"""Offline lifecycle, immutable snapshot, freshness, and migration boundaries."""

import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from test_ingestion import synthetic_pdf
from test_retrieval import corpus  # noqa: F401 -- shared offline fixture

from app.chunking.core import build_chunk_artifact, validate_normalized_document
from app.ingestion.cli import main
from app.ingestion.html import extract_html
from app.ingestion.lifecycle import audit, freshness
from app.ingestion.models import Manifest, Source, active_sources, source_identity
from app.ingestion.pdf import extract_pdf
from app.ingestion.snapshots import read_pair, snapshot_path, write_pair
from app.ingestion.writer import build_document, build_pdf_document
from app.retrieval.table_index import _sources


def source(**changes):
    return Source.model_validate(
        {
            "id": "example",
            "logical_document_id": "example-document",
            "version": 1,
            "title": "Источник",
            "url": "https://unecon.ru/example/",
            "source_type": "html",
            "category": "test",
            "admission_year": 2027,
            "status": "active",
            "supersedes": None,
            "published_at": None,
            "effective_from": None,
            "effective_to": None,
            "processing": {"table_aware": False},
            **changes,
        }
    )


def manifest(*sources):
    return Manifest.model_validate(
        {"schema_version": 2, "sources": [s.model_dump() for s in sources]}
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"active": True},
        {"unknown": 1},
        {"id": "Bad ID"},
        {"logical_document_id": "bad_id"},
        {"version": True},
        {"version": 0},
        {"version": "1"},
        {"version": 1.5},
        {"status": "archived"},
        {"published_at": "2026-02-30"},
        {"published_at": "2026-9-01"},
        {"published_at": "2026-09-01T00:00:00"},
        {"effective_to": False},
        {"effective_from": "2027-09-01", "effective_to": "2027-01-01"},
        {"processing": {"table_aware": True}},
        {"processing": {"table_aware": 1}},
        {"processing": {"table_aware": False, "other": False}},
    ],
)
def test_strict_source(changes):
    with pytest.raises(ValueError):
        source(**changes)


@pytest.mark.parametrize("schema_version", [1, True, 2.0, "2"])
def test_registry_rejects_legacy_and_noninteger_schema(schema_version):
    with pytest.raises(ValueError):
        Manifest.model_validate(
            {"schema_version": schema_version, "sources": [source().model_dump()]}
        )


def test_suspicious_superseded_dates_only_warn():
    older = source(status="superseded", effective_from="2028-01-01", effective_to="2029-01-01")
    newer = source(id="example-v2", version=2, supersedes=older.id, effective_from="2028-02-01")
    registry = manifest(older, newer)
    assert {w["warning"] for w in audit(registry, date(2027, 1, 1))} == {
        "superseded_future_effective_date",
        "superseded_overlapping_interval",
        "active_not_yet_effective",
    }
    assert active_sources(registry) == [newer]


def test_registry_states_chain_selection_and_dates():
    first = source(effective_to="2020-01-01")
    draft = source(id="example-v2", version=2, status="draft", supersedes=first.id)
    registry = manifest(first, draft)
    assert active_sources(registry) == [first]
    assert "active" not in first.model_dump() and "is_active" not in first.model_dump()
    assert {w["warning"] for w in audit(registry, date(2026, 9, 26))} == {
        "active_expired",
        "draft_successor_of_active",
    }
    older = first.model_copy(update={"status": "superseded"})
    newer = draft.model_copy(update={"status": "active"})
    assert active_sources(manifest(older, newer)) == [newer]
    assert active_sources(manifest(source(status="draft"))) == []
    assert active_sources(manifest(source(effective_from="2099-01-01")))


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_id",
        "duplicate_version",
        "two_active",
        "missing",
        "self",
        "other_logical",
        "other_year",
        "not_lower",
        "cycle",
        "orphan",
    ],
)
def test_invalid_chains(kind):
    a = source()
    b = source(id="example-v2", version=2, status="draft", supersedes=a.id)
    if kind == "duplicate_id":
        b = b.model_copy(update={"id": a.id})
    if kind == "duplicate_version":
        b = b.model_copy(update={"version": 1})
    if kind == "two_active":
        b = b.model_copy(update={"status": "active"})
    if kind == "missing":
        b = b.model_copy(update={"supersedes": "missing"})
    if kind == "self":
        b = b.model_copy(update={"supersedes": b.id})
    if kind == "other_logical":
        b = b.model_copy(update={"logical_document_id": "other"})
    if kind == "other_year":
        b = b.model_copy(update={"admission_year": 2028})
    if kind == "not_lower":
        b = b.model_copy(update={"version": 1})
    if kind == "cycle":
        a = a.model_copy(update={"supersedes": b.id})
    if kind == "orphan":
        b = b.model_copy(update={"status": "superseded"})
    with pytest.raises(ValueError):
        manifest(a, b)


def make_pair(tmp_path, *, pdf=False):
    s = source(source_type="pdf" if pdf else "html")
    raw = synthetic_pdf() if pdf else b"<main><h1>Title</h1><p>Useful admission text.</p></main>"
    artifact = (
        build_pdf_document(s, s.url, raw, extract_pdf(raw))
        if pdf
        else build_document(s, s.url, extract_html(raw.decode(), s.title), raw)
    )
    root = tmp_path / "processed"
    originals = tmp_path / "originals"
    write_pair(s, root / s.source_type, originals, artifact, raw)
    return s, raw, artifact, root, originals


@pytest.mark.parametrize("pdf", [False, True])
def test_snapshot_roundtrip_noop_and_mutable_registry_metadata(tmp_path, pdf):
    s, raw, artifact, root, originals = make_pair(tmp_path, pdf=pdf)
    before = snapshot_path(originals, s).stat().st_mtime_ns
    write_pair(s, root / s.source_type, originals, artifact, raw)
    assert snapshot_path(originals, s).stat().st_mtime_ns == before
    assert read_pair(s, root / s.source_type, originals) == (artifact, raw)
    edited = s.model_copy(update={"status": "draft", "published_at": "2026-01-01"})
    assert source_identity(s) == source_identity(edited)
    assert build_chunk_artifact(s, artifact) == build_chunk_artifact(edited, artifact)
    changed_version = s.model_copy(update={"version": 2})
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_normalized_document(changed_version, artifact)
    with pytest.raises(ValueError, match="schema_version"):
        validate_normalized_document(s, {**artifact, "schema_version": 1})


def test_html_presentation_change_requires_new_version(tmp_path):
    s, raw, artifact, root, originals = make_pair(tmp_path)
    changed = raw + b"<!-- layout -->"
    new = build_document(s, s.url, extract_html(changed.decode(), s.title), changed)
    assert new["document"] == artifact["document"]
    with pytest.raises(ValueError, match="immutable"):
        write_pair(s, root / "html", originals, new, changed)
    assert read_pair(s, root / "html", originals) == (artifact, raw)


@pytest.mark.parametrize("fault", ["snapshot", "text", "incomplete", "json"])
def test_corrupt_pair_rejected(tmp_path, fault):
    s, raw, artifact, root, originals = make_pair(tmp_path)
    path = root / "html" / f"{s.id}.json"
    if fault == "snapshot":
        snapshot_path(originals, s).write_bytes(raw + b"changed")
    if fault == "text":
        artifact["document"]["text"] = "tampered"
        artifact["document"]["content_sha256"] = hashlib.sha256(b"tampered").hexdigest()
        path.write_text(json.dumps(artifact), encoding="utf-8")
    if fault == "incomplete":
        snapshot_path(originals, s).unlink()
    if fault == "json":
        path.write_text("broken", encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        read_pair(s, root / "html", originals)


def test_new_pair_write_failure_cleans_up_and_does_not_replace(tmp_path, monkeypatch):
    from app.ingestion import snapshots

    s = source()
    raw = b"<p>Text</p>"
    artifact = build_document(s, s.url, extract_html(raw.decode(), s.title), raw)
    original_replace = snapshots.os.replace

    def fail_second(src, dst):
        if Path(dst).suffix == ".json":
            raise OSError("disk full")
        original_replace(src, dst)

    monkeypatch.setattr(snapshots.os, "replace", fail_second)
    with pytest.raises(OSError):
        write_pair(s, tmp_path / "html", tmp_path / "originals", artifact, raw)
    assert not snapshot_path(tmp_path / "originals", s).exists()
    assert not list(tmp_path.rglob("*.tmp")) and not list(tmp_path.rglob("*.lock"))


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unchanged", "unchanged"),
        ("presentation", "presentation_changed"),
        ("content", "content_changed"),
        ("redirect", "redirect_changed"),
        ("http", "error"),
        ("empty", "error"),
        ("missing", "missing_local_artifact"),
        ("corrupt", "error"),
        ("pdf", "file_changed"),
    ],
)
def test_freshness_is_read_only(tmp_path, case, expected):
    s, raw, artifact, root, originals = make_pair(tmp_path, pdf=case == "pdf")
    if case == "missing":
        (root / "html" / f"{s.id}.json").unlink()
    if case == "corrupt":
        snapshot_path(originals, s).write_bytes(b"broken")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    requests = []

    def respond(request):
        requests.append(str(request.url))
        if case == "http":
            return httpx.Response(500)
        if case == "redirect" and len(requests) == 1:
            return httpx.Response(302, headers={"location": "/moved/"})
        content = raw
        if case == "presentation":
            content += b"<!-- changed -->"
        if case in {"content", "redirect"}:
            content = b"<p>New contents</p>"
        if case == "empty":
            content = b"<main></main>"
        if case == "pdf":
            content += b"\n% changed"
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf" if case == "pdf" else "text/html"},
            content=content,
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        report = freshness(manifest(s), root, originals, client)
    assert report["sources"][0]["status"] == expected
    assert report["active_count"] == 1 and report["counts"] == {expected: 1}
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    if case in {"missing", "corrupt"}:
        assert not requests
    if case == "redirect":
        assert "content_sha256" in report["sources"][0]["changes"]


def test_dynamic_table_sources_order_and_future_year(tmp_path):
    a = source(
        id="future-a",
        logical_document_id="future-a",
        source_type="pdf",
        processing={"table_aware": True},
    )
    b = source(
        id="future-b",
        logical_document_id="future-b",
        source_type="pdf",
        processing={"table_aware": True},
    )
    c = source(
        id="draft",
        logical_document_id="draft",
        source_type="pdf",
        status="draft",
        processing={"table_aware": True},
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest(b, a, c, source()).model_dump_json(), encoding="utf-8")
    assert [s.id for s in _sources(path)] == [b.id, a.id]


def test_cli_list_audit_and_inactive_selection_do_not_use_network(tmp_path, capsys):
    path = tmp_path / "manifest.json"
    path.write_text(manifest(source(status="draft")).model_dump_json(), encoding="utf-8")
    transport = httpx.MockTransport(lambda _: pytest.fail("unexpected network"))
    assert main(["list", "--manifest", str(path)], transport) == 0
    assert "status=draft" in capsys.readouterr().out
    assert main(["audit", "--manifest", str(path), "--as-of", "2026-09-26"], transport) == 0
    assert main(["fetch", "--manifest", str(path), "--source-id", "example"], transport) == 1
    assert main(["fetch", "--manifest", str(path)], transport) == 0


def test_snapshot_hash_is_bound_to_chunks_and_index(corpus, tmp_path):  # noqa: F811
    from test_retrieval import FakeEmbedder

    from app.retrieval.index import RetrievalSession, build_index

    path, chunks_dir, index_dir, artifacts = corpus
    build_index(path, chunks_dir, index_dir, "fake", embedder_factory=FakeEmbedder)
    altered = copy.deepcopy(artifacts["faq"])
    altered["source"]["snapshot_sha256"] = "b" * 64
    (chunks_dir / "faq.json").write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        RetrievalSession(
            path,
            chunks_dir,
            index_dir,
            "fake",
            embedder_factory=lambda: pytest.fail("model loaded"),
        )


@pytest.mark.parametrize(
    "case", ["unchanged", "redirect_changed", "error", "missing_local_artifact"]
)
def test_pdf_freshness_cases(tmp_path, case):
    s, raw, _, root, originals = make_pair(tmp_path, pdf=True)
    if case == "missing_local_artifact":
        snapshot_path(originals, s).unlink()
    count = 0

    def respond(_request):
        nonlocal count
        count += 1
        if case == "error":
            raise httpx.ConnectError("offline")
        if case == "redirect_changed" and count == 1:
            return httpx.Response(302, headers={"location": "/other.pdf"})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=raw)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        report = freshness(manifest(s), root, originals, client)
    assert report["sources"][0]["status"] == case
    if case == "missing_local_artifact":
        assert count == 0


@pytest.mark.parametrize("changed", [False, True])
def test_cli_freshness_exit_and_independent_writer(tmp_path, monkeypatch, changed):
    s, raw, _, root, originals = make_pair(tmp_path)
    registry = tmp_path / "manifest.json"
    registry.write_text(manifest(s).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.writer.write_document",
        lambda *_args: pytest.fail("ingestion writer called by audit"),
    )
    content = b"<p>Changed</p>" if changed else raw
    code = main(
        [
            "freshness",
            "--manifest",
            str(registry),
            "--input-root",
            str(root),
            "--originals-root",
            str(originals),
            "--output-dir",
            str(tmp_path / "audit"),
        ],
        httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/html"}, content=content)
        ),
    )
    assert code == int(changed)
    report = json.loads((tmp_path / "audit/report.json").read_text(encoding="utf-8"))
    assert report["registry_schema_version"] == 2 and report["checked_at"]
    assert "Changed" not in json.dumps(report)


@pytest.mark.parametrize("fault", [None, "missing", "hash", "normalized"])
def test_table_builder_real_local_snapshot_boundary(tmp_path, monkeypatch, fault):
    import faiss
    import numpy as np
    import pymupdf

    import app.retrieval.table_index as tables
    from app.retrieval.table_extract import row_id

    s, raw, normalized, root, originals = make_pair(tmp_path, pdf=True)
    s = s.model_copy(update={"processing": s.processing.model_copy(update={"table_aware": True})})
    registry = tmp_path / "manifest.json"
    registry.write_text(manifest(s).model_dump_json(), encoding="utf-8")
    production = faiss.IndexFlatIP(2)
    production.add(np.array([[1.0, 0.0]], dtype=np.float32))
    metadata = {
        "embedding": {"dimension": 2},
        "index": {"file_sha256": "a" * 64},
        "records": [{"chunk_id": "existing"}],
    }
    monkeypatch.setattr(tables, "_production", lambda *_: (production, metadata))
    monkeypatch.setattr("app.ingestion.fetcher.fetch_pdf", lambda *_: pytest.fail("network fetch"))
    monkeypatch.setattr(httpx.Client, "send", lambda *_a, **_kw: pytest.fail("network"))
    if fault == "missing":
        snapshot_path(originals, s).unlink()
    if fault == "hash":
        snapshot_path(originals, s).write_bytes(raw + b"changed")
    if fault == "normalized":
        normalized["document"]["file_sha256"] = "b" * 64
        (root / "pdf" / f"{s.id}.json").write_text(json.dumps(normalized), encoding="utf-8")

    def extraction(source, final_url, content):
        assert content == raw
        text = "column = value"
        return {
            "source": {"final_url": final_url},
            "extraction": {
                "library": "PyMuPDF",
                "version": pymupdf.VersionBind,
                "strategy": "lines_strict",
            },
            "tables": [
                {
                    "rows": [
                        {
                            "row_id": row_id(
                                source.id, hashlib.sha256(raw).hexdigest(), 1, 1, 1, text
                            ),
                            "text": text,
                            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "page": 1,
                            "table_index": 1,
                            "row_index": 1,
                        }
                    ]
                }
            ],
        }

    monkeypatch.setattr(tables, "extract_pdf_tables", extraction)

    class Embedder:
        def __init__(self):
            assert fault is None, "model loaded before local validation"

        def encode_documents(self, texts):
            assert texts == ["column = value"]
            return np.array([[1.0, 0.0]], dtype=np.float32)

    args = (registry, tmp_path, tmp_path, root / "pdf", tmp_path / "table", "fake")
    if fault:
        with pytest.raises(ValueError):
            tables.build_table_index(*args, embedder_factory=Embedder, originals_root=originals)
        assert not (tmp_path / "table").exists()
    else:
        result = tables.build_table_index(
            *args, embedder_factory=Embedder, originals_root=originals
        )
        assert result["sources"][0]["version"] == 1
        tables.validated_table_index(registry, root / "pdf", tmp_path / "table", metadata, "fake")
        s = s.model_copy(update={"version": 2})
        registry.write_text(manifest(s).model_dump_json(), encoding="utf-8")
        with pytest.raises(ValueError, match="metadata mismatch"):
            tables.validated_table_index(
                registry, root / "pdf", tmp_path / "table", metadata, "fake"
            )


@pytest.mark.parametrize("drift", [False, True])
def test_explicit_migration_checks_all_sources_before_writing(tmp_path, drift):
    from app.ingestion.migration import LEGACY_KEYS, prepare

    current = Manifest.model_validate_json(
        (Path(__file__).resolve().parents[2] / "data/source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    legacy = {
        "schema_version": 1,
        "sources": [
            {**{key: source_identity(s)[key] for key in LEGACY_KEYS}, "active": True}
            for s in current.sources
        ],
    }
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    snapshots = {}
    input_root = tmp_path / "old"
    for s in current.sources:
        raw = synthetic_pdf() if s.source_type == "pdf" else b"<p>Source text</p>"
        snapshots[s.url] = raw
        artifact = (
            build_pdf_document(s, s.url, raw, extract_pdf(raw))
            if s.source_type == "pdf"
            else build_document(s, s.url, extract_html(raw.decode(), s.title), raw)
        )
        chunks = build_chunk_artifact(s, artifact)
        for folder, saved in ((s.source_type, artifact), ("chunks", chunks)):
            old = copy.deepcopy(saved)
            old["schema_version"] = 1
            for key in ("logical_document_id", "version", "snapshot_sha256"):
                del old["source"][key]
            directory = input_root / folder
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{s.id}.json").write_text(json.dumps(old), encoding="utf-8")
    if drift:
        snapshots[current.sources[-1].url] += b"\n% new version"
    requests = []

    def respond(request):
        url = str(request.url)
        requests.append(url)
        payload = snapshots[url]
        return httpx.Response(
            200,
            headers={
                "content-type": "application/pdf" if payload.startswith(b"%PDF") else "text/html"
            },
            content=payload,
        )

    destination = tmp_path / "candidate"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        if drift:
            with pytest.raises(ValueError):
                prepare(legacy_path, input_root, destination, client)
            assert not destination.exists()
        else:
            assert len(prepare(legacy_path, input_root, destination, client)["sources"]) == 12
    assert requests == [s.url for s in current.sources]
