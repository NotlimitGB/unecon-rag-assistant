"""Publication exercises real extraction/FAISS with explicit offline fake models."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.chunking.core import build_source_chunks
from app.config import Settings
from app.corpus import operations, storage, validation
from app.corpus.checks import check_corpus
from app.corpus.incoming import read_package, transition
from app.corpus.models import Release, State
from app.corpus.paths import CorpusError, CorpusPaths, safe_id
from app.corpus.storage import (
    atomic_json,
    current_paths,
    read_json,
    read_state,
    sha,
    verify_release,
)
from app.ingestion.html import extract_html
from app.ingestion.models import Manifest, Source
from app.ingestion.pdf import extract_pdf
from app.ingestion.snapshots import write_pair
from app.ingestion.writer import build_document, build_pdf_document, write_document
from app.main import create_app
from app.retrieval.index import build_index
from app.retrieval.service import RetrievalService
from app.retrieval.table_index import build_table_index

ROOT = Path(__file__).resolve().parents[2]


class Embedder:
    document_calls = 0

    def encode_documents(self, texts):
        type(self).document_calls += 1
        return np.stack([self.encode_query(t) for t in texts])

    def encode_query(self, text):
        raw = np.array(list(hashlib.sha256(text.encode()).digest()[:8]), dtype=np.float32) + 1
        return raw / np.linalg.norm(raw)


class Reranker:
    def score(self, query, passages):
        return [float(int(hashlib.sha256(t.encode()).hexdigest()[:5], 16)) for t in passages]


def table_pdf(value="100"):
    with pymupdf.open() as doc:
        page = doc.new_page()
        xs, ys = [40, 260, 430], list(range(80, 321, 30))
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]))
        for y in ys:
            page.draw_line((xs[0], y), (xs[-1], y))
        for i in range(8):
            page.insert_text((45, 99 + i * 30), "Programme" if i == 0 else f"Economics {i}")
            page.insert_text((265, 99 + i * 30), "Seats" if i == 0 else value)
        return doc.tobytes()


def source(identity, category, pdf=False):
    return Source(
        id=identity,
        logical_document_id=identity,
        version=1,
        title=identity,
        url=f"https://unecon.ru/{identity}",
        source_type="pdf" if pdf else "html",
        category=category,
        admission_year=2026,
        status="active",
        supersedes=None,
        published_at=None,
        effective_from=None,
        effective_to=None,
        processing={"table_aware": pdf},
    )


@pytest.fixture
def environment(tmp_path, monkeypatch):
    # Prevent every network call, including accidental model downloads.
    import socket

    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("network forbidden"))
    config = Settings(embedding_model="fake", reranker_model="fake")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    categories = [
        "admission_rules",
        "admission_deadlines",
        "admission_capacity",
        "entrance_exams",
        "tuition",
        "faq",
        "admissions_overview",
    ]
    sources = [source(f"source-{i}", category) for i, category in enumerate(categories)]
    sources.append(source("table", "admission_capacity", True))
    manifest = Manifest(schema_version=2, sources=sources)
    registry = legacy / "registry.json"
    atomic_json(registry, manifest.model_dump())
    for s in sources:
        data = (
            table_pdf()
            if s.source_type == "pdf"
            else (
                "<p>" + (f"{s.id} Official admission text and requirements. " * 80) + "</p>"
            ).encode()
        )
        artifact = (
            build_pdf_document(s, s.url, data, extract_pdf(data))
            if s.source_type == "pdf"
            else build_document(s, s.url, extract_html(data.decode(), s.title), data)
        )
        write_pair(s, legacy / s.source_type, legacy / "originals", artifact, data)
        build_source_chunks(s, legacy, legacy / "chunks")
    build_index(registry, legacy / "chunks", legacy / "index", "fake", embedder_factory=Embedder)
    build_table_index(
        registry,
        legacy / "chunks",
        legacy / "index",
        legacy / "pdf",
        legacy / "table_index",
        "fake",
        embedder_factory=Embedder,
        originals_root=legacy / "originals",
    )
    # A synthetic 80-question gold set uses the real schema/distributions, separate from production.
    dataset = read_json(ROOT / "data/evaluation/retrieval_questions.json")
    for q in dataset["questions"]:
        s = sources[categories.index(q["category"])]
        q.update(primary_source_id=s.id, acceptable_source_ids=[s.id], expected_pages=None)
    dataset_path = tmp_path / "dataset.json"
    atomic_json(dataset_path, dataset)
    monkeypatch.setattr(validation, "RETRIEVAL_DATASET_SHA", sha(dataset_path.read_bytes()))
    root, incoming = tmp_path / "processed", tmp_path / "incoming"
    operations.bootstrap(
        "release-a", root=root, legacy=CorpusPaths(registry, legacy), config=config
    )
    return {
        "root": root,
        "incoming": incoming,
        "config": config,
        "legacy": legacy,
        "manifest": manifest,
        "dataset": dataset_path,
        "reference": registry,
    }


def package(env, *, pdf=False, identity=None, value="999"):
    prior = env["manifest"].sources[-1 if pdf else 0]
    s = prior.model_copy(
        update={"id": identity or prior.id + "-v2", "version": 2, "supersedes": prior.id}
    )
    directory = env["incoming"] / s.id
    directory.mkdir(parents=True)
    atomic_json(
        directory / "metadata.json", {"schema_version": 1, **s.model_dump(exclude={"status"})}
    )
    (directory / f"document.{s.source_type}").write_bytes(
        table_pdf(value)
        if pdf
        else f"<p>Changed official admission information: {value}.</p>".encode()
    )
    return directory


def staged(env, *, pdf=False, name="release-b"):
    p = package(env, pdf=pdf)
    result = operations.stage(
        name,
        [p],
        root=env["root"],
        incoming_root=env["incoming"],
        config=env["config"],
        embedder_factory=Embedder,
    )
    return result, p


def validate(env, name="release-b"):
    return validation.validate(
        name,
        root=env["root"],
        incoming_root=env["incoming"],
        config=env["config"],
        dataset_path=env["dataset"],
        reference_manifest=env["reference"],
        embedder_factory=Embedder,
        reranker_factory=Reranker,
    )


@pytest.mark.parametrize("pdf", [False, True])
def test_full_publication_and_rollback(environment, monkeypatch, pdf):
    env = environment
    root = env["root"]
    before = (root / "release_state.json").read_bytes()
    base = verify_release(current_paths(root).root)
    embedding_calls = Embedder.document_calls
    created, p = staged(env, pdf=pdf)
    assert not created.sealed and (root / "release_state.json").read_bytes() == before
    assert Embedder.document_calls - embedding_calls == 2  # Both indexes, exactly once each.
    report, token = validate(env)
    assert report.publishable, report.errors
    assert len(report.comparison["candidate"]["questions"]) == 80
    assert report.comparison["changed_question_ids"]
    operations.publish("release-b", "release-a", token, "release-b", root=root)
    paths = current_paths(root)
    manifest, meta, rows = check_corpus(paths, "fake")
    prior = "table" if pdf else "source-0"
    assert next(s for s in manifest.sources if s.id == prior).status == "superseded"
    assert prior not in {r["source_id"] for r in meta["records"] + rows}
    assert prior + "-v2" in {r["source_id"] for r in meta["records"] + rows}
    from app.retrieval.index import RetrievalSession

    dense = RetrievalSession(
        paths.manifest, paths.chunks, paths.index, "fake", embedder_factory=Embedder
    )
    new_text = next(r["text"] for r in meta["records"] if r["source_id"] == prior + "-v2")
    assert dense.search(new_text, 1)[0]["source_id"] == prior + "-v2"
    # Published runtime/rollback no longer depend on the external incoming package.
    (p / "metadata.json").unlink()
    monkeypatch.setattr(Embedder, "encode_documents", lambda *_: pytest.fail("rollback rebuilt"))
    operations.rollback("release-a", "release-b", "release-a", root=root)
    assert verify_release(current_paths(root).root) == base
    state = read_state(root)
    assert state.generation == 3 and state.published_releases == ["release-a", "release-b"]
    assert (root / "releases/release-b").is_dir()


@pytest.mark.parametrize("change", ["document", "metadata", "missing"])
def test_incoming_change_blocks_validation(environment, change):
    env = environment
    _, package_path = staged(env)
    target = package_path / ("document.html" if change == "document" else "metadata.json")
    if change == "missing":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b" ")
    before = (env["root"] / "release_state.json").read_bytes()
    report, token = validate(env)
    assert not report.publishable and token is None
    assert (env["root"] / "release_state.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["confirm", "sha", "expected", "tamper", "replace"])
def test_failed_publication_preserves_pointer(environment, monkeypatch, fault):
    env = environment
    staged(env)
    report, token = validate(env)
    assert report.publishable
    root = env["root"]
    before = (root / "release_state.json").read_bytes()
    if fault == "tamper":
        (root / "staging/release-b/chunks/source-0-v2.json").write_text("{}")
    if fault == "replace":
        original = os.replace

        def fail(source, dest):
            if Path(dest).name == "release_state.json":
                raise OSError("injected pointer publication failure")
            return original(source, dest)

        monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises((ValueError, OSError)):
        operations.publish(
            "release-b",
            "other" if fault == "expected" else "release-a",
            "0" * 64 if fault == "sha" else token,
            "wrong" if fault == "confirm" else "release-b",
            root=root,
        )
    assert (root / "release_state.json").read_bytes() == before
    assert not (root / ".corpus.lock").exists()
    if fault == "replace":
        monkeypatch.setattr(storage.os, "replace", original)
        assert (root / "releases/release-b").is_dir()  # Safe orphan retry.
        operations.publish("release-b", "release-a", token, "release-b", root=root)
        assert read_state(root).current_release == "release-b"


def test_aba_stale_candidate_and_unpublished_rollback(environment):
    env = environment
    staged(env)
    report, token = validate(env)
    assert report.publishable
    root = env["root"]
    with pytest.raises(ValueError):
        operations.rollback("release-b", "release-a", "release-b", root=root)
    # A separately named candidate, prepared on the same generation.
    p = env["incoming"] / "source-0-v2"
    operations.stage(
        "release-c",
        [p],
        root=root,
        incoming_root=env["incoming"],
        config=env["config"],
        embedder_factory=Embedder,
    )
    report_c, token_c = validate(env, "release-c")
    assert report_c.publishable
    operations.publish("release-b", "release-a", token, "release-b", root=root)
    operations.rollback("release-a", "release-b", "release-a", root=root)
    before = (root / "release_state.json").read_bytes()
    with pytest.raises(ValueError, match="stale"):
        operations.publish("release-c", "release-a", token_c, "release-c", root=root)
    assert (root / "release_state.json").read_bytes() == before


def test_runtime_pins_release_or_error_and_fake_mode(environment, monkeypatch):
    env = environment
    root = env["root"]
    monkeypatch.setattr("app.retrieval.service.current_paths", lambda: current_paths(root))
    old = RetrievalService(config=env["config"])
    assert old._dense_session is None and old.manifest_path.parent.name == "release-a"
    staged(env)
    _, token = validate(env)
    operations.publish("release-b", "release-a", token, "release-b", root=root)
    assert old.manifest_path.parent.name == "release-a"
    assert RetrievalService(config=env["config"]).manifest_path.parent.name == "release-b"
    (root / "release_state.json").write_text("broken")
    broken = RetrievalService()
    assert broken._dense_session is None
    monkeypatch.setattr(
        "app.retrieval.service.current_paths", lambda: pytest.fail("must stay pinned")
    )
    with pytest.raises(ValueError, match="corpus selection"):
        broken.retrieve("Вопрос")
    explicit = RetrievalService(index_dir=Path("explicit"))
    assert explicit.index_dir == Path("explicit") and explicit._corpus_error is None
    fake = RetrievalService(dense_factory=lambda: None)
    assert fake._corpus_error is None


def test_missing_pointer_keeps_health_but_safe_503(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.service.current_paths",
        lambda: (_ for _ in ()).throw(CorpusError("sensitive internal path")),
    )
    with TestClient(create_app()) as client:
        assert client.get("/api/v1/health").status_code == 200
        response = client.post("/api/v1/answer", json={"question": "Вопрос"})
        assert response.status_code == 503 and "sensitive" not in response.text


@pytest.mark.parametrize(
    "identity", ["../escape", "C:/outside", "con", "nul", "com1", "UPPER", "a/b"]
)
def test_unsafe_ids(identity):
    with pytest.raises(ValueError):
        safe_id(identity)


def test_strict_state_and_lock(environment):
    root = environment["root"]
    state = read_state(root).model_dump()
    for update in (
        {"generation": True},
        {"unknown": 1},
        {"schema_version": True},
        {"updated_at": "2026-01-01"},
        {"published_releases": ["release-a", "release-a"]},
    ):
        with pytest.raises(ValueError):
            State.model_validate({**state, **update})
    with storage.operator_lock(root):
        with pytest.raises(ValueError, match="lock"):
            operations.bootstrap("second", root=root)
    with pytest.raises(ValueError, match="already exists"):
        operations.bootstrap("second", root=root)


def test_batch_conflicts_and_unsafe_packages(environment):
    env = environment
    path = package(env)
    item = read_package(path, env["incoming"])
    with pytest.raises(ValueError, match="competing"):
        transition(env["manifest"], [item, item])
    with pytest.raises(ValueError, match="escapes"):
        read_package(path, path / "other")
    (path / "extra.txt").write_text("no")
    with pytest.raises(ValueError, match="only"):
        read_package(path, env["incoming"])


def test_legacy_writers_refuse_saved_release(environment):
    paths = current_paths(environment["root"])
    with pytest.raises(ValueError, match="immutable"):
        write_document(paths.chunks, "anything", {})
    with pytest.raises(ValueError, match="immutable"):
        build_index(paths.manifest, paths.chunks, paths.index, "fake", embedder_factory=Embedder)
    with pytest.raises(ValueError, match="immutable"):
        build_table_index(
            paths.manifest,
            paths.chunks,
            paths.index,
            paths.pdf,
            paths.tables,
            "fake",
            embedder_factory=Embedder,
        )


def test_corrupt_rollback_and_strict_release_inventory(environment):
    env = environment
    staged(env)
    _, token = validate(env)
    root = env["root"]
    operations.publish("release-b", "release-a", token, "release-b", root=root)
    before = (root / "release_state.json").read_bytes()
    target = root / "releases/release-a"
    metadata = read_json(target / "release.json")
    with pytest.raises(ValueError):
        Release.model_validate({**metadata, "extra": 1})
    (target / "index/index.faiss").write_bytes(b"broken")
    with pytest.raises(ValueError):
        operations.rollback("release-a", "release-b", "release-a", root=root)
    assert (root / "release_state.json").read_bytes() == before


@pytest.mark.parametrize("phase", ["extract", "dense", "tables", "validation"])
def test_failures_never_switch_pointer(environment, monkeypatch, phase):
    env = environment
    before = (env["root"] / "release_state.json").read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("injected failure")

    if phase == "validation":
        staged(env)
        monkeypatch.setattr(validation, "compare", fail)
        report, token = validate(env)
        assert not report.publishable and token is None
    else:
        target = {
            "extract": "app.corpus.incoming.extract_html",
            "dense": "app.corpus.operations.build_index",
            "tables": "app.corpus.operations.build_table_index",
        }[phase]
        monkeypatch.setattr(target, fail)
        with pytest.raises(ValueError, match="injected"):
            staged(env)
    assert (env["root"] / "release_state.json").read_bytes() == before
    assert not (env["root"] / ".corpus.lock").exists()


@pytest.mark.parametrize(
    "mutation",
    ["status", "unknown", "version", "reserved", "url", "duplicate-json", "utf8", "empty"],
)
def test_incoming_strict_metadata_and_content(environment, mutation):
    env = environment
    path = package(env)
    metadata_path = path / "metadata.json"
    data = read_json(metadata_path)
    if mutation in ("status", "unknown"):
        data[mutation] = "active"
    elif mutation == "version":
        data["version"] = True
    elif mutation == "reserved":
        data["id"] = "con"
    elif mutation == "url":
        data["url"] = "https://example.org/"
    atomic_json(metadata_path, data)
    if mutation == "duplicate-json":
        metadata_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    if mutation in ("utf8", "empty"):
        (path / "document.html").write_bytes(b"\xff" if mutation == "utf8" else b"<body></body>")
    with pytest.raises(ValueError):
        read_package(path, env["incoming"])


@pytest.mark.parametrize(
    "mutation",
    [
        "skip-version",
        "wrong-family",
        "inactive-prior",
        "new-family-v2",
        "duplicate-id",
        "same-family-new",
        "dependent",
    ],
)
def test_transition_conflicts(environment, mutation):
    from dataclasses import replace

    env = environment
    item = read_package(package(env), env["incoming"])
    base = env["manifest"]
    update = {
        "skip-version": {"version": 3},
        "wrong-family": {"logical_document_id": "other"},
        "new-family-v2": {"logical_document_id": "new", "supersedes": None},
        "duplicate-id": {"id": "source-0"},
        "same-family-new": {"supersedes": None, "version": 1},
    }
    if mutation == "inactive-prior":
        base = base.model_copy(deep=True)
        base.sources[0].status = "draft"
    if mutation == "dependent":
        second = replace(
            item,
            source=item.source.model_copy(
                update={"id": "source-0-v3", "version": 3, "supersedes": item.source.id}
            ),
        )
        items = [item, second]
    else:
        items = [replace(item, source=item.source.model_copy(update=update.get(mutation, {})))]
    with pytest.raises(ValueError):
        transition(base, items)


def test_new_family_is_appended_in_argument_order(environment):
    from dataclasses import replace

    env = environment
    item = read_package(package(env), env["incoming"])
    packages = [
        replace(
            item,
            source=item.source.model_copy(
                update={
                    "id": identity,
                    "logical_document_id": identity,
                    "version": 1,
                    "supersedes": None,
                }
            ),
        )
        for identity in ("new-b", "new-a")
    ]
    result = transition(env["manifest"], packages)
    assert [s.id for s in result.sources[-2:]] == ["new-b", "new-a"]


def test_reparse_point_rejected_before_file_read(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.corpus.paths import regular_file

    target = tmp_path / "file.json"
    target.write_text("{}")
    original = Path.lstat

    def reparse(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(ValueError, match="reparse"):
        regular_file(target, tmp_path)


def test_report_writer_accepts_parent_relative_output_but_not_release(tmp_path, monkeypatch):
    working = tmp_path / "backend"
    working.mkdir()
    monkeypatch.chdir(working)
    path = write_document(Path("../reports"), "smoke", {"complete": True})
    assert read_json(path) == {"complete": True}
    with pytest.raises(ValueError, match="immutable"):
        write_document(Path("../processed/releases/example"), "smoke", {})


def test_candidate_change_requires_explicit_validation_and_failed_retry_unseals(
    environment, monkeypatch
):
    env = environment
    staged(env)
    report, token = validate(env)
    assert report.publishable
    root = env["root"]
    target = root / "staging/release-b/registry.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="inventory"):
        operations.publish("release-b", "release-a", token, "release-b", root=root)
    # Only an explicit full validation may accept a harmless byte change and issue a new token.
    report, fresh_token = validate(env)
    assert report.publishable and fresh_token != token
    monkeypatch.setattr(
        validation, "compare", lambda *args: (_ for _ in ()).throw(ValueError("failed"))
    )
    report, failed_token = validate(env)
    assert not report.publishable and failed_token is None
    with pytest.raises(ValueError, match="not sealed"):
        operations.publish("release-b", "release-a", fresh_token, "release-b", root=root)


def test_package_change_during_build_or_validation(environment, monkeypatch):
    env = environment
    path = package(env)
    build = operations.build_table_index

    def build_and_change(*args, **kwargs):
        result = build(*args, **kwargs)
        (path / "document.html").write_bytes(b"<p>Changed during build.</p>")
        return result

    monkeypatch.setattr(operations, "build_table_index", build_and_change)
    before = (env["root"] / "release_state.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        operations.stage(
            "changed",
            [path],
            root=env["root"],
            incoming_root=env["incoming"],
            config=env["config"],
            embedder_factory=Embedder,
        )
    assert (env["root"] / "release_state.json").read_bytes() == before


def test_validation_reports_historical_hits_without_remapping_new_id(environment):
    env = environment
    staged(env)
    report, _ = validate(env)
    assert report.publishable
    old_gold = [
        q
        for q in report.comparison["candidate"]["questions"]
        if q["primary_source_id"] == "source-0"
    ]
    assert old_gold and all(q["primary_rank"] is None for q in old_gold)
    assert report.comparison["count_deltas"]["chunks"] < 0


def test_cli_list_and_show_without_document_text(environment, capsys):
    from app.corpus.cli import main

    root = str(environment["root"])
    assert main(["list", "--processed-root", root]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["releases"][0]["current"] and data["releases"][0]["integrity_valid"]
    assert "Official admission" not in json.dumps(data)
    assert main(["show", "--release", "release-a", "--processed-root", root]) == 0
    assert json.loads(capsys.readouterr().out)["sealed"]
    assert (
        main(
            [
                "publish",
                "--release",
                "release-a",
                "--expected-current",
                "release-a",
                "--confirm",
                "wrong",
                "--validation-sha",
                "0" * 64,
                "--processed-root",
                root,
            ]
        )
        == 1
    )


@pytest.mark.parametrize("target", ["incoming", "candidate"])
def test_changes_during_validation_do_not_seal(environment, monkeypatch, target):
    env = environment
    _, package_path = staged(env)
    original = validation.compare

    def compare_then_change(*args, **kwargs):
        report = original(*args, **kwargs)
        path = (
            package_path / "document.html"
            if target == "incoming"
            else env["root"] / "staging/release-b/chunks/source-0-v2.json"
        )
        path.write_bytes(path.read_bytes() + b"\n")
        return report

    monkeypatch.setattr(validation, "compare", compare_then_change)
    before = (env["root"] / "release_state.json").read_bytes()
    report, token = validate(env)
    assert not report.publishable and token is None
    assert (env["root"] / "release_state.json").read_bytes() == before
