"""Offline regression checks for the adopted table-aware retrieval path."""

import hashlib
import json
from types import SimpleNamespace

import faiss
import numpy as np
import pymupdf
import pytest

from app.generation.service import AnswerService
from app.retrieval.corpus import RetrievalError
from app.retrieval.service import RetrievalResponse, RetrievedChunk
from app.retrieval.table_extract import row_id
from app.retrieval.table_index import _sources, build_table_index, validated_table_index
from app.retrieval.table_session import (
    TableAwareRerankedRetrievalSession,
    select_pdf_page_diversity,
)


def candidate(number, *, source="capacity", page=1, source_type="pdf"):
    return {
        "chunk_id": f"row-{number}",
        "source_id": source,
        "source_type": source_type,
        "page_start": page,
        "page_end": page,
        "score": float(30 - number),
    }


def test_approved_table_scope_is_frozen():
    from pathlib import Path

    assert tuple(
        s.id for s in _sources(Path(__file__).resolve().parents[2] / "data/source_manifest.json")
    ) == (
        "admission-capacity-pdf",
        "entrance-exams-list-pdf",
        "tuition-order-128-pdf",
    )


def test_pdf_page_cap_is_shared_by_chunks_and_rows():
    hits = [
        {**candidate(1), "representation": "production_chunk"},
        {**candidate(2), "representation": "table_row"},
        {**candidate(3), "representation": "table_row"},
        candidate(4, page=2),
        candidate(5, source_type="html", page=None),
        candidate(6, source_type="html", page=None),
    ]
    selected = select_pdf_page_diversity(hits, 5)
    assert [hit["chunk_id"] for hit in selected] == [
        "row-1",
        "row-2",
        "row-4",
        "row-5",
        "row-6",
    ]
    assert [hit["score"] for hit in selected] == [29.0, 28.0, 26.0, 25.0, 24.0]
    assert select_pdf_page_diversity(hits, 1) == hits[:1]
    with pytest.raises(RetrievalError, match="cannot fill"):
        select_pdf_page_diversity([candidate(i) for i in range(1, 7)], 5)


@pytest.mark.parametrize("top_k", [0, 6, True])
def test_selector_rejects_invalid_limit(top_k):
    with pytest.raises(RetrievalError, match="top_k"):
        select_pdf_page_diversity([candidate(1)], top_k)


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def encode_query(self, query):
        self.queries.append(query)
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeReranker:
    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, passages))
        return np.arange(len(passages), dtype=np.float32)


def test_dual_channel_embeds_once_and_reranks_all_25(monkeypatch, tmp_path):
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    production = faiss.IndexFlatIP(2)
    production.add(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (20, 1)))
    table = faiss.IndexFlatIP(2)
    table.add(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (5, 1)))
    records = [
        {
            "vector_id": i,
            "chunk_id": f"chunk-{i}",
            "text": f"text-{i}",
            "source_id": f"source-{i}",
            "source_type": "html",
            "page_start": None,
            "page_end": None,
        }
        for i in range(25)
    ]
    dense = SimpleNamespace(
        index=production,
        records=records[:20],
        embedder=embedder,
        metadata={"embedding": {"dimension": 2}, "records": records[:20]},
    )
    monkeypatch.setattr(
        "app.retrieval.table_session.validated_table_index",
        lambda *_args: (table, records[20:]),
    )
    session = TableAwareRerankedRetrievalSession(
        dense,
        tmp_path,
        tmp_path,
        tmp_path,
        "test",
        "test",
        "cpu",
        8,
        512,
        20,
        reranker_factory=lambda: reranker,
    )
    hits = session.search("question", 5)
    assert embedder.queries == ["question"]
    assert len(reranker.calls) == 1 and len(reranker.calls[0][1]) == 25
    assert len(hits) == 5
    assert [hit["chunk_id"] for hit in hits] == [
        "chunk-24",
        "chunk-23",
        "chunk-22",
        "chunk-21",
        "chunk-20",
    ]
    assert hits[0]["dense_score"] == 1.0
    assert hits[0]["score"] == hits[0]["rerank_score"] == 24.0
    assert session.pairs_scored == 25


def test_candidate_k_drift_fails_before_table_or_reranker_load(tmp_path):
    with pytest.raises(RetrievalError, match="RERANKER_CANDIDATE_K=20"):
        TableAwareRerankedRetrievalSession(
            None, tmp_path, tmp_path, tmp_path, "test", "test", "cpu", 8, 512, 19
        )


def test_table_row_uses_unchanged_answer_citation_contract():
    table_row = RetrievedChunk(
        rank=1,
        chunk_id="capacity:p0002:t001:r0001:0123456789ab",
        text="Источник: Места\nСтраница: 2\nПрограмма = 686",
        source_id="admission-capacity-pdf",
        source_title="Количество мест",
        source_url="https://unecon.ru/capacity.pdf",
        source_type="pdf",
        category="admission_capacity",
        admission_year=2026,
        page=2,
        score=3.0,
        dense_score=0.7,
        rerank_score=3.0,
    )
    response = RetrievalResponse(
        query="Сколько мест?", mode="reranked", top_k=5, results=[table_row]
    )

    class Retrieval:
        def retrieve(self, _question):
            return response

    class Generator:
        def generate(self, _question, contexts):
            assert contexts[0].chunk.text == table_row.text
            return {"status": "answered", "answer": "686 мест.", "cited_context_ids": ["C1"]}

    answer = AnswerService(retrieval_service=Retrieval(), generator=Generator()).answer(
        "Сколько мест?"
    )
    assert answer.citations[0].chunk_id == table_row.chunk_id
    assert answer.citations[0].page == 2
    assert answer.citations[0].source_url == table_row.source_url


def test_offline_builder_and_strict_local_index_validation(monkeypatch, tmp_path):
    import app.retrieval.table_index as table_module

    pdf = b"%PDF-deterministic-test"
    pdf_sha = hashlib.sha256(pdf).hexdigest()
    source_ids = ("admission-capacity-pdf", "entrance-exams-list-pdf", "tuition-order-128-pdf")
    sources = [
        SimpleNamespace(
            id=sid,
            logical_document_id=sid,
            version=1,
            title=sid,
            url=f"https://unecon.ru/{sid}.pdf",
            source_type="pdf",
            category="test",
            admission_year=2026,
        )
        for sid in source_ids
    ]
    production = faiss.IndexFlatIP(2)
    production.add(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (20, 1)))
    production_meta = {
        "embedding": {"dimension": 2},
        "index": {"file_sha256": "a" * 64},
        "records": [{"chunk_id": f"production-{i}"} for i in range(20)],
    }
    monkeypatch.setattr(table_module, "_production", lambda *_args: (production, production_meta))
    monkeypatch.setattr(table_module, "_sources", lambda _path: sources)
    monkeypatch.setattr(
        table_module,
        "_normalized",
        lambda source, _root: (
            {"source": {"final_url": source.url, "snapshot_sha256": pdf_sha}},
            {"file_sha256": pdf_sha, "page_count": 1, "content_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        table_module,
        "read_pair",
        lambda source, *_args: ({}, pdf),
    )

    def extract(source, final_url, content):
        text = f"Источник: {source.title}\nСтраница: 1\ncolumn_1 = value"
        return {
            "source": {"id": source.id, "final_url": final_url},
            "extraction": {
                "library": "PyMuPDF",
                "version": pymupdf.VersionBind,
                "strategy": "lines_strict",
            },
            "tables": [
                {
                    "rows": [
                        {
                            "row_id": row_id(source.id, pdf_sha, 1, 1, 1, text),
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

    monkeypatch.setattr(table_module, "extract_pdf_tables", extract)

    class Embedder:
        def encode_documents(self, texts):
            assert len(texts) == 3
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1))

    table_dir = tmp_path / "table"
    metadata = build_table_index(
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
        table_dir,
        "test",
        embedder_factory=Embedder,
        originals_root=tmp_path,
    )
    assert [record["vector_id"] for record in metadata["records"]] == [20, 21, 22]
    index, records = validated_table_index(tmp_path, tmp_path, table_dir, production_meta, "test")
    assert index.ntotal == len(records) == 3
    saved_index = (table_dir / "index.faiss").read_bytes()
    saved_metadata = (table_dir / "metadata.json").read_bytes()

    def bad_snapshot(*_args):
        raise ValueError("snapshot SHA-256 mismatch")

    monkeypatch.setattr(table_module, "read_pair", bad_snapshot)
    with pytest.raises(RetrievalError, match="snapshot"):
        build_table_index(
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            table_dir,
            "test",
            embedder_factory=lambda: pytest.fail("model initialized before PDF validation"),
            originals_root=tmp_path,
        )
    assert (table_dir / "index.faiss").read_bytes() == saved_index
    assert (table_dir / "metadata.json").read_bytes() == saved_metadata
    original = json.loads((table_dir / "metadata.json").read_text(encoding="utf-8"))

    def reject_change(change, message):
        altered = json.loads(json.dumps(original))
        change(altered)
        (table_dir / "metadata.json").write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(RetrievalError, match=message):
            validated_table_index(tmp_path, tmp_path, table_dir, production_meta, "test")

    reject_change(lambda m: m["embedding"].update(model="wrong"), "model")
    reject_change(lambda m: m["embedding"].update(dimension=3), "dimension")
    reject_change(lambda m: m["index"].update(file_sha256="0" * 64), "hash")
    reject_change(lambda m: m["index"].update(vector_count=4), "count")
    reject_change(lambda m: m["records"][1].update(vector_id=20), "vector ID")
    reject_change(lambda m: m["records"][1].update(chunk_id=m["records"][0]["chunk_id"]), "row ID")
    reject_change(lambda m: m["sources"][0].update(source_id="other"), "source artifact")
    reject_change(lambda m: m["sources"][0].update(file_sha256="0" * 64), "source artifact")
    reject_change(lambda m: m["records"][0].update(page_start=0), "PDF page")
    reject_change(lambda m: m["records"][0].update(source_id="other"), "unexpected table source")
    (table_dir / "metadata.json").write_text(json.dumps(original), encoding="utf-8")
    stale = {**production_meta, "index": {"file_sha256": "c" * 64}}
    with pytest.raises(RetrievalError, match="stale against production"):
        validated_table_index(tmp_path, tmp_path, table_dir, stale, "test")
    bad_index = faiss.IndexFlatIP(2)
    bad_index.add(np.array([[float("nan"), 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    faiss.write_index(bad_index, str(table_dir / "index.faiss"))
    original["index"]["file_sha256"] = hashlib.sha256(
        (table_dir / "index.faiss").read_bytes()
    ).hexdigest()
    (table_dir / "metadata.json").write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(RetrievalError, match="non-finite"):
        validated_table_index(tmp_path, tmp_path, table_dir, production_meta, "test")
