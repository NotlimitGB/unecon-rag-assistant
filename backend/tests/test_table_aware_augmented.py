"""Offline contract checks for the isolated augmented experiment."""

import copy

import faiss
import numpy as np
import pytest

from app.experiments.table_aware.augmented import (
    AugmentedRetrievalAdapter,
    baseline_matches,
    build_augmented,
    compare_generation,
    critical_token,
    dense_search,
    gate,
    movement,
    rerank,
    save_augmented,
    summarize_ranks,
    table_records,
)
from app.experiments.table_aware.extract import ExperimentError
from app.ingestion.models import Source


class FakeEmbedder:
    def __init__(self, dimension=2):
        self.dimension = dimension
        self.documents = []

    def encode_documents(self, texts):
        self.documents.extend(texts)
        return np.tile(np.eye(1, self.dimension, dtype=np.float32), (len(texts), 1))

    def encode_query(self, query):
        return np.eye(1, self.dimension, dtype=np.float32)[0]


class FakeReranker:
    def score(self, query, passages):
        return np.arange(len(passages), dtype=np.float32)


def corpus():
    index = faiss.IndexFlatIP(2)
    vectors = np.tile(np.array([[0, 1]], dtype=np.float32), (338, 1))
    vectors[0] = [1, 0]
    index.add(vectors)
    production = [{"vector_id": i, "chunk_id": f"prod:{i}", "text": f"old-{i}"} for i in range(338)]
    rows = [{"vector_id": i + 338, "chunk_id": f"row:{i}", "text": f"new-{i}"} for i in range(250)]
    return index, vectors, production, rows


def test_reconstructs_existing_vectors_and_embeds_only_rows():
    old, vectors, production, rows = corpus()
    embedder = FakeEmbedder()
    augmented, records = build_augmented(old, production, rows, embedder)
    assert embedder.documents == [r["text"] for r in rows]
    np.testing.assert_array_equal(augmented.reconstruct_n(0, 338), vectors)
    assert augmented.ntotal == 588
    assert [r["vector_id"] for r in records] == list(range(588))
    assert records[0]["representation"] == "production_chunk"


def test_saved_index_retains_hash_and_record_order(tmp_path):
    import hashlib
    import json

    old, _, production, rows = corpus()
    augmented, records = build_augmented(old, production, rows, FakeEmbedder())
    save_augmented(tmp_path, augmented, records, "BAAI/bge-m3", "a" * 64)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert (
        metadata["index_sha256"]
        == hashlib.sha256((tmp_path / "index.faiss").read_bytes()).hexdigest()
    )
    assert metadata["vector_count"] == 588
    assert metadata["records"][337]["chunk_id"] == "prod:337"
    assert metadata["records"][338]["chunk_id"] == "row:0"


def test_table_rows_keep_manifest_and_page_provenance():
    source = Source(
        id="admission-capacity-pdf",
        title="Количество мест",
        url="https://unecon.ru/a.pdf",
        source_type="pdf",
        category="capacity",
        admission_year=2026,
        active=True,
    )
    artifact = {
        "source": {"id": source.id, "final_url": "https://unecon.ru/final.pdf"},
        "tables": [
            {
                "page": 3,
                "rows": [
                    {"row_id": f"row:{i}", "text": f"row-{i}", "content_sha256": "a" * 64}
                    for i in range(250)
                ],
            }
        ],
    }
    records = table_records([artifact], {source.id: source})
    assert records[0]["vector_id"] == 338
    assert records[-1]["vector_id"] == 587
    assert records[0]["page_start"] == records[0]["page_end"] == 3
    assert records[0]["final_url"] == "https://unecon.ru/final.pdf"
    assert records[0]["category"] == "capacity"
    assert records[0]["content_sha256"] == "a" * 64


@pytest.mark.parametrize("broken", ["count", "order", "zero", "nan"])
def test_rejects_invalid_corpus_or_vectors(broken):
    old, _, production, rows = corpus()
    embedder = FakeEmbedder()
    if broken == "count":
        rows.pop()
    if broken == "order":
        rows[0]["vector_id"] = 999
    if broken in {"zero", "nan"}:
        embedder.encode_documents = lambda texts: np.full(
            (len(texts), 2), np.nan if broken == "nan" else 0, dtype=np.float32
        )
    with pytest.raises((ExperimentError, ValueError)):
        build_augmented(old, production, rows, embedder)


def test_dense_ties_use_vector_id_and_reranker_uses_only_candidates():
    old, _, production, rows = corpus()
    index, records = build_augmented(old, production, rows, FakeEmbedder())
    hits = dense_search(index, records, FakeEmbedder(), "вопрос")
    assert len(hits) == 20
    assert hits[0]["vector_id"] == 0
    assert [h["vector_id"] for h in hits[1:]] == list(range(338, 357))
    ranked = rerank("вопрос", hits, FakeReranker())
    assert [h["dense_rank"] for h in ranked] == [20, 19, 18, 17, 16]
    assert ranked[0]["score"] == ranked[0]["rerank_score"]
    assert ranked[0]["dense_score"] == hits[19]["score"]


def test_adapter_retains_official_pdf_provenance():
    index = faiss.IndexFlatIP(2)
    index.add(np.tile(np.array([[1, 0]], dtype=np.float32), (20, 1)))
    records = [
        {
            "vector_id": i,
            "chunk_id": f"row:{i}",
            "text": "Табличная строка",
            "source_id": "admission-capacity-pdf",
            "source_title": "Количество мест",
            "final_url": "https://unecon.ru/a.pdf",
            "source_type": "pdf",
            "category": "capacity",
            "admission_year": 2026,
            "page_start": 2,
            "representation": "table_row",
        }
        for i in range(20)
    ]
    response = AugmentedRetrievalAdapter(index, records, FakeEmbedder(), FakeReranker()).retrieve(
        "  вопрос  "
    )
    assert response.mode == "reranked"
    assert response.query == "вопрос"
    assert [r.rank for r in response.results] == [1, 2, 3, 4, 5]
    assert all(r.page == 2 and r.source_url == "https://unecon.ru/a.pdf" for r in response.results)
    assert "representation" not in response.model_dump_json()
    assert "vector_id" not in response.model_dump_json()


def test_rank_metrics_and_movement():
    questions = [
        {
            "question_id": "a",
            "primary_source_id": "x",
            "acceptable_source_ids": ["x"],
            "expected_pages": [2],
        },
        {
            "question_id": "b",
            "primary_source_id": "y",
            "acceptable_source_ids": ["y", "z"],
            "expected_pages": None,
        },
    ]
    hits = {
        "a": [{"source_id": "x", "page_start": 2}],
        "b": [{"source_id": "z", "page_start": None}],
    }
    metrics, ranks = summarize_ranks(questions, hits)
    assert metrics["primary_recall_at_1"] == 0.5
    assert metrics["accepted_recall_at_1"] == 1
    assert metrics["page_denominator"] == 1
    assert metrics["page_hits_at_5"] == 1
    later = copy.deepcopy(ranks)
    later["b"]["primary"] = 2
    assert movement(ranks, later)["primary"]["improved"] == ["b"]


def test_gate_is_frozen_and_rejects_each_guardrail():
    base = {
        "primary_hits_at_5": 70,
        "accepted_hits_at_5": 71,
        "page_hits_at_5": 47,
        "primary_recall_at_1": 61 / 80,
        "primary_mrr_at_5": 0.8045833333333332,
        "primary_denominator": 80,
        "accepted_denominator": 80,
        "page_denominator": 56,
        "primary_hits_at_1": 61,
        "primary_hits_at_3": 67,
        "accepted_hits_at_1": 62,
        "accepted_hits_at_3": 69,
        "page_hits_at_1": 32,
        "page_hits_at_3": 41,
        "accepted_mrr_at_5": 0.8208333333333334,
        "page_mrr_at_5": 0.6657738095238095,
    }
    assert baseline_matches(base)
    drifted = copy.deepcopy(base)
    drifted["primary_hits_at_5"] = 69
    assert not baseline_matches(drifted)
    probes = [{"reranked_row_rank_top5": 1}] * 4
    checks = gate(base, base, base, base, base, base, probes)
    assert len(checks) == 9 and all(checks.values())
    for field, key, value in (
        ("overall", "primary_hits_at_5", 69),
        ("overall", "accepted_hits_at_5", 70),
        ("overall", "page_hits_at_5", 46),
        ("overall", "primary_recall_at_1", 0.7),
        ("overall", "primary_mrr_at_5", 0.7),
        ("table", "page_hits_at_5", 46),
        ("non", "primary_hits_at_5", 69),
    ):
        changed = copy.deepcopy(base)
        changed[key] = value
        args = {
            "overall": (base, changed, base, base, base, base),
            "table": (base, base, base, changed, base, base),
            "non": (base, base, base, base, base, changed),
        }[field]
        assert not all(gate(*args, probes).values())
    assert not gate(base, base, base, base, base, base, probes[:3])["mandatory_rows_top5"]


def test_critical_tokens_and_generation_comparison_preserve_answers():
    assert critical_token("Стоимость 179 500 рублей", "179500")
    assert not critical_token("1795000", "179500")
    ids = ["gen-008", "gen-011", "gen-021", "gen-027"] + [f"gen-{i:03d}" for i in range(40, 52)]
    baseline = [{"question_id": q, "actual_status": "answered", "answer": "исходный"} for q in ids]
    augmented = [
        {
            "question_id": q,
            "actual_status": "answered",
            "answer": "686 8 60 179 500",
            "elapsed_seconds": 1.0,
        }
        for q in ids
    ]
    original = copy.deepcopy(baseline)
    result = compare_generation(baseline, augmented)
    assert all(p["critical_fact_token_present"] for p in result["critical_probes"])
    assert result["status_movement"]["remained_answered"] == 16
    assert baseline == original
    assert "semantic_score" not in str(result)
