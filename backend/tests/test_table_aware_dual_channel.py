"""Offline invariants for the frozen dual-channel table experiment."""

import copy
import hashlib
import json

import faiss
import numpy as np
import pytest

from app.experiments.table_aware.augmented import baseline_matches, critical_token
from app.experiments.table_aware.dual_channel import (
    DualChannelRetrievalAdapter,
    build_table_channel,
    merge_candidates,
    new_misses,
    partition_questions,
    preservation_check,
    recovered_misses,
    rerank_merged,
    retrieval_gate,
    save_table_channel,
    search_channels,
)
from app.experiments.table_aware.extract import ExperimentError
from app.ingestion.models import Source


class FakeEmbedder:
    def __init__(self):
        self.queries = 0
        self.documents = []

    def encode_documents(self, texts):
        self.documents.extend(texts)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def encode_query(self, query):
        self.queries += 1
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeReranker:
    def __init__(self):
        self.passages = None

    def score(self, query, passages):
        self.passages = list(passages)
        return np.arange(len(passages), dtype=np.float32)


def corpus():
    production_index = faiss.IndexFlatIP(2)
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (338, 1))
    production_index.add(vectors)
    production = [
        {
            "vector_id": i,
            "chunk_id": f"prod:{i}",
            "text": f"production-{i}",
            "source_id": "admission-capacity-pdf",
            "source_title": "Количество мест",
            "final_url": "https://unecon.ru/a.pdf",
            "source_type": "pdf",
            "category": "capacity",
            "admission_year": 2026,
            "page_start": 1,
        }
        for i in range(338)
    ]
    source = Source(
        id="admission-capacity-pdf",
        title="Количество мест",
        url="https://unecon.ru/a.pdf",
        source_type="pdf",
        category="capacity",
        admission_year=2026,
        active=True,
    )
    artifacts = [
        {
            "source": {"id": source.id, "final_url": "https://unecon.ru/a.pdf"},
            "tables": [
                {
                    "page": 2,
                    "rows": [
                        {
                            "row_id": f"row:{i}",
                            "text": f"table-{i}",
                            "content_sha256": hashlib.sha256(f"table-{i}".encode()).hexdigest(),
                        }
                        for i in range(250)
                    ],
                }
            ],
        }
    ]
    return production_index, production, artifacts, {source.id: source}


def test_two_channels_preserve_production_and_use_one_query_embedding():
    index, production, artifacts, sources = corpus()
    embedder = FakeEmbedder()
    table_index, records = build_table_channel(artifacts, sources, embedder, 2)
    assert len(embedder.documents) == 250
    assert embedder.documents == [f"table-{i}" for i in range(250)]
    assert [record["vector_id"] for record in records] == list(range(338, 588))
    before = copy.deepcopy(production)
    prod, table = search_channels(index, production, table_index, records, embedder, "вопрос")
    assert embedder.queries == 1
    assert production == before
    assert [hit["chunk_id"] for hit in prod] == [f"prod:{i}" for i in range(20)]
    assert [hit["chunk_id"] for hit in table] == [f"row:{i}" for i in range(5)]
    assert preservation_check(prod, prod)
    damaged = copy.deepcopy(prod)
    damaged[0]["score"] -= 1e-3
    assert not preservation_check(prod, damaged)
    merged = merge_candidates(prod, table)
    assert len(merged) == 25
    assert [hit["vector_id"] for hit in merged] == [*range(20), *range(338, 343)]
    assert len({hit["vector_id"] for hit in merged}) == 25
    reranker = FakeReranker()
    final = rerank_merged("вопрос", merged, reranker)
    assert len(reranker.passages) == 25
    assert len(final) == 5
    assert final[0]["score"] == final[0]["rerank_score"] == 24
    assert final[0]["dense_score"] == 1.0
    assert final[0]["representation"] == "table_row"


def test_merge_dense_order_and_ties_are_global_and_immutable():
    production = [
        {"vector_id": i, "representation": "production_chunk", "score": 0.5, "text": "p"}
        for i in range(20)
    ]
    table = [
        {"vector_id": i + 338, "representation": "table_row", "score": 0.5, "text": "t"}
        for i in range(5)
    ]
    table[0]["score"] = 0.6
    before = copy.deepcopy((production, table))
    merged = merge_candidates(production, table)
    assert [hit["vector_id"] for hit in merged] == [338, *range(20), *range(339, 343)]
    assert (production, table) == before
    table[0]["vector_id"] = 0
    with pytest.raises(ExperimentError):
        merge_candidates(production, table)


def test_invalid_table_vectors_and_index_pair(tmp_path):
    _, _, artifacts, sources = corpus()
    embedder = FakeEmbedder()
    index, records = build_table_channel(artifacts, sources, embedder, 2)
    save_table_channel(tmp_path, index, records, "BAAI/bge-m3", "a" * 64)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["vector_count"] == 250
    assert (
        metadata["index_sha256"]
        == hashlib.sha256((tmp_path / "index.faiss").read_bytes()).hexdigest()
    )
    embedder.encode_documents = lambda texts: np.zeros((len(texts), 2), dtype=np.float32)
    with pytest.raises(ValueError):
        build_table_channel(artifacts, sources, embedder, 2)


def test_adapter_keeps_official_provenance_and_hides_representation():
    index, production, artifacts, sources = corpus()
    embedder = FakeEmbedder()
    table_index, records = build_table_channel(artifacts, sources, embedder, 2)
    adapter = DualChannelRetrievalAdapter(
        index, production, table_index, records, embedder, FakeReranker()
    )
    response = adapter.retrieve("  вопрос  ")
    assert response.query == "вопрос"
    assert response.mode == "reranked"
    assert [hit.rank for hit in response.results] == [1, 2, 3, 4, 5]
    assert all(hit.source_url == "https://unecon.ru/a.pdf" for hit in response.results)
    assert all(hit.page == 2 for hit in response.results)
    assert "representation" not in response.model_dump_json()
    assert "vector_id" not in response.model_dump_json()


def baseline():
    return {
        "primary_denominator": 80,
        "accepted_denominator": 80,
        "page_denominator": 56,
        "primary_hits_at_1": 61,
        "primary_hits_at_3": 67,
        "primary_hits_at_5": 70,
        "accepted_hits_at_1": 62,
        "accepted_hits_at_3": 69,
        "accepted_hits_at_5": 71,
        "page_hits_at_1": 32,
        "page_hits_at_3": 41,
        "page_hits_at_5": 47,
        "primary_recall_at_1": 61 / 80,
        "primary_mrr_at_5": 0.8045833333333332,
        "accepted_mrr_at_5": 0.8208333333333334,
        "page_mrr_at_5": 0.6657738095238095,
    }


def test_all_eleven_frozen_gates_fail_independently():
    base = baseline()
    assert baseline_matches(base)
    probes = [{"reranked_row_rank_top5": 1}] * 4
    assert len(retrieval_gate(base, base, base, base, probes, True, [], [])) == 11
    assert all(retrieval_gate(base, base, base, base, probes, True, [], []).values())
    drift = copy.deepcopy(base)
    drift["primary_hits_at_5"] = 69
    assert not retrieval_gate(drift, base, base, base, probes, True, [], [])["baseline_reproduced"]
    assert not retrieval_gate(base, base, base, base, probes, False, [], [])[
        "production_candidates_preserved"
    ]
    assert not retrieval_gate(base, base, base, base, probes[:3], True, [], [])[
        "mandatory_rows_top5"
    ]
    for field, gate_name, value in (
        ("primary_hits_at_5", "primary_r5", 69),
        ("accepted_hits_at_5", "accepted_r5", 70),
        ("page_hits_at_5", "page_r5", 46),
        ("primary_recall_at_1", "primary_r1_guard", 0.7),
        ("primary_mrr_at_5", "primary_mrr_guard", 0.7),
    ):
        dual = copy.deepcopy(base)
        dual[field] = value
        assert not retrieval_gate(base, dual, base, base, probes, True, [], [])[gate_name]
    table_dual = copy.deepcopy(base)
    table_dual["page_hits_at_5"] = 46
    assert not retrieval_gate(base, base, base, table_dual, probes, True, [], [])["table_page_r5"]
    assert not retrieval_gate(base, base, base, base, probes, True, ["ret-001"], [])[
        "non_table_no_new_primary_miss"
    ]
    assert not retrieval_gate(base, base, base, base, probes, True, [], ["ret-001"])[
        "non_table_no_new_accepted_miss"
    ]


def test_new_miss_is_not_offset_by_recovery():
    before = {"a": {"primary": 1, "accepted": 1}, "b": {"primary": None, "accepted": None}}
    after = {"a": {"primary": None, "accepted": None}, "b": {"primary": 1, "accepted": 1}}
    assert new_misses(before, after, {"a", "b"}, "primary") == ["a"]
    assert recovered_misses(before, after, {"a", "b"}, "primary") == ["b"]


def test_exact_question_partition_and_numeric_boundaries():
    questions = [{"question_id": f"ret-{i:03d}"} for i in range(1, 81)]
    selected = {question["question_id"] for question in questions[:16]}
    table, non_table = partition_questions(questions, selected)
    assert len(table) == 16 and len(non_table) == 64
    assert not table & non_table
    assert table | non_table == {question["question_id"] for question in questions}
    with pytest.raises(ExperimentError):
        partition_questions(questions, selected | {"ret-999"})
    for text in ("179500", "179 500", "179\u00a0500", "179\u202f500"):
        assert critical_token(text, "179500")
    for token in ("686", "8", "60"):
        assert critical_token(f"({token})", token)
        assert not critical_token(f"1{token}0", token)
    assert not critical_token("1795000", "179500")
