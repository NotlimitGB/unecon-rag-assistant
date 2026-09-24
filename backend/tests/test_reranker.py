"""Offline checks for optional reranking and comparison."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from test_retrieval import FakeEmbedder, build
from test_retrieval import corpus as shared_corpus

from app.config import Settings
from app.evaluation.comparison import compare, render_comparison, write_comparison
from app.retrieval.cli import main as retrieval_main
from app.retrieval.corpus import RetrievalError
from app.retrieval.index import RetrievalSession
from app.retrieval.reranker import RerankedRetrievalSession, rerank_candidates

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/evaluation/retrieval_questions.json"
DATASET_SHA256 = "52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e"


@pytest.fixture(name="corpus")
def local_corpus(tmp_path):
    return shared_corpus.__wrapped__(tmp_path)


class FakeReranker:
    model_name = "fake-reranker"
    device = "cpu"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, passages))
        return self.scores


def test_frozen_dataset_bytes_and_distribution():
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == DATASET_SHA256
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    assert dataset["dataset_id"] == "unecon-retrieval-2026-v1"
    assert len(dataset["questions"]) == 80
    assert Counter(row["difficulty"] for row in dataset["questions"]) == {
        "easy": 25,
        "medium": 35,
        "hard": 20,
    }


def test_rerank_order_provenance_and_dense_regression(corpus):
    build(corpus)
    created = []
    dense = RetrievalSession(
        corpus[0],
        corpus[1],
        corpus[2],
        "fake",
        embedder_factory=lambda: created.append(FakeEmbedder()) or created[-1],
    )
    original = dense.search("правила", 3)
    fake = FakeReranker([0.1, 0.9, 0.9])
    session = RerankedRetrievalSession(dense, candidate_k=20, reranker_factory=lambda: fake)
    hits = session.search("правила", 5)
    assert [hit["dense_rank"] for hit in hits] == [2, 3, 1]
    assert len(hits) == 3
    for hit in hits:
        prior = original[hit["dense_rank"] - 1]
        assert all(hit[key] == value for key, value in prior.items() if key != "score")
        assert hit["dense_score"] == prior["score"]
        assert hit["score"] == hit["rerank_score"]
    assert dense.search("правила", 3) == original
    session.search("правила", 5)
    assert len(created) == 1
    assert len(fake.calls) == 2
    assert fake.calls[0][1] == [hit["text"] for hit in original]
    assert session.pairs_scored == 6


@pytest.mark.parametrize(
    "scores",
    [
        [1],
        [1, 2, 3, 4],
        [float("nan"), 2, 3],
        [float("inf"), 2, 3],
        1.0,
        ["bad", 2, 3],
        ["1", "2", "3"],
        [True, False, True],
    ],
)
def test_bad_scores_rejected(scores):
    candidates = [{"vector_id": i, "text": f"text{i}", "score": float(i)} for i in range(3)]
    with pytest.raises(RetrievalError):
        rerank_candidates("question", candidates, FakeReranker(scores), 3)


def test_candidate_pool_and_validation(corpus):
    build(corpus)
    dense = RetrievalSession(corpus[0], corpus[1], corpus[2], "fake", embedder_factory=FakeEmbedder)
    session = RerankedRetrievalSession(
        dense, candidate_k=5, reranker_factory=lambda: FakeReranker([0.1, 0.2, 0.3])
    )
    assert len(session.search("q")) == 3
    with pytest.raises(RetrievalError):
        session.search(" ")
    with pytest.raises(RetrievalError):
        session.search("q", 6)
    with pytest.raises(RetrievalError):
        session.rerank("q", dense.search("q", 3) * 2)
    with pytest.raises(RetrievalError):
        rerank_candidates("q", [{"score": float("nan"), "text": "x"}], FakeReranker([1]), 1)


@pytest.mark.parametrize("damage", ["stale", "corrupt", "model"])
def test_bad_dense_state_fails_before_reranker(corpus, damage):
    build(corpus)
    if damage == "stale":
        artifact = json.loads((corpus[1] / "faq.json").read_text(encoding="utf-8"))
        artifact["source"]["final_url"] = "https://unecon.ru/changed/"
        (corpus[1] / "faq.json").write_text(json.dumps(artifact), encoding="utf-8")
    elif damage == "corrupt":
        (corpus[2] / "index.faiss").write_bytes(b"bad")
    calls = []
    with pytest.raises(RetrievalError):
        dense = RetrievalSession(
            corpus[0],
            corpus[1],
            corpus[2],
            "other" if damage == "model" else "fake",
            embedder_factory=FakeEmbedder,
        )
        RerankedRetrievalSession(dense, reranker_factory=lambda: calls.append(1))
    assert calls == []


class SyntheticDense:
    metadata = {
        "embedding": {"model": "fake"},
        "index": {"type": "IndexFlatIP", "vector_count": 20},
    }

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[query][:top_k]


def _hit(source, vector_id):
    return {
        "source_id": source,
        "vector_id": vector_id,
        "score": 1 - vector_id / 100,
        "text": source,
        "chunk_id": f"{source}:{vector_id}",
        "page_start": 2,
    }


def test_comparison_metrics_movement_and_reports(tmp_path):
    dataset = {
        "dataset_id": "synthetic",
        "questions": [
            {
                "question_id": "a",
                "question": "qa",
                "category": "entrance_exams",
                "difficulty": "easy",
                "primary_source_id": "target",
                "acceptable_source_ids": ["target"],
                "expected_pages": [2],
            },
            {
                "question_id": "b",
                "question": "qb",
                "category": "entrance_exams",
                "difficulty": "hard",
                "primary_source_id": "target",
                "acceptable_source_ids": ["target"],
                "expected_pages": None,
            },
        ],
    }
    pools = {
        "qa": [_hit("other", i) for i in range(5)]
        + [_hit("target", 5)]
        + [_hit("other", i) for i in range(6, 20)],
        "qb": [_hit("target", 0)] + [_hit("other", i) for i in range(1, 20)],
    }
    dense = SyntheticDense(pools)

    class TargetBoost:
        model_name = "fake"
        device = "cpu"

        def score(self, query, passages):
            if query == "qa":
                return np.array([10 if p == "target" else 0 for p in passages])
            return np.array([0 if p == "target" else 10 for p in passages])

    report, session = compare(dataset, dense, reranker_factory=TargetBoost, require_baseline=False)
    assert dense.calls == [("qa", 20), ("qb", 20)]
    assert session.pairs_scored == 40
    assert report["dense"]["metrics"]["primary_recall_at_5"] == 0.5
    assert report["reranked"]["metrics"]["primary_recall_at_5"] == 0.5
    assert report["candidate_recall"]["primary_candidate_recall_at_20"] == 1
    assert report["candidate_recall"]["page_candidate_recall_at_20"] == 1
    assert report["deltas"]["primary_mrr_at_5"] == 0
    assert [r["status"] for r in report["questions"]] == ["improved", "worsened"]
    markdown = render_comparison(report)
    assert "Восстановленные промахи" in markdown and "Новые промахи" in markdown
    paths = write_comparison(report, tmp_path)
    assert json.loads(paths[0].read_text(encoding="utf-8")) == report
    assert "entrance_exams" in paths[1].read_text(encoding="utf-8")


def test_comparison_stops_on_baseline_mismatch_before_reranker():
    dataset = {
        "dataset_id": "synthetic",
        "questions": [
            {
                "question_id": "ret-001",
                "question": "question",
                "category": "a",
                "difficulty": "easy",
                "primary_source_id": "target",
                "acceptable_source_ids": ["target"],
                "expected_pages": None,
            }
        ],
    }
    pool = [_hit("other", i) for i in range(20)] + [_hit("target", 20)]
    dense = SyntheticDense({"question": pool})
    created = []
    with pytest.raises(RetrievalError, match="baseline"):
        compare(
            dataset,
            dense,
            reranker_factory=lambda: created.append(FakeReranker([])),
        )
    assert dense.calls == [("question", 20)]
    assert created == []


def test_outside_top_20_cannot_be_reranked():
    pool = [_hit("other", i) for i in range(20)] + [_hit("target", 20)]
    dense = SyntheticDense({"question": pool})
    session = RerankedRetrievalSession(
        dense, reranker_factory=lambda: FakeReranker([float(i) for i in range(20)])
    )
    hits = session.search("question")
    assert dense.calls == [("question", 20)]
    assert len(hits) == 5
    assert all(hit["source_id"] != "target" for hit in hits)


def test_optional_cli_keeps_dense_command_separate(monkeypatch, capsys):
    created = []

    class FakeSession:
        def __init__(self, dense, *args):
            created.append(dense)

        def search(self, query, top_k):
            return [
                {
                    "page_start": 2,
                    "source_id": "rules",
                    "chunk_id": "rules:0001:abcdef",
                    "text": "Текст",
                    "rerank_score": 3.0,
                    "dense_score": 0.5,
                    "dense_rank": 4,
                }
            ]

    monkeypatch.setattr("app.retrieval.cli.RetrievalSession", lambda *args: "dense")
    monkeypatch.setattr("app.retrieval.cli.RerankedRetrievalSession", FakeSession)
    assert retrieval_main(["rerank-search", "вопрос"]) == 0
    output = capsys.readouterr().out
    assert created == ["dense"]
    assert "rerank_score=3.000000 dense_rank=4 dense_score=0.500000" in output
    assert "source=rules page=2 chunk_id=rules:0001:abcdef" in output


@pytest.mark.parametrize(
    "values",
    [
        {"RERANKER_MODEL": " "},
        {"RERANKER_DEVICE": "other"},
        {"RERANKER_BATCH_SIZE": 0},
        {"RERANKER_BATCH_SIZE": 65},
        {"RERANKER_MAX_LENGTH": 31},
        {"RERANKER_MAX_LENGTH": 4097},
        {"RERANKER_CANDIDATE_K": 4},
        {"RERANKER_CANDIDATE_K": 101},
    ],
)
def test_invalid_reranker_settings(values):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)
