"""Offline diagnostics for the frozen 25-candidate ranking."""

import copy
import math

import numpy as np
import pytest

from app.experiments.table_aware.extract import ExperimentError
from app.experiments.table_aware.reranker_analysis import (
    cutoff_diagnostics,
    distribution,
    group_statistics,
    lexical_diagnostics,
    saturation,
    token_diagnostics,
    trace_candidates,
    validate_patterns,
)
from app.experiments.table_aware.reranker_analysis_cli import (
    _hit_changes,
    _tokenizer,
    main,
)
from app.retrieval.reranker import rerank_candidates


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        assert len(passages) == 25
        return np.arange(25, dtype=np.float64)


class FakeTokenizer:
    def __call__(self, query, passage, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        return {"input_ids": list(range(len(query.split()) + len(passage.split()) + 2))}


def fixture():
    question = {
        "question_id": "ret-020",
        "question": "Сколько мест 179 500?",
        "primary_source_id": "admission-capacity-pdf",
        "acceptable_source_ids": ["admission-capacity-pdf"],
        "expected_pages": [4],
    }
    candidates = [
        {
            "vector_id": i,
            "chunk_id": f"item-{i}",
            "source_id": "admission-capacity-pdf" if i < 20 else "other",
            "page_start": 4 if i == 0 else 1,
            "representation": "production_chunk" if i < 20 else "table_row",
            "text": f"Мест 179 500: {i}",
            "score": 1.0 - i / 100,
        }
        for i in range(25)
    ]
    return question, candidates


def test_full_trace_preserves_ranking_and_provenance():
    question, merged = fixture()
    before = copy.deepcopy(merged)
    model = FakeReranker()
    trace = trace_candidates(question, merged, model, {i: i + 1 for i in range(25)}, {})
    assert model.calls == 1
    assert merged == before
    assert len(trace) == 25
    assert [row["reranker_rank"] for row in trace] == list(range(1, 26))
    assert [row["chunk_id"] for row in trace[:5]] == [f"item-{i}" for i in range(24, 19, -1)]
    assert sum(row["in_final_top5"] for row in trace) == 5
    assert trace[-1]["is_expected_page"]
    assert trace[0]["representation"] == "table_row"
    assert trace[0]["dense_channel_rank"] == 25
    assert trace[0]["merged_dense_position"] == 25
    direct = rerank_candidates(question["question"], merged, FakeReranker(), 5)
    assert [row["chunk_id"] for row in trace[:5]] == [row["chunk_id"] for row in direct]


def test_cutoff_and_saturation():
    question, merged = fixture()
    trace = trace_candidates(question, merged, FakeReranker(), {i: i + 1 for i in range(25)}, {})
    cutoff = cutoff_diagnostics(trace, True)
    assert cutoff["rank1_score"] == 24
    assert cutoff["rank5_score"] == 20
    assert cutoff["rank6_score"] == 19
    assert cutoff["rank5_rank6_margin"] == 1
    assert cutoff["best_expected_page"]["rank"] == 25
    assert cutoff["best_expected_page"]["gap_to_rank5"] == 20
    assert saturation(trace)["three_same_source"]
    assert saturation(trace)["max_same_source_page"] == 5
    assert saturation(trace)["table_ranks_1_5"] == 5
    assert saturation(trace)["table_ranks_6_10"] == 0
    assert saturation(trace)["table_ranks_11_25"] == 0
    for row in trace[:5]:
        row["representation"] = "table_row"
    assert saturation(trace)["three_table_same_source"]
    assert saturation(trace)["three_table_same_page"]
    for row in trace:
        row["is_expected_page"] = False
    assert cutoff_diagnostics(trace, True)["best_expected_page"] is None


def test_statistics_and_empty_group():
    assert distribution([1, 2, 3, 4]) == {
        "count": 4,
        "mean": 2.5,
        "median": 2.5,
        "p25": 1.75,
        "p75": 3.25,
        "minimum": 1,
        "maximum": 4,
    }
    with pytest.raises(ExperimentError):
        distribution([])
    question, merged = fixture()
    trace = trace_candidates(question, merged, FakeReranker(), {i: i + 1 for i in range(25)}, {})
    stats = group_statistics([trace])
    assert stats["table_row"]["scores"]["count"] == 5
    assert stats["production_chunk"]["scores"]["count"] == 20
    assert stats["table_row"]["top5_rate"] == 1
    assert stats["table_row"]["ranks"]["median"] == 3


def test_unicode_lexical_and_numeric_diagnostics():
    result = lexical_diagnostics("МЕСТ мест 2026?", "Мест: 179 500 (50%).")
    assert result["query_token_coverage"] == 0.5
    assert result["numeric_token_count"] == 2
    assert math.isclose(result["numeric_density"], 2 / result["word_count"])
    assert lexical_diagnostics("вопрос", "!!!")["numeric_density"] == 0


def test_exact_token_boundary_with_fake():
    tokenizer = FakeTokenizer()
    assert token_diagnostics(tokenizer, "а б", "в", 5)["would_truncate"] is False
    assert token_diagnostics(tokenizer, "а б", "в г", 5)["would_truncate"] is True
    assert token_diagnostics(tokenizer, "а б", "в г", 5)["tokens_over_limit"] == 1


def test_tokenizer_access_requires_exact_model_limit():
    model = type("Model", (), {"max_seq_length": 512, "tokenizer": FakeTokenizer()})()
    wrapper = type("Wrapper", (), {"model": model})()
    assert _tokenizer(wrapper, 512)[0] is model.tokenizer
    tokenizer, limitation = _tokenizer(wrapper, 256)
    assert tokenizer is None and "differs" in limitation


def test_cli_reports_offline_preflight_error(monkeypatch, capsys):
    def fail():
        raise ExperimentError("frozen dataset changed")

    monkeypatch.setattr("app.experiments.table_aware.reranker_analysis_cli.run", fail)
    assert main(["run"]) == 1
    assert "frozen dataset changed" in capsys.readouterr().err


def test_lost_recovered_and_patterns():
    question, merged = fixture()
    trace = trace_candidates(question, merged, FakeReranker(), {i: i + 1 for i in range(25)}, {})
    lost, recovered = _hit_changes(
        [question],
        {"ret-020": {"primary": 1, "accepted": 1, "page": 1}},
        {"ret-020": {"primary": None, "accepted": None, "page": None}},
        {"ret-020": trace},
        {"ret-020": [merged[0], merged[1], merged[2], merged[3], merged[4]]},
    )
    assert {item["criterion"] for item in lost} == {"primary", "accepted", "page"}
    assert recovered == []
    assert lost[-1]["best_displaced"]["reranker_rank"] == 25
    validate_patterns("candidate_absence", ["source_saturation"])
    with pytest.raises(ExperimentError):
        validate_patterns("candidate_absence", ["candidate_absence"])
