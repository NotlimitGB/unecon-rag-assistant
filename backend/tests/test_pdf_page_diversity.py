"""Offline invariants for the single frozen PDF-page diversity policy."""

import copy

import pytest

from app.experiments.table_aware.augmented import BASELINE_HITS
from app.experiments.table_aware.diversity import (
    CachedDiversityAdapter,
    retrieval_gate,
    saturation,
    select_with_pdf_page_diversity,
)
from app.experiments.table_aware.diversity_cli import _compare_task017
from app.experiments.table_aware.extract import ExperimentError


def hit(index, source="pdf-a", page=1, representation="production_chunk", source_type="pdf"):
    return {
        "chunk_id": f"item-{index}",
        "vector_id": index,
        "text": f"Текст {index}",
        "source_id": source,
        "source_title": source,
        "final_url": "https://unecon.ru/example.pdf",
        "source_type": source_type,
        "category": "admission",
        "admission_year": 2026,
        "page_start": page,
        "representation": representation,
        "score": float(25 - index),
        "dense_score": float(100 - index),
        "rerank_score": float(25 - index),
        "dense_rank": index + 1,
    }


def test_distinct_pages_unchanged_and_scores_immutable():
    ranked = [hit(i, page=i + 1) for i in range(25)]
    before = copy.deepcopy(ranked)
    choice = select_with_pdf_page_diversity(ranked)
    assert [x["chunk_id"] for x in choice["selected"]] == [f"item-{i}" for i in range(5)]
    assert choice["selected_original_ranks"] == [1, 2, 3, 4, 5]
    assert choice["skipped"] == choice["replacement_pairs"] == []
    assert ranked == before


def test_cap_two_across_representations_and_replacements():
    ranked = [
        hit(i, page=1, representation="table_row" if i % 2 else "production_chunk")
        for i in range(3)
    ]
    ranked += [hit(i, page=i - 1) for i in range(3, 25)]
    choice = select_with_pdf_page_diversity(ranked)
    assert choice["selected_original_ranks"] == [1, 2, 4, 5, 6]
    assert choice["skipped"][0]["original_reranker_rank"] == 3
    assert choice["skipped"][0]["source_id"] == "pdf-a"
    assert choice["skipped"][0]["page"] == 1
    assert choice["replacement_pairs"][0]["replacement"]["original_reranker_rank"] == 6
    assert saturation(choice["selected"])["max_same_pdf_page"] == 2
    assert [h["rerank_score"] for h in choice["selected"]] == [25, 24, 22, 21, 20]


def test_html_unrestricted_and_different_pdf_page_allowed():
    ranked = [hit(i, source="html-a", page=None, source_type="html") for i in range(5)]
    ranked += [hit(i, page=i) for i in range(5, 25)]
    assert len(select_with_pdf_page_diversity(ranked)["selected"]) == 5
    assert select_with_pdf_page_diversity(ranked)["skipped"] == []
    with pytest.raises(ExperimentError):
        select_with_pdf_page_diversity([hit(0, page=None)])


def test_pathological_single_page_stops_at_two_without_fallback():
    ranked = [hit(i) for i in range(25)]
    choice = select_with_pdf_page_diversity(ranked)
    assert len(choice["selected"]) == 2
    assert len(choice["skipped"]) == 23
    assert choice["replacement_pairs"] == []


def test_later_cap_skip_is_recorded_without_inventing_a_replacement():
    ranked = [hit(i) for i in range(4)]
    ranked += [hit(4, page=2), hit(5), hit(6, page=3), hit(7, page=4)]
    ranked += [hit(i, page=i) for i in range(8, 25)]
    choice = select_with_pdf_page_diversity(ranked)
    assert choice["selected_original_ranks"] == [1, 2, 5, 7, 8]
    assert [x["original_reranker_rank"] for x in choice["skipped"]] == [3, 4, 6]
    assert [x["replacement"]["original_reranker_rank"] for x in choice["replacement_pairs"]] == [
        7,
        8,
    ]
    assert [x["skipped"]["original_reranker_rank"] for x in choice["replacement_pairs"]] == [
        3,
        4,
    ]


def test_cached_adapter_preserves_provenance_without_representation():
    ranked = [hit(i, page=i + 1) for i in range(5)]
    adapter = CachedDiversityAdapter({"Вопрос": ranked})
    result = adapter.retrieve("Вопрос")
    assert [h.rank for h in result.results] == [1, 2, 3, 4, 5]
    assert [h.page for h in result.results] == [1, 2, 3, 4, 5]
    assert result.results[0].score == result.results[0].rerank_score
    assert "representation" not in result.model_dump_json()
    assert "vector_id" not in result.model_dump_json()
    assert adapter.history["Вопрос"] == ranked
    with pytest.raises(ExperimentError):
        adapter.retrieve("Другой вопрос")
    with pytest.raises(ExperimentError):
        adapter.retrieve("Вопрос", 6)


def gate_inputs():
    baseline = {
        "primary_denominator": 80,
        "accepted_denominator": 80,
        "page_denominator": 56,
        "primary_mrr_at_5": 0.8046,
        "accepted_mrr_at_5": 0.8208,
        "page_mrr_at_5": 0.6658,
    }
    baseline.update({key.replace("recall", "hits"): value for key, value in BASELINE_HITS.items()})
    ranks = {
        qid: {"primary": 4, "accepted": 4, "page": 4}
        for qid in ("ret-020", "ret-044", "ret-052", "ret-other")
    }
    return {
        "production": baseline,
        "raw_dual_reproduced": True,
        "reranker_unchanged": True,
        "final_sizes_valid": True,
        "mandatory_rows_selected": True,
        "diversity": {"primary_hits_at_5": 73, "accepted_hits_at_5": 74, "page_hits_at_5": 47},
        "table16": {"primary_hits_at_5": 16, "accepted_hits_at_5": 16, "page_hits_at_5": 13},
        "production_ranks": ranks,
        "diversity_ranks": copy.deepcopy(ranks),
        "non_table_ids": {"ret-052", "ret-other"},
        "labeled_ids": {"ret-020", "ret-044", "ret-other"},
    }


@pytest.mark.parametrize(
    ("failed", "change"),
    [
        ("01_production_baseline", ("production", "primary_mrr_at_5", 0.5)),
        ("02_raw_task016", ("raw_dual_reproduced", None, False)),
        ("03_reranker_unchanged", ("reranker_unchanged", None, False)),
        ("04_final_size", ("final_sizes_valid", None, False)),
        ("05_mandatory_rows", ("mandatory_rows_selected", None, False)),
        ("06_overall_primary_r5", ("diversity", "primary_hits_at_5", 69)),
        ("07_overall_accepted_r5", ("diversity", "accepted_hits_at_5", 70)),
        ("08_overall_page_r5", ("diversity", "page_hits_at_5", 46)),
        ("09_table_primary_r5", ("table16", "primary_hits_at_5", 15)),
        ("10_table_accepted_r5", ("table16", "accepted_hits_at_5", 15)),
        ("11_table_page_r5", ("table16", "page_hits_at_5", 12)),
        ("12_non_table_primary_lossless", ("diversity_ranks", "ret-other.primary", None)),
        ("13_non_table_accepted_lossless", ("diversity_ranks", "ret-other.accepted", None)),
        ("14_labeled_page_lossless", ("diversity_ranks", "ret-other.page", None)),
        ("15_ret020_page", ("diversity_ranks", "ret-020.page", None)),
        ("16_ret044_page", ("diversity_ranks", "ret-044.page", None)),
        ("17_ret052_primary", ("diversity_ranks", "ret-052.primary", None)),
    ],
)
def test_each_frozen_gate_can_fail(failed, change):
    kwargs = gate_inputs()
    assert len(retrieval_gate(**kwargs)) == 17
    assert all(retrieval_gate(**kwargs).values())
    field, key, value = change
    if key is None:
        kwargs[field] = value
    elif "." in key:
        qid, label = key.split(".")
        kwargs[field][qid][label] = value
    else:
        kwargs[field][key] = value
    assert retrieval_gate(**kwargs)[failed] is False


def test_recovered_page_does_not_compensate_for_a_new_page_loss():
    kwargs = gate_inputs()
    kwargs["production_ranks"]["ret-044"]["page"] = None
    kwargs["diversity_ranks"]["ret-044"]["page"] = 1
    kwargs["diversity_ranks"]["ret-other"]["page"] = None
    assert kwargs["diversity"]["page_hits_at_5"] >= 47
    assert retrieval_gate(**kwargs)["14_labeled_page_lossless"] is False


def test_task017_invariance_compares_all_25_scores_and_order():
    ranked = [hit(i, page=i + 1) for i in range(25)]
    frozen = [
        {
            "chunk_id": h["chunk_id"],
            "vector_id": h["vector_id"],
            "source_id": h["source_id"],
            "page": h["page_start"],
            "representation": h["representation"],
            "merged_dense_position": i,
            "dense_score": h["dense_score"],
            "reranker_score": h["rerank_score"],
        }
        for i, h in enumerate(ranked, 1)
    ]
    assert _compare_task017(ranked, frozen)
    assert not _compare_task017(ranked[:24], frozen)
    moved = copy.deepcopy(frozen)
    moved[5]["reranker_score"] += 0.01
    assert not _compare_task017(ranked, moved)
