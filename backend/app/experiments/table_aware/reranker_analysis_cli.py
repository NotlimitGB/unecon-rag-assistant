"""Reproduce Task016 and write a complete, local 25-candidate diagnostic trace."""

import argparse
import json
import sys
from collections import Counter
from typing import Any

import httpx
import pymupdf

from app.config import PROJECT_ROOT, settings
from app.evaluation.dataset import validate_page_labels
from app.evaluation.generation_cli import load_datasets
from app.experiments.table_aware.augmented import (
    EXPECTED_PRODUCTION,
    EXPECTED_ROWS,
    baseline_matches,
    movement,
    summarize_ranks,
)
from app.experiments.table_aware.augmented_cli import (
    GENERATION_SHA,
    RETRIEVAL_SHA,
    _sha,
    _verified_tables,
)
from app.experiments.table_aware.dual_channel import (
    build_table_channel,
    merge_candidates,
    partition_questions,
    preservation_check,
    search_channels,
)
from app.experiments.table_aware.evaluation import select_questions
from app.experiments.table_aware.extract import ExperimentError
from app.experiments.table_aware.reranker_analysis import (
    cutoff_diagnostics,
    group_statistics,
    saturation,
    token_diagnostics,
    trace_candidates,
    validate_patterns,
)
from app.ingestion.manifest import load_manifest
from app.ingestion.writer import write_document
from app.retrieval.index import RetrievalSession
from app.retrieval.reranker import RerankedRetrievalSession

EXPERIMENT_ID = "reranker-competition-v1"
EVALUATED_COMMIT = "f06b6c2b3fba6af50287e9b7b95a21e931edbd1e"
DEEP_IDS = ("ret-020", "ret-044", "ret-052")
EXPECTED_DUAL = {
    "primary_hits_at_1": 63,
    "primary_hits_at_3": 69,
    "primary_hits_at_5": 73,
    "accepted_hits_at_5": 74,
    "page_hits_at_1": 36,
    "page_hits_at_3": 45,
    "page_hits_at_5": 47,
}
EXPECTED_GROUPS = {
    "table16": {"primary_hits_at_5": 16, "page_hits_at_5": 12},
    "non_table64": {
        "primary_hits_at_5": 57,
        "accepted_hits_at_5": 58,
        "page_hits_at_5": 35,
    },
}


def _check_task016(
    questions: list[dict[str, Any]],
    production: dict[str, list[dict[str, Any]]],
    dual: dict[str, list[dict[str, Any]]],
    table_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    production_metrics, production_ranks = summarize_ranks(questions, production)
    if not baseline_matches(production_metrics):
        raise ExperimentError("production baseline drift")
    dual_metrics, dual_ranks = summarize_ranks(questions, dual)
    if any(dual_metrics[key] != expected for key, expected in EXPECTED_DUAL.items()):
        raise ExperimentError("Task016 overall metrics do not reproduce")
    groups = {}
    for name, ids in (
        ("table16", table_ids),
        ("non_table64", {q["question_id"] for q in questions} - table_ids),
    ):
        subset = [q for q in questions if q["question_id"] in ids]
        metrics = summarize_ranks(subset, dual)[0]
        if any(metrics[key] != expected for key, expected in EXPECTED_GROUPS[name].items()):
            raise ExperimentError(f"Task016 {name} metrics do not reproduce")
        groups[name] = metrics
    if [
        q["question_id"]
        for q in questions
        if production_ranks[q["question_id"]]["primary"] is not None
        and dual_ranks[q["question_id"]]["primary"] is None
    ] != ["ret-052"]:
        raise ExperimentError("Task016 new primary miss changed")
    if [
        q["question_id"]
        for q in questions
        if q["question_id"] in table_ids
        and production_ranks[q["question_id"]]["page"] is not None
        and dual_ranks[q["question_id"]]["page"] is None
    ] != ["ret-020", "ret-044"]:
        raise ExperimentError("Task016 lost table pages changed")
    return (
        production_metrics,
        dual_metrics,
        {
            "production": production_ranks,
            "dual": dual_ranks,
            "groups": groups,
        },
    )


def _hit_changes(
    questions: list[dict[str, Any]],
    old: dict[str, dict[str, int | None]],
    new: dict[str, dict[str, int | None]],
    trace: dict[str, list[dict[str, Any]]],
    production_final: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lost, recovered = [], []
    for question in questions:
        question_id = question["question_id"]
        candidates = trace[question_id]
        previous_ids = {item["chunk_id"] for item in production_final[question_id]}
        entering = [
            {
                "chunk_id": item["chunk_id"],
                "source_id": item["source_id"],
                "page": item["page"],
                "representation": item["representation"],
                "reranker_rank": item["reranker_rank"],
            }
            for item in candidates[:5]
            if item["chunk_id"] not in previous_ids
        ]
        for field, predicate in (
            ("primary", lambda item: item["is_primary_source"]),
            ("accepted", lambda item: item["is_acceptable_source"]),
            ("page", lambda item: item["is_expected_page"]),
        ):
            if field == "page" and question["expected_pages"] is None:
                continue
            before, after = old[question_id][field], new[question_id][field]
            if before is not None and after is None:
                best = next((item for item in candidates if predicate(item)), None)
                lost.append(
                    {
                        "question_id": question_id,
                        "criterion": field,
                        "production_rank": before,
                        "best_displaced": None
                        if best is None
                        else {
                            "chunk_id": best["chunk_id"],
                            "reranker_rank": best["reranker_rank"],
                            "reranker_score": best["reranker_score"],
                            "gap_to_cutoff": candidates[4]["reranker_score"]
                            - best["reranker_score"],
                        },
                        "entering_finalists": entering,
                    }
                )
            if before is None and after is not None:
                winner = candidates[after - 1]
                recovered.append(
                    {
                        "question_id": question_id,
                        "criterion": field,
                        "dual_rank": after,
                        "chunk_id": winner["chunk_id"],
                        "representation": winner["representation"],
                        "source_id": winner["source_id"],
                        "page": winner["page"],
                    }
                )
    return lost, recovered


def _deep_case(
    question: dict[str, Any],
    trace: list[dict[str, Any]],
    production: list[dict[str, Any]],
    table: list[dict[str, Any]],
) -> dict[str, Any]:
    cutoff = cutoff_diagnostics(trace, question["expected_pages"] is not None)
    saturated = saturation(trace)
    expected_field = "is_expected_page" if question["expected_pages"] else "is_primary_source"
    expected = [item for item in trace if item[expected_field]]
    winners = trace[:5]
    if not expected:
        primary, contributing = "candidate_absence", []
    elif question["question_id"] == "ret-052" and saturated["three_table_same_source"]:
        primary = "source_saturation"
        contributing = ["expected_evidence_low_reranker_score"]
        if saturated["three_table_same_page"]:
            contributing.append("same_page_row_saturation")
    else:
        primary = "expected_evidence_low_reranker_score"
        contributing = []
        if saturated["three_table_same_source"]:
            contributing.append("source_saturation")
        if saturated["three_table_same_page"]:
            contributing.append("same_page_row_saturation")
    validate_patterns(primary, contributing)
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "primary_source": question["primary_source_id"],
        "acceptable_sources": question["acceptable_source_ids"],
        "expected_pages": question["expected_pages"],
        "candidate_trace": trace,
        "rank5_cutoff_score": cutoff["rank5_score"],
        "cutoff": cutoff,
        "expected_candidate_diagnostics": expected,
        "winning_candidate_diagnostics": winners,
        "source_saturation": saturated,
        "table_channel_siblings": [
            {
                "chunk_id": item["chunk_id"],
                "source_id": item["source_id"],
                "page": item["page_start"],
                "dense_rank": rank,
            }
            for rank, item in enumerate(table, 1)
        ],
        "production_candidates": [
            {
                "chunk_id": item["chunk_id"],
                "source_id": item["source_id"],
                "page": item["page_start"],
                "dense_rank": rank,
            }
            for rank, item in enumerate(production, 1)
            if item["source_id"] == question["primary_source_id"]
            or (question["expected_pages"] and item["page_start"] in question["expected_pages"])
        ],
        "truncation_evidence": [
            {"chunk_id": item["chunk_id"], "token_diagnostics": item["token_diagnostics"]}
            for item in expected
        ],
        "order_128_mentions": (
            [
                {"chunk_id": item["chunk_id"], "mentions_order_128": "№ 128" in item["text"]}
                for item in [*winners, *expected]
            ]
            if question["question_id"] == "ret-052"
            else None
        ),
        "primary_pattern": primary,
        "contributing_patterns": contributing,
        "evidence_summary": (
            f"best expected rank {expected[0]['reranker_rank']} score "
            f"{expected[0]['reranker_score']:.6f}; rank-5 cutoff "
            f"{cutoff['rank5_score']:.6f}; table rows in top-5 "
            f"{saturated['table_ranks_1_5']}"
            if expected
            else "Expected evidence absent from the merged 25 candidates."
        ),
    }


def _tokenizer(reranker: Any, max_length: int) -> tuple[Any | None, str | None]:
    model = getattr(reranker, "model", None)
    if getattr(model, "max_seq_length", None) != max_length:
        return None, "CrossEncoder max sequence length differs from configured limit"
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        return None, "CrossEncoder tokenizer unavailable; exact token lengths not claimed"
    return tokenizer, None


def _table_family(table: list[dict[str, Any]]) -> dict[str, int]:
    sources = Counter(hit["source_id"] for hit in table)
    pages = Counter((hit["source_id"], hit["page_start"]) for hit in table)
    return {
        "distinct_sources": len(sources),
        "max_same_source": max(sources.values()),
        "distinct_source_pages": len(pages),
        "max_same_source_page": max(pages.values()),
    }


def run() -> dict[str, Any]:
    expected = {
        "embedding_model": "BAAI/bge-m3",
        "embedding_batch_size": 16,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_batch_size": 8,
        "reranker_max_length": 512,
        "reranker_candidate_k": 20,
    }
    if any(getattr(settings, key) != value for key, value in expected.items()):
        raise ExperimentError("accepted Task016 model or reranker configuration changed")
    retrieval_path = PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
    generation_path = PROJECT_ROOT / "data/evaluation/generation_questions.json"
    if _sha(retrieval_path) != RETRIEVAL_SHA or _sha(generation_path) != GENERATION_SHA:
        raise ExperimentError("frozen dataset hash changed")
    generation, retrieval = load_datasets(generation_path, retrieval_path)
    questions = retrieval["questions"]
    if len(questions) != 80 or len({q["question_id"] for q in questions}) != 80:
        raise ExperimentError("frozen question set changed")
    manifest_path = PROJECT_ROOT / "data/source_manifest.json"
    manifest = load_manifest(manifest_path)
    selected = select_questions(generation, retrieval)
    table_ids = {
        item["retrieval_question_id"]
        for item in generation["questions"]
        if item["question_id"] in {entry["question_id"] for entry in selected}
    }
    table_ids, other_ids = partition_questions(questions, table_ids)
    artifacts, source_checks = _verified_tables(manifest)
    sources = {source.id: source for source in manifest.sources}
    row_lookup = {
        row["row_id"]: row
        for artifact in artifacts
        for table in artifact["tables"]
        for row in table["rows"]
    }
    if len(row_lookup) != EXPECTED_ROWS:
        raise ExperimentError("table row ID drift")
    dense = RetrievalSession(
        manifest_path,
        PROJECT_ROOT / "data/processed/chunks",
        PROJECT_ROOT / "data/processed/index",
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
    )
    if len(dense.records) != EXPECTED_PRODUCTION or dense.index.ntotal != EXPECTED_PRODUCTION:
        raise ExperimentError("production corpus count drift")
    validate_page_labels(retrieval, dense.records)
    table_index, table_records = build_table_channel(
        artifacts, sources, dense.embedder, dense.index.d
    )
    if len(table_records) != EXPECTED_ROWS:
        raise ExperimentError("table corpus count drift")
    reranked = RerankedRetrievalSession(
        dense,
        settings.reranker_model,
        settings.reranker_device,
        settings.reranker_batch_size,
        settings.reranker_max_length,
        20,
    )
    tokenizer, token_limitation = _tokenizer(reranked.reranker, settings.reranker_max_length)
    if tokenizer is not None:
        try:
            token_diagnostics(tokenizer, "проверка", "текст", settings.reranker_max_length)
        except (KeyError, TypeError, ValueError) as exc:
            tokenizer = None
            token_limitation = f"Exact tokenizer diagnostics unavailable: {exc}"
    prior_path = (
        PROJECT_ROOT / "data/processed/evaluation/dual_channel_table_retrieval_experiment.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("experiment_id") != "table-aware-dual-channel-v1":
        raise ExperimentError("Task016 local report unavailable or changed")
    prior_by_id = {item["question_id"]: item for item in prior["competition"]}
    if len(prior_by_id) != 80:
        raise ExperimentError("Task016 report question count changed")
    production_final, dual_final, traces = {}, {}, {}
    production_channels, table_channels = {}, {}
    for position, question in enumerate(questions, 1):
        question_id, query = question["question_id"], question["question"]
        production, table = search_channels(
            dense.index, dense.records, table_index, table_records, dense.embedder, query
        )
        canonical = dense.search(query, 20)
        if not preservation_check(production, canonical):
            raise ExperimentError(f"production top-20 not preserved: {question_id}")
        production_channels[question_id], table_channels[question_id] = production, table
        production_final[question_id] = reranked.rerank(query, canonical, 5)
        merged = merge_candidates(production, table)
        channel_ranks = {
            hit["vector_id"]: rank
            for channel in (production, table)
            for rank, hit in enumerate(channel, 1)
        }
        trace = trace_candidates(
            question,
            merged,
            reranked.reranker,
            channel_ranks,
            row_lookup,
            tokenizer,
            settings.reranker_max_length,
        )
        traces[question_id] = trace
        dual_final[question_id] = [
            next(hit for hit in merged if hit["vector_id"] == item["vector_id"])
            for item in trace[:5]
        ]
        frozen = prior_by_id[question_id]["final_top5"]
        if [item["chunk_id"] for item in trace[:5]] != [item["chunk_id"] for item in frozen]:
            raise ExperimentError(f"Task016 final top-5 changed: {question_id}")
        if any(
            abs(item["reranker_score"] - old["rerank_score"]) > 1e-4
            for item, old in zip(trace[:5], frozen, strict=True)
        ):
            raise ExperimentError(f"Task016 reranker scores changed: {question_id}")
        if position % 10 == 0:
            print(f"evaluated={position}/80", flush=True)
    production_metrics, dual_metrics, ranks = _check_task016(
        questions, production_final, dual_final, table_ids
    )
    lost, recovered = _hit_changes(
        questions, ranks["production"], ranks["dual"], traces, production_final
    )
    by_id = {item["question_id"]: item for item in questions}
    deep = {
        question_id: _deep_case(
            by_id[question_id],
            traces[question_id],
            production_channels[question_id],
            table_channels[question_id],
        )
        for question_id in DEEP_IDS
    }
    groups = {
        name: group_statistics(
            [
                traces[q["question_id"]]
                for q in questions
                if name == "all80" or q["question_id"] in ids
            ]
        )
        for name, ids in (("all80", set()), ("table16", table_ids), ("non_table64", other_ids))
    }
    saturations = {question_id: saturation(trace) for question_id, trace in traces.items()}
    table_winner_ids = {
        question_id
        for question_id, trace in traces.items()
        if any(item["representation"] == "table_row" for item in trace[:5])
    }
    shifts = movement(ranks["production"], ranks["dual"])
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluated_commit": EVALUATED_COMMIT,
        "runtime": {
            "embedding_device": dense.embedder.device,
            "reranker_device": reranked.reranker.device,
            "reranker_model": settings.reranker_model,
            "reranker_max_length": settings.reranker_max_length,
            "tokenizer_class": type(tokenizer).__name__ if tokenizer else None,
            "tokenizer_limitation": token_limitation,
        },
        "source_verification": source_checks,
        "corpus_counts": {
            "production_chunks": 338,
            "table_rows": 250,
            "questions": 80,
            "candidates": 2000,
        },
        "candidate_trace": [
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "group": "table16" if q["question_id"] in table_ids else "non_table64",
                "candidates": traces[q["question_id"]],
                "cutoff": cutoff_diagnostics(
                    traces[q["question_id"]], q["expected_pages"] is not None
                ),
                "saturation": saturations[q["question_id"]],
                "table_channel_family": _table_family(table_channels[q["question_id"]]),
            }
            for q in questions
        ],
        "global_statistics": {
            "representation": groups,
            "metrics": {
                "production": production_metrics,
                "dual": dual_metrics,
                "table16": ranks["groups"]["table16"],
                "non_table64": ranks["groups"]["non_table64"],
            },
            "saturation_ids": {
                field: [qid for qid, value in saturations.items() if value[field]]
                for field in (
                    "three_same_source",
                    "three_table_same_source",
                    "three_table_same_page",
                )
            },
            "table_winner_question_ids": sorted(table_winner_ids),
            "table_winner_movement": {
                field: {
                    bucket: sorted(set(ids) & table_winner_ids) for bucket, ids in value.items()
                }
                for field, value in shifts.items()
            },
            "rank_movement": shifts,
            "production_top20_preserved": 80,
        },
        "lost_hits": lost,
        "recovered_hits": recovered,
        "deep_cases": deep,
        "conclusions": {
            "scope": "descriptive reranker-stage diagnostics; no semantic scoring",
            "task016_reproduced": True,
        },
    }
    write_document(
        PROJECT_ROOT / "data/processed/evaluation", "reranker_competition_analysis", report
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze frozen dual-channel reranker competition")
    parser.add_argument("command", choices=["run"])
    parser.parse_args(argv)
    try:
        report = run()
        print(f"questions={report['corpus_counts']['questions']} candidates=2000")
        print("task016_reproduced=true")
        return 0
    except (ExperimentError, ValueError, OSError, httpx.HTTPError, pymupdf.FileDataError) as exc:
        print(f"reranker analysis stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
