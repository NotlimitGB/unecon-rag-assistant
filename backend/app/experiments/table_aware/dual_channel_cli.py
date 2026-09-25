"""Run the frozen 20+5 dual-channel experiment and conditional generation."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from app.config import PROJECT_ROOT, settings
from app.evaluation.dataset import validate_page_labels
from app.evaluation.generation_cli import check_baseline, load_datasets, preflight_ollama
from app.evaluation.generation_runner import RecordingRetrieval, evaluate_generation
from app.experiments.table_aware.augmented import (
    EXPECTED_PRODUCTION,
    EXPECTED_ROWS,
    baseline_matches,
    compare_generation,
    movement,
    probe_results,
    representation_counts,
    summarize_ranks,
)
from app.experiments.table_aware.augmented_cli import (
    BASELINE_SHA,
    GENERATION_SHA,
    RETRIEVAL_SHA,
    _sha,
    _structural_metrics,
    _verified_tables,
)
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
from app.experiments.table_aware.evaluation import EXPECTED_IDS, PROBE_IDS, select_questions
from app.experiments.table_aware.extract import ExperimentError
from app.generation.service import AnswerService
from app.ingestion.manifest import load_manifest
from app.ingestion.writer import write_document
from app.retrieval.index import RetrievalSession
from app.retrieval.reranker import RerankedRetrievalSession

EXPERIMENT_ID = "table-aware-dual-channel-v1"


def _hit_summary(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": hit["chunk_id"],
            "source_id": hit["source_id"],
            "page": hit["page_start"],
            "dense_score": hit.get("dense_score", hit["score"]),
            "rerank_score": hit.get("rerank_score"),
            "representation": hit["representation"],
        }
        for rank, hit in enumerate(hits, 1)
    ]


def _source_distribution(
    ids: set[str], table_hits: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                hit["source_id"] for question_id in ids for hit in table_hits[question_id]
            ).items()
        )
    )


def _comparison_ids(
    before: dict[str, dict[str, int | None]],
    after: dict[str, dict[str, int | None]],
    ids: set[str],
) -> dict[str, list[str]]:
    return {
        "new_primary_misses": new_misses(before, after, ids, "primary"),
        "recovered_primary_misses": recovered_misses(before, after, ids, "primary"),
        "new_accepted_misses": new_misses(before, after, ids, "accepted"),
        "new_page_misses": new_misses(before, after, ids, "page"),
    }


def _generation_phase(
    generation: dict[str, Any],
    retrieval: dict[str, Any],
    adapter: DualChannelRetrievalAdapter,
    probes: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    check_baseline()
    baseline_path = output_dir / "generation_report.json"
    if _sha(baseline_path) != BASELINE_SHA:
        raise ExperimentError("Task011 baseline report SHA mismatch")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = [row for row in baseline["questions"] if row["question_id"] in EXPECTED_IDS]
    if (
        len(baseline_rows) != 16
        or sum(row["actual_status"] == "answered" for row in baseline_rows) != 12
        or sum(row["actual_status"] == "insufficient_evidence" for row in baseline_rows) != 4
    ):
        raise ExperimentError("Task011 16-question status baseline mismatch")
    preflight_ollama()
    subset = {
        **generation,
        "questions": [
            question
            for question in generation["questions"]
            if question["question_id"] in EXPECTED_IDS
        ],
    }
    recording = RecordingRetrieval(adapter)
    service = AnswerService(retrieval_service=recording, config=settings)
    config = {
        "model": settings.ollama_model,
        "prompt_version": "grounded-answer-v1",
        "temperature": settings.generation_temperature,
        "max_tokens": settings.generation_max_tokens,
        "think": False,
        "retrieval_mode": "reranked",
        "retrieval_top_k": 5,
    }
    try:
        evaluated = evaluate_generation(subset, retrieval, service, recording, config)
    finally:
        service.close()
    comparison = compare_generation(baseline_rows, evaluated["questions"])
    expected_rows = {probe["question_id"]: probe["row_id"] for probe in probes}
    for row in comparison["questions"]:
        hits = adapter.history.get(row["augmented"]["question"], [])
        row["dual_contexts"] = [
            {
                "context_id": f"C{rank}",
                "chunk_id": hit["chunk_id"],
                "representation": hit["representation"],
                "text": hit["text"],
            }
            for rank, hit in enumerate(hits, 1)
        ]
        cited_ids = {citation["chunk_id"] for citation in row["augmented"]["citations"]}
        row["table_row_cited"] = any(
            context["representation"] == "table_row" and context["chunk_id"] in cited_ids
            for context in row["dual_contexts"]
        )
        if row["question_id"] in expected_rows:
            expected_id = expected_rows[row["question_id"]]
            expected_context = next(
                (context for context in row["dual_contexts"] if context["chunk_id"] == expected_id),
                None,
            )
            row["mandatory_row_retrieved"] = expected_context is not None
            row["mandatory_row_cited"] = expected_id in cited_ids
            if row["question_id"] == "gen-027":
                row["mandatory_row_contains_semester"] = bool(
                    expected_context and "семестр" in expected_context["text"].casefold()
                )
    baseline_metrics = _structural_metrics(baseline_rows)
    dual_metrics = _structural_metrics(evaluated["questions"])
    for label, rows, metrics in (
        ("baseline", baseline_rows, baseline_metrics),
        ("dual", evaluated["questions"], dual_metrics),
    ):
        metrics["answered_with_citation"] = sum(
            row["success"] and row["actual_status"] == "answered" and bool(row["citations"])
            for row in rows
        )
        if label == "dual":
            metrics["latency_mean_seconds"] = comparison["augmented_latency_mean"]
            metrics["latency_median_seconds"] = comparison["augmented_latency_median"]
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "baseline_report_sha256": BASELINE_SHA,
        "configuration": config,
        "baseline_metrics": baseline_metrics,
        "dual_metrics": dual_metrics,
        **comparison,
    }
    write_document(output_dir, "dual_channel_table_generation_experiment", report)
    return report


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
        raise ExperimentError("accepted embedding or reranker configuration changed")
    retrieval_path = PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
    generation_path = PROJECT_ROOT / "data/evaluation/generation_questions.json"
    if _sha(retrieval_path) != RETRIEVAL_SHA or _sha(generation_path) != GENERATION_SHA:
        raise ExperimentError("frozen dataset hash changed")
    generation, retrieval = load_datasets(generation_path, retrieval_path)
    manifest_path = PROJECT_ROOT / "data/source_manifest.json"
    manifest = load_manifest(manifest_path)
    selected = select_questions(generation, retrieval)
    artifacts, source_checks = _verified_tables(manifest)
    sources = {source.id: source for source in manifest.sources}
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
    production_final: dict[str, list[dict[str, Any]]] = {}
    dual_final: dict[str, list[dict[str, Any]]] = {}
    table_dense: dict[str, list[dict[str, Any]]] = {}
    merged_by_id: dict[str, list[dict[str, Any]]] = {}
    candidate_checks = []
    for position, question in enumerate(retrieval["questions"], 1):
        question_id, text = question["question_id"], question["question"]
        production, table = search_channels(
            dense.index, dense.records, table_index, table_records, dense.embedder, text
        )
        canonical = dense.search(text, 20)
        preserved = preservation_check(production, canonical)
        candidate_checks.append({"question_id": question_id, "preserved": preserved})
        if not preserved:
            raise ExperimentError(f"production top-20 differs from canonical: {question_id}")
        table_dense[question_id] = table
        production_final[question_id] = reranked.rerank(text, canonical, 5)
        merged = merge_candidates(production, table)
        merged_by_id[question_id] = merged
        dual_final[question_id] = rerank_merged(text, merged, reranked.reranker)
        if position % 10 == 0:
            print(f"evaluated={position}/80", flush=True)
    questions = retrieval["questions"]
    production_metrics, production_ranks = summarize_ranks(questions, production_final)
    if not baseline_matches(production_metrics):
        raise ExperimentError("production baseline does not reproduce accepted metrics")
    dual_metrics, dual_ranks = summarize_ranks(questions, dual_final)
    table_ids = {
        question["retrieval_question_id"]
        for question in generation["questions"]
        if question["question_id"] in {item["question_id"] for item in selected}
    }
    table_ids, non_table_ids = partition_questions(questions, table_ids)
    all_ids = table_ids | non_table_ids
    groups = {}
    for name, ids in (("table16", table_ids), ("non_table64", non_table_ids)):
        subset = [question for question in questions if question["question_id"] in ids]
        groups[name] = {
            "production": summarize_ranks(subset, production_final)[0],
            "dual": summarize_ranks(subset, dual_final)[0],
        }
    generation_to_retrieval = {
        question["question_id"]: question["retrieval_question_id"]
        for question in generation["questions"]
        if question["retrieval_question_id"]
    }
    probe_map = {question_id: generation_to_retrieval[question_id] for question_id in PROBE_IDS}
    probes = probe_results(artifacts, table_dense, dual_final, probe_map)
    for probe in probes:
        question_id = probe_map[probe["question_id"]]
        probe["table_channel_rank_top5"] = probe.pop("dense_row_rank_top20")
        probe["merged_dense_position_top25"] = next(
            (
                rank
                for rank, hit in enumerate(merged_by_id[question_id], 1)
                if hit["chunk_id"] == probe["row_id"]
            ),
            None,
        )
    non_table_changes = _comparison_ids(production_ranks, dual_ranks, non_table_ids)
    gate_results = retrieval_gate(
        production_metrics,
        dual_metrics,
        groups["table16"]["production"],
        groups["table16"]["dual"],
        probes,
        all(result["preserved"] for result in candidate_checks),
        non_table_changes["new_primary_misses"],
        non_table_changes["new_accepted_misses"],
    )
    competition = [
        {
            "question_id": question["question_id"],
            "group": "table16" if question["question_id"] in table_ids else "non_table64",
            "table_channel_top5": _hit_summary(table_dense[question["question_id"]]),
            "final_top5": _hit_summary(dual_final[question["question_id"]]),
            "final_representation_counts": representation_counts(
                dual_final[question["question_id"]]
            ),
        }
        for question in questions
    ]
    occupancy = {
        name: sum(
            row["final_representation_counts"]["table_row"]
            for row in competition
            if name == "all80" or row["group"] == name
        )
        / (80 if name == "all80" else 16 if name == "table16" else 64)
        for name in ("all80", "table16", "non_table64")
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluated_commit": "807c263aff36fff9a73702de304de18b9d4044d1",
        "source_checks": source_checks,
        "counts": {"production_chunks": len(dense.records), "table_rows": len(table_records)},
        "allocation": {"production_k": 20, "table_k": 5, "merged_k": 25, "final_k": 5},
        "devices": {"embedding": dense.embedder.device, "reranker": reranked.reranker.device},
        "baseline_reproduced": True,
        "candidate_preservation": candidate_checks,
        "metrics": {"production": production_metrics, "dual": dual_metrics},
        "groups": groups,
        "movement": movement(production_ranks, dual_ranks),
        "non_table_changes": non_table_changes,
        "question_ranks": [
            {
                "question_id": question["question_id"],
                "production": production_ranks[question["question_id"]],
                "dual": dual_ranks[question["question_id"]],
            }
            for question in questions
        ],
        "table_source_distribution": {
            "all80": _source_distribution(all_ids, table_dense),
            "table16": _source_distribution(table_ids, table_dense),
            "non_table64": _source_distribution(non_table_ids, table_dense),
        },
        "competition": competition,
        "mean_table_rows_final_top5": occupancy,
        "task015_mean_table_rows_final_top5": {
            "all80": 1.0125,
            "table16": 2.8125,
            "non_table64": 0.5625,
        },
        "probes": probes,
        "gate": gate_results,
        "phase_a_verdict": "dual_channel_retrieval_gate_passed"
        if all(gate_results.values())
        else "dual_channel_retrieval_gate_failed",
    }
    output_dir = PROJECT_ROOT / "data/processed/evaluation"
    save_table_channel(
        PROJECT_ROOT / "data/processed/experiments/table_aware_dual_channel",
        table_index,
        table_records,
        settings.embedding_model,
        dense.metadata["index"]["file_sha256"],
    )
    write_document(output_dir, "dual_channel_table_retrieval_experiment", report)
    generation_report = None
    generation_blocked = None
    if report["phase_a_verdict"] == "dual_channel_retrieval_gate_passed":
        adapter = DualChannelRetrievalAdapter(
            dense.index,
            dense.records,
            table_index,
            table_records,
            dense.embedder,
            reranked.reranker,
        )
        try:
            generation_report = _generation_phase(
                generation, retrieval, adapter, probes, output_dir
            )
        except (ExperimentError, ValueError, OSError, httpx.HTTPError) as exc:
            generation_blocked = str(exc)
    if report["phase_a_verdict"] == "dual_channel_retrieval_gate_failed":
        verdict = "dual_channel_retrieval_failed_gate"
    elif generation_report is None:
        verdict = "dual_channel_retrieval_promising_generation_inconclusive"
    else:
        mandatory = {
            row["question_id"]: row
            for row in generation_report["questions"]
            if row["question_id"] in PROBE_IDS
        }
        promising = (
            generation_report["status_movement"]["error"] == 0
            and generation_report["status_movement"]["new_refusal"] == 0
            and all(
                mandatory[question_id]["augmented"]["actual_status"] == "answered"
                for question_id in PROBE_IDS
            )
            and all(mandatory[question_id]["mandatory_row_retrieved"] for question_id in PROBE_IDS)
            and all(
                probe["critical_fact_token_present"]
                for probe in generation_report["critical_probes"]
            )
        )
        verdict = (
            "dual_channel_integration_promising"
            if promising
            else "dual_channel_retrieval_promising_generation_inconclusive"
        )
    return {
        "retrieval": report,
        "generation": generation_report,
        "generation_blocked": generation_blocked,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen dual-channel table retrieval experiment")
    parser.add_argument("command", choices=["run"])
    parser.parse_args(argv)
    try:
        result = run()
        print(f"verdict={result['verdict']} gate={result['retrieval']['phase_a_verdict']}")
        if result["generation_blocked"]:
            print(f"generation_blocked={result['generation_blocked']}")
        return 0
    except (ExperimentError, ValueError, OSError, httpx.HTTPError, pymupdf.FileDataError) as exc:
        print(f"dual-channel experiment stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
