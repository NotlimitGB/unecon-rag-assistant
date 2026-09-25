"""Run one frozen PDF-page diversity policy after unchanged dual-channel reranking."""

import argparse
import json
import sys
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
    compare_generation,
    probe_results,
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
from app.experiments.table_aware.diversity import (
    CachedDiversityAdapter,
    retrieval_gate,
    saturation,
    select_with_pdf_page_diversity,
)
from app.experiments.table_aware.dual_channel import (
    build_table_channel,
    merge_candidates,
    new_misses,
    partition_questions,
    preservation_check,
    recovered_misses,
    search_channels,
)
from app.experiments.table_aware.evaluation import EXPECTED_IDS, PROBE_IDS, select_questions
from app.experiments.table_aware.extract import ExperimentError
from app.experiments.table_aware.reranker_analysis_cli import _check_task016
from app.generation.service import AnswerService
from app.ingestion.manifest import load_manifest
from app.ingestion.writer import write_document
from app.retrieval.index import RetrievalSession
from app.retrieval.reranker import RerankedRetrievalSession, rerank_candidates

EXPERIMENT_ID = "pdf-page-diversity-v1"
EVALUATED_COMMIT = "0c3f942799324c21cffdf01c06ea0ff11e431f67"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/evaluation"
SCORE_TOLERANCE = 1e-4


def _summary(hits: list[dict[str, Any]], original_ranks: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "final_rank": rank,
            "original_reranker_rank": original_ranks[hit["chunk_id"]],
            "chunk_id": hit["chunk_id"],
            "source_id": hit["source_id"],
            "page": hit["page_start"],
            "representation": hit["representation"],
            "dense_score": hit["dense_score"],
            "reranker_score": hit["rerank_score"],
        }
        for rank, hit in enumerate(hits, 1)
    ]


def _compare_task017(ranked: list[dict[str, Any]], frozen: list[dict[str, Any]]) -> bool:
    if len(ranked) != 25 or len(frozen) != 25:
        return False
    return all(
        current["chunk_id"] == old["chunk_id"]
        and current["vector_id"] == old["vector_id"]
        and current["source_id"] == old["source_id"]
        and current["page_start"] == old["page"]
        and current["representation"] == old["representation"]
        and current["dense_rank"] == old["merged_dense_position"]
        and abs(current["dense_score"] - old["dense_score"]) <= 1e-6
        and abs(current["rerank_score"] - old["reranker_score"]) <= SCORE_TOLERANCE
        for current, old in zip(ranked, frozen, strict=True)
    )


def _compare_metrics(actual: dict[str, Any], frozen: dict[str, Any]) -> bool:
    return all(
        key in actual
        and (
            abs(actual[key] - value) <= SCORE_TOLERANCE
            if isinstance(value, float)
            else actual[key] == value
        )
        for key, value in frozen.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _losses(
    old: dict[str, dict[str, int | None]],
    new: dict[str, dict[str, int | None]],
    ids: set[str],
) -> dict[str, dict[str, list[str]]]:
    return {
        field: {
            "new_losses": new_misses(old, new, ids, field),
            "recovered": recovered_misses(old, new, ids, field),
        }
        for field in ("primary", "accepted", "page")
    }


def _impact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["changed"]]
    skipped = [item for row in rows for item in row["skipped"]]
    replacements = [pair["replacement"] for row in rows for pair in row["replacement_pairs"]]
    bins = {
        label: sum(start <= item["original_reranker_rank"] <= end for item in replacements)
        for label, start, end in (
            ("6-10", 6, 10),
            ("11-15", 11, 15),
            ("16-20", 16, 20),
            ("21-25", 21, 25),
        )
    }
    return {
        "changed_questions": [row["question_id"] for row in changed],
        "changed_count": len(changed),
        "unchanged_count": len(rows) - len(changed),
        "skipped_total": len(skipped),
        "mean_skipped_on_changed": len(skipped) / len(changed) if changed else 0.0,
        "max_skipped": max((len(row["skipped"]) for row in rows), default=0),
        "replacement_rank_bins": bins,
        "groups": {
            name: {
                "questions": sum(row["group"] == name for row in rows),
                "changed": sum(row["group"] == name and row["changed"] for row in rows),
                "skipped": sum(len(row["skipped"]) for row in rows if row["group"] == name),
            }
            for name in ("table16", "non_table64")
        },
    }


def _saturation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        stage: {
            field: [row["question_id"] for row in rows if row[stage][field]]
            for field in ("three_same_source", "three_same_pdf_page", "three_table_same_pdf_page")
        }
        for stage in ("raw_saturation", "diversity_saturation")
    }


def _repair_probe(
    question_id: str,
    frozen: dict[str, Any],
    raw_ranked: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    field = "is_primary_source" if question_id == "ret-052" else "is_expected_page"
    expected_ids = {c["chunk_id"] for c in frozen["candidates"] if c[field]}
    raw_rank = next(
        (rank for rank, hit in enumerate(raw_ranked, 1) if hit["chunk_id"] in expected_ids),
        None,
    )
    selected_rank = next(
        (rank for rank, hit in enumerate(selected, 1) if hit["chunk_id"] in expected_ids),
        None,
    )
    return {
        "question_id": question_id,
        "raw_expected_rank": raw_rank,
        "diversity_expected_rank": selected_rank,
        "repaired": selected_rank is not None,
        "skipped_siblings": [
            item
            for item in skipped
            if (
                (
                    question_id == "ret-052"
                    and item["source_id"] == "tuition-order-128-pdf"
                    and item["page"] == 3
                )
                or (
                    question_id == "ret-020"
                    and item["source_id"] == "admission-capacity-pdf"
                    and item["page"] == 1
                )
                or (
                    question_id == "ret-044"
                    and item["source_id"] == "entrance-exams-list-pdf"
                    and item["page"] == 1
                )
            )
        ],
    }


def _generation_phase(
    generation: dict[str, Any],
    retrieval: dict[str, Any],
    final_by_question: dict[str, list[dict[str, Any]]],
    mandatory: list[dict[str, Any]],
) -> dict[str, Any]:
    check_baseline()
    baseline_path = OUTPUT_DIR / "generation_report.json"
    if _sha(baseline_path) != BASELINE_SHA:
        raise ExperimentError("Task011 baseline report SHA mismatch")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = [row for row in baseline["questions"] if row["question_id"] in EXPECTED_IDS]
    if (
        len(baseline_rows) != 16
        or sum(row["actual_status"] == "answered" for row in baseline_rows) != 12
        or sum(row["actual_status"] == "insufficient_evidence" for row in baseline_rows) != 4
    ):
        raise ExperimentError("Task011 16-question baseline changed")
    preflight_ollama()
    subset = {
        **generation,
        "questions": [q for q in generation["questions"] if q["question_id"] in EXPECTED_IDS],
    }
    if tuple(q["question_id"] for q in subset["questions"]) != EXPECTED_IDS:
        raise ExperimentError("frozen generation subset changed")
    by_retrieval = {q["question_id"]: q["question"] for q in retrieval["questions"]}
    hits_by_text = {by_retrieval[qid]: hits for qid, hits in final_by_question.items()}
    adapter = CachedDiversityAdapter(hits_by_text)
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
    mandatory_ids = {item["question_id"]: item["row_id"] for item in mandatory}
    for row in comparison["questions"]:
        hits = adapter.history.get(row["augmented"]["question"], [])
        row["diversity_contexts"] = [
            {
                "context_id": f"C{rank}",
                "chunk_id": hit["chunk_id"],
                "representation": hit["representation"],
                "text": hit["text"],
            }
            for rank, hit in enumerate(hits, 1)
        ]
        cited = {c["chunk_id"] for c in row["augmented"]["citations"]}
        row["mandatory_row_selected"] = (
            mandatory_ids[row["question_id"]] in {hit["chunk_id"] for hit in hits}
            if row["question_id"] in mandatory_ids
            else None
        )
        row["mandatory_row_cited"] = (
            mandatory_ids[row["question_id"]] in cited
            if row["question_id"] in mandatory_ids
            else None
        )
    for label, gen_id in (("ret-020", "gen-012"), ("ret-044", "gen-022")):
        if gen_id not in {row["question_id"] for row in comparison["questions"]}:
            raise ExperimentError(f"missing frozen generation mapping for {label}")
    baseline_metrics = _structural_metrics(baseline_rows)
    diversity_metrics = _structural_metrics(evaluated["questions"])
    for rows, metrics in (
        (baseline_rows, baseline_metrics),
        (evaluated["questions"], diversity_metrics),
    ):
        metrics["answered_with_citation"] = sum(
            row["success"] and row["actual_status"] == "answered" and bool(row["citations"])
            for row in rows
        )
    diversity_metrics["latency_mean_seconds"] = comparison["augmented_latency_mean"]
    diversity_metrics["latency_median_seconds"] = comparison["augmented_latency_median"]
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "baseline_report_sha256": BASELINE_SHA,
        "configuration": config,
        "generation_question_ids": list(EXPECTED_IDS),
        "baseline_metrics": baseline_metrics,
        "diversity_metrics": diversity_metrics,
        **comparison,
    }
    write_document(OUTPUT_DIR, "pdf_page_diversity_generation_experiment", report)
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
        raise ExperimentError("accepted Task016 model or reranker configuration changed")
    retrieval_path = PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
    generation_path = PROJECT_ROOT / "data/evaluation/generation_questions.json"
    if _sha(retrieval_path) != RETRIEVAL_SHA or _sha(generation_path) != GENERATION_SHA:
        raise ExperimentError("frozen dataset hash changed")
    generation, retrieval = load_datasets(generation_path, retrieval_path)
    questions = retrieval["questions"]
    selected_questions = select_questions(generation, retrieval)
    table_ids = {
        q["retrieval_question_id"]
        for q in generation["questions"]
        if q["question_id"] in {item["question_id"] for item in selected_questions}
    }
    table_ids, other_ids = partition_questions(questions, table_ids)
    trace_path = OUTPUT_DIR / "reranker_competition_analysis.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if (
        trace.get("experiment_id") != "reranker-competition-v1"
        or trace.get("evaluated_commit") != "f06b6c2b3fba6af50287e9b7b95a21e931edbd1e"
        or len(trace.get("candidate_trace", [])) != 80
    ):
        raise ExperimentError("Task017 complete local trace unavailable or changed")
    frozen = {row["question_id"]: row for row in trace["candidate_trace"]}
    if set(frozen) != {q["question_id"] for q in questions}:
        raise ExperimentError("Task017 question IDs changed")
    task016_path = OUTPUT_DIR / "dual_channel_table_retrieval_experiment.json"
    task016 = json.loads(task016_path.read_text(encoding="utf-8"))
    if task016.get("experiment_id") != "table-aware-dual-channel-v1":
        raise ExperimentError("Task016 report unavailable or changed")
    table_metadata = json.loads(
        (
            PROJECT_ROOT / "data/processed/experiments/table_aware_dual_channel/metadata.json"
        ).read_text(encoding="utf-8")
    )
    if _sha(PROJECT_ROOT / "data/processed/index/index.faiss") != table_metadata.get(
        "production_index_sha256"
    ):
        raise ExperimentError("production index SHA differs from Task016")
    manifest_path = PROJECT_ROOT / "data/source_manifest.json"
    manifest = load_manifest(manifest_path)
    artifacts, source_checks = _verified_tables(manifest)
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
        artifacts, {s.id: s for s in manifest.sources}, dense.embedder, dense.index.d
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
    production_final, raw_final, diversity_final = {}, {}, {}
    raw_ranked, table_dense, selections = {}, {}, []
    for position, question in enumerate(questions, 1):
        qid, query = question["question_id"], question["question"]
        production, table = search_channels(
            dense.index, dense.records, table_index, table_records, dense.embedder, query
        )
        canonical = dense.search(query, 20)
        if not preservation_check(production, canonical):
            raise ExperimentError(f"production top-20 drift: {qid}")
        table_dense[qid] = table
        production_final[qid] = reranked.rerank(query, canonical, 5)
        ranked = rerank_candidates(
            query, merge_candidates(production, table), reranked.reranker, 25
        )
        if not _compare_task017(ranked, frozen[qid]["candidates"]):
            raise ExperimentError(f"Task017 full reranker ranking drift: {qid}")
        raw_ranked[qid] = ranked
        raw_final[qid] = ranked[:5]
        original_ranks = {hit["chunk_id"]: i for i, hit in enumerate(ranked, 1)}
        choice = select_with_pdf_page_diversity(ranked)
        if len(choice["selected"]) != 5:
            raise ExperimentError(f"PDF page cap cannot yield five candidates: {qid}")
        diversity_final[qid] = choice["selected"]
        raw_sat, diverse_sat = saturation(ranked[:5]), saturation(choice["selected"])
        if diverse_sat["max_same_pdf_page"] > 2:
            raise ExperimentError(f"PDF page cap invariant failed: {qid}")
        selections.append(
            {
                "question_id": qid,
                "group": "table16" if qid in table_ids else "non_table64",
                "raw_top5": _summary(ranked[:5], original_ranks),
                "diversity_top5": _summary(choice["selected"], original_ranks),
                "selected_original_ranks": choice["selected_original_ranks"],
                "skipped": choice["skipped"],
                "replacement_pairs": choice["replacement_pairs"],
                "skipped_before_fill": len(choice["skipped"]),
                "changed": [h["chunk_id"] for h in ranked[:5]]
                != [h["chunk_id"] for h in choice["selected"]],
                "raw_saturation": raw_sat,
                "diversity_saturation": diverse_sat,
            }
        )
        if position % 10 == 0:
            print(f"evaluated={position}/80", flush=True)
    production_metrics, raw_metrics, rank_info = _check_task016(
        questions, production_final, raw_final, table_ids
    )
    if not _compare_metrics(
        production_metrics, task016["metrics"]["production"]
    ) or not _compare_metrics(raw_metrics, task016["metrics"]["dual"]):
        raise ExperimentError("Task016 complete metric baseline drift")
    diversity_metrics, diversity_ranks = summarize_ranks(questions, diversity_final)
    groups = {}
    for name, ids in (("table16", table_ids), ("non_table64", other_ids)):
        subset = [q for q in questions if q["question_id"] in ids]
        groups[name] = {
            "production": summarize_ranks(subset, production_final)[0],
            "raw_dual": summarize_ranks(subset, raw_final)[0],
            "diversity": summarize_ranks(subset, diversity_final)[0],
        }
    gen_to_ret = {
        q["question_id"]: q["retrieval_question_id"]
        for q in generation["questions"]
        if q["retrieval_question_id"]
    }
    raw_probes = probe_results(
        artifacts,
        table_dense,
        raw_final,
        {gen: gen_to_ret[gen] for gen in PROBE_IDS},
    )
    mandatory = []
    for probe in raw_probes:
        qid = gen_to_ret[probe["question_id"]]
        row_id = probe["row_id"]
        row_hit = next((h for h in raw_ranked[qid] if h["chunk_id"] == row_id), None)
        raw_rank = next(
            (i for i, h in enumerate(raw_ranked[qid], 1) if h["chunk_id"] == row_id), None
        )
        selected_rank = next(
            (i for i, h in enumerate(diversity_final[qid], 1) if h["chunk_id"] == row_id), None
        )
        skipped = next(
            (
                s
                for s in next(row for row in selections if row["question_id"] == qid)["skipped"]
                if s["chunk_id"] == row_id
            ),
            None,
        )
        sibling = [
            h["chunk_id"]
            for h in diversity_final[qid]
            if row_hit is not None
            and h["source_id"] == row_hit["source_id"]
            and h["page_start"] == row_hit["page_start"]
            and h["chunk_id"] != row_id
        ]
        mandatory.append(
            {
                "question_id": probe["question_id"],
                "retrieval_question_id": qid,
                "row_id": row_id,
                "source_id": row_hit["source_id"] if row_hit else None,
                "page": row_hit["page_start"] if row_hit else None,
                "original_reranker_rank": raw_rank,
                "diversity_final_rank": selected_rank,
                "selected": selected_rank is not None,
                "skipped_by_cap": skipped is not None,
                "same_page_selected_siblings": sibling,
                "structural_probe_passed": probe["passed"],
            }
        )
    labeled_ids = {q["question_id"] for q in questions if q["expected_pages"] is not None}
    gates = retrieval_gate(
        production_metrics,
        True,
        True,
        True,
        len(mandatory) == 4
        and all(p["selected"] and p["structural_probe_passed"] for p in mandatory),
        diversity_metrics,
        groups["table16"]["diversity"],
        rank_info["production"],
        diversity_ranks,
        other_ids,
        labeled_ids,
    )
    probes = {
        qid: _repair_probe(
            qid,
            frozen[qid],
            raw_ranked[qid],
            diversity_final[qid],
            next(row for row in selections if row["question_id"] == qid)["skipped"],
        )
        for qid in ("ret-020", "ret-044", "ret-052")
    }
    all_ids = {q["question_id"] for q in questions}
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluated_commit": EVALUATED_COMMIT,
        "source_checks": source_checks,
        "dataset_sha256": {"retrieval": RETRIEVAL_SHA, "generation": GENERATION_SHA},
        "corpus_counts": {
            "production_chunks": len(dense.records),
            "table_rows": len(table_records),
            "questions": len(questions),
        },
        "allocation": {
            "production_k": 20,
            "table_k": 5,
            "merged_k": 25,
            "pdf_page_cap": 2,
            "final_k": 5,
        },
        "devices": {"embedding": dense.embedder.device, "reranker": reranked.reranker.device},
        "reproduction": {
            "production_top20_preserved": 80,
            "task017_full_rankings_preserved": 80,
            "production": production_metrics,
            "raw_dual": raw_metrics,
        },
        "diversity_metrics": diversity_metrics,
        "groups": groups,
        "losses_vs_production": _losses(rank_info["production"], diversity_ranks, all_ids),
        "non_table_losses_vs_production": _losses(
            rank_info["production"], diversity_ranks, other_ids
        ),
        "selection_diagnostics": selections,
        "selection_impact": _impact(selections),
        "saturation": _saturation(selections),
        "mandatory_rows": mandatory,
        "repair_probes": probes,
        "gate": gates,
        "phase_a_verdict": "diversity_retrieval_gate_passed"
        if all(gates.values())
        else "diversity_retrieval_gate_failed",
    }
    write_document(OUTPUT_DIR, "pdf_page_diversity_retrieval_experiment", report)
    generation_report = None
    generation_blocked = None
    if report["phase_a_verdict"] == "diversity_retrieval_gate_passed":
        try:
            generation_report = _generation_phase(generation, retrieval, diversity_final, mandatory)
        except (ExperimentError, ValueError, OSError, httpx.HTTPError) as exc:
            generation_blocked = str(exc)
    if report["phase_a_verdict"] == "diversity_retrieval_gate_failed":
        verdict = "diversity_retrieval_failed_gate"
    elif generation_report is None:
        verdict = "diversity_retrieval_promising_generation_inconclusive"
    else:
        mandatory_generation = {
            row["question_id"]: row
            for row in generation_report["questions"]
            if row["question_id"] in PROBE_IDS
        }
        promising = (
            generation_report["status_movement"]["error"] == 0
            and generation_report["status_movement"]["new_refusal"] == 0
            and all(
                mandatory_generation[q]["augmented"]["actual_status"] == "answered"
                for q in PROBE_IDS
            )
            and all(
                probe["critical_fact_token_present"]
                for probe in generation_report["critical_probes"]
            )
            and all(row["selected"] for row in mandatory)
        )
        verdict = (
            "diversity_integration_promising"
            if promising
            else "diversity_retrieval_promising_generation_inconclusive"
        )
    if generation_report is not None:
        generation_report["verdict"] = verdict
        write_document(OUTPUT_DIR, "pdf_page_diversity_generation_experiment", generation_report)
    return {
        "retrieval": report,
        "generation": generation_report,
        "generation_blocked": generation_blocked,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen PDF page diversity experiment")
    parser.add_argument("command", choices=["run"])
    parser.parse_args(argv)
    try:
        result = run()
        print(f"verdict={result['verdict']} gate={result['retrieval']['phase_a_verdict']}")
        if result["generation_blocked"]:
            print(f"generation_blocked={result['generation_blocked']}")
        return 0
    except (ExperimentError, ValueError, OSError, httpx.HTTPError, pymupdf.FileDataError) as exc:
        print(f"PDF page diversity experiment stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
