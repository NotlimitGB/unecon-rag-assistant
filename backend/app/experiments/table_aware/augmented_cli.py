"""Run the controlled augmented-corpus experiment without changing production."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from app.chunking.core import validate_normalized_document
from app.config import PROJECT_ROOT, settings
from app.evaluation.dataset import validate_page_labels
from app.evaluation.generation_cli import check_baseline, load_datasets, preflight_ollama
from app.evaluation.generation_runner import RecordingRetrieval, evaluate_generation
from app.experiments.table_aware.augmented import (
    EXPECTED_PRODUCTION,
    EXPECTED_ROWS,
    AugmentedRetrievalAdapter,
    baseline_matches,
    build_augmented,
    compare_generation,
    dense_search,
    gate,
    movement,
    probe_results,
    representation_counts,
    rerank,
    save_augmented,
    summarize_ranks,
    table_records,
)
from app.experiments.table_aware.evaluation import EXPECTED_IDS, TARGET_IDS, select_questions
from app.experiments.table_aware.extract import ExperimentError, extract_pdf_tables
from app.generation.service import AnswerService
from app.ingestion.fetcher import fetch_pdf
from app.ingestion.manifest import load_manifest
from app.ingestion.writer import write_document
from app.retrieval.index import RetrievalSession
from app.retrieval.reranker import RerankedRetrievalSession

RETRIEVAL_SHA = "52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e"
GENERATION_SHA = "11a284b4279f627ae787d0ec4777e0884733b4ec937d6658be3857c2dd15d0a4"
BASELINE_SHA = "9e9a2d8052bc74b3bc397f98a83b9b703e3d5fb52d90c75a2fe0f0cc998ef6ff"
PDF_SHA = {
    "admission-capacity-pdf": "748b164c0d1a5e7c82d7fcc4e8e417334be73f062b02c0bed5245e66ae2850c6",
    "entrance-exams-list-pdf": "656ccb00805760f0685c703587c0d5ae5f6dad9ca11ed098f72e6a2180f3d4d7",
    "tuition-order-128-pdf": "455d11c699f84b20739df78aa830bd37ff091b3a13ad776eb11f466a5893544e",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_tables(manifest: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [source for source in manifest.sources if source.id in TARGET_IDS]
    if tuple(source.id for source in sources) != TARGET_IDS or not all(s.active for s in sources):
        raise ExperimentError("target PDF sources changed")
    checked = []
    with httpx.Client(timeout=60) as client:
        for source in sources:
            raw = json.loads(
                (PROJECT_ROOT / "data/processed/pdf" / f"{source.id}.json").read_text(
                    encoding="utf-8"
                )
            )
            normalized = validate_normalized_document(source, raw)
            if normalized["file_sha256"] != PDF_SHA[source.id]:
                raise ExperimentError(f"normalized source version changed: {source.id}")
            fetched = fetch_pdf(source, client)
            digest = hashlib.sha256(fetched.content).hexdigest()
            if digest != PDF_SHA[source.id] or fetched.final_url != raw["source"]["final_url"]:
                raise ExperimentError(f"fetched source version changed: {source.id}")
            checked.append((source, fetched, digest))
    artifacts = [
        extract_pdf_tables(source, fetched.final_url, fetched.content)
        for source, fetched, _ in checked
    ]
    for artifact in artifacts:
        prior_path = (
            PROJECT_ROOT
            / "data/processed/experiments/table_aware"
            / f"{artifact['source']['id']}.json"
        )
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior != artifact:
            raise ExperimentError(
                f"Task014 extraction no longer reproduces: {artifact['source']['id']}"
            )
    if sum(len(table["rows"]) for a in artifacts for table in a["tables"]) != EXPECTED_ROWS:
        raise ExperimentError("Task014 table extraction drift")
    checks = [
        {
            "source_id": source.id,
            "sha256": digest,
            "rows": sum(len(t["rows"]) for t in artifact["tables"]),
        }
        for (source, _, digest), artifact in zip(checked, artifacts, strict=True)
    ]
    return artifacts, checks


def _averages(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = ("production_chunk", "table_row")
    return {
        stage: {
            kind: {
                "total": sum(row[stage][kind] for row in rows),
                "mean": sum(row[stage][kind] for row in rows) / len(rows),
                "distribution": {
                    str(n): sum(row[stage][kind] == n for row in rows)
                    for n in range(21 if stage == "dense_top20" else 6)
                },
            }
            for kind in kinds
        }
        for stage in ("dense_top20", "reranked_top5")
    }


def _generation_phase(
    generation: dict[str, Any],
    retrieval: dict[str, Any],
    selected: list[dict[str, Any]],
    baseline_path: Path,
    adapter: AugmentedRetrievalAdapter,
    output_dir: Path,
) -> dict[str, Any]:
    check_baseline()
    if _sha(baseline_path) != BASELINE_SHA:
        raise ExperimentError("Task011 baseline report SHA mismatch")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = [row for row in baseline["questions"] if row["question_id"] in EXPECTED_IDS]
    if (
        len(baseline_rows) != 16
        or sum(row["actual_status"] == "answered" for row in baseline_rows) != 12
        or sum(row["actual_status"] == "insufficient_evidence" for row in baseline_rows) != 4
    ):
        raise ExperimentError("Task011 table-question status baseline mismatch")
    preflight_ollama()
    subset = {
        **generation,
        "questions": [q for q in generation["questions"] if q["question_id"] in EXPECTED_IDS],
    }
    recording = RecordingRetrieval(adapter)
    service = AnswerService(retrieval_service=recording, config=settings)
    config = {
        "model": settings.ollama_model,
        "prompt_version": "grounded-answer-v1",
        "temperature": settings.generation_temperature,
        "max_tokens": settings.generation_max_tokens,
        "retrieval_mode": "reranked",
        "retrieval_top_k": 5,
    }
    try:
        evaluated = evaluate_generation(subset, retrieval, service, recording, config)
    finally:
        service.close()
    comparison = compare_generation(baseline_rows, evaluated["questions"])
    by_id = {row["question_id"]: row for row in comparison["questions"]}
    for row in comparison["questions"]:
        hits = adapter.history.get(row["augmented"]["question"], [])
        row["augmented_contexts"] = [
            {
                "context_id": f"C{rank}",
                "chunk_id": hit["chunk_id"],
                "representation": hit["representation"],
                "text": hit["text"],
            }
            for rank, hit in enumerate(hits, 1)
        ]
        cited = {citation["chunk_id"] for citation in row["augmented"]["citations"]}
        row["augmented_table_row_cited"] = any(
            context["representation"] == "table_row" and context["chunk_id"] in cited
            for context in row["augmented_contexts"]
        )
    for question_id in ("gen-008", "gen-011", "gen-021", "gen-027"):
        row = by_id[question_id]
        if question_id == "gen-027":
            row["table_row_contains_semester"] = any(
                "семестр" in context["text"].casefold()
                for context in row["augmented_contexts"]
                if context["representation"] == "table_row"
            )
    report = {
        "schema_version": 1,
        "experiment_id": "table-aware-augmented-v1",
        "baseline_sha256": BASELINE_SHA,
        "configuration": config,
        "baseline_metrics": _structural_metrics(baseline_rows),
        "augmented_metrics": _structural_metrics(evaluated["questions"]),
        **comparison,
    }
    write_document(output_dir, "augmented_table_generation_experiment", report)
    return report


def _structural_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in rows if r["success"] and r["actual_status"] == "answered"]
    return {
        "structured_success": sum(r["success"] for r in rows),
        "answered": len(answered),
        "insufficient_evidence": sum(r["actual_status"] == "insufficient_evidence" for r in rows),
        "acceptable_source_hits": sum(r["acceptable_source_hit"] is True for r in answered),
        "primary_source_hits": sum(r["primary_source_hit"] is True for r in answered),
        "expected_page_hits": sum(r["expected_page_hit"] is True for r in answered),
        "citation_denominator": len(answered),
    }


def run() -> dict[str, Any]:
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
    rows = table_records(artifacts, sources)
    dense = RetrievalSession(
        manifest_path,
        PROJECT_ROOT / "data/processed/chunks",
        PROJECT_ROOT / "data/processed/index",
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
    )
    if dense.index.ntotal != EXPECTED_PRODUCTION:
        raise ExperimentError("production chunk count drift")
    validate_page_labels(retrieval, dense.records)
    augmented_index, records = build_augmented(dense.index, dense.records, rows, dense.embedder)
    reranked = RerankedRetrievalSession(
        dense,
        settings.reranker_model,
        settings.reranker_device,
        settings.reranker_batch_size,
        settings.reranker_max_length,
        20,
    )
    prod_final, aug_final, aug_dense = {}, {}, {}
    for q in retrieval["questions"]:
        question_id, question = q["question_id"], q["question"]
        prod_final[question_id] = reranked.rerank(question, dense.search(question, 20), 5)
        aug_dense[question_id] = dense_search(augmented_index, records, dense.embedder, question)
        aug_final[question_id] = rerank(question, aug_dense[question_id], reranked.reranker)
    all_questions = retrieval["questions"]
    prod_metrics, prod_ranks = summarize_ranks(all_questions, prod_final)
    if not baseline_matches(prod_metrics):
        raise ExperimentError("production reranked metrics do not reproduce accepted baseline")
    aug_metrics, aug_ranks = summarize_ranks(all_questions, aug_final)
    selected_ids = {item["question_id"] for item in selected}
    gen_by_ret = {
        item["retrieval_question_id"]: item["question_id"]
        for item in generation["questions"]
        if item["retrieval_question_id"]
    }
    table_ids = {
        item["retrieval_question_id"]
        for item in generation["questions"]
        if item["question_id"] in selected_ids
    }
    if len(table_ids) != 16 or len(all_questions) - len(table_ids) != 64:
        raise ExperimentError("frozen table/non-table partition changed")
    groups = {}
    for label, subset in (
        ("table16", [q for q in all_questions if q["question_id"] in table_ids]),
        ("non_table64", [q for q in all_questions if q["question_id"] not in table_ids]),
    ):
        groups[label] = {
            "production": summarize_ranks(subset, prod_final)[0],
            "augmented": summarize_ranks(subset, aug_final)[0],
        }
    question_to_retrieval = {
        item["question_id"]: next(
            g["retrieval_question_id"]
            for g in generation["questions"]
            if g["question_id"] == item["question_id"]
        )
        for item in selected
    }
    probes = probe_results(artifacts, aug_dense, aug_final, question_to_retrieval)
    checks = gate(
        prod_metrics,
        aug_metrics,
        groups["table16"]["production"],
        groups["table16"]["augmented"],
        groups["non_table64"]["production"],
        groups["non_table64"]["augmented"],
        probes,
    )
    competition = [
        {
            "question_id": q["question_id"],
            "group": "table16" if q["question_id"] in table_ids else "non_table64",
            "dense_top20": representation_counts(aug_dense[q["question_id"]]),
            "reranked_top5": representation_counts(aug_final[q["question_id"]]),
        }
        for q in all_questions
    ]
    report = {
        "schema_version": 1,
        "experiment_id": "table-aware-augmented-v1",
        "evaluated_commit": "441a5ef511b29669490a700daf36b8ff0ca0145e",
        "counts": {
            "production": len(dense.records),
            "table_rows": len(rows),
            "augmented": len(records),
        },
        "devices": {
            "embedding": dense.embedder.device,
            "reranker": reranked.reranker.device,
        },
        "source_checks": source_checks,
        "baseline_reproduced": True,
        "metrics": {"production": prod_metrics, "augmented": aug_metrics},
        "groups": groups,
        "movement": movement(prod_ranks, aug_ranks),
        "question_ranks": [
            {
                "question_id": q["question_id"],
                "generation_question_id": gen_by_ret.get(q["question_id"]),
                "production": prod_ranks[q["question_id"]],
                "augmented": aug_ranks[q["question_id"]],
            }
            for q in all_questions
        ],
        "competition": competition,
        "competition_summary": {
            "all80": _averages(competition),
            "table16": _averages([r for r in competition if r["group"] == "table16"]),
            "non_table64": _averages([r for r in competition if r["group"] == "non_table64"]),
        },
        "probes": probes,
        "gate": checks,
        "phase_a_verdict": "retrieval_gate_passed"
        if all(checks.values())
        else "retrieval_gate_failed",
    }
    output_dir = PROJECT_ROOT / "data/processed/evaluation"
    save_augmented(
        PROJECT_ROOT / "data/processed/experiments/table_aware_augmented",
        augmented_index,
        records,
        settings.embedding_model,
        dense.metadata["index"]["file_sha256"],
    )
    write_document(output_dir, "augmented_table_retrieval_experiment", report)
    generation_report = None
    blocked = None
    if report["phase_a_verdict"] == "retrieval_gate_passed":
        try:
            generation_report = _generation_phase(
                generation,
                retrieval,
                selected,
                PROJECT_ROOT / "data/processed/evaluation/generation_report.json",
                AugmentedRetrievalAdapter(
                    augmented_index, records, dense.embedder, reranked.reranker
                ),
                output_dir,
            )
        except (ExperimentError, ValueError, OSError, httpx.HTTPError) as exc:
            blocked = str(exc)
    if report["phase_a_verdict"] == "retrieval_gate_failed":
        verdict = "augmented_retrieval_failed_gate"
    elif generation_report is None:
        verdict = "retrieval_promising_generation_inconclusive"
    else:
        statuses = generation_report["status_movement"]
        gen_probes = generation_report["critical_probes"]
        mandatory = {r["question_id"]: r for r in generation_report["questions"]}
        promising = (
            statuses["new_refusal"] == 0
            and statuses["error"] == 0
            and all(p["critical_fact_token_present"] for p in gen_probes)
            and all(
                mandatory[q]["augmented"]["actual_status"] == "answered"
                for q in ("gen-008", "gen-011", "gen-021", "gen-027")
            )
        )
        verdict = (
            "integration_promising" if promising else "retrieval_promising_generation_inconclusive"
        )
    return {
        "retrieval": report,
        "generation": generation_report,
        "generation_blocked": blocked,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Augmented table-row integration experiment")
    parser.add_argument("command", choices=["run"])
    parser.parse_args(argv)
    try:
        result = run()
        print(f"verdict={result['verdict']} gate={result['retrieval']['phase_a_verdict']}")
        if result["generation_blocked"]:
            print(f"generation_blocked={result['generation_blocked']}")
        return 0
    except (ExperimentError, ValueError, OSError, httpx.HTTPError, pymupdf.FileDataError) as exc:
        print(f"augmented experiment stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
