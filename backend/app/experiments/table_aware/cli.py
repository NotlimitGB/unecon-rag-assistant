"""Run the isolated table-aware PDF representation experiment."""

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
from app.evaluation.dataset import load_dataset
from app.evaluation.generation_dataset import load_generation_dataset
from app.experiments.table_aware.evaluation import (
    TARGET_IDS,
    evaluate_probes,
    select_questions,
    summarize,
)
from app.experiments.table_aware.extract import (
    EXPERIMENT_ID,
    STRATEGY,
    ExperimentError,
    extract_pdf_tables,
)
from app.experiments.table_aware.retrieval import (
    build_row_index,
    dense_search,
    reranked_search,
    save_row_index,
)
from app.ingestion.fetcher import fetch_pdf
from app.ingestion.manifest import load_manifest
from app.ingestion.writer import write_document
from app.retrieval.index import RetrievalSession
from app.retrieval.reranker import RerankedRetrievalSession


def run(
    *,
    manifest_path: Path = PROJECT_ROOT / "data/source_manifest.json",
    normalized_dir: Path = PROJECT_ROOT / "data/processed/pdf",
    artifacts_dir: Path = PROJECT_ROOT / "data/processed/experiments/table_aware",
    report_dir: Path = PROJECT_ROOT / "data/processed/evaluation",
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    sources = [source for source in manifest.sources if source.id in TARGET_IDS]
    if tuple(source.id for source in sources) != TARGET_IDS or any(
        not source.active for source in sources
    ):
        raise ExperimentError("target sources are missing, reordered, or inactive")

    retrieval_dataset = load_dataset(
        PROJECT_ROOT / "data/evaluation/retrieval_questions.json", manifest
    )
    generation_dataset = load_generation_dataset(
        PROJECT_ROOT / "data/evaluation/generation_questions.json", retrieval_dataset
    )
    questions = select_questions(generation_dataset, retrieval_dataset)

    # Fetch and compare every source before extracting or writing any experimental artifact.
    checked = []
    with httpx.Client(timeout=60) as client:
        for source in sources:
            original = json.loads(
                (normalized_dir / f"{source.id}.json").read_text(encoding="utf-8")
            )
            document = validate_normalized_document(source, original)
            fetched = fetch_pdf(source, client)
            actual_sha = hashlib.sha256(fetched.content).hexdigest()
            expected_sha = document["file_sha256"]
            if actual_sha != expected_sha:
                raise ExperimentError(f"source version changed: {source.id}")
            if fetched.final_url != original["source"]["final_url"]:
                raise ExperimentError(f"source final URL changed: {source.id}")
            checked.append((source, fetched, expected_sha))

    artifacts = [
        extract_pdf_tables(source, fetched.final_url, fetched.content)
        for source, fetched, _ in checked
    ]
    if any(not any(table["rows"] for table in artifact["tables"]) for artifact in artifacts):
        raise ExperimentError("generic strategy found no usable rows in a source")
    for artifact in artifacts:
        write_document(artifacts_dir, artifact["source"]["id"], artifact)

    source_checks = []
    for artifact, (_, fetched, expected_sha) in zip(artifacts, checked, strict=True):
        columns = [column for table in artifact["tables"] for column in table["columns"]]
        source_checks.append(
            {
                "source_id": artifact["source"]["id"],
                "expected_sha256": expected_sha,
                "fetched_sha256": hashlib.sha256(fetched.content).hexdigest(),
                "sha_match": True,
                "tables": len(artifact["tables"]),
                "rows": sum(len(table["rows"]) for table in artifact["tables"]),
                "ambiguous_headers": sum(column["ambiguous"] for column in columns),
            }
        )

    dense = RetrievalSession(
        manifest_path,
        PROJECT_ROOT / "data/processed/chunks",
        PROJECT_ROOT / "data/processed/index",
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
    )
    reranker = RerankedRetrievalSession(
        dense,
        settings.reranker_model,
        settings.reranker_device,
        settings.reranker_batch_size,
        settings.reranker_max_length,
        20,
    )
    index, records, index_bytes = build_row_index(artifacts, dense.embedder)
    save_row_index(artifacts_dir, index_bytes, records, settings.embedding_model)

    production: dict[str, list[dict[str, Any]]] = {}
    experimental: dict[str, list[dict[str, Any]]] = {}
    experimental_dense: dict[str, list[dict[str, Any]]] = {}
    for item in questions:
        question_id, question = item["question_id"], item["question"]
        candidates = dense.search(question, 20)
        production[question_id] = reranker.rerank(question, candidates, 5)
        table_candidates = dense_search(index, records, dense.embedder, question, 20)
        experimental_dense[question_id] = table_candidates
        experimental[question_id] = reranked_search(
            question, table_candidates, reranker.reranker, 5
        )

    probes = evaluate_probes(artifacts, experimental_dense, experimental)
    comparison = summarize(questions, production, experimental, probes, source_checks)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluated_commit": "4d815d31b6df81171884f2206b79dd8f87cde2be",
        "extraction": {"library": "PyMuPDF", "version": pymupdf.VersionBind, "strategy": STRATEGY},
        "models": {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "candidate_k": 20,
            "top_k": 5,
        },
        "sources": source_checks,
        "probes": probes,
        **comparison,
    }
    write_document(report_dir, "table_aware_pdf_experiment", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated table-aware PDF experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run()
            print(f"verdict={result['verdict']} rows={sum(s['rows'] for s in result['sources'])}")
            for mode, scores in result["metrics"].items():
                print(f"{mode}: " + " ".join(f"{key}={value:.3f}" for key, value in scores.items()))
            print("report=data/processed/evaluation/table_aware_pdf_experiment.json")
            return 0
    except (ExperimentError, ValueError, OSError, httpx.HTTPError, pymupdf.FileDataError) as exc:
        print(f"table-aware experiment failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
