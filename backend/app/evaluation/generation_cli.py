"""Evaluate the unchanged Task009 answer baseline and manual review."""

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.config import settings
from app.evaluation.dataset import DatasetError, load_dataset, validate_page_labels
from app.evaluation.generation_dataset import load_generation_dataset
from app.evaluation.generation_runner import (
    RecordingRetrieval,
    evaluate_generation,
    summarize_manual_review,
    validate_manual_review,
    write_generation_reports,
)
from app.generation.prompt import PROMPT_VERSION
from app.generation.service import AnswerService
from app.ingestion.manifest import ManifestError, load_manifest
from app.retrieval.corpus import RetrievalError, load_corpus
from app.retrieval.index import _validated_index
from app.retrieval.service import RetrievalService

ROOT = Path(__file__).resolve().parents[3]
FROZEN_RETRIEVAL_SHA256 = "52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e"


def load_datasets(dataset_path: Path, retrieval_path: Path) -> tuple[dict, dict]:
    if hashlib.sha256(retrieval_path.read_bytes()).hexdigest() != FROZEN_RETRIEVAL_SHA256:
        raise DatasetError("Task006 retrieval dataset SHA-256 differs from frozen baseline")
    manifest = load_manifest(ROOT / "data/source_manifest.json")
    retrieval = load_dataset(retrieval_path, manifest)
    generation = load_generation_dataset(dataset_path, retrieval)
    return generation, retrieval


def check_baseline() -> None:
    expected = {
        "generation_provider": "ollama",
        "ollama_model": "qwen3.5:9b",
        "generation_temperature": 0.1,
        "generation_max_tokens": 512,
        "retrieval_mode": "reranked",
        "retrieval_top_k": 5,
    }
    if any(getattr(settings, key) != value for key, value in expected.items()):
        raise ValueError("Task009 baseline configuration differs from the accepted settings")
    if PROMPT_VERSION != "grounded-answer-v1":
        raise ValueError("Task009 prompt version differs from accepted baseline")


def preflight_corpus(retrieval: dict) -> None:
    _, metadata = _validated_index(ROOT / "data/processed/index", settings.embedding_model)
    fingerprints, records = load_corpus(
        ROOT / "data/source_manifest.json", ROOT / "data/processed/chunks"
    )
    if metadata["corpus"]["sources"] != fingerprints or metadata["records"] != records:
        raise RetrievalError("index is stale relative to source corpus")
    validate_page_labels(retrieval, records)


def preflight_ollama() -> None:
    if shutil.which("ollama") is None:
        raise ValueError("Ollama command is unavailable; install Ollama before real evaluation")
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        response.raise_for_status()
        payload = response.json()
        names = {model.get("name") for model in payload["models"]}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("local Ollama API is unavailable or returned invalid model data") from exc
    if settings.ollama_model not in names:
        raise ValueError(f"model missing; run: ollama pull {settings.ollama_model}")


def run(dataset: dict, retrieval: dict, output_dir: Path) -> dict:
    check_baseline()
    preflight_corpus(retrieval)
    preflight_ollama()
    recording = RecordingRetrieval(RetrievalService(config=settings))
    service = AnswerService(retrieval_service=recording, config=settings)
    config = {
        "generation_provider": settings.generation_provider,
        "model": settings.ollama_model,
        "retrieval_mode": settings.retrieval_mode,
        "retrieval_top_k": settings.retrieval_top_k,
        "prompt_version": PROMPT_VERSION,
        "temperature": settings.generation_temperature,
        "max_tokens": settings.generation_max_tokens,
        "supported_questions": 42,
        "unsupported_questions": 18,
    }
    try:
        report = evaluate_generation(dataset, retrieval, service, recording, config)
    finally:
        service.close()
    write_generation_reports(report, output_dir)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate local grounded answers")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate-dataset", "validate-review", "summarize-review"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--dataset", type=Path, default=ROOT / "data/evaluation/generation_questions.json"
        )
        sub.add_argument(
            "--retrieval-dataset",
            type=Path,
            default=ROOT / "data/evaluation/retrieval_questions.json",
        )
        sub.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/evaluation")
        if name in ("validate-review", "summarize-review"):
            sub.add_argument("--review", type=Path)
        if name == "summarize-review":
            sub.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        dataset, retrieval = load_datasets(args.dataset, args.retrieval_dataset)
        if args.command == "validate-dataset":
            print(
                f"dataset={dataset['dataset_id']} questions={len(dataset['questions'])} valid=true"
            )
            return 0
        if args.command == "run":
            report = run(dataset, retrieval, args.output_dir)
            print(
                f"questions={len(report['questions'])} "
                f"structured={report['metrics']['reliability']['successful_structured_responses']}"
            )
            print(f"status_accuracy={report['metrics']['status']['overall_status_accuracy']:.4f}")
            for path in (
                "generation_report.json",
                "generation_report.md",
                "generation_manual_review.json",
            ):
                print(f"report={args.output_dir / path}")
            return 0
        report = json.loads(
            (args.output_dir / "generation_report.json").read_text(encoding="utf-8")
        )
        review_path = args.review or args.output_dir / "generation_manual_review.json"
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if args.command == "validate-review":
            reviewed = validate_manual_review(review, report)
            print(f"valid=true reviewed={len(reviewed)}")
        else:
            summary = summarize_manual_review(review, report, partial=args.partial)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (
        DatasetError,
        ManifestError,
        RetrievalError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
