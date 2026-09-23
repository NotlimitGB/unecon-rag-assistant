"""Run the fixed 80-question retrieval baseline from local artifacts."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config import settings
from app.evaluation.dataset import DatasetError, load_dataset, validate_page_labels
from app.evaluation.runner import evaluate_retrieval, write_reports
from app.ingestion.manifest import ManifestError, load_manifest
from app.retrieval.corpus import RetrievalError, load_corpus
from app.retrieval.index import RetrievalSession, _validated_index

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_evaluation(
    dataset_path: Path,
    manifest_path: Path,
    chunks_dir: Path,
    index_dir: Path,
    output_dir: Path,
    device: str = "auto",
    batch_size: int = 16,
) -> dict:
    manifest = load_manifest(manifest_path)
    dataset = load_dataset(dataset_path, manifest)
    _, metadata = _validated_index(index_dir, settings.embedding_model)
    fingerprints, records = load_corpus(manifest_path, chunks_dir)
    if metadata["corpus"]["sources"] != fingerprints or metadata["records"] != records:
        raise RetrievalError("index is stale relative to the source corpus")
    validate_page_labels(dataset, metadata["records"])
    session = RetrievalSession(
        manifest_path, chunks_dir, index_dir, settings.embedding_model, device, batch_size
    )
    report = evaluate_retrieval(dataset, session)
    write_reports(report, output_dir)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval over 80 questions")
    parser.add_argument("command", choices=["retrieval"])
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/source_manifest.json")
    parser.add_argument("--chunks-dir", type=Path, default=PROJECT_ROOT / "data/processed/chunks")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "data/processed/index")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/evaluation"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=settings.embedding_device
    )
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128:
        parser.error("--batch-size must be from 1 to 128")
    try:
        report = run_evaluation(
            args.dataset,
            args.manifest,
            args.chunks_dir,
            args.index_dir,
            args.output_dir,
            args.device,
            args.batch_size,
        )
    except (DatasetError, ManifestError, RetrievalError, OSError, ValueError, RuntimeError) as exc:
        print(f"FAILED: {exc}")
        return 1
    metrics = report["metrics"]
    print(f"dataset={report['dataset']['dataset_id']}")
    print(f"questions={report['dataset']['question_count']}")
    for key in (
        "primary_recall_at_1",
        "primary_recall_at_3",
        "primary_recall_at_5",
        "primary_mrr_at_5",
        "accepted_recall_at_5",
        "page_recall_at_5",
    ):
        value = metrics[key]
        print(f"{key}={'N/A' if value is None else f'{value:.4f}'}")
    print(f"report_json={args.output_dir / 'retrieval_report.json'}")
    print(f"report_md={args.output_dir / 'retrieval_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
