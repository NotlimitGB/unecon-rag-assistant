"""Run the fixed 80-question retrieval baseline from local artifacts."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app.config import settings
from app.evaluation.canonical import run_canonical_evaluation
from app.evaluation.comparison import compare, write_comparison
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
    parser.add_argument("command", choices=["retrieval", "compare-reranker", "canonical-retrieval"])
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/source_manifest.json")
    parser.add_argument("--chunks-dir", type=Path, default=PROJECT_ROOT / "data/processed/chunks")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "data/processed/index")
    parser.add_argument("--pdf-root", type=Path, default=PROJECT_ROOT / "data/processed/pdf")
    parser.add_argument(
        "--table-index-dir", type=Path, default=PROJECT_ROOT / "data/processed/table_index"
    )
    parser.add_argument(
        "--task018-report",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/evaluation/pdf_page_diversity_retrieval_experiment.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/evaluation"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=settings.embedding_device
    )
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument(
        "--reranker-device", choices=["auto", "cpu", "cuda"], default=settings.reranker_device
    )
    parser.add_argument("--reranker-batch-size", type=int, default=settings.reranker_batch_size)
    parser.add_argument("--reranker-max-length", type=int, default=settings.reranker_max_length)
    parser.add_argument("--candidate-k", type=int, default=settings.reranker_candidate_k)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128:
        parser.error("--batch-size must be from 1 to 128")
    if args.command == "compare-reranker" and (
        args.candidate_k != 20
        or not 1 <= args.reranker_batch_size <= 64
        or not 32 <= args.reranker_max_length <= 4096
    ):
        parser.error("invalid reranker comparison options")
    if args.command == "canonical-retrieval" and (
        not 1 <= args.reranker_batch_size <= 64 or not 32 <= args.reranker_max_length <= 4096
    ):
        parser.error("invalid reranker options")
    try:
        if args.command == "retrieval":
            report = run_evaluation(
                args.dataset,
                args.manifest,
                args.chunks_dir,
                args.index_dir,
                args.output_dir,
                args.device,
                args.batch_size,
            )
        elif args.command == "canonical-retrieval":
            report = run_canonical_evaluation(
                args.dataset,
                args.manifest,
                args.chunks_dir,
                args.index_dir,
                args.pdf_root,
                args.table_index_dir,
                args.task018_report,
                args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                reranker_device=args.reranker_device,
                reranker_batch_size=args.reranker_batch_size,
                reranker_max_length=args.reranker_max_length,
            )
        else:
            manifest = load_manifest(args.manifest)
            dataset = load_dataset(args.dataset, manifest)
            with (args.index_dir / "metadata.json").open(encoding="utf-8") as handle:
                page_metadata = json.load(handle)
            try:
                validate_page_labels(dataset, page_metadata["records"])
            except (KeyError, TypeError) as exc:
                raise RetrievalError(f"invalid local index metadata: {exc}") from exc
            dense_session = RetrievalSession(
                args.manifest,
                args.chunks_dir,
                args.index_dir,
                settings.embedding_model,
                args.device,
                args.batch_size,
            )
            report, reranked_session = compare(
                dataset,
                dense_session,
                model_name=settings.reranker_model,
                device=args.reranker_device,
                batch_size=args.reranker_batch_size,
                max_length=args.reranker_max_length,
                candidate_k=args.candidate_k,
            )
            write_comparison(report, args.output_dir)
    except (DatasetError, ManifestError, RetrievalError, OSError, ValueError, RuntimeError) as exc:
        print(f"FAILED: {exc}")
        return 1
    if args.command == "canonical-retrieval":
        print(f"migration_matches={report['migration_matches']}/80")
        for label, metrics in report["metrics"].items():
            print(
                f"{label}={metrics['hits_at_1']}/{metrics['denominator']},"
                f"{metrics['hits_at_3']}/{metrics['denominator']},"
                f"{metrics['hits_at_5']}/{metrics['denominator']} "
                f"mrr_at_5={metrics['mrr_at_5']:.4f}"
            )
        print(f"report_json={args.output_dir / 'canonical_retrieval.json'}")
        print(f"report_md={args.output_dir / 'canonical_retrieval.md'}")
        return 0
    if args.command == "compare-reranker":
        print(
            f"embedding_model={settings.embedding_model} "
            f"embedding_device={dense_session.embedder.device}"
        )
        print(
            f"reranker_model={settings.reranker_model} "
            f"reranker_device={reranked_session.reranker.device}"
        )
        print(
            f"candidate_k={args.candidate_k} final_k=5 "
            f"questions={report['dataset']['question_count']} "
            f"vectors={report['dense']['vector_count']} "
            f"pairs={report['reranked']['pairs_scored']}"
        )
        for key, value in report["dense"]["metrics"].items():
            if key != "page_labeled_questions":
                print(
                    f"{key}: dense={value:.4f} "
                    f"reranked={report['reranked']['metrics'][key]:.4f} "
                    f"delta={report['deltas'][key]:+.4f}"
                )
        print(f"report_json={args.output_dir / 'reranker_comparison.json'}")
        print(f"report_md={args.output_dir / 'reranker_comparison.md'}")
        return 0
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
