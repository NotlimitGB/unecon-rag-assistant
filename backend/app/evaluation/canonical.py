"""Frozen evaluation of the adopted application retrieval path."""

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.evaluation.dataset import load_dataset, validate_page_labels
from app.evaluation.metrics import rank_metrics
from app.ingestion.manifest import load_manifest
from app.retrieval.corpus import RetrievalError
from app.retrieval.index import _validated_index
from app.retrieval.service import RetrievalService

EXPECTED_HITS = {
    "primary": (63, 70, 74, 80, 0.83625),
    "accepted": (65, 71, 74, 80, 0.8545833333333333),
    "page": (36, 45, 49, 56, 0.7273809523809524),
}
RETRIEVAL_DATASET_SHA = "52de939e1ba13d1558c3e96fa6cec2cabdb69fca9158996aa1a4282d4167511e"


def _rank(results: list[Any], predicate: Any) -> int | None:
    return next((position for position, hit in enumerate(results, 1) if predicate(hit)), None)


def _nearest_p95(values: list[float]) -> float:
    return sorted(values)[(95 * len(values) + 99) // 100 - 1]


def run_canonical_evaluation(
    dataset_path: Path,
    manifest_path: Path,
    chunks_dir: Path,
    index_dir: Path,
    pdf_root: Path,
    table_index_dir: Path,
    frozen_path: Path,
    output_dir: Path,
    *,
    service: RetrievalService | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    reranker_device: str | None = None,
    reranker_batch_size: int | None = None,
    reranker_max_length: int | None = None,
) -> dict[str, Any]:
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != RETRIEVAL_DATASET_SHA:
        raise RetrievalError("frozen Task006 dataset hash changed")
    manifest = load_manifest(manifest_path)
    dataset = load_dataset(dataset_path, manifest)
    _, metadata = _validated_index(index_dir, settings.embedding_model)
    validate_page_labels(dataset, metadata["records"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        frozen.get("experiment_id") != "pdf-page-diversity-v1"
        or frozen.get("dataset_sha256", {}).get("retrieval") != RETRIEVAL_DATASET_SHA
    ):
        raise RetrievalError("Task018 migration evidence has invalid identity")
    references = frozen.get("selection_diagnostics")
    if not isinstance(references, list) or len(references) != 80:
        raise RetrievalError("Task018 migration evidence is incomplete")
    by_id = {row["question_id"]: row for row in references}
    if len(by_id) != 80:
        raise RetrievalError("Task018 migration evidence has duplicate question IDs")
    config = settings.model_copy(
        update={
            "embedding_device": device if device is not None else settings.embedding_device,
            "embedding_batch_size": (
                batch_size if batch_size is not None else settings.embedding_batch_size
            ),
            "reranker_device": (
                reranker_device if reranker_device is not None else settings.reranker_device
            ),
            "reranker_batch_size": (
                reranker_batch_size
                if reranker_batch_size is not None
                else settings.reranker_batch_size
            ),
            "reranker_max_length": (
                reranker_max_length
                if reranker_max_length is not None
                else settings.reranker_max_length
            ),
        }
    )
    service = service or RetrievalService(
        config=config,
        mode="reranked",
        manifest_path=manifest_path,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        pdf_root=pdf_root,
        table_index_dir=table_index_dir,
    )
    rows = []
    durations = []
    mismatches = []
    for question in dataset["questions"]:
        start = time.perf_counter()
        response = service.retrieve(question["question"], top_k=5)
        durations.append(time.perf_counter() - start)
        hits = response.results
        qid = question["question_id"]
        old = by_id[qid]["diversity_top5"]
        if (
            len(hits) != 5
            or len(old) != 5
            or any(
                hit.chunk_id != prior["chunk_id"]
                or hit.source_id != prior["source_id"]
                or hit.page != prior["page"]
                or abs(hit.rerank_score - prior["reranker_score"]) > 1e-4
                for hit, prior in zip(hits, old, strict=True)
            )
        ):
            mismatches.append(qid)
        primary = question["primary_source_id"]
        accepted = question["acceptable_source_ids"]
        pages = question["expected_pages"]
        rows.append(
            {
                "question_id": qid,
                "question": question["question"],
                "primary_rank": _rank(hits, lambda hit, primary=primary: hit.source_id == primary),
                "accepted_rank": _rank(
                    hits, lambda hit, accepted=accepted: hit.source_id in accepted
                ),
                "page_rank": (
                    _rank(
                        hits,
                        lambda hit, primary=primary, pages=pages: (
                            hit.source_id == primary and hit.page in pages
                        ),
                    )
                    if pages is not None
                    else None
                ),
                "expected_pages": pages,
                "results": [
                    {
                        "rank": hit.rank,
                        "chunk_id": hit.chunk_id,
                        "source_id": hit.source_id,
                        "page": hit.page,
                        "score": hit.score,
                        "dense_score": hit.dense_score,
                    }
                    for hit in hits
                ],
            }
        )
    metrics = {}
    for label, subset in (
        ("primary", rows),
        ("accepted", rows),
        ("page", [row for row in rows if row["expected_pages"] is not None]),
    ):
        ranks = [row[f"{label}_rank"] for row in subset]
        metrics[label] = {
            **rank_metrics(ranks),
            "denominator": len(ranks),
            "hits_at_1": sum(rank is not None and rank <= 1 for rank in ranks),
            "hits_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
            "hits_at_5": sum(rank is not None and rank <= 5 for rank in ranks),
        }
    latency = {
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": _nearest_p95(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
    }
    report = {
        "schema_version": 1,
        "dataset_id": dataset["dataset_id"],
        "question_count": len(rows),
        "migration_matches": len(rows) - len(mismatches),
        "migration_mismatch_ids": mismatches,
        "metrics": metrics,
        "latency": latency,
        "questions": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canonical_retrieval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Canonical retrieval evaluation",
        "",
        f"Task018 exact matches: {report['migration_matches']}/80.",
        "",
        "| Metric | R@1 | R@3 | R@5 | MRR@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics.items():
        lines.append(
            f"| {label} | {values['recall_at_1']:.4f} | {values['recall_at_3']:.4f} "
            f"| {values['recall_at_5']:.4f} | {values['mrr_at_5']:.4f} |"
        )
    lines.extend(["", f"Latency (seconds): {latency}", ""])
    (output_dir / "canonical_retrieval.md").write_text("\n".join(lines), encoding="utf-8")
    if mismatches:
        raise RetrievalError(f"Task018 migration mismatch: {', '.join(mismatches)}")
    for label, expected in EXPECTED_HITS.items():
        actual = metrics[label]
        if (
            actual["hits_at_1"],
            actual["hits_at_3"],
            actual["hits_at_5"],
            actual["denominator"],
        ) != expected[:4] or abs(actual["mrr_at_5"] - expected[4]) > 1e-6:
            raise RetrievalError(f"frozen {label} retrieval metrics changed")
    return report
