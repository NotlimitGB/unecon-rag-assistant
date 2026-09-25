"""Isolated augmentation of the accepted FAISS corpus with Task014 table rows."""

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.evaluation.metrics import rank_metrics
from app.evaluation.runner import _first_rank
from app.experiments.table_aware.evaluation import PROBE_IDS, evaluate_probes
from app.experiments.table_aware.extract import ExperimentError
from app.experiments.table_aware.retrieval import _atomic_bytes
from app.retrieval.index import _vectors
from app.retrieval.reranker import rerank_candidates
from app.retrieval.service import RetrievalResponse, RetrievedChunk

EXPECTED_PRODUCTION = 338
EXPECTED_ROWS = 250
BASELINE_HITS = {
    "primary_recall_at_1": 61,
    "primary_recall_at_3": 67,
    "primary_recall_at_5": 70,
    "accepted_recall_at_1": 62,
    "accepted_recall_at_3": 69,
    "accepted_recall_at_5": 71,
    "page_recall_at_1": 32,
    "page_recall_at_3": 41,
    "page_recall_at_5": 47,
}


def table_records(artifacts: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, Any]]:
    """Carry only verified official provenance into the augmented index."""
    records = []
    for artifact in artifacts:
        meta = artifact["source"]
        source = sources[meta["id"]]
        for table in artifact["tables"]:
            for row in table["rows"]:
                records.append(
                    {
                        "vector_id": EXPECTED_PRODUCTION + len(records),
                        "chunk_id": row["row_id"],
                        "text": row["text"],
                        "content_sha256": row["content_sha256"],
                        "source_id": source.id,
                        "source_title": source.title,
                        "url": source.url,
                        "final_url": meta["final_url"],
                        "source_type": "pdf",
                        "category": source.category,
                        "admission_year": source.admission_year,
                        "page_start": table["page"],
                        "page_end": table["page"],
                        "representation": "table_row",
                    }
                )
    if len(records) != EXPECTED_ROWS:
        raise ExperimentError(f"table extraction drift: {len(records)} rows")
    return records


def build_augmented(
    production: Any,
    production_records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    embedder: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(production, faiss.IndexFlatIP) or production.ntotal != EXPECTED_PRODUCTION:
        raise ExperimentError("production index drift or unsupported type")
    if len(production_records) != EXPECTED_PRODUCTION or len(rows) != EXPECTED_ROWS:
        raise ExperimentError("corpus count drift")
    if [r["vector_id"] for r in production_records] != list(range(EXPECTED_PRODUCTION)):
        raise ExperimentError("production vector order changed")
    if [r["vector_id"] for r in rows] != list(
        range(EXPECTED_PRODUCTION, EXPECTED_PRODUCTION + EXPECTED_ROWS)
    ):
        raise ExperimentError("table vector order changed")
    records = [
        {**record, "representation": "production_chunk"} for record in production_records
    ] + rows
    if len({r["chunk_id"] for r in records}) != len(records):
        raise ExperimentError("duplicate augmented record ID")
    existing = _vectors(
        production.reconstruct_n(0, production.ntotal), EXPECTED_PRODUCTION, production.d
    )
    added = _vectors(
        embedder.encode_documents([r["text"] for r in rows]), EXPECTED_ROWS, production.d
    )
    index = faiss.IndexFlatIP(production.d)
    index.add(np.ascontiguousarray(np.concatenate((existing, added))))
    if index.ntotal != EXPECTED_PRODUCTION + EXPECTED_ROWS:
        raise ExperimentError("augmented index count mismatch")
    return index, records


def save_augmented(
    output_dir: Path, index: Any, records: list[dict[str, Any]], model: str, production_sha: str
) -> None:
    content = faiss.serialize_index(index).tobytes()
    metadata = {
        "schema_version": 1,
        "experiment_id": "table-aware-augmented-v1",
        "model": model,
        "production_index_sha256": production_sha,
        "index_sha256": hashlib.sha256(content).hexdigest(),
        "vector_count": index.ntotal,
        "records": records,
    }
    _atomic_bytes(output_dir / "index.faiss", content)
    _atomic_bytes(
        output_dir / "metadata.json",
        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode(),
    )


def dense_search(
    index: Any, records: list[dict[str, Any]], embedder: Any, query: str, top_k: int = 20
) -> list[dict[str, Any]]:
    if (
        not isinstance(query, str)
        or not query.strip()
        or type(top_k) is not int
        or not 1 <= top_k <= 20
    ):
        raise ExperimentError("invalid augmented search request")
    raw = np.asarray(embedder.encode_query(query))
    if raw.ndim != 1:
        raise ExperimentError("invalid query vector shape")
    vector = _vectors(raw.reshape(1, -1), 1, index.d)
    scores, ids = index.search(vector, index.ntotal)
    ranked = sorted(
        zip(scores[0].tolist(), ids[0].tolist(), strict=True), key=lambda p: (-p[0], p[1])
    )
    return [{**records[i], "score": float(score)} for score, i in ranked[:top_k]]


def rerank(
    query: str, candidates: list[dict[str, Any]], model: Any, top_k: int = 5
) -> list[dict[str, Any]]:
    if len(candidates) != 20 or type(top_k) is not int or not 1 <= top_k <= 5:
        raise ExperimentError("reranking requires exactly 20 candidates and top-k <= 5")
    return rerank_candidates(query, candidates, model, top_k)


def _rank(hits: list[dict[str, Any]], question: dict[str, Any]) -> dict[str, int | None]:
    primary = question["primary_source_id"]
    accepted = set(question["acceptable_source_ids"])
    pages = question["expected_pages"]
    return {
        "primary": _first_rank(hits, lambda h: h["source_id"] == primary),
        "accepted": _first_rank(hits, lambda h: h["source_id"] in accepted),
        "page": None
        if pages is None
        else _first_rank(hits, lambda h: h["source_id"] == primary and h["page_start"] in pages),
    }


def summarize_ranks(
    dataset: list[dict[str, Any]], hits: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, dict[str, int | None]]]:
    ranks = {q["question_id"]: _rank(hits[q["question_id"]], q) for q in dataset}
    metrics = {}
    for field in ("primary", "accepted", "page"):
        values = [
            ranks[q["question_id"]][field]
            for q in dataset
            if field != "page" or q["expected_pages"] is not None
        ]
        metrics.update({f"{field}_{key}": value for key, value in rank_metrics(values).items()})
        metrics[f"{field}_denominator"] = len(values)
        for k in (1, 3, 5):
            metrics[f"{field}_hits_at_{k}"] = sum(
                value is not None and value <= k for value in values
            )
    return metrics, ranks


def baseline_matches(metrics: dict[str, Any]) -> bool:
    return (
        metrics["primary_denominator"] == 80
        and metrics["accepted_denominator"] == 80
        and metrics["page_denominator"] == 56
        and all(
            metrics[key.replace("recall", "hits")] == count for key, count in BASELINE_HITS.items()
        )
        and abs(metrics["primary_mrr_at_5"] - 0.8046) < 0.0001
        and abs(metrics["accepted_mrr_at_5"] - 0.8208) < 0.0001
        and abs(metrics["page_mrr_at_5"] - 0.6658) < 0.0001
    )


def gate(
    prod: dict[str, Any],
    aug: dict[str, Any],
    table_prod: dict[str, Any],
    table_aug: dict[str, Any],
    non_prod: dict[str, Any],
    non_aug: dict[str, Any],
    probes: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "baseline_reproduced": baseline_matches(prod),
        "mandatory_rows_top5": len(probes) == 4
        and all(p["reranked_row_rank_top5"] is not None for p in probes),
        "primary_r5": aug["primary_hits_at_5"] >= prod["primary_hits_at_5"],
        "accepted_r5": aug["accepted_hits_at_5"] >= prod["accepted_hits_at_5"],
        "page_r5": aug["page_hits_at_5"] >= prod["page_hits_at_5"],
        "primary_r1_guard": aug["primary_recall_at_1"]
        >= prod["primary_recall_at_1"] - 0.025 - 1e-12,
        "primary_mrr_guard": aug["primary_mrr_at_5"] >= prod["primary_mrr_at_5"] - 0.025 - 1e-12,
        "table_page_r5": table_aug["page_hits_at_5"] >= table_prod["page_hits_at_5"],
        "non_table_primary_r5": non_aug["primary_hits_at_5"] >= non_prod["primary_hits_at_5"],
    }


def movement(
    before: dict[str, dict[str, int | None]], after: dict[str, dict[str, int | None]]
) -> dict[str, dict[str, list[str]]]:
    result = {}
    for field in ("primary", "accepted", "page"):
        group = {"improved": [], "worsened": [], "unchanged": []}
        for question_id, old in before.items():
            a, b = old[field] or 6, after[question_id][field] or 6
            group["improved" if b < a else "worsened" if b > a else "unchanged"].append(question_id)
        result[field] = group
    return result


def representation_counts(hits: list[dict[str, Any]]) -> dict[str, int]:
    return {
        kind: sum(h["representation"] == kind for h in hits)
        for kind in ("production_chunk", "table_row")
    }


def probe_results(
    artifacts: list[dict[str, Any]],
    dense: dict[str, list[dict[str, Any]]],
    final: dict[str, list[dict[str, Any]]],
    question_to_retrieval: dict[str, str],
) -> list[dict[str, Any]]:
    keyed_dense = {gen: dense[question_to_retrieval[gen]] for gen in PROBE_IDS}
    keyed_final = {gen: final[question_to_retrieval[gen]] for gen in PROBE_IDS}
    probes = evaluate_probes(artifacts, keyed_dense, keyed_final)
    for p in probes:
        gen = p["question_id"]
        top = keyed_final[gen]
        production = [h for h in top if h["representation"] == "production_chunk"]
        p["production_chunk_present"] = bool(production)
        p["first_representation"] = top[0]["representation"]
    return probes


class AugmentedRetrievalAdapter:
    """Experiment-only bridge into the unchanged AnswerService."""

    def __init__(self, index: Any, records: list[dict[str, Any]], embedder: Any, reranker: Any):
        self.index, self.records, self.embedder, self.reranker = index, records, embedder, reranker
        self.last_hits: list[dict[str, Any]] = []
        self.history: dict[str, list[dict[str, Any]]] = {}

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResponse:
        count = 5 if top_k is None else top_k
        if (
            type(count) is not int
            or not 1 <= count <= 5
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise ExperimentError("invalid augmented retrieval request")
        query = question.strip()
        self.last_hits = rerank(
            query,
            dense_search(self.index, self.records, self.embedder, query),
            self.reranker,
            count,
        )
        self.history[query] = list(self.last_hits)
        chunks = [
            RetrievedChunk(
                rank=rank,
                chunk_id=h["chunk_id"],
                text=h["text"],
                source_id=h["source_id"],
                source_title=h["source_title"],
                source_url=h["final_url"],
                source_type=h["source_type"],
                category=h["category"],
                admission_year=h["admission_year"],
                page=h["page_start"],
                score=h["score"],
                dense_score=h["dense_score"],
                rerank_score=h["rerank_score"],
            )
            for rank, h in enumerate(self.last_hits, 1)
        ]
        return RetrievalResponse(query=query, mode="reranked", top_k=count, results=chunks)


def critical_token(answer: str, token: str) -> bool:
    groups = [token[: len(token) % 3 or 3]] + [
        token[i : i + 3] for i in range(len(token) % 3 or 3, len(token), 3)
    ]
    pattern = r"[\s\u00a0\u202f]*".join(map(re.escape, groups))
    return re.search(rf"(?<!\d){pattern}(?!\d)", answer) is not None


def compare_generation(
    baseline: list[dict[str, Any]], augmented: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {r["question_id"]: r for r in baseline}
    if (
        len(augmented) != 16
        or len(by_id) != 16
        or set(by_id) != {r["question_id"] for r in augmented}
    ):
        raise ExperimentError("generation comparison question mismatch")
    rows = []
    for row in augmented:
        old = by_id[row["question_id"]]
        before, after = old["actual_status"], row["actual_status"]
        if before not in {"answered", "insufficient_evidence"} or after not in {
            "answered",
            "insufficient_evidence",
        }:
            status = "error"
        else:
            status = {
                ("insufficient_evidence", "answered"): "recovered_from_refusal",
                ("answered", "answered"): "remained_answered",
                ("answered", "insufficient_evidence"): "new_refusal",
                ("insufficient_evidence", "insufficient_evidence"): "remained_refused",
            }[(before, after)]
        rows.append(
            {
                "question_id": row["question_id"],
                "movement": status,
                "baseline": old,
                "augmented": row,
            }
        )
    tokens = {"gen-008": "686", "gen-011": "8", "gen-021": "60", "gen-027": "179500"}
    probes = [
        {
            "question_id": qid,
            "critical_fact_token_present": critical_token(
                next(r["augmented"]["answer"] or "" for r in rows if r["question_id"] == qid), token
            ),
        }
        for qid, token in tokens.items()
    ]
    return {
        "questions": rows,
        "critical_probes": probes,
        "status_movement": {
            name: sum(r["movement"] == name for r in rows)
            for name in (
                "recovered_from_refusal",
                "remained_answered",
                "new_refusal",
                "remained_refused",
                "error",
            )
        },
        "augmented_latency_mean": statistics.mean(r["elapsed_seconds"] for r in augmented),
        "augmented_latency_median": statistics.median(r["elapsed_seconds"] for r in augmented),
    }
