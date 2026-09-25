"""Two isolated dense channels with one unchanged cross-encoder reranker."""

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.experiments.table_aware.augmented import (
    EXPECTED_PRODUCTION,
    EXPECTED_ROWS,
    baseline_matches,
    table_records,
)
from app.experiments.table_aware.extract import ExperimentError
from app.experiments.table_aware.retrieval import _atomic_bytes
from app.retrieval.index import _vectors
from app.retrieval.reranker import rerank_candidates
from app.retrieval.service import RetrievalResponse, RetrievedChunk

PRODUCTION_K = 20
TABLE_K = 5
FINAL_K = 5


def build_table_channel(
    artifacts: list[dict[str, Any]], sources: dict[str, Any], embedder: Any, dimension: int
) -> tuple[Any, list[dict[str, Any]]]:
    """Embed only the verified Task014 rows; global IDs follow production IDs."""
    records = table_records(artifacts, sources)
    vectors = _vectors(
        embedder.encode_documents([record["text"] for record in records]),
        EXPECTED_ROWS,
        dimension,
    )
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    if index.ntotal != EXPECTED_ROWS:
        raise ExperimentError("table-only index count mismatch")
    return index, records


def save_table_channel(
    output_dir: Path, index: Any, records: list[dict[str, Any]], model: str, production_sha: str
) -> None:
    content = faiss.serialize_index(index).tobytes()
    metadata = {
        "schema_version": 1,
        "experiment_id": "table-aware-dual-channel-v1",
        "model": model,
        "production_index_sha256": production_sha,
        "index_sha256": hashlib.sha256(content).hexdigest(),
        "vector_count": index.ntotal,
        "records": records,
    }
    _atomic_bytes(output_dir / "index.faiss", content)
    _atomic_bytes(
        output_dir / "metadata.json",
        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _rank_index(
    index: Any, records: list[dict[str, Any]], vector: np.ndarray, count: int
) -> list[dict[str, Any]]:
    scores, ids = index.search(vector, index.ntotal)
    ranked = sorted(
        zip(scores[0].tolist(), ids[0].tolist(), strict=True),
        key=lambda pair: (-pair[0], records[pair[1]]["vector_id"]),
    )
    return [{**records[position], "score": float(score)} for score, position in ranked[:count]]


def search_channels(
    production_index: Any,
    production_records: list[dict[str, Any]],
    table_index: Any,
    table_records: list[dict[str, Any]],
    embedder: Any,
    query: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(query, str) or not query.strip():
        raise ExperimentError("query must not be empty")
    if (
        not isinstance(production_index, faiss.IndexFlatIP)
        or not isinstance(table_index, faiss.IndexFlatIP)
        or production_index.ntotal != EXPECTED_PRODUCTION
        or table_index.ntotal != EXPECTED_ROWS
        or production_index.d != table_index.d
    ):
        raise ExperimentError("dual-channel index mismatch")
    raw = np.asarray(embedder.encode_query(query))
    if raw.ndim != 1:
        raise ExperimentError("query embedding must be one-dimensional")
    vector = _vectors(raw.reshape(1, -1), 1, production_index.d)
    production = _rank_index(production_index, production_records, vector, PRODUCTION_K)
    table = _rank_index(table_index, table_records, vector, TABLE_K)
    return (
        [{**hit, "representation": "production_chunk"} for hit in production],
        [{**hit, "representation": "table_row"} for hit in table],
    )


def preservation_check(
    production: list[dict[str, Any]], canonical: list[dict[str, Any]], tolerance: float = 1e-6
) -> bool:
    return len(production) == len(canonical) == PRODUCTION_K and all(
        left["chunk_id"] == right["chunk_id"]
        and left["vector_id"] == right["vector_id"]
        and abs(left["score"] - right["score"]) <= tolerance
        for left, right in zip(production, canonical, strict=True)
    )


def merge_candidates(
    production: list[dict[str, Any]], table: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(production) != PRODUCTION_K or len(table) != TABLE_K:
        raise ExperimentError("dual channel requires 20 production and 5 table candidates")
    if any(hit["representation"] != "production_chunk" for hit in production) or any(
        hit["representation"] != "table_row" for hit in table
    ):
        raise ExperimentError("invalid candidate representation")
    merged = [*production, *table]
    ids = [hit["vector_id"] for hit in merged]
    if len(set(ids)) != 25 or any(type(i) is not int for i in ids):
        raise ExperimentError("duplicate or invalid experiment-global vector ID")
    if any(not np.isfinite(hit["score"]) for hit in merged):
        raise ExperimentError("non-finite dense score")
    return sorted(merged, key=lambda hit: (-hit["score"], hit["vector_id"]))


def rerank_merged(query: str, merged: list[dict[str, Any]], reranker: Any) -> list[dict[str, Any]]:
    if len(merged) != PRODUCTION_K + TABLE_K:
        raise ExperimentError("reranker requires 25 merged candidates")
    return rerank_candidates(query, merged, reranker, FINAL_K)


def new_misses(
    before: dict[str, dict[str, int | None]],
    after: dict[str, dict[str, int | None]],
    ids: set[str],
    field: str,
) -> list[str]:
    return sorted(
        question_id
        for question_id in ids
        if before[question_id][field] is not None and after[question_id][field] is None
    )


def recovered_misses(
    before: dict[str, dict[str, int | None]],
    after: dict[str, dict[str, int | None]],
    ids: set[str],
    field: str,
) -> list[str]:
    return sorted(
        question_id
        for question_id in ids
        if before[question_id][field] is None and after[question_id][field] is not None
    )


def partition_questions(
    all_questions: list[dict[str, Any]], table_question_ids: set[str]
) -> tuple[set[str], set[str]]:
    all_ids = {question["question_id"] for question in all_questions}
    non_table_ids = all_ids - table_question_ids
    if (
        len(all_questions) != 80
        or len(all_ids) != 80
        or len(table_question_ids) != 16
        or not table_question_ids <= all_ids
        or len(non_table_ids) != 64
        or table_question_ids & non_table_ids
        or table_question_ids | non_table_ids != all_ids
    ):
        raise ExperimentError("frozen question partition changed")
    return table_question_ids, non_table_ids


def retrieval_gate(
    production: dict[str, Any],
    dual: dict[str, Any],
    table_production: dict[str, Any],
    table_dual: dict[str, Any],
    probes: list[dict[str, Any]],
    preserved: bool,
    new_non_table_primary: list[str],
    new_non_table_accepted: list[str],
) -> dict[str, bool]:
    """The eleven predeclared Task016 criteria; never adjusted by CLI settings."""
    return {
        "baseline_reproduced": baseline_matches(production),
        "production_candidates_preserved": preserved,
        "mandatory_rows_top5": len(probes) == 4
        and all(probe["reranked_row_rank_top5"] is not None for probe in probes),
        "primary_r5": dual["primary_hits_at_5"] >= production["primary_hits_at_5"],
        "accepted_r5": dual["accepted_hits_at_5"] >= production["accepted_hits_at_5"],
        "page_r5": dual["page_hits_at_5"] >= production["page_hits_at_5"],
        "primary_r1_guard": dual["primary_recall_at_1"]
        >= production["primary_recall_at_1"] - 0.025 - 1e-12,
        "primary_mrr_guard": dual["primary_mrr_at_5"]
        >= production["primary_mrr_at_5"] - 0.025 - 1e-12,
        "table_page_r5": table_dual["page_hits_at_5"] >= table_production["page_hits_at_5"],
        "non_table_no_new_primary_miss": not new_non_table_primary,
        "non_table_no_new_accepted_miss": not new_non_table_accepted,
    }


class DualChannelRetrievalAdapter:
    """Experiment-local RetrievalResponse boundary for the unchanged AnswerService."""

    def __init__(
        self,
        production_index: Any,
        production_records: list[dict[str, Any]],
        table_index: Any,
        table_records: list[dict[str, Any]],
        embedder: Any,
        reranker: Any,
    ):
        self.production_index = production_index
        self.production_records = production_records
        self.table_index = table_index
        self.table_records = table_records
        self.embedder = embedder
        self.reranker = reranker
        self.history: dict[str, list[dict[str, Any]]] = {}

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResponse:
        count = FINAL_K if top_k is None else top_k
        if type(count) is not int or not 1 <= count <= FINAL_K:
            raise ExperimentError("invalid dual-channel top_k")
        query = question.strip() if isinstance(question, str) else ""
        production, table = search_channels(
            self.production_index,
            self.production_records,
            self.table_index,
            self.table_records,
            self.embedder,
            query,
        )
        hits = rerank_candidates(query, merge_candidates(production, table), self.reranker, count)
        self.history[query] = hits
        chunks = [
            RetrievedChunk(
                rank=rank,
                chunk_id=hit["chunk_id"],
                text=hit["text"],
                source_id=hit["source_id"],
                source_title=hit["source_title"],
                source_url=hit["final_url"],
                source_type=hit["source_type"],
                category=hit["category"],
                admission_year=hit["admission_year"],
                page=hit["page_start"],
                score=hit["score"],
                dense_score=hit["dense_score"],
                rerank_score=hit["rerank_score"],
            )
            for rank, hit in enumerate(hits, 1)
        ]
        return RetrievalResponse(query=query, mode="reranked", top_k=count, results=chunks)
