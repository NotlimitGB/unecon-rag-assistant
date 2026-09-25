"""Validated dual-channel retrieval with the accepted PDF-page diversity rule."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from app.retrieval.corpus import RetrievalError
from app.retrieval.index import RetrievalSession, _vectors
from app.retrieval.reranker import CrossEncoderReranker, Reranker, rerank_candidates
from app.retrieval.table_index import validated_table_index

PRODUCTION_K = 20
TABLE_K = 5
MERGED_K = 25
FINAL_K = 5
PDF_PAGE_LIMIT = 2


def select_pdf_page_diversity(ranked: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if type(top_k) is not int or not 1 <= top_k <= FINAL_K:
        raise RetrievalError("reranked top_k must be from 1 to 5")
    selected = []
    page_counts: Counter[tuple[str, int]] = Counter()
    for hit in ranked:
        if hit["source_type"] == "pdf":
            page = hit["page_start"]
            if type(page) is not int or page < 1 or hit["page_end"] != page:
                raise RetrievalError("invalid PDF page in reranked candidate")
            key = (hit["source_id"], page)
            if page_counts[key] >= PDF_PAGE_LIMIT:
                continue
            page_counts[key] += 1
        elif hit["source_type"] != "html" or hit["page_start"] is not None:
            raise RetrievalError("invalid HTML candidate page")
        selected.append(hit)
        if len(selected) == top_k:
            return selected
    raise RetrievalError("PDF page diversity cannot fill requested result count")


def _rank(index: Any, records: list[dict[str, Any]], query_vector: np.ndarray, count: int):
    scores, ids = index.search(query_vector, index.ntotal)
    ordered = sorted(
        zip(scores[0].tolist(), ids[0].tolist(), strict=True),
        key=lambda pair: (-pair[0], records[pair[1]]["vector_id"]),
    )
    return [{**records[position], "score": float(score)} for score, position in ordered[:count]]


class TableAwareRerankedRetrievalSession:
    """Load both validated indices and models once; never fetch sources at request time."""

    def __init__(
        self,
        dense_session: RetrievalSession,
        manifest_path: Path,
        pdf_root: Path,
        table_dir: Path,
        model_name: str,
        reranker_model: str,
        reranker_device: str,
        reranker_batch_size: int,
        reranker_max_length: int,
        candidate_k: int,
        reranker_factory: Callable[[], Reranker] | None = None,
    ):
        if candidate_k != PRODUCTION_K:
            raise RetrievalError("canonical reranked retrieval requires RERANKER_CANDIDATE_K=20")
        if (
            not reranker_model.strip()
            or reranker_device not in {"auto", "cpu", "cuda"}
            or type(reranker_batch_size) is not int
            or not 1 <= reranker_batch_size <= 64
            or type(reranker_max_length) is not int
            or not 32 <= reranker_max_length <= 4096
        ):
            raise RetrievalError("invalid reranker configuration")
        self.dense_session = dense_session
        self.table_index, self.table_records = validated_table_index(
            manifest_path, pdf_root, table_dir, dense_session.metadata, model_name
        )
        self.reranker = (
            reranker_factory
            or (
                lambda: CrossEncoderReranker(
                    reranker_model, reranker_device, reranker_batch_size, reranker_max_length
                )
            )
        )()
        self.pairs_scored = 0

    def search(self, query: str, top_k: int = FINAL_K) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must not be empty")
        if type(top_k) is not int or not 1 <= top_k <= FINAL_K:
            raise RetrievalError("reranked top_k must be from 1 to 5")
        raw = np.asarray(self.dense_session.embedder.encode_query(query))
        if raw.ndim != 1:
            raise RetrievalError("query embedding must be one-dimensional")
        vector = _vectors(raw.reshape(1, -1), 1, self.dense_session.index.d)
        production = _rank(
            self.dense_session.index, self.dense_session.records, vector, PRODUCTION_K
        )
        table = _rank(self.table_index, self.table_records, vector, TABLE_K)
        if len(production) != PRODUCTION_K or len(table) != TABLE_K:
            raise RetrievalError("dual-channel retrieval requires 20+5 candidates")
        merged = [
            *({**hit, "representation": "production_chunk"} for hit in production),
            *({**hit, "representation": "table_row"} for hit in table),
        ]
        ids = [hit["vector_id"] for hit in merged]
        chunk_ids = [hit["chunk_id"] for hit in merged]
        if len(set(ids)) != MERGED_K or len(set(chunk_ids)) != MERGED_K:
            raise RetrievalError("duplicate merged retrieval candidate")
        if any(not np.isfinite(hit["score"]) for hit in merged):
            raise RetrievalError("non-finite dense score")
        merged.sort(key=lambda hit: (-hit["score"], hit["vector_id"]))
        ranked = rerank_candidates(query, merged, self.reranker, MERGED_K)
        self.pairs_scored += MERGED_K
        return select_pdf_page_diversity(ranked, top_k)
