"""Optional cross-encoder reranking of validated dense candidates."""

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from app.retrieval.corpus import RetrievalError
from app.retrieval.index import RetrievalSession


class Reranker(Protocol):
    model_name: str
    device: str

    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str, batch_size: int, max_length: int):
        import torch
        from sentence_transformers import CrossEncoder

        if device == "cuda" and not torch.cuda.is_available():
            raise RetrievalError("CUDA was requested for reranker but is unavailable")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        import torch

        scores = self.model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            activation_fn=torch.nn.Identity(),
            apply_softmax=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(scores)


def rerank_candidates(
    query: str, candidates: list[dict[str, Any]], reranker: Reranker, top_k: int
) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must not be empty")
    if type(top_k) is not int or top_k < 1 or top_k > len(candidates):
        raise RetrievalError("invalid reranked top_k")
    for hit in candidates:
        score = hit.get("score")
        if type(score) not in (float, int) or not np.isfinite(score):
            raise RetrievalError("invalid dense score")
    try:
        scores = np.asarray(reranker.score(query, [hit["text"] for hit in candidates]))
    except (TypeError, ValueError, KeyError) as exc:
        raise RetrievalError(f"invalid reranker output: {exc}") from exc
    if scores.ndim != 1 or len(scores) != len(candidates):
        raise RetrievalError("reranker score count or shape mismatch")
    if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
        scores.dtype, np.complexfloating
    ):
        raise RetrievalError("reranker scores must be real numbers")
    try:
        numeric = scores.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetrievalError(f"invalid reranker scores: {exc}") from exc
    if not np.isfinite(numeric).all():
        raise RetrievalError("reranker scores must be finite")
    ranked = [
        {
            **hit,
            "score": float(score),
            "dense_rank": rank,
            "dense_score": hit["score"],
            "rerank_score": float(score),
        }
        for rank, (hit, score) in enumerate(zip(candidates, numeric, strict=True), 1)
    ]
    ranked.sort(key=lambda hit: (-hit["rerank_score"], hit["dense_rank"], hit["vector_id"]))
    return ranked[:top_k]


class RerankedRetrievalSession:
    """Reuse one dense session and one lazily initialized reranker."""

    def __init__(
        self,
        dense_session: RetrievalSession,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        candidate_k: int = 20,
        reranker_factory: Callable[[], Reranker] | None = None,
    ):
        if (
            not model_name.strip()
            or device not in {"auto", "cpu", "cuda"}
            or type(batch_size) is not int
            or not 1 <= batch_size <= 64
            or type(max_length) is not int
            or not 32 <= max_length <= 4096
            or type(candidate_k) is not int
            or not 5 <= candidate_k <= 100
        ):
            raise RetrievalError("invalid reranker configuration")
        self.dense_session = dense_session
        self.candidate_k = candidate_k
        self.reranker = (
            reranker_factory
            or (lambda: CrossEncoderReranker(model_name, device, batch_size, max_length))
        )()
        self.pairs_scored = 0

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5):
        if type(top_k) is not int or not 1 <= top_k <= self.candidate_k:
            raise RetrievalError("top_k must be from 1 to candidate_k")
        if len(candidates) > self.candidate_k or not candidates:
            raise RetrievalError("invalid dense candidate count")
        self.pairs_scored += len(candidates)
        return rerank_candidates(query, candidates, self.reranker, min(top_k, len(candidates)))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must not be empty")
        if type(top_k) is not int or not 1 <= top_k <= self.candidate_k:
            raise RetrievalError("top_k must be from 1 to candidate_k")
        return self.rerank(query, self.dense_session.search(query, self.candidate_k), top_k)
