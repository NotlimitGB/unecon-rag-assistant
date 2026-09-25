"""Stable application retrieval boundary over the existing dense and reranked sessions."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.config import Settings, settings
from app.ingestion.models import validate_official_url
from app.retrieval.corpus import RetrievalError
from app.retrieval.index import RetrievalSession
from app.retrieval.table_session import TableAwareRerankedRetrievalSession

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RetrievalMode = Literal["dense", "reranked"]


class SearchSession(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class RetrievedChunk(BaseModel):
    """One exact chunk and its user-facing official provenance."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rank: int = Field(ge=1)
    chunk_id: str
    text: str
    source_id: str
    source_title: str
    source_url: str
    source_type: Literal["html", "pdf"]
    category: str
    admission_year: int = Field(gt=0)
    page: int | None
    score: float
    dense_score: float
    rerank_score: float | None

    @field_validator("chunk_id", "text", "source_id", "source_title", "category")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("retrieved field must not be empty")
        return value

    @field_validator("source_url")
    @classmethod
    def official_url(cls, value: str) -> str:
        return validate_official_url(value)

    @field_validator("score", "dense_score", "rerank_score")
    @classmethod
    def finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval scores must be finite")
        return value

    @model_validator(mode="after")
    def page_matches_source(self) -> "RetrievedChunk":
        if self.source_type == "pdf" and (type(self.page) is not int or self.page < 1):
            raise ValueError("PDF result requires a positive page")
        if self.source_type == "html" and self.page is not None:
            raise ValueError("HTML result must not have a page")
        return self


class RetrievalResponse(BaseModel):
    """Portable result consumed by future answer generation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    query: str
    mode: RetrievalMode
    top_k: int = Field(ge=1, le=20)
    results: list[RetrievedChunk]

    @model_validator(mode="after")
    def consistent_ranks_and_scores(self) -> "RetrievalResponse":
        if len(self.results) > self.top_k:
            raise ValueError("too many retrieved results")
        for rank, chunk in enumerate(self.results, 1):
            if chunk.rank != rank:
                raise ValueError("retrieval ranks must be sequential")
            if self.mode == "dense":
                if chunk.rerank_score is not None or chunk.score != chunk.dense_score:
                    raise ValueError("invalid dense score semantics")
            elif chunk.rerank_score is None or chunk.score != chunk.rerank_score:
                raise ValueError("invalid reranked score semantics")
        return self


class RetrievalService:
    """Initialize validated sessions on first use and reuse them for later requests."""

    def __init__(
        self,
        config: Settings | None = None,
        mode: RetrievalMode | None = None,
        manifest_path: Path = PROJECT_ROOT / "data/source_manifest.json",
        chunks_dir: Path = PROJECT_ROOT / "data/processed/chunks",
        index_dir: Path = PROJECT_ROOT / "data/processed/index",
        dense_factory: Callable[[], SearchSession] | None = None,
        reranked_factory: Callable[[SearchSession], SearchSession] | None = None,
        pdf_root: Path = PROJECT_ROOT / "data/processed/pdf",
        table_index_dir: Path = PROJECT_ROOT / "data/processed/table_index",
    ):
        self.config = config or settings
        self.mode = mode if mode is not None else self.config.retrieval_mode
        if self.mode not in ("dense", "reranked"):
            raise RetrievalError("retrieval mode must be dense or reranked")
        self.manifest_path = manifest_path
        self.chunks_dir = chunks_dir
        self.index_dir = index_dir
        self.pdf_root = pdf_root
        self.table_index_dir = table_index_dir
        self._dense_factory = dense_factory
        self._reranked_factory = reranked_factory
        self._dense_session: SearchSession | None = None
        self._reranked_session: SearchSession | None = None

    def _session(self) -> SearchSession:
        if self._dense_session is None:
            self._dense_session = (
                self._dense_factory()
                if self._dense_factory is not None
                else RetrievalSession(
                    self.manifest_path,
                    self.chunks_dir,
                    self.index_dir,
                    self.config.embedding_model,
                    self.config.embedding_device,
                    self.config.embedding_batch_size,
                )
            )
        if self.mode == "dense":
            return self._dense_session
        if self._reranked_session is None:
            try:
                self._reranked_session = (
                    self._reranked_factory(self._dense_session)
                    if self._reranked_factory is not None
                    else TableAwareRerankedRetrievalSession(
                        self._dense_session,
                        self.manifest_path,
                        self.pdf_root,
                        self.table_index_dir,
                        self.config.embedding_model,
                        self.config.reranker_model,
                        self.config.reranker_device,
                        self.config.reranker_batch_size,
                        self.config.reranker_max_length,
                        self.config.reranker_candidate_k,
                    )
                )
            except Exception as exc:
                raise RetrievalError(f"reranked retrieval initialization failed: {exc}") from exc
        return self._reranked_session

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResponse:
        if not isinstance(question, str) or not question.strip():
            raise RetrievalError("question must be a non-empty string")
        query = question.strip()
        count = self.config.retrieval_top_k if top_k is None else top_k
        limit = 20 if self.mode == "dense" else 5
        if type(count) is not int or not 1 <= count <= limit:
            raise RetrievalError(f"top_k must be from 1 to {limit} in {self.mode} mode")
        if self.mode == "reranked" and self.config.reranker_candidate_k != 20:
            raise RetrievalError("canonical reranked retrieval requires RERANKER_CANDIDATE_K=20")
        session = self._session()
        try:
            hits = session.search(query, count)
            if not isinstance(hits, list):
                raise RetrievalError("retrieval session returned a non-list result")
            results = [
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
                    dense_score=hit["score"] if self.mode == "dense" else hit["dense_score"],
                    rerank_score=None if self.mode == "dense" else hit["rerank_score"],
                )
                for rank, hit in enumerate(hits, 1)
            ]
            return RetrievalResponse(query=query, mode=self.mode, top_k=count, results=results)
        except (KeyError, TypeError, ValidationError, ValueError, RuntimeError) as exc:
            raise RetrievalError(f"invalid {self.mode} retrieval result: {exc}") from exc
