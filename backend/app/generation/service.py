"""One retrieval, one generation, and trusted citation mapping per answer."""

from typing import Protocol

from pydantic import ValidationError

from app.config import Settings, settings
from app.generation.models import (
    REFUSAL_MESSAGE,
    AnswerCitation,
    AnswerResponse,
    ContextBlock,
    GeneratedAnswer,
    GenerationError,
)
from app.generation.ollama import OllamaGroundedGenerator
from app.retrieval.corpus import RetrievalError
from app.retrieval.service import RetrievalService


class GroundedGenerator(Protocol):
    def generate(self, question: str, contexts: list[ContextBlock]) -> GeneratedAnswer: ...


class AnswerService:
    """Reuse canonical retrieval and a single local generation provider."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generator: GroundedGenerator | None = None,
        config: Settings | None = None,
    ):
        self.config = config or settings
        self.retrieval_service = retrieval_service or RetrievalService(config=self.config)
        self.generator = generator or OllamaGroundedGenerator(config=self.config)
        self._owns_generator = generator is None

    def close(self) -> None:
        if self._owns_generator:
            self.generator.close()

    def answer(self, question: str) -> AnswerResponse:
        if not isinstance(question, str) or not question.strip():
            raise GenerationError("question must be a non-empty string")
        try:
            retrieved = self.retrieval_service.retrieve(question)
        except RetrievalError as exc:
            raise GenerationError(f"retrieval failed: {exc}") from exc
        contexts = [
            ContextBlock(context_id=f"C{rank}", chunk=chunk)
            for rank, chunk in enumerate(retrieved.results, 1)
        ]
        try:
            generated = GeneratedAnswer.model_validate(
                self.generator.generate(retrieved.query, contexts)
            )
        except ValidationError as exc:
            raise GenerationError(f"invalid generated answer: {exc}") from exc
        if generated.status == "insufficient_evidence":
            return AnswerResponse(
                query=retrieved.query,
                status="insufficient_evidence",
                answer=REFUSAL_MESSAGE,
                retrieval_mode=retrieved.mode,
                citations=[],
            )
        indexed = {context.context_id: context.chunk for context in contexts}
        unknown = [item for item in generated.cited_context_ids if item not in indexed]
        if unknown:
            raise GenerationError(f"unknown cited context IDs: {', '.join(unknown)}")
        citations = [
            AnswerCitation(
                context_id=context_id,
                chunk_id=indexed[context_id].chunk_id,
                source_id=indexed[context_id].source_id,
                source_title=indexed[context_id].source_title,
                source_url=indexed[context_id].source_url,
                source_type=indexed[context_id].source_type,
                page=indexed[context_id].page,
            )
            for context_id in generated.cited_context_ids
        ]
        return AnswerResponse(
            query=retrieved.query,
            status="answered",
            answer=generated.answer,
            retrieval_mode=retrieved.mode,
            citations=citations,
        )
