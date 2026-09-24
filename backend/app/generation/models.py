"""Strict model output and public answer contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.ingestion.models import validate_official_url
from app.retrieval.service import RetrievalMode, RetrievedChunk

REFUSAL_MESSAGE = (
    "В предоставленных официальных материалах недостаточно информации, чтобы уверенно "
    "ответить на этот вопрос. Рекомендуется проверить актуальную информацию на официальном "
    "сайте СПбГЭУ или обратиться в приёмную комиссию."
)


class GenerationError(ValueError):
    """The local generator failed or returned untrustworthy output."""


class GeneratedAnswer(BaseModel):
    """The only fields the LLM may return."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    cited_context_ids: list[str]

    @model_validator(mode="after")
    def valid_status_and_citations(self) -> "GeneratedAnswer":
        if self.status == "answered" and (
            not self.answer.strip() or not self.cited_context_ids
        ):
            raise ValueError("answered result requires an answer and citations")
        if self.status == "insufficient_evidence" and self.cited_context_ids:
            raise ValueError("insufficient evidence cannot cite contexts")
        if len(self.cited_context_ids) != len(set(self.cited_context_ids)):
            raise ValueError("duplicate context IDs")
        return self


class ContextBlock(BaseModel):
    """One retrieved chunk with a request-local citation ID."""

    model_config = ConfigDict(extra="forbid", strict=True)

    context_id: str
    chunk: RetrievedChunk


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context_id: str
    chunk_id: str
    source_id: str
    source_title: str
    source_url: str
    source_type: Literal["html", "pdf"]
    page: int | None

    @model_validator(mode="after")
    def valid_source(self) -> "AnswerCitation":
        validate_official_url(self.source_url)
        if self.source_type == "pdf" and (type(self.page) is not int or self.page < 1):
            raise ValueError("PDF citation requires a positive page")
        if self.source_type == "html" and self.page is not None:
            raise ValueError("HTML citation must not have a page")
        return self


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    retrieval_mode: RetrievalMode
    citations: list[AnswerCitation]

    @model_validator(mode="after")
    def valid_answer(self) -> "AnswerResponse":
        if self.status == "answered" and (not self.answer.strip() or not self.citations):
            raise ValueError("answered response requires answer and citations")
        if self.status == "insufficient_evidence" and (
            self.answer != REFUSAL_MESSAGE or self.citations
        ):
            raise ValueError("insufficient response must use the fixed refusal")
        return self
