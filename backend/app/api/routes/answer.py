"""Thin synchronous adapter over the canonical answer service."""

import logging
import traceback
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.dependencies import get_answer_service
from app.generation.models import AnswerResponse, GenerationError
from app.generation.service import AnswerService

router = APIRouter()
logger = logging.getLogger(__name__)
UNAVAILABLE_MESSAGE = "Сервис ответов временно недоступен. Попробуйте повторить запрос позже."


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(max_length=2000)

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


@router.post("/answer", response_model=AnswerResponse)
def answer(
    request: AnswerRequest,
    service: Annotated[AnswerService, Depends(get_answer_service)],
) -> AnswerResponse:
    try:
        return service.answer(request.question)
    except GenerationError as exc:
        # Exception strings can contain Pydantic input values. Log only types and locations.
        logger.error(
            "AnswerService failed: %s; cause=%s; frames=%s",
            type(exc).__name__,
            type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
            [(frame.name, frame.lineno) for frame in traceback.extract_tb(exc.__traceback__)],
        )
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc
