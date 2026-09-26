"""Application-scoped dependencies exposed through FastAPI."""

from fastapi import Request

from app.generation.service import AnswerService


def get_answer_service(request: Request) -> AnswerService:
    return request.app.state.answer_service
