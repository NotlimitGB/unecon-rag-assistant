from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.config import settings
from app.generation.service import AnswerService


def create_app(answer_service_factory: Callable[[], AnswerService] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = (answer_service_factory or AnswerService)()
        application.state.answer_service = service
        try:
            yield
        finally:
            service.close()

    application = FastAPI(title=settings.app_name, lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    application.include_router(router)
    return application


app = create_app()
