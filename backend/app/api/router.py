from fastapi import APIRouter

from app.api.routes import answer, health

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(answer.router)
