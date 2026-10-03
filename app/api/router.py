"""Application-level API router registration."""

from fastapi import APIRouter

from app.api.application.controller import router as application_router
from app.api.extractor.controller import router as extractor_router
from app.api.systemone.controller import router as systemone_router

router = APIRouter()
router.include_router(systemone_router)
router.include_router(extractor_router)
router.include_router(application_router)

__all__ = ["router"]
