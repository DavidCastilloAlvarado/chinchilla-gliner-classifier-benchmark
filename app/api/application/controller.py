"""FastAPI controller for health, legacy, and stored-application endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.dependencies import get_runtime
from app.core.lifespan import Runtime

from .schema import AppUsageRequest, ClassificationRequest, InferenceResponse, UsageType
from .service import ApplicationService, get_application_service

router = APIRouter()

CLASSIFICATION_REQUEST_EXAMPLES = {
    "banking_intent": {
        "summary": "Banking intent classification",
        "value": {
            "query": "Quiero consultar el saldo de mi cuenta.",
            "classes": ["check_balance", "transfer_money", "pay_bill"],
            "descriptions": {
                "check_balance": "Consultar el saldo disponible de una cuenta",
                "transfer_money": "Mover dinero a otra cuenta o persona",
                "pay_bill": "Pagar una factura o servicio",
            },
        },
    }
}


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health/ready")
async def readiness(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return {
        "status": "ready",
        "model": runtime.settings.model_name,
        "device": runtime.model.device,
        "batch_window_ms": runtime.settings.batch_window_ms,
        "n_concurrency": runtime.settings.max_batch_size,
        "max_queue_size": runtime.settings.max_queue_size,
        "request_timeout_seconds": runtime.settings.request_timeout_seconds,
        "compile_model": runtime.settings.compile_model,
        "warmup_seconds": runtime.warmup_seconds,
        "env_file_loaded": runtime.env_file_loaded,
    }


@router.post("/api/v1/classify", response_model=InferenceResponse)
async def classify(
    payload: ClassificationRequest = Body(..., openapi_examples=CLASSIFICATION_REQUEST_EXAMPLES),
    service: ApplicationService = Depends(get_application_service),
) -> InferenceResponse:
    return await service.classify(payload)


@router.post("/app/{app_id}/usage/{usage_type}/", response_model=InferenceResponse)
async def app_usage(
    app_id: str,
    usage_type: UsageType,
    payload: AppUsageRequest,
    service: ApplicationService = Depends(get_application_service),
) -> InferenceResponse:
    return await service.use_app(app_id, usage_type, payload)
