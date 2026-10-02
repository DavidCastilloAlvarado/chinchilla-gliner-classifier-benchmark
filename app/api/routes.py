"""Inference HTTP endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.batching import QueueFullError, RequestTimeoutError
from app.core.dependencies import get_runtime
from app.core.lifespan import Runtime

from .schemas import AppUsageRequest, ClassificationRequest, InferenceResponse, UsageType

router = APIRouter()


def _labels_with_descriptions(classes: list[str], descriptions: dict[str, str] | None) -> list[str] | dict[str, str]:
    if not descriptions:
        return classes
    unknown = set(descriptions).difference(classes)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Descriptions contain labels not present in classes: {sorted(unknown)}",
        )
    return {label: descriptions.get(label, label) for label in classes}


def _inference_response(payload: dict[str, Any]) -> InferenceResponse:
    return InferenceResponse.model_validate(payload)


async def _submit(runtime: Runtime, query: str, operation: dict[str, Any]) -> dict[str, Any]:
    try:
        return await runtime.batcher.submit(query, operation)
    except QueueFullError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
    except RequestTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Inference failed") from exc


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
    }


@router.post("/api/v1/classify", response_model=InferenceResponse)
async def classify(
    payload: ClassificationRequest,
    runtime: Runtime = Depends(get_runtime),
) -> InferenceResponse:
    operation = {
        "kind": "classification",
        "labels": _labels_with_descriptions(payload.classes, payload.descriptions),
        "multi_label": payload.multi_label,
        "threshold": payload.threshold,
    }
    result = await _submit(runtime, payload.query, operation)
    return _inference_response(result)


@router.post("/app/{app_id}/usage/{usage_type}/", response_model=InferenceResponse)
async def app_usage(
    app_id: str,
    usage_type: UsageType,
    payload: AppUsageRequest,
    runtime: Runtime = Depends(get_runtime),
) -> InferenceResponse:
    try:
        operation = runtime.registry.operation(app_id, usage_type)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = await _submit(runtime, payload.query, operation)
    return _inference_response(result)
