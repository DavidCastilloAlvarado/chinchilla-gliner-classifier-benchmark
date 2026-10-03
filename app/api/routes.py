"""Inference HTTP endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.batching import QueueFullError, RequestTimeoutError
from app.core.dependencies import get_runtime
from app.core.lifespan import Runtime

from .schemas import (
    AppUsageRequest,
    ClassificationRequest,
    InferenceResponse,
    SystemOneRequest,
    SystemOneResponse,
    UsageType,
)

router = APIRouter()


SYSTEM_ONE_REQUEST_EXAMPLES = {
    "banking_triage": {
        "summary": "Banking intent triage",
        "description": "Evaluate several independent decisions over one banking message.",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "state": {
                "message": "Perdí mi tarjeta y necesito bloquearla de inmediato.",
                "language": "es",
            },
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which banking intent best describes the message?",
                    "criteria": {
                        "card_block": "The customer wants to block or freeze a lost or stolen card.",
                        "check_balance": "The customer wants to know an account balance.",
                        "transfer_money": "The customer wants to send money to another account.",
                    },
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Does this request require immediate attention?",
                    "criteria": {
                        "true": "A lost or stolen payment card needs prompt blocking.",
                        "false": "The request can wait without immediate risk.",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the customer issue?",
                    "criteria": ["Low impact", "Moderate impact", "High impact"],
                },
            },
        },
    }
}


SYSTEM_ONE_RESPONSE_EXAMPLES = {
    "banking_triage": {
        "summary": "Structured System One decisions",
        "value": {
            "model": "fastino/GLiNER2.5-multi-Decide",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": "card_block",
                    "probabilities": {
                        "card_block": 0.94,
                        "check_balance": 0.03,
                        "transfer_money": 0.03,
                    },
                    "confidence": 0.94,
                },
                "urgent": {
                    "type": "noul",
                    "noul": 0.97,
                },
                "severity": {
                    "type": "score",
                    "score": 1.84,
                    "legend": {
                        "0": "Low impact",
                        "1": "Moderate impact",
                        "2": "High impact",
                    },
                    "probabilities": {
                        "0": 0.02,
                        "1": 0.12,
                        "2": 0.86,
                    },
                    "confidence": 0.86,
                },
            },
            "usage": {
                "input_tokens": 123,
                "output_tokens": 0,
            },
        },
    }
}


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


def _system_one_state_text(state: str | dict[str, Any] | list[Any]) -> str:
    """Serialize System One state without losing structured input."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


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
        "env_file_loaded": runtime.env_file_loaded,
    }


@router.post(
    "/v1/systemone",
    response_model=SystemOneResponse,
    response_model_exclude_none=True,
    summary="Evaluate typed System One decisions",
    description=(
        "Evaluate independent choice, noul, and score questions over one state. "
        "The request and response follow the TypeSafe/System One shape."
    ),
    responses={
        200: {
            "description": "Typed decisions for every submitted question.",
            "content": {
                "application/json": {
                    "examples": SYSTEM_ONE_RESPONSE_EXAMPLES,
                }
            },
        }
    },
)
async def system_one(
    payload: SystemOneRequest = Body(
        ...,
        openapi_examples=SYSTEM_ONE_REQUEST_EXAMPLES,
    ),
    runtime: Runtime = Depends(get_runtime),
) -> SystemOneResponse:
    """Evaluate independent choice, noul, and score questions over one state."""
    served_model = runtime.settings.model_repo
    if payload.model != served_model:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "The requested model is not loaded by this server",
                "requested_model": payload.model,
                "loaded_model": served_model,
            },
        )

    operation = {
        "kind": "systemone",
        "model": served_model,
        "questions": {
            question_id: question.model_dump(mode="json", exclude_none=True)
            for question_id, question in payload.questions.items()
        },
    }
    result = await _submit(runtime, _system_one_state_text(payload.state), operation)
    return SystemOneResponse.model_validate(result["result"])


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
