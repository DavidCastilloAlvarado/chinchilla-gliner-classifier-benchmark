"""Shared transport helpers for endpoint services."""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from app.core.batching import QueueFullError, RequestTimeoutError
from app.core.lifespan import Runtime


def state_to_text(state: str | dict[str, Any] | list[Any]) -> str:
    """Serialize a state for decision inference."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def extraction_state_to_text(state: str | dict[str, Any] | list[Any]) -> str:
    """Use a text-bearing state field so entity offsets map to source text."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        for key in ("text", "message", "content"):
            value = state.get(key)
            if isinstance(value, str) and value:
                return value
    return state_to_text(state)


def require_loaded_model(runtime: Runtime, requested_model: str) -> str:
    """Require that a single-model server receives its loaded model identifier."""
    loaded_model = runtime.settings.model_repo
    if requested_model != loaded_model:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "The requested model is not loaded by this server",
                "requested_model": requested_model,
                "loaded_model": loaded_model,
            },
        )
    return loaded_model


async def submit_inference(runtime: Runtime, query: str, operation: dict[str, Any]) -> dict[str, Any]:
    """Submit work and translate queue failures into HTTP responses."""
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
