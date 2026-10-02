"""FastAPI dependency providers for the initialized runtime."""

from __future__ import annotations

from fastapi import HTTPException, Request

from .lifespan import Runtime


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.ready:
        raise HTTPException(status_code=503, detail="Inference service is not ready")
    return runtime
