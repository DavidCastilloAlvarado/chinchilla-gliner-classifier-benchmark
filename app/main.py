"""FastAPI application entrypoint.

Run from the project root with:
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI

from app.api.router import router
from app.core.lifespan import lifespan

app = FastAPI(
    title="GLiNER2 Fast Classification POC",
    version="0.1.0",
    description="Micro-batched classification and extraction with local GLiNER2 models.",
    docs_url="/api/doc",
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.include_router(router)
