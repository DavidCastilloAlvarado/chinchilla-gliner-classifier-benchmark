"""FastAPI lifespan: load resources before readiness and release them on shutdown."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI

from .apps import AppRegistry
from .batching import MicroBatcher
from .config import Settings, load_environment
from .model import ModelService


@dataclass
class Runtime:
    settings: Settings
    registry: AppRegistry
    model: ModelService
    batcher: MicroBatcher
    ready: bool = False
    warmup_seconds: float | None = None
    env_file_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    env_file_loaded = load_environment()
    settings = Settings.from_env()
    registry = AppRegistry(settings.apps_dir)
    registry.load()

    model = ModelService(settings)
    model.load()

    warmup_seconds = None
    if settings.compile_model:
        operation = registry.first_operation() or {
            "kind": "classification",
            "labels": ["warmup"],
            "multi_label": False,
            "threshold": 0.5,
        }
        warmup_seconds = model.warmup(operation)

    batcher = MicroBatcher(
        model=model,
        window_ms=settings.batch_window_ms,
        max_batch_size=settings.max_batch_size,
        max_queue_size=settings.max_queue_size,
        request_timeout_seconds=settings.request_timeout_seconds,
    )
    await batcher.start()

    runtime = Runtime(
        settings=settings,
        registry=registry,
        model=model,
        batcher=batcher,
        ready=True,
        warmup_seconds=warmup_seconds,
        env_file_loaded=env_file_loaded,
    )
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.ready = False
        await batcher.stop()
