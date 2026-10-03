"""FastAPI lifespan: load resources before readiness and release them on shutdown."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI

from .apps import AppRegistry
from .batching import MicroBatcher
from .config import ENV_FILE, Settings
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
    settings = Settings()
    env_file_loaded = ENV_FILE.is_file()
    registry = AppRegistry(settings.apps_dir)
    registry.load()

    model = ModelService(settings)
    model.load()

    # Always execute one real inference before accepting traffic. This primes
    # CUDA lazy initialization and model kernels; when torch.compile is enabled,
    # it also traces the loaded model before the first user request arrives.
    warmup_operation = registry.first_operation() or {
        "kind": "classification",
        "labels": ["warmup"],
        "multi_label": False,
        "threshold": 0.5,
    }
    warmup_seconds = model.warmup(warmup_operation)

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
