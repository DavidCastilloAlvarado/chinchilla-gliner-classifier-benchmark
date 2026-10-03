"""Startup lifecycle tests, including the pre-readiness warmup."""

import asyncio
from types import SimpleNamespace

import app.core.lifespan as lifespan_module


class FakeSettings:
    model_name = "decide"
    gpu_mode = "cpu"
    compile_model = False
    batch_window_ms = 10
    max_batch_size = 8
    max_queue_size = 8
    request_timeout_seconds = 30
    apps_dir = "app/apps"


class FakeRegistry:
    def __init__(self, apps_dir) -> None:
        self.apps_dir = apps_dir

    def load(self) -> None:
        events.append("registry.load")

    def first_operation(self) -> dict:
        return {"kind": "classification", "labels": ["warmup"]}


class FakeModel:
    device = "cpu"

    def __init__(self, settings) -> None:
        self.settings = settings

    def load(self) -> None:
        events.append("model.load")

    def warmup(self, operation: dict) -> float:
        events.append(("model.warmup", operation))
        return 0.01


class FakeBatcher:
    def __init__(self, **kwargs) -> None:
        events.append("batcher.init")

    async def start(self) -> None:
        events.append("batcher.start")

    async def stop(self) -> None:
        events.append("batcher.stop")


events: list = []


def test_model_warmup_runs_before_batcher_and_readiness(monkeypatch) -> None:
    events.clear()
    monkeypatch.setattr(lifespan_module, "Settings", lambda: FakeSettings())
    monkeypatch.setattr(lifespan_module, "AppRegistry", FakeRegistry)
    monkeypatch.setattr(lifespan_module, "ModelService", FakeModel)
    monkeypatch.setattr(lifespan_module, "MicroBatcher", FakeBatcher)

    app = SimpleNamespace(state=SimpleNamespace())

    async def exercise() -> None:
        async with lifespan_module.lifespan(app):
            assert app.state.runtime.ready is True
            assert events.index(("model.warmup", {"kind": "classification", "labels": ["warmup"]})) < events.index(
                "batcher.start"
            )

    asyncio.run(exercise())
    assert events[-1] == "batcher.stop"
