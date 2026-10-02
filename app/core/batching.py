"""A small async micro-batcher for one model instance.

Requests wait up to ``batch_window_ms`` for compatible requests, then the model
call runs in a worker thread so the asyncio event loop remains responsive.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from .metrics import (
    INFLIGHT_BATCHES,
    QUEUE_DEPTH,
    QUEUE_REJECTED,
    REQUEST_TIMEOUTS,
    observe_batch,
)
from .model import ModelService


@dataclass
class _Pending:
    query: str
    operation: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]
    submitted_at: float


class QueueFullError(RuntimeError):
    """Raised when bounded queue capacity is exhausted."""


class RequestTimeoutError(TimeoutError):
    """Raised when a request waits too long for an inference result."""


class MicroBatcher:
    def __init__(
        self,
        model: ModelService,
        window_ms: float,
        max_batch_size: int,
        max_queue_size: int,
        request_timeout_seconds: float,
    ) -> None:
        self.model = model
        self.window_ms = max(0.0, window_ms)
        self.max_batch_size = max(1, max_batch_size)
        self.request_timeout_seconds = max(0.1, request_timeout_seconds)
        self.queue: asyncio.Queue[_Pending] = asyncio.Queue(maxsize=max(1, max_queue_size))
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="inference-microbatcher")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def submit(self, query: str, operation: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        operation_name = operation["kind"]
        pending = _Pending(query, operation, future, time.perf_counter())
        try:
            self.queue.put_nowait(pending)
        except asyncio.QueueFull as exc:
            QUEUE_REJECTED.labels(operation_name).inc()
            future.cancel()
            raise QueueFullError("Inference queue is full; retry later") from exc

        QUEUE_DEPTH.set(self.queue.qsize())
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            future.cancel()
            REQUEST_TIMEOUTS.labels(operation_name).inc()
            raise RequestTimeoutError("Inference request timed out") from exc
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            QUEUE_DEPTH.set(self.queue.qsize())
            if first.future.cancelled():
                self.queue.task_done()
                continue

            pending = [first]
            deadline = time.perf_counter() + self.window_ms / 1000.0

            while len(pending) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    QUEUE_DEPTH.set(self.queue.qsize())
                    if candidate.future.cancelled():
                        self.queue.task_done()
                        continue
                    pending.append(candidate)
                except asyncio.TimeoutError:
                    break

            groups: dict[str, list[_Pending]] = {}
            for item in pending:
                key = json.dumps(item.operation, sort_keys=True, ensure_ascii=False)
                groups.setdefault(key, []).append(item)

            for group in groups.values():
                for offset in range(0, len(group), self.max_batch_size):
                    await self._process(group[offset : offset + self.max_batch_size])

    async def _process(self, items: list[_Pending]) -> None:
        if not items:
            return

        operation = items[0].operation["kind"]
        device = self.model.device
        started = time.perf_counter()
        status = "success"
        INFLIGHT_BATCHES.labels(operation, device).inc()
        try:
            results = await asyncio.to_thread(
                self.model.infer_batch,
                [item.query for item in items],
                items[0].operation,
            )
            inference_ms = (time.perf_counter() - started) * 1000
            if len(results) != len(items):
                raise RuntimeError(f"Model returned {len(results)} results for {len(items)} requests")
            for item, result in zip(items, results):
                if not item.future.done():
                    item.future.set_result(
                        {
                            "result": result,
                            "meta": {
                                "batch_size": len(items),
                                "queue_wait_ms": round((started - item.submitted_at) * 1000, 3),
                                "inference_ms": round(inference_ms, 3),
                                "device": device,
                            },
                        }
                    )
        except Exception as exc:
            status = "error"
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)
        finally:
            duration_seconds = time.perf_counter() - started
            observe_batch(
                operation=operation,
                device=device,
                batch_size=len(items),
                status=status,
                duration_seconds=duration_seconds,
                queue_wait_seconds=[started - item.submitted_at for item in items],
            )
            INFLIGHT_BATCHES.labels(operation, device).dec()
            for _ in items:
                self.queue.task_done()
