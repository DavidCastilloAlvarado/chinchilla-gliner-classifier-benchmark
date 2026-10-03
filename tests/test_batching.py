"""Unit tests for async micro-batching and backpressure."""

import asyncio
import unittest

from app.core.batching import MicroBatcher, QueueFullError, RequestTimeoutError


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def infer_batch(self, texts: list[str], operation: dict) -> list[dict]:
        self.calls.append((texts, operation))
        return [{"text": text} for text in texts]


class MicroBatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_compatible_requests_share_a_model_batch(self) -> None:
        model = FakeModel()
        batcher = MicroBatcher(
            model=model,
            window_ms=50,
            max_batch_size=2,
            max_queue_size=8,
            request_timeout_seconds=1,
        )
        operation = {"kind": "classification", "labels": ["a", "b"]}
        await batcher.start()
        try:
            first, second = await asyncio.gather(
                batcher.submit("first", operation),
                batcher.submit("second", operation),
            )
        finally:
            await batcher.stop()

        assert len(model.calls) == 1
        assert model.calls[0][0] == ["first", "second"]
        assert first["result"] == {"text": "first"}
        assert second["result"] == {"text": "second"}
        assert first["meta"]["batch_size"] == 2
        assert second["meta"]["batch_size"] == 2

    async def test_request_timeout_is_reported_when_worker_is_not_started(self) -> None:
        batcher = MicroBatcher(
            model=FakeModel(),
            window_ms=1,
            max_batch_size=1,
            max_queue_size=2,
            request_timeout_seconds=0.01,
        )

        with self.assertRaises(RequestTimeoutError):
            await batcher.submit("never processed", {"kind": "classification"})

    async def test_full_queue_is_rejected(self) -> None:
        batcher = MicroBatcher(
            model=FakeModel(),
            window_ms=1,
            max_batch_size=1,
            max_queue_size=1,
            request_timeout_seconds=1,
        )
        first = asyncio.create_task(
            batcher.submit("first", {"kind": "classification"})
        )
        while batcher.queue.qsize() == 0:
            await asyncio.sleep(0)

        try:
            with self.assertRaises(QueueFullError):
                await batcher.submit("second", {"kind": "classification"})
        finally:
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
