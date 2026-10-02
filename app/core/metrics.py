"""Prometheus metrics for proving and tuning micro-batching."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

BATCHES = Counter(
    "gliner_inference_batches_total",
    "Completed or failed model batches, partitioned by operation and batch size.",
    ["operation", "device", "batch_size", "status"],
)
BATCH_INPUTS = Counter(
    "gliner_inference_batch_inputs_total",
    "Input requests processed by model batches.",
    ["operation", "device", "status"],
)
BATCH_SIZE = Histogram(
    "gliner_inference_batch_size",
    "Observed number of requests in each model batch.",
    ["operation", "device"],
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
BATCH_DURATION = Histogram(
    "gliner_inference_batch_duration_seconds",
    "Wall-clock model inference duration per batch.",
    ["operation", "device"],
)
QUEUE_WAIT = Histogram(
    "gliner_inference_queue_wait_seconds",
    "Time requests wait before their model batch starts.",
    ["operation", "device"],
)
INFLIGHT_BATCHES = Gauge(
    "gliner_inference_inflight_batches",
    "Model batches currently executing.",
    ["operation", "device"],
)
QUEUE_DEPTH = Gauge(
    "gliner_inference_queue_depth",
    "Requests currently waiting in the micro-batcher queue.",
)
QUEUE_REJECTED = Counter(
    "gliner_inference_queue_rejected_total",
    "Requests rejected because the bounded micro-batcher queue was full.",
    ["operation"],
)
REQUEST_TIMEOUTS = Counter(
    "gliner_inference_request_timeouts_total",
    "Requests that timed out while waiting for a model result.",
    ["operation"],
)
LAST_BATCH_SIZE = Gauge(
    "gliner_inference_last_batch_size",
    "Size of the most recently started model batch.",
    ["operation", "device"],
)


def observe_batch(
    *,
    operation: str,
    device: str,
    batch_size: int,
    status: str,
    duration_seconds: float,
    queue_wait_seconds: list[float],
) -> None:
    """Record one model call and its request-level queue waits."""
    BATCHES.labels(operation, device, str(batch_size), status).inc()
    BATCH_INPUTS.labels(operation, device, status).inc(batch_size)
    BATCH_SIZE.labels(operation, device).observe(batch_size)
    BATCH_DURATION.labels(operation, device).observe(duration_seconds)
    for wait_seconds in queue_wait_seconds:
        QUEUE_WAIT.labels(operation, device).observe(wait_seconds)
    LAST_BATCH_SIZE.labels(operation, device).set(batch_size)
