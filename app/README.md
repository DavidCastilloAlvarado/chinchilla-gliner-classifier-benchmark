# GLiNER2 FastAPI proof of concept

This directory is an independent serving application. It does not import the
benchmark implementation under `src/`.

## Configuration loading

At startup, the FastAPI lifespan first looks for the project-root `.env` file.
If it exists, `python-dotenv` loads its values into `os.environ`. Existing system
environment variables are not overwritten, so container/Kubernetes environment
variables take precedence over `.env`. If `.env` does not exist, the application
uses the system environment directly.

Create a local configuration with:

```bash
cp .env.example .env
```

`.env` is ignored by Git. The readiness response reports whether the file was
found as `env_file_loaded`.

## Run

The default configuration serves the locally downloaded multilingual Decide
checkpoint on CPU:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For the RTX 5060 Ti benchmark configuration:

```bash
MODEL_NAME=decide \
GPU_MODE=cuda \
COMPILE_MODEL=true \
N_CONCURRENCY=8 \
BATCH_WINDOW_MS=10 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Compilation happens during startup. The readiness endpoint is not available as
ready until model loading and the compilation warmup have finished.

## Endpoints

### Ad-hoc classification

Send candidate classes with optional descriptions:

```bash
curl -s http://localhost:8000/api/v1/classify \
  -H 'content-type: application/json' \
  -d '{
    "query": "Quiero consultar el saldo de mi cuenta.",
    "classes": ["check_balance", "transfer_money", "pay_bill"],
    "descriptions": {
      "check_balance": "Consultar el saldo disponible de una cuenta",
      "transfer_money": "Mover dinero a otra cuenta o persona",
      "pay_bill": "Pagar una factura o servicio"
    }
  }'
```

`descriptions` is optional. The labels can therefore be sent by themselves:

```json
{
  "query": "I need to send money to another account",
  "classes": ["check_balance", "transfer_money", "pay_bill"]
}
```

The response contains `result` and serving metadata:

```json
{
  "result": {"intent": {"label": "transfer_money", "confidence": 0.99}},
  "meta": {
    "batch_size": 1,
    "queue_wait_ms": 10.2,
    "inference_ms": 7.8,
    "device": "cuda"
  }
}
```

### Stored app schemas

A client only sends a query when the schema is stored in `app/apps/`:

```bash
curl -s http://localhost:8000/app/banking_es/usage/classification/ \
  -H 'content-type: application/json' \
  -d '{"query": "Perdí mi tarjeta de débito y necesito bloquearla."}'
```

Extraction uses the same app and URL shape:

```bash
curl -s http://localhost:8000/app/banking_es/usage/extraction/ \
  -H 'content-type: application/json' \
  -d '{"query": "Transferiré 250 dólares a la cuenta de Ana el viernes."}'
```

Each app JSON can define `classification`, `extraction`, or both. Classification
uses `labels`, which can be a list or a label-to-description object. Extraction
uses `entities`, also as a list or a label-to-description object. See
`app/apps/banking_es.json`.

## Architecture

```text
app/
├── main.py                 FastAPI application and router registration
├── api/
│   ├── routes.py           HTTP endpoints and dependency injection
│   └── schemas.py          Pydantic request/response models
├── core/
│   ├── apps.py             Validated JSON app-schema registry
│   ├── batching.py         Async 10 ms micro-batcher
│   ├── config.py           Environment-backed settings
│   ├── dependencies.py     Runtime dependency provider
│   ├── lifespan.py         Startup/shutdown and readiness lifecycle
│   ├── metrics.py          Prometheus batch metrics
│   └── model.py             Local GLiNER2 loading and batched inference
└── apps/
    └── banking_es.json     Stored classification/extraction schemas
```

The lifespan loads the registry and one model instance before starting the
batch worker. Endpoints receive the initialized `Runtime` through FastAPI's
`Depends` mechanism. The worker keeps model execution out of the asyncio event
loop by running each model batch in a dedicated `asyncio.to_thread` call.

## Micro-batching design

Each request is put into a **bounded** `asyncio.Queue`. The worker:

1. Takes the first request.
2. Waits up to `BATCH_WINDOW_MS` (default: 10 ms) for more requests.
3. Groups requests with the same operation/schema.
4. Processes each group in chunks of at most `N_CONCURRENCY` (default: 8).
5. Resolves each request's future with its corresponding result.

`MAX_QUEUE_SIZE` bounds memory usage. If the queue is full, the API immediately
returns HTTP `429` with `Retry-After: 1`; it does not keep allocating pending
request objects. Requests waiting longer than `REQUEST_TIMEOUT_SECONDS` receive
HTTP `504` and are removed from future batching when encountered by the worker.

A Python thread cannot be safely force-killed if a native model call hangs. The
request timeout therefore releases the HTTP request, but the stuck model thread
may still occupy the single model worker. If that happens, the bounded queue
will eventually return `429` rather than growing without limit. For a hard kill
and recovery from a truly hung CUDA/native call, run the model server in a
separate process and let a supervisor or Kubernetes restart the process.

Requests with different ad-hoc labels cannot safely share one GLiNER2
classification task, so they are grouped only when their normalized operation
schemas match. Stored app requests naturally batch well when they use the same
`app_id` and usage type.

This is dynamic batching rather than LLM continuous batching. GLiNER2 is a
stateless encoder/classifier: the complete input batch can be executed in one
forward pass, with no token-by-token decode loop or KV-cache scheduler. The
10 ms window is an intentional latency/throughput tradeoff. `meta.batch_size`
shows whether a request actually shared a batch.

### Relevant serving guidance

- [NVIDIA Triton dynamic batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
  describes combining stateless requests, maximum batch size, and queue delay.
- [vLLM documentation](https://docs.vllm.ai/en/latest/) describes continuous
  batching for autoregressive generation; its scheduler is more sophisticated
  than needed for this stateless GLiNER2 workload.
- [GLiNER2 batch classification tutorial](https://github.com/fastino-ai/gliner2/blob/main/tutorial/1-classification.md)
  documents multi-task and batch classification APIs.
- [FastAPI lifespan events](https://fastapi.tiangolo.com/advanced/events/)
  documents startup/shutdown resource management.

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_NAME` | `decide` | `base`, `multi`, or `decide` |
| `MODEL_DIR` | `temp/<model>` | Override local model directory |
| `GPU_MODE` | `cpu` | `cpu` or `cuda` |
| `COMPILE_MODEL` | `false` | Compile and warm up during startup |
| `BATCH_WINDOW_MS` | `10` | Maximum queueing window |
| `N_CONCURRENCY` | `8` | Maximum requests per model batch |
| `MAX_QUEUE_SIZE` | `256` | Hard cap on queued requests before HTTP 429 |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Maximum request wait before HTTP 504 |
| `APPS_DIR` | `app/apps` | Directory containing app JSON files |

Values may come from `.env` or the system environment. If both define a value,
the system environment wins. `HOST` and `PORT` are Uvicorn command-line options,
not application settings; configure them in the `uvicorn` command.

### Capacity and failure responses

| HTTP status | Cause | Behavior |
|---:|---|---|
| `200` | Inference completed | Returns the model result and batch metadata |
| `429` | Bounded queue is full | Request is rejected immediately with `Retry-After: 1` |
| `503` | Model inference failed or service is not ready | No unbounded retry is performed by the API |
| `504` | Request exceeded `REQUEST_TIMEOUT_SECONDS` | The request future is cancelled |

A timeout cannot force-kill a native PyTorch/CUDA thread. If the model itself
hangs, the worker remains occupied, the bounded queue fills, and new requests
receive `429`. Use process-level supervision for hard recovery.

## Observed GPU stress benchmark

A five-minute Locust run with 40 concurrent virtual users produced the following
result using one FastAPI worker and one NVIDIA RTX 5060 Ti with 16 GB VRAM. The
server used `N_CONCURRENCY=8`, meaning a maximum of eight requests per model
microbatch:

| Metric | Result |
|---|---:|
| Requests | 36,066 |
| Failures | 0 (0.00%) |
| Average response time | 315 ms |
| Median response time | 320 ms |
| Minimum response time | 34 ms |
| Maximum response time | 703 ms |
| Throughput | 120.28 requests/s |
| Locust virtual users | 40 |
| Maximum model microbatch | 8 requests |
| FastAPI workers | 1 |
| GPUs | 1 × RTX 5060 Ti 16 GB |

Approximate response-time percentiles were p50 320 ms, p90 330 ms, p95 340 ms,
p99 410 ms, p99.9 480 ms, p99.99 630 ms, and p100 700 ms. These are complete
HTTP response times, including queueing and model execution. They are distinct
from the Prometheus batch-duration metric, which measures model execution only.

## Prometheus observability

`GET /metrics` exposes Prometheus text-format metrics. The key proof-of-batching
counter is:

```text
gliner_inference_batches_total{operation="classification",device="cuda",batch_size="8",status="success"}
```

It increments once per actual model call containing eight requests.

Filter batch-duration p95 for a specific batch size:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(gliner_inference_batch_duration_seconds_bucket{
      operation="classification",
      device="cuda",
      batch_size="50"
    }[5m])
  )
)
```

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `gliner_inference_batches_total` | Counter | `operation`, `device`, `batch_size`, `status` | Actual model calls; `status` is `success` or `error` |
| `gliner_inference_batch_inputs_total` | Counter | `operation`, `device`, `status` | Requests processed by model calls |
| `gliner_inference_batch_size` | Histogram | `operation`, `device` | Distribution of requests per batch |
| `gliner_inference_batch_duration_seconds` | Histogram | `operation`, `device`, `batch_size` | Wall-clock model duration per batch, filterable by batch size |
| `gliner_inference_queue_wait_seconds` | Histogram | `operation`, `device` | Request wait time before batch execution |
| `gliner_inference_queue_depth` | Gauge | none | Requests currently waiting in the bounded queue |
| `gliner_inference_inflight_batches` | Gauge | `operation`, `device` | Model batches currently executing |
| `gliner_inference_last_batch_size` | Gauge | `operation`, `device` | Most recently started batch size |
| `gliner_inference_queue_rejected_total` | Counter | `operation` | Requests rejected with HTTP 429 because the queue was full |
| `gliner_inference_request_timeouts_total` | Counter | `operation` | Requests that returned HTTP 504 after waiting too long |

The application disables the generated `_created` helper series. Histogram
metrics still expose the standard `_bucket`, `_sum`, and `_count` series.
`operation` is currently `classification` or `extraction`; `batch_size` is a
string label such as `1`, `4`, or `8`. See [`stress/README.md`](../stress/README.md)
for Locust commands, PromQL queries, and interpretation guidance.

Health endpoints:

- `GET /health/live` — process liveness
- `GET /health/ready` — model, batcher, and environment readiness/configuration
- `GET /metrics` — Prometheus metrics
- `GET /api/doc` — Swagger UI
- `GET /api/redoc` — ReDoc
- `GET /api/openapi.json` — OpenAPI schema
