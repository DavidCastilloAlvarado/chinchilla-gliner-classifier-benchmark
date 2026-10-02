# Locust micro-batching stress test

This load test sends concurrent requests to one FastAPI pod. It validates the
server-side batcher through the `meta.batch_size` response field and the
Prometheus `/metrics` endpoint.

## Start the server

```bash
MODEL_NAME=decide \
GPU_MODE=cuda \
COMPILE_MODEL=true \
N_CONCURRENCY=8 \
BATCH_WINDOW_MS=10 \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait until this reports `status: ready`:

```bash
curl -s http://127.0.0.1:8000/health/ready
```

## Run Locust

Interactive UI:

```bash
uv run locust \
  -f stress/locustfile.py \
  -H http://127.0.0.1:8000
```

Open <http://127.0.0.1:8089> and configure Locust users independently from the
server's `N_CONCURRENCY`. Locust `-u` is the number of concurrent virtual users;
`N_CONCURRENCY` is the maximum number of requests inside one model microbatch.
Use enough Locust users and a suitable spawn rate to create overlapping requests
within the batching window.

Headless run:

```bash
uv run locust \
  -f stress/locustfile.py \
  --headless \
  -u 16 \
  -r 16 \
  -t 30s \
  -H http://127.0.0.1:8000
```

The default test uses the stored `banking_es` classification schema.

## RTX 5060 Ti benchmark: 40 concurrent users

The following five-minute Locust run used one FastAPI worker on one NVIDIA RTX
5060 Ti with 16 GB VRAM and one GPU. The server was configured with
`N_CONCURRENCY=8`, meaning at most eight requests per model microbatch. The
Locust load used `-u 40`, meaning 40 concurrent virtual users. It used the stored
`banking_es` classification endpoint:

```bash
uv run locust \
  -f stress/locustfile.py \
  --headless \
  -u 40 \
  -r 2 \
  -t 300s \
  -H http://127.0.0.1:8000
```

| Metric | Result |
|---|---:|
| Requests | 36,066 |
| Failures | 0 (0.00%) |
| Average response time | 315 ms |
| Median response time | 320 ms |
| Minimum response time | 34 ms |
| Maximum response time | 703 ms |
| Throughput | 120.28 requests/s |
| Failures/s | 0.00 |
| Locust virtual users | 40 |
| Maximum model microbatch | 8 requests |
| FastAPI workers | 1 |
| GPUs | 1 × RTX 5060 Ti 16 GB |

Approximate response-time percentiles reported by Locust:

| Percentile | Response time |
|---:|---:|
| p50 | 320 ms |
| p66 | 330 ms |
| p75 | 330 ms |
| p80 | 330 ms |
| p90 | 330 ms |
| p95 | 340 ms |
| p98 | 350 ms |
| p99 | 410 ms |
| p99.9 | 480 ms |
| p99.99 | 630 ms |
| p100 | 700 ms |

This is client-observed HTTP latency, including queueing, batching, model
execution, and response handling. It should not be compared directly with the
Prometheus batch-duration histogram, which measures only model execution per
batch. The run sustained approximately 120 requests/s with no failed requests.

To test client-provided classes and descriptions instead:

```bash
SERVER_MODE=adhoc uv run locust \
  -f stress/locustfile.py \
  --headless -u 16 -r 16 -t 30s \
  -H http://127.0.0.1:8000
```

To test extraction:

```bash
LOCUST_USAGE_TYPE=extraction uv run locust \
  -f stress/locustfile.py \
  --headless -u 16 -r 16 -t 30s \
  -H http://127.0.0.1:8000
```

## Prove batching happened

Inspect the response metadata from any request:

```json
{
  "meta": {
    "batch_size": 8,
    "queue_wait_ms": 3.1,
    "inference_ms": 7.8,
    "device": "cuda"
  }
}
```

The authoritative server counters are exposed here:

```bash
curl -s http://127.0.0.1:8000/metrics | grep gliner_inference
```

The most important series are:

```text
gliner_inference_batches_total{
  operation="classification",
  device="cuda",
  batch_size="8",
  status="success"
}
```

This counter increments once for every actual model call containing eight
requests. The complete metric catalog is:

| Metric | Type | Labels | What it proves or measures |
|---|---|---|---|
| `gliner_inference_batches_total` | Counter | `operation`, `device`, `batch_size`, `status` | Actual model calls; `status` is `success` or `error` |
| `gliner_inference_batch_inputs_total` | Counter | `operation`, `device`, `status` | Requests processed by model calls |
| `gliner_inference_batch_size` | Histogram | `operation`, `device` | Distribution of requests per model call |
| `gliner_inference_batch_duration_seconds` | Histogram | `operation`, `device`, `batch_size` | Wall-clock model duration per batch, filterable by batch size |
| `gliner_inference_queue_wait_seconds` | Histogram | `operation`, `device` | Request wait time before inference |
| `gliner_inference_queue_depth` | Gauge | none | Current bounded queue depth |
| `gliner_inference_queue_rejected_total` | Counter | `operation` | HTTP 429 responses due to a full queue |
| `gliner_inference_request_timeouts_total` | Counter | `operation` | HTTP 504 responses after request timeout |
| `gliner_inference_inflight_batches` | Gauge | `operation`, `device` | Currently executing model batches |
| `gliner_inference_last_batch_size` | Gauge | `operation`, `device` | Most recently started batch size |

The application disables `_created` helper series. Histograms still expose the
standard `_bucket`, `_sum`, and `_count` series.

If the test is really batching, `batch_size="8"` should increase and the
histogram should show values above 1. If every request executes alone, only
`batch_size="1"` will increase and `gliner_inference_batch_size_sum` will be
close to `gliner_inference_batch_size_count`.

## Prometheus scrape configuration

For a Prometheus server scraping the API pod:

```yaml
scrape_configs:
  - job_name: gliner-api
    metrics_path: /metrics
    static_configs:
      - targets: ["gliner-api:8000"]
```

Useful PromQL:

```promql
sum by (batch_size) (rate(gliner_inference_batches_total{status="success"}[1m]))
```

```promql
sum(rate(gliner_inference_batch_inputs_total{status="success"}[1m]))
/
sum(rate(gliner_inference_batches_total{status="success"}[1m]))
```

The second query estimates the average number of requests per model batch.

Filter batch duration for one exact batch size:

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

Compare p95 batch duration across batch sizes:

```promql
histogram_quantile(
  0.95,
  sum by (batch_size, le) (
    rate(gliner_inference_batch_duration_seconds_bucket[5m])
  )
)
```

The duration metric now includes `batch_size` as a label, so every bucket, sum,
and count can be filtered by the exact batch size.

## Locust environment variables

These variables configure `stress/locustfile.py`:

| Variable | Default | Meaning |
|---|---|---|
| `LOCUST_APP_ID` | `banking_es` | Stored app ID used by the test |
| `LOCUST_USAGE_TYPE` | `classification` | Stored usage type: `classification` or `extraction` |
| `SERVER_MODE` | `stored` | `stored` uses an app schema; `adhoc` sends classes/descriptions |
| `LOCUST_WAIT_MIN` | `0` | Minimum pause between requests from one Locust user, seconds |
| `LOCUST_WAIT_MAX` | `0` | Maximum pause between requests from one Locust user, seconds |

For example, to reduce pressure while retaining multiple users:

```bash
LOCUST_WAIT_MIN=0.01 LOCUST_WAIT_MAX=0.02 \
  uv run locust -f stress/locustfile.py --headless -u 16 -r 16 -t 30s \
  -H http://127.0.0.1:8000
```

## Tuning notes

- `N_CONCURRENCY` is the maximum model batch size, not the number of model
  worker threads. This implementation deliberately has one serial model worker
  per process/GPU.
- `BATCH_WINDOW_MS` adds intentional queueing delay to collect compatible
  requests.
- Requests with different ad-hoc schemas are not compatible and will be
  counted as separate batches.
- Run one server process per GPU. Multiple Uvicorn workers would each load and
  compile a separate model.
