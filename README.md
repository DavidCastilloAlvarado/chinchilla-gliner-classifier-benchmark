# Banking Intent Classifier (GLiNER2)

Zero-shot banking intent classification with GLiNER2 checkpoints (CPU-friendly),
managed with **uv**. Three models selectable at benchmark time:

| `--model` | Checkpoint | Size / architecture | Language | Focus |
|---|---|---|---|---|
| `base` (default) | `fastino/gliner2-base-v1` | 205M, span | English | General extraction/classification |
| `multi` | `fastino/gliner2.5-multi-v1` | 287M, boundary | Multilingual | General extraction/classification |
| `decide` | `fastino/GLiNER2.5-multi-Decide` | 287M, decision | Multilingual | Intent/routing/operational decisions |

## Layout

```
classifier/
├── pyproject.toml            # uv project (gliner2[local], huggingface-hub)
├── .env.example              # FastAPI configuration template
├── temp/
│   ├── gliner2-base-v1/          # base model weights (read from local disk)
│   ├── gliner2.5-multi-v1/       # general multilingual model weights
│   └── GLiNER2.5-multi-Decide/   # multilingual decision model weights
├── data/
│   ├── banking_intents.jsonl      # 1,000 English examples (10 intents × 100)
│   ├── banking_intents_es.jsonl   # 1,000 Spanish examples (10 intents × 100)
│   ├── predictions_*.jsonl        # predictions + confidence + per-example latency_ms
│   └── metrics_*.json             # latency percentiles, throughput, accuracy
├── app/                            # independent FastAPI serving proof of concept
│   ├── main.py                     # FastAPI entrypoint
│   ├── api/                        # routes and Pydantic request/response models
│   ├── core/                       # lifespan, model, registry, dependencies, batching
│   ├── apps/                       # stored classification/extraction JSON schemas
│   └── README.md                   # server API and deployment documentation
├── stress/
│   ├── locustfile.py               # concurrent Locust load test
│   └── README.md                   # batching verification and PromQL queries
└── src/classifier/
    ├── download_model.py        # downloads the model into temp/
    ├── generate_examples.py     # generates the 1,000 English examples
    ├── generate_examples_es.py  # generates the 1,000 Spanish examples
    ├── classify.py              # loads model from temp/, classifies, measures latency
    └── review_errors.py         # lists misclassified examples + confusion pairs
```

## Setup

```bash
uv sync
```

For local FastAPI configuration:

```bash
cp .env.example .env
```

The server loads the project-root `.env` during startup when it exists. Existing
system environment variables take precedence; if `.env` is absent, system
environment variables are used directly. The real `.env` is ignored by Git.

## FastAPI serving proof of concept

The independent `app/` package serves the local GLiNER2 model with a modular
FastAPI architecture and a small dynamic micro-batcher. It supports the
TypeSafe/System One-compatible free endpoint at `POST /v1/systemone`, a
GLiNER2 entity extraction endpoint at `POST /v1/extraction`, stored app schemas
for classification/extraction, legacy GLiNER2-native ad-hoc classification,
CUDA model loading, optional `torch.compile`, and a configurable
10 ms batching window.

```bash
# CPU by default; use GPU_MODE=cuda for NVIDIA inference
MODEL_NAME=decide GPU_MODE=cuda COMPILE_MODEL=true \
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

See [`app/README.md`](app/README.md) for the endpoint contracts, app JSON
format, batching design, environment variables, and Kubernetes readiness guidance.

### Docker and Docker Compose

The Docker image installs dependencies during the image build but does not copy
application source or model weights into the image. Compose mounts the checkout at
`/workspace` and mounts the local `temp/` model directory read-only.

CPU mode is the default:

```bash
make build
make run
# or build and run in one command
make runbuild
```

Use the CUDA image and GPU profile with:

```bash
make build PROFILE=cuda
make run PROFILE=cuda
# or
make runbuild PROFILE=cuda
```

The Compose services define their native `healthcheck` against
`/health/live`, with a 300-second startup window for model loading, compilation,
and warmup. The CUDA profile uses the NVIDIA Container Toolkit and `gpus: all`.
Both profiles run Uvicorn with developer reload enabled and watch the mounted
`/workspace/app` directory, so Python source changes automatically restart the
server and repeat model warmup. Dependency changes still require an image rebuild.
The CPU and CUDA variants are separate Compose profiles, so only the selected
service starts. See [`docker-compose.yml`](docker-compose.yml), [`Dockerfile`](Dockerfile), and
[`Makefile`](Makefile) for the complete configuration.

The Locust stress test and Prometheus verification instructions are in
[`stress/README.md`](stress/README.md):

```bash
uv run locust -f stress/locustfile.py --headless -u 16 -r 16 -t 30s \
  -H http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/metrics | grep gliner_inference
```

### Observed GPU stress benchmark

A five-minute Locust run used 40 concurrent Locust virtual users (`-u 40`),
with one FastAPI worker and one NVIDIA RTX 5060 Ti with 16 GB VRAM. The server
used `N_CONCURRENCY=8`, meaning a maximum of eight requests per model
microbatch:

```bash
uv run locust -f stress/locustfile.py \
  --headless -u 40 -r 2 -t 300s \
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
| Locust virtual users | 40 |
| Maximum model microbatch | 8 requests |
| FastAPI workers | 1 |
| GPUs | 1 × RTX 5060 Ti 16 GB |

Approximate Locust response-time percentiles: p50 320 ms, p90 330 ms, p95
340 ms, p99 410 ms, p99.9 480 ms, p99.99 630 ms, and p100 700 ms. These are
complete HTTP response times, including queueing, batching, model execution, and
response handling; they are not the same as the Prometheus batch-duration metric.

### FastAPI environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_NAME` | `decide` | Local checkpoint: `base`, `multi`, or `decide` |
| `MODEL_DIR` | `temp/<model>` | Override the local checkpoint directory |
| `GPU_MODE` | `cpu` | Inference device: `cpu` or `cuda` |
| `COMPILE_MODEL` | `false` | Run `torch.compile` during startup; warmup always runs before readiness |
| `BATCH_WINDOW_MS` | `10` | Maximum time to collect compatible requests |
| `N_CONCURRENCY` | `8` | Maximum requests in one model batch |
| `MAX_QUEUE_SIZE` | `256` | Maximum queued requests before HTTP 429 |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Maximum wait for a result before HTTP 504 |
| `APPS_DIR` | `app/apps` | Directory containing stored app JSON schemas |

These variables can be placed in `.env` or provided by the system/container
environment. System environment values take precedence over `.env`. `HOST` and
`PORT` are Uvicorn command-line options rather than application settings.

The queue is bounded. A full queue returns `429` with `Retry-After: 1`; a model
failure returns `503`; and a request timeout returns `504`. A hard hang inside
native PyTorch/CUDA code requires process-level supervision and restart.

### Prometheus metrics

`GET /metrics` exposes the following metrics:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `gliner_inference_batches_total` | Counter | `operation`, `device`, `batch_size`, `status` | Actual model calls; `status` is `success` or `error` |
| `gliner_inference_batch_inputs_total` | Counter | `operation`, `device`, `status` | Requests processed by model calls |
| `gliner_inference_batch_size` | Histogram | `operation`, `device` | Requests per model call |
| `gliner_inference_batch_duration_seconds` | Histogram | `operation`, `device`, `batch_size` | Model duration per batch, filterable by batch size |
| `gliner_inference_queue_wait_seconds` | Histogram | `operation`, `device` | Time waiting before inference |
| `gliner_inference_queue_depth` | Gauge | none | Current bounded queue depth |
| `gliner_inference_queue_rejected_total` | Counter | `operation` | HTTP 429 queue rejections |
| `gliner_inference_request_timeouts_total` | Counter | `operation` | HTTP 504 request timeouts |
| `gliner_inference_inflight_batches` | Gauge | `operation`, `device` | Currently executing batches |
| `gliner_inference_last_batch_size` | Gauge | `operation`, `device` | Most recently started batch size |

The primary proof that batching is active is a growing series such as:

```text
gliner_inference_batches_total{operation="classification",device="cuda",batch_size="8",status="success"}
```

Batch duration can be filtered by exact batch size:

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

Histogram metrics also expose the standard `_bucket`, `_sum`, and `_count`
series. The application disables the generated `_created` helper series. See
[`app/README.md`](app/README.md) and
[`stress/README.md`](stress/README.md) for full endpoint, PromQL, and Locust
documentation.

### Locust stress-test variables

These configure [`stress/locustfile.py`](stress/locustfile.py):

| Variable | Default | Meaning |
|---|---|---|
| `LOCUST_APP_ID` | `banking_es` | Stored app schema ID |
| `LOCUST_USAGE_TYPE` | `classification` | `classification` or `extraction` |
| `SERVER_MODE` | `stored` | `stored` uses an app schema; `adhoc` sends labels/descriptions |
| `LOCUST_WAIT_MIN` | `0` | Minimum per-user pause, seconds |
| `LOCUST_WAIT_MAX` | `0` | Maximum per-user pause, seconds |

## FastAPI tests

The app tests are managed by `uv` and run without loading a model checkpoint:

```bash
uv run tests
```

The `tests` command is a project script backed by pytest. The equivalent direct
command is `uv run pytest -q`. The suite includes pytest validation tests and async `unittest` cases for routing,
backpressure, micro-batching, and startup warmup. To run only unittest cases:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

## 1. Download the models into `temp/`

```bash
uv run python src/classifier/download_model.py          # base (default)
uv run python src/classifier/download_model.py multi    # gliner2.5-multi-v1
uv run python src/classifier/download_model.py decide   # GLiNER2.5-multi-Decide
uv run python src/classifier/download_model.py all      # all checkpoints
```

Downloads via `huggingface_hub.snapshot_download` into `temp/<model-name>/`
(subsequent runs are cached).

## 2. Generate 1,000 banking intent examples

```bash
uv run python src/classifier/generate_examples.py    # English  -> data/banking_intents.jsonl
uv run python src/classifier/generate_examples_es.py # Spanish  -> data/banking_intents_es.jsonl
```

Each writes 1,000 unique examples, 100 per intent:

`check_balance`, `transfer_money`, `pay_bill`, `report_lost_card`, `report_fraud`,
`apply_for_loan`, `apply_for_credit_card`, `close_account`, `open_account`, `reset_password`

## 3. Classify (with latency metrics)

```bash
uv run python src/classifier/classify.py                                  # base on CPU (default)
uv run python src/classifier/classify.py --model multi                     # general multilingual model
uv run python src/classifier/classify.py --model decide                   # multilingual decision model
uv run python src/classifier/classify.py --model decide --gpu-mode cuda   # Decide on NVIDIA GPU
uv run python src/classifier/classify.py --data data/banking_intents_es.jsonl --model decide --gpu-mode cuda
uv run python src/classifier/classify.py --limit 20                       # quick smoke test
```

Loads the selected model from `temp/` (no Hub access at inference time) and runs
`model.classify_text(text, {"intent": [...]})` over the dataset. `--model base` uses
`GLiNER2.from_pretrained` (span checkpoint); `--model multi` and `--model decide`
use `AutoExtractor.from_pretrained` (boundary/decision checkpoints). The `decide`
checkpoint is specialized for intent, routing, triage, and other operational decisions.

`--gpu-mode` selects the inference device: `cpu` is the default; use
`--gpu-mode cuda` on a machine with a compatible NVIDIA driver. The script checks
`torch.cuda.is_available()` and fails with a clear message if CUDA is unavailable.
The selected device is recorded in each `metrics_*.json` file as `device`.
Only the NVIDIA driver is required at runtime; the CUDA Toolkit is not required.

`--lang {en,es}` injects a short description per intent into the model prompt
(label → description dict instead of a plain label list). This improves accuracy
on confusable intents (ES: 88.7% → 93.4%) at the cost of ~2.4x latency
(longer prompt). Default: no descriptions.

`--compile` enables `torch.compile` (a warmup/tracing call is excluded from timing).
Measured on the development CPU: **not worth it** — 65.7 s one-time tracing cost
and steady-state latency went up (234.6 ms → 294.4 ms mean on ES). On the RTX 5060 Ti,
`decide --compile` reduced mean latency from 11.44 ms to 7.75 ms and increased
throughput from 87.3 to 128.8 examples/s, but required a 44.8 s one-time warmup.
Keep it off for CPU; on GPU, enable it for long-running workloads where the warmup
can be amortized.

**Metrics** (wall-clock, end-to-end per example):
- Per-example `latency_ms` in each `predictions_*.jsonl` line
- `metrics_*.json`: min / mean / median / p90 / p95 / p99 / max / stddev,
  total inference time, throughput (examples/s and tokens/s), model load time,
  overall + per-intent accuracy and per-intent latency

## Benchmark results

All benchmark runs use 1,000 examples and measure wall-clock end-to-end latency per example.
CPU results below were measured on the development machine; the GPU run was measured
on an NVIDIA GeForce RTX 5060 Ti with 16 GB VRAM. Full numbers are written to
`data/metrics_*.json`.

### Model comparison

| | **ES** `base` | **ES** `multi` | **EN** `base` | **EN** `multi` |
|---|---|---|---|---|
| **Accuracy** | 77.1% | **88.7%** | **97.1%** | 91.6% |
| Mean latency | 276.6 ms | **234.6 ms** | 308.7 ms | **228.6 ms** |
| Median | 274.4 ms | **232.5 ms** | 261.7 ms | **226.2 ms** |
| p95 | 328.9 ms | **287.0 ms** | 582.1 ms | **263.3 ms** |
| p99 | 357.1 ms | **322.8 ms** | 885.2 ms | **293.3 ms** |
| Max | 538.6 ms | **404.3 ms** | 4,832.7 ms | **342.8 ms** |
| Throughput | 3.61 ex/s | **4.26 ex/s** | 3.24 ex/s | **4.37 ex/s** |
| Model load | 2.49 s | 5.82 s | 2.62 s | 5.79 s |

`multi` is the better all-rounder: +11.6 pts on Spanish (and fixes the
`apply_for_credit_card` collapse, 3% → 96%), ~15% faster, tighter tail latency.
Cost: −5.5 pts on English.

The `decide` model is integrated as an optional third checkpoint and has now been
benchmarked on the same Spanish dataset and GPU.

```bash
uv run python src/classifier/classify.py \
  --data data/banking_intents_es.jsonl \
  --model decide \
  --gpu-mode cuda
```

It uses `fastino/GLiNER2.5-multi-Decide`, the multilingual operational-decision
checkpoint for intent, routing, triage, and related tasks.

### GPU benchmark: RTX 5060 Ti 16 GB

Command used:

```bash
uv run python src/classifier/classify.py \
  --data data/banking_intents_es.jsonl \
  --model multi \
  --gpu-mode cuda
```

Configuration: `fastino/gliner2.5-multi-v1`, Spanish dataset, 1,000 examples,
no label descriptions, no `torch.compile`, single-example inference.

| Metric | RTX 5060 Ti 16 GB (`cuda`) | CPU `multi` reference |
|---|---:|---:|
| Accuracy | **88.7%** (887/1,000) | 88.7% (887/1,000) |
| Model load | **3.44 s** | 5.82 s |
| Min latency | 10.97 ms | 174.53 ms |
| Mean latency | **11.54 ms** | 234.62 ms |
| Median latency | **11.08 ms** | 232.46 ms |
| p90 | **11.15 ms** | 260.60 ms |
| p95 | **11.23 ms** | 286.96 ms |
| p99 | **11.67 ms** | 322.84 ms |
| Max latency | 225.81 ms | 404.29 ms |
| Stddev | 7.99 ms | 27.02 ms |
| Total inference | **11.56 s** | 234.68 s |
| Throughput (examples/s) | **86.53** | 4.26 |
| Throughput (tokens/s) | **940.0** | 46.3 |

Compared with the CPU reference, the RTX 5060 Ti provides approximately **20×
higher throughput** and **20× lower mean latency**, with the same accuracy.
The GPU p99 remains close to the normal per-request latency; the 225.81 ms
maximum is an isolated outlier, which explains why the mean and standard
percentiles differ.

Per-intent accuracy on the GPU was unchanged from the CPU `multi` run:
`check_balance` 0.68, `transfer_money` 1.00, `pay_bill` 0.74,
`report_lost_card` 0.65, `report_fraud` 0.85, `apply_for_loan` 1.00,
`apply_for_credit_card` 0.96, `close_account` 1.00, `open_account` 0.99,
and `reset_password` 1.00.

GPU runtime emitted non-fatal compatibility warnings for legacy tokenizer
metadata and an SDPA fallback to eager attention. The model loaded and completed
successfully using the standard CUDA path.

### GPU benchmark: `GLiNER2.5-multi-Decide`

Command used:

```bash
uv run python src/classifier/classify.py \
  --data data/banking_intents_es.jsonl \
  --model decide \
  --gpu-mode cuda
```

Configuration: `fastino/GLiNER2.5-multi-Decide`, Spanish dataset, 1,000 examples,
no label descriptions, no `torch.compile`, single-example inference, same RTX
5060 Ti 16 GB.

| Metric | `decide` on RTX 5060 Ti | `multi` on RTX 5060 Ti |
|---|---:|---:|
| **Accuracy** | **93.7%** (937/1,000) | 88.7% (887/1,000) |
| Model load | 3.44 s | 3.44 s |
| Min latency | 10.95 ms | 10.97 ms |
| Mean latency | **11.44 ms** | 11.54 ms |
| Median latency | **11.05 ms** | 11.08 ms |
| p90 | **11.12 ms** | 11.15 ms |
| p95 | **11.18 ms** | 11.23 ms |
| p99 | **11.63 ms** | 11.67 ms |
| Max latency | 225.38 ms | 225.81 ms |
| Stddev | **7.53 ms** | 7.99 ms |
| Total inference | **11.46 s** | 11.56 s |
| Throughput (examples/s) | **87.3** | 86.53 |
| Throughput (tokens/s) | **948.4** | 940.0 |

On this Spanish benchmark, `decide` improves accuracy by **5.0 percentage points**
(88.7% → 93.7%) while maintaining essentially the same GPU latency and
throughput. It also reduces errors from 113 to 63, a **44.2% reduction**.
The Decide model is therefore the preferred Fastino checkpoint for this banking
intent task, while `multi` remains the general-purpose multilingual model.

Per-intent accuracy for `decide`:
`check_balance` 0.81, `transfer_money` 1.00, `pay_bill` 1.00,
`report_lost_card` 0.71, `report_fraud` 1.00, `apply_for_loan` 0.93,
`apply_for_credit_card` 0.96, `close_account` 0.96, `open_account` 1.00,
and `reset_password` 1.00.

### Decide GPU error analysis

The Decide run produced **63/1,000 errors (6.3%)**. The most frequent confusion
pairs were:

| Gold intent | Predicted intent | Count |
|---|---|---:|
| `check_balance` | `transfer_money` | 18 |
| `report_lost_card` | `apply_for_credit_card` | 13 |
| `report_lost_card` | `reset_password` | 10 |
| `apply_for_loan` | `apply_for_credit_card` | 7 |
| `report_lost_card` | `close_account` | 6 |
| `apply_for_credit_card` | `open_account` | 4 |
| `close_account` | `reset_password` | 4 |
| `check_balance` | `open_account` | 1 |

The remaining errors are concentrated around card actions (`lost`, `credit card`,
`close account`) and account-balance language that is interpreted as a transfer.

### GPU benchmark: `decide` with `torch.compile`

Command used:

```bash
uv run python src/classifier/classify.py \
  --data data/banking_intents_es.jsonl \
  --compile \
  --model decide \
  --gpu-mode cuda
```

Configuration: `fastino/GLiNER2.5-multi-Decide`, RTX 5060 Ti 16 GB, Spanish dataset,
1,000 examples, no label descriptions, single-example inference. The compilation
warmup took **44.8 s** and was excluded from the per-example latency metrics.

| Metric | `decide` + `compile` | `decide` without compile |
|---|---:|---:|
| **Accuracy** | **93.7%** (937/1,000) | 93.7% (937/1,000) |
| Model load | 3.45 s | 3.44 s |
| Compile warmup | 44.8 s | — |
| Min latency | **7.58 ms** | 10.95 ms |
| Mean latency | **7.75 ms** | 11.44 ms |
| Median latency | **7.74 ms** | 11.05 ms |
| p90 | **7.83 ms** | 11.12 ms |
| p95 | **7.86 ms** | 11.18 ms |
| p99 | **7.97 ms** | 11.63 ms |
| Max latency | **8.74 ms** | 225.38 ms |
| Stddev | **0.08 ms** | 7.53 ms |
| Total inference | **7.76 s** | 11.46 s |
| Throughput (examples/s) | **128.8** | 87.3 |
| Throughput (tokens/s) | **1,399.3** | 948.4 |

`torch.compile` delivered a **32.3% lower mean latency** and **47.5% higher
throughput** with no accuracy change. Including the one-time 44.8 s compilation,
the run took approximately 52.6 s instead of 11.5 s. At this measured saving,
the warmup breaks even after roughly **12,000 examples**, so compilation is
recommended for long-running GPU services or large batches, not short one-off jobs.

Compilation emitted non-fatal Triton warnings about `max_autotune_gemm`, a C macro
redefinition, and deprecated TorchScript usage. The compiled benchmark completed
successfully.

### Per-intent accuracy (no descriptions)

| Intent | ES base | ES multi | EN base | EN multi |
|---|---|---|---|---|
| check_balance | 1.00 | 0.68 | 1.00 | 0.72 |
| transfer_money | 0.99 | 1.00 | 1.00 | 1.00 |
| pay_bill | 0.77 | 0.74 | 1.00 | 0.83 |
| report_lost_card | 0.94 | 0.65 | 1.00 | 0.82 |
| report_fraud | 0.76 | 0.85 | 1.00 | 0.83 |
| apply_for_loan | 0.79 | 1.00 | 0.93 | 1.00 |
| apply_for_credit_card | 0.03 | 0.96 | 0.86 | 0.96 |
| close_account | 0.80 | 1.00 | 0.92 | 1.00 |
| open_account | 0.84 | 0.99 | 1.00 | 1.00 |
| reset_password | 0.79 | 1.00 | 1.00 | 1.00 |

### Effect of label descriptions (`--lang es`, `multi` model, ES dataset)

| | no descriptions | with Spanish descriptions |
|---|---|---|
| **Accuracy** | 88.7% | **93.4%** (+4.7) |
| Mean latency | 234.6 ms | 557.8 ms (2.4×) |
| p95 / p99 | 287.0 / 322.8 ms | 682.0 / 974.2 ms |
| Throughput | 4.26 ex/s | 1.79 ex/s |

Per-intent (no-desc → with-desc): `check_balance` 0.68 → **1.00**, `pay_bill`
0.74 → 0.90, `report_lost_card` 0.65 → 0.77, `report_fraud` 0.85 → 0.93;
slight regressions on `transfer_money` (1.00 → 0.87), `apply_for_credit_card`
(0.96 → 0.92), `open_account` (0.99 → 0.95).

Top remaining confusions (ES, with descriptions):
`report_lost_card → close_account` (20×, "cancelar tarjeta" vs "cerrar cuenta"),
`transfer_money → check_balance` (13×), `pay_bill → check_balance` (9×).

## Reviewing misclassifications

```bash
uv run python src/classifier/review_errors.py                 # ES predictions (default)
uv run python src/classifier/review_errors.py --file data/predictions_banking_intents.jsonl
uv run python src/classifier/review_errors.py --intent report_lost_card
uv run python src/classifier/review_errors.py --top 5 --out errors.jsonl
```

Prints the confusion-pair summary plus example texts (id, gold, predicted,
confidence) per pair. Quick one-liner alternative:

```bash
jq -c 'select(.predicted != .intent)' data/predictions_banking_intents_es.jsonl
```

### GPU error analysis (RTX 5060 Ti)

The GPU run produced **113/1,000 errors (11.3%)**, consistent with its 88.7%
overall accuracy. The most frequent confusion pairs were:

| Gold intent | Predicted intent | Count |
|---|---|---:|
| `check_balance` | `transfer_money` | 32 |
| `report_lost_card` | `close_account` | 26 |
| `pay_bill` | `transfer_money` | 25 |
| `report_fraud` | `transfer_money` | 8 |
| `report_fraud` | `check_balance` | 7 |
| `report_lost_card` | `check_balance` | 5 |
| `apply_for_credit_card` | `open_account` | 4 |
| `report_lost_card` | `report_fraud` | 4 |

The main improvement opportunity is separating account actions that mention an
account or a card: balance queries are often interpreted as transfers, bill
payments as transfers, and card cancellation/blocking as account closure.

## Single prediction

```python
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("temp/gliner2-base-v1")
INTENTS = ["check_balance", "transfer_money", "pay_bill", "report_lost_card",
           "report_fraud", "apply_for_loan", "apply_for_credit_card",
           "close_account", "open_account", "reset_password"]

result = model.classify_text(
    "I want to transfer 250 dollars to my savings.",
    {"intent": INTENTS},
    include_confidence=True,
)
print(result)  # {'intent': {'label': 'transfer_money', 'confidence': 1.0}}
```
