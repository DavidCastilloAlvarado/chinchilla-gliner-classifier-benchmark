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
├── temp/
│   ├── gliner2-base-v1/          # base model weights (read from local disk)
│   ├── gliner2.5-multi-v1/       # general multilingual model weights
│   └── GLiNER2.5-multi-Decide/   # multilingual decision model weights
├── data/
│   ├── banking_intents.jsonl      # 1,000 English examples (10 intents × 100)
│   ├── banking_intents_es.jsonl   # 1,000 Spanish examples (10 intents × 100)
│   ├── predictions_*.jsonl        # predictions + confidence + per-example latency_ms
│   └── metrics_*.json             # latency percentiles, throughput, accuracy
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
Measured on this CPU: **not worth it** — 65.7 s one-time tracing cost and steady-state
latency went *up* (234.6 ms → 294.4 ms mean on ES). Keep it off for CPU; it is
meant to pay off on GPU or large batches.

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
