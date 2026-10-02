"""Banking intent classification with fastino/gliner2-base-v1 loaded from temp/.

Measures end-to-end inference latency (per example, wall clock around
classify_text) plus aggregate metrics: percentiles, throughput, tokens/s,
model load time, and per-intent latency.

Usage:
    uv run python src/classifier/classify.py                        # EN dataset, base model (default)
    uv run python src/classifier/classify.py --model multi          # multilingual model
    uv run python src/classifier/classify.py --data data/banking_intents_es.jsonl --model multi
    uv run python src/classifier/classify.py --data data/banking_intents_es.jsonl --model multi --lang es
    uv run python src/classifier/classify.py --limit 20             # quick smoke test
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from gliner2 import AutoExtractor, GLiNER2

TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "banking_intents.jsonl"

# name -> (repo id, local dir, loader class)
MODELS = {
    "base": ("fastino/gliner2-base-v1", TEMP_DIR / "gliner2-base-v1", GLiNER2),
    "multi": ("fastino/gliner2.5-multi-v1", TEMP_DIR / "gliner2.5-multi-v1", AutoExtractor),
}

INTENTS = [
    "check_balance",
    "transfer_money",
    "pay_bill",
    "report_lost_card",
    "report_fraud",
    "apply_for_loan",
    "apply_for_credit_card",
    "close_account",
    "open_account",
    "reset_password",
]

# Optional label descriptions (label -> description), injected into the model prompt.
# Keep them short and discriminative, in the same language as the input text.
INTENT_DESCRIPTIONS = {
    "en": {
        "check_balance": "Check or inquire about the balance or available funds in an account",
        "transfer_money": "Move or send money from one account to another account or person",
        "pay_bill": "Pay a bill or invoice for a service (electricity, water, phone, rent, etc.)",
        "report_lost_card": "Report a lost or misplaced card and request to block or cancel it",
        "report_fraud": "Report an unauthorized transaction, suspicious charge, or fraud on a card or account",
        "apply_for_loan": "Apply for or request a loan or credit (personal, auto, mortgage, etc.)",
        "apply_for_credit_card": "Apply for, request, or open a new credit card",
        "close_account": "Close, cancel, or terminate a bank account",
        "open_account": "Open or start a new bank account",
        "reset_password": "Reset, recover, or change the online banking or app password",
    },
    "es": {
        "check_balance": "Consultar el saldo o el dinero disponible en una cuenta",
        "transfer_money": "Mover o enviar dinero de una cuenta a otra cuenta o a otra persona",
        "pay_bill": "Pagar una factura o recibo de un servicio (luz, agua, teléfono, renta, etc.)",
        "report_lost_card": "Reportar que una tarjeta se perdió o se extravió y solicitar bloquearla o cancelarla",
        "report_fraud": "Reportar una transacción no autorizada, un cargo sospechoso o fraude en una tarjeta o cuenta",
        "apply_for_loan": "Solicitar o pedir un préstamo o crédito (personal, automotriz, hipoteca, etc.)",
        "apply_for_credit_card": "Solicitar, pedir o abrir una tarjeta de crédito nueva",
        "close_account": "Cerrar, cancelar o dar de baja una cuenta bancaria",
        "open_account": "Abrir o iniciar una cuenta bancaria nueva",
        "reset_password": "Restablecer, recuperar o cambiar la contraseña de la banca en línea o la app",
    },
}


def load_model(name: str) -> tuple[object, float, str]:
    repo, model_dir, loader = MODELS[name]
    if not model_dir.exists():
        raise SystemExit(
            f"Model not found at {model_dir}. Run: uv run python src/classifier/download_model.py {name}"
        )
    print(f"Loading {repo} from {model_dir} ...")
    t0 = time.perf_counter()
    model = loader.from_pretrained(str(model_dir))
    load_s = time.perf_counter() - t0
    print(f"Model loaded in {load_s:.2f}s")
    return model, load_s, repo


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method)."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def latency_stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    n = len(vals)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return {
        "count": n,
        "min_ms": round(s[0] * 1000, 2),
        "max_ms": round(s[-1] * 1000, 2),
        "mean_ms": round(mean * 1000, 2),
        "median_ms": round(percentile(s, 50) * 1000, 2),
        "p90_ms": round(percentile(s, 90) * 1000, 2),
        "p95_ms": round(percentile(s, 95) * 1000, 2),
        "p99_ms": round(percentile(s, 99) * 1000, 2),
        "stddev_ms": round(math.sqrt(var) * 1000, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Dataset JSONL path")
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="base",
        help="base=fastino/gliner2-base-v1 (EN span) | multi=fastino/gliner2.5-multi-v1 (multilingual)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "es"],
        default=None,
        help="Inject label descriptions in this language into the model prompt (default: no descriptions)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (first call traces the graph; a warmup call is excluded from timing)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only classify the first N examples")
    args = parser.parse_args()

    # Plain label list, or {label: description} dict when --lang is set.
    tasks = {"intent": INTENTS}
    if args.lang:
        tasks = {"intent": {label: INTENT_DESCRIPTIONS[args.lang][label] for label in INTENTS}}

    model, load_s, repo = load_model(args.model)

    if args.compile:
        print("Enabling torch.compile ...")
        model.compile()
        t0 = time.perf_counter()
        model.classify_text("warmup", tasks)  # tracing happens here, excluded from timing
        print(f"Warmup (tracing) took {time.perf_counter() - t0:.1f}s")

    examples = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"Classifying {len(examples)} examples with {len(INTENTS)} intents ...")

    latencies: list[float] = []          # seconds, per example
    tokens_total = 0
    correct = 0
    per_intent_correct: dict[str, int] = Counter()
    per_intent_total: dict[str, int] = Counter()
    per_intent_latency: dict[str, list[float]] = {i: [] for i in INTENTS}

    dataset_stem = args.data.stem
    results_path = args.data.parent / f"predictions_{dataset_stem}.jsonl"
    metrics_path = args.data.parent / f"metrics_{dataset_stem}.json"

    infer_start = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as out:
        for i, ex in enumerate(examples, start=1):
            t0 = time.perf_counter()
            result = model.classify_text(ex["text"], tasks, include_confidence=True)
            dt = time.perf_counter() - t0
            latencies.append(dt)
            tokens_total += len(ex["text"].split())

            pred = result["intent"]
            confidence = None
            if isinstance(pred, dict):
                confidence = pred.get("confidence")
                pred = pred.get("label")
            ok = pred == ex["intent"]
            correct += ok
            per_intent_total[ex["intent"]] += 1
            per_intent_correct[ex["intent"]] += ok
            per_intent_latency[ex["intent"]].append(dt)

            out.write(json.dumps(
                {**ex, "predicted": pred, "confidence": confidence, "latency_ms": round(dt * 1000, 2)},
                ensure_ascii=False,
            ) + "\n")
            if i % 100 == 0 or i == len(examples):
                print(f"  {i}/{len(examples)}  (running accuracy: {correct / i:.3f}, "
                      f"avg latency: {sum(latencies) / len(latencies) * 1000:.1f} ms)")
    total_infer_s = time.perf_counter() - infer_start

    n = len(examples)
    accuracy = correct / n
    stats = latency_stats(latencies)
    metrics = {
        "dataset": str(args.data),
        "model": repo,
        "label_descriptions": args.lang,
        "compiled": args.compile,
        "model_load_seconds": round(load_s, 2),
        "n_examples": n,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "inference": {
            **stats,
            "total_seconds": round(total_infer_s, 2),
            "throughput_examples_per_sec": round(n / total_infer_s, 2),
            "tokens_per_sec": round(tokens_total / total_infer_s, 1),
        },
        "per_intent": {
            intent: {
                "n": per_intent_total[intent],
                "accuracy": round(per_intent_correct[intent] / per_intent_total[intent], 4),
                "latency": latency_stats(per_intent_latency[intent]),
            }
            for intent in INTENTS
            if per_intent_total[intent]
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nOverall accuracy: {accuracy:.4f} ({correct}/{n})")
    print("\nPer-intent accuracy:")
    for intent in INTENTS:
        if per_intent_total[intent]:
            acc = per_intent_correct[intent] / per_intent_total[intent]
            print(f"  {intent:22s} {acc:.3f}  (n={per_intent_total[intent]})")

    inf = metrics["inference"]
    print("\nInference latency (end-to-end, per example, wall clock):")
    print(f"  min    : {inf['min_ms']:>10.2f} ms")
    print(f"  mean   : {inf['mean_ms']:>10.2f} ms")
    print(f"  median : {inf['median_ms']:>10.2f} ms")
    print(f"  p90    : {inf['p90_ms']:>10.2f} ms")
    print(f"  p95    : {inf['p95_ms']:>10.2f} ms")
    print(f"  p99    : {inf['p99_ms']:>10.2f} ms")
    print(f"  max    : {inf['max_ms']:>10.2f} ms")
    print(f"  stddev : {inf['stddev_ms']:>10.2f} ms")
    print(f"  total  : {inf['total_seconds']} s for {n} examples")
    print(f"  throughput: {inf['throughput_examples_per_sec']} examples/s, {inf['tokens_per_sec']} tokens/s")
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Predictions saved to {results_path}")


if __name__ == "__main__":
    main()
