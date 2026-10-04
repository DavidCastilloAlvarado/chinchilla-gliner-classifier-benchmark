"""Benchmark SystemOne on English and machine-translated Banking77 test data.

Run from the project root with::

    uv run python playground/benchmark_banking77.py

The script downloads the official Banking77 test CSV, creates a Spanish
translation while preserving labels and row order, then submits both versions
as choice questions to the existing SystemOne endpoint from the project .env.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_PATH = PROJECT_ROOT / "playground" / "BANKING77_BENCHMARK_REPORT.md"
METRICS_PATH = DATA_DIR / "banking77_benchmark_metrics.json"
TEST_CSV_PATH = DATA_DIR / "banking77_test.csv"
SPANISH_CSV_PATH = DATA_DIR / "banking77_test_es.csv"

BANKING77_TEST_CSV_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
DEFAULT_MODEL = "fastino/GLiNER2.5-multi-Decide"
QUESTION_ID = "intent"
QUESTION_INSTRUCTIONS = (
    "Choose the single best-matching banking intent for this customer-support "
    "utterance. Select exactly one option."
)
MARKER_START = "⟪B77_"
MARKER_END = "⟪END_B77_"


class BenchmarkError(RuntimeError):
    """Raised for invalid source data or an unusable benchmark response."""


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _http_get(url: str, *, timeout: float = 45.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "classifier-banking77-benchmark/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise BenchmarkError(f"{path.name} must contain a 'text' column")
        label_column = "category" if "category" in reader.fieldnames else "label"
        if label_column not in reader.fieldnames:
            raise BenchmarkError(f"{path.name} must contain a 'category' or 'label' column")
        return [
            {"text": (row.get("text") or "").strip(), "category": (row.get(label_column) or "").strip()}
            for row in reader
        ]


def validate_rows(rows: list[dict[str, str]], *, expected_count: int | None = None) -> list[str]:
    if not rows:
        raise BenchmarkError("The Banking77 test split is empty")
    if expected_count is not None and len(rows) != expected_count:
        raise BenchmarkError(
            f"Expected {expected_count} Banking77 test examples, got {len(rows)}"
        )
    if any(not row["text"] or not row["category"] for row in rows):
        raise BenchmarkError("Banking77 rows must have non-empty text and category values")
    labels = sorted({row["category"] for row in rows})
    if len(labels) != 77:
        raise BenchmarkError(f"Expected 77 Banking77 intents, found {len(labels)}")
    return labels


def download_test_csv(*, force: bool = False) -> tuple[list[dict[str, str]], str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not TEST_CSV_PATH.exists():
        payload = _http_get(BANKING77_TEST_CSV_URL)
        text = payload.decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        normalized = [
            {
                "text": (row.get("text") or "").strip(),
                "category": (row.get("category") or row.get("label") or "").strip(),
            }
            for row in rows
        ]
        validate_rows(normalized, expected_count=3080)
        atomic_write_text(TEST_CSV_PATH, text)
    rows = read_rows(TEST_CSV_PATH)
    validate_rows(rows, expected_count=3080)
    digest = hashlib.sha256(TEST_CSV_PATH.read_bytes()).hexdigest()
    return rows, digest


def translate_batch(
    items: list[tuple[int, str]],
    *,
    timeout: float = 45.0,
    attempts: int = 4,
) -> dict[int, str]:
    """Translate a marked batch so sentence segmentation cannot shift rows."""

    marked = "\n".join(
        f"{MARKER_START}{row_id:05d}⟫ {text} {MARKER_END}{row_id:05d}⟫"
        for row_id, text in items
    )
    query = urllib.parse.urlencode(
        {"client": "gtx", "sl": "en", "tl": "es", "dt": "t", "q": marked}
    )
    url = f"{TRANSLATE_URL}?{query}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Banking77 benchmark)"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                translated_payload = json.loads(response.read().decode("utf-8"))
            segments = translated_payload[0]
            translated = "".join(
                str(segment[0])
                for segment in segments
                if isinstance(segment, list) and segment and segment[0]
            )
            result: dict[int, str] = {}
            for row_id, _ in items:
                start_marker = f"{MARKER_START}{row_id:05d}⟫"
                end_marker = f"{MARKER_END}{row_id:05d}⟫"
                start = translated.find(start_marker)
                end = translated.find(end_marker, start + len(start_marker))
                if start < 0 or end < 0:
                    raise BenchmarkError("Translation response did not preserve row markers")
                value = translated[start + len(start_marker) : end].strip()
                if not value:
                    raise BenchmarkError("Translation response contained an empty row")
                result[row_id] = value
            return result
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, IndexError, BenchmarkError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2.0**attempt, 8.0))
    raise BenchmarkError(
        f"Could not translate a batch of {len(items)} examples after {attempts} attempts"
    ) from last_error


def translate_rows(
    rows: list[dict[str, str]],
    *,
    batch_size: int = 16,
    workers: int = 3,
    timeout: float = 45.0,
) -> list[dict[str, str]]:
    translated: list[str | None] = [None] * len(rows)
    batches = [
        [(index, rows[index]["text"]) for index in range(start, min(start + batch_size, len(rows)))]
        for start in range(0, len(rows), batch_size)
    ]
    print(
        f"Translating {len(rows):,} utterances into Spanish "
        f"({len(batches)} marked batches, {workers} workers)...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_batch = {
            executor.submit(translate_batch, batch, timeout=timeout): batch
            for batch in batches
        }
        for completed, future in enumerate(as_completed(future_to_batch), start=1):
            result = future.result()
            for row_id, value in result.items():
                translated[row_id] = value
            if completed % 20 == 0 or completed == len(batches):
                print(f"  translated batches: {completed}/{len(batches)}", flush=True)
    if any(value is None for value in translated):
        raise BenchmarkError("Spanish translation is incomplete")
    return [
        {"text": str(translated[index]), "category": row["category"]}
        for index, row in enumerate(rows)
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["text", "category"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, output.getvalue())


def get_or_create_spanish_csv(
    english_rows: list[dict[str, str]],
    *,
    force: bool = False,
    batch_size: int = 16,
    workers: int = 3,
) -> tuple[list[dict[str, str]], str]:
    if force or not SPANISH_CSV_PATH.exists():
        spanish_rows = translate_rows(
            english_rows,
            batch_size=batch_size,
            workers=workers,
        )
        write_csv(SPANISH_CSV_PATH, spanish_rows)
    spanish_rows = read_rows(SPANISH_CSV_PATH)
    validate_rows(spanish_rows, expected_count=len(english_rows))
    if [row["category"] for row in spanish_rows] != [row["category"] for row in english_rows]:
        raise BenchmarkError("Spanish CSV labels/order do not match the English test split")
    digest = hashlib.sha256(SPANISH_CSV_PATH.read_bytes()).hexdigest()
    return spanish_rows, digest


def _category_description(category: str) -> str:
    readable = category.replace("_", " ").replace("?", "").strip()
    return f"The intent is {readable}."


def _parse_choice(response: Any) -> tuple[str, float | None]:
    result = response.get("result", response) if isinstance(response, dict) else {}
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    answer = answers.get(QUESTION_ID, {}) if isinstance(answers, dict) else {}
    choice = answer.get("choice") if isinstance(answer, dict) else None
    if choice is None:
        raise BenchmarkError("SystemOne response has no intent choice")
    predicted = str(choice).strip()
    confidence: float | None = None
    if isinstance(answer, dict):
        value = answer.get("confidence")
        if value is None and isinstance(answer.get("probabilities"), dict):
            value = answer["probabilities"].get(predicted)
        try:
            confidence = float(value) if value is not None else None
        except (TypeError, ValueError):
            confidence = None
    return predicted, confidence


def request_prediction(
    index: int,
    row: dict[str, str],
    language: str,
    endpoint: str,
    model: str,
    labels: list[str],
    timeout: float,
    attempts: int = 4,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "state": row["text"],
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_INSTRUCTIONS,
                # The same canonical English class names and prompt are used for
                # both datasets to isolate the language of the utterance.
                "criteria": {label: _category_description(label) for label in labels},
            }
        },
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            predicted, confidence = _parse_choice(response_data)
            return {
                "row_id": index,
                "language": language,
                "gold": row["category"],
                "predicted": predicted,
                "correct": predicted == row["category"],
                "confidence": confidence,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt + 1 >= attempts:
                detail = exc.read(500).decode("utf-8", "replace").replace("\n", " ")
                error_message = f"HTTP {exc.code}: {detail[:250]}"
                print(
                    f"ERROR {language.upper()} row {index + 1}: {error_message}",
                    flush=True,
                )
                return {
                    "row_id": index,
                    "language": language,
                    "gold": row["category"],
                    "predicted": None,
                    "correct": False,
                    "confidence": None,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": error_message,
                }
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(0.5 * (2**attempt), 4.0)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, BenchmarkError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(0.5 * (2**attempt), 4.0))
    error_message = f"{type(last_error).__name__}: {str(last_error)[:250]}"
    print(
        f"ERROR {language.upper()} row {index + 1}: {error_message}",
        flush=True,
    )
    return {
        "row_id": index,
        "language": language,
        "gold": row["category"],
        "predicted": None,
        "correct": False,
        "confidence": None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "error": error_message,
    }


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percent * len(ordered)) - 1))
    return round(ordered[index], 3)


def summarize_predictions(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]], elapsed_seconds: float
) -> dict[str, Any]:
    labels = sorted({row["category"] for row in rows})
    supports = Counter(row["category"] for row in rows)
    true_positives: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    for prediction in predictions:
        gold = prediction["gold"]
        predicted = prediction["predicted"]
        if predicted is not None:
            predicted_counts[predicted] += 1
            if predicted == gold:
                true_positives[gold] += 1
                per_class_correct[gold] += 1
            elif predicted in labels:
                confusion[(gold, predicted)] += 1

    per_intent = []
    for label in labels:
        support = supports[label]
        tp = true_positives[label]
        predicted_count = predicted_counts[label]
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_intent.append(
            {
                "intent": label,
                "support": support,
                "correct": per_class_correct[label],
                "accuracy": round(per_class_correct[label] / support, 6) if support else 0.0,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    per_intent.sort(key=lambda metric: (-metric["f1"], -metric["accuracy"], metric["intent"]))

    total = len(rows)
    correct = sum(prediction["correct"] for prediction in predictions)
    latencies = [float(prediction["latency_ms"]) for prediction in predictions]
    errors = [prediction for prediction in predictions if prediction["error"]]
    return {
        "examples": total,
        "correct": correct,
        "incorrect": total - correct,
        "request_failures": len(errors),
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_precision": round(sum(item["precision"] for item in per_intent) / len(labels), 6),
        "macro_recall": round(sum(item["recall"] for item in per_intent) / len(labels), 6),
        "macro_f1": round(sum(item["f1"] for item in per_intent) / len(labels), 6),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_examples_per_second": round(total / elapsed_seconds, 3) if elapsed_seconds > 0 else None,
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "median": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "per_intent_rankings": per_intent,
        "top_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(15)
        ],
        "errors": [
            {"row_id": item["row_id"], "message": item["error"]}
            for item in errors[:20]
        ],
    }


def run_language(
    language: str,
    rows: list[dict[str, str]],
    labels: list[str],
    endpoint: str,
    model: str,
    *,
    workers: int,
    timeout: float,
    limit: int | None,
) -> dict[str, Any]:
    selected_rows = rows[:limit] if limit is not None else rows
    output_path = DATA_DIR / f"banking77_predictions_{language}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(
        f"Benchmarking {language.upper()}: {len(selected_rows):,} examples, "
        f"{len(labels)} intents, {workers} request workers...",
        flush=True,
    )
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    request_prediction,
                    index,
                    row,
                    language,
                    endpoint,
                    model,
                    labels,
                    timeout,
                ): index
                for index, row in enumerate(selected_rows)
            }
            for completed, future in enumerate(as_completed(future_to_index), start=1):
                prediction = future.result()
                predictions.append(prediction)
                output.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
                if completed % 100 == 0 or completed == len(selected_rows):
                    output.flush()
                    print(f"  {language.upper()} completed: {completed}/{len(selected_rows)}", flush=True)
    elapsed = time.perf_counter() - started
    # Keep metrics/predictions aligned with the source split, regardless of completion order.
    predictions.sort(key=lambda prediction: prediction["row_id"])
    # The on-disk checkpoint is written as futures complete; normalize final
    # output ordering so prediction row N always corresponds to CSV row N.
    atomic_write_text(
        output_path,
        "".join(
            json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n"
            for prediction in predictions
        ),
    )
    summary = summarize_predictions(selected_rows, predictions, elapsed)
    summary["predictions_file"] = str(output_path.relative_to(PROJECT_ROOT))
    return summary


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(metrics: dict[str, Any]) -> str:
    results = metrics["results"]
    ranked_languages = sorted(
        results.items(),
        key=lambda item: (-item[1]["macro_f1"], -item[1]["accuracy"], item[0]),
    )
    language_rows = []
    previous_score: tuple[float, float] | None = None
    current_rank = 0
    for position, (language, result) in enumerate(ranked_languages, start=1):
        score = (result["macro_f1"], result["accuracy"])
        if score != previous_score:
            current_rank = position
            previous_score = score
        language_rows.append(
            [
                current_rank,
                language.upper(),
                f"{result['accuracy']:.2%}",
                f"{result['macro_f1']:.2%}",
                f"{result['latency_ms']['mean']:.1f} ms",
                f"{result['latency_ms']['p95']:.1f} ms",
                f"{result['throughput_examples_per_second']:.2f}",
                result["request_failures"],
            ]
        )
    top_score = (ranked_languages[0][1]["macro_f1"], ranked_languages[0][1]["accuracy"])
    leaders = [
        language.upper()
        for language, result in ranked_languages
        if (result["macro_f1"], result["accuracy"]) == top_score
    ]
    if len(leaders) > 1:
        ranking_conclusion = f"**Top-ranked variants (tied):** {', '.join(leaders)}."
    else:
        leader = ranked_languages[0][0]
        ranking_conclusion = (
            f"**Higher-ranked variant:** {leader.upper()} "
            f"(macro-F1 {results[leader]['macro_f1']:.2%}; "
            f"accuracy {results[leader]['accuracy']:.2%})."
        )
    evaluated_counts = sorted({result["examples"] for result in results.values()})
    evaluated_note = (
        f"This run evaluated the first {evaluated_counts[0]:,} examples per language (a limited run)."
        if metrics["configuration"].get("limit") is not None
        else f"This run evaluated all {metrics['dataset']['examples']:,} test examples per language."
    )
    report = [
        "# Banking77 bilingual SystemOne benchmark report",
        "",
        f"Run timestamp (UTC): `{metrics['run_timestamp_utc']}`",
        "",
        "## Result ranking",
        "",
        "Language variants are ranked by macro-F1, then accuracy. The test split "
        "contains 3,080 utterances and 77 intents. " + evaluated_note + " Spanish rows are translations "
        "of the corresponding English rows with labels and order preserved.",
        "",
        markdown_table(
            ["Rank", "Test language", "Accuracy", "Macro-F1", "Mean latency", "P95 latency", "Examples/s", "Failed requests"],
            language_rows,
        ),
        "",
        ranking_conclusion,
        "",
        "## Per-language details",
        "",
        "| Language | Examples | Correct | Accuracy | Macro precision | Macro recall | Macro-F1 | Mean / median latency | P90 / P95 / P99 latency | Throughput | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for language in ("en", "es"):
        result = results.get(language)
        if not result:
            continue
        latency = result["latency_ms"]
        report.append(
            f"| {language.upper()} | {result['examples']:,} | {result['correct']:,} | "
            f"{result['accuracy']:.2%} | {result['macro_precision']:.2%} | "
            f"{result['macro_recall']:.2%} | {result['macro_f1']:.2%} | "
            f"{latency['mean']:.1f} / {latency['median']:.1f} ms | "
            f"{latency['p90']:.1f} / {latency['p95']:.1f} / {latency['p99']:.1f} ms | "
            f"{result['throughput_examples_per_second']:.2f} ex/s | {result['elapsed_seconds']:.1f} s |"
        )
    report.extend(
        [
            "",
            "## Benchmark setup",
            "",
            f"- Dataset: [PolyAI/Banking77 on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77), test split; 3,080 utterances and 77 intents.",
            f"- English CSV: [`data/banking77_test.csv`](../data/banking77_test.csv) (SHA-256 `{metrics['dataset']['english_sha256']}`).",
            f"- Spanish CSV: [`data/banking77_test_es.csv`](../data/banking77_test_es.csv) (SHA-256 `{metrics['dataset']['spanish_sha256']}`).",
            f"- Model requested: `{metrics['model']}`. The endpoint URL is loaded from the project `.env`; it is intentionally not copied into this report.",
            f"- Requests: one SystemOne `choice` request per row; {metrics['configuration']['workers']} concurrent workers, timeout {metrics['configuration']['timeout_seconds']:.1f}s, up to four attempts for transient failures.",
            "- Both languages use identical English instructions and canonical English intent option keys/descriptions. Only the utterance language changes, so this evaluates cross-lingual robustness under the same label prompt.",
            "- Latency is client-observed HTTP round-trip time, including endpoint queueing and response handling; throughput is completed examples divided by wall-clock benchmark time.",
            "- Spanish data was machine-translated from the English test text using Google Translate's public web translation endpoint. Labels and row order were preserved. Translation artifacts or ambiguity can affect Spanish scores; the Spanish scores are not a native-human-translation result.",
            "- Accuracy counts failed or invalid choices as incorrect. Macro-F1 gives equal weight to each of the 77 intents. Per-intent rank is descending F1, with accuracy used as a tiebreaker.",
            "",
            "## Intent rankings by language",
            "",
            "The full per-intent leaderboard is shown below for each language. `Accuracy` is class recall in this single-label task; `F1` incorporates both false positives and false negatives.",
            "",
        ]
    )
    for language in ("en", "es"):
        result = results.get(language)
        if not result:
            continue
        intent_rows = [
            [rank, item["intent"], f"{item['f1']:.2%}", f"{item['accuracy']:.2%}", item["support"]]
            for rank, item in enumerate(result["per_intent_rankings"], start=1)
        ]
        report.extend(
            [
                f"### {language.upper()} intent ranking",
                "",
                markdown_table(["Rank", "Intent", "F1", "Accuracy", "Support"], intent_rows),
                "",
            ]
        )
        confusions = result["top_confusions"][:10]
        if confusions:
            confusion_rows = [[item["gold"], item["predicted"], item["count"]] for item in confusions]
            report.extend(
                [
                    f"### {language.upper()} most common confusions",
                    "",
                    markdown_table(["Gold intent", "Predicted intent", "Count"], confusion_rows),
                    "",
                ]
            )
    report.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "uv run python playground/benchmark_banking77.py",
            "```",
            "",
            "The script downloads/validates the official test CSV, creates the Spanish CSV if absent, then writes the JSONL predictions, aggregate metrics, and this report. Use `--limit 20` for a small endpoint smoke run; use `--force-translation` to regenerate the Spanish file.",
            "",
            "## Dataset citation and sources",
            "",
            "BANKING77: Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders (2020)](https://arxiv.org/abs/2003.04807). Dataset card: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77). The dataset is listed as CC BY 4.0 on its Hugging Face card. Source test CSV: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/banking_data/test.csv).",
            "",
            f"Machine translation: [Google Translate](https://translate.google.com/); generated `{metrics['translation']['generated_at_utc']}`.",
            "",
        ]
    )
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("both", "en", "es"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="Benchmark only the first N rows (smoke test)")
    parser.add_argument("--workers", type=int, default=int(os.getenv("BANKING77_WORKERS", "8")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("BANKING77_TIMEOUT_SECONDS", "30")))
    parser.add_argument("--translation-workers", type=int, default=3)
    parser.add_argument("--translation-batch-size", type=int, default=16)
    parser.add_argument("--force-download", action="store_true", help="Download the source test CSV again")
    parser.add_argument("--force-translation", action="store_true", help="Regenerate the Spanish CSV")
    parser.add_argument("--prepare-only", action="store_true", help="Download and translate data without calling SystemOne")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")
    if args.workers < 1 or args.translation_workers < 1 or args.translation_batch_size < 1:
        raise SystemExit("worker counts and translation batch size must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    endpoint = os.getenv("SYSTEMONE_URL") or os.getenv("DINO_SYSTEMONE_URL") or os.getenv("DOOM_SYSTEMONE_URL")
    if not args.prepare_only and not endpoint:
        raise SystemExit("SYSTEMONE_URL is missing; set it in the project .env file")
    model = os.getenv("BANKING77_MODEL", os.getenv("DINO_MODEL", DEFAULT_MODEL))

    english_rows, english_sha = download_test_csv(force=args.force_download)
    labels = validate_rows(english_rows, expected_count=3080)
    spanish_rows, spanish_sha = get_or_create_spanish_csv(
        english_rows,
        force=args.force_translation,
        batch_size=args.translation_batch_size,
        workers=args.translation_workers,
    )
    translation_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(
        f"Prepared Banking77 test data: {len(english_rows):,} rows, "
        f"{len(labels)} intents; Spanish labels/order verified.",
        flush=True,
    )
    if args.prepare_only:
        print(f"English CSV: {TEST_CSV_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Spanish CSV: {SPANISH_CSV_PATH.relative_to(PROJECT_ROOT)}")
        return

    languages = ("en", "es") if args.language == "both" else (args.language,)
    source_rows = {"en": english_rows, "es": spanish_rows}
    results: dict[str, Any] = {}
    for language in languages:
        assert endpoint is not None
        results[language] = run_language(
            language,
            source_rows[language],
            labels,
            endpoint,
            model,
            workers=args.workers,
            timeout=args.timeout,
            limit=args.limit,
        )

    metrics = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "PolyAI/Banking77",
            "split": "test",
            "examples": len(english_rows),
            "intents": len(labels),
            "english_csv": str(TEST_CSV_PATH.relative_to(PROJECT_ROOT)),
            "english_sha256": english_sha,
            "spanish_csv": str(SPANISH_CSV_PATH.relative_to(PROJECT_ROOT)),
            "spanish_sha256": spanish_sha,
        },
        "translation": {
            "provider": "Google Translate public web translation endpoint",
            "source_language": "en",
            "target_language": "es",
            "generated_at_utc": translation_timestamp,
            "labels_and_order_preserved": True,
        },
        "model": model,
        "configuration": {
            "endpoint_source": "SYSTEMONE_URL from project .env",
            "workers": args.workers,
            "timeout_seconds": args.timeout,
            "attempts_per_example": 4,
            "limit": args.limit,
            "same_english_prompt_and_intent_options_for_both_languages": True,
        },
        "results": results,
    }
    atomic_write_text(METRICS_PATH, json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(REPORT_PATH, render_report(metrics))
    print(f"Metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    for language, summary in results.items():
        print(
            f"{language.upper()}: accuracy={summary['accuracy']:.2%}, "
            f"macro-F1={summary['macro_f1']:.2%}, "
            f"mean_latency={summary['latency_ms']['mean']:.1f}ms, "
            f"failures={summary['request_failures']}"
        )


if __name__ == "__main__":
    main()
