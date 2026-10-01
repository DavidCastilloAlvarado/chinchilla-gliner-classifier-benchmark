"""List misclassified examples from a predictions_*.jsonl file.

Usage:
    uv run python src/classifier/review_errors.py
    uv run python src/classifier/review_errors.py --file data/predictions_banking_intents_es.jsonl
    uv run python src/classifier/review_errors.py --intent check_balance   # only one intent
    uv run python src/classifier/review_errors.py --top 5                  # 5 per confusion pair
    uv run python src/classifier/review_errors.py --out errors.jsonl       # also save
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_FILE = DATA_DIR / "predictions_banking_intents_es.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="predictions_*.jsonl file")
    parser.add_argument("--intent", default=None, help="Only show errors where the GOLD intent is this")
    parser.add_argument("--top", type=int, default=3, help="Max examples printed per confusion pair")
    parser.add_argument("--out", type=Path, default=None, help="Also write misclassified rows to this JSONL")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = [r for r in rows if r["predicted"] != r["intent"]]

    print(f"{args.file.name}: {len(errors)}/{len(rows)} misclassified ({len(errors) / len(rows):.1%})")

    pairs = Counter((e["intent"], e["predicted"]) for e in errors)
    print("\nConfusion pairs (gold -> predicted):")
    for (gold, pred), n in pairs.most_common():
        print(f"  {gold:22s} -> {pred:22s} {n}")

    print("\nExamples (up to --top per pair):")
    shown: dict[tuple, int] = Counter()
    for e in errors:
        if args.intent and e["intent"] != args.intent:
            continue
        key = (e["intent"], e["predicted"])
        if shown[key] >= args.top:
            continue
        shown[key] += 1
        conf = f"{e['confidence']:.3f}" if e.get("confidence") is not None else "n/a"
        print(f"\n  [{e['id']}] gold={e['intent']}  pred={e['predicted']}  conf={conf}")
        print(f"      {e['text']}")

    if args.out:
        with args.out.open("w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(errors)} rows to {args.out}")


if __name__ == "__main__":
    main()
