"""Evaluate a saved model on a labeled external CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import evaluate_external
from src.predict import DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on labeled unseen profiles.")
    parser.add_argument("--data", type=Path, required=True, help="Labeled CSV to evaluate.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination.")
    arguments = parser.parse_args()
    try:
        report = evaluate_external(arguments.data, arguments.model, arguments.output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Evaluated {report['evaluated_rows']} of {report['total_rows']} rows; excluded {report['training_overlap_rows']} training-profile overlaps.")
    print(f"Macro F1={report['macro_f1']:.3f}, accuracy={report['accuracy']:.3f}")
    print(f"Saved evaluation to: {arguments.output}")


if __name__ == "__main__":
    main()
