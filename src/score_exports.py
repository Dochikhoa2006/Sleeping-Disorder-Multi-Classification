"""Score labeled prediction exports without loading or rerunning a model."""

import argparse
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path

import pandas as pd

from src.evaluation import classification_metrics
from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, read_prediction_csv
from src.preprocessing import TARGET_COLUMN
from src.split_data import LABELS


def score_export(snapshot: bytes, *, min_accuracy: float | None = None,
                 min_macro_f1: float | None = None) -> dict:
    thresholds = {"accuracy": min_accuracy, "macro_f1": min_macro_f1}
    for name, value in thresholds.items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Minimum {name} must be a finite number from 0 to 1.")
    columns, rows, predictions = read_prediction_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
    if TARGET_COLUMN not in columns:
        raise ValueError(f"Prediction export must include actual labels in '{TARGET_COLUMN}'.")
    target = [row[TARGET_COLUMN] for row in rows]
    if any(value not in LABELS for value in target + predictions):
        raise ValueError("Actual and predicted labels must be Healthy, Insomnia, or Sleep Apnea.")
    metrics = classification_metrics(pd.Series(target), predictions, list(LABELS))
    checks = {name: metrics[name] >= value for name, value in thresholds.items() if value is not None}
    return {"dataset_sha256": hashlib.sha256(snapshot).hexdigest(), "labels": list(LABELS),
            "evaluated_rows": len(rows), "scoring_source": "saved prediction export",
            "training_overlap_checked": False, "class_counts": {label: target.count(label) for label in LABELS},
            **metrics, "thresholds": {name: value for name, value in thresholds.items() if value is not None},
            "threshold_checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Atomic JSON report; omit for standard output.")
    parser.add_argument("--min-accuracy", type=float)
    parser.add_argument("--min-macro-f1", type=float)
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in {args.input_csv.resolve(), DEFAULT_MODEL_PATH.resolve()}:
            raise ValueError("Output must differ from the input and shipped model paths.")
        report = score_export(args.input_csv.read_bytes(), min_accuracy=args.min_accuracy, min_macro_f1=args.min_macro_f1)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
