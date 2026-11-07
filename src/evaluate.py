"""Evaluate a saved model on a labeled external CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import evaluate_external
from src.predict import DEFAULT_MODEL_PATH
from src.preprocessing import CATEGORICAL_FEATURES, RANDOM_STATE


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on labeled unseen profiles.")
    parser.add_argument("--data", type=Path, required=True, help="Labeled CSV or gzip-compressed CSV (.gz) to evaluate.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, help="JSON report destination; omit to print JSON to standard output.")
    parser.add_argument("--bootstrap-runs", type=int, default=0,
                        help="Add 95%% profile-cluster percentile intervals using 100–2000 resamples; 0 disables.")
    parser.add_argument("--bootstrap-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--group-by", nargs="+", choices=CATEGORICAL_FEATURES,
                        help="Add separate score breakdowns for categorical input fields.")
    parser.add_argument("--min-group-rows", type=int, default=5,
                        help="Minimum subgroup size for displaying scores; smaller groups retain counts only.")
    arguments = parser.parse_args()
    try:
        report = evaluate_external(arguments.data, arguments.model, arguments.output,
                                   bootstrap_runs=arguments.bootstrap_runs, bootstrap_seed=arguments.bootstrap_seed,
                                   group_by=arguments.group_by or (), min_group_rows=arguments.min_group_rows)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    if arguments.output is None:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print(f"Evaluated {report['evaluated_rows']} of {report['total_rows']} rows; excluded {report['training_overlap_rows']} training-profile overlaps.")
    print(f"Macro F1={report['macro_f1']:.3f}, accuracy={report['accuracy']:.3f}")
    print(f"Saved evaluation to: {arguments.output}")


if __name__ == "__main__":
    main()
