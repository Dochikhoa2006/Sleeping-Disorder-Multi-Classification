"""Compare patient input distributions without loading a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.output import atomic_text_output
from src.predict import read_patient_csv
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SleepFeatureEngineer


def numeric_shift(reference: np.ndarray, current: np.ndarray) -> dict:
    """Describe samples and their empirical CDF distance; no significance test."""
    def describe(values):
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "min": float(np.min(values)), "max": float(np.max(values)),
                "std": float(np.std(values))}
    first, second = describe(reference), describe(current)
    points = np.unique(np.concatenate((reference, current)))
    first_cdf = np.searchsorted(np.sort(reference), points, side="right") / len(reference)
    second_cdf = np.searchsorted(np.sort(current), points, side="right") / len(current)
    delta = second["mean"] - first["mean"]
    return {"reference": first, "current": second, "mean_change": delta,
            "standardized_mean_change": delta / first["std"] if first["std"] else None,
            "ks_distance": float(np.max(np.abs(first_cdf - second_cdf)))}


def categorical_shift(reference: pd.Series, current: pd.Series) -> dict:
    first, second = reference.value_counts().to_dict(), current.value_counts().to_dict()
    categories = sorted(set(first) | set(second))
    distributions = [{"category": str(category), "reference_count": int(first.get(category, 0)),
                      "current_count": int(second.get(category, 0)),
                      "reference_share": first.get(category, 0) / len(reference),
                      "current_share": second.get(category, 0) / len(current)}
                     for category in categories]
    return {"new_categories": sorted(set(second) - set(first)),
            "absent_categories": sorted(set(first) - set(second)),
            "total_variation_distance": min(1.0, sum(abs(row["current_share"] - row["reference_share"])
                                                       for row in distributions) / 2),
            "distribution": distributions}


def _distance_limits(numeric, categorical) -> dict:
    limits = {name: value for name, value in (("numeric", numeric), ("categorical", categorical)) if value is not None}
    for name, value in limits.items():
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f"Maximum {name} distance must be a finite number from 0 to 1.")
    return limits


def compare_datasets(reference_path: Path, current_path: Path, *,
                     max_ks_distance: float | None = None, max_category_distance: float | None = None) -> dict:
    _distance_limits(max_ks_distance, max_category_distance)
    return compare_dataset_bytes(reference_path.read_bytes(), current_path.read_bytes(),
                                 max_ks_distance=max_ks_distance, max_category_distance=max_category_distance)


def compare_dataset_bytes(reference_bytes: bytes, current_bytes: bytes, *,
                          max_ks_distance: float | None = None, max_category_distance: float | None = None) -> dict:
    """Compare uploaded or file-backed snapshots using identical validation rules."""
    limits = _distance_limits(max_ks_distance, max_category_distance)
    frames, descriptions = [], []
    for snapshot in (reference_bytes, current_bytes):
        _, _, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
        frames.append(SleepFeatureEngineer().transform(pd.DataFrame(patients)))
        descriptions.append({"rows": len(patients), "sha256": hashlib.sha256(snapshot).hexdigest()})
    reference, current = frames
    report = {"reference": descriptions[0], "current": descriptions[1],
            "numeric": {feature: numeric_shift(reference[feature].to_numpy(dtype=float),
                                               current[feature].to_numpy(dtype=float))
                        for feature in NUMERIC_FEATURES},
            "categorical": {feature: categorical_shift(reference[feature], current[feature])
                            for feature in CATEGORICAL_FEATURES}}
    if limits:
        metrics = {"numeric": "ks_distance", "categorical": "total_variation_distance"}
        checks = {group: {feature: values[metrics[group]] <= limit for feature, values in report[group].items()}
                  for group, limit in limits.items()}
        report.update(limits=limits, limit_checks=checks,
                      passed=all(value for fields in checks.values() for value in fields.values()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Save JSON atomically; omit for standard output.")
    parser.add_argument("--max-ks-distance", type=float, help="Maximum numeric KS distance (0–1); fail after reporting any exceedance.")
    parser.add_argument("--max-category-distance", type=float, help="Maximum categorical total variation distance (0–1).")
    args = parser.parse_args()
    if args.output and args.output.resolve() in {args.reference.resolve(), args.current.resolve()}:
        raise SystemExit("Error: output must differ from both input paths.")
    try:
        report = compare_datasets(args.reference, args.current, max_ks_distance=args.max_ks_distance,
                                   max_category_distance=args.max_category_distance)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if not report.get("passed", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
