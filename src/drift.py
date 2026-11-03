"""Compare patient input distributions without loading a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
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
            "total_variation_distance": sum(abs(row["current_share"] - row["reference_share"])
                                             for row in distributions) / 2,
            "distribution": distributions}


def compare_datasets(reference_path: Path, current_path: Path) -> dict:
    frames, descriptions = [], []
    for path in (reference_path, current_path):
        snapshot = path.read_bytes()
        _, _, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
        frames.append(SleepFeatureEngineer().transform(pd.DataFrame(patients)))
        descriptions.append({"rows": len(patients), "sha256": hashlib.sha256(snapshot).hexdigest()})
    reference, current = frames
    return {"reference": descriptions[0], "current": descriptions[1],
            "numeric": {feature: numeric_shift(reference[feature].to_numpy(dtype=float),
                                               current[feature].to_numpy(dtype=float))
                        for feature in NUMERIC_FEATURES},
            "categorical": {feature: categorical_shift(reference[feature], current[feature])
                            for feature in CATEGORICAL_FEATURES}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Save JSON atomically; omit for standard output.")
    args = parser.parse_args()
    if args.output and args.output.resolve() in {args.reference.resolve(), args.current.resolve()}:
        raise SystemExit("Error: output must differ from both input paths.")
    try:
        report = compare_datasets(args.reference, args.current)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
