"""Export reproducible, profile-disjoint training and test CSVs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluation import feature_fingerprints
from src.predict import read_patient_csv
from src.preprocessing import RANDOM_STATE, TARGET_COLUMN

LABELS = ("Healthy", "Insomnia", "Sleep Apnea")


def split_dataset(source: Path, destination: Path, *, folds: int = 5,
                  fold: int = 0, seed: int = RANDOM_STATE) -> dict:
    """Publish a new directory only after both CSVs and their manifest are ready."""
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new directory.")
    if not 2 <= folds <= 20 or not 0 <= fold < folds:
        raise ValueError("Choose 2–20 folds and a zero-based fold index smaller than the fold count.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer from 0 to 4294967295.")
    snapshot = source.read_bytes()
    columns, rows, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
    if TARGET_COLUMN not in columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    target = pd.Series([row[TARGET_COLUMN] for row in rows])
    if not target.isin(LABELS).all():
        raise ValueError("Every row must have an explicit Healthy, Insomnia, or Sleep Apnea label.")
    groups = feature_fingerprints(pd.DataFrame(patients))
    profiles = pd.DataFrame({"group": groups, "label": target})
    if (profiles.groupby("group")["label"].nunique() > 1).any():
        raise ValueError("Equivalent patient profiles have conflicting labels.")
    counts = profiles.drop_duplicates().groupby("label").size()
    if any(count < folds for count in counts):
        raise ValueError("Each present class needs at least as many unique profiles as folds.")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    train, test = list(splitter.split(pd.DataFrame(patients), target, groups))[fold]
    if set(groups.iloc[train]) & set(groups.iloc[test]):
        raise RuntimeError("Equivalent profiles crossed the split boundary.")
    if set(target.iloc[train]) != set(target) or set(target.iloc[test]) != set(target):
        raise ValueError("This fold does not cover every present class on both sides; choose another fold or seed.")
    report = {"source_sha256": hashlib.sha256(snapshot).hexdigest(),
              "folds": folds, "test_fold": fold, "seed": seed, "partitions": {}}
    with TemporaryDirectory(prefix=".sleep-split-", dir=destination.parent) as staging:
        for name, indices in (("train", train), ("test", test)):
            output = StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows[index] for index in indices)
            payload = output.getvalue().encode("utf-8")
            Path(staging, name + ".csv").write_bytes(payload)
            report["partitions"][name] = {
                "rows": len(indices), "unique_profiles": int(groups.iloc[indices].nunique()),
                "class_counts": dict(sorted(Counter(target.iloc[indices]).items())),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        Path(staging, "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        Path(staging).rename(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    try:
        report = split_dataset(args.input_csv, args.output_dir, folds=args.folds, fold=args.fold, seed=args.seed)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
