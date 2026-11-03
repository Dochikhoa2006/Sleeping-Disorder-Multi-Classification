"""Measure raw-input permutation importance on labeled, unseen patient profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

from src.evaluation import feature_fingerprints
from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, read_patient_csv
from src.preprocessing import RANDOM_STATE, RAW_FEATURES, TARGET_COLUMN


def measure_importance(data_path: Path, model_path: Path = DEFAULT_MODEL_PATH,
                       *, repeats: int = 5, seed: int = RANDOM_STATE) -> dict:
    if not 2 <= repeats <= 30:
        raise ValueError("Choose between 2 and 30 permutation repeats.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be between 0 and 4294967295.")
    dataset_bytes = data_path.read_bytes()
    columns, rows, patients = read_patient_csv(StringIO(dataset_bytes.decode("utf-8-sig"), newline=""))
    if TARGET_COLUMN not in columns:
        raise ValueError(f"CSV must contain '{TARGET_COLUMN}'.")
    model_bytes = model_path.read_bytes()
    model = joblib.load(BytesIO(model_bytes))
    if not hasattr(model, "training_feature_hashes_"):
        raise ValueError("Model lacks training fingerprints; retrain it before measuring importance.")
    labels = [str(label) for label in model.classes_]
    target = pd.Series([row[TARGET_COLUMN] for row in rows])
    if not target.isin(labels).all():
        raise ValueError("Every target value must be one of the model's explicit class labels.")
    features = pd.DataFrame(patients)[RAW_FEATURES]
    fingerprints = feature_fingerprints(features)
    overlap = fingerprints.isin(model.training_feature_hashes_)
    if overlap.all():
        raise ValueError("All input rows match training profiles; no unseen rows to score.")
    features, target = features.loc[~overlap], target.loc[~overlap]
    scorer = make_scorer(f1_score, labels=labels, average="macro", zero_division=0, pos_label=None)
    baseline = float(scorer(model, features, target))
    importance = permutation_importance(model, features, target, scoring=scorer,
                                       n_repeats=repeats, random_state=seed, n_jobs=1)
    if not np.isfinite(importance.importances).all():
        raise ValueError("Model produced non-finite importance scores.")
    results = [{"feature": feature, "mean_score_drop": float(importance.importances_mean[index]),
                "score_drop_std": float(importance.importances_std[index]),
                "score_drops": importance.importances[index].tolist()}
               for index, feature in enumerate(RAW_FEATURES)]
    results.sort(key=lambda row: (-row["mean_score_drop"], row["feature"]))
    return {"dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "scoring": "macro_f1", "labels": labels, "baseline_score": baseline,
            "total_rows": len(rows), "training_overlap_rows": int(overlap.sum()),
            "evaluated_rows": len(features),
            "unique_evaluated_profiles": int(fingerprints.loc[~overlap].nunique()),
            "class_counts": {label: int(target.eq(label).sum()) for label in labels},
            "repeats": repeats, "seed": seed, "features": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve() in {args.data.resolve(), args.model.resolve()}:
        raise SystemExit("Error: output must differ from data and model paths.")
    try:
        report = measure_importance(args.data, args.model, repeats=args.repeats, seed=args.seed)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, AttributeError, EOFError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
