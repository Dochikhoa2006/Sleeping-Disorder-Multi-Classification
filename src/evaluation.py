"""Metrics shared by training holdout and external labeled evaluation."""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET_COLUMN, SleepFeatureEngineer
from src.output import atomic_text_output


def classification_metrics(target: pd.Series, predictions: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "macro_f1": float(f1_score(target, predictions, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(target, predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(target, predictions, labels=labels).tolist(),
    }


def feature_fingerprints(features: pd.DataFrame) -> pd.Series:
    """Hash only the engineered features actually consumed by the model."""
    engineered = SleepFeatureEngineer().transform(features)
    for column in NUMERIC_FEATURES:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce").astype("float64")
    for column in CATEGORICAL_FEATURES:
        engineered[column] = engineered[column].astype("string")
    return pd.util.hash_pandas_object(engineered, index=False)


def validate_bootstrap_options(runs: int, seed: int) -> None:
    if type(runs) is not int or (runs != 0 and not 100 <= runs <= 2000):
        raise ValueError("Bootstrap runs must be 0 (disabled) or an integer from 100 to 2000.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Bootstrap seed must be an integer from 0 to 4294967295.")


def bootstrap_intervals(target, predictions, groups, labels: list[str], *,
                        runs: int = 500, seed: int = RANDOM_STATE) -> dict:
    """Resample complete engineered-profile clusters with the classifier held fixed."""
    validate_bootstrap_options(runs, seed)
    if runs == 0:
        raise ValueError("Interval calculation requires at least 100 bootstrap runs.")
    actual, predicted, fingerprints = map(np.asarray, (target, predictions, groups))
    if (any(values.ndim != 1 for values in (actual, predicted, fingerprints))
            or not len(actual) or len(actual) != len(predicted) or len(actual) != len(fingerprints)):
        raise ValueError("Bootstrap targets, predictions, and groups must be aligned non-empty vectors.")
    if not labels or not np.isin(actual, labels).all() or not np.isin(predicted, labels).all():
        raise ValueError("Bootstrap targets and predictions must use the supplied class labels.")
    clusters = {}
    for index, group in enumerate(fingerprints):
        clusters.setdefault(group, []).append(index)
    if len(clusters) < 2:
        raise ValueError("Bootstrap intervals require at least two unseen unique profiles.")
    members = [np.asarray(indices) for indices in clusters.values()]
    rng = np.random.default_rng(seed)
    scores = {"accuracy": [], "macro_f1": []}
    row_counts = []
    for _ in range(runs):
        indices = np.concatenate([members[index] for index in rng.integers(0, len(members), size=len(members))])
        sampled_actual, sampled_predictions = actual[indices], predicted[indices]
        scores["accuracy"].append(float(accuracy_score(sampled_actual, sampled_predictions)))
        scores["macro_f1"].append(float(f1_score(sampled_actual, sampled_predictions,
                                               labels=labels, average="macro", zero_division=0)))
        row_counts.append(len(indices))
    intervals = {name: {"lower": float(np.percentile(values, 2.5)),
                        "upper": float(np.percentile(values, 97.5))}
                 for name, values in scores.items()}
    return {"method": "profile_cluster_percentile", "confidence_level": 0.95,
            "resamples": runs, "seed": seed, "unique_profiles": len(members),
            "sampled_rows_min": min(row_counts), "sampled_rows_max": max(row_counts),
            **intervals}


def validate_group_options(group_by, min_rows: int) -> tuple[str, ...]:
    if (not isinstance(group_by, (tuple, list)) or any(not isinstance(field, str) or field not in CATEGORICAL_FEATURES for field in group_by)
            or len(group_by) != len(set(group_by))):
        raise ValueError("Choose distinct categorical grouping fields: " + ", ".join(CATEGORICAL_FEATURES))
    if type(min_rows) is not int or not 1 <= min_rows <= 100000:
        raise ValueError("Minimum group rows must be an integer from 1 to 100000.")
    return tuple(group_by)


def subgroup_metrics(features: pd.DataFrame, target: pd.Series, predictions: list[str],
                     fingerprints: pd.Series, labels: list[str], *, group_by, min_rows: int = 5) -> dict:
    """Describe independent categorical slices of an already-scored unseen batch."""
    fields = validate_group_options(group_by, min_rows)
    if (not len(features) or len(predictions) != len(features)
            or not features.index.equals(target.index) or not features.index.equals(fingerprints.index)):
        raise ValueError("Subgroup features, targets, predictions, and fingerprints must be aligned.")
    engineered = SleepFeatureEngineer().transform(features)
    predicted = pd.Series(predictions, index=target.index)
    results = {}
    for field in fields:
        groups = []
        values = engineered[field]
        for value in pd.unique(values):
            missing = bool(pd.isna(value))
            mask = values.isna() if missing else values.eq(value)
            rows = int(mask.sum())
            groups.append({"value": None if missing else str(value), "rows": rows,
                           "share": rows / len(features),
                           "unique_profiles": int(fingerprints.loc[mask].nunique()),
                           "class_counts": {label: int(target.loc[mask].eq(label).sum()) for label in labels},
                           "status": "scored" if rows >= min_rows else "insufficient_rows",
                           "metrics": classification_metrics(target.loc[mask], predicted.loc[mask].tolist(), labels)
                                      if rows >= min_rows else None})
        groups.sort(key=lambda group: (-group["rows"], group["value"] is not None, group["value"] or ""))
        results[field] = groups
    return {"min_rows": min_rows, "fields": results}


def evaluate_external(data_path: Path, model_path: Path, output_path: Path | None = None, *,
                      bootstrap_runs: int = 0, bootstrap_seed: int = RANDOM_STATE,
                      group_by=(), min_group_rows: int = 5) -> dict:
    """Score labeled, unseen rows and account for training-profile overlap."""
    if output_path is not None and output_path.resolve() in {data_path.resolve(), model_path.resolve()}:
        raise ValueError("Evaluation output path must differ from data and model paths.")
    report = evaluate_bytes(data_path.read_bytes(), model_path, compressed=data_path.suffix.lower() == ".gz",
                            bootstrap_runs=bootstrap_runs, bootstrap_seed=bootstrap_seed,
                            group_by=group_by, min_group_rows=min_group_rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_text_output(output_path) as output_file:
            json.dump(report, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
    return report


def evaluate_bytes(dataset_bytes: bytes, model_path: Path, *, compressed: bool = False, include_rows: bool = False,
                   bootstrap_runs: int = 0, bootstrap_seed: int = RANDOM_STATE,
                   group_by=(), min_group_rows: int = 5) -> dict:
    """Evaluate CSV bytes in memory using the same rules as file evaluation."""
    validate_bootstrap_options(bootstrap_runs, bootstrap_seed)
    group_by = validate_group_options(group_by, min_group_rows)
    model_bytes = model_path.read_bytes()
    pipeline = joblib.load(BytesIO(model_bytes))
    if not hasattr(pipeline, "training_feature_hashes_"):
        raise ValueError("Model lacks training fingerprints; retrain it before external evaluation.")
    if compressed:
        try:
            csv_bytes = gzip.decompress(dataset_bytes)
        except (OSError, EOFError, zlib.error) as error:
            raise ValueError("Evaluation gzip input is invalid or truncated.") from error
    else:
        csv_bytes = dataset_bytes
    dataset = pd.read_csv(BytesIO(csv_bytes))
    if dataset.empty:
        raise ValueError("Evaluation CSV must contain at least one row.")
    if TARGET_COLUMN not in dataset:
        raise ValueError(f"Evaluation CSV must contain '{TARGET_COLUMN}'.")
    target = dataset[TARGET_COLUMN]
    labels = [str(label) for label in pipeline.classes_]
    if target.isna().any() or not target.isin(labels).all():
        raise ValueError(f"Every '{TARGET_COLUMN}' value must be one of: {', '.join(labels)}.")

    features = dataset.drop(columns=[TARGET_COLUMN])
    fingerprints = feature_fingerprints(features)
    overlap = fingerprints.isin(pipeline.training_feature_hashes_)
    unseen = ~overlap
    if not unseen.any():
        raise ValueError("All evaluation rows match training profiles; no unseen rows to score.")
    unseen_target = target.loc[unseen]
    predictions = [str(value) for value in pipeline.predict(features.loc[unseen])]
    report = {
        "model_name": pipeline.model_name_,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "total_rows": len(dataset),
        "training_overlap_rows": int(overlap.sum()),
        "evaluated_rows": int(unseen.sum()),
        "unique_evaluated_profiles": int(fingerprints.loc[unseen].nunique()),
        "duplicate_evaluated_rows": int(fingerprints.loc[unseen].duplicated().sum()),
        "class_row_counts": {
            label: {
                "total_rows": int(target.eq(label).sum()),
                "training_overlap_rows": int((target.eq(label) & overlap).sum()),
                "evaluated_rows": int((target.eq(label) & unseen).sum()),
            }
            for label in labels
        },
        "labels": labels,
        **classification_metrics(unseen_target, predictions, labels),
    }
    if group_by:
        report["subgroups"] = subgroup_metrics(features.loc[unseen], unseen_target, predictions,
                                               fingerprints.loc[unseen], labels, group_by=group_by,
                                               min_rows=min_group_rows)
    if bootstrap_runs:
        report["bootstrap"] = bootstrap_intervals(unseen_target.to_numpy(), predictions,
                                                  fingerprints.loc[unseen].to_numpy(), labels,
                                                  runs=bootstrap_runs, seed=bootstrap_seed)
    if include_rows:
        remaining = iter(predictions)
        diagnostics = []
        for index, (actual, excluded) in enumerate(zip(target, overlap, strict=True), start=1):
            predicted = None if excluded else next(remaining)
            diagnostics.append({
                "record_number": index, "actual": str(actual), "predicted": predicted,
                "status": "training_overlap" if excluded else "correct" if predicted == actual else "incorrect",
            })
        report["row_results"] = diagnostics
    return report
