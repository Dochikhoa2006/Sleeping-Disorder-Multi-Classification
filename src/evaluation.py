"""Metrics shared by training holdout and external labeled evaluation."""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, SleepFeatureEngineer
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


def evaluate_external(data_path: Path, model_path: Path, output_path: Path | None = None) -> dict:
    """Score labeled, unseen rows and account for training-profile overlap."""
    if output_path is not None and output_path.resolve() in {data_path.resolve(), model_path.resolve()}:
        raise ValueError("Evaluation output path must differ from data and model paths.")
    model_bytes = model_path.read_bytes()
    pipeline = joblib.load(BytesIO(model_bytes))
    if not hasattr(pipeline, "training_feature_hashes_"):
        raise ValueError("Model lacks training fingerprints; retrain it before external evaluation.")
    dataset_bytes = data_path.read_bytes()
    if data_path.suffix.lower() == ".gz":
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
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_text_output(output_path) as output_file:
            json.dump(report, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
    return report
