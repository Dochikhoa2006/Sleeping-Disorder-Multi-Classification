"""Inspect a saved classifier without supplying patient records."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.predict import DEFAULT_MODEL_PATH


def inspect_model(model_path: Path = DEFAULT_MODEL_PATH, *, include_features: bool = False) -> dict:
    """Return the model's provenance, classes, and stored training results."""
    model_bytes = model_path.read_bytes()
    model = joblib.load(BytesIO(model_bytes))
    if not hasattr(model, "classes_"):
        raise ValueError("Model must be a fitted classifier with class labels.")
    report = {
        "model_name": getattr(model, "model_name_", type(model).__name__),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "labels": [str(label) for label in model.classes_],
        "training_metadata": getattr(model, "training_metadata_", None),
        "cross_validation": getattr(model, "cv_results_", None),
    }
    if include_features:
        if not isinstance(model, Pipeline) or len(model.steps) < 2:
            raise ValueError("Feature inspection requires a pipeline with preprocessing steps.")
        report["transformed_features"] = [str(name) for name in model[:-1].get_feature_names_out()]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Print saved model information as JSON.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--include-features", action="store_true", help="Include transformed feature names in the JSON report.")
    arguments = parser.parse_args()
    try:
        report = inspect_model(arguments.model, include_features=arguments.include_features)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
