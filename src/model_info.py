"""Inspect a saved classifier without supplying patient records."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import joblib

from src.predict import DEFAULT_MODEL_PATH


def inspect_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Return the model's provenance, classes, and stored training results."""
    model_bytes = model_path.read_bytes()
    model = joblib.load(BytesIO(model_bytes))
    if not hasattr(model, "classes_"):
        raise ValueError("Model must be a fitted classifier with class labels.")
    return {
        "model_name": getattr(model, "model_name_", type(model).__name__),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "labels": [str(label) for label in model.classes_],
        "training_metadata": getattr(model, "training_metadata_", None),
        "cross_validation": getattr(model, "cv_results_", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print saved model information as JSON.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    arguments = parser.parse_args()
    try:
        report = inspect_model(arguments.model)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
