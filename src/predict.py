"""Run sleep-disorder predictions with the persisted sklearn pipeline."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from numbers import Real
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import RAW_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sleep_disorder_pipeline.joblib"


def load_patient_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        patient = json.load(input_file)
    if isinstance(patient, list):
        if not patient:
            raise ValueError("Patient JSON array must contain at least one object.")
        for index, item in enumerate(patient, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Patient {index} must be a JSON object.")
    elif not isinstance(patient, dict):
        raise ValueError("Patient JSON must contain an object or an array of objects.")
    return patient


def prompt_for_patient() -> dict[str, Any]:
    print("\n--- New Patient Inference ---")
    return {
        "Gender": input("Gender (Male/Female): ").strip(),
        "Age": int(input("Age: ")),
        "Occupation": input("Occupation: ").strip(),
        "Sleep Duration": float(input("Sleep duration in hours: ")),
        "Quality of Sleep": int(input("Quality of sleep (1-10): ")),
        "Physical Activity Level": int(
            input("Physical activity in minutes/day: ")
        ),
        "Stress Level": int(input("Stress level (1-10): ")),
        "BMI Category": input(
            "BMI category (Normal/Normal Weight/Overweight/Obese): "
        ).strip(),
        "Blood Pressure": input("Blood pressure (for example, 120/80): ").strip(),
        "Heart Rate": int(input("Heart rate (BPM): ")),
        "Daily Steps": int(input("Daily steps: ")),
    }


def validate_patient(patient: dict[str, Any]) -> None:
    missing = [feature for feature in RAW_FEATURES if feature not in patient]
    if missing:
        raise ValueError(
            "Missing required feature(s): " + ", ".join(sorted(missing))
        )

    for feature in ("Gender", "Occupation", "BMI Category"):
        if not isinstance(patient[feature], str) or not patient[feature].strip():
            raise ValueError(f"'{feature}' must be a non-empty string.")

    ranges = {
        "Age": (1, 120),
        "Sleep Duration": (0, 24),
        "Quality of Sleep": (1, 10),
        "Physical Activity Level": (0, 1_440),
        "Stress Level": (1, 10),
        "Heart Rate": (20, 250),
        "Daily Steps": (0, 100_000),
    }
    for feature, (minimum, maximum) in ranges.items():
        value = patient[feature]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"'{feature}' must be a number from {minimum} to {maximum}."
            )

    pressure_match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*",
        str(patient["Blood Pressure"]),
    )
    if pressure_match is None:
        raise ValueError("'Blood Pressure' must use the systolic/diastolic format.")

    systolic, diastolic = map(float, pressure_match.groups())
    if not (50 <= systolic <= 300 and 30 <= diastolic <= 200):
        raise ValueError("'Blood Pressure' is outside the supported range.")
    if systolic <= diastolic:
        raise ValueError("Systolic blood pressure must exceed diastolic pressure.")


def predict(patient: dict[str, Any], model_path: Path = DEFAULT_MODEL_PATH) -> str:
    return predict_many([patient], model_path)[0]


def predict_many(
    patients: list[dict[str, Any]], model_path: Path = DEFAULT_MODEL_PATH
) -> list[str]:
    """Validate all records, then run one batch through the saved pipeline."""
    if not patients:
        raise ValueError("At least one patient is required.")
    for index, patient in enumerate(patients, start=1):
        try:
            validate_patient(patient)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Patient {index}: {error}") from error

    pipeline = joblib.load(model_path)
    patient_frame = pd.DataFrame(
        [{key: patient[key] for key in RAW_FEATURES} for patient in patients]
    )
    return [str(label) for label in pipeline.predict(patient_frame)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sleep-disorder classes for one or more patients."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Path to a patient JSON object or an array of patient objects.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write predictions as a JSON array in input order.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.output_json and not arguments.input_json:
        raise SystemExit("Error: --output-json requires --input-json.")
    try:
        patient_data = (
            load_patient_json(arguments.input_json)
            if arguments.input_json
            else prompt_for_patient()
        )
        predictions = predict_many(
            patient_data if isinstance(patient_data, list) else [patient_data],
            arguments.model,
        )
        if arguments.output_json:
            arguments.output_json.write_text(
                json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Error: {error}") from error

    if arguments.output_json:
        print(f"Saved {len(predictions)} prediction(s) to: {arguments.output_json}")
    elif len(predictions) == 1:
        print(f"\nPredicted sleep disorder: {predictions[0]}")
    else:
        for index, prediction in enumerate(predictions, start=1):
            print(f"Patient {index}: {prediction}")
    print("This educational prediction is not medical advice.")


if __name__ == "__main__":
    main()
