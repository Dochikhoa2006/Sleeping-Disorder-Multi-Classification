"""Run sleep-disorder predictions with the persisted sklearn pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from numbers import Real
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import RAW_FEATURES
from src.output import atomic_text_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sleep_disorder_pipeline.joblib"
CSV_NUMERIC_FEATURES = (
    "Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "Heart Rate", "Daily Steps",
)
PREDICTION_COLUMN = "Predicted Sleep Disorder"


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    patient = {}
    for key, value in pairs:
        if key in patient:
            raise ValueError(f"Duplicate JSON field: {key!r}.")
        patient[key] = value
    return patient


def load_patient_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as input_file:
        patient = json.load(input_file, object_pairs_hook=_unique_json_object)
    if isinstance(patient, list):
        if not patient:
            raise ValueError("Patient JSON array must contain at least one object.")
        for index, item in enumerate(patient, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Patient {index} must be a JSON object.")
    elif not isinstance(patient, dict):
        raise ValueError("Patient JSON must contain an object or an array of objects.")
    return patient


def load_patient_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[dict[str, Any]]]:
    """Read CSV rows without changing their original values for output."""
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.reader(input_file)
        columns = next(reader, None)
        if not columns:
            raise ValueError("Patient CSV must have a header and at least one row.")
        if len(columns) != len(set(columns)) or any(not column for column in columns):
            raise ValueError("Patient CSV headers must be unique and non-empty.")
        missing = sorted(set(RAW_FEATURES) - set(columns))
        if missing:
            raise ValueError("Missing required feature(s): " + ", ".join(missing))
        if PREDICTION_COLUMN in columns:
            raise ValueError(f"Patient CSV already has a '{PREDICTION_COLUMN}' column.")

        original_rows: list[dict[str, str]] = []
        patients: list[dict[str, Any]] = []
        for row_number, values in enumerate(reader, start=2):
            if len(values) != len(columns):
                raise ValueError(f"CSV row {row_number}: expected {len(columns)} fields, got {len(values)}.")
            original = dict(zip(columns, values))
            patient: dict[str, Any] = dict(original)
            for feature in CSV_NUMERIC_FEATURES:
                try:
                    patient[feature] = float(original[feature])
                except ValueError as error:
                    raise ValueError(f"CSV row {row_number}: '{feature}' must be a number.") from error
            try:
                validate_patient(patient)
            except (ValueError, TypeError) as error:
                raise ValueError(f"CSV row {row_number}: {error}") from error
            original_rows.append(original)
            patients.append(patient)
    if not patients:
        raise ValueError("Patient CSV must have at least one patient row.")
    return columns, original_rows, patients


def write_prediction_csv(
    path: Path, columns: list[str], rows: list[dict[str, str]], predictions: list[str]
) -> None:
    with atomic_text_output(path, newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=[*columns, PREDICTION_COLUMN])
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow({**row, PREDICTION_COLUMN: prediction})


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
            or not minimum <= value <= maximum
            or not math.isfinite(float(value))
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


def validate_patients(patients: list[dict[str, Any]]) -> None:
    """Validate a batch without loading a model or running predictions."""
    if not patients:
        raise ValueError("At least one patient is required.")
    for index, patient in enumerate(patients, start=1):
        try:
            validate_patient(patient)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Patient {index}: {error}") from error


def predict_many(
    patients: list[dict[str, Any]], model_path: Path = DEFAULT_MODEL_PATH
) -> list[str]:
    """Validate all records, then run one batch through the saved pipeline."""
    validate_patients(patients)
    pipeline = joblib.load(model_path)
    patient_frame = pd.DataFrame(
        [{key: patient[key] for key in RAW_FEATURES} for patient in patients]
    )
    return [str(label) for label in pipeline.predict(patient_frame)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sleep-disorder classes for one or more patients."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-json",
        type=Path,
        help="Path to a patient JSON object or an array of patient objects.",
    )
    input_group.add_argument(
        "--input-csv", type=Path, help="Path to a CSV with one patient per row."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write predictions as a JSON array in input order.",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        help="Write input CSV rows with a predicted class column.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate input records without loading a model or predicting.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.validate_only and (arguments.output_json or arguments.output_csv):
        raise SystemExit("Error: --validate-only cannot be combined with output options.")
    if arguments.output_json and not arguments.input_json:
        raise SystemExit("Error: --output-json requires --input-json.")
    if arguments.output_csv and not arguments.input_csv:
        raise SystemExit("Error: --output-csv requires --input-csv.")
    output = arguments.output_json or arguments.output_csv
    source = arguments.input_json or arguments.input_csv
    if output and source and output.resolve() == source.resolve():
        raise SystemExit("Error: output path must differ from input path.")
    if output and output.resolve() == arguments.model.resolve():
        raise SystemExit("Error: output path must differ from model path.")
    try:
        if arguments.input_csv:
            columns, original_rows, patients = load_patient_csv(arguments.input_csv)
        else:
            patient_data = (
                load_patient_json(arguments.input_json)
                if arguments.input_json else prompt_for_patient()
            )
            patients = patient_data if isinstance(patient_data, list) else [patient_data]
        if arguments.validate_only:
            validate_patients(patients)
            print(f"Validated {len(patients)} patient record(s).")
            return
        predictions = predict_many(patients, arguments.model)
        if arguments.output_json:
            with atomic_text_output(arguments.output_json) as output_file:
                output_file.write(json.dumps(predictions, indent=2) + "\n")
        if arguments.output_csv:
            write_prediction_csv(arguments.output_csv, columns, original_rows, predictions)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Error: {error}") from error

    if arguments.output_json or arguments.output_csv:
        print(f"Saved {len(predictions)} prediction(s) to: {arguments.output_json or arguments.output_csv}")
    elif len(predictions) == 1:
        print(f"\nPredicted sleep disorder: {predictions[0]}")
    else:
        for index, prediction in enumerate(predictions, start=1):
            print(f"Patient {index}: {prediction}")
    print("This educational prediction is not medical advice.")


if __name__ == "__main__":
    main()
