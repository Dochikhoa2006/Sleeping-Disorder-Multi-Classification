"""Create traceable prediction receipts without exporting patient input fields."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path

import joblib
import pandas as pd

from src.output import atomic_text_output
from src.predict import (CSV_NUMERIC_FEATURES, DEFAULT_MODEL_PATH, read_patient_csv,
                         read_patient_json, validate_patient, validate_patients)
from src.preprocessing import RAW_FEATURES


def input_fingerprint(patient: dict) -> str:
    validate_patient(patient)
    fields = {key: float(patient[key]) if key in CSV_NUMERIC_FEATURES else str(patient[key])
              for key in RAW_FEATURES}
    canonical = json.dumps(fields, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_receipt(patients: list[dict], model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    validate_patients(patients)
    fingerprints = [input_fingerprint(patient) for patient in patients]
    snapshot = model_path.read_bytes()
    model = joblib.load(BytesIO(snapshot))
    if not hasattr(model, "classes_"):
        raise ValueError("Model must be a fitted classifier with class labels.")
    labels = [str(label) for label in model.classes_]
    frame = pd.DataFrame([{key: patient[key] for key in RAW_FEATURES} for patient in patients])
    predictions = [str(label) for label in model.predict(frame)]
    if len(predictions) != len(patients) or any(label not in labels for label in predictions):
        raise ValueError("Model must return one known class label per patient.")
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": {"name": getattr(model, "model_name_", type(model).__name__),
                      "sha256": hashlib.sha256(snapshot).hexdigest(), "labels": labels},
            "count": len(patients),
            "predictions": [{"patient": index, "input_sha256": fingerprint, "prediction": prediction}
                            for index, (fingerprint, prediction) in enumerate(zip(fingerprints, predictions, strict=True), start=1)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-json", type=Path)
    inputs.add_argument("--input-csv", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, help="Save receipt JSON atomically; omit for standard output.")
    args = parser.parse_args()
    source = args.input_json or args.input_csv
    if args.output and args.output.resolve() in {source.resolve(), args.model.resolve()}:
        raise SystemExit("Error: output must differ from input and model paths.")
    try:
        snapshot = source.read_bytes()
        stream = StringIO(snapshot.decode("utf-8-sig"), newline="")
        if args.input_csv:
            _, _, patients = read_patient_csv(stream)
        else:
            records = read_patient_json(stream)
            patients = records if isinstance(records, list) else [records]
        report = create_receipt(patients, args.model)
        report["source_sha256"] = hashlib.sha256(snapshot).hexdigest()
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, AttributeError, EOFError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
