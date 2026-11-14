"""Compare two trusted saved classifiers on the same patient batch."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.output import atomic_text_output
from src.predict import load_patient_csv, load_patient_json, validate_patients
from src.preprocessing import RAW_FEATURES


def compare_models(patients: list[dict], baseline: Path, candidate: Path) -> dict:
    """Report prediction changes, loading each model from a hashed snapshot."""
    validate_patients(patients)
    frame = pd.DataFrame([{key: patient[key] for key in RAW_FEATURES} for patient in patients])
    predictions = []
    hashes = []
    for path in (baseline, candidate):
        snapshot = path.read_bytes()
        model = joblib.load(BytesIO(snapshot))
        classes = np.asarray(getattr(model, "classes_", []))
        if classes.ndim != 1 or not len(classes):
            raise ValueError("Each model must be a fitted classifier with a one-dimensional class list.")
        known = [str(label) for label in classes]
        if len(set(known)) != len(known):
            raise ValueError("Each model must expose distinct class labels.")
        values = np.asarray(model.predict(frame.copy()))
        if values.ndim != 1 or len(values) != len(patients):
            raise ValueError("Each model must return exactly one prediction per patient in a one-dimensional array.")
        labels = [str(label) for label in values]
        if any(label not in known for label in labels):
            raise ValueError("Each model must predict only its declared class labels.")
        predictions.append(labels)
        hashes.append(hashlib.sha256(snapshot).hexdigest())
    first, second = predictions
    transitions = Counter(zip(first, second, strict=True))
    changes = [{"patient": index, "baseline": before, "candidate": after}
               for index, (before, after) in enumerate(zip(first, second, strict=True), start=1)
               if before != after]
    return {
        "baseline_sha256": hashes[0], "candidate_sha256": hashes[1],
        "total_records": len(patients), "changed_records": len(changes),
        "unchanged_records": len(patients) - len(changes),
        "agreement_rate": (len(patients) - len(changes)) / len(patients),
        "transitions": [{"baseline": before, "candidate": after, "count": count}
                        for (before, after), count in sorted(transitions.items())],
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-json", type=Path)
    inputs.add_argument("--input-csv", type=Path)
    parser.add_argument("--output", type=Path, help="Save JSON atomically; omit for standard output.")
    parser.add_argument("--fail-on-change", action="store_true",
                        help="Exit with status 1 after reporting any changed predictions.")
    args = parser.parse_args()
    source = args.input_json or args.input_csv
    if args.output and args.output.resolve() in {path.resolve() for path in (source, args.baseline, args.candidate)}:
        raise SystemExit("Error: output must differ from input and both model paths.")
    try:
        if args.input_csv:
            _, _, patients = load_patient_csv(source)
        else:
            records = load_patient_json(source)
            patients = records if isinstance(records, list) else [records]
        report = compare_models(patients, args.baseline, args.candidate)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, AttributeError, EOFError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if args.fail_on_change and report["changed_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
