"""Find conflicting labels on equivalent patient profiles without loading a model."""

import argparse
from collections import Counter
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

import pandas as pd

from src.evaluation import feature_fingerprints
from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, read_patient_csv
from src.preprocessing import TARGET_COLUMN

LABELS = ("Healthy", "Insomnia", "Sleep Apnea")


def audit_labels(snapshot: bytes) -> dict:
    columns, rows, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
    if TARGET_COLUMN not in columns:
        raise ValueError(f"CSV must contain '{TARGET_COLUMN}'.")
    labels = [row[TARGET_COLUMN] for row in rows]
    invalid = [index + 2 for index, label in enumerate(labels) if label not in LABELS]
    if invalid:
        raise ValueError(f"Labels must be Healthy, Insomnia, or Sleep Apnea; invalid CSV rows: {invalid}.")
    groups = {}
    for index, fingerprint in enumerate(feature_fingerprints(pd.DataFrame(patients))):
        groups.setdefault(int(fingerprint), []).append(index)
    conflicts = []
    for indices in groups.values():
        counts = Counter(labels[index] for index in indices)
        if len(counts) > 1:
            conflicts.append({"csv_rows": [index + 2 for index in indices], "rows": len(indices),
                              "label_counts": {label: counts[label] for label in LABELS}})
    return {"source_sha256": hashlib.sha256(snapshot).hexdigest(), "total_rows": len(rows),
            "unique_profiles": len(groups), "duplicate_rows": len(rows) - len(groups),
            "class_counts": {label: labels.count(label) for label in LABELS},
            "conflicting_profiles": len(conflicts),
            "rows_in_conflicting_profiles": sum(group["rows"] for group in conflicts),
            "conflicts": conflicts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Atomic JSON report; omit for standard output.")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in {args.input_csv.resolve(), DEFAULT_MODEL_PATH.resolve()}:
            raise ValueError("Output must differ from the input and shipped model paths.")
        report = audit_labels(args.input_csv.read_bytes())
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if report["conflicting_profiles"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
