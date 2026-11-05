"""Remove equivalent patient profiles while retaining original CSV field values."""

from __future__ import annotations

import argparse
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


def deduplicate_csv(source: Path, destination: Path, *, keep: str = "first") -> dict:
    if keep not in ("first", "last"):
        raise ValueError("Choose first or last retention.")
    if destination.resolve() in {source.resolve(), DEFAULT_MODEL_PATH.resolve()}:
        raise ValueError("Output must differ from the input and shipped model paths.")
    snapshot = source.read_bytes()
    columns, rows, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
    groups = {}
    for index, fingerprint in enumerate(feature_fingerprints(pd.DataFrame(patients))):
        groups.setdefault(int(fingerprint), []).append(index)
    if TARGET_COLUMN in columns:
        for indices in groups.values():
            if len({rows[index][TARGET_COLUMN] for index in indices}) > 1:
                numbers = ", ".join(str(index + 2) for index in indices)
                raise ValueError(f"Equivalent profiles have conflicting '{TARGET_COLUMN}' values at CSV rows {numbers}.")
    retained, duplicates = [], []
    for indices in groups.values():
        chosen = indices[0] if keep == "first" else indices[-1]
        retained.append(chosen)
        if len(indices) > 1:
            duplicates.append({"retained_csv_row": chosen + 2,
                               "removed_csv_rows": [index + 2 for index in indices if index != chosen]})
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows[index] for index in sorted(retained))
    payload = output.getvalue()
    report = {"input_rows": len(rows), "retained_rows": len(retained),
              "removed_rows": len(rows) - len(retained), "keep": keep,
              "source_sha256": hashlib.sha256(snapshot).hexdigest(),
              "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
              "duplicate_groups": duplicates}
    with atomic_text_output(destination, newline="") as output_file:
        output_file.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", choices=("first", "last"), default="first")
    args = parser.parse_args()
    try:
        report = deduplicate_csv(args.input_csv, args.output, keep=args.keep)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
