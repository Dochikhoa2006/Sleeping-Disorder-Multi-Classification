"""Audit every record in a JSON or CSV patient batch without loading a model."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
from typing import TextIO
import sys

from src.output import atomic_text_output
from src.predict import (load_patient_json, read_patient_json, validate_patient,
                         validate_csv_columns, parse_csv_patient)


def audit_patients(records: dict | list[dict]) -> dict:
    """Return one validation result per record, without echoing patient values."""
    patients = records if isinstance(records, list) else [records]
    if not patients:
        raise ValueError("At least one patient is required.")
    results = []
    for index, patient in enumerate(patients, start=1):
        try:
            if not isinstance(patient, dict):
                raise ValueError("Patient must be a JSON object.")
            validate_patient(patient)
        except (ValueError, TypeError) as error:
            results.append({"patient": index, "valid": False, "error": str(error)})
        else:
            results.append({"patient": index, "valid": True})
    valid = sum(record["valid"] for record in results)
    return {"total_records": len(results), "valid_records": valid,
            "invalid_records": len(results) - valid, "records": results}


def audit_csv(input_file: TextIO) -> dict:
    """Collect row errors while rejecting unparseable files and invalid headers."""
    reader = csv.reader(input_file, strict=True)
    columns = next(reader, None)
    validate_csv_columns(columns)
    results = []
    for number, values in enumerate(reader, start=2):
        result = {"patient": number - 1, "csv_row": number, "valid": True}
        try:
            parse_csv_patient(columns, values, number)
        except (ValueError, TypeError) as error:
            result.update(valid=False, error=str(error))
        results.append(result)
    if not results:
        raise ValueError("Patient CSV must have at least one patient row.")
    valid = sum(record["valid"] for record in results)
    return {"total_records": len(results), "valid_records": valid,
            "invalid_records": len(results) - valid, "records": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-json", type=Path,
                        help="Patient JSON file; use - for standard input.")
    inputs.add_argument("--input-csv", type=Path,
                        help="Patient CSV file; use - for standard input.")
    parser.add_argument("--output", type=Path,
                        help="Save the audit as JSON; omit or use - for standard output.")
    args = parser.parse_args()
    source = args.input_json or args.input_csv
    if (args.output and args.output != Path("-") and source != Path("-")
            and args.output.resolve() == source.resolve()):
        raise SystemExit("Error: output path must differ from input path.")
    try:
        if args.input_csv:
            if source == Path("-"):
                report = audit_csv(StringIO(sys.stdin.read().removeprefix("\ufeff")))
            else:
                with source.open(encoding="utf-8-sig", newline="") as input_file:
                    report = audit_csv(input_file)
        else:
            records = (read_patient_json(StringIO(sys.stdin.read().removeprefix("\ufeff")))
                       if source == Path("-") else load_patient_json(source))
            report = audit_patients(records)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output and args.output != Path("-"):
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            sys.stdout.write(payload)
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if report["invalid_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
