"""Audit every record in a JSON patient batch without loading a model."""

from __future__ import annotations

import argparse
from io import StringIO
import json
from pathlib import Path
import sys

from src.output import atomic_text_output
from src.predict import load_patient_json, read_patient_json, validate_patient


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True,
                        help="Patient JSON file; use - for standard input.")
    parser.add_argument("--output", type=Path,
                        help="Save the audit as JSON; omit or use - for standard output.")
    args = parser.parse_args()
    if (args.output and args.output != Path("-") and args.input_json != Path("-")
            and args.output.resolve() == args.input_json.resolve()):
        raise SystemExit("Error: output path must differ from input path.")
    try:
        records = (read_patient_json(StringIO(sys.stdin.read().removeprefix("\ufeff")))
                   if args.input_json == Path("-") else load_patient_json(args.input_json))
        report = audit_patients(records)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output and args.output != Path("-"):
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            sys.stdout.write(payload)
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from error
    if report["invalid_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
