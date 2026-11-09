"""Convert validated patient inputs to a feature-only CSV or JSON batch."""

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import sys

from src.output import atomic_text_output
from src.predict import (CSV_NUMERIC_FEATURES, DEFAULT_MODEL_PATH, read_patient_csv,
                         read_patient_json, validate_patients)
from src.preprocessing import RAW_FEATURES


def convert_text(content: str, input_format: str, output_format: str) -> str:
    if input_format not in ("csv", "json") or output_format not in ("csv", "json"):
        raise ValueError("Choose csv or json input and output formats.")
    source = StringIO(content.removeprefix("\ufeff"), newline="")
    if input_format == "csv":
        _, _, patients = read_patient_csv(source)
    else:
        records = read_patient_json(source)
        patients = records if isinstance(records, list) else [records]
        validate_patients(patients)
    projected = [{field: float(patient[field]) if field in CSV_NUMERIC_FEATURES else patient[field]
                  for field in RAW_FEATURES} for patient in patients]
    if output_format == "json":
        return json.dumps(projected, indent=2, allow_nan=False) + "\n"
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RAW_FEATURES)
    writer.writeheader()
    writer.writerows(projected)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-json", type=Path, help="JSON object or batch; - reads standard input.")
    inputs.add_argument("--input-csv", type=Path, help="Patient CSV; - reads standard input.")
    parser.add_argument("--to", choices=("csv", "json"), required=True)
    parser.add_argument("--output", type=Path, help="Atomic output; omit or use - for standard output.")
    args = parser.parse_args()
    source = args.input_json if args.input_json is not None else args.input_csv
    try:
        if args.output and args.output != Path("-") and (
            args.output.resolve() == DEFAULT_MODEL_PATH.resolve()
            or (source != Path("-") and args.output.resolve() == source.resolve())
        ):
            raise ValueError("Output must differ from the input and shipped model paths.")
        if source == Path("-"):
            content = sys.stdin.read()
        else:
            with source.open(encoding="utf-8-sig", newline="") as input_file:
                content = input_file.read()
        payload = convert_text(content, "json" if args.input_json is not None else "csv", args.to)
        if args.output and args.output != Path("-"):
            with atomic_text_output(args.output, newline="") as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
