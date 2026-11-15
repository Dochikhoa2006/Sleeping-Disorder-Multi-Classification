"""Convert validated patient inputs to feature-only CSV, JSON, or JSON Lines."""

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
    if input_format not in ("csv", "json", "jsonl") or output_format not in ("csv", "json", "jsonl"):
        raise ValueError("Choose csv, json, or jsonl input and output formats.")
    source = StringIO(content.removeprefix("\ufeff"), newline="")
    if input_format == "csv":
        _, _, patients = read_patient_csv(source)
    elif input_format == "jsonl":
        lines = source.read().split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        patients = []
        for number, line in enumerate(lines, start=1):
            try:
                record = read_patient_json(StringIO(line))
                if not isinstance(record, dict):
                    raise ValueError("Each line must contain one patient object.")
                validate_patients([record])
            except (ValueError, TypeError) as error:
                raise ValueError(f"JSON Lines line {number}: {error}") from error
            patients.append(record)
        if not patients:
            raise ValueError("JSON Lines input must contain at least one patient object.")
    else:
        records = read_patient_json(source)
        patients = records if isinstance(records, list) else [records]
        validate_patients(patients)
    projected = [{field: float(patient[field]) if field in CSV_NUMERIC_FEATURES else patient[field]
                  for field in RAW_FEATURES} for patient in patients]
    if output_format == "json":
        return json.dumps(projected, indent=2, allow_nan=False) + "\n"
    if output_format == "jsonl":
        return "".join(json.dumps(patient, ensure_ascii=False, allow_nan=False) + "\n" for patient in projected)
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
    inputs.add_argument("--input-jsonl", type=Path, help="One patient object per line; - reads standard input.")
    parser.add_argument("--to", choices=("csv", "json", "jsonl"), required=True)
    parser.add_argument("--output", type=Path, help="Atomic output; omit or use - for standard output.")
    args = parser.parse_args()
    source, input_format = next((path, kind) for path, kind in
                                ((args.input_json, "json"), (args.input_csv, "csv"), (args.input_jsonl, "jsonl"))
                                if path is not None)
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
        payload = convert_text(content, input_format, args.to)
        if args.output and args.output != Path("-"):
            with atomic_text_output(args.output, newline="") as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
