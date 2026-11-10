"""Export valid patient CSV records while reporting rejected records."""

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, parse_csv_patient, validate_csv_columns


def clean_csv(source: Path, output: Path) -> dict:
    if output.resolve() in {source.resolve(), DEFAULT_MODEL_PATH.resolve()}:
        raise ValueError("Output must differ from the input and shipped model paths.")
    snapshot = source.read_bytes()
    reader = csv.reader(StringIO(snapshot.decode("utf-8-sig"), newline=""), strict=True)
    columns = next(reader, None)
    validate_csv_columns(columns)
    retained, rejected, kept_numbers = [], [], []
    for number, values in enumerate(reader, start=2):
        try:
            original, _ = parse_csv_patient(columns, values, number)
        except (ValueError, TypeError) as error:
            rejected.append({"csv_row": number, "error": str(error)})
        else:
            retained.append(original)
            kept_numbers.append(number)
    if not retained:
        raise ValueError("No valid patient rows remain; output was not written. Use src.audit for a full error report.")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(retained)
    payload = buffer.getvalue()
    report = {"total_rows": len(retained) + len(rejected), "retained_rows": len(retained),
              "rejected_rows": len(rejected), "retained_csv_rows": kept_numbers, "rejections": rejected,
              "source_sha256": hashlib.sha256(snapshot).hexdigest(),
              "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
    with atomic_text_output(output, newline="") as destination:
        destination.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="CSV containing only valid records.")
    args = parser.parse_args()
    try:
        report = clean_csv(args.input_csv, args.output)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if report["rejected_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
