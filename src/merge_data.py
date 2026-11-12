"""Merge validated patient CSV batches with matching columns."""

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, read_patient_csv


def merge_csvs(sources: list[Path], output: Path) -> dict:
    if len(sources) < 2:
        raise ValueError("Choose at least two input CSV files.")
    resolved = [source.resolve() for source in sources]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Input paths must be distinct; the same file cannot be merged twice.")
    if output.resolve() in {*resolved, DEFAULT_MODEL_PATH.resolve()}:
        raise ValueError("Output must differ from every input and the shipped model path.")
    columns, combined, descriptions = None, [], []
    for position, source in enumerate(sources, start=1):
        snapshot = source.read_bytes()
        current_columns, rows, _ = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
        if columns is None:
            columns = current_columns
        elif set(columns) != set(current_columns):
            raise ValueError(f"Input {position} has different columns; all inputs must share the same field names.")
        descriptions.append({"input": position, "sha256": hashlib.sha256(snapshot).hexdigest(),
                             "rows": len(rows), "first_output_csv_row": len(combined) + 2,
                             "last_output_csv_row": len(combined) + len(rows) + 1})
        combined.extend(rows)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(combined)
    payload = buffer.getvalue()
    with atomic_text_output(output, newline="") as destination:
        destination.write(payload)
    return {"input_files": len(sources), "output_rows": len(combined), "columns": columns,
            "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "inputs": descriptions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", nargs="+", type=Path, required=True, help="Two or more input CSVs, in output order.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(merge_csvs(args.input_csv, args.output), indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
