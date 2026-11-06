"""Check a prediction receipt against supplied inputs and a trusted model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

from src.output import atomic_text_output
from src.predict import (DEFAULT_MODEL_PATH, _unique_json_object, read_patient_csv,
                         read_patient_json, validate_patients)
from src.receipt import create_receipt_from_bytes, input_fingerprint


def _is_hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_receipt(receipt: dict) -> None:
    if not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError("Expected a schema-version-1 receipt object.")
    count, rows, model = receipt.get("count"), receipt.get("predictions"), receipt.get("model")
    if type(count) is not int or count < 1 or not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Receipt count must match a non-empty predictions array.")
    if not isinstance(model, dict) or not _is_hash(model.get("sha256")):
        raise ValueError("Receipt must contain a model SHA-256.")
    labels = model.get("labels")
    if (not isinstance(model.get("name"), str) or not model["name"].strip()
            or not isinstance(labels, list) or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("Receipt must contain a model name and unique class labels.")
    for index, row in enumerate(rows, start=1):
        if (not isinstance(row, dict) or type(row.get("patient")) is not int or row["patient"] != index
                or not _is_hash(row.get("input_sha256")) or row.get("prediction") not in labels):
            raise ValueError("Receipt predictions must have sequential patient numbers, hashes, and known labels.")
    if "source_sha256" in receipt and not _is_hash(receipt["source_sha256"]):
        raise ValueError("Receipt source SHA-256 is invalid.")
    try:
        stamp = datetime.fromisoformat(receipt["generated_at"])
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Missing time zone.")
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Receipt timestamp must include a time zone.") from error


def verify_receipt(receipt: dict, patients: list[dict], model_path: Path = DEFAULT_MODEL_PATH,
                   *, source_sha256: str | None = None) -> dict:
    validate_receipt(receipt)
    validate_patients(patients)
    actual = [input_fingerprint(patient) for patient in patients]
    expected = [row["input_sha256"] for row in receipt["predictions"]]
    input_mismatches = [index + 1 for index in range(max(len(actual), len(expected)))
                        if index >= len(actual) or index >= len(expected) or actual[index] != expected[index]]
    snapshot = model_path.read_bytes()
    digest = hashlib.sha256(snapshot).hexdigest()
    checks = {"model_hash": digest == receipt["model"]["sha256"], "ordered_inputs": not input_mismatches,
              "source_file_hash": source_sha256 == receipt["source_sha256"] if "source_sha256" in receipt else None,
              "model_metadata": None, "predictions": None}
    prediction_mismatches = []
    replayed = checks["model_hash"] and checks["ordered_inputs"]
    if replayed:
        replay = create_receipt_from_bytes(patients, snapshot)
        checks["model_metadata"] = replay["model"] == receipt["model"]
        prediction_mismatches = [index for index, (before, after) in enumerate(
            zip(receipt["predictions"], replay["predictions"], strict=True), start=1)
            if before["prediction"] != after["prediction"]]
        checks["predictions"] = not prediction_mismatches
    return {"verified": (replayed and checks["source_file_hash"] is not False
                         and checks["model_metadata"] is True and checks["predictions"] is True),
            "replayed": replayed, "checks": checks, "model_sha256": digest,
            "input_mismatches": input_mismatches, "prediction_mismatches": prediction_mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-json", type=Path)
    inputs.add_argument("--input-csv", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, help="Save verification JSON atomically; omit for standard output.")
    args = parser.parse_args()
    source = args.input_json or args.input_csv
    if args.output and args.output.resolve() in {path.resolve() for path in (source, args.receipt, args.model)}:
        raise SystemExit("Error: output must differ from input, receipt, and model paths.")
    try:
        with args.receipt.open(encoding="utf-8-sig") as receipt_file:
            receipt = json.load(receipt_file, object_pairs_hook=_unique_json_object)
        snapshot = source.read_bytes()
        stream = StringIO(snapshot.decode("utf-8-sig"), newline="")
        if args.input_csv:
            _, _, patients = read_patient_csv(stream)
        else:
            records = read_patient_json(stream)
            patients = records if isinstance(records, list) else [records]
        report = verify_receipt(receipt, patients, args.model, source_sha256=hashlib.sha256(snapshot).hexdigest())
        payload = json.dumps(report, indent=2) + "\n"
        if args.output:
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, AttributeError, EOFError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
