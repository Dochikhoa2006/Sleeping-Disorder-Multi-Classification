"""Audit shared patient profiles across two CSV datasets without loading a model."""

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


def audit_overlap(reference: bytes, current: bytes) -> dict:
    profiles, summaries = [], []
    for snapshot in (reference, current):
        _, _, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
        fingerprints = feature_fingerprints(pd.DataFrame(patients)).tolist()
        profiles.append(fingerprints)
        summaries.append({"sha256": hashlib.sha256(snapshot).hexdigest(), "rows": len(fingerprints),
                          "unique_profiles": len(set(fingerprints)),
                          "duplicate_rows": len(fingerprints) - len(set(fingerprints))})
    first, second = map(set, profiles)
    shared = first & second
    for fingerprints, summary in zip(profiles, summaries, strict=True):
        shared_rows = sum(value in shared for value in fingerprints)
        summary.update(shared_rows=shared_rows, shared_row_fraction=shared_rows / len(fingerprints),
                       exclusive_unique_profiles=summary["unique_profiles"] - len(shared))
    return {"reference": summaries[0], "current": summaries[1],
            "shared_unique_profiles": len(shared), "union_unique_profiles": len(first | second),
            "profile_jaccard_similarity": len(shared) / len(first | second)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Atomic JSON report; omit for standard output.")
    parser.add_argument("--fail-on-overlap", action="store_true",
                        help="Exit 1 after reporting any shared profile.")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in {
            args.reference.resolve(), args.current.resolve(), DEFAULT_MODEL_PATH.resolve()
        }:
            raise ValueError("Output must differ from both inputs and the shipped model path.")
        report = audit_overlap(args.reference.read_bytes(), args.current.read_bytes())
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if args.fail_on_overlap and report["shared_unique_profiles"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
