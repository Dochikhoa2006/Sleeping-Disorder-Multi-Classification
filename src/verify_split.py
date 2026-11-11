"""Verify an exported train/test split and optionally its original dataset."""

import argparse
from collections import Counter
import hashlib
from io import StringIO
import json
from pathlib import Path
import csv

import pandas as pd

from src.evaluation import feature_fingerprints
from src.predict import _unique_json_object, read_patient_csv
from src.preprocessing import TARGET_COLUMN
from src.split_data import LABELS


def verify_split(directory: Path, source: Path | None = None) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"),
                          object_pairs_hook=_unique_json_object)
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("partitions"), dict)
            or any(not isinstance(manifest["partitions"].get(name), dict) for name in ("train", "test"))):
        raise ValueError("Manifest must describe train and test partitions.")
    checks, partitions, groups = {}, {}, {}
    for name in ("train", "test"):
        snapshot = (directory / f"{name}.csv").read_bytes()
        columns, rows, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
        if TARGET_COLUMN not in columns or any(row[TARGET_COLUMN] not in LABELS for row in rows):
            raise ValueError("Each split record must have an explicit valid sleep-disorder label.")
        fingerprints = feature_fingerprints(pd.DataFrame(patients))
        actual = {"sha256": hashlib.sha256(snapshot).hexdigest(), "rows": len(rows),
                  "unique_profiles": int(fingerprints.nunique()),
                  "class_counts": dict(sorted(Counter(row[TARGET_COLUMN] for row in rows).items()))}
        expected = manifest["partitions"][name]
        checks[name] = {key: type(expected.get(key)) is type(value) and expected[key] == value
                        for key, value in actual.items()}
        expected_counts = expected.get("class_counts")
        if isinstance(expected_counts, dict) and any(type(count) is not int for count in expected_counts.values()):
            checks[name]["class_counts"] = False
        partitions[name] = (columns, rows)
        groups[name] = set(fingerprints)
    checks["matching_columns"] = partitions["train"][0] == partitions["test"][0]
    shared = len(groups["train"] & groups["test"])
    checks["profile_disjoint"] = shared == 0
    checks["source_hash"] = None
    checks["source_rows_preserved"] = None
    if source is not None:
        snapshot = source.read_bytes()
        columns, rows, _ = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
        checks["source_hash"] = hashlib.sha256(snapshot).hexdigest() == manifest.get("source_sha256")
        combined = partitions["train"][1] + partitions["test"][1]
        # Compare multisets, so duplicate rows must also occur the same number of times.
        checks["source_rows_preserved"] = (
            columns == partitions["train"][0] == partitions["test"][0]
            and Counter(tuple(row[column] for column in columns) for row in rows)
            == Counter(tuple(row[column] for column in columns) for row in combined))
    verdicts = [value for name, value in checks.items() if name not in ("train", "test")]
    verdicts += [value for name in ("train", "test") for value in checks[name].values()]
    return {"verified": all(value is not False for value in verdicts), "checks": checks,
            "shared_unique_profiles": shared, "source_checked": source is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Also verify the original CSV hash and exact record multiset.")
    args = parser.parse_args()
    try:
        report = verify_split(args.directory, args.source)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
