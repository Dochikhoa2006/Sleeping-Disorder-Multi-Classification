"""Compare saved prediction CSV exports by Record ID without loading a model."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH
from src.prediction_diff import compare_prediction_exports


def compare_files(baseline: Path, candidate: Path, output: Path | None = None) -> dict:
    if output is not None and output.resolve() in {
        baseline.resolve(), candidate.resolve(), DEFAULT_MODEL_PATH.resolve()
    }:
        raise ValueError("Output must differ from both inputs and the shipped model path.")
    before, after = baseline.read_bytes(), candidate.read_bytes()
    report = compare_prediction_exports(before, after)
    report.update(baseline_sha256=hashlib.sha256(before).hexdigest(),
                  candidate_sha256=hashlib.sha256(after).hexdigest())
    if output is not None:
        with atomic_text_output(output) as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Atomic JSON report; omit for standard output.")
    parser.add_argument("--fail-on-change", action="store_true",
                        help="Exit 1 after reporting any input, prediction, added, or removed record change.")
    args = parser.parse_args()
    try:
        report = compare_files(args.baseline, args.candidate, args.output)
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if args.output is None:
        print(json.dumps(report, indent=2, allow_nan=False))
    if args.fail_on_change and any(count for status, count in report["counts"].items() if status != "unchanged"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
