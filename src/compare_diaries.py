"""Preview differences between diary CSV snapshots without changing either file."""

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

from src.diary import COLUMNS, read_diary
from src.output import atomic_text_output


def compare_diaries(baseline: bytes, candidate: bytes) -> dict:
    batches = [{row["date"]: row for row in read_diary(StringIO(snapshot.decode("utf-8-sig"), newline=""))}
               for snapshot in (baseline, candidate)]
    before, after = batches
    added, removed, changed, unchanged = [], [], [], []
    for day in sorted(set(before) | set(after)):
        if day not in before:
            added.append(day)
        elif day not in after:
            removed.append(day)
        else:
            fields = [field for field in COLUMNS if field != "date" and before[day][field] != after[day][field]]
            if fields:
                changed.append({"date": day, "fields": fields})
            else:
                unchanged.append(day)
    return {"baseline_sha256": hashlib.sha256(baseline).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
            "baseline_nights": len(before), "candidate_nights": len(after),
            "counts": {"added": len(added), "removed": len(removed),
                       "changed": len(changed), "unchanged": len(unchanged)},
            "added_dates": added, "removed_dates": removed, "changes": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Atomic JSON report; omit for standard output.")
    parser.add_argument("--fail-on-change", action="store_true")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in {args.baseline.resolve(), args.candidate.resolve()}:
            raise ValueError("Output must differ from both diary paths.")
        report = compare_diaries(args.baseline.read_bytes(), args.candidate.read_bytes())
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if args.fail_on_change and any(report["counts"][name] for name in ("added", "removed", "changed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
