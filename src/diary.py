"""Keep a local sleep diary independently of the classification model."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import json
from pathlib import Path
import re

from src.output import atomic_text_output

DEFAULT_DIARY = Path(__file__).resolve().parents[1] / "data" / "sleep_diary.csv"
COLUMNS = ["date", "bedtime", "wake_time", "awake_minutes", "quality"]


def parse_date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD.")
    return date.fromisoformat(value)


def clock_minutes(value: str) -> int:
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("Times must use 24-hour HH:MM.")
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def night_metrics(record: dict) -> dict:
    """Calculate wall-clock intervals; no time-zone or DST adjustment is inferred."""
    parse_date(record["date"])
    start, stop = clock_minutes(record["bedtime"]), clock_minutes(record["wake_time"])
    minutes = (stop - start) % (24 * 60)
    if not minutes:
        raise ValueError("Bedtime and wake time must differ; intervals must be under 24 hours.")
    awake = int(record["awake_minutes"])
    if not 0 <= awake <= minutes:
        raise ValueError("Awake minutes must be between zero and the time in bed.")
    quality = int(record["quality"]) if record["quality"] != "" else None
    if quality is not None and not 1 <= quality <= 10:
        raise ValueError("Quality must be an integer from 1 to 10.")
    return {"time_in_bed_minutes": minutes, "estimated_sleep_minutes": minutes - awake,
            "quality": quality}


def load_diary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Not a sleep diary: expected columns " + ", ".join(COLUMNS))
        records = []
        seen = set()
        for index, record in enumerate(reader, start=2):
            if None in record or any(value is None for value in record.values()):
                raise ValueError(f"Diary row {index} has the wrong number of fields.")
            try:
                night_metrics(record)
                if record["date"] in seen:
                    raise ValueError("Duplicate night date.")
            except (ValueError, TypeError) as error:
                raise ValueError(f"Diary row {index}: {error}") from error
            seen.add(record["date"])
            records.append(record)
    return records


def add_night(path: Path, record: dict, *, replace: bool = False) -> None:
    night_metrics(record)
    records = load_diary(path)
    if any(row["date"] == record["date"] for row in records) and not replace:
        raise ValueError("This night is already recorded; use --replace to edit it.")
    records = [row for row in records if row["date"] != record["date"]] + [record]
    with atomic_text_output(path, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["date"]))


def summarize(records: list[dict], end: date, days: int = 7) -> dict:
    if not 1 <= days <= 366:
        raise ValueError("Choose a summary period from 1 to 366 days.")
    start = end - timedelta(days=days - 1)
    selected = sorted((row for row in records if start <= parse_date(row["date"]) <= end),
                      key=lambda row: row["date"])
    nights = [{"date": row["date"], **night_metrics(row)} for row in selected]
    qualities = [night["quality"] for night in nights if night["quality"] is not None]
    return {"start_date": start.isoformat(), "end_date": end.isoformat(),
            "recorded_nights": len(nights), "unrecorded_nights": days - len(nights),
            "average_estimated_sleep_hours": (sum(n["estimated_sleep_minutes"] for n in nights) / len(nights) / 60) if nights else None,
            "average_quality": sum(qualities) / len(qualities) if qualities else None,
            "nights": nights}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_DIARY, help="Local diary CSV (default: data/sleep_diary.csv).")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Record one night, dated by when you went to bed.")
    add.add_argument("--date", required=True)
    add.add_argument("--bedtime", required=True)
    add.add_argument("--wake-time", required=True)
    add.add_argument("--awake-minutes", type=int, default=0, help="Estimated total awake time in bed, including time to fall asleep.")
    add.add_argument("--quality", type=int, help="Optional personal rating from 1 to 10.")
    add.add_argument("--replace", action="store_true")
    summary = commands.add_parser("summary", help="Print a date-range summary as JSON.")
    summary.add_argument("--end", default=date.today().isoformat())
    summary.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    try:
        if args.command == "add":
            record = {"date": args.date, "bedtime": args.bedtime, "wake_time": args.wake_time,
                      "awake_minutes": str(args.awake_minutes),
                      "quality": str(args.quality) if args.quality is not None else ""}
            add_night(args.file, record, replace=args.replace)
            print(f"Saved night {args.date} to {args.file}")
        else:
            print(json.dumps(summarize(load_diary(args.file), parse_date(args.end), args.days), indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, csv.Error, OverflowError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
