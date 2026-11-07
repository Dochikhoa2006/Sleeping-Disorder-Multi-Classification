"""Keep a local sleep diary independently of the classification model."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import json
import hashlib
from html import escape
from string import Template
from pathlib import Path
from typing import TextIO
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
        return read_diary(source)


def read_diary(source: TextIO) -> list[dict]:
    """Read and validate a complete diary CSV, including duplicate dates."""
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


def merge_nights(path: Path, incoming: list[dict], *, replace: bool = False) -> dict:
    """Validate and merge a backup in one atomic write; reject conflicts by default."""
    if not incoming:
        raise ValueError("The backup must contain at least one night.")
    additions = {}
    for record in incoming:
        if set(record) != set(COLUMNS):
            raise ValueError("Imported records must contain exactly the diary columns.")
        night_metrics(record)
        if record["date"] in additions:
            raise ValueError("Duplicate night date in backup.")
        additions[record["date"]] = record
    existing = {record["date"]: record for record in load_diary(path)}
    conflicts = set(existing) & set(additions)
    if conflicts and not replace:
        raise ValueError(f"{len(conflicts)} date(s) already exist. Choose replacement explicitly or use --replace.")
    merged = {**existing, **additions}
    with atomic_text_output(path, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))
    return {"added": len(additions) - len(conflicts), "replaced": len(conflicts), "total": len(merged)}


def night_fingerprint(record: dict) -> str:
    canonical = json.dumps({key: str(record[key]) for key in COLUMNS}, sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def remove_night(path: Path, day: str, *, expected_fingerprint: str | None = None) -> dict:
    """Remove exactly one date, rejecting stale browser confirmations."""
    parse_date(day)
    records = load_diary(path)
    selected = next((record for record in records if record["date"] == day), None)
    if selected is None:
        raise ValueError("No diary entry exists for this date.")
    if expected_fingerprint is not None and night_fingerprint(selected) != expected_fingerprint:
        raise ValueError("This entry changed since you reviewed it. Review the removal again.")
    remaining = [record for record in records if record["date"] != day]
    with atomic_text_output(path, newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(remaining, key=lambda record: record["date"]))
    return {"removed_date": day, "remaining_entries": len(remaining)}


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


def render_report(report: dict) -> str:
    """Create a portable, script-free chart with an explicit row for missing nights."""
    nights = {row["date"]: row for row in report["nights"]}
    start, end = parse_date(report["start_date"]), parse_date(report["end_date"])
    rows = []
    current = start
    while current <= end:
        day = current.isoformat()
        night = nights.get(day)
        if night is None:
            details = '<td class="missing" colspan="3">Not recorded</td>'
        else:
            sleep = night["estimated_sleep_minutes"]
            bed = night["time_in_bed_minutes"]
            quality = str(night["quality"]) if night["quality"] is not None else "Not rated"
            details = (f'<td><div class="track" aria-hidden="true"><span class="bed" style="width:{bed / 1440 * 100:.4f}%"></span>'
                       f'<span class="sleep" style="width:{sleep / 1440 * 100:.4f}%"></span></div>'
                       f'<span>{sleep / 60:.2f} h estimated sleep</span></td>'
                       f'<td>{bed / 60:.2f} h</td><td>{escape(quality)}</td>')
        rows.append(f'<tr><th scope="row"><time datetime="{day}">{day}</time></th>{details}</tr>')
        if current == end:
            break
        current += timedelta(days=1)
    average = report["average_estimated_sleep_hours"]
    quality = report["average_quality"]
    return Template(Path(__file__).with_name("diary_report.html").read_text(encoding="utf-8")).substitute(
        start=escape(report["start_date"]), end=escape(report["end_date"]),
        recorded=report["recorded_nights"], missing=report["unrecorded_nights"],
        average=f"{average:.2f} h" if average is not None else "No entries",
        quality=f"{quality:.2f} / 10" if quality is not None else "Not rated",
        rows="".join(rows))


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
    summary.add_argument("--html", type=Path, help="Also save a printable, self-contained HTML chart.")
    export = commands.add_parser("export-calendar", help="Save diary nights as an iCalendar .ics file.")
    export.add_argument("--output", type=Path, required=True)
    restore = commands.add_parser("import", help="Merge a diary CSV backup into the local diary.")
    restore.add_argument("--input-csv", type=Path, required=True)
    restore.add_argument("--replace", action="store_true", help="Replace matching dates; otherwise any conflict rejects the import.")
    remove = commands.add_parser("remove", help="Remove one diary night by its bedtime date.")
    remove.add_argument("--date", required=True)
    args = parser.parse_args()
    try:
        if args.command == "add":
            record = {"date": args.date, "bedtime": args.bedtime, "wake_time": args.wake_time,
                      "awake_minutes": str(args.awake_minutes),
                      "quality": str(args.quality) if args.quality is not None else ""}
            add_night(args.file, record, replace=args.replace)
            print(f"Saved night {args.date} to {args.file}")
        elif args.command == "remove":
            print(json.dumps(remove_night(args.file, args.date), indent=2))
        elif args.command == "import":
            if args.input_csv.resolve() == args.file.resolve():
                raise ValueError("Backup input must differ from the destination diary.")
            with args.input_csv.open(encoding="utf-8-sig", newline="") as source:
                incoming = read_diary(source)
            result = merge_nights(args.file, incoming, replace=args.replace)
            print(json.dumps(result, indent=2))
        elif args.command == "export-calendar":
            from src.calendar_export import render_calendar
            if args.output.resolve() == args.file.resolve():
                raise ValueError("Calendar output must differ from the diary path.")
            payload = render_calendar(load_diary(args.file), calendar_id=str(args.file.resolve()))
            with atomic_text_output(args.output, newline="") as output:
                output.write(payload)
            print(f"Saved calendar to {args.output}")
        else:
            if args.html and args.html.resolve() == args.file.resolve():
                raise ValueError("HTML report path must differ from the diary path.")
            report = summarize(load_diary(args.file), parse_date(args.end), args.days)
            if args.html:
                with atomic_text_output(args.html) as output:
                    output.write(render_report(report))
            print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, csv.Error, OverflowError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
