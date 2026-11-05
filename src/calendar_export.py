"""Export local diary records as RFC 5545 calendar events."""

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from src.diary import clock_minutes, night_metrics, parse_date


def escape_text(value: str) -> str:
    return (value.replace("\\", "\\\\").replace("\r\n", "\n").replace("\r", "\n")
            .replace("\n", "\\n").replace(";", "\\;").replace(",", "\\,"))


def fold_line(line: str) -> str:
    """Fold at 75 UTF-8 octets without splitting a multibyte character."""
    pieces, current, length = [], "", 0
    for character in line:
        size = len(character.encode("utf-8"))
        if length + size > 75:
            pieces.append(current)
            current, length = " ", 1
        current += character
        length += size
    pieces.append(current)
    return "\r\n".join(pieces)


def render_calendar(records: list[dict], *, calendar_id: str = "sleep-diary",
                    timestamp: datetime | None = None) -> str:
    """Use floating local clock times; do not infer a time zone from diary data."""
    stamp = timestamp or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Calendar export timestamp must include a time zone.")
    stamp_text = stamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Sleep and Lifestyle//Diary Export//EN", "CALSCALE:GREGORIAN"]
    seen = set()
    for record in sorted(records, key=lambda row: row["date"]):
        metrics = night_metrics(record)
        if record["date"] in seen:
            raise ValueError("Calendar export cannot contain duplicate diary dates.")
        seen.add(record["date"])
        start = parse_date(record["date"])
        try:
            end = start + timedelta(days=int(clock_minutes(record["wake_time"]) < clock_minutes(record["bedtime"])))
        except OverflowError as error:
            raise ValueError("An overnight event extends beyond the supported date range.") from error
        start_text = start.isoformat().replace("-", "") + "T" + record["bedtime"].replace(":", "") + "00"
        end_text = end.isoformat().replace("-", "") + "T" + record["wake_time"].replace(":", "") + "00"
        sleep = metrics["estimated_sleep_minutes"] / 60
        quality = f'{metrics["quality"]}/10' if metrics["quality"] is not None else 'Not rated'
        description = (f'Time in bed: {metrics["time_in_bed_minutes"] / 60:.2f} h\n'
                       f'Estimated sleep: {sleep:.2f} h\nAwake minutes: {record["awake_minutes"]}\n'
                       f'Quality: {quality}\nSelf-reported wall-clock estimates.')
        uid = uuid5(NAMESPACE_URL, f"urn:sleep-diary:{calendar_id}:{record['date']}")
        lines.extend(["BEGIN:VEVENT", f"UID:{uid}@sleep-diary.local", f"DTSTAMP:{stamp_text}",
                      f"DTSTART:{start_text}", f"DTEND:{end_text}",
                      "SUMMARY:" + escape_text(f"Sleep diary: {sleep:.2f} h estimated sleep"),
                      "DESCRIPTION:" + escape_text(description), "CLASS:PRIVATE", "TRANSP:TRANSPARENT", "END:VEVENT"])
    lines.append("END:VCALENDAR")
    return "\r\n".join(fold_line(line) for line in lines) + "\r\n"
