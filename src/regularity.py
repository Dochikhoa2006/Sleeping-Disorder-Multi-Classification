"""Describe adjacent-night clock changes without interpreting them medically."""

from datetime import date, timedelta

from src.diary import clock_minutes, night_metrics, parse_date


def clock_distance(first: str, second: str) -> int:
    """Shortest unsigned distance on a 24-hour clock, in minutes."""
    difference = abs(clock_minutes(first) - clock_minutes(second))
    return min(difference, 1440 - difference)


def schedule_regularity(records: list[dict], end: date, days: int = 14) -> dict:
    if type(days) is not int or not 1 <= days <= 366:
        raise ValueError("Choose a regularity period from 1 to 366 days.")
    start = end - timedelta(days=days - 1)
    selected = {}
    for row in records:
        day = parse_date(row["date"])
        if start <= day <= end:
            night_metrics(row)
            if day in selected:
                raise ValueError("Duplicate night date.")
            selected[day] = row
    changes = {"bedtime": [], "wake_time": []}
    previous = None
    for day in sorted(selected):
        if previous is not None and (day - previous).days == 1:
            for field in changes:
                changes[field].append(clock_distance(selected[previous][field], selected[day][field]))
        previous = day
    def describe(values):
        return {"mean_absolute_change_minutes": sum(values) / len(values) if values else None,
                "max_absolute_change_minutes": max(values) if values else None}
    pairs = len(changes["bedtime"])
    return {"start_date": start.isoformat(), "end_date": end.isoformat(),
            "recorded_nights": len(selected), "unrecorded_nights": days - len(selected),
            "consecutive_night_pairs": pairs, "possible_night_pairs": days - 1,
            "pair_coverage": pairs / (days - 1) if days > 1 else None,
            **{field: describe(values) for field, values in changes.items()}}
