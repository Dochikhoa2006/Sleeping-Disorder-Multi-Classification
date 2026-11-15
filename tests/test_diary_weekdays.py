from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.diary import weekday_summary


class DiaryWeekdayTests(unittest.TestCase):
    def night(self, day, awake="0", quality=""):
        return dict(date=day, bedtime="23:00", wake_time="07:00", awake_minutes=awake, quality=quality)

    def test_bedtime_weekday_means_exclude_missing_nights_and_ratings(self):
        rows = [self.night("2026-11-02", "60", "8"), self.night("2026-11-09"),
                self.night("2026-10-26"), self.night("2026-11-16")]
        report = weekday_summary(rows, date(2026, 11, 15), 14)
        monday = report["weekdays"][0]
        self.assertEqual(monday["weekday"], "Monday")
        self.assertEqual(monday["possible_nights"], 2)
        self.assertEqual(monday["recorded_nights"], 2)
        self.assertEqual(monday["average_estimated_sleep_hours"], 7.5)
        self.assertEqual(monday["average_quality"], 8)
        self.assertEqual(monday["rated_nights"], 1)
        self.assertIsNone(report["weekdays"][1]["average_estimated_sleep_hours"])
        self.assertEqual(report["unrecorded_nights"], 12)

    def test_partial_week_possible_counts_and_boundary_date(self):
        report = weekday_summary([], date(2026, 11, 9), 3)
        self.assertEqual([group["possible_nights"] for group in report["weekdays"]], [1, 0, 0, 0, 0, 1, 1])
        self.assertEqual(sum(group["unrecorded_nights"] for group in report["weekdays"]), 3)
        self.assertEqual(weekday_summary([], date.min, 1)["weekdays"][0]["possible_nights"], 1)
        json.dumps(report, allow_nan=False)

    def test_invalid_period_duplicate_dates_and_underflow(self):
        for days in (0, 367, True, 1.5):
            with self.subTest(days=days), self.assertRaises(ValueError):
                weekday_summary([], date(2026, 11, 15), days)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            weekday_summary([self.night("2026-11-09")] * 2, date(2026, 11, 15))
        with self.assertRaises(OverflowError):
            weekday_summary([], date.min)

    def test_stdlib_cli_missing_diary_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            result = subprocess.run([sys.executable, "-S", "-m", "src.diary", "--file", str(path),
                                     "weekdays", "--end", "2026-11-15"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(len(report["weekdays"]), 7)
            self.assertEqual(report["unrecorded_nights"], 28)
            self.assertFalse(path.exists())
