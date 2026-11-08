from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.diary import compare_periods, main


class DiaryComparisonTests(unittest.TestCase):
    def night(self, day, awake="0", quality="", wake="07:00"):
        return dict(date=day, bedtime="23:00", wake_time=wake,
                    awake_minutes=awake, quality=quality)

    def test_adjacent_windows_and_signed_changes(self):
        rows = [self.night("2026-10-25", "60", "6"),
                self.night("2026-11-01", "0", "8"),
                self.night("2026-11-02", "0"),
                self.night("2026-10-24"), self.night("2026-11-08")]
        report = compare_periods(rows, date(2026, 11, 7))
        self.assertEqual(report["previous"]["start_date"], "2026-10-25")
        self.assertEqual(report["previous"]["end_date"], "2026-10-31")
        self.assertEqual(report["current"]["recorded_nights"], 2)
        self.assertEqual(report["current"]["unrecorded_nights"], 5)
        self.assertEqual(report["current"]["rated_nights"], 1)
        self.assertEqual(report["change_current_minus_previous"], {
            "average_estimated_sleep_hours": 1, "average_quality": 2,
            "average_awake_minutes": -60, "average_sleep_efficiency_percent": 12.5})

    def test_empty_period_and_unrated_quality_are_null(self):
        report = compare_periods([self.night("2026-11-01")], date(2026, 11, 1), 1)
        self.assertEqual(report["current"]["average_estimated_sleep_hours"], 8)
        self.assertIsNone(report["current"]["average_quality"])
        self.assertTrue(all(value is None for value in report["change_current_minus_previous"].values()))
        json.dumps(report, allow_nan=False)

    def test_efficiency_is_mean_of_nightly_percentages(self):
        rows = [self.night("2026-11-01", "60"), self.night("2026-11-02", "60", wake="03:00")]
        report = compare_periods(rows, date(2026, 11, 2), 2)
        self.assertEqual(report["current"]["average_sleep_efficiency_percent"], 81.25)

    def test_invalid_period_and_calendar_underflow(self):
        for days in (0, 367, True, 1.5):
            with self.subTest(days=days), self.assertRaises(ValueError):
                compare_periods([], date(2026, 11, 1), days)
        with self.assertRaises(OverflowError):
            compare_periods([], date.min, 1)

    def test_cli_missing_diary_is_read_only_and_stdlib_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent.csv"
            result = subprocess.run([sys.executable, "-S", "-m", "src.diary", "--file", str(path),
                                     "compare", "--end", "2026-11-01", "--days", "1"],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["previous"]["end_date"], "2026-10-31")
            self.assertFalse(path.exists())

    def test_cli_date_underflow_has_clean_error(self):
        with patch("sys.argv", ["diary", "compare", "--end", "0001-01-01"]):
            with self.assertRaisesRegex(SystemExit, "Error:"):
                main()
