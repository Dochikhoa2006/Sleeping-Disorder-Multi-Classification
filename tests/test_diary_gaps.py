from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.diary import diary_gaps


class DiaryGapTests(unittest.TestCase):
    def night(self, day):
        return dict(date=day, bedtime="23:00", wake_time="07:00", awake_minutes="0", quality="")

    def test_missing_ranges_and_streaks_use_only_selected_dates(self):
        rows = [self.night(f"2026-11-{day:02d}") for day in (2, 3, 6, 7, 8)]
        report = diary_gaps(rows, date(2026, 11, 7), 7)
        self.assertEqual(report["missing_ranges"], [
            {"start_date": "2026-11-01", "end_date": "2026-11-01", "nights": 1},
            {"start_date": "2026-11-04", "end_date": "2026-11-05", "nights": 2}])
        self.assertEqual(report["longest_recorded_streak"], 2)
        self.assertEqual(report["ending_recorded_streak"], 2)
        self.assertEqual(report["longest_missing_streak"], 2)
        self.assertEqual(report["recorded_fraction"], 4 / 7)

    def test_empty_complete_and_supported_date_boundaries(self):
        empty = diary_gaps([], date(2026, 11, 7), 7)
        self.assertEqual(empty["missing_ranges"], [{"start_date": "2026-11-01", "end_date": "2026-11-07", "nights": 7}])
        self.assertEqual(empty["ending_recorded_streak"], 0)
        rows = [self.night(f"2026-11-{day:02d}") for day in range(1, 8)]
        complete = diary_gaps(rows, date(2026, 11, 7), 7)
        self.assertEqual(complete["missing_ranges"], [])
        self.assertEqual(complete["ending_recorded_streak"], 7)
        self.assertEqual(complete["longest_missing_streak"], 0)
        for boundary in (date.min, date.max):
            self.assertEqual(diary_gaps([], boundary, 1)["unrecorded_nights"], 1)

    def test_invalid_period_and_duplicate_entries_rejected(self):
        for days in (0, 367, True, 1.5):
            with self.subTest(days=days), self.assertRaises(ValueError):
                diary_gaps([], date(2026, 11, 7), days)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            diary_gaps([self.night("2026-11-07")] * 2, date(2026, 11, 7))

    def test_stdlib_cli_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            result = subprocess.run([sys.executable, "-S", "-m", "src.diary", "--file", str(path),
                                     "gaps", "--end", "2026-11-07", "--days", "7"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["longest_missing_streak"], 7)
            self.assertFalse(path.exists())
