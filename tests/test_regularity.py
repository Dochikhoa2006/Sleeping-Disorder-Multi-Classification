from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.regularity import clock_distance, schedule_regularity


class RegularityTests(unittest.TestCase):
    def night(self, day, bedtime="23:00", wake="07:00"):
        return dict(date=day, bedtime=bedtime, wake_time=wake, awake_minutes="0", quality="")

    def test_clock_distances_wrap_midnight_and_are_unsigned(self):
        self.assertEqual(clock_distance("23:45", "00:15"), 30)
        self.assertEqual(clock_distance("00:15", "23:45"), 30)
        self.assertEqual(clock_distance("00:00", "12:00"), 720)
        self.assertEqual(clock_distance("07:00", "07:00"), 0)

    def test_adjacent_pairs_unsorted_records_and_boundary_exclusion(self):
        rows = [self.night("2026-11-03", "00:30", "08:00"),
                self.night("2026-11-01", "23:30", "07:00"),
                self.night("2026-11-02", "00:00", "07:20"),
                self.night("2026-10-31"), self.night("2026-11-04")]
        report = schedule_regularity(rows, date(2026, 11, 3), 3)
        self.assertEqual(report["recorded_nights"], 3)
        self.assertEqual(report["consecutive_night_pairs"], 2)
        self.assertEqual(report["pair_coverage"], 1)
        self.assertEqual(report["bedtime"]["mean_absolute_change_minutes"], 30)
        self.assertEqual(report["wake_time"]["mean_absolute_change_minutes"], 30)
        self.assertEqual(report["wake_time"]["max_absolute_change_minutes"], 40)

    def test_gaps_do_not_form_pairs_and_absence_is_not_zero(self):
        rows = [self.night("2026-11-01"), self.night("2026-11-03")]
        report = schedule_regularity(rows, date(2026, 11, 3), 3)
        self.assertEqual(report["unrecorded_nights"], 1)
        self.assertEqual(report["consecutive_night_pairs"], 0)
        self.assertIsNone(report["bedtime"]["mean_absolute_change_minutes"])
        self.assertEqual(report["pair_coverage"], 0)
        single = schedule_regularity([], date(2026, 11, 3), 1)
        self.assertIsNone(single["pair_coverage"])
        json.dumps(single, allow_nan=False)

    def test_duplicates_invalid_period_and_dates_rejected(self):
        for days in (0, 367, True, 1.5):
            with self.subTest(days=days), self.assertRaises(ValueError):
                schedule_regularity([], date(2026, 11, 3), days)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            schedule_regularity([self.night("2026-11-03")] * 2, date(2026, 11, 3))
        with self.assertRaises(OverflowError):
            schedule_regularity([], date.min)

    def test_stdlib_cli_missing_diary_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "missing.csv"
            result = subprocess.run([sys.executable, "-S", "-m", "src.diary", "--file", str(source),
                                     "regularity", "--end", "2026-11-03", "--days", "3"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["recorded_nights"], 0)
            self.assertEqual(report["possible_night_pairs"], 2)
            self.assertFalse(source.exists())
