from contextlib import redirect_stdout
from datetime import date
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.diary import add_night, load_diary, main, night_metrics, summarize


class DiaryTests(unittest.TestCase):
    def night(self, **changes):
        return {"date": "2026-11-01", "bedtime": "23:00", "wake_time": "07:00",
                "awake_minutes": "30", "quality": "8", **changes}

    def test_overnight_and_same_day_intervals(self):
        self.assertEqual(night_metrics(self.night()), {"time_in_bed_minutes": 480, "estimated_sleep_minutes": 450, "quality": 8})
        self.assertEqual(night_metrics(self.night(bedtime="13:00", wake_time="14:30", quality=""))["estimated_sleep_minutes"], 60)

    def test_invalid_entries_never_create_a_file(self):
        changes = [{"date": "2026-02-30"}, {"date": "20261101"}, {"bedtime": "24:00"},
                   {"wake_time": "23:00"}, {"awake_minutes": "481"}, {"awake_minutes": "-1"},
                   {"quality": "11"}, {"quality": "2.5"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    add_night(path, self.night(**change))
                self.assertFalse(path.exists())

    def test_add_sorts_and_requires_explicit_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            add_night(path, self.night())
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, '--replace'):
                add_night(path, self.night(quality="2"))
            self.assertEqual(path.read_bytes(), before)
            add_night(path, self.night(date="2026-10-31"))
            add_night(path, self.night(quality="2"), replace=True)
            self.assertEqual([row['date'] for row in load_diary(path)], ['2026-10-31', '2026-11-01'])
            self.assertEqual(load_diary(path)[1]['quality'], '2')

    def test_summary_uses_recorded_nights_and_optional_ratings(self):
        rows = [self.night(date="2026-10-25"), self.night(), self.night(date="2026-11-02", quality="", awake_minutes="0")]
        report = summarize(rows, date(2026, 11, 2), 7)
        self.assertEqual(report['recorded_nights'], 2)
        self.assertEqual(report['unrecorded_nights'], 5)
        self.assertEqual(report['average_estimated_sleep_hours'], 7.75)
        self.assertEqual(report['average_quality'], 8)

    def test_missing_diary_summary_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.csv'
            report = summarize(load_diary(path), date(2026, 11, 2))
            self.assertIsNone(report['average_estimated_sleep_hours'])
            self.assertEqual(report['unrecorded_nights'], 7)
            self.assertFalse(path.exists())

    def test_corrupt_existing_diary_is_never_overwritten(self):
        for contents in ('wrong,header\n', 'date,bedtime,wake_time,awake_minutes,quality\n2026-11-01,23:00,07:00,0\n'):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'diary.csv'
                path.write_text(contents)
                with self.assertRaises(ValueError):
                    add_night(path, self.night())
                self.assertEqual(path.read_text(), contents)

    def test_cli_add_then_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'diary.csv')
            with patch('sys.argv', ['diary', '--file', path, 'add', '--date', '2026-11-01', '--bedtime', '23:00', '--wake-time', '07:00']), redirect_stdout(StringIO()):
                main()
            with patch('sys.argv', ['diary', '--file', path, 'summary', '--end', '2026-11-02']), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(json.loads(output.getvalue())['recorded_nights'], 1)

    def test_summary_rejects_invalid_period(self):
        for days in (0, 367):
            with self.assertRaises(ValueError):
                summarize([], date(2026, 11, 2), days)
