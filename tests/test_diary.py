from contextlib import redirect_stdout
from datetime import date
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.diary import add_night, load_diary, main, night_metrics, summarize, render_report, merge_nights, read_diary, remove_night, night_fingerprint


class DiaryTests(unittest.TestCase):
    def test_write_helpers_reject_lossy_integer_values_before_changing_files(self):
        for field in ("awake_minutes", "quality"):
            for value in (1.5, 1.0, True, False, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "diary.csv"
                    add_night(path, self.night())
                    original = path.read_bytes()
                    invalid = self.night(date="2026-11-02", **{field: value})
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        add_night(path, invalid)
                    self.assertEqual(path.read_bytes(), original)
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        merge_nights(path, [invalid])
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(len(load_diary(path)), 1)

    def test_integer_values_and_integer_strings_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diary.csv"
            for index, (awake, quality) in enumerate(((30, 8), (" 030 ", "+8"), ("+30", "08")), start=1):
                add_night(path, self.night(date=f"2026-11-0{index}", awake_minutes=awake, quality=quality))
            for row in load_diary(path):
                self.assertEqual(night_metrics(row)["estimated_sleep_minutes"], 450)
                self.assertEqual(night_metrics(row)["quality"], 8)

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

    def test_html_report_shows_missing_nights_and_exact_values(self):
        report = summarize([self.night()], date(2026, 11, 2), 3)
        html = render_report(report)
        self.assertEqual(html.count('Not recorded</td>'), 2)
        self.assertIn('7.50 h estimated sleep', html)
        self.assertIn('8.00 h</td>', html)
        self.assertIn('width:31.2500%', html)
        self.assertIn('width:33.3333%', html)
        self.assertIn('1 recorded', html)
        self.assertNotIn('<script', html)
        self.assertNotIn('https://', html)

    def test_empty_report_has_no_invented_zero_sleep(self):
        html = render_report(summarize([], date(2026, 11, 2), 7))
        self.assertIn('No entries', html)
        self.assertEqual(html.count('Not recorded</td>'), 7)
        self.assertNotIn('0.00 h estimated sleep', html)

    def test_cli_html_export_preserves_json_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            diary = Path(directory) / 'diary.csv'
            output = Path(directory) / 'report.html'
            add_night(diary, self.night())
            with patch('sys.argv', ['diary', '--file', str(diary), 'summary', '--end', '2026-11-02', '--html', str(output)]), redirect_stdout(StringIO()) as stdout:
                main()
            self.assertEqual(json.loads(stdout.getvalue())['recorded_nights'], 1)
            self.assertIn('7.50 h estimated sleep', output.read_text())

    def test_html_export_cannot_overwrite_diary(self):
        with tempfile.TemporaryDirectory() as directory:
            diary = Path(directory) / 'diary.csv'
            add_night(diary, self.night())
            before = diary.read_bytes()
            with patch('sys.argv', ['diary', '--file', str(diary), 'summary', '--html', str(diary)]):
                with self.assertRaisesRegex(SystemExit, 'differ'):
                    main()
            self.assertEqual(diary.read_bytes(), before)

    def test_merge_adds_new_dates_preserves_existing_and_requires_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            add_night(path, self.night())
            before = path.read_bytes()
            incoming = [self.night(quality='2'), self.night(date='2026-11-02')]
            with self.assertRaisesRegex(ValueError, '--replace'):
                merge_nights(path, incoming)
            self.assertEqual(path.read_bytes(), before)
            result = merge_nights(path, incoming, replace=True)
            self.assertEqual(result, {'added': 1, 'replaced': 1, 'total': 2})
            self.assertEqual(load_diary(path)[0]['quality'], '2')
            self.assertEqual(merge_nights(path, [self.night(date='2026-11-03')])['total'], 3)

    def test_invalid_or_duplicate_backup_does_not_partially_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            add_night(path, self.night())
            before = path.read_bytes()
            for incoming in ([], [self.night(date='2026-11-02'), self.night(quality='99')],
                             [self.night(date='2026-11-02')] * 2):
                with self.subTest(incoming=incoming), self.assertRaises(ValueError):
                    merge_nights(path, incoming, replace=True)
                self.assertEqual(path.read_bytes(), before)

    def test_import_cli_restores_backup_and_rejects_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup, target = root / 'backup.csv', root / 'restored.csv'
            add_night(backup, self.night())
            with patch('sys.argv', ['diary', '--file', str(target), 'import', '--input-csv', str(backup)]), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(json.loads(output.getvalue()), {'added': 1, 'replaced': 0, 'total': 1})
            self.assertEqual(load_diary(target), load_diary(backup))
            with patch('sys.argv', ['diary', '--file', str(backup), 'import', '--input-csv', str(backup), '--replace']):
                with self.assertRaisesRegex(SystemExit, 'differ'):
                    main()

    def test_removal_retains_other_dates_and_can_leave_empty_diary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            add_night(path, self.night())
            add_night(path, self.night(date='2026-11-02'))
            self.assertEqual(remove_night(path, '2026-11-01')['remaining_entries'], 1)
            self.assertEqual(load_diary(path)[0]['date'], '2026-11-02')
            remove_night(path, '2026-11-02')
            self.assertEqual(load_diary(path), [])
            self.assertTrue(path.read_text().startswith('date,bedtime'))

    def test_stale_removal_and_missing_dates_leave_diary_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            add_night(path, self.night())
            fingerprint = night_fingerprint(self.night())
            add_night(path, self.night(quality='3'), replace=True)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'changed'):
                remove_night(path, '2026-11-01', expected_fingerprint=fingerprint)
            with self.assertRaisesRegex(ValueError, 'No diary entry'):
                remove_night(path, '2026-11-02')
            self.assertEqual(path.read_bytes(), before)

    def test_remove_cli_and_missing_file_do_not_create_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            with self.assertRaises(ValueError):
                remove_night(path, '2026-11-01')
            self.assertFalse(path.exists())
            add_night(path, self.night())
            with patch('sys.argv', ['diary', '--file', str(path), 'remove', '--date', '2026-11-01']), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(json.loads(output.getvalue())['remaining_entries'], 0)
