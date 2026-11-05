from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.calendar_export import escape_text, fold_line, render_calendar
from src.diary import add_night, main

STAMP = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


class CalendarExportTests(unittest.TestCase):
    def night(self, **changes):
        return {'date': '2026-12-31', 'bedtime': '23:00', 'wake_time': '07:00',
                'awake_minutes': '30', 'quality': '8', **changes}

    def test_overnight_event_crosses_year_boundary_with_exact_clock_times(self):
        calendar = render_calendar([self.night()], timestamp=STAMP)
        unfolded = calendar.replace('\r\n ', '')
        self.assertIn('DTSTART:20261231T230000\r\n', calendar)
        self.assertIn('DTEND:20270101T070000\r\n', calendar)
        self.assertIn('DTSTAMP:20261105T120000Z\r\n', calendar)
        self.assertIn('Estimated sleep: 7.50 h\\n', unfolded)
        self.assertIn('Quality: 8/10', unfolded)
        self.assertIn('TRANSP:TRANSPARENT', calendar)
        self.assertNotIn('VALARM', calendar)
        self.assertTrue(all(len(line.encode()) <= 75 for line in calendar.split('\r\n')))
        self.assertNotIn('\n', calendar.replace('\r\n', ''))

    def test_same_day_event_and_stable_identifiers_after_edit(self):
        first = render_calendar([self.night(bedtime='13:00', wake_time='14:00')], timestamp=STAMP)
        changed = render_calendar([self.night(quality='2')], timestamp=STAMP)
        first_uid = next(line for line in first.splitlines() if line.startswith('UID:'))
        self.assertIn(first_uid, changed)
        self.assertIn('DTEND:20261231T140000', first)
        other = render_calendar([self.night()], calendar_id='other', timestamp=STAMP)
        self.assertNotIn(first_uid, other)

    def test_utf8_folding_and_text_escaping_round_trip(self):
        original = 'DESCRIPTION:' + escape_text('é' * 100 + ',;\\\nnext')
        folded = fold_line(original)
        self.assertEqual(folded.replace('\r\n ', ''), original)
        self.assertTrue(all(len(line.encode()) <= 75 for line in folded.split('\r\n')))
        self.assertIn('\\,\\;\\\\\\nnext', original)

    def test_empty_calendar_and_invalid_duplicate_dates(self):
        self.assertNotIn('BEGIN:VEVENT', render_calendar([], timestamp=STAMP))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            render_calendar([self.night(), self.night()], timestamp=STAMP)
        with self.assertRaisesRegex(ValueError, 'date range'):
            render_calendar([self.night(date='9999-12-31')], timestamp=STAMP)

    def test_cli_exports_crlf_and_protects_diary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diary.csv'
            output = Path(directory) / 'diary.ics'
            add_night(path, self.night())
            prefix = ['diary', '--file', str(path), 'export-calendar', '--output']
            with patch('sys.argv', prefix + [str(output)]), redirect_stdout(StringIO()):
                main()
            self.assertIn(b'DTEND:20270101T070000\r\n', output.read_bytes())
            before = path.read_bytes()
            with patch('sys.argv', prefix + [str(path)]):
                with self.assertRaisesRegex(SystemExit, 'differ'):
                    main()
            self.assertEqual(path.read_bytes(), before)
