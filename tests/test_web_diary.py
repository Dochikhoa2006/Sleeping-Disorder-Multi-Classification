import csv
import http.client
from io import StringIO
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from src.diary import load_diary
from src.web import make_server


class BrowserDiaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'diary.csv'
        self.server = make_server(port=0, diary_path=self.path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.values = {'date': '2026-11-01', 'bedtime': '23:00', 'wake_time': '07:00',
                       'awake_minutes': '30', 'quality': '8'}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def request(self, method='GET', path='/diary', body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read().decode()
        connection.close()
        return result

    def add(self, values=None):
        return self.request('POST', body=urlencode(self.values if values is None else values),
                            headers={'Content-Type': 'application/x-www-form-urlencoded'})

    def test_add_download_and_report_work_without_model_loading(self):
        with patch('src.predict.joblib.load') as load:
            self.assertEqual(self.request()[0], 200)
            self.assertFalse(self.path.exists())
            status, _, body = self.add()
            self.assertEqual(status, 200)
            self.assertIn('7.50 h', body)
            status, headers, body = self.request(path='/diary.csv')
            self.assertEqual(status, 200)
            self.assertIn('attachment', headers['Content-Disposition'])
            self.assertEqual(list(csv.DictReader(StringIO(body))), [self.values])
            self.assertEqual(self.request(path='/diary-report')[0], 200)
        load.assert_not_called()
        self.assertEqual(load_diary(self.path), [self.values])

    def test_duplicate_requires_checkbox_and_invalid_input_preserves_diary(self):
        self.add()
        original = self.path.read_bytes()
        self.assertEqual(self.add()[0], 400)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.add({**self.values, 'awake_minutes': '900'})[0], 400)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.add({**self.values, 'replace': '1', 'quality': '4'})[0], 200)
        self.assertEqual(load_diary(self.path)[0]['quality'], '4')

    def test_nonlocal_requests_and_bad_form_are_rejected(self):
        status, _, _ = self.request('POST', body=urlencode(self.values), headers={
            'Content-Type': 'application/x-www-form-urlencoded', 'Origin': 'https://example.com'})
        self.assertEqual(status, 403)
        self.assertEqual(self.request('POST', body='x', headers={'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request('POST', body='x', headers={'Content-Length': '2049',
                         'Content-Type': 'application/x-www-form-urlencoded'})[0], 413)
        self.assertFalse(self.path.exists())

    def test_corrupt_diary_is_not_overwritten(self):
        self.path.write_text('not,a,diary\n')
        self.assertEqual(self.request()[0], 503)
        self.assertEqual(self.add()[0], 503)
        self.assertEqual(self.path.read_text(), 'not,a,diary\n')

    def test_disabled_diary_cannot_write(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server = make_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.assertIn('--diary', self.request()[2])
        self.assertEqual(self.add()[0], 403)
        self.assertEqual(self.request(path='/diary.csv')[0], 404)
        self.assertEqual(self.request(path='/diary.ics')[0], 404)
        self.assertFalse(self.path.exists())

    def test_month_calendar_shows_recorded_and_missing_nights(self):
        self.add()
        status, headers, body = self.request(path='/diary?month=2026-11')
        self.assertEqual(status, 200)
        self.assertIn('<caption>November 2026</caption>', body)
        self.assertIn('1 of 30 nights recorded', body)
        self.assertIn('7.50 h average estimated sleep per recorded night', body)
        self.assertIn('<time datetime="2026-11-01">1</time><strong>7.50 h</strong>', body)
        self.assertIn('href="/diary?month=2026-10"', body)
        self.assertIn('href="/diary?month=2026-12"', body)
        self.assertEqual(headers['Cache-Control'], 'no-store')

    def test_leap_month_and_year_bounds_render_without_writing(self):
        for month, count in (('2024-02', 29), ('2025-02', 28), ('0001-01', 31), ('9999-12', 31)):
            with self.subTest(month=month):
                status, _, body = self.request(path='/diary?month=' + month)
                self.assertEqual(status, 200)
                self.assertEqual(body.count('<time datetime="' + month), count)
                self.assertIn(f'0 of {count} nights recorded', body)
        self.assertFalse(self.path.exists())

    def test_invalid_calendar_queries_are_rejected_without_writing(self):
        for query in ('month=2026-13', 'month=', 'month=0000-01', 'month=202611',
                      'month=2026-01&month=2026-02', 'unknown=2026-01'):
            with self.subTest(query=query):
                self.assertEqual(self.request(path='/diary?' + query)[0], 400)
        self.assertFalse(self.path.exists())

    def test_calendar_distinguishes_zero_sleep_from_missing(self):
        self.add({**self.values, 'awake_minutes': '480'})
        _, _, body = self.request(path='/diary?month=2026-11')
        self.assertIn('<time datetime="2026-11-01">1</time><strong>0.00 h</strong>', body)
        self.assertIn('<time datetime="2026-11-02">2</time><span class="diary-missing">Not recorded</span>', body)

    def test_calendar_download_exports_overnight_event_without_model_loading(self):
        self.add()
        with patch('src.predict.joblib.load') as load:
            status, headers, body = self.request(path='/diary.ics')
        load.assert_not_called()
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'text/calendar; charset=utf-8')
        self.assertIn('sleep-diary.ics', headers['Content-Disposition'])
        self.assertIn('DTSTART:20261101T230000', body)
        self.assertIn('DTEND:20261102T070000', body)
        self.assertIn('BEGIN:VCALENDAR', body)
