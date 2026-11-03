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
        self.assertFalse(self.path.exists())
