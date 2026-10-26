import http.client
import csv
from io import StringIO
import threading
import unittest
from urllib.parse import urlencode
from unittest.mock import patch

from src.predict import PREDICTION_COLUMN, predict, predict_many
from src.web import DEFAULT_PATIENT, MAX_UPLOAD_BYTES, make_server


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path="/", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode("utf-8")
        finally:
            connection.close()

    def submit(self, patient):
        return self.request("POST", "/predict", urlencode(patient), {
            "Content-Type": "application/x-www-form-urlencoded",
        })

    def upload(self, contents, *, duplicate=False):
        boundary = "test-upload-boundary"
        part = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="patients"; filename="patients.csv"\r\n'
            'Content-Type: text/csv\r\n\r\n'
        ).encode() + contents + b"\r\n"
        body = part * (2 if duplicate else 1) + f"--{boundary}--\r\n".encode()
        return self.request("POST", "/batch", body, {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        })

    def test_csv_upload_downloads_ordered_predictions_with_original_columns(self) -> None:
        patients = [DEFAULT_PATIENT, {**DEFAULT_PATIENT, "Age": 55}]
        rows = [{**patient, "Record ID": str(index), "Notes": 'Quoted, café\nsecond line'}
                for index, patient in enumerate(patients)]
        source = StringIO(newline="")
        writer = csv.DictWriter(source, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        status, headers, body = self.upload(source.getvalue().encode("utf-8-sig"))
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn('filename="sleep-predictions.csv"', headers["Content-Disposition"])
        result = list(csv.DictReader(StringIO(body, newline="")))
        self.assertEqual([row["Record ID"] for row in result], ["0", "1"])
        self.assertEqual([row[PREDICTION_COLUMN] for row in result], predict_many(patients))
        for actual, original in zip(result, rows, strict=True):
            for key, value in original.items():
                self.assertEqual(actual[key], str(value))

    def test_invalid_csv_row_returns_error_instead_of_partial_download(self) -> None:
        source = StringIO()
        writer = csv.DictWriter(source, fieldnames=list(DEFAULT_PATIENT))
        writer.writeheader()
        writer.writerows([DEFAULT_PATIENT, {**DEFAULT_PATIENT, "Age": 0}])
        with patch("src.predict.joblib.load") as load_model:
            status, headers, body = self.upload(source.getvalue().encode())
        load_model.assert_not_called()
        self.assertEqual(status, 400)
        self.assertNotIn("Content-Disposition", headers)
        self.assertIn("CSV row 3", body)
        self.assertIn('role="alert"', body)

    def test_malformed_empty_and_duplicate_uploads_are_rejected(self) -> None:
        self.assertEqual(self.upload(b"")[0], 400)
        self.assertEqual(self.upload(b"Age,Age\n31,32\n")[0], 400)
        self.assertEqual(self.upload(b"Age\n31\n", duplicate=True)[0], 400)
        self.assertEqual(self.request("POST", "/batch", b"invalid", {
            "Content-Type": "multipart/form-data; boundary=missing",
        })[0], 400)

    def test_batch_limits_and_external_requests_are_rejected_before_processing(self) -> None:
        self.assertEqual(self.request("POST", "/batch", b"", {
            "Content-Type": "multipart/form-data; boundary=test",
            "Content-Length": str(MAX_UPLOAD_BYTES + 1),
        })[0], 413)
        self.assertEqual(self.request("POST", "/batch", b"data", {
            "Content-Type": "text/csv",
        })[0], 415)
        self.assertEqual(self.request("POST", "/batch", b"data", {
            "Origin": "https://example.com",
        })[0], 403)

    def test_example_csv_can_be_downloaded(self) -> None:
        status, headers, body = self.request("GET", "/example.csv")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        rows = list(csv.DictReader(StringIO(body)))
        self.assertEqual(len(rows), 2)
        self.assertIn("Record ID", rows[0])

    def test_home_page_contains_complete_accessible_form(self) -> None:
        status, headers, body = self.request("GET")
        self.assertEqual(status, 200)
        self.assertEqual(body.count("<input "), 12)
        self.assertIn('enctype="multipart/form-data"', body)
        self.assertIn('action="/predict"', body)
        self.assertIn('for="field-0"', body)
        self.assertIn("not medical advice", body)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])

    def test_form_prediction_matches_cli_model(self) -> None:
        status, _, body = self.submit(DEFAULT_PATIENT)
        self.assertEqual(status, 200)
        self.assertIn(f"<h2>{predict(DEFAULT_PATIENT)}</h2>", body)
        self.assertIn('role="status"', body)

    def test_invalid_inputs_are_retained_and_escaped_without_model_loading(self) -> None:
        patient = {**DEFAULT_PATIENT, "Age": 0, "Occupation": '<script>alert("x")</script>'}
        with patch("src.predict.joblib.load") as load_model:
            status, _, body = self.submit(patient)
        load_model.assert_not_called()
        self.assertEqual(status, 400)
        self.assertIn('role="alert"', body)
        self.assertIn("Age", body)
        self.assertIn("&lt;script&gt;", body)
        self.assertNotIn("<script>", body)

    def test_duplicate_form_values_are_rejected(self) -> None:
        status, _, body = self.request("POST", "/predict", "Age=31&Age=55", {
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self.assertEqual(status, 400)
        self.assertIn("exactly once", body)

    def test_external_origins_and_unknown_routes_are_rejected(self) -> None:
        status, _, _ = self.request("POST", "/predict", "Age=31", {
            "Origin": "https://example.com", "Content-Type": "application/x-www-form-urlencoded",
        })
        self.assertEqual(status, 403)
        self.assertEqual(self.request("GET", "/models/sleep_disorder_pipeline.joblib")[0], 404)
        self.assertEqual(self.request("POST", "/predict", "{}", {"Content-Type": "application/json"})[0], 415)


if __name__ == "__main__":
    unittest.main()
