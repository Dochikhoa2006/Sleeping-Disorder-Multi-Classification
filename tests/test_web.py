import http.client
import threading
import unittest
from urllib.parse import urlencode
from unittest.mock import patch

from src.predict import predict
from src.web import DEFAULT_PATIENT, make_server


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

    def test_home_page_contains_complete_accessible_form(self) -> None:
        status, headers, body = self.request("GET")
        self.assertEqual(status, 200)
        self.assertEqual(body.count("<input "), 11)
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
