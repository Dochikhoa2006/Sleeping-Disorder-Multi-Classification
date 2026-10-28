import http.client
import base64
import csv
import json
import re
from datetime import datetime
from io import StringIO
import threading
import unittest
from urllib.parse import urlencode
from unittest.mock import patch

from src.predict import PREDICTION_COLUMN, predict, predict_many, read_patient_json
from src.web import DEFAULT_PATIENT, MAX_UPLOAD_BYTES, make_server
from src.model_info import inspect_model


class WebTests(unittest.TestCase):
    @staticmethod
    def csv_source(rows):
        source = StringIO(newline="")
        writer = csv.DictWriter(source, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return source.getvalue().encode("utf-8-sig")

    @staticmethod
    def preview_download(body):
        encoded = re.search(r'href="data:text/csv;charset=utf-8;base64,([^"]+)"', body).group(1)
        return list(csv.DictReader(StringIO(base64.b64decode(encoded).decode("utf-8"), newline="")))

    def test_batch_preview_shows_counts_and_downloads_matching_predictions(self) -> None:
        patients = [DEFAULT_PATIENT, {**DEFAULT_PATIENT, "Age": 55, "Sleep Duration": 5.5}]
        rows = [{**patient, "Record ID": f"row-{index}", "Notes": '<script>café</script>'}
                for index, patient in enumerate(patients)]
        expected = predict_many(patients)
        status, headers, body = self.upload(self.csv_source(rows), path="/batch-preview")
        self.assertEqual(status, 200)
        self.assertNotIn("Content-Disposition", headers)
        self.assertIn("Showing the first 2 of 2 records", body)
        self.assertIn("&lt;script&gt;café&lt;/script&gt;", body)
        self.assertNotIn("<script>", body)
        for label in ("Healthy", "Insomnia", "Sleep Apnea"):
            self.assertIn(f'<p>{label}</p><strong>{expected.count(label)}</strong>', body)
        downloaded = self.preview_download(body)
        self.assertEqual([row[PREDICTION_COLUMN] for row in downloaded], expected)
        self.assertEqual([row["Record ID"] for row in downloaded], ["row-0", "row-1"])
        self.assertEqual(downloaded[0]["Notes"], '<script>café</script>')

    def test_preview_limits_visible_rows_but_keeps_the_entire_download(self) -> None:
        rows = [{**DEFAULT_PATIENT, "Record ID": f"record-{index}"} for index in range(101)]
        status, _, body = self.upload(self.csv_source(rows), path="/batch-preview")
        self.assertEqual(status, 200)
        self.assertIn("Showing the first 100 of 101 records", body)
        self.assertEqual(body.count("<tr>"), 101)  # Header plus 100 patient rows.
        self.assertNotIn("<td>record-100</td>", body)
        downloaded = self.preview_download(body)
        self.assertEqual(len(downloaded), 101)
        self.assertEqual(downloaded[-1]["Record ID"], "record-100")
        predicted = predict(DEFAULT_PATIENT)
        self.assertIn(f'<p>{predicted}</p><strong>101</strong>', body)

    def test_invalid_preview_has_no_results_or_download_and_is_not_stored(self) -> None:
        rows = [DEFAULT_PATIENT, {**DEFAULT_PATIENT, "Age": 0}]
        with patch("src.predict.joblib.load") as load_model:
            status, _, body = self.upload(self.csv_source(rows), path="/batch-preview")
        load_model.assert_not_called()
        self.assertEqual(status, 400)
        self.assertIn("CSV row 3", body)
        self.assertNotIn("data:text/csv", body)
        self.assertNotIn("Your results are ready", body)
        self.assertEqual(self.request("GET", "/batch-preview")[0], 404)

    def test_profile_export_is_valid_cli_json_without_loading_a_model(self) -> None:
        patient = {**DEFAULT_PATIENT, "Occupation": "Ingénieur"}
        with patch("src.predict.joblib.load") as load_model:
            status, headers, body = self.request("POST", "/profile.json", urlencode(patient), {
                "Content-Type": "application/x-www-form-urlencoded",
            })
        load_model.assert_not_called()
        self.assertEqual(status, 200)
        self.assertIn('filename="patient-profile.json"', headers["Content-Disposition"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(read_patient_json(StringIO(body)), patient)

    def test_saved_profile_can_be_restored_without_predicting(self) -> None:
        patient = {**DEFAULT_PATIENT, "Age": 55, "Occupation": "Ingénieur"}
        status, _, saved = self.request("POST", "/profile.json", urlencode(patient), {
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self.assertEqual(status, 200)
        with patch("src.predict.joblib.load") as load_model:
            status, _, body = self.upload(saved.encode("utf-8-sig"), field="profile", path="/import-profile")
        load_model.assert_not_called()
        self.assertEqual(status, 200)
        self.assertIn("Profile loaded", body)
        self.assertIn("No prediction has run yet", body)
        self.assertIn('name="Age" type="number" min="1" max="120" step="any" value="55.0"', body)
        self.assertIn('value="Ingénieur"', body)
        self.assertNotIn("MODEL PREDICTION", body)

    def test_profile_import_rejects_batches_duplicates_and_invalid_records(self) -> None:
        for contents in ('[{"Age": 31}]', '{"Age": 31, "Age": 55}',
                         json.dumps({**DEFAULT_PATIENT, "Age": 0}), 'not JSON'):
            with self.subTest(contents=contents), patch("src.predict.joblib.load") as load_model:
                status, _, body = self.upload(contents.encode(), field="profile", path="/import-profile")
            load_model.assert_not_called()
            self.assertEqual(status, 400)
            self.assertIn('role="alert"', body)
            self.assertNotIn("Profile loaded", body)

    def test_imported_text_is_escaped_and_invalid_export_is_not_downloaded(self) -> None:
        patient = {**DEFAULT_PATIENT, "Occupation": "<script>bad()</script>"}
        status, _, body = self.upload(json.dumps(patient).encode(), field="profile", path="/import-profile")
        self.assertEqual(status, 200)
        self.assertIn("&lt;script&gt;", body)
        self.assertNotIn("<script>", body)
        status, headers, _ = self.request("POST", "/profile.json", urlencode({**patient, "Age": 0}), {
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self.assertEqual(status, 400)
        self.assertNotIn("Content-Disposition", headers)

    def test_profile_import_rejects_oversize_and_foreign_origin_uploads(self) -> None:
        self.assertEqual(self.request("POST", "/import-profile", b"", {
            "Content-Type": "multipart/form-data; boundary=test",
            "Content-Length": str(MAX_UPLOAD_BYTES + 1),
        })[0], 413)
        self.assertEqual(self.request("POST", "/import-profile", b"{}", {
            "Content-Type": "application/json",
        })[0], 415)
        for path in ("/import-profile", "/profile.json"):
            self.assertEqual(self.request("POST", path, b"data", {"Origin": "https://example.com"})[0], 403)

    def test_patient_form_offers_report_for_current_values_in_a_new_tab(self) -> None:
        status, _, body = self.request("GET")
        self.assertEqual(status, 200)
        self.assertIn('formaction="/report" formtarget="_blank"', body)
        self.assertIn("runs a fresh prediction", body)

    def test_printable_report_contains_prediction_all_inputs_and_utc_timestamp(self) -> None:
        patient = {**DEFAULT_PATIENT, "Age": 55}
        status, headers, body = self.request("POST", "/report", urlencode(patient), {
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("<h1>Educational prediction report</h1>", body)
        self.assertIn(f"<h2>{predict(patient)}</h2>", body)
        self.assertEqual(body.count('<th scope="row">'), 11)
        self.assertIn('<th scope="row">Age</th><td>55</td>', body)
        self.assertIn("Software Engineer", body)
        self.assertIn("@media print", body)
        self.assertNotIn("<form", body)
        generated = datetime.fromisoformat(re.search(r"<time>([^<]+)</time>", body).group(1))
        self.assertEqual(generated.utcoffset().total_seconds(), 0)
        self.assertIn("not medical advice or a diagnosis", body)

    def test_report_rejects_invalid_input_before_loading_model(self) -> None:
        with patch("src.predict.joblib.load") as load_model:
            status, _, body = self.request("POST", "/report", urlencode({**DEFAULT_PATIENT, "Age": 0}), {
                "Content-Type": "application/x-www-form-urlencoded",
            })
        load_model.assert_not_called()
        self.assertEqual(status, 400)
        self.assertIn('role="alert"', body)
        self.assertNotIn("<h1>Educational prediction report</h1>", body)

    def test_reports_escape_patient_text_and_cannot_be_read_back_via_get(self) -> None:
        patient = {**DEFAULT_PATIENT, "Occupation": "<script>bad()</script>"}
        status, _, body = self.request("POST", "/report", urlencode(patient), {
            "Content-Type": "application/x-www-form-urlencoded",
        })
        self.assertEqual(status, 200)
        self.assertIn("&lt;script&gt;", body)
        self.assertNotIn("<script>", body)
        self.assertEqual(self.request("GET", "/report")[0], 404)
        self.assertEqual(self.request("POST", "/report", urlencode(patient), {
            "Origin": "https://example.com", "Content-Type": "application/x-www-form-urlencoded",
        })[0], 403)

    def test_model_dashboard_displays_the_actual_saved_model_results(self) -> None:
        report = inspect_model()
        status, _, body = self.request("GET", "/model")
        self.assertEqual(status, 200)
        self.assertIn(f'<h2>{report["model_name"]}</h2>', body)
        self.assertIn(report["model_sha256"], body)
        self.assertIn(str(report["training_metadata"]["training_rows"]), body)
        for name, metrics in report["cross_validation"].items():
            self.assertIn(name, body)
            self.assertIn(f'{metrics["f1_macro"]:.3f}', body)
        self.assertIn("Selected</span>", body)
        self.assertIn('href="/model.json"', body)
        self.assertIn("not confidence intervals", body)

    def test_model_details_download_matches_model_inspection(self) -> None:
        status, headers, body = self.request("GET", "/model.json")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertIn('filename="model-info.json"', headers["Content-Disposition"])
        self.assertEqual(json.loads(body), inspect_model())

    def test_dashboard_handles_models_without_training_metadata(self) -> None:
        report = {"model_name": "Custom classifier", "labels": ["Healthy"],
                  "model_sha256": "example-fingerprint", "training_metadata": None,
                  "cross_validation": None}
        with patch("src.web.inspect_model", return_value=report):
            status, _, body = self.request("GET", "/model")
        self.assertEqual(status, 200)
        self.assertIn("Not recorded", body)
        self.assertIn("No cross-validation results were stored", body)
        self.assertNotIn("Selected</span>", body)

    def test_dashboard_escapes_model_supplied_text(self) -> None:
        report = {"model_name": "<script>bad()</script>", "labels": ["<img src=x>"],
                  "training_metadata": {"validation": "<b>untrusted</b>"},
                  "cross_validation": {"<svg>": {"f1_macro": 0.5, "accuracy": 0.6}}}
        with patch("src.web.inspect_model", return_value=report):
            status, _, body = self.request("GET", "/model")
        self.assertEqual(status, 200)
        self.assertIn("&lt;script&gt;", body)
        self.assertIn("&lt;svg&gt;", body)
        self.assertNotIn("<script>", body)
        self.assertNotIn("<img", body)
        self.assertNotIn("<svg>", body)

    def test_model_read_failures_show_a_dashboard_error(self) -> None:
        with patch("src.web.inspect_model", side_effect=OSError("private filesystem details")):
            for path in ("/model", "/model.json"):
                with self.subTest(path=path):
                    status, _, body = self.request("GET", path)
                    self.assertEqual(status, 503)
                    self.assertIn("Model details unavailable", body)
                    self.assertNotIn("private filesystem details", body)

    @staticmethod
    def comparison_values(first=None, second=None):
        return {prefix + key: value for prefix, patient in (
            ("a.", first or DEFAULT_PATIENT), ("b.", second or DEFAULT_PATIENT),
        ) for key, value in patient.items()}

    def compare(self, values):
        return self.request("POST", "/compare", urlencode(values), {
            "Content-Type": "application/x-www-form-urlencoded",
        })

    def test_comparison_page_has_two_independently_labeled_profiles(self) -> None:
        status, _, body = self.request("GET", "/compare")
        self.assertEqual(status, 200)
        self.assertEqual(body.count("<input "), 22)
        self.assertIn('name="a.Age"', body)
        self.assertIn('name="b.Age"', body)
        self.assertIn('action="/compare"', body)
        ids = re.findall(r'\bid="([^"]+)"', body)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("not the effect of changing a habit", body)

    def test_comparison_shows_ordered_predictions_and_edited_fields(self) -> None:
        alternative = {**DEFAULT_PATIENT, "Age": 55, "Sleep Duration": 5.5}
        expected = predict_many([DEFAULT_PATIENT, alternative])
        status, _, body = self.compare(self.comparison_values(second=alternative))
        self.assertEqual(status, 200)
        for label, prediction in zip(("A", "B"), expected):
            self.assertIn(f'PROFILE {label}</p><h2>{prediction}</h2>', body)
        self.assertIn("Edited fields", body)
        self.assertIn('<th scope="row">Age</th><td>31</td><td>55</td>', body)
        self.assertIn('name="b.Sleep Duration" type="number" min="0" max="24" step="any" value="5.5"', body)

    def test_identical_profiles_produce_same_class_without_edited_fields(self) -> None:
        status, _, body = self.compare(self.comparison_values())
        self.assertEqual(status, 200)
        self.assertIn("Both profiles received the same predicted class", body)
        self.assertIn("identical entries", body)
        self.assertNotIn("<caption>Edited fields</caption>", body)

    def test_invalid_comparison_retains_both_profiles_without_loading_model(self) -> None:
        for value in (0, "not-a-number"):
            values = self.comparison_values(second={**DEFAULT_PATIENT, "Age": value})
            values["a.Occupation"] = '<script>alert("test")</script>'
            with self.subTest(age=value), patch("src.predict.joblib.load") as load_model:
                status, _, body = self.compare(values)
            load_model.assert_not_called()
            self.assertEqual(status, 400)
            self.assertIn("Profile B:", body)
            self.assertIn("&lt;script&gt;", body)
            self.assertNotIn("<script>", body)
            self.assertIn(f'value="{value}"', body)
            self.assertNotIn("MODEL PREDICTION", body)

    def test_comparison_changes_table_escapes_submitted_text(self) -> None:
        values = self.comparison_values(second={**DEFAULT_PATIENT, "Occupation": "<img src=x>"})
        status, _, body = self.compare(values)
        self.assertEqual(status, 200)
        self.assertIn("&lt;img src=x&gt;", body)
        self.assertNotIn("<img", body)

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

    def upload(self, contents, *, duplicate=False, field="patients", path="/batch"):
        boundary = "test-upload-boundary"
        part = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="upload"\r\n'
            'Content-Type: text/csv\r\n\r\n'
        ).encode() + contents + b"\r\n"
        body = part * (2 if duplicate else 1) + f"--{boundary}--\r\n".encode()
        return self.request("POST", path, body, {
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
        self.assertEqual(body.count("<input "), 13)
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
