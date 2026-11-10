import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.coverage import category_coverage
from src.preprocessing import build_pipeline


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.joblib"
        self.patient = json.loads(Path("examples/patient.json").read_text())
        model = build_pipeline(DummyClassifier()).fit(pd.DataFrame([self.patient]), ["Healthy"])
        joblib.dump(model, self.path)

    def batch(self, changes):
        rows = [{**self.patient, **change} for change in changes]
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode()

    def test_known_and_normalized_aliases_have_coverage_without_prediction(self):
        with patch("sklearn.pipeline.Pipeline.predict", side_effect=AssertionError("No inference")):
            report = category_coverage(self.batch([{}, {"BMI Category": " Normal Weight "}]), self.path)
        self.assertEqual(report["fully_covered_rows"], 2)
        self.assertEqual(report["rows_with_unknown_categories"], 0)

    def test_counts_unknowns_per_field_and_unique_affected_rows(self):
        report = category_coverage(self.batch([{"Occupation": "Astronaut", "Gender": "Other"},
                                               {"Occupation": "Astronaut"}, {}]), self.path)
        self.assertEqual(report["rows_with_unknown_categories"], 2)
        self.assertEqual(report["fields"]["Occupation"]["unknown_categories"], [{"value": "Astronaut", "rows": 2}])
        self.assertEqual(report["fields"]["Gender"]["unknown_rows"], 1)
        self.assertEqual(report["fully_covered_rows"], 1)

    def test_invalid_input_fails_before_loading(self):
        with patch("joblib.load") as load, self.assertRaises(ValueError):
            category_coverage(b"invalid", self.path)
        load.assert_not_called()

    def test_incompatible_model_has_clear_error(self):
        joblib.dump(DummyClassifier().fit([[1]], ["Healthy"]), self.path)
        with self.assertRaisesRegex(ValueError, "fitted project pipeline"):
            category_coverage(self.batch([{}]), self.path)

    def test_cli_failure_flag_report_and_collision(self):
        source = self.path.with_name("input.csv")
        output = self.path.with_name("report.json")
        source.write_bytes(self.batch([{"Occupation": "Astronaut"}]))
        command = [sys.executable, "-m", "src.coverage", "--input-csv", str(source), "--model", str(self.path)]
        result = subprocess.run([*command, "--output", str(output), "--fail-on-unknown"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(output.read_text())["rows_with_unknown_categories"], 1)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["total_rows"], 1)
        snapshot = self.path.read_bytes()
        result = subprocess.run([*command, "--output", str(self.path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(snapshot, self.path.read_bytes())
