import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluation import classification_metrics
from src.evaluation_report import render_evaluation


class EvaluationReportTests(unittest.TestCase):
    def test_extreme_integer_metrics_are_validation_errors(self):
        for name in ("accuracy", "macro_f1"):
            for value in (10**1000, -(10**1000)):
                with self.subTest(name=name, negative=value < 0), self.assertRaisesRegex(ValueError, "finite number from 0 to 1"):
                    render_evaluation({**self.report(), name: value})

    def test_cli_extreme_metric_has_no_traceback_and_preserves_html(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evaluation.json"
            output = source.with_suffix(".html")
            output.write_text("existing report")
            for name in ("accuracy", "macro_f1"):
                source.write_text(json.dumps({**self.report(), name: 10**1000}))
                result = subprocess.run([sys.executable, "-m", "src.evaluation_report", "--input-json", str(source),
                                         "--output", str(output)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("Error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(output.read_text(), "existing report")

    def test_per_class_metrics_come_from_counts_instead_of_unchecked_saved_details(self):
        report = self.report()
        report["per_class"] = {"Healthy": {"precision": 0, "recall": 0, "f1-score": 0}}
        html = render_evaluation(report)
        self.assertIn("Class scores and observation counts", html)
        self.assertIn('<th scope="row">Healthy</th><td>1.000</td><td>0.500</td><td>0.667</td><td>2</td><td>1</td>', html)
        self.assertIn('<th scope="row">Insomnia</th><td>0.500</td><td>1.000</td><td>0.667</td><td>1</td><td>2</td>', html)

    def test_per_class_metrics_show_absent_classes_and_escape_labels(self):
        report = {"labels": ["Healthy", "<img src=x>"], "evaluated_rows": 1,
                  "confusion_matrix": [[1, 0], [0, 0]], "accuracy": 1, "macro_f1": .5}
        html = render_evaluation(report)
        self.assertIn('<th scope="row">&lt;img src=x&gt;</th><td>0.000</td><td>0.000</td><td>0.000</td><td>0</td><td>0</td>', html)
        self.assertNotIn("<img", html)

    def report(self):
        return {"labels": ["Healthy", "Insomnia"], "evaluated_rows": 3,
                "model_name": '<script>alert("x")</script>',
                **classification_metrics(pd.Series(["Healthy", "Healthy", "Insomnia"]),
                                         ["Healthy", "Insomnia", "Insomnia"], ["Healthy", "Insomnia"])}

    def test_renders_escaped_portable_matrix_without_loading_model(self):
        with patch("joblib.load") as load:
            html = render_evaluation(self.report())
        load.assert_not_called()
        self.assertIn("0.667", html)
        self.assertIn('scope="row">Healthy', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn('src="', html)
        self.assertIn("@media print", html)

    def test_rejects_inconsistent_metrics_counts_and_shapes(self):
        for changes in ({"accuracy": .9}, {"macro_f1": .9}, {"evaluated_rows": 4},
                        {"confusion_matrix": [[1, -1], [0, 1]]}, {"confusion_matrix": [[True, 1], [0, 1]]},
                        {"confusion_matrix": [[1]]}, {"labels": ["Healthy", "Healthy"]}, {"accuracy": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                render_evaluation({**self.report(), **changes})

    def test_absent_class_zero_f1_convention(self):
        report = {"labels": ["Healthy", "Insomnia"], "evaluated_rows": 1,
                  **classification_metrics(pd.Series(["Healthy"]), ["Healthy"], ["Healthy", "Insomnia"])}
        self.assertIn("0.500", render_evaluation(report))

    def test_cli_hashes_input_and_preserves_report_on_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evaluation.json"
            output = source.with_suffix(".html")
            source.write_text(json.dumps(self.report()))
            command = [sys.executable, "-m", "src.evaluation_report", "--input-json", str(source), "--output"]
            result = subprocess.run([*command, str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Report file SHA-256", output.read_text())
            original = output.read_bytes()
            source.write_text('{"labels": [], "labels": []}')
            result = subprocess.run([*command, str(output)], capture_output=True, text=True)
            self.assertIn("Error:", result.stderr)
            self.assertEqual(output.read_bytes(), original)
            snapshot = source.read_bytes()
            result = subprocess.run([*command, str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), snapshot)
