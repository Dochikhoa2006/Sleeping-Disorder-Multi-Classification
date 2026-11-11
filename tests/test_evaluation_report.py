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
