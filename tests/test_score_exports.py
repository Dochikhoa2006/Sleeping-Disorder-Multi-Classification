import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation_report import render_evaluation
from src.predict import PREDICTION_COLUMN
from src.score_exports import score_export


class ScoreExportsTests(unittest.TestCase):
    def batch(self, labels):
        patient = json.loads(Path("examples/patient.json").read_text())
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=[*patient, "Sleep Disorder", PREDICTION_COLUMN])
        writer.writeheader()
        writer.writerows({**patient, "Sleep Disorder": actual, PREDICTION_COLUMN: predicted}
                         for actual, predicted in labels)
        return output.getvalue().encode("utf-8-sig")

    def test_known_scores_fixed_labels_and_no_model(self):
        snapshot = self.batch([("Healthy", "Healthy"), ("Insomnia", "Healthy"), ("Sleep Apnea", "Sleep Apnea")])
        with patch("joblib.load") as load:
            report = score_export(snapshot)
        load.assert_not_called()
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 0 + 1) / 3)
        self.assertEqual(report["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        self.assertFalse(report["training_overlap_checked"])
        self.assertIn("Classifier evaluation report", render_evaluation(report))

    def test_threshold_equality_and_missing_class_zero_f1(self):
        snapshot = self.batch([("Healthy", "Healthy")])
        report = score_export(snapshot, min_accuracy=1, min_macro_f1=1 / 3)
        self.assertTrue(report["passed"])
        self.assertEqual(report["class_counts"]["Insomnia"], 0)
        self.assertFalse(score_export(snapshot, min_macro_f1=.5)["passed"])

    def test_invalid_labels_missing_target_and_thresholds(self):
        for pair in (("None", "Healthy"), ("Healthy", "Other")):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                score_export(self.batch([pair]))
        snapshot = self.batch([("Healthy", "Healthy")])
        with self.assertRaisesRegex(ValueError, "actual labels"):
            score_export(snapshot.replace(b",Sleep Disorder,", b",Other Label,"))
        for value in (-.1, 1.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                score_export(snapshot, min_accuracy=value)

    def test_cli_writes_failed_threshold_report_and_preserves_invalid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "predictions.csv"
            output = source.with_name("scores.json")
            source.write_bytes(self.batch([("Healthy", "Insomnia")]))
            command = [sys.executable, "-m", "src.score_exports", "--input-csv", str(source)]
            result = subprocess.run([*command, "--min-accuracy", "0.8", "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(json.loads(output.read_text())["passed"])
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["accuracy"], 0)
            saved = output.read_bytes()
            source.write_text("invalid")
            result = subprocess.run([*command, "--output", str(output)], capture_output=True, text=True)
            self.assertIn("Error:", result.stderr)
            self.assertEqual(output.read_bytes(), saved)
