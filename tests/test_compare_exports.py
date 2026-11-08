import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.compare_exports import compare_files
import test_prediction_diff as fixtures


class ExportComparisonCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name) / "baseline.csv"
        self.candidate = self.base.with_name("candidate.csv")
        self.output = self.base.with_name("report.json")
        self.payload = fixtures.PredictionDiffTests().export([{"Record ID": "A"}])
        self.base.write_bytes(self.payload)
        self.candidate.write_bytes(self.payload)

    def run_cli(self, *options):
        return subprocess.run([sys.executable, "-m", "src.compare_exports", "--baseline", str(self.base),
                               "--candidate", str(self.candidate), *options], capture_output=True, text=True)

    def test_stdout_json_and_identical_exit_zero(self):
        result = self.run_cli("--fail-on-change")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["counts"]["unchanged"], 1)
        self.assertEqual(report["baseline_sha256"], hashlib.sha256(self.payload).hexdigest())

    def test_changed_report_written_before_failure_exit(self):
        self.candidate.write_bytes(fixtures.PredictionDiffTests().export([{"Record ID": "B"}]))
        result = self.run_cli("--fail-on-change", "--output", str(self.output))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        report = json.loads(self.output.read_text())
        self.assertEqual(report["counts"]["added"], 1)
        self.assertEqual(report["counts"]["removed"], 1)
        self.assertNotEqual(report["baseline_sha256"], report["candidate_sha256"])
        self.assertEqual(self.run_cli().returncode, 0)

    def test_invalid_input_preserves_existing_report(self):
        self.output.write_text("old report")
        self.candidate.write_text("invalid CSV")
        result = self.run_cli("--output", str(self.output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.output.read_text(), "old report")

    def test_output_collision_rejected_before_reading(self):
        for path in (self.base, self.candidate):
            with self.subTest(path=path), patch.object(Path, "read_bytes") as read:
                with self.assertRaises(ValueError):
                    compare_files(self.base, self.candidate, path)
                read.assert_not_called()

    def test_does_not_load_model(self):
        with patch("joblib.load") as load:
            report = compare_files(self.base, self.candidate, self.output)
        load.assert_not_called()
        self.assertEqual(report, json.loads(self.output.read_text()))
