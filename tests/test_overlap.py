import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.overlap import audit_overlap


class OverlapTests(unittest.TestCase):
    def batch(self, changes):
        patient = json.loads(Path("examples/patient.json").read_text())
        rows = [{**patient, "Record ID": str(index), **change} for index, change in enumerate(changes)]
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode("utf-8-sig")

    def test_normalization_metadata_and_duplicate_weighting(self):
        first = self.batch([{}, {}, {"Age": 50}])
        second = self.batch([{"BMI Category": " Normal Weight ", "Sleep Duration": "7.20"}, {"Age": 60}])
        with patch("joblib.load") as load:
            report = audit_overlap(first, second)
        load.assert_not_called()
        self.assertEqual(report["shared_unique_profiles"], 1)
        self.assertEqual(report["union_unique_profiles"], 3)
        self.assertAlmostEqual(report["profile_jaccard_similarity"], 1 / 3)
        self.assertEqual(report["reference"]["duplicate_rows"], 1)
        self.assertEqual(report["reference"]["shared_rows"], 2)
        self.assertEqual(report["current"]["shared_row_fraction"], .5)
        self.assertEqual(report["current"]["exclusive_unique_profiles"], 1)
        self.assertEqual(report["reference"]["sha256"], hashlib.sha256(first).hexdigest())
        self.assertNotIn("Record ID", json.dumps(report))

    def test_identical_and_disjoint_profiles(self):
        source = self.batch([{}, {}])
        self.assertEqual(audit_overlap(source, source)["profile_jaccard_similarity"], 1)
        report = audit_overlap(source, self.batch([{"Age": 60}]))
        self.assertEqual(report["shared_unique_profiles"], 0)
        self.assertEqual(report["reference"]["shared_rows"], 0)
        self.assertEqual(report["profile_jaccard_similarity"], 0)

    def test_invalid_and_empty_csv_rejected(self):
        for invalid in (b"invalid", self.batch([{}]).splitlines(keepends=True)[0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                audit_overlap(self.batch([{}]), invalid)

    def test_cli_reports_overlap_before_failure_and_preserves_invalid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = source.with_name("report.json")
            source.write_bytes(self.batch([{}]))
            command = [sys.executable, "-m", "src.overlap", "--reference", str(source),
                       "--current", str(source), "--output", str(output), "--fail-on-overlap"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(output.read_text())["shared_unique_profiles"], 1)
            original = output.read_bytes()
            source.write_text("bad")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertIn("Error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_stdout_and_input_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            snapshot = self.batch([{}])
            source.write_bytes(snapshot)
            command = [sys.executable, "-m", "src.overlap", "--reference", str(source), "--current", str(source)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["shared_unique_profiles"], 1)
            result = subprocess.run([*command, "--output", str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), snapshot)
