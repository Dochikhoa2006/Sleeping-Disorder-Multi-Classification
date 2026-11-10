import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.label_audit import audit_labels
from src.preprocessing import TARGET_COLUMN


class LabelAuditTests(unittest.TestCase):
    def batch(self, changes):
        patient = json.loads(Path("examples/patient.json").read_text())
        rows = [{**patient, "Record ID": str(index), TARGET_COLUMN: "Healthy", **change}
                for index, change in enumerate(changes)]
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode("utf-8-sig")

    def test_reports_every_conflict_with_normalized_profiles(self):
        snapshot = self.batch([{}, {"BMI Category": " Normal Weight ", TARGET_COLUMN: "Insomnia"},
                               {"Age": 50}, {"Age": 50, TARGET_COLUMN: "Sleep Apnea"}, {"Age": 60}])
        with patch("joblib.load") as load:
            report = audit_labels(snapshot)
        load.assert_not_called()
        self.assertEqual(report["conflicting_profiles"], 2)
        self.assertEqual(report["rows_in_conflicting_profiles"], 4)
        self.assertEqual([group["csv_rows"] for group in report["conflicts"]], [[2, 3], [4, 5]])
        self.assertEqual(report["conflicts"][0]["label_counts"], {"Healthy": 1, "Insomnia": 1, "Sleep Apnea": 0})
        self.assertNotIn("Record ID", json.dumps(report))
        self.assertNotIn("Occupation", json.dumps(report))

    def test_consistent_duplicates_are_not_conflicts(self):
        report = audit_labels(self.batch([{}, {}, {"Age": 50, TARGET_COLUMN: "Insomnia"}]))
        self.assertEqual(report["conflicting_profiles"], 0)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["class_counts"], {"Healthy": 2, "Insomnia": 1, "Sleep Apnea": 0})

    def test_invalid_missing_and_empty_labels_rejected(self):
        for label in ("", "None", " healthy "):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "invalid CSV rows"):
                audit_labels(self.batch([{TARGET_COLUMN: label}]))
        with self.assertRaisesRegex(ValueError, "must contain"):
            audit_labels(self.batch([{}]).replace(b"Sleep Disorder", b"Other Label"))

    def test_cli_writes_full_report_before_conflict_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = source.with_name("audit.json")
            source.write_bytes(self.batch([{}, {TARGET_COLUMN: "Insomnia"}]))
            command = [sys.executable, "-m", "src.label_audit", "--input-csv", str(source)]
            result = subprocess.run([*command, "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(output.read_text())["conflicting_profiles"], 1)
            source.write_bytes(self.batch([{}]))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["conflicting_profiles"], 0)

    def test_invalid_input_and_collision_preserve_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = source.with_name("audit.json")
            source.write_bytes(self.batch([{}]))
            snapshot = source.read_bytes()
            command = [sys.executable, "-m", "src.label_audit", "--input-csv", str(source), "--output"]
            result = subprocess.run([*command, str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), snapshot)
            source.write_text("invalid")
            output.write_text("previous report")
            result = subprocess.run([*command, str(output)], capture_output=True, text=True)
            self.assertIn("Error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(output.read_text(), "previous report")
