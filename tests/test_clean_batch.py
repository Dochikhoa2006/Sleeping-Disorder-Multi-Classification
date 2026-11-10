import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.clean_batch import clean_csv
from src.predict import load_patient_csv


class CleanBatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.csv"
        self.output = self.source.with_name("clean.csv")
        self.patient = json.loads(Path("examples/patient.json").read_text())

    def write(self, changes):
        with self.source.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=[*self.patient, "Note"])
            writer.writeheader()
            for change in changes:
                writer.writerow({**self.patient, "Note": 'private, "note"\nsecond line', **change})

    def test_preserves_valid_values_order_and_reports_all_invalid_records(self):
        self.write([{"Age": "31.00"}, {"Age": "bad"}, {"Sleep Duration": 25}, {"Age": 40}])
        with patch("joblib.load") as load:
            report = clean_csv(self.source, self.output)
        load.assert_not_called()
        _, rows, _ = load_patient_csv(self.output)
        self.assertEqual([row["Age"] for row in rows], ["31.00", "40"])
        self.assertEqual(rows[0]["Note"], 'private, "note"\nsecond line')
        self.assertEqual(report["retained_csv_rows"], [2, 5])
        self.assertEqual([row["csv_row"] for row in report["rejections"]], [3, 4])
        self.assertEqual(report["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertNotIn("private", json.dumps(report))

    def test_wrong_width_record_rejected_but_following_record_retained(self):
        self.write([{}])
        with self.source.open("a", newline="") as destination:
            destination.write("wrong,width\n")
        report = clean_csv(self.source, self.output)
        self.assertEqual(report["retained_rows"], 1)
        self.assertEqual(report["rejected_rows"], 1)

    def test_empty_all_invalid_and_malformed_preserve_output(self):
        self.output.write_text("old export")
        for changes in ([], [{"Age": "bad"}]):
            self.write(changes)
            with self.assertRaises(ValueError):
                clean_csv(self.source, self.output)
            self.assertEqual(self.output.read_text(), "old export")
        self.write([{}])
        with self.source.open("a") as destination:
            destination.write('"unterminated')
        with self.assertRaises(csv.Error):
            clean_csv(self.source, self.output)
        self.assertEqual(self.output.read_text(), "old export")

    def test_source_collision_preserves_input(self):
        self.write([{}])
        snapshot = self.source.read_bytes()
        with self.assertRaises(ValueError):
            clean_csv(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), snapshot)

    def test_cli_reports_partial_export_before_failure_exit(self):
        self.write([{}, {"Age": "bad"}])
        command = [sys.executable, "-m", "src.clean_batch", "--input-csv", str(self.source), "--output", str(self.output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["rejected_rows"], 1)
        self.assertEqual(len(load_patient_csv(self.output)[1]), 1)
        self.write([{}])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
