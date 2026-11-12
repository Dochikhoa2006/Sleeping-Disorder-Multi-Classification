import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.merge_data import merge_csvs
from src.predict import load_patient_csv


class MergeDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.first, self.second, self.output = [root / name for name in ("first.csv", "second.csv", "merged.csv")]
        self.patient = json.loads(Path("examples/patient.json").read_text())
        self.write(self.first, ["31.00", "32"])
        self.write(self.second, ["40"], reverse=True)

    def write(self, path, ages, reverse=False, extra=True):
        columns = [*self.patient, "Note"] if extra else list(self.patient)
        if reverse:
            columns.reverse()
        with path.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns)
            writer.writeheader()
            for age in ages:
                row = {**self.patient, "Age": age}
                if extra:
                    row["Note"] = 'quoted, "value"\nsecond line'
                writer.writerow(row)

    def test_preserves_values_order_metadata_and_provenance_without_model(self):
        with patch("joblib.load") as load:
            report = merge_csvs([self.first, self.second], self.output)
        load.assert_not_called()
        columns, rows, _ = load_patient_csv(self.output)
        self.assertEqual(columns, [*self.patient, "Note"])
        self.assertEqual([row["Age"] for row in rows], ["31.00", "32", "40"])
        self.assertEqual(rows[2]["Note"], 'quoted, "value"\nsecond line')
        self.assertEqual(report["inputs"][1]["first_output_csv_row"], 4)
        self.assertEqual(report["inputs"][1]["last_output_csv_row"], 4)
        self.assertEqual(report["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_schema_difference_or_late_invalid_input_preserves_output(self):
        self.output.write_text("old output")
        self.write(self.second, ["40"], extra=False)
        with self.assertRaisesRegex(ValueError, "different columns"):
            merge_csvs([self.first, self.second], self.output)
        self.assertEqual(self.output.read_text(), "old output")
        self.write(self.second, ["bad"])
        with self.assertRaises(ValueError):
            merge_csvs([self.first, self.second], self.output)
        self.assertEqual(self.output.read_text(), "old output")

    def test_duplicate_paths_too_few_inputs_and_output_collision(self):
        for sources, output in (([self.first], self.output), ([self.first, self.first], self.output),
                                ([self.first, self.second], self.first)):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                merge_csvs(sources, output)
        self.assertEqual(len(load_patient_csv(self.first)[1]), 2)

    def test_duplicate_profiles_retained_and_cli_report_parseable(self):
        self.write(self.second, ["31.00"])
        result = subprocess.run([sys.executable, "-m", "src.merge_data", "--input-csv", str(self.first),
                                 str(self.second), "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["output_rows"], 3)
        self.assertEqual([row["Age"] for row in load_patient_csv(self.output)[1]], ["31.00", "32", "31.00"])
