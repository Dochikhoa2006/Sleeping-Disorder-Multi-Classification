import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.sample_data import sample_csv


class SampleDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.csv"
        self.output = self.source.with_name("sample.csv")
        patient = json.loads(Path("examples/patient.json").read_text())
        self.rows = [{**patient, "Record ID": str(i), "Age": age, "Note": "original, value"}
                     for i, age in enumerate([30, 40, 30, 50, 40, 50])]
        self.write(self.rows)

    def write(self, rows):
        with self.source.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def read(self):
        with self.output.open(newline="") as source:
            return list(csv.DictReader(source))

    def test_whole_groups_original_order_values_and_hashes(self):
        with patch("joblib.load") as load:
            report = sample_csv(self.source, self.output, profiles=2)
        load.assert_not_called()
        rows = self.read()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["Age"] for row in rows}), 2)
        self.assertEqual([int(row["Record ID"]) for row in rows], sorted(int(row["Record ID"]) for row in rows))
        self.assertTrue(all(row["Note"] == "original, value" for row in rows))
        self.assertEqual(report["sampled_rows"], 4)
        self.assertEqual(report["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_repeatable_and_independent_of_row_order(self):
        sample_csv(self.source, self.output, profiles=1, seed=45)
        first = self.output.read_bytes()
        sample_csv(self.source, self.output, profiles=1, seed=45)
        self.assertEqual(first, self.output.read_bytes())
        selected = {row["Age"] for row in self.read()}
        self.write(list(reversed(self.rows)))
        sample_csv(self.source, self.output, profiles=1, seed=45)
        self.assertEqual(selected, {row["Age"] for row in self.read()})

    def test_all_profiles_preserves_all_records(self):
        sample_csv(self.source, self.output, profiles=3)
        self.assertEqual([row["Record ID"] for row in self.read()], [str(i) for i in range(6)])

    def test_invalid_options_and_collision_preserve_files(self):
        self.output.write_text("existing")
        for options in ({"profiles": 0}, {"profiles": 4}, {"profiles": True},
                        {"profiles": 1, "seed": -1}, {"profiles": 1, "seed": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sample_csv(self.source, self.output, **options)
            self.assertEqual(self.output.read_text(), "existing")
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            sample_csv(self.source, self.source, profiles=1)
        self.assertEqual(original, self.source.read_bytes())

    def test_cli_json_and_invalid_csv_preserves_output(self):
        command = [sys.executable, "-m", "src.sample_data", "--input-csv", str(self.source),
                   "--output", str(self.output), "--profiles", "1"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["sampled_unique_profiles"], 1)
        original = self.output.read_bytes()
        self.source.write_text("invalid")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(original, self.output.read_bytes())
