import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.convert import convert_text
from src.predict import read_patient_csv
from src.preprocessing import RAW_FEATURES


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(Path("examples/patient.json").read_text())

    def test_json_object_to_csv_projects_fields_without_model(self):
        source = {**self.patient, "Record ID": "private", "Notes": {"nested": "extra"}}
        with patch("joblib.load") as load:
            payload = convert_text(json.dumps(source), "json", "csv")
        load.assert_not_called()
        columns, _, patients = read_patient_csv(StringIO(payload))
        self.assertEqual(columns, RAW_FEATURES)
        self.assertEqual(patients[0]["Age"], self.patient["Age"])
        self.assertNotIn("private", payload)
        self.assertNotIn("Notes", payload)

    def test_batch_round_trip_retains_order_and_quoted_fields(self):
        second = {**self.patient, "Occupation": 'Researcher, "night shift"', "Age": 40}
        payload = convert_text(json.dumps([self.patient, second]), "json", "csv")
        restored = json.loads(convert_text("\ufeff" + payload, "csv", "json"))
        self.assertEqual(restored, [self.patient, second])
        self.assertEqual(len(list(csv.reader(StringIO(payload)))), 3)

    def test_json_output_is_always_array_and_numeric_values_are_numbers(self):
        payload = convert_text(json.dumps(self.patient), "json", "json")
        restored = json.loads(payload)
        self.assertIsInstance(restored, list)
        self.assertIsInstance(restored[0]["Age"], (int, float))

    def test_invalid_batch_duplicate_keys_and_formats_rejected(self):
        for content in ("[]", '{"Age": 30, "Age": 40}', json.dumps([self.patient, {"Age": 40}])):
            with self.subTest(content=content), self.assertRaises(ValueError):
                convert_text(content, "json", "csv")
        with self.assertRaises(ValueError):
            convert_text("", "xml", "csv")

    def test_cli_stdin_stdout_is_parseable(self):
        result = subprocess.run([sys.executable, "-m", "src.convert", "--input-json", "-", "--to", "csv"],
                                input=json.dumps(self.patient), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_patient_csv(StringIO(result.stdout))[0], RAW_FEATURES)

    def test_invalid_input_and_collision_preserve_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            output = source.with_name("output.csv")
            source.write_text(json.dumps(self.patient))
            command = [sys.executable, "-m", "src.convert", "--input-json", str(source), "--to", "csv", "--output"]
            result = subprocess.run([*command, str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(source.read_text()), self.patient)
            source.write_text("invalid")
            output.write_text("previous export")
            result = subprocess.run([*command, str(output)], capture_output=True, text=True)
            self.assertIn("Error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(output.read_text(), "previous export")
