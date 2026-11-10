from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from src.predict import NUMERIC_RANGES, validate_patient
from src.preprocessing import RAW_FEATURES
from src.schema import main, patient_schema


class SchemaTests(unittest.TestCase):
    def test_required_fields_and_shared_limits_match_runtime(self):
        schema = patient_schema()
        self.assertEqual(schema["required"], RAW_FEATURES)
        self.assertEqual(list(schema["properties"]), RAW_FEATURES)
        self.assertTrue(schema["additionalProperties"])
        patient = json.loads(Path("examples/patient.json").read_text())
        for field, (low, high) in NUMERIC_RANGES.items():
            prop = schema["properties"][field]
            self.assertEqual((prop["minimum"], prop["maximum"]), (low, high))
            for value in (low, high):
                validate_patient({**patient, field: value})
            with self.assertRaises(ValueError):
                validate_patient({**patient, field: high + 1})

    def test_text_constraints_and_pressure_relationship_caveat(self):
        properties = patient_schema()["properties"]
        pattern = properties["Occupation"]["pattern"]
        self.assertIsNone(re.search(pattern, "  "))
        self.assertIsNotNone(re.search(pattern, " Nurse "))
        pressure = properties["Blood Pressure"]
        self.assertIsNotNone(re.search(pressure["pattern"], "120 / 80"))
        self.assertIsNone(re.search(pressure["pattern"], "120-80"))
        self.assertIn("separately", pressure["description"])

    def test_batch_nonempty_and_independent_schema_objects(self):
        schema = patient_schema(batch=True)
        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["minItems"], 1)
        self.assertEqual(schema["items"]["required"], RAW_FEATURES)
        schema["items"]["required"].clear()
        self.assertEqual(patient_schema()["required"], RAW_FEATURES)

    def test_cli_stdout_and_atomic_file_without_model_load(self):
        output = StringIO()
        with patch("sys.argv", ["schema", "--batch"]), patch("joblib.load") as load, redirect_stdout(output):
            main()
        load.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), patient_schema(batch=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            with patch("sys.argv", ["schema", "--output", str(path)]):
                main()
            self.assertEqual(json.loads(path.read_text()), patient_schema())
