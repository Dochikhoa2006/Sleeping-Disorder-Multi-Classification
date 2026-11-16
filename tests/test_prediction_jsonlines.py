from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.predict import main, predict_many, read_patient_json_lines


class JsonLinesPredictionTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(Path("examples/patient.json").read_text())

    def payload(self, rows):
        return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"

    def test_stdin_predictions_preserve_order_and_keep_streams_open(self):
        rows = [self.patient, {**self.patient, "Age": 55}]
        source, output, errors = StringIO(self.payload(rows)), StringIO(), StringIO()
        with patch("sys.argv", ["predict", "--input-jsonl", "-", "--output-json", "-"]), \
                patch("sys.stdin", source), redirect_stdout(output), redirect_stderr(errors):
            main()
        self.assertEqual(json.loads(output.getvalue()), predict_many(rows))
        self.assertIn("not medical advice", errors.getvalue())
        self.assertFalse(source.closed or output.closed)

    def test_validate_only_does_not_load_model(self):
        with patch("sys.argv", ["predict", "--input-jsonl", "-", "--validate-only"]), \
                patch("sys.stdin", StringIO(self.payload([self.patient] * 2))), \
                patch("src.predict.joblib.load") as load, redirect_stdout(StringIO()) as output:
            main()
        load.assert_not_called()
        self.assertEqual(output.getvalue(), "Validated 2 patient record(s).\n")

    def test_late_invalid_line_preserves_export_and_never_loads_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "patients.jsonl", Path(directory) / "predictions.json"
            source.write_text(self.payload([self.patient, {}]))
            output.write_text("existing export")
            args = ["predict", "--input-jsonl", str(source), "--output-json"]
            with patch("sys.argv", [*args, str(output)]), patch("src.predict.joblib.load") as load:
                with self.assertRaisesRegex(SystemExit, "JSON Lines line 2"):
                    main()
            load.assert_not_called()
            self.assertEqual(output.read_text(), "existing export")
            original = source.read_bytes()
            with patch("sys.argv", [*args, str(source)]), self.assertRaisesRegex(SystemExit, "differ from input"):
                main()
            self.assertEqual(source.read_bytes(), original)

    def test_parser_handles_crlf_bom_and_unicode_without_splitting_string_values(self):
        patient = {**self.patient, "Occupation": "Research\u2028night shift"}
        self.assertEqual(read_patient_json_lines(StringIO("\ufeff" + self.payload([patient]).replace("\n", "\r\n"))), [patient])
        for bad in ('{"Age": 30, "Age": 40}', "[]", "null", "", "{}"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_patient_json_lines(StringIO(bad))
