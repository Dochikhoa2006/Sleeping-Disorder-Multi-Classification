from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.audit import audit_json_lines, main


class JsonLinesAuditTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(Path("examples/patient.json").read_text())

    def test_collects_every_syntax_duplicate_shape_and_patient_error(self):
        valid = json.dumps(self.patient)
        lines = [valid, json.dumps({**self.patient, "Age": -1}), "not json",
                 '{"Age": 30, "Age": 40}', "[]", "", valid]
        with patch("src.predict.joblib.load") as load:
            report = audit_json_lines(StringIO("\n".join(lines) + "\n"))
        load.assert_not_called()
        self.assertEqual((report["total_records"], report["valid_records"], report["invalid_records"]), (7, 2, 5))
        self.assertEqual([row["json_line"] for row in report["records"]], list(range(1, 8)))
        self.assertTrue(report["records"][-1]["valid"])
        self.assertNotIn(self.patient["Occupation"], json.dumps(report))

    def test_bom_crlf_unicode_and_optional_final_newline(self):
        patient = {**self.patient, "Occupation": "Research\u2028night shift"}
        line = json.dumps(patient, ensure_ascii=False)
        for payload in (line, "\ufeff" + line + "\r\n"):
            self.assertEqual(audit_json_lines(StringIO(payload))["valid_records"], 1)
        with self.assertRaises(ValueError):
            audit_json_lines(StringIO(""))

    def test_cli_stdin_reports_all_records_before_failure(self):
        source = StringIO(json.dumps(self.patient) + "\n{}\n")
        with patch("sys.argv", ["audit", "--input-jsonl", "-"]), patch("sys.stdin", source), redirect_stdout(StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(json.loads(output.getvalue())["invalid_records"], 1)
        self.assertFalse(source.closed)

    def test_empty_input_preserves_existing_report_and_input_collision_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "patients.jsonl", Path(directory) / "audit.json"
            source.write_text("")
            output.write_text("existing report")
            args = ["audit", "--input-jsonl", str(source), "--output"]
            with patch("sys.argv", [*args, str(output)]), self.assertRaisesRegex(SystemExit, "at least one"):
                main()
            self.assertEqual(output.read_text(), "existing report")
            with patch("sys.argv", [*args, str(source)]), self.assertRaisesRegex(SystemExit, "differ"):
                main()
