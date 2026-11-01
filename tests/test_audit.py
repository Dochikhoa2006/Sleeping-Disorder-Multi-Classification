import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.audit import audit_patients, main


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(Path('examples/patient.json').read_text())

    def test_reports_all_invalid_records_without_model_loading(self):
        patients = [self.patient, {**self.patient, 'Age': -1},
                    {**self.patient, 'Blood Pressure': 'bad'}, self.patient]
        with patch('src.predict.joblib.load') as load:
            report = audit_patients(patients)
        load.assert_not_called()
        self.assertEqual((report['total_records'], report['valid_records'], report['invalid_records']), (4, 2, 2))
        self.assertEqual([row['patient'] for row in report['records']], [1, 2, 3, 4])
        self.assertIn('Age', report['records'][1]['error'])
        self.assertIn('Blood Pressure', report['records'][2]['error'])
        self.assertNotIn(self.patient['Occupation'], json.dumps(report))

    def test_cli_stdin_returns_report_and_failure_status_for_invalid_batch(self):
        output = StringIO()
        with patch('sys.argv', ['audit', '--input-json', '-']), \
                patch('sys.stdin', StringIO(json.dumps([self.patient, {}]))), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(json.loads(output.getvalue())['invalid_records'], 1)

    def test_valid_single_patient_and_atomic_file_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit.json'
            with patch('sys.argv', ['audit', '--input-json', 'examples/patient.json', '--output', str(output)]):
                main()
            self.assertEqual(json.loads(output.read_text())['valid_records'], 1)

    def test_malformed_input_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'audit.json'
            output.write_text('keep')
            with patch('sys.argv', ['audit', '--input-json', '-', '--output', str(output)]), \
                    patch('sys.stdin', StringIO('{broken')):
                with self.assertRaisesRegex(SystemExit, 'Error:'):
                    main()
            self.assertEqual(output.read_text(), 'keep')

    def test_rejects_output_that_would_overwrite_input(self):
        with patch('sys.argv', ['audit', '--input-json', 'examples/patient.json', '--output', 'examples/patient.json']):
            with self.assertRaisesRegex(SystemExit, 'differ'):
                main()
