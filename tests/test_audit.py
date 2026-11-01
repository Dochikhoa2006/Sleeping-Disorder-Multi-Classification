import json
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.audit import audit_patients, audit_csv, main


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

    def csv_source(self, rows):
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(list(self.patient) + ['Notes'])
        for row in rows:
            writer.writerow(row)
        output.seek(0)
        return output

    def test_csv_collects_bad_width_numeric_and_range_rows(self):
        valid = list(self.patient.values()) + ['private, multiline\nnotes']
        numeric = dict(self.patient, Age='not a number')
        bounds = dict(self.patient, Age=200)
        source = self.csv_source([valid, ['short'], list(numeric.values()) + [''],
                                  list(bounds.values()) + [''], valid])
        with patch('src.predict.joblib.load') as load:
            report = audit_csv(source)
        load.assert_not_called()
        self.assertEqual(report['valid_records'], 2)
        self.assertEqual(report['invalid_records'], 3)
        self.assertEqual([row['csv_row'] for row in report['records']], [2, 3, 4, 5, 6])
        self.assertNotIn('private', json.dumps(report))
        self.assertIn('expected', report['records'][1]['error'])
        self.assertIn('must be a number', report['records'][2]['error'])
        self.assertIn('120', report['records'][3]['error'])

    def test_csv_file_and_bom_stdin_cli(self):
        for stdin in (False, True):
            with self.subTest(stdin=stdin), redirect_stdout(StringIO()) as output:
                source = Path('examples/patients.csv')
                with patch('sys.argv', ['audit', '--input-csv', '-' if stdin else str(source)]), \
                        patch('sys.stdin', StringIO('\ufeff' + source.read_text(encoding='utf-8-sig'))):
                    main()
                self.assertEqual(json.loads(output.getvalue())['invalid_records'], 0)

    def test_csv_file_level_errors_do_not_replace_report(self):
        header = self.csv_source([]).getvalue()
        for contents in ('Age,Age\n1,2\n', header, header + '"unfinished'):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'report.json'
                output.write_text('keep')
                with patch('sys.argv', ['audit', '--input-csv', '-', '--output', str(output)]), \
                        patch('sys.stdin', StringIO(contents)):
                    with self.assertRaisesRegex(SystemExit, 'Error:'):
                        main()
                self.assertEqual(output.read_text(), 'keep')

    def test_invalid_csv_cli_writes_complete_report_and_exits_one(self):
        source = self.csv_source([['short'], ['also short']])
        with patch('sys.argv', ['audit', '--input-csv', '-']), patch('sys.stdin', source), \
                redirect_stdout(StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(json.loads(output.getvalue())['invalid_records'], 2)
