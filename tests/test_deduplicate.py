from contextlib import redirect_stdout
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.deduplicate import deduplicate_csv, main
from src.predict import DEFAULT_MODEL_PATH


class DeduplicationTests(unittest.TestCase):
    def fixture(self, root, *, conflicting=False):
        patient = json.loads(Path('examples/patient.json').read_text())
        rows = [{**patient, 'Record ID': 'first', 'Notes': 'comma, and\nline break', 'Sleep Disorder': 'Healthy'},
                {**patient, 'Age': 50, 'Record ID': 'other', 'Notes': 'keep me', 'Sleep Disorder': 'Insomnia'},
                {**patient, 'Sleep Duration': '7.20', 'BMI Category': ' Normal Weight ',
                 'Record ID': 'last', 'Notes': 'different metadata', 'Sleep Disorder': 'Insomnia' if conflicting else 'Healthy'}]
        source = root / 'patients.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return source

    def read(self, path):
        with path.open(newline='') as source:
            return list(csv.DictReader(source))

    def test_first_keeps_original_values_and_reports_removed_rows_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            output = root / 'clean.csv'
            with patch('joblib.load') as load:
                report = deduplicate_csv(source, output)
            load.assert_not_called()
            rows = self.read(output)
            self.assertEqual([row['Record ID'] for row in rows], ['first', 'other'])
            self.assertEqual(rows[0]['Notes'], 'comma, and\nline break')
            self.assertEqual(report['duplicate_groups'], [{'retained_csv_row': 2, 'removed_csv_rows': [4]}])
            self.assertEqual(report['removed_rows'], 1)
            self.assertEqual(report['output_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_last_preserves_retained_order_and_original_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'clean.csv'
            deduplicate_csv(self.fixture(root), output, keep='last')
            rows = self.read(output)
            self.assertEqual([row['Record ID'] for row in rows], ['other', 'last'])
            self.assertEqual(rows[1]['Sleep Duration'], '7.20')
            self.assertEqual(rows[1]['BMI Category'], ' Normal Weight ')

    def test_conflicting_labels_leave_existing_destination_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root, conflicting=True)
            output = root / 'clean.csv'
            output.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                deduplicate_csv(source, output)
            self.assertEqual(output.read_text(), 'keep')

    def test_cli_emits_json_report_and_protects_input_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            with patch('sys.argv', ['deduplicate', '--input-csv', str(source), '--output', str(root / 'clean.csv')]), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(json.loads(output.getvalue())['retained_rows'], 2)
            for destination in (source, DEFAULT_MODEL_PATH):
                with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, 'differ'):
                    deduplicate_csv(source, destination)

    def test_invalid_patient_values_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            source.write_text(source.read_text(encoding='utf-8-sig').replace('120/80', 'bad'))
            output = root / 'clean.csv'
            with self.assertRaises(ValueError):
                deduplicate_csv(source, output)
            self.assertFalse(output.exists())
