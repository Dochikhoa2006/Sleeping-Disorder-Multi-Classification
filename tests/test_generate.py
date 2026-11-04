from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.generate import generate_patients, main, serialize_patients
from src.predict import DEFAULT_MODEL_PATH, read_patient_csv, read_patient_json, validate_patients


class GenerateTests(unittest.TestCase):
    def test_seed_is_repeatable_and_does_not_load_a_model(self):
        with patch('src.predict.joblib.load') as load:
            first = generate_patients(100, 42)
            self.assertEqual(first, generate_patients(100, 42))
            self.assertNotEqual(first, generate_patients(100, 43))
        load.assert_not_called()
        validate_patients(first)
        self.assertEqual(len({row['Record ID'] for row in first}), 100)
        self.assertTrue(all(row['Data Source'].startswith('Synthetic') for row in first))
        self.assertTrue(all('Sleep Disorder' not in row for row in first))

    def test_both_formats_round_trip_through_prediction_readers(self):
        patients = generate_patients(50, edge_cases=True)
        self.assertEqual(read_patient_json(StringIO(serialize_patients(patients, 'json'))), patients)
        _, rows, parsed = read_patient_csv(StringIO(serialize_patients(patients, 'csv')))
        self.assertEqual(len(rows), 50)
        self.assertEqual(parsed, patients)

    def test_edge_cases_cover_valid_numeric_limits(self):
        low, high = generate_patients(2, edge_cases=True)
        self.assertEqual((low['Age'], high['Age']), (1, 120))
        self.assertEqual((low['Sleep Duration'], high['Sleep Duration']), (0, 24))
        self.assertEqual((low['Blood Pressure'], high['Blood Pressure']), ('50/30', '300/200'))
        validate_patients([low, high])

    def test_invalid_options_leave_existing_output_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'patients.csv'
            output.write_text('keep')
            for options in (['--rows', '0'], ['--rows', '10001'], ['--seed', '-1'],
                            ['--rows', '1', '--edge-cases']):
                with self.subTest(options=options), patch('sys.argv', ['generate', '--output', str(output), *options]):
                    with self.assertRaisesRegex(SystemExit, 'Error:'):
                        main()
                self.assertEqual(output.read_text(), 'keep')

    def test_cli_stdout_contains_only_json(self):
        with patch('sys.argv', ['generate', '--rows', '3', '--format', 'json', '--output', '-']), redirect_stdout(StringIO()) as output:
            main()
        self.assertEqual(len(json.loads(output.getvalue())), 3)

    def test_cli_saves_csv_and_protects_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'patients.csv'
            with patch('sys.argv', ['generate', '--rows', '4', '--output', str(output)]):
                main()
            self.assertEqual(len(read_patient_csv(StringIO(output.read_text()))[2]), 4)
        with patch('sys.argv', ['generate', '--output', str(DEFAULT_MODEL_PATH)]):
            with self.assertRaisesRegex(SystemExit, 'model'):
                main()
