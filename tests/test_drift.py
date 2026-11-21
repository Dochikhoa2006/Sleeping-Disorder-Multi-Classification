from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from src.drift import compare_datasets, compare_dataset_bytes, main, numeric_shift


class DriftTests(unittest.TestCase):
    def test_distance_limits_are_inclusive_and_report_each_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self.write(root, "reference.csv", [{"Age": 20, "Occupation": "Engineer"}])
            current = self.write(root, "current.csv", [{"Age": 60, "Occupation": "Teacher"}])
            with patch("joblib.load") as load:
                report = compare_datasets(reference, current, max_ks_distance=.5, max_category_distance=.5)
                boundary = compare_datasets(reference, current, max_ks_distance=1, max_category_distance=1)
            load.assert_not_called()
            self.assertFalse(report["passed"])
            self.assertFalse(report["limit_checks"]["numeric"]["Age"])
            self.assertFalse(report["limit_checks"]["categorical"]["Occupation"])
            self.assertTrue(report["limit_checks"]["numeric"]["Daily Steps"])
            self.assertTrue(boundary["passed"])
            self.assertNotIn("passed", compare_datasets(reference, current))
            reference = self.write(root, "many-reference.csv", [{"Occupation": f"Reference {i}"} for i in range(9)])
            current = self.write(root, "many-current.csv", [{"Occupation": f"Current {i}"} for i in range(9)])
            self.assertTrue(compare_datasets(reference, current, max_category_distance=1)["passed"])

    def test_invalid_distance_limits_fail_before_file_reads(self):
        for field in ("max_ks_distance", "max_category_distance"):
            for value in (-.1, 1.1, 10**1000, float("nan"), float("inf"), True, "0.5"):
                with self.subTest(field=field, value=value), patch.object(Path, "read_bytes") as read:
                    with self.assertRaises(ValueError):
                        compare_datasets(Path("missing-a.csv"), Path("missing-b.csv"), **{field: value})
                    read.assert_not_called()

    def test_cli_writes_complete_failed_limit_report_and_preserves_invalid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self.write(root, "reference.csv", [{"Age": 20}])
            current = self.write(root, "current.csv", [{"Age": 60}])
            output = root / "drift.json"
            args = ["drift", "--reference", str(reference), "--current", str(current), "--output", str(output), "--max-ks-distance"]
            with patch("sys.argv", [*args, "0.5"]):
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 1)
            self.assertFalse(json.loads(output.read_text())["passed"])
            before = output.read_bytes()
            with patch("sys.argv", [*args, "nan"]), self.assertRaisesRegex(SystemExit, "finite number"):
                main()
            self.assertEqual(output.read_bytes(), before)

    def write(self, root, name, changes):
        patient = json.loads(Path('examples/patient.json').read_text())
        path = root / name
        with path.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(patient))
            writer.writeheader()
            writer.writerows({**patient, **change} for change in changes)
        return path

    def test_identical_inputs_have_zero_distances_without_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(Path(directory), 'patients.csv', [{'Age': 20}, {'Age': 40}])
            with patch('joblib.load') as load:
                report = compare_datasets(path, path)
        load.assert_not_called()
        self.assertTrue(all(row['ks_distance'] == 0 for row in report['numeric'].values()))
        self.assertTrue(all(row['total_variation_distance'] == 0 for row in report['categorical'].values()))
        self.assertEqual(report['numeric']['Age']['standardized_mean_change'], 0)

    def test_numeric_shift_and_constant_reference_are_explicit(self):
        result = numeric_shift(np.array([10., 20.]), np.array([20., 30.]))
        self.assertEqual(result['mean_change'], 10)
        self.assertEqual(result['standardized_mean_change'], 2)
        self.assertEqual(result['ks_distance'], .5)
        constant = numeric_shift(np.array([10., 10.]), np.array([20., 20.]))
        self.assertIsNone(constant['standardized_mean_change'])
        self.assertEqual(constant['ks_distance'], 1)

    def test_categorical_shares_and_new_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self.write(root, 'reference.csv', [{'Occupation': 'Engineer'}] * 2)
            current = self.write(root, 'current.csv', [{'Occupation': 'Engineer'}, {'Occupation': 'Teacher'}])
            result = compare_datasets(reference, current)['categorical']['Occupation']
        self.assertEqual(result['new_categories'], ['Teacher'])
        self.assertEqual(result['total_variation_distance'], .5)
        self.assertEqual(sum(row['current_share'] for row in result['distribution']), 1)

    def test_normalization_matches_model_and_pressure_is_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self.write(root, 'reference.csv', [{'BMI Category': 'Normal'}])
            current = self.write(root, 'current.csv', [{'BMI Category': ' Normal Weight ', 'Blood Pressure': '130/85'}])
            report = compare_datasets(reference, current)
        self.assertEqual(report['categorical']['BMI Category']['new_categories'], [])
        self.assertEqual(report['numeric']['Systolic Blood Pressure']['mean_change'], 10)
        self.assertEqual(report['numeric']['Diastolic Blood Pressure']['mean_change'], 5)

    def test_cli_stdout_is_json_and_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(Path(directory), 'patients.csv', [{}])
            argv = ['drift', '--reference', str(path), '--current', str(path)]
            with patch('sys.argv', argv), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(json.loads(output.getvalue())['current']['rows'], 1)
            with patch('sys.argv', argv + ['--output', str(path)]):
                with self.assertRaisesRegex(SystemExit, 'differ'):
                    main()

    def test_invalid_input_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write(root, 'patients.csv', [{'Age': -1}])
            output = root / 'report.json'
            output.write_text('keep')
            with patch('sys.argv', ['drift', '--reference', str(path), '--current', str(path), '--output', str(output)]):
                with self.assertRaisesRegex(SystemExit, 'Error:'):
                    main()
            self.assertEqual(output.read_text(), 'keep')

    def test_uploaded_snapshots_match_file_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self.write(root, 'reference.csv', [{'Age': 20}])
            current = self.write(root, 'current.csv', [{'Age': 30}, {'Age': 40}])
            self.assertEqual(compare_dataset_bytes(reference.read_bytes(), current.read_bytes()),
                             compare_datasets(reference, current))
