from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

from src.evaluation import evaluate_bytes, feature_fingerprints
from src.evaluate import main
from src.predict import DEFAULT_MODEL_PATH
from src.preprocessing import TARGET_COLUMN


class SubgroupEvaluationTests(unittest.TestCase):
    def fixture(self, missing=False):
        patient = json.loads(Path('examples/patient.json').read_text())
        rows = [{**patient, 'Age': 96, TARGET_COLUMN: 'Healthy'},
                {**patient, 'Age': 97, TARGET_COLUMN: 'Healthy'},
                {**patient, 'Age': 98, 'Gender': None if missing else 'Female',
                 'BMI Category': 'Normal Weight', TARGET_COLUMN: 'Healthy'},
                {**patient, 'Age': 99, 'Gender': 'Female', 'BMI Category': ' Normal ', TARGET_COLUMN: 'Insomnia'}]
        model = joblib.load(DEFAULT_MODEL_PATH)
        model.training_feature_hashes_ = set(feature_fingerprints(pd.DataFrame(rows[:1])))
        return pd.DataFrame(rows).to_csv(index=False).encode(), model

    def test_slices_exclude_training_overlap_and_suppress_small_groups(self):
        payload, model = self.fixture()
        with patch('src.evaluation.joblib.load', return_value=model), \
                patch.object(model, 'predict', return_value=['Healthy'] * 3) as predict:
            report = evaluate_bytes(payload, DEFAULT_MODEL_PATH, group_by=['Gender', 'BMI Category'], min_group_rows=2)
        predict.assert_called_once()
        groups = report['subgroups']['fields']['Gender']
        female = next(group for group in groups if group['value'] == 'Female')
        male = next(group for group in groups if group['value'] == 'Male')
        self.assertEqual((female['rows'], male['rows']), (2, 1))
        self.assertEqual(female['metrics']['accuracy'], .5)
        self.assertEqual(female['class_counts']['Insomnia'], 1)
        self.assertIsNone(male['metrics'])
        self.assertEqual(male['status'], 'insufficient_rows')
        bmi = report['subgroups']['fields']['BMI Category']
        self.assertEqual(len(bmi), 1)
        self.assertEqual(bmi[0]['value'], 'Normal')
        self.assertEqual(bmi[0]['rows'], 3)
        self.assertAlmostEqual(sum(group['share'] for group in groups), 1)

    def test_missing_categories_are_explicit_and_not_dropped(self):
        payload, model = self.fixture(missing=True)
        with patch('src.evaluation.joblib.load', return_value=model):
            report = evaluate_bytes(payload, DEFAULT_MODEL_PATH, group_by=['Gender'], min_group_rows=1)
        groups = report['subgroups']['fields']['Gender']
        missing = next(group for group in groups if group['value'] is None)
        self.assertEqual(missing['rows'], 1)
        self.assertEqual(sum(group['rows'] for group in groups), report['evaluated_rows'])

    def test_bad_group_configuration_is_rejected_before_loading(self):
        for options in ({'group_by': ['Age']}, {'group_by': ['Gender', 'Gender']}, {'min_group_rows': 0}):
            with self.subTest(options=options), patch('src.evaluation.joblib.load') as load:
                with self.assertRaises(ValueError):
                    evaluate_bytes(b'', DEFAULT_MODEL_PATH, **options)
                load.assert_not_called()

    def test_groups_can_be_combined_with_bootstrap_without_repredicting(self):
        payload, model = self.fixture()
        with patch('src.evaluation.joblib.load', return_value=model), \
                patch.object(model, 'predict', return_value=['Healthy'] * 3) as predict:
            report = evaluate_bytes(payload, DEFAULT_MODEL_PATH, group_by=['Gender'], bootstrap_runs=100)
        predict.assert_called_once()
        self.assertIn('bootstrap', report)
        self.assertIn('subgroups', report)

    def test_cli_json_includes_selected_group_fields(self):
        payload, model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'evaluation.csv'
            source.write_bytes(payload)
            with patch('sys.argv', ['evaluate', '--data', str(source), '--group-by', 'Gender', 'BMI Category', '--min-group-rows', '2']), \
                    patch('src.evaluation.joblib.load', return_value=model), redirect_stdout(StringIO()) as output:
                main()
            report = json.loads(output.getvalue())
            self.assertEqual(set(report['subgroups']['fields']), {'Gender', 'BMI Category'})
            self.assertEqual(report['subgroups']['min_rows'], 2)
