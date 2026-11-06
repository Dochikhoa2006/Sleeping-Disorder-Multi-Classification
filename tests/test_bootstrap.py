from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.evaluation import bootstrap_intervals, evaluate_bytes, feature_fingerprints
from src.evaluate import main
from src.predict import DEFAULT_MODEL_PATH
from src.preprocessing import TARGET_COLUMN


class BootstrapTests(unittest.TestCase):
    def test_perfect_single_class_predictions_have_constant_intervals(self):
        report = bootstrap_intervals(['A', 'A', 'A'], ['A', 'A', 'A'], [1, 1, 2], ['A'], runs=100)
        self.assertEqual(report['accuracy'], {'lower': 1, 'upper': 1})
        self.assertEqual(report['macro_f1'], {'lower': 1, 'upper': 1})
        self.assertEqual(report['unique_profiles'], 2)

    def test_whole_duplicate_clusters_are_resampled_together(self):
        samples = []
        def capture(actual, predicted):
            samples.append(list(actual))
            return 1.
        with patch('src.evaluation.accuracy_score', side_effect=capture):
            report = bootstrap_intervals(['A', 'A', 'B'], ['A', 'A', 'B'], [1, 1, 2], ['A', 'B'], runs=100)
        self.assertTrue(all(sample.count('A') % 2 == 0 for sample in samples))
        self.assertEqual(report['sampled_rows_min'], 2)
        self.assertEqual(report['sampled_rows_max'], 4)

    def test_repeatable_intervals_are_bounded(self):
        args = (['A', 'B', 'A', 'B'], ['A', 'A', 'B', 'B'], [1, 2, 3, 4], ['A', 'B'])
        first = bootstrap_intervals(*args, runs=100, seed=42)
        self.assertEqual(first, bootstrap_intervals(*args, runs=100, seed=42))
        for name in ('accuracy', 'macro_f1'):
            self.assertTrue(0 <= first[name]['lower'] <= first[name]['upper'] <= 1)

    def test_bad_options_rejected_before_loading_model(self):
        for options in ({'bootstrap_runs': 99}, {'bootstrap_runs': 2001}, {'bootstrap_seed': -1}):
            with self.subTest(options=options), patch('src.evaluation.joblib.load') as load:
                with self.assertRaises(ValueError):
                    evaluate_bytes(b'', DEFAULT_MODEL_PATH, **options)
                load.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'two unseen'):
            bootstrap_intervals(['A'], ['A'], [1], ['A'], runs=100)

    def fixture(self):
        patient = json.loads(Path('examples/patient.json').read_text())
        rows = [{**patient, 'Age': 96, TARGET_COLUMN: 'Healthy'},
                {**patient, 'Age': 97, TARGET_COLUMN: 'Healthy'},
                {**patient, 'Age': 97, TARGET_COLUMN: 'Healthy'},
                {**patient, 'Age': 98, TARGET_COLUMN: 'Insomnia'}]
        model = joblib.load(DEFAULT_MODEL_PATH)
        model.training_feature_hashes_ = set(feature_fingerprints(pd.DataFrame(rows[:1])))
        return pd.DataFrame(rows).to_csv(index=False).encode(), model

    def test_external_evaluation_excludes_overlap_and_predicts_only_once(self):
        payload, model = self.fixture()
        with patch('src.evaluation.joblib.load', return_value=model), \
                patch.object(model, 'predict', return_value=np.array(['Healthy', 'Healthy', 'Healthy'])) as predict:
            report = evaluate_bytes(payload, DEFAULT_MODEL_PATH, bootstrap_runs=100)
        predict.assert_called_once()
        self.assertEqual(report['training_overlap_rows'], 1)
        self.assertEqual(report['bootstrap']['unique_profiles'], 2)
        self.assertEqual(report['bootstrap']['resamples'], 100)

    def test_cli_can_add_intervals_to_json_stdout(self):
        payload, model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'evaluation.csv'
            source.write_bytes(payload)
            with patch('sys.argv', ['evaluate', '--data', str(source), '--bootstrap-runs', '100']), \
                    patch('src.evaluation.joblib.load', return_value=model), redirect_stdout(StringIO()) as output:
                main()
            report = json.loads(output.getvalue())
            self.assertIn('bootstrap', report)
            self.assertEqual(report['bootstrap']['confidence_level'], .95)
