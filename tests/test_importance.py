from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.evaluation import feature_fingerprints
from src.importance import measure_importance, main
from src.preprocessing import build_pipeline, RAW_FEATURES, TARGET_COLUMN


class ImportanceTests(unittest.TestCase):
    def fixtures(self, root):
        patient = json.loads(Path('examples/patient.json').read_text())
        patients = [{**patient, 'Age': 25 + index} for index in range(6)]
        labels = ['Healthy', 'Insomnia', 'Sleep Apnea'] * 2
        model = build_pipeline(DummyClassifier(strategy='most_frequent'))
        model.fit(pd.DataFrame(patients), labels)
        model.training_feature_hashes_ = {feature_fingerprints(pd.DataFrame(patients)).iloc[0]}
        model_path = root / 'model.joblib'
        joblib.dump(model, model_path)
        data = root / 'patients.csv'
        with data.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=[*RAW_FEATURES, TARGET_COLUMN])
            writer.writeheader()
            for patient, label in zip(patients, labels):
                writer.writerow({**patient, TARGET_COLUMN: label})
        return data, model_path, model

    def test_constant_classifier_has_zero_importance_and_training_rows_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            data, model, _ = self.fixtures(Path(directory))
            report = measure_importance(data, model, repeats=2)
            again = measure_importance(data, model, repeats=2)
        self.assertEqual(report, again)
        self.assertEqual(report['training_overlap_rows'], 1)
        self.assertEqual(report['evaluated_rows'], 5)
        self.assertEqual({row['feature'] for row in report['features']}, set(RAW_FEATURES))
        self.assertTrue(all(row['score_drops'] == [0, 0] for row in report['features']))
        self.assertGreater(report['baseline_score'], 0)

    def test_informative_age_ranks_above_constant_inputs(self):
        from sklearn.tree import DecisionTreeClassifier
        with tempfile.TemporaryDirectory() as directory:
            data, path, original = self.fixtures(Path(directory))
            frame = pd.read_csv(data)
            model = build_pipeline(DecisionTreeClassifier(random_state=0))
            model.fit(frame[RAW_FEATURES], frame[TARGET_COLUMN])
            model.training_feature_hashes_ = original.training_feature_hashes_
            joblib.dump(model, path)
            report = measure_importance(data, path, repeats=5)
        self.assertEqual(report['baseline_score'], 1)
        self.assertEqual(report['features'][0]['feature'], 'Age')
        self.assertGreater(report['features'][0]['mean_score_drop'], 0)
        self.assertTrue(all(row['mean_score_drop'] == 0 for row in report['features'][1:]))

    def test_rejects_complete_overlap_and_missing_training_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            data, path, model = self.fixtures(Path(directory))
            model.training_feature_hashes_ = set(feature_fingerprints(pd.read_csv(data)))
            joblib.dump(model, path)
            with self.assertRaisesRegex(ValueError, 'no unseen'):
                measure_importance(data, path, repeats=2)
            del model.training_feature_hashes_
            joblib.dump(model, path)
            with self.assertRaisesRegex(ValueError, 'fingerprints'):
                measure_importance(data, path, repeats=2)

    def test_invalid_parameters_rejected_before_loading(self):
        for options in ({'repeats': 1}, {'repeats': 31}, {'seed': -1}):
            with self.subTest(options=options), patch('src.importance.joblib.load') as load:
                with self.assertRaises(ValueError):
                    measure_importance(Path('missing.csv'), **options)
                load.assert_not_called()

    def test_cli_stdout_is_json_and_output_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data, model, _ = self.fixtures(Path(directory))
            argv = ['importance', '--data', str(data), '--model', str(model), '--repeats', '2']
            with patch('sys.argv', argv), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(json.loads(output.getvalue())['scoring'], 'macro_f1')
            for destination in (data, model):
                with patch('sys.argv', argv + ['--output', str(destination)]):
                    with self.assertRaisesRegex(SystemExit, 'differ'):
                        main()

    def test_failure_preserves_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, model, _ = self.fixtures(root)
            output = root / 'importance.json'
            output.write_text('keep')
            data.write_text(data.read_text().replace('Healthy', 'unknown'))
            with patch('sys.argv', ['importance', '--data', str(data), '--model', str(model), '--output', str(output)]):
                with self.assertRaisesRegex(SystemExit, 'Error:'):
                    main()
            self.assertEqual(output.read_text(), 'keep')
