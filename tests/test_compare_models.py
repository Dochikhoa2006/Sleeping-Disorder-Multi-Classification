from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
from sklearn.dummy import DummyClassifier

from src.compare_models import compare_models, main
from src.predict import DEFAULT_MODEL_PATH


class CompareModelsTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(Path('examples/patient.json').read_text())

    def save_model(self, directory, name, label):
        model = DummyClassifier(strategy='constant', constant=label)
        model.fit([[0], [1]], ['Healthy', 'Insomnia'])
        path = Path(directory) / name
        joblib.dump(model, path)
        return path

    def test_changed_predictions_have_counts_indices_transitions_and_snapshot_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.save_model(directory, 'baseline.joblib', 'Healthy')
            candidate = self.save_model(directory, 'candidate.joblib', 'Insomnia')
            report = compare_models([self.patient] * 3, baseline, candidate)
            self.assertEqual(report['baseline_sha256'], hashlib.sha256(baseline.read_bytes()).hexdigest())
            self.assertEqual(report['candidate_sha256'], hashlib.sha256(candidate.read_bytes()).hexdigest())
        self.assertEqual(report['changed_records'], 3)
        self.assertEqual(report['agreement_rate'], 0)
        self.assertEqual(report['transitions'], [{'baseline': 'Healthy', 'candidate': 'Insomnia', 'count': 3}])
        self.assertEqual([row['patient'] for row in report['changes']], [1, 2, 3])
        self.assertNotIn(self.patient['Occupation'], json.dumps(report))

    def test_same_shipped_model_has_complete_agreement(self):
        report = compare_models([self.patient] * 2, DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH)
        self.assertEqual(report['agreement_rate'], 1)
        self.assertEqual(report['unchanged_records'], 2)
        self.assertEqual(report['changes'], [])

    def test_invalid_patients_are_rejected_before_loading_models(self):
        with patch('src.compare_models.joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'Patient 2'):
                compare_models([self.patient, {}], DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH)
        load.assert_not_called()

    def test_cli_writes_report_before_failure_exit_when_predictions_change(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.save_model(directory, 'baseline.joblib', 'Healthy')
            candidate = self.save_model(directory, 'candidate.joblib', 'Insomnia')
            output = Path(directory) / 'comparison.json'
            with patch('sys.argv', ['compare', '--baseline', str(baseline), '--candidate', str(candidate),
                                    '--input-json', 'examples/patient.json', '--output', str(output), '--fail-on-change']):
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(json.loads(output.read_text())['changed_records'], 1)

    def test_csv_cli_stdout_is_json(self):
        with patch('sys.argv', ['compare', '--baseline', str(DEFAULT_MODEL_PATH), '--candidate', str(DEFAULT_MODEL_PATH),
                                '--input-csv', 'examples/patients.csv', '--fail-on-change']), redirect_stdout(StringIO()) as output:
            main()
        self.assertEqual(json.loads(output.getvalue())['total_records'], 2)

    def test_cli_rejects_output_collisions(self):
        for output in ('examples/patient.json', str(DEFAULT_MODEL_PATH)):
            with self.subTest(output=output), patch('sys.argv', ['compare', '--baseline', str(DEFAULT_MODEL_PATH),
                    '--candidate', str(DEFAULT_MODEL_PATH), '--input-json', 'examples/patient.json', '--output', output]):
                with self.assertRaisesRegex(SystemExit, 'differ'):
                    main()

    def test_model_failure_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            output.write_text('keep')
            with patch('sys.argv', ['compare', '--baseline', str(Path(directory) / 'missing.joblib'),
                    '--candidate', str(DEFAULT_MODEL_PATH), '--input-json', 'examples/patient.json', '--output', str(output)]):
                with self.assertRaisesRegex(SystemExit, 'Error:'):
                    main()
            self.assertEqual(output.read_text(), 'keep')
