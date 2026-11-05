from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.benchmark import benchmark_model, main
from src.predict import DEFAULT_MODEL_PATH


class BenchmarkTests(unittest.TestCase):
    def test_metrics_use_measured_calls_and_exclude_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'model.joblib'
            model_path.write_bytes(b'model snapshot')
            model = Mock()
            model.model_name_ = 'Fixture'
            model.predict.side_effect = lambda frame: ['Healthy'] * len(frame)
            with patch('src.benchmark.joblib.load', return_value=model), \
                    patch('src.benchmark.perf_counter', side_effect=[1., 1.05, 2., 2.01, 3., 3.02]):
                report = benchmark_model(model_path, batch_sizes=[1], runs=2, warmup=3)
        result = report['batches'][0]
        self.assertEqual(model.predict.call_count, 5)
        self.assertAlmostEqual(report['model_read_and_load_ms'], 50)
        self.assertAlmostEqual(result['mean_ms'], 15)
        self.assertAlmostEqual(result['median_ms'], 15)
        self.assertAlmostEqual(result['p95_ms'], 19.5)
        self.assertAlmostEqual(result['rows_per_second'], 1000 / 15)
        self.assertEqual(report['model_sha256'], hashlib.sha256(b'model snapshot').hexdigest())

    def test_shipped_model_runs_on_multiple_batch_sizes(self):
        report = benchmark_model(batch_sizes=[1, 3], runs=2, warmup=0)
        self.assertEqual([row['rows'] for row in report['batches']], [1, 3])
        self.assertTrue(all(len(row['samples_ms']) == 2 for row in report['batches']))
        self.assertTrue(all(row['rows_per_second'] > 0 for row in report['batches']))
        self.assertEqual(report['input_source'], 'synthetic demo patients')

    def test_invalid_options_rejected_before_loading_model(self):
        options = [{'batch_sizes': []}, {'batch_sizes': [0]}, {'batch_sizes': [1, 1]},
                   {'batch_sizes': [10001]}, {'runs': 0}, {'warmup': 11}, {'seed': -1}]
        for values in options:
            with self.subTest(values=values), patch('src.benchmark.joblib.load') as load:
                with self.assertRaises(ValueError):
                    benchmark_model(**values)
                load.assert_not_called()

    def test_wrong_prediction_count_is_rejected(self):
        with patch('src.benchmark.joblib.load') as load:
            load.return_value.predict.return_value = []
            with self.assertRaisesRegex(ValueError, 'one prediction'):
                benchmark_model(batch_sizes=[1], runs=1, warmup=1)

    def test_cli_json_stdout_and_model_collision_protection(self):
        with patch('sys.argv', ['benchmark', '--batch-sizes', '1', '--runs', '1', '--warmup', '0']), redirect_stdout(StringIO()) as output:
            main()
        self.assertEqual(json.loads(output.getvalue())['batches'][0]['rows'], 1)
        with patch('sys.argv', ['benchmark', '--output', str(DEFAULT_MODEL_PATH)]):
            with self.assertRaisesRegex(SystemExit, 'differ'):
                main()

    def test_invalid_run_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            path.write_text('keep')
            with patch('sys.argv', ['benchmark', '--runs', '0', '--output', str(path)]):
                with self.assertRaisesRegex(SystemExit, 'Error:'):
                    main()
            self.assertEqual(path.read_text(), 'keep')
