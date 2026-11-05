from contextlib import redirect_stdout
from importlib import metadata
from io import StringIO
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.doctor import ASSETS, check_installation, main


class DoctorTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'src').mkdir()
        (root / 'models').mkdir()
        (root / 'examples').mkdir()
        for name in ASSETS:
            (root / 'src' / name).write_text('asset')
        for name in ('patient.json', 'patients.csv'):
            (root / 'examples' / name).write_text('example')
        (root / 'requirements.txt').write_text('example-package==1.2.3\n')
        model = root / 'models' / 'sleep_disorder_pipeline.joblib'
        model.write_bytes(b'not deserialized by default')
        return model

    def test_default_diagnostics_never_deserialize_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch('src.doctor.metadata.version', return_value='1.2.3'), patch('src.doctor._smoke_test') as smoke:
                report = check_installation(root=root)
            smoke.assert_not_called()
            self.assertTrue(report['healthy'])
            self.assertEqual(len(report['model_sha256']), 64)

    def test_missing_and_mismatched_dependencies_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            for effect in (metadata.PackageNotFoundError('example-package'), None):
                with patch('src.doctor.metadata.version', return_value='9.0', side_effect=effect), patch('src.doctor._smoke_test') as smoke:
                    report = check_installation(root=root, smoke_test=True)
                self.assertFalse(report['healthy'])
                self.assertEqual(report['checks'][-1]['status'], 'skipped')
                smoke.assert_not_called()
                dependency = next(row for row in report['checks'] if row['name'] == 'dependency:example-package')
                self.assertIn('1.2.3', dependency['detail'])

    def test_missing_files_and_empty_model_are_reported_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.fixture(root)
            model.write_bytes(b'')
            (root / 'src' / 'web_timer.js').unlink()
            with patch('src.doctor.metadata.version', return_value='1.2.3'):
                report = check_installation(root=root)
            failures = {row['name'] for row in report['checks'] if row['status'] == 'failed'}
            self.assertEqual(failures, {'runtime_files', 'model_file'})

    def test_smoke_test_failure_becomes_diagnostic_instead_of_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch('src.doctor.metadata.version', return_value='1.2.3'), patch('src.doctor._smoke_test', side_effect=ValueError('broken artifact')):
                report = check_installation(root=root, smoke_test=True)
            self.assertFalse(report['healthy'])
            self.assertIn('broken artifact', report['checks'][-1]['detail'])

    def test_shipped_model_passes_actual_prediction_smoke_test(self):
        report = check_installation(smoke_test=True)
        self.assertTrue(report['healthy'], report)
        self.assertEqual(report['checks'][-1]['name'], 'prediction')
        self.assertIn('Healthy', report['checks'][-1]['detail'])

    def test_cli_emits_report_before_failure_exit(self):
        with patch('src.doctor.check_installation', return_value={'healthy': False, 'model_sha256': None, 'checks': []}), \
                patch('sys.argv', ['doctor', '--json']), redirect_stdout(StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        self.assertFalse(json.loads(output.getvalue())['healthy'])

    def test_diagnostics_run_without_site_packages(self):
        result = subprocess.run([sys.executable, '-S', '-m', 'src.doctor', '--json'],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertFalse(report['healthy'])
        self.assertTrue(any(row['name'].startswith('dependency:') and row['status'] == 'failed'
                            for row in report['checks']))
        self.assertNotIn('Traceback', result.stderr)
