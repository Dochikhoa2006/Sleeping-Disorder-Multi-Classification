from contextlib import redirect_stdout
import copy
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.predict import DEFAULT_MODEL_PATH
from src.receipt import create_receipt
from src.verify_receipt import validate_receipt, verify_receipt, main


class VerifyReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patient = json.loads(Path('examples/patient.json').read_text())
        cls.receipt = create_receipt([cls.patient])

    def test_original_receipt_replays_successfully(self):
        report = verify_receipt(self.receipt, [self.patient])
        self.assertTrue(report['verified'])
        self.assertTrue(report['replayed'])
        self.assertIsNone(report['checks']['source_file_hash'])

    def test_changed_model_or_inputs_skip_deserialization(self):
        wrong_model = copy.deepcopy(self.receipt)
        wrong_model['model']['sha256'] = '0' * 64
        for receipt, patient in ((wrong_model, self.patient),
                                 (self.receipt, {**self.patient, 'Age': 55})):
            with self.subTest(patient=patient), patch('src.verify_receipt.create_receipt_from_bytes') as replay:
                report = verify_receipt(receipt, [patient])
            self.assertFalse(report['verified'])
            self.assertFalse(report['replayed'])
            replay.assert_not_called()

    def test_modified_prediction_and_model_metadata_are_detected(self):
        altered = copy.deepcopy(self.receipt)
        altered['predictions'][0]['prediction'] = 'Insomnia'
        report = verify_receipt(altered, [self.patient])
        self.assertEqual(report['prediction_mismatches'], [1])
        self.assertFalse(report['verified'])
        altered = copy.deepcopy(self.receipt)
        altered['model']['name'] = 'altered name'
        self.assertFalse(verify_receipt(altered, [self.patient])['checks']['model_metadata'])

    def test_source_byte_changes_fail_even_with_matching_logical_inputs(self):
        receipt = copy.deepcopy(self.receipt)
        receipt['source_sha256'] = '0' * 64
        report = verify_receipt(receipt, [self.patient], source_sha256='1' * 64)
        self.assertFalse(report['verified'])
        self.assertTrue(report['checks']['predictions'])
        self.assertFalse(report['checks']['source_file_hash'])

    def test_malformed_receipt_schema_is_rejected(self):
        changes = [{'schema_version': True}, {'count': 2}, {'generated_at': 'not-a-date'},
                   {'predictions': [{'patient': 1, 'input_sha256': 'bad', 'prediction': 'Healthy'}]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_receipt({**self.receipt, **change})

    def test_cli_writes_failed_verification_before_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / 'receipt.json', root / 'verification.json'
            altered = copy.deepcopy(self.receipt)
            altered['predictions'][0]['prediction'] = 'Insomnia'
            path.write_text(json.dumps(altered))
            with patch('sys.argv', ['verify', '--receipt', str(path), '--input-json', 'examples/patient.json', '--output', str(output)]):
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(json.loads(output.read_text())['prediction_mismatches'], [1])

    def test_cli_valid_receipt_and_output_collision_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            receipt = copy.deepcopy(self.receipt)
            receipt['source_sha256'] = hashlib.sha256(Path('examples/patient.json').read_bytes()).hexdigest()
            path.write_text(json.dumps(receipt))
            argv = ['verify', '--receipt', str(path), '--input-json', 'examples/patient.json']
            with patch('sys.argv', argv), redirect_stdout(StringIO()) as output:
                main()
            self.assertTrue(json.loads(output.getvalue())['verified'])
            for destination in (path, Path('examples/patient.json'), DEFAULT_MODEL_PATH):
                with patch('sys.argv', argv + ['--output', str(destination)]):
                    with self.assertRaisesRegex(SystemExit, 'differ'):
                        main()
