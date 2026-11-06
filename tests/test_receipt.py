from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib

from src.predict import DEFAULT_MODEL_PATH, predict_many
from src.receipt import create_receipt, input_fingerprint, main


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(Path('examples/patient.json').read_text())

    def test_receipt_matches_model_snapshot_and_predictions_without_input_values(self):
        patients = [self.patient, {**self.patient, 'Age': 55}]
        real_load = joblib.load
        snapshots = []
        def capture(stream):
            snapshots.append(stream.getvalue())
            return real_load(stream)
        with patch('src.receipt.joblib.load', side_effect=capture):
            receipt = create_receipt(patients)
        self.assertEqual(snapshots, [DEFAULT_MODEL_PATH.read_bytes()])
        self.assertEqual(receipt['model']['sha256'], hashlib.sha256(snapshots[0]).hexdigest())
        self.assertEqual([row['prediction'] for row in receipt['predictions']], predict_many(patients))
        self.assertEqual(receipt['count'], 2)
        self.assertNotIn(self.patient['Occupation'], json.dumps(receipt))
        self.assertNotEqual(receipt['predictions'][0]['input_sha256'], receipt['predictions'][1]['input_sha256'])

    def test_fingerprint_ignores_key_order_metadata_and_integer_float_format(self):
        reordered = dict(reversed(list(self.patient.items())))
        reordered['Age'] = float(reordered['Age'])
        reordered['Notes'] = 'not part of model input'
        self.assertEqual(input_fingerprint(self.patient), input_fingerprint(reordered))

    def test_invalid_batch_never_loads_a_model(self):
        with patch('src.receipt.joblib.load') as load:
            with self.assertRaisesRegex(ValueError, 'Patient 2'):
                create_receipt([self.patient, {}])
        load.assert_not_called()

    def test_csv_cli_receipt_reports_source_hash_and_order(self):
        source = Path('examples/patients.csv')
        with patch('sys.argv', ['receipt', '--input-csv', str(source)]), redirect_stdout(StringIO()) as output:
            main()
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual([row['patient'] for row in receipt['predictions']], [1, 2])

    def test_cli_output_is_atomic_and_cannot_overwrite_inputs(self):
        source = Path('examples/patient.json')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'receipt.json'
            with patch('sys.argv', ['receipt', '--input-json', str(source), '--output', str(output)]):
                main()
            self.assertEqual(json.loads(output.read_text())['count'], 1)
            output.write_text('keep')
            with patch('sys.argv', ['receipt', '--input-json', str(source), '--model', str(Path(directory) / 'missing.joblib'), '--output', str(output)]):
                with self.assertRaisesRegex(SystemExit, 'Error:'):
                    main()
            self.assertEqual(output.read_text(), 'keep')
        for target in (source, DEFAULT_MODEL_PATH):
            with patch('sys.argv', ['receipt', '--input-json', str(source), '--output', str(target)]):
                with self.assertRaisesRegex(SystemExit, 'differ'):
                    main()
