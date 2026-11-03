import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation import feature_fingerprints
from src.predict import load_patient_csv
from src.split_data import split_dataset
from src.preprocessing import TARGET_COLUMN
import pandas as pd


class SplitDataTests(unittest.TestCase):
    def write_source(self, root):
        patient = json.loads(Path('examples/patient.json').read_text())
        rows = []
        for label in ('Healthy', 'Insomnia', 'Sleep Apnea'):
            for index in range(6):
                for repeat in range(2):
                    rows.append({**patient, 'Age': 20 + index,
                                 'Occupation': label + ' worker', TARGET_COLUMN: label,
                                 'Record ID': f'{label}-{index}-{repeat}', 'Notes': 'comma, newline\ntext'})
        path = root / 'source.csv'
        with path.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path, rows

    def test_splits_profiles_without_leakage_and_preserves_all_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, original = self.write_source(root)
            destination = root / 'split'
            report = split_dataset(source, destination, folds=3)
            _, train_rows, train = load_patient_csv(destination / 'train.csv')
            _, test_rows, test = load_patient_csv(destination / 'test.csv')
            self.assertFalse(set(feature_fingerprints(pd.DataFrame(train))) & set(feature_fingerprints(pd.DataFrame(test))))
            self.assertEqual(len(train) + len(test), len(original))
            self.assertEqual({r['Record ID'] for r in train_rows + test_rows}, {r['Record ID'] for r in original})
            self.assertTrue(all(row['Notes'] == 'comma, newline\ntext' for row in train_rows + test_rows))
            self.assertEqual(report, json.loads((destination / 'manifest.json').read_text()))
            for name in ('train', 'test'):
                self.assertEqual(report['partitions'][name]['sha256'], hashlib.sha256((destination / (name + '.csv')).read_bytes()).hexdigest())

    def test_same_seed_produces_identical_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.write_source(root)
            for name in ('one', 'two'):
                split_dataset(source, root / name, folds=3, seed=42)
            for filename in ('train.csv', 'test.csv', 'manifest.json'):
                self.assertEqual((root / 'one' / filename).read_bytes(), (root / 'two' / filename).read_bytes())

    def test_existing_directory_is_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.write_source(root)
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                split_dataset(source, root)
            self.assertEqual(source.read_bytes(), before)

    def test_bad_configuration_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.write_source(root)
            for options in ({'folds': 1}, {'fold': 5}, {'folds': 20}, {'seed': -1}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    split_dataset(source, root / 'output', **options)
                self.assertFalse((root / 'output').exists())

    def test_missing_or_conflicting_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.write_source(root)
            text = source.read_text()
            for changed in (text.replace(',Healthy,', ',,', 1), text.replace(',Healthy,', ',Insomnia,', 1)):
                source.write_text(changed)
                with self.assertRaises(ValueError):
                    split_dataset(source, root / 'output', folds=3)
                self.assertFalse((root / 'output').exists())
