import csv
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.prediction_diff import compare_prediction_exports
from src.predict import PREDICTION_COLUMN


class PredictionDiffTests(unittest.TestCase):
    def export(self, records):
        patient = json.loads(Path('examples/patient.json').read_text())
        rows = [{**patient, 'Record ID': str(index), PREDICTION_COLUMN: 'Healthy', **changes}
                for index, changes in enumerate(records)]
        output = StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode('utf-8-sig')

    def test_ids_match_across_reordering_and_equivalent_input_formats(self):
        first = self.export([{'Record ID': 'A'}, {'Record ID': 'B'}])
        second = self.export([{'Record ID': 'B', 'Sleep Duration': '7.20'},
                              {'Record ID': 'A', 'BMI Category': ' Normal Weight '}])
        with patch('joblib.load') as load:
            report = compare_prediction_exports(first, second)
        load.assert_not_called()
        self.assertEqual(report['counts']['unchanged'], 2)
        self.assertEqual([row['Record ID'] for row in report['records']], ['A', 'B'])

    def test_distinguishes_prediction_change_input_change_addition_and_removal(self):
        first = self.export([{'Record ID': 'A'}, {'Record ID': 'B'}, {'Record ID': 'C'}, {'Record ID': 'D'}])
        second = self.export([{'Record ID': 'A', PREDICTION_COLUMN: 'Insomnia'},
                              {'Record ID': 'B', 'Age': 45}, {'Record ID': 'D'}, {'Record ID': 'E'}])
        report = compare_prediction_exports(first, second)
        self.assertEqual(report['counts'], {'unchanged': 1, 'prediction_changed': 1,
                                           'inputs_changed': 1, 'added': 1, 'removed': 1})
        self.assertEqual(report['records'][0]['Candidate Prediction'], 'Insomnia')
        self.assertEqual(report['records'][2]['Candidate Prediction'], '')
        self.assertEqual(report['records'][-1]['Record ID'], 'E')

    def test_input_change_takes_precedence_when_prediction_also_changes(self):
        first = self.export([{}])
        second = self.export([{'Age': 50, PREDICTION_COLUMN: 'Sleep Apnea'}])
        report = compare_prediction_exports(first, second)
        self.assertEqual(report['records'][0]['Status'], 'inputs_changed')
        self.assertEqual(report['records'][0]['Candidate Prediction'], 'Sleep Apnea')

    def test_duplicate_blank_and_missing_ids_are_rejected(self):
        for bad in (self.export([{'Record ID': 'A'}, {'Record ID': 'A'}]),
                    self.export([{'Record ID': ' '}]),
                    self.export([{}]).replace(b'Record ID', b'Other ID')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'Record ID'):
                compare_prediction_exports(self.export([{}]), bad)
