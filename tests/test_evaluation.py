import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation import evaluate_external
from src.predict import predict
from src.preprocessing import TARGET_COLUMN
from src.train import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, split_training_holdout


class ExternalEvaluationTests(unittest.TestCase):
    @staticmethod
    def new_patient(age: int) -> dict:
        return {
            "Gender": "Male", "Age": age, "Occupation": "Engineer",
            "Sleep Duration": 7.2, "Quality of Sleep": 8,
            "Physical Activity Level": 60, "Stress Level": 4,
            "BMI Category": "Normal", "Blood Pressure": "120/80",
            "Heart Rate": 70, "Daily Steps": 8_000,
        }

    def test_external_report_excludes_training_profiles(self) -> None:
        dataset = pd.read_csv(DEFAULT_DATA_PATH)
        features = dataset.drop(columns=[TARGET_COLUMN])
        target = dataset[TARGET_COLUMN].fillna("Healthy")
        train_indices, _ = split_training_holdout(features, target)
        overlapping = features.iloc[train_indices[0]].to_dict()
        overlapping["Age"] = float(overlapping["Age"])
        overlapping[TARGET_COLUMN] = target.iloc[train_indices[0]]
        unseen = self.new_patient(97)
        unseen[TARGET_COLUMN] = predict(unseen, DEFAULT_MODEL_PATH)

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "external.csv"
            output_path = Path(directory) / "evaluation.json"
            pd.DataFrame([overlapping, unseen]).to_csv(data_path, index=False)
            report = evaluate_external(data_path, DEFAULT_MODEL_PATH, output_path)

            self.assertEqual(report["training_overlap_rows"], 1)
            self.assertEqual(report["evaluated_rows"], 1)
            self.assertEqual(report["accuracy"], 1.0)
            self.assertTrue(output_path.exists())

    def test_external_report_rejects_only_overlapping_rows(self) -> None:
        dataset = pd.read_csv(DEFAULT_DATA_PATH)
        features = dataset.drop(columns=[TARGET_COLUMN])
        target = dataset[TARGET_COLUMN].fillna("Healthy")
        train_indices, _ = split_training_holdout(features, target)
        row = features.iloc[train_indices[0]].to_dict()
        row[TARGET_COLUMN] = target.iloc[train_indices[0]]
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "external.csv"
            output_path = Path(directory) / "evaluation.json"
            pd.DataFrame([row]).to_csv(data_path, index=False)
            with self.assertRaisesRegex(ValueError, "no unseen rows"):
                evaluate_external(data_path, DEFAULT_MODEL_PATH, output_path)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
