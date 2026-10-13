import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from src.evaluation import evaluate_external, feature_fingerprints
from src.predict import predict
from src.preprocessing import TARGET_COLUMN
from src.train import DEFAULT_MODEL_PATH


class ExternalEvaluationTests(unittest.TestCase):
    def test_output_cannot_overwrite_data_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "external.csv"
            data_path.write_text("Sleep Disorder\nHealthy\n", encoding="utf-8")
            original = data_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "output path must differ"):
                evaluate_external(data_path, DEFAULT_MODEL_PATH, data_path)
            self.assertEqual(data_path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "output path must differ"):
                evaluate_external(data_path, DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH)

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
        overlapping = self.new_patient(96)
        overlapping[TARGET_COLUMN] = predict(overlapping, DEFAULT_MODEL_PATH)
        unseen = self.new_patient(97)
        unseen[TARGET_COLUMN] = predict(unseen, DEFAULT_MODEL_PATH)
        pipeline = joblib.load(DEFAULT_MODEL_PATH)
        pipeline.training_feature_hashes_ = set(feature_fingerprints(pd.DataFrame([overlapping])))

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "external.csv"
            output_path = Path(directory) / "evaluation.json"
            pd.DataFrame([overlapping, unseen]).to_csv(data_path, index=False)
            with patch("src.evaluation.joblib.load", return_value=pipeline):
                report = evaluate_external(data_path, DEFAULT_MODEL_PATH, output_path)

            self.assertEqual(report["training_overlap_rows"], 1)
            self.assertEqual(report["evaluated_rows"], 1)
            self.assertEqual(report["accuracy"], 1.0)
            self.assertTrue(output_path.exists())

    def test_external_report_rejects_only_overlapping_rows(self) -> None:
        row = self.new_patient(96)
        row[TARGET_COLUMN] = predict(row, DEFAULT_MODEL_PATH)
        pipeline = joblib.load(DEFAULT_MODEL_PATH)
        pipeline.training_feature_hashes_ = set(feature_fingerprints(pd.DataFrame([row])))
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "external.csv"
            output_path = Path(directory) / "evaluation.json"
            pd.DataFrame([row]).to_csv(data_path, index=False)
            with patch("src.evaluation.joblib.load", return_value=pipeline):
                with self.assertRaisesRegex(ValueError, "no unseen rows"):
                    evaluate_external(data_path, DEFAULT_MODEL_PATH, output_path)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
