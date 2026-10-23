import gzip
import hashlib
import json
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
    def test_report_counts_equivalent_unseen_profiles_without_dropping_rows(self) -> None:
        first = {**self.new_patient(97), TARGET_COLUMN: "Healthy", "Person ID": 1}
        equivalent = {**first, "Person ID": 2, "BMI Category": "Normal Weight",
                      "Blood Pressure": " 120 / 80 "}
        different = {**self.new_patient(98), TARGET_COLUMN: "Healthy", "Person ID": 3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "patients.csv"
            pd.DataFrame([first, equivalent, different]).to_csv(data_path, index=False)
            report = evaluate_external(data_path, DEFAULT_MODEL_PATH, root / "report.json")
            self.assertEqual(report["evaluated_rows"], 3)
            self.assertEqual(report["unique_evaluated_profiles"], 2)
            self.assertEqual(report["duplicate_evaluated_rows"], 1)
            self.assertEqual(sum(sum(row) for row in report["confusion_matrix"]), 3)

    def test_gzip_evaluation_matches_plain_csv_and_hashes_original_file(self) -> None:
        row = {**self.new_patient(97), TARGET_COLUMN: "Healthy"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain_path = root / "patients.csv"
            gzip_path = root / "patients.csv.gz"
            pd.DataFrame([row]).to_csv(plain_path, index=False)
            gzip_path.write_bytes(gzip.compress(plain_path.read_bytes()))
            plain = evaluate_external(plain_path, DEFAULT_MODEL_PATH, root / "plain.json")
            compressed = evaluate_external(gzip_path, DEFAULT_MODEL_PATH, root / "gzip.json")
            for key in ("accuracy", "macro_f1", "confusion_matrix", "class_row_counts", "evaluated_rows"):
                self.assertEqual(plain[key], compressed[key])
            self.assertEqual(compressed["dataset_sha256"], hashlib.sha256(gzip_path.read_bytes()).hexdigest())
            self.assertNotEqual(plain["dataset_sha256"], compressed["dataset_sha256"])

    def test_report_accounts_for_class_coverage_after_overlap_filtering(self) -> None:
        pipeline = joblib.load(DEFAULT_MODEL_PATH)
        overlapping = {**self.new_patient(96), TARGET_COLUMN: "Insomnia"}
        unseen = {**self.new_patient(97), TARGET_COLUMN: "Healthy"}
        pipeline.training_feature_hashes_ = set(feature_fingerprints(pd.DataFrame([overlapping])))
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "external.csv"
            output_path = Path(directory) / "report.json"
            pd.DataFrame([overlapping, unseen, unseen]).to_csv(data_path, index=False)
            with patch("src.evaluation.joblib.load", return_value=pipeline):
                report = evaluate_external(data_path, DEFAULT_MODEL_PATH, output_path)
            self.assertEqual(report["class_row_counts"], {
                "Healthy": {"total_rows": 2, "training_overlap_rows": 0, "evaluated_rows": 2},
                "Insomnia": {"total_rows": 1, "training_overlap_rows": 1, "evaluated_rows": 0},
                "Sleep Apnea": {"total_rows": 0, "training_overlap_rows": 0, "evaluated_rows": 0},
            })
            for key in ("total_rows", "training_overlap_rows", "evaluated_rows"):
                self.assertEqual(sum(counts[key] for counts in report["class_row_counts"].values()), report[key])
            self.assertEqual(json.loads(output_path.read_text())["class_row_counts"], report["class_row_counts"])

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
            self.assertEqual(report["dataset_sha256"], hashlib.sha256(data_path.read_bytes()).hexdigest())
            self.assertEqual(report["model_sha256"], hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), report)

    def test_report_fingerprints_change_when_input_files_change(self) -> None:
        pipeline = joblib.load(DEFAULT_MODEL_PATH)
        pipeline.training_feature_hashes_ = set()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "external.csv"
            model_path = root / "model.joblib"
            output_path = root / "evaluation.json"
            row = self.new_patient(97)
            row[TARGET_COLUMN] = predict(row, DEFAULT_MODEL_PATH)
            pd.DataFrame([row]).to_csv(data_path, index=False)
            joblib.dump(pipeline, model_path)
            first = evaluate_external(data_path, model_path, output_path)

            # A different CSV serialization preserves predictions but changes provenance.
            with data_path.open("a", encoding="utf-8") as data_file:
                data_file.write("\n")
            pipeline.model_name_ = "Updated model artifact"
            joblib.dump(pipeline, model_path)
            second = evaluate_external(data_path, model_path, output_path)

            self.assertNotEqual(first["dataset_sha256"], second["dataset_sha256"])
            self.assertNotEqual(first["model_sha256"], second["model_sha256"])
            self.assertEqual(first["accuracy"], second["accuracy"])
            self.assertEqual(first["confusion_matrix"], second["confusion_matrix"])
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), second)

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
