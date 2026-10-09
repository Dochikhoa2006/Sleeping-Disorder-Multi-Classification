import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import sklearn

from src.predict import load_patient_json, main, predict, predict_many

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sleep_disorder_pipeline.joblib"


class PredictionTests(unittest.TestCase):
    @staticmethod
    def sample_patient() -> dict:
        return {
            "Gender": "Male", "Age": 31, "Occupation": "Engineer",
            "Sleep Duration": 7.2, "Quality of Sleep": 8,
            "Physical Activity Level": 60, "Stress Level": 4,
            "BMI Category": "Normal", "Blood Pressure": "120/80",
            "Heart Rate": 70, "Daily Steps": 8_000,
        }

    def test_batch_predictions_preserve_input_order(self) -> None:
        first = self.sample_patient()
        second = {**first, "Age": 55, "Sleep Duration": 5.5}

        results = predict_many([first, second], MODEL_PATH)

        self.assertEqual(results, [predict(first, MODEL_PATH), predict(second, MODEL_PATH)])

    def test_batch_validation_identifies_invalid_record(self) -> None:
        invalid = {**self.sample_patient(), "Age": 0}

        with self.assertRaisesRegex(ValueError, "Patient 2: 'Age'"):
            predict_many([self.sample_patient(), invalid], MODEL_PATH)

    def test_cli_writes_batch_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.json"
            output_path = Path(directory) / "predictions.json"
            patients = [self.sample_patient(), {**self.sample_patient(), "Age": 55}]
            input_path.write_text(json.dumps(patients), encoding="utf-8")
            with patch("sys.argv", ["predict", "--input-json", str(input_path),
                                    "--output-json", str(output_path)]):
                main()

            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")),
                             predict_many(patients, MODEL_PATH))

    def test_rejects_empty_or_non_object_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.json"
            for contents, message in [("[]", "at least one"),
                                      ('[{"Age": 30}, 2]', "Patient 2")]:
                input_path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_patient_json(input_path)

    def test_saved_pipeline_matches_pinned_sklearn_version(self) -> None:
        pipeline = joblib.load(MODEL_PATH)
        expected_model = max(
            pipeline.cv_results_,
            key=lambda name: (
                pipeline.cv_results_[name]["f1_macro"],
                pipeline.cv_results_[name]["accuracy"],
            ),
        )

        self.assertEqual(
            pipeline.training_metadata_["scikit_learn_version"],
            sklearn.__version__,
        )
        self.assertEqual(pipeline.model_name_, expected_model)

    def test_saved_pipeline_predicts_known_shape(self) -> None:
        patient = {
            "Gender": "Male",
            "Age": 31,
            "Occupation": "Software Engineer",
            "Sleep Duration": 7.2,
            "Quality of Sleep": 8,
            "Physical Activity Level": 60,
            "Stress Level": 4,
            "BMI Category": "Normal",
            "Blood Pressure": "120/80",
            "Heart Rate": 70,
            "Daily Steps": 8_000,
        }

        result = predict(patient, MODEL_PATH)

        self.assertIn(result, {"Healthy", "Insomnia", "Sleep Apnea"})

    def test_saved_pipeline_handles_unseen_categories(self) -> None:
        patient = {
            "Gender": "Non-binary",
            "Age": 29,
            "Occupation": "Astronaut",
            "Sleep Duration": 6.5,
            "Quality of Sleep": 6,
            "Physical Activity Level": 35,
            "Stress Level": 7,
            "BMI Category": "Normal",
            "Blood Pressure": "128/78",
            "Heart Rate": 72,
            "Daily Steps": 6_500,
        }

        result = predict(patient, MODEL_PATH)

        self.assertIn(result, {"Healthy", "Insomnia", "Sleep Apnea"})

    def test_rejects_out_of_range_patient_values(self) -> None:
        patient = {
            "Gender": "Male",
            "Age": 31,
            "Occupation": "Engineer",
            "Sleep Duration": 7.2,
            "Quality of Sleep": 1_000,
            "Physical Activity Level": 60,
            "Stress Level": 4,
            "BMI Category": "Normal",
            "Blood Pressure": "120/80",
            "Heart Rate": 70,
            "Daily Steps": 8_000,
        }

        with self.assertRaisesRegex(ValueError, "Quality of Sleep"):
            predict(patient, MODEL_PATH)

    def test_rejects_impossible_blood_pressure(self) -> None:
        patient = {
            "Gender": "Male",
            "Age": 31,
            "Occupation": "Engineer",
            "Sleep Duration": 7.2,
            "Quality of Sleep": 8,
            "Physical Activity Level": 60,
            "Stress Level": 4,
            "BMI Category": "Normal",
            "Blood Pressure": "80/120",
            "Heart Rate": 70,
            "Daily Steps": 8_000,
        }

        with self.assertRaisesRegex(ValueError, "Systolic"):
            predict(patient, MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
