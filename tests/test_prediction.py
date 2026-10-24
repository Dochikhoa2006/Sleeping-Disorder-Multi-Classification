import unittest
import csv
import json
from contextlib import redirect_stdout
from io import StringIO
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import sklearn

from src.predict import PREDICTION_COLUMN, load_patient_csv, load_patient_json, main, predict, predict_many

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sleep_disorder_pipeline.joblib"


class PredictionTests(unittest.TestCase):
    def test_extreme_integers_are_validation_errors_before_model_loading(self) -> None:
        numeric_fields = ("Age", "Sleep Duration", "Quality of Sleep",
                          "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps")
        for feature in numeric_fields:
            for value in (10 ** 1000, -(10 ** 1000)):
                with self.subTest(feature=feature, negative=value < 0), \
                        patch("src.predict.joblib.load") as load_model:
                    with self.assertRaisesRegex(ValueError, f"Patient 1: '{feature}'.*must be a number"):
                        predict_many([{**self.sample_patient(), feature: value}], MODEL_PATH)
                load_model.assert_not_called()

    def test_cli_reports_extreme_json_integer_without_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patient.json"
            path.write_text(json.dumps({**self.sample_patient(), "Age": 10 ** 1000}), encoding="utf-8")
            with patch("sys.argv", ["predict", "--input-json", str(path), "--validate-only"]):
                with self.assertRaisesRegex(SystemExit, "Error: Patient 1: 'Age'.*1 to 120"):
                    main()

    def test_cli_rejects_output_overwriting_input_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.json"
            input_path.write_text(json.dumps(self.sample_patient()), encoding="utf-8")
            original = input_path.read_bytes()
            for output_path in (input_path, Path(directory) / "alias.json"):
                if output_path != input_path:
                    output_path.symlink_to(input_path)
                with patch("sys.argv", ["predict", "--input-json", str(input_path),
                                        "--output-json", str(output_path)]):
                    with self.assertRaisesRegex(SystemExit, "output path must differ"):
                        main()
                self.assertEqual(input_path.read_bytes(), original)
            with patch("sys.argv", ["predict", "--input-json", str(input_path),
                                    "--output-json", str(MODEL_PATH)]):
                with self.assertRaisesRegex(SystemExit, "output path must differ"):
                    main()

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

    def test_validate_only_accepts_json_and_csv_without_loading_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patients = [self.sample_patient(), {**self.sample_patient(), "Age": 55}]
            json_path = root / "patients.json"
            json_path.write_text(json.dumps(patients), encoding="utf-8")
            csv_path = root / "patients.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(patients[0]))
                writer.writeheader()
                writer.writerows(patients)

            for option, path in (("--input-json", json_path), ("--input-csv", csv_path)):
                with self.subTest(option=option):
                    output = StringIO()
                    with patch("sys.argv", ["predict", option, str(path), "--validate-only",
                                            "--model", str(root / "missing.joblib")]), \
                            patch("src.predict.joblib.load") as load_model, redirect_stdout(output):
                        main()
                    load_model.assert_not_called()
                    self.assertEqual(output.getvalue(), "Validated 2 patient record(s).\n")

    def test_validate_only_reports_invalid_record_without_loading_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patients.json"
            path.write_text(json.dumps([
                self.sample_patient(), {**self.sample_patient(), "Age": 0},
            ]), encoding="utf-8")
            with patch("sys.argv", ["predict", "--input-json", str(path), "--validate-only"]), \
                    patch("src.predict.joblib.load") as load_model:
                with self.assertRaisesRegex(SystemExit, "Patient 2: 'Age'"):
                    main()
            load_model.assert_not_called()

    def test_validate_only_rejects_output_options_without_creating_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "output"
            for option in ("--output-json", "--output-csv"):
                with self.subTest(option=option), patch("sys.argv", [
                    "predict", "--validate-only", option, str(output_path),
                ]):
                    with self.assertRaisesRegex(SystemExit, "cannot be combined with output"):
                        main()
                self.assertFalse(output_path.exists())

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

    def test_cli_csv_preserves_rows_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.csv"
            output_path = Path(directory) / "predictions.csv"
            patients = [self.sample_patient(), {**self.sample_patient(), "Age": 55}]
            rows = [{**{key: str(value) for key, value in patient.items()},
                     "Record ID": str(index)} for index, patient in enumerate(patients)]
            with input_path.open("w", newline="", encoding="utf-8") as input_file:
                writer = csv.DictWriter(input_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with patch("sys.argv", ["predict", "--input-csv", str(input_path),
                                    "--output-csv", str(output_path)]):
                main()

            with output_path.open(newline="", encoding="utf-8") as output_file:
                results = list(csv.DictReader(output_file))
            self.assertEqual(
                results,
                [{**row, PREDICTION_COLUMN: prediction} for row, prediction
                 in zip(rows, predict_many(patients, MODEL_PATH))],
            )

    def test_csv_reports_source_row_and_does_not_write_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.csv"
            output_path = Path(directory) / "predictions.csv"
            rows = [self.sample_patient(), {**self.sample_patient(), "Age": "invalid"}]
            with input_path.open("w", newline="", encoding="utf-8") as input_file:
                writer = csv.DictWriter(input_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "CSV row 3: 'Age'"):
                load_patient_csv(input_path)
            with patch("sys.argv", ["predict", "--input-csv", str(input_path),
                                    "--output-csv", str(output_path)]):
                with self.assertRaisesRegex(SystemExit, "CSV row 3: 'Age'"):
                    main()
            self.assertFalse(output_path.exists())

    def test_csv_rejects_duplicate_headers_and_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.csv"
            input_path.write_text("Age,Age\n31,32\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_patient_csv(input_path)
            input_path.write_text(",".join(self.sample_patient()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one patient"):
                load_patient_csv(input_path)

    def test_rejects_empty_or_non_object_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.json"
            for contents, message in [("[]", "at least one"),
                                      ('[{"Age": 30}, 2]', "Patient 2")]:
                input_path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_patient_json(input_path)

    def test_json_rejects_duplicate_fields_in_objects_and_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "patients.json"
            for contents in (
                '{"Age": 0, "Age": 31}',
                '[{"Age": 30}, {"Age": 0, "Age": 31}]',
                '{"Age": 31, "Age": 31}',
                '{"Age": 0, "\\u0041ge": 31}',
            ):
                with self.subTest(contents=contents):
                    input_path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Duplicate JSON field: 'Age'"):
                        load_patient_json(input_path)

    def test_duplicate_json_fields_stop_cli_before_model_loading_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "patients.json"
            patient = json.dumps(self.sample_patient())
            input_path.write_text(patient[:-1] + ', "Age": 55}', encoding="utf-8")
            output_path = root / "predictions.json"
            output_path.write_bytes(b"existing predictions")
            for options in (
                ["--output-json", str(output_path)], ["--validate-only"],
            ):
                with self.subTest(options=options), patch("sys.argv", [
                    "predict", "--input-json", str(input_path), *options,
                ]), patch("src.predict.joblib.load") as load_model:
                    with self.assertRaisesRegex(SystemExit, "Duplicate JSON field: 'Age'"):
                        main()
                load_model.assert_not_called()
                self.assertEqual(output_path.read_bytes(), b"existing predictions")

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
