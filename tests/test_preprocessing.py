import unittest

import pandas as pd
import joblib

from src.preprocessing import SleepFeatureEngineer
from src.predict import DEFAULT_MODEL_PATH


def patient_frame(blood_pressure: str = "120/80") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Person ID": 999,
                "Gender": "Female",
                "Age": 35,
                "Occupation": "Engineer",
                "Sleep Duration": 7.5,
                "Quality of Sleep": 8,
                "Physical Activity Level": 45,
                "Stress Level": 4,
                "BMI Category": "Normal Weight",
                "Blood Pressure": blood_pressure,
                "Heart Rate": 68,
                "Daily Steps": 8_000,
            }
        ]
    )


class SleepFeatureEngineerTests(unittest.TestCase):
    def test_categorical_whitespace_is_normalized_without_mutating_input(self) -> None:
        clean = patient_frame()
        padded = clean.copy()
        for column in ("Gender", "Occupation", "BMI Category"):
            padded[column] = " \t" + padded[column] + "\n "
        original = padded.copy(deep=True)
        transformer = SleepFeatureEngineer()
        pd.testing.assert_frame_equal(transformer.transform(clean), transformer.transform(padded))
        pd.testing.assert_frame_equal(padded, original)

    def test_saved_model_preprocessing_matches_for_padded_categories(self) -> None:
        model = joblib.load(DEFAULT_MODEL_PATH)
        clean = patient_frame()
        padded = clean.copy()
        for column in ("Gender", "Occupation", "BMI Category"):
            padded[column] = " " + padded[column] + " "
        self.assertEqual(model.predict(clean).tolist(), model.predict(padded).tolist())
        self.assertEqual(model[:-1].transform(clean).tolist(), model[:-1].transform(padded).tolist())

    def test_duplicate_input_columns_are_rejected_before_feature_engineering(self) -> None:
        for column in ("Age", "Blood Pressure", "Person ID"):
            frame = patient_frame()
            duplicate = pd.concat([frame, frame[[column]]], axis=1)
            original = duplicate.copy(deep=True)
            for operation in ("fit", "transform"):
                with self.subTest(column=column, operation=operation):
                    with self.assertRaisesRegex(ValueError, f"Duplicate input column.*{column}"):
                        getattr(SleepFeatureEngineer(), operation)(duplicate)
                    pd.testing.assert_frame_equal(duplicate, original)

    def test_engineered_feature_names_match_transformed_columns(self) -> None:
        transformer = SleepFeatureEngineer()
        frame = patient_frame()
        transformed = transformer.fit_transform(frame)
        self.assertEqual(transformer.get_feature_names_out().tolist(), transformed.columns.tolist())
        self.assertEqual(transformer.get_feature_names_out(frame.columns).tolist(), transformed.columns.tolist())
        self.assertNotIn("Person ID", transformer.get_feature_names_out())

    def test_saved_preprocessing_pipeline_exposes_names_for_every_output(self) -> None:
        preprocessing = joblib.load(DEFAULT_MODEL_PATH)[:-1]
        transformed = preprocessing.transform(patient_frame())
        names = preprocessing.get_feature_names_out().tolist()
        self.assertEqual(len(names), transformed.shape[1])
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("numeric__Systolic Blood Pressure", names)
        self.assertIn("categorical__BMI Category_Normal", names)

    def test_drops_identifier_and_parses_blood_pressure(self) -> None:
        transformed = SleepFeatureEngineer().fit_transform(patient_frame())

        self.assertNotIn("Person ID", transformed.columns)
        self.assertEqual(transformed.loc[0, "Systolic Blood Pressure"], 120)
        self.assertEqual(transformed.loc[0, "Diastolic Blood Pressure"], 80)
        self.assertEqual(
            transformed.loc[0, "Blood Pressure Category"],
            "stage 1 hypertension",
        )
        self.assertEqual(transformed.loc[0, "BMI Category"], "Normal")

    def test_severe_hypertension_is_not_shadowed_by_stage_two(self) -> None:
        transformed = SleepFeatureEngineer().fit_transform(patient_frame("180/120"))

        self.assertEqual(
            transformed.loc[0, "Blood Pressure Category"],
            "severe hypertension",
        )

    def test_malformed_blood_pressure_becomes_measurement_error(self) -> None:
        transformed = SleepFeatureEngineer().fit_transform(patient_frame("unknown"))

        self.assertEqual(
            transformed.loc[0, "Blood Pressure Category"],
            "measurement error",
        )
        self.assertTrue(
            pd.isna(transformed.loc[0, "Systolic Blood Pressure"])
        )

    def test_rejects_missing_features(self) -> None:
        incomplete = patient_frame().drop(columns=["Gender"])

        with self.assertRaisesRegex(ValueError, "Gender"):
            SleepFeatureEngineer().fit_transform(incomplete)


if __name__ == "__main__":
    unittest.main()
