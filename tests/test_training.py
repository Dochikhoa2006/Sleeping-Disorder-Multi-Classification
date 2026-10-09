import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocessing import RAW_FEATURES, build_pipeline
from src.train import make_feature_groups


class TrainingTests(unittest.TestCase):
    def test_duplicate_features_share_a_validation_group(self) -> None:
        base = {
            "Gender": "Female", "Age": 30, "Occupation": "Engineer",
            "Sleep Duration": 7.5, "Quality of Sleep": 8,
            "Physical Activity Level": 45, "Stress Level": 4,
            "BMI Category": "Normal", "Blood Pressure": "120/80",
            "Heart Rate": 68, "Daily Steps": 8_000,
        }
        features = pd.DataFrame([
            {**base, "Person ID": 1, "Unused Column": "first"},
            {**base, "Person ID": 2, "Unused Column": "second",
             "BMI Category": "Normal Weight", "Blood Pressure": " 120 / 80 "},
            {**base, "Person ID": 3, "Age": 45},
        ])

        groups = make_feature_groups(features)

        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])

    def test_grouping_requires_model_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "Gender"):
            make_feature_groups(pd.DataFrame([{key: 1 for key in RAW_FEATURES if key != "Gender"}]))

    def test_pipeline_can_train_without_external_preprocessing(self) -> None:
        rows = []
        targets = []
        for index, label in enumerate(
            ["Healthy", "Insomnia", "Sleep Apnea"] * 3
        ):
            rows.append(
                {
                    "Gender": "Female" if index % 2 else "Male",
                    "Age": 25 + index * 3,
                    "Occupation": f"Occupation {index % 4}",
                    "Sleep Duration": 5.5 + index * 0.25,
                    "Quality of Sleep": 4 + index % 6,
                    "Physical Activity Level": 20 + index * 5,
                    "Stress Level": 1 + index % 10,
                    "BMI Category": "Normal",
                    "Blood Pressure": f"{110 + index * 3}/{70 + index}",
                    "Heart Rate": 60 + index,
                    "Daily Steps": 4_000 + index * 500,
                }
            )
            targets.append(label)

        pipeline = build_pipeline(LogisticRegression(max_iter=1_000))
        pipeline.fit(pd.DataFrame(rows), pd.Series(targets))

        self.assertEqual(len(pipeline.predict(pd.DataFrame(rows))), len(rows))


if __name__ == "__main__":
    unittest.main()
