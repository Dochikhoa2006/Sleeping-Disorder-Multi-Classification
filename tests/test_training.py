import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocessing import RAW_FEATURES, build_pipeline
from src.train import load_training_data, make_feature_groups, split_training_holdout, train


class TrainingTests(unittest.TestCase):
    def test_loading_preserves_supported_labels_and_maps_missing_to_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text(
                "Age,Sleep Disorder\n30,Healthy\n31,Insomnia\n32,Sleep Apnea\n33,\n",
                encoding="utf-8",
            )
            features, target = load_training_data(path)

        self.assertEqual(target.tolist(), ["Healthy", "Insomnia", "Sleep Apnea", "Healthy"])
        self.assertEqual(features.columns.tolist(), ["Age"])
        self.assertEqual(features["Age"].tolist(), [30, 31, 32, 33])

    def test_invalid_training_labels_fail_before_model_selection_or_writing(self) -> None:
        for label in ("Insomina", "insomnia", " ", "123"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "training.csv"
                path.write_text(f"Age,Sleep Disorder\n30,Healthy\n31,{label}\n", encoding="utf-8")
                outputs = [root / name for name in ("model.joblib", "plot.png", "report.json")]
                for output in outputs:
                    output.write_bytes(b"existing artifact")

                with patch("src.train.compare_models") as compare:
                    with self.assertRaisesRegex(ValueError, "Unsupported 'Sleep Disorder' label"):
                        train(path, *outputs)
                compare.assert_not_called()
                for output in outputs:
                    self.assertEqual(output.read_bytes(), b"existing artifact")

    def test_loading_rejects_empty_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text("Age,Sleep Disorder\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one row"):
                load_training_data(path)

    def test_loading_requires_target_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text("Age\n30\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a 'Sleep Disorder' column"):
                load_training_data(path)

    def test_holdout_keeps_duplicate_profiles_together(self) -> None:
        base = {
            "Gender": "Female", "Age": 30, "Occupation": "Engineer",
            "Sleep Duration": 7.5, "Quality of Sleep": 8,
            "Physical Activity Level": 45, "Stress Level": 4,
            "BMI Category": "Normal", "Blood Pressure": "120/80",
            "Heart Rate": 68, "Daily Steps": 8_000,
        }
        rows = []
        labels = []
        for index in range(30):
            label = ["Healthy", "Insomnia", "Sleep Apnea"][index % 3]
            for duplicate in range(2):
                rows.append({**base, "Age": 25 + index, "Person ID": len(rows),
                             "BMI Category": "Normal Weight" if duplicate else "Normal"})
                labels.append(label)
        features = pd.DataFrame(rows)
        target = pd.Series(labels)

        train_indices, holdout_indices = split_training_holdout(features, target)
        groups = make_feature_groups(features)

        self.assertFalse(set(groups.iloc[train_indices]) & set(groups.iloc[holdout_indices]))
        self.assertEqual(len(train_indices) + len(holdout_indices), len(features))
        self.assertEqual(set(target.iloc[holdout_indices]), set(target))

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
