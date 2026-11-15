import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

from src.evaluation import classification_metrics, evaluate_external
from src.predict import DEFAULT_MODEL_PATH


class MetricIntegrityTests(unittest.TestCase):
    def test_unknown_actual_and_predicted_classes_rejected(self):
        for actual, predicted in ((["Healthy"], ["Unknown"]), (["Unknown"], ["Healthy"])):
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, "declared class labels"):
                classification_metrics(pd.Series(actual), predicted, ["Healthy"])

    def test_invalid_vectors_and_label_schemas_rejected(self):
        for actual, predicted in (([], []), (["Healthy"], []), (["Healthy"], [["Healthy"]]),
                                  ([["Healthy"]], ["Healthy"]), (["Healthy"], "Healthy")):
            with self.subTest(actual=actual, predicted=predicted), self.assertRaisesRegex(ValueError, "vectors"):
                classification_metrics(actual, predicted, ["Healthy"])
        for labels in ([], ["Healthy", "Healthy"], [""], [1], "Healthy"):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "Metric labels"):
                classification_metrics(["Healthy"], ["Healthy"], labels)

    def test_absent_declared_classes_keep_zero_f1_and_complete_counts(self):
        report = classification_metrics(pd.Series(["Healthy"]), ["Healthy"], ["Healthy", "Insomnia"])
        self.assertEqual(report["macro_f1"], .5)
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), 1)

    def test_invalid_model_label_preserves_external_evaluation_report(self):
        patient = json.loads(Path("examples/patient.json").read_text())
        model = joblib.load(DEFAULT_MODEL_PATH)
        model.training_feature_hashes_ = set()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "patients.csv"
            output = source.with_name("report.json")
            pd.DataFrame([{**patient, "Sleep Disorder": "Healthy"}]).to_csv(source, index=False)
            output.write_text("existing report")
            with patch("src.evaluation.joblib.load", return_value=model), \
                    patch.object(model, "predict", return_value=["Unknown"]):
                with self.assertRaisesRegex(ValueError, "declared class labels"):
                    evaluate_external(source, DEFAULT_MODEL_PATH, output)
            self.assertEqual(output.read_text(), "existing report")
