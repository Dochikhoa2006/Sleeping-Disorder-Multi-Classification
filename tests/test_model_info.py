import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import joblib
from sklearn.dummy import DummyClassifier

from src.model_info import inspect_model, main
from src.predict import DEFAULT_MODEL_PATH


class ModelInfoTests(unittest.TestCase):
    def test_inspection_reports_saved_provenance_and_classes(self) -> None:
        report = inspect_model()
        self.assertEqual(report["model_name"], "Support Vector Machine")
        self.assertEqual(set(report["labels"]), {"Healthy", "Insomnia", "Sleep Apnea"})
        self.assertEqual(report["model_sha256"], hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest())
        self.assertGreater(report["training_metadata"]["training_rows"], 0)
        self.assertEqual(report["training_metadata"]["selection_metric"], "mean macro F1")
        self.assertIn(report["model_name"], report["cross_validation"])

    def test_cli_prints_machine_readable_json_for_custom_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "classifier.joblib"
            classifier = DummyClassifier().fit([[0], [1]], ["Healthy", "Insomnia"])
            joblib.dump(classifier, model_path)
            original = model_path.read_bytes()
            output = StringIO()
            with patch("sys.argv", ["model_info", "--model", str(model_path)]), redirect_stdout(output):
                main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["model_name"], "DummyClassifier")
            self.assertEqual(report["labels"], ["Healthy", "Insomnia"])
            self.assertIsNone(report["training_metadata"])
            self.assertIsNone(report["cross_validation"])
            self.assertEqual(model_path.read_bytes(), original)

    def test_unfitted_classifier_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "unfitted.joblib"
            joblib.dump(DummyClassifier(), model_path)
            with patch("sys.argv", ["model_info", "--model", str(model_path)]):
                with self.assertRaisesRegex(SystemExit, "fitted classifier with class labels"):
                    main()

    def test_missing_model_has_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "missing.joblib"
            with patch("sys.argv", ["model_info", "--model", str(model_path)]):
                with self.assertRaisesRegex(SystemExit, "Error:.*missing.joblib"):
                    main()


if __name__ == "__main__":
    unittest.main()
