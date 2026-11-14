import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.verify_split import verify_split


class VerifySplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.patient = json.loads(Path("examples/patient.json").read_text())
        self.source = self.root / "source.csv"
        self.write(self.root / "train.csv", [30, 30])
        self.write(self.root / "test.csv", [40])
        self.write(self.source, [30, 30, 40])
        self.manifest = {"source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(), "partitions": {}}
        for name, count in (("train", 2), ("test", 1)):
            self.manifest["partitions"][name] = {
                "sha256": hashlib.sha256((self.root / f"{name}.csv").read_bytes()).hexdigest(),
                "rows": count, "unique_profiles": 1, "class_counts": {"Healthy": count}}
        self.save_manifest()

    def write(self, path, ages):
        with path.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=[*self.patient, "Sleep Disorder"])
            writer.writeheader()
            writer.writerows({**self.patient, "Age": age, "Sleep Disorder": "Healthy"} for age in ages)

    def save_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_valid_split_with_optional_source_and_no_model(self):
        with patch("joblib.load") as load:
            report = verify_split(self.root, self.source)
        load.assert_not_called()
        self.assertTrue(report["verified"])
        self.assertTrue(report["source_checked"])
        self.assertTrue(report["checks"]["source_rows_preserved"])
        self.assertIsNone(verify_split(self.root)["checks"]["source_hash"])

    def test_changed_file_is_reported_and_overlap_detected(self):
        self.write(self.root / "test.csv", [30])
        report = verify_split(self.root)
        self.assertFalse(report["verified"])
        self.assertFalse(report["checks"]["test"]["sha256"])
        self.assertFalse(report["checks"]["profile_disjoint"])
        self.assertEqual(report["shared_unique_profiles"], 1)

    def test_source_duplicate_multiplicity_checked_independently_of_hash(self):
        self.write(self.source, [30, 40, 40])
        self.manifest["source_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.save_manifest()
        report = verify_split(self.root, self.source)
        self.assertTrue(report["checks"]["source_hash"])
        self.assertFalse(report["checks"]["source_rows_preserved"])

    def test_conflicting_profile_labels_fail_even_with_matching_manifest(self):
        path = self.root / "train.csv"
        path.write_bytes(path.read_bytes().replace(b"Healthy", b"Insomnia", 1))
        self.manifest["partitions"]["train"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest["partitions"]["train"]["class_counts"] = {"Healthy": 1, "Insomnia": 1}
        self.save_manifest()
        report = verify_split(self.root)
        self.assertTrue(report["checks"]["train"]["sha256"])
        self.assertTrue(report["checks"]["train"]["class_counts"])
        self.assertFalse(report["checks"]["train"]["consistent_profile_labels"])
        self.assertFalse(report["verified"])

    def test_class_coverage_fails_even_with_matching_manifest(self):
        path = self.root / "test.csv"
        path.write_bytes(path.read_bytes().replace(b"Healthy", b"Insomnia"))
        self.manifest["partitions"]["test"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest["partitions"]["test"]["class_counts"] = {"Insomnia": 1}
        self.save_manifest()
        report = verify_split(self.root)
        self.assertTrue(report["checks"]["profile_disjoint"])
        self.assertTrue(report["checks"]["test"]["consistent_profile_labels"])
        self.assertFalse(report["checks"]["class_coverage"])
        self.assertFalse(report["verified"])

    def test_manifest_count_mismatch_and_malformed_manifest(self):
        self.manifest["partitions"]["train"]["rows"] = 3
        self.save_manifest()
        self.assertFalse(verify_split(self.root)["checks"]["train"]["rows"])
        (self.root / "manifest.json").write_text('{"partitions": {}, "partitions": {}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_split(self.root)

    def test_cli_machine_readable_failure_and_success(self):
        command = [sys.executable, "-m", "src.verify_split", "--directory", str(self.root), "--source", str(self.source)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["verified"])
        self.manifest["partitions"]["test"]["class_counts"] = {"Insomnia": 1}
        self.save_manifest()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(result.stdout)["verified"])
