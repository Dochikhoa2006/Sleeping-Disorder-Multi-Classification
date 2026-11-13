import json
from pathlib import Path
import unittest

import pandas as pd

from src.evaluation import feature_fingerprints
from src.overlap import audit_overlap


class FingerprintNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.patient = json.loads(Path("examples/patient.json").read_text())

    def test_signed_zero_matches_for_every_zero_permitted_input(self):
        for field in ("Sleep Duration", "Physical Activity Level", "Daily Steps"):
            with self.subTest(field=field):
                rows = [{**self.patient, field: value} for value in (0, 0.0, -0.0, "-0.0", "0")]
                self.assertEqual(feature_fingerprints(pd.DataFrame(rows)).nunique(), 1)

    def test_nonzero_values_remain_distinct(self):
        rows = [{**self.patient, "Sleep Duration": value} for value in (0.0, -0.0, .1)]
        fingerprints = feature_fingerprints(pd.DataFrame(rows)).tolist()
        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertNotEqual(fingerprints[0], fingerprints[2])

    def test_existing_positive_zero_hash_is_unchanged(self):
        from src.preprocessing import SleepFeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES
        frame = pd.DataFrame([{**self.patient, "Sleep Duration": 0.0}, self.patient])
        engineered = SleepFeatureEngineer().transform(frame)
        for column in NUMERIC_FEATURES:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce").astype("float64")
        for column in CATEGORICAL_FEATURES:
            engineered[column] = engineered[column].astype("string")
        self.assertEqual(feature_fingerprints(frame).tolist(), pd.util.hash_pandas_object(engineered, index=False).tolist())

    def test_cross_dataset_overlap_detects_signed_zero_alias(self):
        before = pd.DataFrame([{**self.patient, "Sleep Duration": 0.0}]).to_csv(index=False).encode()
        after = pd.DataFrame([{**self.patient, "Sleep Duration": -0.0}]).to_csv(index=False).encode()
        report = audit_overlap(before, after)
        self.assertEqual(report["shared_unique_profiles"], 1)
        self.assertEqual(report["profile_jaccard_similarity"], 1)
