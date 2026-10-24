"""Shared preprocessing used by both training and prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Sleep Disorder"
RANDOM_STATE = 150

RAW_FEATURES = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps",
]

NUMERIC_FEATURES = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart Rate",
    "Daily Steps",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Occupation",
    "BMI Category",
    "Blood Pressure Category",
]


def _blood_pressure_category(
    systolic: pd.Series, diastolic: pd.Series
) -> np.ndarray:
    invalid = (
        systolic.isna()
        | diastolic.isna()
        | (systolic <= 0)
        | (diastolic <= 0)
        | (systolic > 300)
        | (diastolic > 200)
    )

    nullable_conditions = [
        invalid,
        (systolic >= 180) | (diastolic >= 120),
        (systolic >= 140) | (diastolic >= 90),
        (systolic >= 130) | (diastolic >= 80),
        (systolic >= 120) & (diastolic < 80),
        (systolic < 120) & (diastolic < 80),
    ]
    conditions = [
        condition.fillna(False).to_numpy(dtype=bool)
        for condition in nullable_conditions
    ]
    labels = [
        "measurement error",
        "severe hypertension",
        "stage 2 hypertension",
        "stage 1 hypertension",
        "elevated",
        "normal",
    ]
    return np.select(conditions, labels, default="measurement error")


class SleepFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert raw patient fields into stable model features."""

    def fit(self, X: Any, y: Any = None) -> "SleepFeatureEngineer":
        self._validate_input(X)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = self._validate_input(X).copy()

        pressure = frame["Blood Pressure"].astype("string").str.extract(
            r"^\s*(?P<systolic>\d+(?:\.\d+)?)\s*/\s*"
            r"(?P<diastolic>\d+(?:\.\d+)?)\s*$"
        )
        systolic = pd.to_numeric(
            pressure["systolic"], errors="coerce"
        ).astype("float64")
        diastolic = pd.to_numeric(
            pressure["diastolic"], errors="coerce"
        ).astype("float64")

        frame["Systolic Blood Pressure"] = systolic
        frame["Diastolic Blood Pressure"] = diastolic
        frame["Blood Pressure Category"] = _blood_pressure_category(
            systolic, diastolic
        )
        frame["BMI Category"] = frame["BMI Category"].replace(
            {"Normal Weight": "Normal"}
        )

        return frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Expose engineered column names to downstream sklearn transformers."""
        return np.asarray(NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)

    @staticmethod
    def _validate_input(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected prediction data as a pandas DataFrame.")

        duplicates = X.columns[X.columns.duplicated()].unique()
        if len(duplicates):
            raise ValueError(
                "Duplicate input column(s): " + ", ".join(map(str, duplicates))
            )

        missing = [column for column in RAW_FEATURES if column not in X.columns]
        if missing:
            raise ValueError(
                "Missing required feature(s): " + ", ".join(sorted(missing))
            )
        return X


def build_pipeline(estimator: Any | None = None) -> Pipeline:
    """Build a complete preprocessing and classification pipeline."""

    if estimator is None:
        estimator = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=2,
            random_state=RANDOM_STATE,
        )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", SleepFeatureEngineer()),
            ("preprocessing", column_transformer),
            ("classifier", estimator),
        ]
    )
