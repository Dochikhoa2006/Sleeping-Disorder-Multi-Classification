"""Train, compare, and persist sleep-disorder classification pipelines."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import matplotlib
import sklearn

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.svm import SVC

from src.preprocessing import (
    RANDOM_STATE,
    TARGET_COLUMN,
    SleepFeatureEngineer,
    build_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sleep_disorder_pipeline.joblib"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison.png"


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(path)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    target = dataset[TARGET_COLUMN].fillna("Healthy")
    features = dataset.drop(columns=[TARGET_COLUMN])
    return features, target


def candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000,
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": SVC(
            kernel="poly",
            degree=3,
            C=1.0,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=80,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


def make_feature_groups(features: pd.DataFrame) -> pd.Series:
    """Group rows with identical model inputs, ignoring unused raw columns."""

    model_features = SleepFeatureEngineer().transform(features)
    return pd.util.hash_pandas_object(model_features, index=False)


def compare_models(
    features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, dict[str, float]], str]:
    cross_validation = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    groups = make_feature_groups(features)
    results: dict[str, dict[str, float]] = {}

    for name, estimator in candidate_models().items():
        scores = cross_validate(
            build_pipeline(estimator),
            features,
            target,
            groups=groups,
            cv=cross_validation,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
            n_jobs=1,
        )
        accuracy = scores["test_accuracy"]
        results[name] = {
            "f1_macro": float(np.mean(scores["test_f1_macro"])),
            "accuracy": float(np.mean(accuracy)),
            "accuracy_fold_std": float(np.std(accuracy, ddof=1)),
        }

    best_model = max(
        results,
        key=lambda name: (results[name]["f1_macro"], results[name]["accuracy"]),
    )
    return results, best_model


def save_comparison_plot(
    results: dict[str, dict[str, float]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results)
    f1_scores = [results[name]["f1_macro"] for name in names]
    accuracies = [results[name]["accuracy"] for name in names]
    fold_standard_deviations = [
        results[name]["accuracy_fold_std"] for name in names
    ]

    sns.set_theme(style="whitegrid")
    figure, (f1_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(15, 6))

    sns.barplot(x=names, y=f1_scores, ax=f1_axis, color="#4C72B0")
    f1_axis.set_title("Group-Aware Five-Fold Cross-Validation Macro F1")
    f1_axis.set_xlabel("Model")
    f1_axis.set_ylabel("Macro F1")
    f1_axis.set_ylim(0, 1)
    f1_axis.tick_params(axis="x", rotation=12)
    for index, score in enumerate(f1_scores):
        f1_axis.text(index, score + 0.015, f"{score:.3f}", ha="center")

    accuracy_axis.errorbar(
        names,
        accuracies,
        yerr=fold_standard_deviations,
        fmt="o",
        capsize=6,
        color="#4C72B0",
        ecolor="#C44E52",
    )
    accuracy_axis.set_title("Mean Accuracy with Fold-to-Fold Standard Deviation")
    accuracy_axis.set_xlabel("Model")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.tick_params(axis="x", rotation=12)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def train(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, dict[str, float]]:
    features, target = load_training_data(data_path)
    results, best_model_name = compare_models(features, target)

    final_pipeline = build_pipeline(candidate_models()[best_model_name])
    final_pipeline.fit(features, target)
    final_pipeline.model_name_ = best_model_name
    final_pipeline.cv_results_ = results
    final_pipeline.training_metadata_ = {
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "training_rows": len(features),
        "unique_feature_groups": int(make_feature_groups(features).nunique()),
        "validation": "5-fold stratified group cross-validation",
        "selection_metric": "mean macro F1",
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    save_comparison_plot(results, report_path)

    print(f"Selected model: {best_model_name}")
    for name, metrics in results.items():
        print(
            f"{name}: macro F1={metrics['f1_macro']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f} "
            f"± {metrics['accuracy_fold_std']:.3f} fold SD"
        )
    print(f"Saved pipeline to: {model_path}")
    print(f"Saved report to: {report_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare sleep-disorder classifiers."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.data, arguments.model_output, arguments.report_output)
