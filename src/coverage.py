"""Check patient categories against a trusted saved pipeline's fitted encoder."""

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path

import joblib
import pandas as pd

from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, read_patient_csv
from src.preprocessing import CATEGORICAL_FEATURES


def category_coverage(snapshot: bytes, model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    _, _, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
    model_bytes = model_path.read_bytes()
    model = joblib.load(BytesIO(model_bytes))
    try:
        preprocessing = model.named_steps["preprocessing"]
        encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
        fields = next(columns for name, _, columns in preprocessing.transformers_ if name == "categorical")
        if list(fields) != CATEGORICAL_FEATURES or len(encoder.categories_) != len(fields):
            raise ValueError("Unexpected categorical feature schema.")
        engineered = model.named_steps["feature_engineering"].transform(pd.DataFrame(patients))
    except (AttributeError, KeyError, StopIteration) as error:
        raise ValueError("Coverage requires a fitted project pipeline with a categorical encoder.") from error
    unknown_rows = set()
    results = {}
    for field, categories in zip(fields, encoder.categories_, strict=True):
        known = set(categories)
        counts = {}
        for index, value in enumerate(engineered[field]):
            if value not in known:
                unknown_rows.add(index)
                counts[str(value)] = counts.get(str(value), 0) + 1
        count = sum(counts.values())
        results[field] = {"known_category_count": len(known), "unknown_rows": count,
                          "unknown_row_fraction": count / len(patients),
                          "unknown_categories": [{"value": value, "rows": counts[value]} for value in sorted(counts)]}
    return {"source_sha256": hashlib.sha256(snapshot).hexdigest(),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(), "total_rows": len(patients),
            "rows_with_unknown_categories": len(unknown_rows),
            "fully_covered_rows": len(patients) - len(unknown_rows), "fields": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, help="Atomic JSON report; omit for standard output.")
    parser.add_argument("--fail-on-unknown", action="store_true")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() in {args.input_csv.resolve(), args.model.resolve()}:
            raise ValueError("Output must differ from input and model paths.")
        report = category_coverage(args.input_csv.read_bytes(), args.model)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, AttributeError, EOFError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error
    if args.fail_on_unknown and report["rows_with_unknown_categories"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
