"""Match exported prediction batches by record ID without running a model."""

from collections import Counter
from io import StringIO

import pandas as pd

from src.evaluation import normalized_feature_values
from src.predict import read_prediction_csv
from src.preprocessing import RAW_FEATURES

ID_COLUMN = "Record ID"


def compare_prediction_exports(baseline: bytes, candidate: bytes) -> dict:
    batches = []
    for content in (baseline, candidate):
        columns, rows, predictions = read_prediction_csv(StringIO(content.decode("utf-8-sig"), newline=""))
        if ID_COLUMN not in columns:
            raise ValueError("Both exports must contain a 'Record ID' column.")
        ids = [row[ID_COLUMN] for row in rows]
        if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Record IDs must be non-empty and unique within each export.")
        profiles = normalized_feature_values(pd.DataFrame(rows)).to_dict(orient="records")
        batches.append({identifier: (prediction, profile) for identifier, prediction, profile
                        in zip(ids, predictions, profiles, strict=True)})
    before, after = batches
    results = []
    for identifier in [*before, *(value for value in after if value not in before)]:
        first, second = before.get(identifier), after.get(identifier)
        changed_fields = []
        if first is not None and second is not None:
            for field in RAW_FEATURES:
                columns = ("Systolic Blood Pressure", "Diastolic Blood Pressure") if field == "Blood Pressure" else (field,)
                if any(first[1][column] != second[1][column] for column in columns):
                    changed_fields.append(field)
        if first is None:
            status = "added"
        elif second is None:
            status = "removed"
        elif changed_fields:
            status = "inputs_changed"
        elif first[0] != second[0]:
            status = "prediction_changed"
        else:
            status = "unchanged"
        results.append({ID_COLUMN: identifier, "Status": status,
                        "Baseline Prediction": first[0] if first else "",
                        "Candidate Prediction": second[0] if second else "",
                        "Changed Input Fields": "; ".join(changed_fields)})
    counts = Counter(row["Status"] for row in results)
    return {"baseline_rows": len(before), "candidate_rows": len(after),
            "counts": {key: counts[key] for key in
                       ("unchanged", "prediction_changed", "inputs_changed", "added", "removed")},
            "records": results}
