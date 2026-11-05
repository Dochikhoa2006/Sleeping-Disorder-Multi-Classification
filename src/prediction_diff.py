"""Match exported prediction batches by record ID without running a model."""

from collections import Counter
from io import StringIO

import pandas as pd

from src.evaluation import feature_fingerprints
from src.predict import read_prediction_csv

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
        fingerprints = feature_fingerprints(pd.DataFrame(rows))
        batches.append({identifier: (prediction, int(fingerprint)) for identifier, prediction, fingerprint
                        in zip(ids, predictions, fingerprints, strict=True)})
    before, after = batches
    results = []
    for identifier in [*before, *(value for value in after if value not in before)]:
        first, second = before.get(identifier), after.get(identifier)
        if first is None:
            status = "added"
        elif second is None:
            status = "removed"
        elif first[1] != second[1]:
            status = "inputs_changed"
        elif first[0] != second[0]:
            status = "prediction_changed"
        else:
            status = "unchanged"
        results.append({ID_COLUMN: identifier, "Status": status,
                        "Baseline Prediction": first[0] if first else "",
                        "Candidate Prediction": second[0] if second else ""})
    counts = Counter(row["Status"] for row in results)
    return {"baseline_rows": len(before), "candidate_rows": len(after),
            "counts": {key: counts[key] for key in
                       ("unchanged", "prediction_changed", "inputs_changed", "added", "removed")},
            "records": results}
