"""Render saved evaluation metrics as a portable, script-free HTML report."""

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path

from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, _unique_json_object


def render_evaluation(report: dict, *, source_sha256: str = "") -> str:
    if not isinstance(report, dict):
        raise ValueError("Evaluation report must be a JSON object.")
    labels, matrix = report.get("labels"), report.get("confusion_matrix")
    if (not isinstance(labels, list) or not 1 <= len(labels) <= 100
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Evaluation labels must be 1–100 distinct nonempty strings.")
    if (not isinstance(matrix, list) or len(matrix) != len(labels)
            or any(not isinstance(row, list) or len(row) != len(labels)
                   or any(type(value) is not int or value < 0 for value in row) for row in matrix)):
        raise ValueError("Confusion matrix must be square with nonnegative integer counts.")
    total = sum(map(sum, matrix))
    if not total or type(report.get("evaluated_rows")) is not int or report["evaluated_rows"] != total:
        raise ValueError("Evaluated row count must match a nonempty confusion matrix.")
    for name in ("accuracy", "macro_f1"):
        value = report.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number from 0 to 1.")
    expected_accuracy = sum(matrix[i][i] for i in range(len(labels))) / total
    if not math.isclose(report["accuracy"], expected_accuracy, abs_tol=1e-9):
        raise ValueError("Accuracy does not match the confusion matrix.")
    expected_f1 = sum(2 * matrix[i][i] / denominator if (denominator := sum(matrix[i]) + sum(row[i] for row in matrix)) else 0
                      for i in range(len(labels))) / len(labels)
    if not math.isclose(report["macro_f1"], expected_f1, abs_tol=1e-9):
        raise ValueError("Macro F1 does not match the confusion matrix.")
    headers = "".join(f'<th scope="col">{escape(label)}</th>' for label in labels)
    rows = []
    for index, (label, counts) in enumerate(zip(labels, matrix, strict=True)):
        cells = "".join(f'<td class="{"correct" if index == column else "error"}">{count}</td>'
                        for column, count in enumerate(counts))
        rows.append(f'<tr><th scope="row">{escape(label)}</th>{cells}<td>{sum(counts)}</td></tr>')
    provenance = "".join(f'<dt>{escape(key)}</dt><dd>{escape(str(report[key]))}</dd>'
                         for key in ("model_name", "model_sha256", "dataset_sha256", "training_overlap_rows", "total_rows")
                         if key in report)
    if source_sha256:
        provenance += f'<dt>Report file SHA-256</dt><dd>{escape(source_sha256)}</dd>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Classifier evaluation report</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#182830;background:#fff;max-width:1100px;margin:40px auto;padding:0 24px}}
h1,h2{{line-height:1.2}}.metrics{{font-size:1.2rem;padding:16px;background:#edf5f4}}.matrix{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #94a7ab;padding:10px;text-align:right}}
th{{text-align:left}}.correct{{background:#e4f3eb;font-weight:bold}}caption{{text-align:left}}
dd{{margin:0 0 12px;overflow-wrap:anywhere}}dt{{font-weight:bold}}@media print{{body{{margin:0;max-width:none;padding:0}}tr{{break-inside:avoid}}}}
</style></head><body><h1>Classifier evaluation report</h1>
<p class="metrics">Evaluated rows: <strong>{total}</strong> · Accuracy: <strong>{report['accuracy']:.3f}</strong>
 · Macro F1: <strong>{report['macro_f1']:.3f}</strong></p>
<h2>Confusion matrix</h2><p>Rows are actual labels; columns are predicted labels. Counts on the diagonal are correct predictions.</p>
<div class="matrix"><table><caption>Prediction counts by actual and predicted class</caption><thead><tr><th scope="col">Actual / predicted</th>
{headers}<th scope="col">Actual total</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Provenance</h2><dl>{provenance}</dl>
<p>This is a summary of saved aggregate metrics. Optional subgroup, bootstrap, and row diagnostics remain in the original JSON.
Scores depend on the evaluated sample and do not establish clinical validity. This educational classifier is not medical advice.</p>
</body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.resolve() in {args.input_json.resolve(), DEFAULT_MODEL_PATH.resolve()}:
            raise ValueError("Output must differ from the input and shipped model paths.")
        snapshot = args.input_json.read_bytes()
        report = json.loads(snapshot.decode("utf-8-sig"), object_pairs_hook=_unique_json_object)
        html = render_evaluation(report, source_sha256=hashlib.sha256(snapshot).hexdigest())
        with atomic_text_output(args.output) as destination:
            destination.write(html)
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
