"""Serve a local, browser-based patient prediction form."""

from __future__ import annotations

import argparse
import base64
import calendar
from collections import Counter
import csv
import json
import math
from datetime import date, datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from string import Template
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from src.predict import (
    CSV_NUMERIC_FEATURES, DEFAULT_MODEL_PATH, PREDICTION_COLUMN,
    predict, predict_many, read_patient_csv, read_patient_json, validate_patient,
)
from src.preprocessing import RAW_FEATURES
from src.model_info import inspect_model
from src.evaluation import evaluate_bytes
from src.calendar_export import render_calendar
from src.diary import COLUMNS as DIARY_COLUMNS, add_night, load_diary, night_metrics, summarize, render_report as render_diary_report

TEMPLATE_PATH = Path(__file__).with_name("web_ui.html")
EXAMPLE_CSV_PATH = Path(__file__).resolve().parents[1] / "examples" / "patients.csv"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
PREVIEW_ROWS = 100
DEFAULT_PATIENT = {
    "Gender": "Male", "Age": 31, "Occupation": "Software Engineer",
    "Sleep Duration": 7.2, "Quality of Sleep": 8,
    "Physical Activity Level": 60, "Stress Level": 4,
    "BMI Category": "Normal", "Blood Pressure": "120/80",
    "Heart Rate": 70, "Daily Steps": 8000,
}
LIMITS = {
    "Age": (1, 120), "Sleep Duration": (0, 24), "Quality of Sleep": (1, 10),
    "Physical Activity Level": (0, 1440), "Stress Level": (1, 10),
    "Heart Rate": (20, 250), "Daily Steps": (0, 100000),
}
LABELS = {
    "Sleep Duration": "Sleep duration · hours", "Quality of Sleep": "Sleep quality · 1–10",
    "Physical Activity Level": "Activity · minutes per day", "Stress Level": "Stress · 1–10",
    "Blood Pressure": "Blood pressure · systolic/diastolic", "Heart Rate": "Heart rate · BPM",
    "Daily Steps": "Daily steps",
}


def render_input_guide() -> str:
    """Explain the input format without inspecting the model or patient data."""
    notes = {
        "Gender": "Non-empty text, for example Male or Female.",
        "Occupation": "Non-empty text, for example Software Engineer.",
        "BMI Category": "Non-empty text. Common values: Normal, Normal Weight, Overweight, Obese. Normal Weight is normalized to Normal.",
        "Blood Pressure": "Systolic/diastolic, for example 120/80. Systolic: 50–300; diastolic: 30–200. Systolic must exceed diastolic.",
    }
    rows = []
    for feature in RAW_FEATURES:
        if feature in LIMITS:
            low, high = LIMITS[feature]
            rule = f"Number from {low} to {high}, inclusive."
        else:
            rule = notes[feature]
        rows.append(f'<tr><th scope="row">{escape(feature)}</th>'
                    f'<td>{escape(LABELS.get(feature, feature))}</td><td>{escape(rule)}</td></tr>')
    workspace = Template(TEMPLATE_PATH.with_name("web_input_guide.html").read_text(encoding="utf-8")).substitute(rows="".join(rows))
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="dashboard")


def render_diary_calendar(records: list[dict], month: date) -> str:
    first = month.replace(day=1)
    month_key = first.isoformat()[:7]
    nights = {record["date"]: record for record in records}
    weeks = []
    for week in calendar.Calendar(firstweekday=0).monthdayscalendar(first.year, first.month):
        cells = []
        for day in week:
            if not day:
                cells.append('<td class="diary-outside" aria-hidden="true"></td>')
                continue
            day_key = first.replace(day=day).isoformat()
            record = nights.get(day_key)
            details = '<span class="diary-missing">Not recorded</span>'
            if record is not None:
                metrics = night_metrics(record)
                quality = f'Quality {record["quality"]}/10' if record["quality"] else 'Not rated'
                details = f'<strong>{metrics["estimated_sleep_minutes"] / 60:.2f} h</strong><small>{escape(quality)}</small>'
            cells.append(f'<td class="diary-day{ " diary-recorded" if record else ""}">'
                         f'<time datetime="{day_key}">{day}</time>{details}</td>')
        weeks.append('<tr>' + ''.join(cells) + '</tr>')
    last = first.replace(day=calendar.monthrange(first.year, first.month)[1])
    report = summarize(records, last, last.day)
    average = report["average_estimated_sleep_hours"]
    summary = f'{report["recorded_nights"]} of {last.day} nights recorded. '
    summary += f'{average:.2f} h average estimated sleep per recorded night.' if average is not None else 'No sleep estimates for this month.'
    links = []
    for offset, text in ((-1, '← Previous month'), (1, 'Next month →')):
        index = (first.year - 1) * 12 + first.month - 1 + offset
        if 0 <= index < 9999 * 12:
            year, zero_month = divmod(index, 12)
            key = date(year + 1, zero_month + 1, 1).isoformat()[:7]
            links.append(f'<a href="/diary?month={key}">{text}</a>')
    return Template(TEMPLATE_PATH.with_name("web_diary_calendar.html").read_text(encoding="utf-8")).substitute(
        month=month_key, title=escape(first.strftime("%B %Y")), summary=escape(summary),
        links=''.join(links), weeks=''.join(weeks))


def render_diary(records: list[dict] | None = None, *, values: dict | None = None,
                 error: str | None = None, saved: bool = False, month: date | None = None) -> str:
    if records is None:
        workspace = ('<section class="model-panel"><h2>Enable your local diary</h2>'
                     '<p>Start the app with <code>python -m src.web --diary data/sleep_diary.csv</code> to enable diary storage.</p>'
                     '<p>The diary is optional. It saves nightly records to the local CSV you choose, separate from patient predictions.</p></section>')
    else:
        values = values or {"date": (date.today() - timedelta(days=1)).isoformat(),
                            "bedtime": "23:00", "wake_time": "07:00", "awake_minutes": "0", "quality": ""}
        rows = []
        for record in sorted(records, key=lambda row: row["date"], reverse=True)[:90]:
            metrics = night_metrics(record)
            cells = [record["date"], record["bedtime"], record["wake_time"],
                     f'{metrics["estimated_sleep_minutes"] / 60:.2f} h',
                     record["quality"] or "Not rated"]
            rows.append('<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in cells) + '</tr>')
        report = summarize(records, date.today(), 7)
        average = report["average_estimated_sleep_hours"]
        summary = f'{report["recorded_nights"]} recorded nights in the last 7 days; '
        summary += f'{average:.2f} h average estimated sleep.' if average is not None else 'no sleep estimates yet.'
        feedback = (f'<p class="feedback error" role="alert">{escape(error)}</p>' if error else
                    '<p class="feedback result" role="status">Night saved to your local diary.</p>' if saved else '')
        workspace = Template(TEMPLATE_PATH.with_name("web_diary.html").read_text(encoding="utf-8")).substitute(
            **{key: escape(str(values.get(key, "")), quote=True) for key in DIARY_COLUMNS},
            checked=' checked' if values.get("replace") == "1" else '', feedback=feedback,
            summary=escape(summary), total=len(records), calendar=render_diary_calendar(records, month or date.today()),
            rows=''.join(rows) or '<tr><td colspan="5">No nights recorded yet.</td></tr>')
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="dashboard")


def render_timer() -> str:
    workspace = TEMPLATE_PATH.with_name("web_timer.html").read_text(encoding="utf-8")
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="dashboard")


def render_fields(values: dict, prefix: str = "") -> str:
    fields = []
    for index, feature in enumerate(RAW_FEATURES):
        value = escape(str(values.get(prefix + feature, "")), quote=True)
        label = escape(LABELS.get(feature, feature))
        field_id = f"{prefix}field-{index}"
        if feature in LIMITS:
            minimum, maximum = LIMITS[feature]
            attributes = f'type="number" min="{minimum}" max="{maximum}" step="any"'
        else:
            attributes = 'type="text"'
        fields.append(
            f'<label for="{field_id}">{label}<input id="{field_id}" '
            f'name="{escape(prefix + feature, quote=True)}" {attributes} value="{value}" required></label>'
        )
    return "\n".join(fields)


def render_page(values: dict | None = None, *, result: str | None = None, error: str | None = None, notice: str | None = None) -> str:
    values = DEFAULT_PATIENT if values is None else values
    if error:
        feedback = f'<section class="feedback error" role="alert"><h2>Check your details</h2><p>{escape(error)}</p></section>'
    elif result:
        feedback = f'<section class="feedback result" role="status"><p class="eyebrow">MODEL PREDICTION</p><h2>{escape(result)}</h2><p>This is an educational model output, not a diagnosis.</p></section>'
    elif notice:
        feedback = f'<section class="feedback" role="status"><h2>Profile loaded</h2><p>{escape(notice)}</p></section>'
    else:
        feedback = '<section class="feedback"><h2>Start with an example</h2><p>The form contains sample values. Replace them with your own details, then select Predict.</p></section>'
    workspace = Template(TEMPLATE_PATH.with_name("web_patient_form.html").read_text(encoding="utf-8")).substitute(
        fields=render_fields(values), feedback=feedback,
    )
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="layout")


def scenario_patients(values: dict) -> tuple[str, list[dict]]:
    """Build a bounded, evenly spaced sweep of one numeric input."""
    feature = values.get("scenario_field", "Sleep Duration")
    if feature not in LIMITS:
        raise ValueError("Choose a numeric scenario field.")
    try:
        start = float(values.get("scenario_start", "5"))
        stop = float(values.get("scenario_stop", "9"))
        count = int(values.get("scenario_count", "9"))
    except (ValueError, TypeError) as error:
        raise ValueError("Enter numeric range endpoints and an integer point count.") from error
    low, high = LIMITS[feature]
    if not (math.isfinite(start) and math.isfinite(stop) and low <= start < stop <= high):
        raise ValueError(f"Choose an increasing range for {feature} within {low}–{high}.")
    if not 2 <= count <= 25:
        raise ValueError("Choose between 2 and 25 scenario points.")
    patient = parse_form_patient(values)
    validate_patient(patient)
    patients = [{**patient, feature: start + (stop - start) * index / (count - 1)}
                for index in range(count)]
    patients[-1][feature] = stop
    return feature, patients


def render_scenarios(values: dict | None = None, *, result=None, error: str | None = None) -> str:
    values = {**DEFAULT_PATIENT, "scenario_field": "Sleep Duration", "scenario_start": "5",
              "scenario_stop": "9", "scenario_count": "9"} if values is None else values
    options = "".join(f'<option value="{escape(feature)}"'
                      + (' selected' if values.get("scenario_field") == feature else '')
                      + f'>{escape(LABELS.get(feature, feature))}</option>' for feature in LIMITS)
    feedback = ""
    if error:
        feedback = f'<section class="feedback error" role="alert"><h2>Check your scenario</h2><p>{escape(error)}</p></section>'
    elif result is not None:
        feature, patients, predictions = result
        rows = "".join(f'<tr><th scope="row">{escape(str(patient[feature]))}</th><td>{escape(prediction)}</td></tr>'
                       for patient, prediction in zip(patients, predictions, strict=True))
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=[*RAW_FEATURES, PREDICTION_COLUMN])
        writer.writeheader()
        for patient, prediction in zip(patients, predictions, strict=True):
            writer.writerow({**patient, PREDICTION_COLUMN: prediction})
        download = base64.b64encode(output.getvalue().encode("utf-8")).decode("ascii")
        feedback = ('<section class="model-panel" role="status"><h2>Scenario results</h2>'
                    f'<p>{len(patients)} profiles evaluated. Only {escape(feature)} varies.</p>'
                    f'<a class="download-link" href="data:text/csv;charset=utf-8;base64,{download}" download="scenario-predictions.csv">Download all scenario profiles ↓</a>'
                    '<div class="table-scroll"><table><caption>Predictions across the chosen range</caption>'
                    f'<thead><tr><th scope="col">{escape(LABELS.get(feature, feature))}</th><th scope="col">Predicted class</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table></div></section>')
    workspace = Template(TEMPLATE_PATH.with_name("web_scenarios.html").read_text(encoding="utf-8")).substitute(
        fields=render_fields(values), options=options, feedback=feedback,
        start=escape(str(values.get("scenario_start", "")), quote=True),
        stop=escape(str(values.get("scenario_stop", "")), quote=True),
        count=escape(str(values.get("scenario_count", "")), quote=True))
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="dashboard")


def render_comparison(values: dict | None = None, *, result: list[str] | None = None, error: str | None = None) -> str:
    if values is None:
        values = {prefix + key: value for prefix in ("a.", "b.") for key, value in DEFAULT_PATIENT.items()}
    feedback = ""
    if error:
        message = str(error).replace("Patient 1:", "Profile A:").replace("Patient 2:", "Profile B:")
        feedback = f'<section class="feedback error" role="alert"><h2>Check your profiles</h2><p>{escape(message)}</p></section>'
    elif result:
        cards = "".join(
            f'<section class="feedback result"><p class="eyebrow">PROFILE {label}</p><h2>{escape(prediction)}</h2></section>'
            for label, prediction in zip(("A", "B"), result, strict=True)
        )
        changes = []
        for feature in RAW_FEATURES:
            first, second = str(values.get("a." + feature, "")), str(values.get("b." + feature, ""))
            if first != second:
                changes.append(f'<tr><th scope="row">{escape(LABELS.get(feature, feature))}</th><td>{escape(first)}</td><td>{escape(second)}</td></tr>')
        summary = "Both profiles received the same predicted class." if result[0] == result[1] else "The profiles received different predicted classes."
        differences = (
            '<div class="table-scroll"><table><caption>Edited fields</caption><thead><tr><th scope="col">Field</th><th scope="col">Profile A</th><th scope="col">Profile B</th></tr></thead><tbody>'
            + "".join(changes) + "</tbody></table></div>"
        ) if changes else "<p>The profiles have identical entries.</p>"
        feedback = f'<section role="status"><div class="profile-grid">{cards}</div><p>{summary}</p>{differences}</section>'
    workspace = Template(TEMPLATE_PATH.with_name("web_compare.html").read_text(encoding="utf-8")).substitute(
        first=render_fields(values, "a."), second=render_fields(values, "b."), feedback=feedback,
    )
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="comparison")


def parse_form_patient(values: dict, prefix: str = "") -> dict:
    patient = {feature: values.get(prefix + feature, "") for feature in RAW_FEATURES}
    for feature in CSV_NUMERIC_FEATURES:
        try:
            patient[feature] = float(patient[feature])
        except ValueError as error:
            label = {"a.": "Profile A: ", "b.": "Profile B: "}.get(prefix, "")
            raise ValueError(f"{label}'{feature}' must be a number.") from error
    return patient


def render_model(report: dict | None = None, *, error: str | None = None) -> str:
    if error:
        workspace = f'<section class="feedback error" role="alert"><h2>Model details unavailable</h2><p>{escape(error)}</p></section>'
    else:
        metadata = report.get("training_metadata") or {}
        results = report.get("cross_validation") or {}

        def display(value) -> str:
            return escape(str(value)) if value is not None else "Not recorded"

        def score(value, *, meter_label=None) -> str:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return "Not recorded"
            text = f"{value:.3f}"
            if meter_label is not None and 0 <= value <= 1:
                text += f'<meter min="0" max="1" value="{value}" aria-label="{escape(meter_label, quote=True)}">{text}</meter>'
            return text

        rows = []
        for name, metrics in results.items():
            selected = '<span class="model-badge">Selected</span>' if name == report.get("model_name") else ""
            rows.append(
                f'<tr><th scope="row">{display(name)} {selected}</th>'
                f'<td>{score(metrics.get("f1_macro"), meter_label=str(name) + " macro F1")}</td>'
                f'<td>{score(metrics.get("accuracy"), meter_label=str(name) + " accuracy")}</td>'
                f'<td>{score(metrics.get("accuracy_fold_std"))}</td></tr>'
            )
        scores = (
            '<div class="table-scroll"><table><caption>Stored cross-validation results</caption>'
            '<thead><tr><th scope="col">Model</th><th scope="col">Macro F1</th><th scope="col">Accuracy</th><th scope="col">Accuracy fold SD</th></tr></thead>'
            '<tbody>' + "".join(rows) + '</tbody></table></div>'
        ) if rows else '<p class="hint">No cross-validation results were stored with this model.</p>'
        details = "".join(
            f'<dt>{label}</dt><dd>{display(metadata.get(key))}</dd>' for key, label in (
                ("selection_metric", "Selection metric"), ("validation", "Validation method"),
                ("scikit_learn_version", "scikit-learn version"), ("pandas_version", "pandas version"),
                ("numpy_version", "NumPy version"), ("joblib_version", "Joblib version"),
            )
        )
        workspace = Template(TEMPLATE_PATH.with_name("web_model.html").read_text(encoding="utf-8")).substitute(
            model=display(report.get("model_name")),
            classes="".join(f'<span class="model-badge">{display(label)}</span>' for label in report.get("labels", [])),
            training_rows=display(metadata.get("training_rows")),
            holdout_rows=display(metadata.get("holdout_rows")),
            groups=display(metadata.get("unique_feature_groups")),
            scores=scores, details=details, fingerprint=display(report.get("model_sha256")),
        )
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="dashboard")


def render_patient_report(values: dict, result: str) -> str:
    rows = "".join(
        f'<tr><th scope="row">{escape(LABELS.get(feature, feature))}</th>'
        f'<td>{escape(str(values.get(feature, "")))}</td></tr>'
        for feature in RAW_FEATURES
    )
    return Template(TEMPLATE_PATH.with_name("web_report.html").read_text(encoding="utf-8")).substitute(
        result=escape(result), rows=rows,
        generated=escape(datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def uploaded_file(content_type: str, body: bytes, field_name: str) -> bytes:
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if not message.is_multipart() or any(part.defects for part in message.walk()):
        raise ValueError("The upload is incomplete or malformed. Choose the file again.")
    parts = list(message.iter_parts())
    if len(parts) != 1 or parts[0].get_param("name", header="content-disposition") != field_name:
        raise ValueError("Upload exactly one file using the matching upload form.")
    upload = parts[0]
    if not upload.get_filename():
        raise ValueError("Choose a file to upload.")
    payload = upload.get_payload(decode=True)
    if payload is None:
        raise ValueError("The uploaded file is empty or malformed.")
    return payload


def render_batch_results(columns: list[str], rows: list[dict], predictions: list[str], csv_content: str, *, query: str = "", selected_class: str = "", sort_by: str = "", direction: str = "asc") -> str:
    counts = Counter(predictions)
    labels = dict.fromkeys(["Healthy", "Insomnia", "Sleep Apnea", *predictions])
    matches = [index for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True))
               if (not selected_class or prediction == selected_class)
               and (not query or query.casefold() in " ".join([*(str(value) for value in row.values()), prediction]).casefold())]
    if sort_by:
        def sort_key(index):
            value = predictions[index] if sort_by == PREDICTION_COLUMN else rows[index][sort_by]
            return float(value) if sort_by in CSV_NUMERIC_FEATURES else str(value).casefold()
        matches.sort(key=sort_key, reverse=direction == "desc")
    cards = "".join(
        f'<article><p>{escape(label)}</p><strong>{counts[label]}</strong></article>' for label in labels
    )
    headers = '<th scope="col">CSV row</th>' + "".join(
        f'<th scope="col">{escape(column)}</th>' for column in [*columns, PREDICTION_COLUMN]
    )
    table_rows = []
    for index in matches[:PREVIEW_ROWS]:
        row, prediction = rows[index], predictions[index]
        cells = "".join(f'<td>{escape(str(row[column]))}</td>' for column in columns)
        table_rows.append(f'<tr><th scope="row">{index + 2}</th>{cells}<td>{escape(prediction)}</td></tr>')
    if not matches:
        table_rows.append(f'<tr><td colspan="{len(columns) + 2}">No records match these filters. Clear the filters to see the batch again.</td></tr>')
    options = '<option value="">All classes</option>' + "".join(
        f'<option value="{escape(label, quote=True)}"{" selected" if label == selected_class else ""}>{escape(label)}</option>'
        for label in labels
    )
    filtered = ""
    if query or selected_class or sort_by:
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=[*columns, PREDICTION_COLUMN])
        writer.writeheader()
        for index in matches:
            writer.writerow({**rows[index], PREDICTION_COLUMN: predictions[index]})
        encoded = base64.b64encode(output.getvalue().encode("utf-8")).decode("ascii")
        filtered = f'<p><a href="data:text/csv;charset=utf-8;base64,{encoded}" download="filtered-predictions.csv">Download {len(matches)} matching predictions ↓</a></p>'
    workspace = Template(TEMPLATE_PATH.with_name("web_batch_results.html").read_text(encoding="utf-8")).substitute(
        total=len(rows), matched=len(matches), shown=min(len(matches), PREVIEW_ROWS), cards=cards,
        headers=headers, rows="".join(table_rows),
        download=base64.b64encode(csv_content.encode("utf-8")).decode("ascii"),
        query=escape(query, quote=True), options=options, filtered=filtered,
        sorting='<option value="">Input order</option>' + "".join(
            f'<option value="{escape(column, quote=True)}"{" selected" if column == sort_by else ""}>{escape(column)}</option>'
            for column in [*columns, PREDICTION_COLUMN]
        ),
        ascending=" selected" if direction == "asc" else "",
        descending=" selected" if direction == "desc" else "",
        ordering=escape(f"{sort_by} ({'descending' if direction == 'desc' else 'ascending'})") if sort_by else "input order",
    )
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="dashboard")


def read_preview_results(encoded: str) -> tuple[list[str], list[dict], list[str], str]:
    content = base64.b64decode(encoded, validate=True).decode("utf-8")
    reader = csv.reader(StringIO(content, newline=""), strict=True)
    headers = next(reader, [])
    if len(headers) != len(set(headers)) or PREDICTION_COLUMN not in headers:
        raise ValueError("The results are malformed. Upload the original CSV again.")
    prediction_index = headers.index(PREDICTION_COLUMN)
    columns = [name for name in headers if name != PREDICTION_COLUMN]
    source = StringIO(newline="")
    writer = csv.writer(source)
    writer.writerow(columns)
    predictions = []
    for values in reader:
        if len(values) != len(headers) or not values[prediction_index]:
            raise ValueError("The results contain an incomplete row.")
        predictions.append(values[prediction_index])
        writer.writerow(values[:prediction_index] + values[prediction_index + 1:])
    source.seek(0)
    columns, rows, _ = read_patient_csv(source)
    return columns, rows, predictions, content


def render_evaluation(report: dict | None = None, *, error: str | None = None, diagnostics: list[dict] | None = None) -> str:
    feedback = f'<section class="feedback error" role="alert"><h2>Check your evaluation file</h2><p>{escape(error)}</p></section>' if error else ""
    results = ""
    if report is not None:
        labels = report["labels"]
        headers = '<th scope="col">Actual ↓ / Predicted →</th>' + "".join(
            f'<th scope="col">{escape(label)}</th>' for label in labels
        )
        matrix = []
        scores = []
        for index, label in enumerate(labels):
            cells = []
            for column, count in enumerate(report["confusion_matrix"][index]):
                style = "matrix-hit" if index == column else "matrix-miss" if count else ""
                cells.append(f'<td class="{style}">{count}</td>')
            matrix.append(f'<tr><th scope="row">{escape(label)}</th>{"".join(cells)}</tr>')
            metrics = report["per_class"][label]
            scores.append(
                f'<tr><th scope="row">{escape(label)}</th>'
                f'<td>{metrics["precision"]:.3f}</td><td>{metrics["recall"]:.3f}</td>'
                f'<td>{metrics["f1-score"]:.3f}</td><td>{int(metrics["support"])}</td>'
                f'<td>{report["class_row_counts"][label]["training_overlap_rows"]}</td></tr>'
            )
        review = ""
        if diagnostics is not None:
            outcomes = {"correct": "Correct", "incorrect": "Incorrect", "training_overlap": "Excluded — training overlap"}
            incorrect = [row for row in diagnostics if row["status"] == "incorrect"]

            def diagnostics_download(records) -> str:
                stream = StringIO(newline="")
                writer = csv.writer(stream)
                writer.writerow(["Data record", "Actual Sleep Disorder", PREDICTION_COLUMN, "Outcome"])
                for row in records:
                    writer.writerow([row["record_number"], row["actual"], row["predicted"], row["status"]])
                return base64.b64encode(stream.getvalue().encode("utf-8")).decode("ascii")

            review_rows = "".join(
                f'<tr><th scope="row">{row["record_number"]}</th><td>{escape(row["actual"])}</td>'
                f'<td>{escape(row["predicted"]) if row["predicted"] is not None else "Not predicted"}</td>'
                f'<td>{outcomes[row["status"]]}</td></tr>'
                for row in diagnostics[:PREVIEW_ROWS]
            )
            review = Template(TEMPLATE_PATH.with_name("web_evaluation_rows.html").read_text(encoding="utf-8")).substitute(
                correct=sum(row["status"] == "correct" for row in diagnostics),
                incorrect=len(incorrect), excluded=report["training_overlap_rows"],
                shown=min(len(diagnostics), PREVIEW_ROWS), total=len(diagnostics), rows=review_rows,
                all_download=diagnostics_download(diagnostics), errors_download=diagnostics_download(incorrect),
            )
        results = Template(TEMPLATE_PATH.with_name("web_evaluation_results.html").read_text(encoding="utf-8")).substitute(
            accuracy=f'{report["accuracy"]:.1%}', f1=f'{report["macro_f1"]:.3f}',
            evaluated=report["evaluated_rows"], total=report["total_rows"], overlap=report["training_overlap_rows"],
            unique=report["unique_evaluated_profiles"], duplicates=report["duplicate_evaluated_rows"],
            headers=headers, matrix="".join(matrix), scores="".join(scores),
            review=review,
            download=base64.b64encode((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")).decode("ascii"),
        )
    workspace = Template(TEMPLATE_PATH.with_name("web_evaluate.html").read_text(encoding="utf-8")).substitute(
        feedback=feedback, results=results,
    )
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="dashboard")


def make_server(model_path: Path = DEFAULT_MODEL_PATH, port: int = 8765, *, bind: str = "127.0.0.1", diary_path: Path | None = None) -> ThreadingHTTPServer:
    if diary_path is not None and diary_path.resolve() == model_path.resolve():
        raise ValueError("Diary path must differ from model path.")
    diary_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:
            # Patient input is neither persisted nor written to access logs.
            pass

        def respond(self, status: int, body: str, *, content_type: str = "text/html; charset=utf-8", filename: str | None = None, allow: str | None = None, timer_script: bool = False) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if allow:
                self.send_header("Allow", allow)
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            policy = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
            if timer_script:
                policy += "; script-src 'self'"
            self.send_header("Content-Security-Policy", policy)
            self.end_headers()
            self.wfile.write(encoded)

        def local_request(self) -> bool:
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (
                origin is None or origin in {f"http://{host}" for host in hosts}
            )

        def respond_json(self, status: int, payload: dict, *, allow: str | None = None) -> None:
            self.respond(status, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         content_type="application/json; charset=utf-8", allow=allow)

        def do_GET(self) -> None:
            if not self.local_request():
                if self.path.startswith("/api/"):
                    self.respond_json(403, {"error": "Local requests only."})
                else:
                    self.respond(403, "Local requests only.")
            elif self.path == "/api/health":
                self.respond_json(200, {"status": "ok"})
            elif self.path.startswith("/api/"):
                self.respond_json(405 if self.path == "/api/predict" else 404,
                                  {"error": "Use POST for predictions." if self.path == "/api/predict" else "API route not found."},
                                  allow="POST" if self.path == "/api/predict" else None)
            elif self.path == "/timer":
                self.respond(200, render_timer(), timer_script=True)
            elif self.path == "/timer.js":
                self.respond(200, TEMPLATE_PATH.with_name("web_timer.js").read_text(encoding="utf-8"),
                             content_type="text/javascript; charset=utf-8")
            elif self.path == "/":
                self.respond(200, render_page())
            elif self.path in ("/diary", "/diary.csv", "/diary.ics", "/diary-report") or self.path.startswith("/diary?"):
                self.diary_get()
            elif self.path == "/explore":
                self.respond(200, render_scenarios())
            elif self.path == "/guide":
                self.respond(200, render_input_guide())
            elif self.path == "/template.csv":
                output = StringIO(newline="")
                csv.writer(output).writerow(RAW_FEATURES)
                self.respond(200, output.getvalue(), content_type="text/csv; charset=utf-8",
                             filename="patient-template.csv")
            elif self.path == "/compare":
                self.respond(200, render_comparison())
            elif self.path == "/evaluate":
                self.respond(200, render_evaluation())
            elif self.path in ("/model", "/model.json"):
                try:
                    report = inspect_model(model_path)
                    if self.path == "/model.json":
                        self.respond(200, json.dumps(report, indent=2, allow_nan=False) + "\n",
                                     content_type="application/json; charset=utf-8", filename="model-info.json")
                    else:
                        self.respond(200, render_model(report))
                except (OSError, ValueError, TypeError, AttributeError):
                    self.respond(503, render_model(error="Check the server's model file and try again."))
            elif self.path == "/example.csv":
                self.respond(200, EXAMPLE_CSV_PATH.read_text(encoding="utf-8"),
                             content_type="text/csv; charset=utf-8", filename="example-patients.csv")
            else:
                self.respond(404, "Page not found.")

        def do_POST(self) -> None:
            if not self.local_request():
                if self.path.startswith("/api/"):
                    self.respond_json(403, {"error": "Local requests only."})
                else:
                    self.respond(403, "Local requests only.")
                return
            if self.path.startswith("/api/"):
                if self.path == "/api/predict":
                    self.api_predict()
                else:
                    self.respond_json(405 if self.path == "/api/health" else 404,
                                      {"error": "Use GET for the health check." if self.path == "/api/health" else "API route not found."},
                                      allow="GET" if self.path == "/api/health" else None)
                return
            if self.path == "/diary":
                self.diary_add()
                return
            if self.path == "/evaluate":
                self.evaluate_upload()
                return
            if self.path == "/batch-filter":
                self.filter_batch()
                return
            if self.path in ("/batch", "/batch-preview"):
                self.batch_predict()
                return
            if self.path == "/import-profile":
                self.import_profile()
                return
            if self.path not in ("/predict", "/compare", "/report", "/profile.json", "/explore"):
                self.respond(404, "Page not found.")
                return
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self.respond(415, "Submit the patient form as URL-encoded data.")
                return
            values = {}
            renderer = (render_scenarios if self.path == "/explore" else
                        render_comparison if self.path == "/compare" else render_page)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    self.respond(413, "Form data must contain between 1 and 16384 bytes.")
                    return
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True, max_num_fields=32)
                if any(len(items) != 1 for items in fields.values()):
                    raise ValueError("Each field must be submitted exactly once.")
                values = {key: items[0] for key, items in fields.items()}
                if self.path == "/profile.json":
                    patient = parse_form_patient(values)
                    validate_patient(patient)
                    self.respond(200, json.dumps(patient, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                 content_type="application/json; charset=utf-8", filename="patient-profile.json")
                    return
                if self.path == "/explore":
                    feature, patients = scenario_patients(values)
                    result = (feature, patients, predict_many(patients, model_path))
                elif self.path == "/compare":
                    patients = [parse_form_patient(values, prefix) for prefix in ("a.", "b.")]
                    result = predict_many(patients, model_path)
                else:
                    result = predict(parse_form_patient(values), model_path)
            except (ValueError, TypeError, UnicodeError) as error:
                self.respond(400, renderer(values, error=str(error)))
                return
            except OSError:
                self.respond(503, renderer(values, error="The model could not be loaded. Check the server's model path."))
                return
            if self.path == "/report":
                self.respond(200, render_patient_report(values, result))
            else:
                self.respond(200, renderer(values, result=result))

        def diary_get(self) -> None:
            if diary_path is None:
                self.respond(200 if self.path == "/diary" else 404, render_diary())
                return
            try:
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True, max_num_fields=1)
                if set(query) - {"month"}:
                    raise ValueError("Choose a month using YYYY-MM.")
                key = query.get("month", [date.today().isoformat()[:7]])[0]
                if len(key) != 7:
                    raise ValueError("Choose a month using YYYY-MM.")
                month = date.fromisoformat(key + "-01")
            except ValueError:
                self.respond(400, "Choose a valid month using YYYY-MM.")
                return
            try:
                with diary_lock:
                    records = load_diary(diary_path)
                if self.path == "/diary.csv":
                    output = StringIO(newline="")
                    writer = csv.DictWriter(output, fieldnames=DIARY_COLUMNS)
                    writer.writeheader()
                    writer.writerows(sorted(records, key=lambda row: row["date"]))
                    self.respond(200, output.getvalue(), content_type="text/csv; charset=utf-8", filename="sleep-diary.csv")
                elif self.path == "/diary.ics":
                    self.respond(200, render_calendar(records, calendar_id=str(diary_path.resolve())),
                                 content_type="text/calendar; charset=utf-8", filename="sleep-diary.ics")
                elif self.path == "/diary-report":
                    self.respond(200, render_diary_report(summarize(records, date.today(), 30)))
                else:
                    self.respond(200, render_diary(records, month=month))
            except (OSError, ValueError, TypeError, csv.Error):
                self.respond(503, "The diary could not be read. Check its format and file permissions.")

        def diary_add(self) -> None:
            if diary_path is None:
                self.respond(403, render_diary())
                return
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self.respond(415, "Submit the diary form as URL-encoded data.")
                return
            values = {}
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2048:
                    self.respond(413, "Diary form must contain between 1 and 2048 bytes.")
                    return
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True, max_num_fields=6)
                if any(len(items) != 1 for items in fields.values()):
                    raise ValueError("Submit each diary field once.")
                values = {key: items[0] for key, items in fields.items()}
                record = {key: values.get(key, "") for key in DIARY_COLUMNS}
                with diary_lock:
                    add_night(diary_path, record, replace=values.get("replace") == "1")
                    records = load_diary(diary_path)
            except (ValueError, TypeError, UnicodeError, csv.Error) as error:
                try:
                    with diary_lock:
                        records = load_diary(diary_path)
                except (OSError, ValueError, TypeError, csv.Error):
                    self.respond(503, "The diary could not be read. Check its format and file permissions.")
                    return
                self.respond(400, render_diary(records, values=values, error=str(error)))
                return
            except OSError:
                self.respond(503, "The diary could not be saved. Check its directory and file permissions.")
                return
            self.respond(200, render_diary(records, saved=True, month=date.fromisoformat(record["date"])))

        def batch_predict(self) -> None:
            if self.headers.get_content_type() != "multipart/form-data":
                self.respond(415, render_page(error="Choose a CSV using the batch upload form."))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD_BYTES:
                    self.respond(413, render_page(error="Choose a CSV smaller than 2 MiB, including upload overhead."))
                    return
                payload = uploaded_file(self.headers.get("Content-Type", ""), self.rfile.read(length), "patients")
                columns, rows, patients = read_patient_csv(StringIO(payload.decode("utf-8-sig"), newline=""))
                predictions = predict_many(patients, model_path)
                output = StringIO(newline="")
                writer = csv.DictWriter(output, fieldnames=[*columns, PREDICTION_COLUMN])
                writer.writeheader()
                for row, prediction in zip(rows, predictions, strict=True):
                    writer.writerow({**row, PREDICTION_COLUMN: prediction})
            except (ValueError, TypeError, UnicodeError, csv.Error) as error:
                self.respond(400, render_page(error=str(error)))
                return
            except OSError:
                self.respond(503, render_page(error="The model could not be loaded. Check the server's model path."))
                return
            if self.path == "/batch-preview":
                self.respond(200, render_batch_results(columns, rows, predictions, output.getvalue()))
            else:
                self.respond(200, output.getvalue(), content_type="text/csv; charset=utf-8", filename="sleep-predictions.csv")

        def api_predict(self) -> None:
            if self.headers.get_content_type() != "application/json":
                self.respond_json(415, {"error": "Content-Type must be application/json."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD_BYTES:
                    self.respond_json(413, {"error": "JSON input must contain between 1 byte and 2 MiB."})
                    return
                records = read_patient_json(StringIO(self.rfile.read(length).decode("utf-8-sig")))
                patients = records if isinstance(records, list) else [records]
                predictions = predict_many(patients, model_path)
            except (ValueError, TypeError, UnicodeError) as error:
                self.respond_json(400, {"error": str(error)})
                return
            except OSError:
                self.respond_json(503, {"error": "The model could not be loaded. Check the server's model path."})
                return
            self.respond_json(200, {"predictions": predictions, "count": len(predictions)})

        def evaluate_upload(self) -> None:
            if self.headers.get_content_type() != "multipart/form-data":
                self.respond(415, render_evaluation(error="Choose a labeled CSV using the evaluation form."))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD_BYTES:
                    self.respond(413, render_evaluation(error="Choose a CSV smaller than 2 MiB, including upload overhead."))
                    return
                payload = uploaded_file(self.headers.get("Content-Type", ""), self.rfile.read(length), "evaluation")
                report = evaluate_bytes(payload, model_path, include_rows=True)
                diagnostics = report.pop("row_results")
            except (ValueError, TypeError, UnicodeError, csv.Error) as error:
                self.respond(400, render_evaluation(error=str(error)))
                return
            except OSError:
                self.respond(503, render_evaluation(error="The model could not be loaded. Check the server's model path."))
                return
            self.respond(200, render_evaluation(report, diagnostics=diagnostics))

        def filter_batch(self) -> None:
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self.respond(415, render_page(error="Use the filters on the batch preview page."))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    self.respond(413, render_page(error="The results form is too large. Upload a smaller batch."))
                    return
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True, max_num_fields=6)
                if any(len(items) != 1 for items in fields.values()):
                    raise ValueError("Each filter field must be submitted exactly once.")
                columns, rows, predictions, content = read_preview_results(fields.get("results", [""])[0])
                query = fields.get("query", [""])[0].strip()
                selected_class = fields.get("class", [""])[0]
                sort_by = fields.get("sort", [""])[0]
                direction = fields.get("direction", ["asc"])[0]
                if "reset" in fields:
                    query = selected_class = sort_by = ""
                    direction = "asc"
                if len(query) > 256:
                    raise ValueError("Search text must contain at most 256 characters.")
                if sort_by not in ["", *columns, PREDICTION_COLUMN] or direction not in ("asc", "desc"):
                    raise ValueError("Choose a listed sort column and ascending or descending order.")
            except (ValueError, TypeError, UnicodeError, csv.Error) as error:
                self.respond(400, render_page(error=str(error)))
                return
            self.respond(200, render_batch_results(columns, rows, predictions, content, query=query, selected_class=selected_class, sort_by=sort_by, direction=direction))

        def import_profile(self) -> None:
            if self.headers.get_content_type() != "multipart/form-data":
                self.respond(415, render_page(error="Choose a JSON file using Load a saved profile."))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD_BYTES:
                    self.respond(413, render_page(error="Choose a profile smaller than 2 MiB, including upload overhead."))
                    return
                payload = uploaded_file(self.headers.get("Content-Type", ""), self.rfile.read(length), "profile")
                patient = read_patient_json(StringIO(payload.decode("utf-8-sig")))
                if not isinstance(patient, dict):
                    raise ValueError("Import one patient JSON object, not a batch.")
                validate_patient(patient)
            except (ValueError, TypeError, UnicodeError) as error:
                self.respond(400, render_page(error=str(error)))
                return
            self.respond(200, render_page(patient, notice="Review the restored details, then predict or edit them. No prediction has run yet."))

    return ThreadingHTTPServer((bind, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a local browser interface for sleep-disorder predictions.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="Listen address; use 0.0.0.0 inside a container with a loopback-only published port.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--diary", type=Path, help="Enable browser diary storage at this local CSV path.")
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if not arguments.model.is_file():
        parser.error(f"Model file does not exist: {arguments.model}")
    if arguments.diary and arguments.diary.resolve() == arguments.model.resolve():
        parser.error("--diary must differ from --model.")
    try:
        with make_server(arguments.model, arguments.port, bind=arguments.bind, diary_path=arguments.diary) as server:
            print(f"Open http://127.0.0.1:{server.server_port} · Press Ctrl+C to stop.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
