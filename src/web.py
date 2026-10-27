"""Serve a local, browser-based patient prediction form."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from string import Template
from urllib.parse import parse_qs

from src.predict import (
    CSV_NUMERIC_FEATURES, DEFAULT_MODEL_PATH, PREDICTION_COLUMN,
    predict, predict_many, read_patient_csv,
)
from src.preprocessing import RAW_FEATURES
from src.model_info import inspect_model

TEMPLATE_PATH = Path(__file__).with_name("web_ui.html")
EXAMPLE_CSV_PATH = Path(__file__).resolve().parents[1] / "examples" / "patients.csv"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
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


def render_page(values: dict | None = None, *, result: str | None = None, error: str | None = None) -> str:
    values = DEFAULT_PATIENT if values is None else values
    if error:
        feedback = f'<section class="feedback error" role="alert"><h2>Check your details</h2><p>{escape(error)}</p></section>'
    elif result:
        feedback = f'<section class="feedback result" role="status"><p class="eyebrow">MODEL PREDICTION</p><h2>{escape(result)}</h2><p>This is an educational model output, not a diagnosis.</p></section>'
    else:
        feedback = '<section class="feedback"><h2>Start with an example</h2><p>The form contains sample values. Replace them with your own details, then select Predict.</p></section>'
    workspace = Template(TEMPLATE_PATH.with_name("web_patient_form.html").read_text(encoding="utf-8")).substitute(
        fields=render_fields(values), feedback=feedback,
    )
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(workspace=workspace, layout_class="layout")


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


def make_server(model_path: Path = DEFAULT_MODEL_PATH, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:
            # Patient input is neither persisted nor written to access logs.
            pass

        def respond(self, status: int, body: str, *, content_type: str = "text/html; charset=utf-8", filename: str | None = None) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(encoded)

        def local_request(self) -> bool:
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (
                origin is None or origin in {f"http://{host}" for host in hosts}
            )

        def do_GET(self) -> None:
            if not self.local_request():
                self.respond(403, "Local requests only.")
            elif self.path == "/":
                self.respond(200, render_page())
            elif self.path == "/compare":
                self.respond(200, render_comparison())
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
                self.respond(403, "Local requests only.")
                return
            if self.path == "/batch":
                self.batch_predict()
                return
            if self.path not in ("/predict", "/compare", "/report"):
                self.respond(404, "Page not found.")
                return
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self.respond(415, "Submit the patient form as URL-encoded data.")
                return
            values = {}
            renderer = render_comparison if self.path == "/compare" else render_page
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    self.respond(413, "Form data must contain between 1 and 16384 bytes.")
                    return
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True, max_num_fields=32)
                if any(len(items) != 1 for items in fields.values()):
                    raise ValueError("Each field must be submitted exactly once.")
                values = {key: items[0] for key, items in fields.items()}
                if self.path == "/compare":
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

        def batch_predict(self) -> None:
            if self.headers.get_content_type() != "multipart/form-data":
                self.respond(415, render_page(error="Choose a CSV using the batch upload form."))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD_BYTES:
                    self.respond(413, render_page(error="Choose a CSV smaller than 2 MiB, including upload overhead."))
                    return
                content_type = self.headers.get("Content-Type", "")
                message = BytesParser(policy=policy.default).parsebytes(
                    f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
                    + self.rfile.read(length)
                )
                if not message.is_multipart() or any(part.defects for part in message.walk()):
                    raise ValueError("The upload is incomplete or malformed. Choose the CSV again.")
                parts = list(message.iter_parts())
                if len(parts) != 1 or parts[0].get_param("name", header="content-disposition") != "patients":
                    raise ValueError("Upload exactly one patient CSV file.")
                upload = parts[0]
                if not upload.get_filename():
                    raise ValueError("Choose a CSV file to upload.")
                payload = upload.get_payload(decode=True)
                if payload is None:
                    raise ValueError("The uploaded file is empty or malformed.")
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
            self.respond(200, output.getvalue(), content_type="text/csv; charset=utf-8", filename="sleep-predictions.csv")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a local browser interface for sleep-disorder predictions.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if not arguments.model.is_file():
        parser.error(f"Model file does not exist: {arguments.model}")
    try:
        with make_server(arguments.model, arguments.port) as server:
            print(f"Open http://127.0.0.1:{server.server_port} · Press Ctrl+C to stop.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
