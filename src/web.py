"""Serve a local, browser-based patient prediction form."""

from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import parse_qs

from src.predict import CSV_NUMERIC_FEATURES, DEFAULT_MODEL_PATH, predict
from src.preprocessing import RAW_FEATURES

TEMPLATE_PATH = Path(__file__).with_name("web_ui.html")
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


def render_page(values: dict | None = None, *, result: str | None = None, error: str | None = None) -> str:
    values = DEFAULT_PATIENT if values is None else values
    fields = []
    for index, feature in enumerate(RAW_FEATURES):
        value = escape(str(values.get(feature, "")), quote=True)
        label = escape(LABELS.get(feature, feature))
        if feature in LIMITS:
            minimum, maximum = LIMITS[feature]
            attributes = f'type="number" min="{minimum}" max="{maximum}" step="any"'
        else:
            attributes = 'type="text"'
        fields.append(
            f'<label for="field-{index}">{label}<input id="field-{index}" '
            f'name="{escape(feature, quote=True)}" {attributes} value="{value}" required></label>'
        )
    if error:
        feedback = f'<section class="feedback error" role="alert"><h2>Check your details</h2><p>{escape(error)}</p></section>'
    elif result:
        feedback = f'<section class="feedback result" role="status"><p class="eyebrow">MODEL PREDICTION</p><h2>{escape(result)}</h2><p>This is an educational model output, not a diagnosis.</p></section>'
    else:
        feedback = '<section class="feedback"><h2>Start with an example</h2><p>The form contains sample values. Replace them with your own details, then select Predict.</p></section>'
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(
        fields="\n".join(fields), feedback=feedback,
    )


def make_server(model_path: Path = DEFAULT_MODEL_PATH, port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:
            # Patient input is neither persisted nor written to access logs.
            pass

        def respond(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
            else:
                self.respond(404, "Page not found.")

        def do_POST(self) -> None:
            if not self.local_request():
                self.respond(403, "Local requests only.")
                return
            if self.path != "/predict":
                self.respond(404, "Page not found.")
                return
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self.respond(415, "Submit the patient form as URL-encoded data.")
                return
            values = {}
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    self.respond(413, "Form data must contain between 1 and 16384 bytes.")
                    return
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True, max_num_fields=32)
                if any(len(items) != 1 for items in fields.values()):
                    raise ValueError("Each field must be submitted exactly once.")
                values = {key: items[0] for key, items in fields.items()}
                patient = dict(values)
                for feature in CSV_NUMERIC_FEATURES:
                    try:
                        patient[feature] = float(values.get(feature, ""))
                    except ValueError as error:
                        raise ValueError(f"'{feature}' must be a number.") from error
                result = predict(patient, model_path)
            except (ValueError, TypeError, UnicodeError) as error:
                self.respond(400, render_page(values, error=str(error)))
                return
            except OSError:
                self.respond(503, render_page(values, error="The model could not be loaded. Check the server's model path."))
                return
            self.respond(200, render_page(values, result=result))

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
