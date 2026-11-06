"""Diagnose a local installation without importing ML packages unless requested."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "web_ui.html", "web_patient_form.html", "web_compare.html", "web_model.html",
    "web_report.html", "web_batch_results.html", "web_batch_comparison.html",
    "web_evaluate.html", "web_evaluation_results.html", "web_evaluation_rows.html",
    "web_input_guide.html", "web_scenarios.html", "web_diary.html", "web_diary_calendar.html",
    "web_timer.html", "web_timer.js", "diary_report.html", "web_drift.html", "web_drift_results.html",
)


def _smoke_test(snapshot: bytes, patient_path: Path) -> str:
    from io import BytesIO
    import joblib
    import pandas as pd
    from src.predict import load_patient_json, validate_patient
    from src.preprocessing import RAW_FEATURES

    patient = load_patient_json(patient_path)
    if not isinstance(patient, dict):
        raise ValueError("The example must contain one patient object.")
    validate_patient(patient)
    model = joblib.load(BytesIO(snapshot))
    predictions = model.predict(pd.DataFrame([{key: patient[key] for key in RAW_FEATURES}]))
    if len(predictions) != 1:
        raise ValueError("Model did not return one prediction for the example.")
    return str(predictions[0])


def check_installation(*, root: Path = PROJECT_ROOT, model_path: Path | None = None,
                       smoke_test: bool = False) -> dict:
    checks = []
    def add(name, passed, detail):
        checks.append({"name": name, "status": "ok" if passed else "failed", "detail": detail})

    add("python", sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}; requires 3.11 or newer.")
    try:
        requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        pins = []
        for line in requirements:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, expected = line.partition("==")
            if not separator or not name or not expected:
                raise ValueError("Expected exact package==version entries in requirements.txt.")
            pins.append((name, expected))
        if not pins:
            raise ValueError("No dependency pins found in requirements.txt.")
        add("requirements", True, f"Read {len(pins)} pinned dependencies.")
        for name, expected in pins:
            try:
                actual = metadata.version(name)
                add("dependency:" + name, actual == expected, f"Expected {expected}; installed {actual}.")
            except metadata.PackageNotFoundError:
                add("dependency:" + name, False, f"Not installed; expected {expected}.")
    except (OSError, ValueError) as error:
        add("requirements", False, str(error))
    missing = ["src/" + name for name in ASSETS if not (root / "src" / name).is_file()]
    missing += ["examples/" + name for name in ("patient.json", "patients.csv")
                if not (root / "examples" / name).is_file()]
    add("runtime_files", not missing, "Missing: " + ", ".join(missing) if missing else "All web assets and examples are present.")
    model_path = model_path or root / "models" / "sleep_disorder_pipeline.joblib"
    snapshot, digest = None, None
    try:
        snapshot = model_path.read_bytes()
        if not snapshot:
            raise ValueError("Model file is empty.")
        digest = hashlib.sha256(snapshot).hexdigest()
        add("model_file", True, f"Readable artifact: {len(snapshot)} bytes. Not deserialized by this check.")
    except (OSError, ValueError) as error:
        add("model_file", False, str(error))
    if smoke_test:
        if any(check["status"] == "failed" for check in checks):
            checks.append({"name": "prediction", "status": "skipped", "detail": "Resolve failed checks before running a prediction."})
        else:
            try:
                prediction = _smoke_test(snapshot, root / "examples" / "patient.json")
                add("prediction", True, f"Synthetic example predicted successfully: {prediction}.")
            except Exception as error:
                add("prediction", False, f"{type(error).__name__}: {error}")
    return {"healthy": all(check["status"] == "ok" for check in checks),
            "model_sha256": digest, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="Load the trusted model and predict the synthetic example.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable diagnostic report.")
    args = parser.parse_args()
    report = check_installation(model_path=args.model, smoke_test=args.smoke_test)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f'[{check["status"].upper()}] {check["name"]}: {check["detail"]}')
        print("Installation checks passed." if report["healthy"] else "Installation needs attention.")
    if not report["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
