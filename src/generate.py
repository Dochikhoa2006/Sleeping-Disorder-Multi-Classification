"""Generate clearly marked synthetic patient inputs for demos and batch testing."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import random

from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, validate_patients
from src.preprocessing import RANDOM_STATE, RAW_FEATURES

COLUMNS = ["Record ID", "Data Source", *RAW_FEATURES]


def generate_patients(rows: int = 100, seed: int = RANDOM_STATE,
                      *, edge_cases: bool = False) -> list[dict]:
    """Create input-valid fixtures, without modeling real population relationships."""
    if not 1 <= rows <= 10000:
        raise ValueError("Choose between 1 and 10000 rows.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be between 0 and 4294967295.")
    if edge_cases and rows < 2:
        raise ValueError("Edge cases require at least two rows.")
    rng = random.Random(seed)
    patients = []
    for index in range(rows):
        systolic = rng.randint(90, 180)
        diastolic = rng.randint(50, min(110, systolic - 1))
        patient = {
            "Record ID": f"SYN-{index + 1:06d}", "Data Source": "Synthetic demo",
            "Gender": rng.choice(["Male", "Female"]), "Age": rng.randint(18, 90),
            "Occupation": rng.choice(["Engineer", "Teacher", "Nurse", "Doctor", "Accountant", "Software Engineer"]),
            "Sleep Duration": rng.randint(30, 100) / 10,
            "Quality of Sleep": rng.randint(1, 10),
            "Physical Activity Level": rng.randint(0, 180), "Stress Level": rng.randint(1, 10),
            "BMI Category": rng.choice(["Normal", "Normal Weight", "Overweight", "Obese"]),
            "Blood Pressure": f"{systolic}/{diastolic}", "Heart Rate": rng.randint(45, 110),
            "Daily Steps": rng.randint(0, 20000),
        }
        if edge_cases and index < 2:
            low, high = (
                {"Age": 1, "Sleep Duration": 0, "Quality of Sleep": 1,
                 "Physical Activity Level": 0, "Stress Level": 1, "Heart Rate": 20,
                 "Daily Steps": 0, "Blood Pressure": "50/30"},
                {"Age": 120, "Sleep Duration": 24, "Quality of Sleep": 10,
                 "Physical Activity Level": 1440, "Stress Level": 10, "Heart Rate": 250,
                 "Daily Steps": 100000, "Blood Pressure": "300/200"},
            )
            patient.update(low if index == 0 else high)
            patient["Data Source"] = "Synthetic boundary fixture"
        patients.append(patient)
    validate_patients(patients)
    return patients


def serialize_patients(patients: list[dict], format_name: str) -> str:
    if format_name == "json":
        return json.dumps(patients, indent=2, allow_nan=False) + "\n"
    if format_name != "csv":
        raise ValueError("Choose csv or json output.")
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(patients)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--edge-cases", action="store_true",
                        help="Use the first two rows for input-validation boundary fixtures.")
    parser.add_argument("--output", type=Path, help="Save atomically; omit or use - for standard output.")
    args = parser.parse_args()
    if args.output and args.output != Path("-") and args.output.resolve() == DEFAULT_MODEL_PATH.resolve():
        raise SystemExit("Error: output must differ from the shipped model path.")
    try:
        payload = serialize_patients(generate_patients(args.rows, args.seed, edge_cases=args.edge_cases), args.format)
        if args.output and args.output != Path("-"):
            with atomic_text_output(args.output, newline="") as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
