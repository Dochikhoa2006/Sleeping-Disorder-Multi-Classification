"""Export a JSON Schema for patient JSON inputs without loading a model."""

import argparse
import json
from pathlib import Path

from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, NUMERIC_RANGES
from src.preprocessing import RAW_FEATURES


def patient_schema(*, batch: bool = False) -> dict:
    properties = {}
    for field in RAW_FEATURES:
        if field in NUMERIC_RANGES:
            low, high = NUMERIC_RANGES[field]
            properties[field] = {"type": "number", "minimum": low, "maximum": high}
        elif field == "Blood Pressure":
            properties[field] = {
                "type": "string", "pattern": r"^\s*[0-9]+(?:\.[0-9]+)?\s*/\s*[0-9]+(?:\.[0-9]+)?\s*$",
                "description": "Systolic/diastolic; systolic 50–300, diastolic 30–200, systolic must exceed diastolic. Validate these numeric relationships separately.",
            }
        else:
            properties[field] = {"type": "string", "minLength": 1, "pattern": r"\S"}
    patient = {"type": "object", "required": list(RAW_FEATURES), "properties": properties,
               "additionalProperties": True}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Sleep classifier patient batch" if batch else "Sleep classifier patient",
            **({"type": "array", "minItems": 1, "items": patient} if batch else patient)}


def api_input_schema() -> dict:
    """Describe both JSON shapes accepted by prediction and validation APIs."""
    patient = patient_schema()
    dialect = patient.pop("$schema")
    patient.pop("title")
    return {"$schema": dialect, "title": "Patient API input",
            "$defs": {"patient": patient},
            "oneOf": [{"$ref": "#/$defs/patient"},
                      {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/patient"}}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument("--batch", action="store_true", help="Describe a nonempty array instead of a single patient.")
    shapes.add_argument("--api", action="store_true", help="Describe either a patient object or nonempty array, as accepted by the API.")
    parser.add_argument("--output", type=Path, help="Atomic JSON file; omit for standard output.")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() == DEFAULT_MODEL_PATH.resolve():
            raise ValueError("Output must differ from the shipped model path.")
        schema = api_input_schema() if args.api else patient_schema(batch=args.batch)
        payload = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as destination:
                destination.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
