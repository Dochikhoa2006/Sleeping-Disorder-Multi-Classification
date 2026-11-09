"""Sample whole patient-profile groups reproducibly without loading a model."""

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import random

import pandas as pd

from src.evaluation import feature_fingerprints
from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH, read_patient_csv
from src.preprocessing import RANDOM_STATE


def sample_csv(source: Path, output: Path, *, profiles: int, seed: int = RANDOM_STATE) -> dict:
    if type(profiles) is not int or profiles < 1:
        raise ValueError("Profile count must be a positive integer.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer from 0 to 4294967295.")
    if output.resolve() in {source.resolve(), DEFAULT_MODEL_PATH.resolve()}:
        raise ValueError("Output must differ from the input and shipped model paths.")
    snapshot = source.read_bytes()
    columns, rows, patients = read_patient_csv(StringIO(snapshot.decode("utf-8-sig"), newline=""))
    fingerprints = feature_fingerprints(pd.DataFrame(patients)).tolist()
    # Sorting makes the chosen profiles independent of input row order.
    unique = sorted(set(fingerprints))
    if profiles > len(unique):
        raise ValueError(f"Requested {profiles} profiles, but the input has only {len(unique)}.")
    selected = set(random.Random(seed).sample(unique, profiles))
    retained = [row for row, fingerprint in zip(rows, fingerprints, strict=True) if fingerprint in selected]
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(retained)
    payload = buffer.getvalue()
    report = {"seed": seed, "input_rows": len(rows), "input_unique_profiles": len(unique),
              "sampled_rows": len(retained), "sampled_unique_profiles": profiles,
              "source_sha256": hashlib.sha256(snapshot).hexdigest(),
              "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
    with atomic_text_output(output, newline="") as destination:
        destination.write(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", type=int, required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    try:
        print(json.dumps(sample_csv(args.input_csv, args.output, profiles=args.profiles, seed=args.seed), indent=2))
    except (OSError, ValueError, TypeError, csv.Error) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
