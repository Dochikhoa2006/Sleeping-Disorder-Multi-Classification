"""Benchmark saved-model inference on reproducible synthetic patient inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.generate import generate_patients
from src.output import atomic_text_output
from src.predict import DEFAULT_MODEL_PATH
from src.preprocessing import RANDOM_STATE, RAW_FEATURES


def benchmark_model(model_path: Path = DEFAULT_MODEL_PATH, *, batch_sizes=(1, 10, 100),
                    runs: int = 10, warmup: int = 2, seed: int = RANDOM_STATE) -> dict:
    sizes = list(batch_sizes)
    if not sizes or len(sizes) > 10 or len(set(sizes)) != len(sizes) or any(
            not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 10000 for size in sizes):
        raise ValueError("Choose 1–10 distinct integer batch sizes, each between 1 and 10000.")
    if not 1 <= runs <= 100 or not 0 <= warmup <= 10:
        raise ValueError("Choose 1–100 measured runs and 0–10 warmup runs per batch size.")
    patients = generate_patients(max(sizes), seed)
    frames = {size: pd.DataFrame(patients[:size])[RAW_FEATURES] for size in sizes}
    started = perf_counter()
    snapshot = model_path.read_bytes()
    model = joblib.load(BytesIO(snapshot))
    load_seconds = perf_counter() - started
    if not np.isfinite(load_seconds) or load_seconds < 0:
        raise ValueError("Invalid model-load timing measurement.")
    batches = []
    for size in sizes:
        frame = frames[size]
        for _ in range(warmup):
            if len(model.predict(frame)) != size:
                raise ValueError("Model must return one prediction per patient.")
        samples = []
        for _ in range(runs):
            started = perf_counter()
            predictions = model.predict(frame)
            elapsed = perf_counter() - started
            if len(predictions) != size:
                raise ValueError("Model must return one prediction per patient.")
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Prediction timing must be positive and finite.")
            samples.append(elapsed * 1000)
        average = float(np.mean(samples))
        batches.append({"rows": size, "runs": runs, "warmup_runs": warmup,
                        "mean_ms": average, "median_ms": float(np.median(samples)),
                        "p95_ms": float(np.percentile(samples, 95)),
                        "min_ms": min(samples), "max_ms": max(samples),
                        "rows_per_second": size * 1000 / average, "samples_ms": samples})
    return {"model_sha256": hashlib.sha256(snapshot).hexdigest(),
            "model_name": getattr(model, "model_name_", type(model).__name__),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed, "input_source": "synthetic demo patients",
            "model_read_and_load_ms": load_seconds * 1000,
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "machine": platform.machine(), "scikit_learn": sklearn.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__},
            "batches": batches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, help="Save JSON atomically; omit for standard output.")
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.model.resolve():
        raise SystemExit("Error: output must differ from model path.")
    try:
        report = benchmark_model(args.model, batch_sizes=args.batch_sizes, runs=args.runs,
                                 warmup=args.warmup, seed=args.seed)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with atomic_text_output(args.output) as output:
                output.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, TypeError, AttributeError, EOFError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
