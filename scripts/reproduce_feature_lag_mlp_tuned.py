"""Reproduce ``submission_feature_lag_mlp_tuned.csv`` from ``src/feature_lag_mlp``.

Uses the frozen :data:`feature_lag_mlp.config.TUNED_CONFIGS`, the same
three-seed raw ensemble as ``feature_lag_mlp.tune``, and official integer
truncation (``maximum(raw, 0).astype(int)``).

When TensorFlow is unavailable (for example low disk space in ``bkk``), falls
back to the research NumPy trainer, which approximates the Keras schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from feature_lag_mlp.config import CITIES, TARGET_COLUMN, TUNED_CONFIGS, TrainingSchedule
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.training import assemble_submission, prediction_frame
from feature_lag_mlp.tune import parse_int_list, train_full_city as train_full_city_tf

from research.candidates import tuned_candidate
from research.submission_training import train_full_city as train_full_city_fallback
from research.training_backend import resolve_backend, tensorflow_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--final-seeds",
        type=parse_int_list,
        default=(17, 42, 73),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--backend",
        choices=("auto", "tensorflow", "numpy"),
        default="auto",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )
    backend = resolve_backend(args.backend)
    train, test, template = load_competition_data(args.data_dir)

    metadata: dict[str, object] = {
        "method": "feature_lag_mlp tuned reproduction",
        "backend_requested": args.backend,
        "backend_used": backend,
        "tensorflow_available": tensorflow_available(),
        "final_seeds": list(args.final_seeds),
        "tuned_configs": {city: TUNED_CONFIGS[city].to_dict() for city in CITIES},
        "cities": {},
    }

    city_predictions = []
    for city in CITIES:
        candidate = tuned_candidate(city)
        if backend == "tensorflow":
            raw, runs = train_full_city_tf(
                train,
                test,
                city,
                candidate,
                args.final_seeds,
                schedule,
                args.output_dir,
                args.verbose,
            )
        else:
            raw, runs, _ = train_full_city_fallback(
                train,
                test,
                city,
                candidate,
                args.final_seeds,
                schedule,
                args.verbose,
                backend=backend,
            )
        cases = np.maximum(raw, 0).astype(int)
        city_predictions.append(prediction_frame(test, city, cases))
        metadata["cities"][city] = {
            "candidate": candidate.name,
            "final_runs": runs,
            "ensemble_prediction_mean": float(np.mean(cases)),
            "ensemble_prediction_max": int(np.max(cases)),
        }

    submission = assemble_submission(template, city_predictions)
    submission_path = args.output_dir / "submission_feature_lag_mlp_tuned.csv"
    submission.to_csv(submission_path, index=False)
    metadata["submission_sha256"] = _sha256(submission_path)
    metadata_path = args.output_dir / "feature_lag_mlp_tuning.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Backend: {backend}")
    print(f"Wrote {submission_path}")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
