"""Train one documented feature-specific lag-window MLP configuration.

This module deliberately contains orchestration only. The transformations and
model are implemented in the smaller modules next to it so each stage can be
read and tested independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from .config import (
    CITIES,
    LAG_WINDOWS,
    NORMALIZATION_CHOICES,
    PROFILE_CONFIGS,
    SHARED_SJ_NORMALIZATION,
    TrainingSchedule,
)
from .data import load_competition_data
from .features import build_test_matrix, build_training_matrix
from .training import (
    assemble_submission,
    fit_arrays,
    predict_cases,
    prediction_frame,
    set_tensorflow_seed,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONFIGS),
        default="improved",
        help="Named architecture from config.py; tuned is the local-CV winner.",
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_CHOICES,
        default=SHARED_SJ_NORMALIZATION,
        help="shared-sj is confirmed; city-specific is the normalization ablation.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--lr-mode",
        choices=("max", "min"),
        default="max",
        help=(
            "ReduceLROnPlateau direction. max preserves the confirmed historical "
            "behavior; min is the logically correct direction for MAE."
        ),
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate matrices without importing TensorFlow.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_metadata(train, test, normalization: str, profile: str) -> dict[str, object]:
    cities: dict[str, object] = {}
    for city in CITIES:
        train_x, train_y = build_training_matrix(train, city, normalization)
        test_x = build_test_matrix(train, test, city, normalization)
        cities[city] = {
            "training_shape": list(train_x.shape),
            "label_shape": list(train_y.shape),
            "test_shape": list(test_x.shape),
            "model": PROFILE_CONFIGS[profile][city].to_dict(),
        }
    return cities


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode=args.lr_mode,
    )

    # Stage 1: read official tables and verify submission keys.
    train, test, template = load_competition_data(args.data_dir)
    stem = args.experiment_name or f"feature_lag_mlp_{args.profile}"
    metadata: dict[str, object] = {
        "method": "feature-specific lag-window MLP",
        "profile": args.profile,
        "normalization": args.normalization,
        "lag_windows": LAG_WINDOWS,
        "uses_external_saved_weights": False,
        "cities": _matrix_metadata(train, test, args.normalization, args.profile),
    }
    if args.prepare_only:
        path = args.output_dir / f"{stem}_prepared.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata["cities"], indent=2))
        print(f"Wrote {path}")
        return

    # Stage 2: fix randomness once, then fit independent city models.
    import tensorflow as tf

    set_tensorflow_seed(args.seed)
    metadata.update(
        {
            "tensorflow": tf.__version__,
            "seed": args.seed,
            **schedule.__dict__,
        }
    )
    city_predictions = []
    for city in CITIES:
        train_x, train_y = build_training_matrix(train, city, args.normalization)
        test_x = build_test_matrix(train, test, city, args.normalization)
        result = fit_arrays(
            train_x,
            train_y,
            city,
            PROFILE_CONFIGS[args.profile][city],
            schedule,
            args.verbose,
        )
        raw, cases = predict_cases(result.model, test_x)
        city_predictions.append(prediction_frame(test, city, cases))

        model_path = args.output_dir / f"{stem}_{city}.keras"
        history_path = args.output_dir / f"{stem}_{city}_history.csv"
        result.model.save(model_path)
        result.history.to_csv(history_path, index=False)
        city_metadata = metadata["cities"][city]  # type: ignore[index]
        city_metadata.update(  # type: ignore[union-attr]
            {
                "final_training_mae": float(result.history["mae"].iloc[-1]),
                "raw_prediction_mean": float(np.mean(raw)),
                "raw_prediction_max": float(np.max(raw)),
                "submission_mean": float(np.mean(cases)),
                "submission_max": int(np.max(cases)),
                "model_sha256": _sha256(model_path),
            }
        )

    # Stage 3: put predictions back in the official row order and audit outputs.
    submission = assemble_submission(template, city_predictions)
    submission_path = args.output_dir / f"submission_{stem}.csv"
    submission.to_csv(submission_path, index=False)
    metadata["submission_sha256"] = _sha256(submission_path)
    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {submission_path}")
    print(
        submission.groupby("city")["total_cases"]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
