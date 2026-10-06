"""Reproduce the documented 16.6-MAE multiscale SJ-only submission.

Pipeline (matches ``feature_lag_mlp.representation_train``):

1. Train raw-lag ``TUNED_CONFIGS`` with seed 42 → baseline CSV.
2. Train ``multiscale_summaries`` with seed 42 for both cities.
3. Write full multiscale submission and ``_sj_only`` hybrid (SJ multiscale,
   IQ from baseline).

Uses TensorFlow when available; otherwise the research NumPy trainer.
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from feature_lag_mlp.config import (
    CITIES,
    KEY_COLUMNS,
    TARGET_COLUMN,
    TUNED_CONFIGS,
    TrainingSchedule,
)
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.features import build_test_matrix, build_training_matrix
from feature_lag_mlp.representations import (
    MULTISCALE_SUMMARIES,
    build_representation_test_matrix,
    build_representation_training_matrix,
)
from feature_lag_mlp.training import assemble_submission, prediction_frame

from research.training_backend import fit_city, predict_raw, resolve_backend, tensorflow_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
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


def _city_hybrid(baseline: pd.DataFrame, candidate: pd.DataFrame, city: str) -> pd.DataFrame:
    hybrid = baseline.copy()
    city_rows = hybrid["city"].eq(city)
    candidate_city = candidate.loc[candidate["city"].eq(city), TARGET_COLUMN]
    hybrid.loc[city_rows, TARGET_COLUMN] = candidate_city.to_numpy()
    return hybrid.astype({TARGET_COLUMN: int})


def _train_city_raw(
    train: pd.DataFrame,
    test: pd.DataFrame,
    city: str,
    seed: int,
    schedule: TrainingSchedule,
    backend: str,
    verbose: int,
) -> np.ndarray:
    train_x, train_y = build_training_matrix(train, city)
    test_x = build_test_matrix(train, test, city)
    model, _ = fit_city(
        train_x,
        train_y,
        city,
        TUNED_CONFIGS[city],
        schedule,
        seed,
        verbose,
        backend=backend,
    )
    return predict_raw(model, test_x, backend)


def _train_city_multiscale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    city: str,
    seed: int,
    schedule: TrainingSchedule,
    backend: str,
    verbose: int,
) -> np.ndarray:
    train_x, train_y = build_representation_training_matrix(
        train, city, MULTISCALE_SUMMARIES
    )
    test_x = build_representation_test_matrix(
        train, test, city, MULTISCALE_SUMMARIES
    )
    model, _ = fit_city(
        train_x,
        train_y,
        city,
        TUNED_CONFIGS[city],
        schedule,
        seed,
        verbose,
        backend=backend,
    )
    return predict_raw(model, test_x, backend)


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

    print(f"[1/3] Raw-lag tuned baseline, seed={args.seed}, backend={backend}")
    baseline_frames = []
    for city in CITIES:
        raw = _train_city_raw(
            train, test, city, args.seed, schedule, backend, args.verbose
        )
        cases = np.maximum(raw, 0).astype(int)
        baseline_frames.append(prediction_frame(test, city, cases))
    baseline = assemble_submission(template, baseline_frames)
    baseline_path = args.output_dir / "submission_feature_lag_mlp_tuned_seed42.csv"
    baseline.to_csv(baseline_path, index=False)
    print(f"  Wrote {baseline_path}")

    print(f"[2/3] Multiscale summaries, seed={args.seed}")
    multiscale_frames = []
    for city in CITIES:
        raw = _train_city_multiscale(
            train, test, city, args.seed, schedule, backend, args.verbose
        )
        cases = np.maximum(raw, 0).astype(int)
        multiscale_frames.append(prediction_frame(test, city, cases))
    candidate = assemble_submission(template, multiscale_frames)
    stem = f"feature_lag_mlp_{MULTISCALE_SUMMARIES}_seed{args.seed}"
    candidate_path = args.output_dir / f"submission_{stem}.csv"
    candidate.to_csv(candidate_path, index=False)
    print(f"  Wrote {candidate_path}")

    print("[3/3] City-isolation hybrids")
    written = [candidate_path]
    for city in CITIES:
        hybrid = _city_hybrid(baseline, candidate, city)
        hybrid_path = args.output_dir / f"submission_{stem}_{city}_only.csv"
        hybrid.to_csv(hybrid_path, index=False)
        written.append(hybrid_path)
        print(f"  Wrote {hybrid_path}")

    best_path = args.output_dir / f"submission_{stem}_sj_only.csv"
    metadata = {
        "method": "multiscale summaries reproduction",
        "documented_hidden_test_mae_sj_only": 16.6,
        "backend_used": backend,
        "tensorflow_available": tensorflow_available(),
        "seed": args.seed,
        "baseline_submission": str(baseline_path),
        "best_submission": str(best_path),
        "all_outputs": [str(path) for path in written],
        "best_submission_sha256": _sha256(best_path),
    }
    meta_path = args.output_dir / "multiscale_best_reproduction.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("\nBest documented candidate:")
    print(f"  {best_path}")
    print(candidate.groupby("city")[TARGET_COLUMN].agg(["min", "median", "mean", "max"]))
    print(
        best_path.read_text(encoding="utf-8").count("\n"),
        "lines (expect 417 data rows + header)",
    )


if __name__ == "__main__":
    main()
