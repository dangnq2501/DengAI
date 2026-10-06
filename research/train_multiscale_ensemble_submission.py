"""Multiscale SJ improvements: seed median ensemble (+ optional supervised mix-in).

t-SNE/PCA global compression did not beat the 180-dim multiscale baseline on
temporal CV in ``search_multiscale_reduction.py``. This script instead:

* averages several multiscale MLP seeds with a **median** (robust to outliers);
* keeps IQ from the confirmed seed-42 raw-lag baseline;
* optionally blends SJ multiscale median with the baseline SJ vector.
"""

from __future__ import annotations

import argparse
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

from feature_lag_mlp.config import CITIES, TARGET_COLUMN, TUNED_CONFIGS, TrainingSchedule
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.representations import (
    MULTISCALE_SUMMARIES,
    build_representation_test_matrix,
    build_representation_training_matrix,
)
from feature_lag_mlp.training import assemble_submission, prediction_frame

from research.training_backend import fit_city, predict_raw, resolve_backend


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--baseline-submission",
        type=Path,
        default=ROOT / "artifacts" / "submission_feature_lag_mlp_tuned_seed42.csv",
    )
    parser.add_argument("--seeds", type=parse_int_list, default=(17, 42, 73))
    parser.add_argument(
        "--aggregate",
        choices=("median", "mean"),
        default="median",
    )
    parser.add_argument(
        "--sj-baseline-weight",
        type=float,
        default=0.0,
        help="Blend SJ multiscale aggregate with SJ rows from baseline (0=off).",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "tensorflow", "numpy"),
        default="auto",
    )
    return parser.parse_args()


def _city_hybrid(baseline: pd.DataFrame, candidate: pd.DataFrame, city: str) -> pd.DataFrame:
    hybrid = baseline.copy()
    rows = hybrid["city"].eq(city)
    hybrid.loc[rows, TARGET_COLUMN] = candidate.loc[
        candidate["city"].eq(city), TARGET_COLUMN
    ].to_numpy()
    hybrid[TARGET_COLUMN] = hybrid[TARGET_COLUMN].astype(int)
    return hybrid


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.baseline_submission.is_file():
        raise FileNotFoundError(f"Missing baseline: {args.baseline_submission}")

    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )
    backend = resolve_backend(args.backend)
    train, test, template = load_competition_data(args.data_dir)
    baseline = pd.read_csv(args.baseline_submission)

    sj_raws: list[np.ndarray] = []
    train_x, train_y = build_representation_training_matrix(
        train, "sj", MULTISCALE_SUMMARIES
    )
    test_x = build_representation_test_matrix(
        train, test, "sj", MULTISCALE_SUMMARIES
    )
    for seed in args.seeds:
        print(f"[sj multiscale] seed={seed}", flush=True)
        model, _ = fit_city(
            train_x,
            train_y,
            "sj",
            TUNED_CONFIGS["sj"],
            schedule,
            seed,
            args.verbose,
            backend=backend,
        )
        sj_raws.append(predict_raw(model, test_x, backend))

    stacked = np.stack(sj_raws, axis=0)
    sj_raw = np.median(stacked, axis=0) if args.aggregate == "median" else stacked.mean(axis=0)
    if args.sj_baseline_weight > 0:
        baseline_sj = baseline.loc[baseline["city"].eq("sj"), TARGET_COLUMN].to_numpy(
            float
        )
        w = args.sj_baseline_weight
        sj_raw = w * baseline_sj + (1.0 - w) * sj_raw

    sj_cases = np.maximum(sj_raw, 0).astype(int)
    iq_cases = baseline.loc[baseline["city"].eq("iq"), TARGET_COLUMN].to_numpy(int)
    city_frames = [
        prediction_frame(test, "sj", sj_cases),
        prediction_frame(test, "iq", iq_cases),
    ]
    candidate = assemble_submission(template, city_frames)
    stem = f"feature_lag_mlp_multiscale_{args.aggregate}_seed{'_'.join(map(str, args.seeds))}"
    out = args.output_dir / f"submission_{stem}_sj_only.csv"
    candidate.to_csv(out, index=False)

    metadata = {
        "backend": backend,
        "seeds": list(args.seeds),
        "aggregate": args.aggregate,
        "sj_baseline_weight": args.sj_baseline_weight,
        "output": str(out),
        "sj_submission_mean": float(sj_cases.mean()),
        "sj_submission_max": int(sj_cases.max()),
    }
    meta = args.output_dir / f"{stem}.json"
    meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(
        candidate.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
    )


if __name__ == "__main__":
    main()
