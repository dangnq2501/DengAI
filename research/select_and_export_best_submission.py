"""Pick the best multiscale SJ strategy on a chronological holdout, then export CSV."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from feature_lag_mlp.config import TARGET_COLUMN, TUNED_CONFIGS, TrainingSchedule
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.features import build_test_matrix, build_training_matrix
from feature_lag_mlp.representations import (
    MULTISCALE_SUMMARIES,
    build_representation_test_matrix,
    build_representation_training_matrix,
)
from feature_lag_mlp.training import assemble_submission, prediction_frame

from research.training_backend import fit_city, predict_raw, resolve_backend


@dataclass(frozen=True)
class Strategy:
    name: str
    seeds: tuple[int, ...]
    aggregate: str  # single | mean | median
    sj_baseline_weight: float = 0.0
    iq_from_baseline: bool = True
    rounding: str = "truncate"  # truncate | nearest


def to_cases(raw: np.ndarray, rounding: str) -> np.ndarray:
    clipped = np.maximum(raw, 0.0)
    if rounding == "nearest":
        return np.floor(clipped + 0.5).astype(int)
    return clipped.astype(int)


def mae(y: np.ndarray, raw: np.ndarray, rounding: str) -> float:
    return float(np.mean(np.abs(y - to_cases(raw, rounding))))


def sj_holdout_split(
    train: pd.DataFrame, holdout_weeks: int = 52
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sj = train.loc[train["city"].eq("sj")].sort_values("week_start_date").reset_index(drop=True)
    if len(sj) <= holdout_weeks + 100:
        raise ValueError("Not enough SJ history for holdout")
    cutoff = len(sj) - holdout_weeks
    prefix_sj = sj.iloc[:cutoff]
    holdout = sj.iloc[cutoff:]
    iq = train.loc[train["city"].eq("iq")].sort_values("week_start_date")
    fraction = cutoff / len(sj)
    iq_end = max(53, int(len(iq) * fraction))
    fold_train = pd.concat([prefix_sj, iq.iloc[:iq_end]], ignore_index=True)
    y_hold = holdout[TARGET_COLUMN].to_numpy(float)
    return fold_train, holdout, y_hold


def predict_sj_multiscale(
    fold_train: pd.DataFrame,
    predict_frame: pd.DataFrame,
    strategy: Strategy,
    schedule: TrainingSchedule,
    backend: str,
    verbose: int,
    baseline_raw: np.ndarray | None = None,
) -> np.ndarray:
    train_x, train_y = build_representation_training_matrix(
        fold_train, "sj", MULTISCALE_SUMMARIES
    )
    test_x = build_representation_test_matrix(
        fold_train, predict_frame, "sj", MULTISCALE_SUMMARIES
    )
    raws: list[np.ndarray] = []
    for seed in strategy.seeds:
        model, _ = fit_city(
            train_x,
            train_y,
            "sj",
            TUNED_CONFIGS["sj"],
            schedule,
            seed,
            verbose,
            backend=backend,
        )
        raws.append(predict_raw(model, test_x, backend))
    if strategy.aggregate == "single":
        out = raws[0]
    elif strategy.aggregate == "mean":
        out = np.mean(np.stack(raws), axis=0)
    elif strategy.aggregate == "median":
        out = np.median(np.stack(raws), axis=0)
    else:
        raise ValueError(strategy.aggregate)
    if strategy.sj_baseline_weight > 0 and baseline_raw is not None:
        w = strategy.sj_baseline_weight
        out = w * baseline_raw + (1.0 - w) * out
    return out


def predict_sj_raw_lag(
    fold_train: pd.DataFrame,
    predict_frame: pd.DataFrame,
    seed: int,
    schedule: TrainingSchedule,
    backend: str,
    verbose: int,
) -> np.ndarray:
    train_x, train_y = build_training_matrix(fold_train, "sj")
    test_x = build_test_matrix(fold_train, predict_frame, "sj")
    model, _ = fit_city(
        train_x, train_y, "sj", TUNED_CONFIGS["sj"], schedule, seed, verbose, backend=backend
    )
    return predict_raw(model, test_x, backend)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    p.add_argument("--holdout-weeks", type=int, default=52)
    p.add_argument("--cv-epochs", type=int, default=28)
    p.add_argument("--cv-steps", type=int, default=120)
    p.add_argument("--final-epochs", type=int, default=40)
    p.add_argument("--final-steps", type=int, default=200)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--backend", choices=("auto", "tensorflow", "numpy"), default="auto")
    p.add_argument(
        "--reference",
        type=Path,
        default=ROOT
        / "artifacts"
        / "submission_feature_lag_mlp_multiscale_summaries_seed42_sj_only.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = resolve_backend(args.backend)
    train, test, template = load_competition_data(args.data_dir)

    strategies = [
        Strategy("multiscale_seed42", (42,), "single"),
        Strategy("multiscale_seed42_nearest", (42,), "single", rounding="nearest"),
        Strategy("multiscale_median_17_42_73", (17, 42, 73), "median"),
        Strategy("multiscale_both_cities_seed42", (42,), "single", iq_from_baseline=False),
        Strategy(
            "multiscale_both_nearest_seed42",
            (42,),
            "single",
            iq_from_baseline=False,
            rounding="nearest",
        ),
        Strategy("multiscale_seed42_blend25_raw", (42,), "single", sj_baseline_weight=0.25),
    ]

    cv_schedule = TrainingSchedule(
        epochs=args.cv_epochs,
        steps_per_epoch=args.cv_steps,
        batch_size=16,
        lr_mode="max",
    )
    fold_train, holdout_frame, y_hold = sj_holdout_split(train, args.holdout_weeks)
    baseline_raw = predict_sj_raw_lag(
        fold_train, holdout_frame, 42, cv_schedule, backend, args.verbose
    )

    rows: list[dict[str, object]] = []
    for strategy in strategies:
        print(f"CV {strategy.name} ...", flush=True)
        raw = predict_sj_multiscale(
            fold_train,
            holdout_frame,
            strategy,
            cv_schedule,
            backend,
            args.verbose,
            baseline_raw if strategy.sj_baseline_weight > 0 else None,
        )
        score = mae(y_hold, raw, strategy.rounding)
        rows.append({"strategy": strategy.name, "holdout_mae": score, **strategy.__dict__})
        print(f"  sj_holdout_mae={score:.4f}", flush=True)

    ranking = sorted(rows, key=lambda r: r["holdout_mae"])  # type: ignore[arg-type]
    best_row = ranking[0]
    best = Strategy(
        str(best_row["strategy"]),
        tuple(best_row["seeds"]),  # type: ignore[arg-type]
        str(best_row["aggregate"]),
        float(best_row["sj_baseline_weight"]),
        bool(best_row["iq_from_baseline"]),
        str(best_row["rounding"]),
    )
    print(f"\nSelected: {best.name} (holdout_mae={best_row['holdout_mae']:.4f})\n")

    final_schedule = TrainingSchedule(
        epochs=args.final_epochs,
        steps_per_epoch=args.final_steps,
        batch_size=16,
        lr_mode="max",
    )
    full_train = train
    baseline_full_raw = None
    if best.sj_baseline_weight > 0:
        baseline_full_raw = predict_sj_raw_lag(
            full_train, test, 42, final_schedule, backend, args.verbose
        )
    sj_raw = predict_sj_multiscale(
        full_train,
        test,
        best,
        final_schedule,
        backend,
        args.verbose,
        baseline_full_raw,
    )
    sj_cases = to_cases(sj_raw, best.rounding)

    baseline_path = args.output_dir / "submission_feature_lag_mlp_tuned_seed42.csv"
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Missing {baseline_path}; run: python scripts/reproduce_multiscale_best.py"
        )
    baseline_sub = pd.read_csv(baseline_path)

    city_frames = [prediction_frame(test, "sj", sj_cases)]
    if best.iq_from_baseline:
        iq_cases = baseline_sub.loc[baseline_sub["city"].eq("iq"), TARGET_COLUMN].to_numpy(
            int
        )
        city_frames.append(prediction_frame(test, "iq", iq_cases))
    else:
        for city in ("iq",):
            train_x, train_y = build_representation_training_matrix(
                full_train, city, MULTISCALE_SUMMARIES
            )
            test_x = build_representation_test_matrix(
                full_train, test, city, MULTISCALE_SUMMARIES
            )
            model, _ = fit_city(
                train_x,
                train_y,
                city,
                TUNED_CONFIGS[city],
                final_schedule,
                best.seeds[0],
                args.verbose,
                backend=backend,
            )
            raw_iq = predict_raw(model, test_x, backend)
            city_frames.append(
                prediction_frame(test, city, to_cases(raw_iq, best.rounding))
            )

    submission = assemble_submission(template, city_frames)
    out = args.output_dir / "submission_feature_lag_mlp_best_candidate.csv"
    submission.to_csv(out, index=False)

    ref_mae = None
    if args.reference.is_file():
        ref = pd.read_csv(args.reference)
        merged = submission.merge(
            ref, on=["city", "year", "weekofyear"], suffixes=("_new", "_ref")
        )
        ref_mae = float(
            np.mean(
                np.abs(
                    merged[f"{TARGET_COLUMN}_ref"]
                    - merged[f"{TARGET_COLUMN}_new"]
                )
            )
        )

    meta = {
        "backend": backend,
        "selected_strategy": best.__dict__,
        "holdout_ranking": ranking,
        "output": str(out),
        "reference_submission": str(args.reference),
        "mean_abs_diff_vs_reference": ref_mae,
        "sj_stats": {
            "min": int(sj_cases.min()),
            "median": float(np.median(sj_cases)),
            "mean": float(sj_cases.mean()),
            "max": int(sj_cases.max()),
        },
    }
    meta_path = args.output_dir / "submission_feature_lag_mlp_best_candidate.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out}")
    print(submission.groupby("city")[TARGET_COLUMN].agg(["min", "median", "mean", "max"]))


if __name__ == "__main__":
    main()
