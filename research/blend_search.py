"""Search MLP vs seasonal blend weights on temporal validation folds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from feature_lag_mlp.config import CITIES, TARGET_COLUMN, TrainingSchedule
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.validation import prepared_folds

from research.baselines import seasonal_median_predict
from research.candidates import tuned_candidate
from research.rounding import blend_raw, to_submission_cases
from research.training_backend import Backend, fit_city, predict_raw, resolve_backend


def parse_weight_grid(value: str) -> tuple[float, ...]:
    weights = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not weights:
        raise argparse.ArgumentTypeError("at least one weight is required")
    for weight in weights:
        if not 0.0 <= weight <= 1.0:
            raise argparse.ArgumentTypeError("weights must lie in [0, 1]")
    return weights


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def fold_predictions(
    train: pd.DataFrame,
    city: str,
    fold: dict[str, object],
    seed: int,
    schedule: TrainingSchedule,
    verbose: int,
    backend: Backend,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MLP raw, seasonal raw, and validation targets for one fold."""
    resolved = resolve_backend(backend)
    candidate = tuned_candidate(city)
    model, _ = fit_city(
        fold["train_x"],
        fold["train_y"],
        city,
        candidate.config,
        schedule,
        seed,
        verbose,
        backend=backend,
    )
    raw_mlp = predict_raw(model, fold["validation_x"], resolved)
    target = np.asarray(fold["validation_y"], dtype=float)

    city_train = (
        train.loc[train["city"].eq(city)]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )
    validation_mask = city_train["week_start_date"].eq(
        pd.Timestamp(fold["validation_start"])
    ).to_numpy()
    if not validation_mask.any():
        raise ValueError(f"Could not locate validation start for {city}")
    start_index = int(np.flatnonzero(validation_mask)[0])
    prefix = city_train.iloc[:start_index]
    predict_weeks = city_train.iloc[start_index : start_index + len(target)][
        "weekofyear"
    ].to_numpy()
    raw_seasonal = seasonal_median_predict(
        prefix["weekofyear"].to_numpy(),
        prefix[TARGET_COLUMN].to_numpy(),
        predict_weeks,
    )
    return raw_mlp, raw_seasonal, target


def search_city_weights(
    train: pd.DataFrame,
    city: str,
    folds: list[dict[str, object]],
    seeds: tuple[int, ...],
    weights: tuple[float, ...],
    rounding_modes: tuple[str, ...],
    schedule: TrainingSchedule,
    verbose: int,
    backend: Backend,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for fold in folds:
            raw_mlp, raw_seasonal, target = fold_predictions(
                train, city, fold, seed, schedule, verbose, backend
            )
            for weight in weights:
                blended = blend_raw(raw_mlp, raw_seasonal, weight)
                for mode in rounding_modes:
                    prediction = to_submission_cases(blended, mode=mode)
                    rows.append(
                        {
                            "city": city,
                            "seed": seed,
                            "fold": fold["fold"],
                            "mlp_weight": weight,
                            "rounding": mode,
                            "mae": mae(target, prediction),
                            "validation_start": fold["validation_start"],
                            "validation_end": fold["validation_end"],
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["city", "mlp_weight", "rounding"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            worst_mae=("mae", "max"),
            runs=("mae", "size"),
        )
        .fillna({"std_mae": 0.0})
    )
    summary["selection_score"] = summary["mean_mae"] + 0.10 * summary["std_mae"]
    return summary.sort_values(["selection_score", "worst_mae"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "research" / "artifacts",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-weeks", type=int, default=52)
    parser.add_argument("--seeds", type=parse_int_list, default=(42,))
    parser.add_argument(
        "--mlp-weights",
        type=parse_weight_grid,
        default=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        help="MLP weight grid; values below 0.5 are excluded so the lag MLP stays primary.",
    )
    parser.add_argument(
        "--rounding",
        default="nearest,floor,truncate",
        help="Comma-separated rounding modes to compare.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "tensorflow", "numpy"),
        default="auto",
        help="Use NumPy when TensorFlow is missing or broken (conda env bkk).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounding_modes = tuple(
        part.strip() for part in args.rounding.split(",") if part.strip()
    )
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )
    train, _, _ = load_competition_data(args.data_dir)

    resolved_backend = resolve_backend(args.backend)
    metadata: dict[str, object] = {
        "backend": resolved_backend,
        "folds": args.folds,
        "fold_weeks": args.fold_weeks,
        "seeds": list(args.seeds),
        "mlp_weights": list(args.mlp_weights),
        "rounding_modes": list(rounding_modes),
        "cities": {},
    }
    for city in CITIES:
        folds = prepared_folds(train, city, args.folds, args.fold_weeks)
        summary = search_city_weights(
            train,
            city,
            folds,
            args.seeds,
            args.mlp_weights,
            rounding_modes,
            schedule,
            args.verbose,
            args.backend,
        )
        summary_path = args.output_dir / f"blend_search_{city}.csv"
        summary.to_csv(summary_path, index=False)
        best = summary.iloc[0]
        metadata["cities"][city] = {
            "mlp_weight": float(best["mlp_weight"]),
            "rounding": str(best["rounding"]),
            "mean_mae": float(best["mean_mae"]),
            "selection_score": float(best["selection_score"]),
            "summary_path": str(summary_path),
        }
        print(f"\n{city.upper()} blend ranking (top 5):\n{summary.head()}\n")

    metadata_path = args.output_dir / "blend_search.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
