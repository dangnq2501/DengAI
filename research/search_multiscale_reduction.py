"""Temporal CV for PCA variants on multiscale features."""

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

from feature_lag_mlp.config import CITIES, TUNED_CONFIGS, TrainingSchedule
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.representations import MULTISCALE_SUMMARIES
from feature_lag_mlp.validation import prepared_folds

from research.multiscale_reduction import ReductionSpec, default_search_grid, reduce_multiscale
from research.training_backend import fit_city, predict_raw, resolve_backend


def mae(y_true: np.ndarray, raw_pred: np.ndarray) -> float:
    pred = np.maximum(raw_pred, 0).astype(int)
    return float(np.mean(np.abs(y_true - pred)))


def evaluate_city(
    train: pd.DataFrame,
    city: str,
    spec: ReductionSpec,
    fold_count: int,
    fold_weeks: int,
    seed: int,
    schedule: TrainingSchedule,
    backend: str,
    verbose: int,
) -> dict[str, float]:
    folds = prepared_folds(
        train, city, fold_count, fold_weeks, representation=MULTISCALE_SUMMARIES
    )
    fold_maes: list[float] = []
    for fold in folds:
        train_x, apply_x, _ = reduce_multiscale(
            fold["train_x"], fold["validation_x"], spec
        )
        model, _ = fit_city(
            train_x,
            fold["train_y"],
            city,
            TUNED_CONFIGS[city],
            schedule,
            seed,
            verbose,
            backend=backend,
        )
        raw = predict_raw(model, apply_x, backend)
        fold_maes.append(
            mae(np.asarray(fold["validation_y"], dtype=float), raw)
        )
    return {
        "mean_mae": float(np.mean(fold_maes)),
        "worst_mae": float(np.max(fold_maes)),
        "std_mae": float(np.std(fold_maes)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "research" / "artifacts",
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--fold-weeks", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "tensorflow", "numpy"),
        default="auto",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Search only baseline + a small PCA grid.",
    )
    return parser.parse_args()


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
    train, _, _ = load_competition_data(args.data_dir)
    specs = default_search_grid()
    if args.quick:
        keep = {
            "baseline_180",
            "global_pca_96",
            "global_pca_120",
            "block_pca_4",
            "block_pca_5",
        }
        specs = [spec for spec in specs if spec.name in keep]

    rows: list[dict[str, object]] = []
    for city in CITIES:
        for spec in specs:
            print(f"[{city}] {spec.name} ...", flush=True)
            metrics = evaluate_city(
                train,
                city,
                spec,
                args.folds,
                args.fold_weeks,
                args.seed,
                schedule,
                backend,
                args.verbose,
            )
            score = metrics["mean_mae"] + 0.1 * metrics["std_mae"]
            rows.append(
                {
                    "city": city,
                    "spec": spec.name,
                    "method": spec.method,
                    "n_components": spec.n_components,
                    "block_components": spec.block_components,
                    "selection_score": score,
                    **metrics,
                }
            )
            print(f"  mean_mae={metrics['mean_mae']:.3f}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        ["city", "selection_score", "worst_mae"]
    )
    results.to_csv(args.output_dir / "multiscale_reduction_cv.csv", index=False)

    selected: dict[str, object] = {"backend": backend, "seed": args.seed, "cities": {}}
    for city in CITIES:
        city_rows = results.loc[results["city"].eq(city)]
        best = city_rows.iloc[0]
        selected["cities"][city] = {
            "spec": best["spec"],
            "method": best["method"],
            "n_components": best["n_components"] if pd.notna(best["n_components"]) else None,
            "block_components": best["block_components"]
            if pd.notna(best["block_components"])
            else None,
            "mean_mae": float(best["mean_mae"]),
            "selection_score": float(best["selection_score"]),
        }
        print(f"\n{city.upper()} winner: {best['spec']} (mean_mae={best['mean_mae']:.3f})")

    path = args.output_dir / "multiscale_reduction_selection.json"
    path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
