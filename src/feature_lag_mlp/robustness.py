"""Screen conservative complexity and climate-noise augmentation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import CITIES, TrainingSchedule
from .data import load_competition_data
from .search_space import robustness_grid
from .tune import run_stage
from .validation import prepared_folds, summarize_results


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--cities", choices=("both", "sj", "iq"), default="both")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--fold-weeks", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, _, _ = load_competition_data(args.data_dir)
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )
    cities = CITIES if args.cities == "both" else (args.cities,)

    for city in cities:
        folds = prepared_folds(train, city, args.folds, args.fold_weeks)
        runs = run_stage(
            city,
            robustness_grid(city),
            folds,
            (args.seed,),
            schedule,
            args.verbose,
            "robustness",
        )
        summary = summarize_results(runs)
        runs_path = args.output_dir / f"feature_lag_mlp_robustness_{city}_runs.csv"
        summary_path = (
            args.output_dir / f"feature_lag_mlp_robustness_{city}_summary.csv"
        )
        runs.to_csv(runs_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"\n{city.upper()} robustness ranking:\n{summary}\n")
        print(f"Wrote {runs_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
