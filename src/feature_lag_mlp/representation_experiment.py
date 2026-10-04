"""Compare original raw lag windows with causal multiscale summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import CITIES, TUNED_CONFIGS, TrainingSchedule
from .data import load_competition_data
from .representations import REPRESENTATIONS
from .search_space import Candidate
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
    parser.add_argument("--seeds", type=int, nargs="+", default=(42,))
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=REPRESENTATIONS,
    )
    parser.add_argument(
        "--experiment-name", default="feature_lag_mlp_representation"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    return parser.parse_args()


def _candidate(city: str, representation: str) -> Candidate:
    config = TUNED_CONFIGS[city]
    return Candidate(
        name=representation,
        hidden_1=config.hidden_1,
        hidden_2=config.hidden_2,
        dropout_1=config.dropout_1,
        dropout_2=config.dropout_2,
        learning_rate=config.learning_rate,
    )


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
        representation_runs = []
        for representation in args.representations:
            folds = prepared_folds(
                train,
                city,
                args.folds,
                args.fold_weeks,
                representation=representation,
            )
            runs = run_stage(
                city,
                [_candidate(city, representation)],
                folds,
                tuple(args.seeds),
                schedule,
                args.verbose,
                "representation",
            )
            runs["representation"] = representation
            runs["input_dimension"] = folds[0]["train_x"].shape[1]
            representation_runs.append(runs)

        all_runs = pd.concat(representation_runs, ignore_index=True)
        summary = summarize_results(all_runs)
        dimensions = all_runs[["candidate", "input_dimension"]].drop_duplicates()
        summary = summary.merge(dimensions, on="candidate", how="left")
        runs_path = args.output_dir / f"{args.experiment_name}_{city}_runs.csv"
        summary_path = args.output_dir / f"{args.experiment_name}_{city}_summary.csv"
        all_runs.to_csv(runs_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"\n{city.upper()} representation ranking:\n{summary}\n")
        print(f"Wrote {runs_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
