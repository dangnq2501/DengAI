"""Tune only the SJ MLP wrapped around the confirmed multiscale features."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd

from .config import TARGET_COLUMN, TrainingSchedule
from .data import load_competition_data
from .representations import (
    MULTISCALE_SUMMARIES,
    build_representation_test_matrix,
    build_representation_training_matrix,
)
from .search_space import (
    Candidate,
    candidate_from_summary,
    multiscale_sj_architecture_grid,
)
from .training import fit_arrays, predict_cases, set_tensorflow_seed
from .tune import parse_int_list, run_stage
from .validation import prepared_folds, summarize_results


CONTROL_NAME = "summary_control"


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--search-folds", type=int, default=2)
    parser.add_argument("--confirm-folds", type=int, default=3)
    parser.add_argument("--fold-weeks", type=int, default=52)
    parser.add_argument("--top-candidates", type=int, default=2)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument(
        "--confirmation-seeds", type=parse_int_list, default=(17, 42, 73)
    )
    parser.add_argument("--final-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--baseline-submission",
        type=Path,
        default=(
            project_dir / "artifacts" / "submission_feature_lag_mlp_tuned_seed42.csv"
        ),
        help="Known 18.8 submission whose IQ predictions remain untouched.",
    )
    parser.add_argument("--skip-final-training", action="store_true")
    return parser.parse_args()


def _save_stage(
    output_dir: Path,
    stage: str,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    summary = summarize_results(runs)
    stem = f"feature_lag_mlp_multiscale_sj_architecture_{stage}"
    runs.to_csv(output_dir / f"{stem}_runs.csv", index=False)
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    return summary


def _finalists(search_summary: pd.DataFrame, count: int) -> list[Candidate]:
    selected = [
        candidate_from_summary(row)
        for _, row in search_summary.head(count).iterrows()
    ]
    control = multiscale_sj_architecture_grid()[0]
    if all(candidate.name != CONTROL_NAME for candidate in selected):
        selected.append(control)
    return selected


def _write_sj_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: Candidate,
    args: argparse.Namespace,
    schedule: TrainingSchedule,
) -> tuple[Path, dict[str, object]]:
    if not args.baseline_submission.exists():
        raise FileNotFoundError(f"Missing baseline: {args.baseline_submission}")
    baseline = pd.read_csv(args.baseline_submission)
    keys = ["city", "year", "weekofyear"]
    if not baseline[keys].equals(test[keys]):
        raise ValueError("Baseline submission keys do not match test rows")

    train_x, train_y = build_representation_training_matrix(
        train, "sj", MULTISCALE_SUMMARIES
    )
    test_x = build_representation_test_matrix(
        train, test, "sj", MULTISCALE_SUMMARIES
    )
    set_tensorflow_seed(args.final_seed, clear_session=True)
    result = fit_arrays(
        train_x,
        train_y,
        "sj",
        candidate.config,
        schedule,
        args.verbose,
    )
    raw, cases = predict_cases(result.model, test_x)
    sj_rows = baseline["city"].eq("sj")
    if int(sj_rows.sum()) != len(cases):
        raise ValueError("SJ prediction count does not match baseline")
    submission = baseline.copy()
    submission.loc[sj_rows, TARGET_COLUMN] = cases
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)

    stem = (
        f"feature_lag_mlp_multiscale_architecture_seed{args.final_seed}_sj_only"
    )
    submission_path = args.output_dir / f"submission_{stem}.csv"
    model_path = args.output_dir / f"{stem}.keras"
    history_path = args.output_dir / f"{stem}_history.csv"
    submission.to_csv(submission_path, index=False)
    result.model.save(model_path)
    result.history.to_csv(history_path, index=False)
    details = {
        "input_dimension": int(train_x.shape[1]),
        "training_rows": int(train_x.shape[0]),
        "final_training_mae": float(result.history["mae"].iloc[-1]),
        "raw_prediction_mean": float(raw.mean()),
        "raw_prediction_max": float(raw.max()),
        "submission_mean": float(cases.mean()),
        "submission_max": int(cases.max()),
        "submission": str(submission_path),
    }
    return submission_path, details


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, _ = load_competition_data(args.data_dir)
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )

    search_folds = prepared_folds(
        train,
        "sj",
        args.search_folds,
        args.fold_weeks,
        representation=MULTISCALE_SUMMARIES,
    )
    search_runs = run_stage(
        "sj",
        multiscale_sj_architecture_grid(),
        search_folds,
        (args.search_seed,),
        schedule,
        args.verbose,
        "multiscale-search",
    )
    search_summary = _save_stage(args.output_dir, "search", search_runs)
    print(f"\nSJ multiscale search ranking:\n{search_summary}\n")

    finalists = _finalists(search_summary, args.top_candidates)
    confirmation_folds = prepared_folds(
        train,
        "sj",
        args.confirm_folds,
        args.fold_weeks,
        representation=MULTISCALE_SUMMARIES,
    )
    confirmation_runs = run_stage(
        "sj",
        finalists,
        confirmation_folds,
        tuple(args.confirmation_seeds),
        schedule,
        args.verbose,
        "multiscale-confirm",
    )
    confirmation_summary = _save_stage(
        args.output_dir, "confirmation", confirmation_runs
    )
    winner = candidate_from_summary(confirmation_summary.iloc[0])
    control_row = confirmation_summary.loc[
        confirmation_summary["candidate"].eq(CONTROL_NAME)
    ].iloc[0]
    winner_row = confirmation_summary.iloc[0]
    metadata: dict[str, object] = {
        "representation": MULTISCALE_SUMMARIES,
        "leaderboard_control_mae": 16.6,
        "search_folds": args.search_folds,
        "confirm_folds": args.confirm_folds,
        "fold_weeks": args.fold_weeks,
        "search_seed": args.search_seed,
        "confirmation_seeds": list(args.confirmation_seeds),
        "selected": asdict(winner),
        "selected_mean_mae": float(winner_row["mean_mae"]),
        "control_mean_mae": float(control_row["mean_mae"]),
        "local_mae_improvement": float(
            control_row["mean_mae"] - winner_row["mean_mae"]
        ),
    }
    print(f"\nSJ multiscale confirmation ranking:\n{confirmation_summary}\n")

    if not args.skip_final_training:
        submission_path, final_details = _write_sj_submission(
            train, test, winner, args, schedule
        )
        metadata["final_training"] = final_details
        print(f"Wrote {submission_path}")

    metadata_path = (
        args.output_dir / "feature_lag_mlp_multiscale_sj_architecture.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
