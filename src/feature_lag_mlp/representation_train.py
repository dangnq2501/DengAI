"""Train the confirmed multiscale representation and export safe candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd

from .config import CITIES, TARGET_COLUMN, TUNED_CONFIGS, TrainingSchedule
from .data import load_competition_data
from .representations import (
    MULTISCALE_SUMMARIES,
    REPRESENTATIONS,
    build_representation_test_matrix,
    build_representation_training_matrix,
)
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
        "--representation", choices=REPRESENTATIONS, default=MULTISCALE_SUMMARIES
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--baseline-submission",
        type=Path,
        default=project_dir / "artifacts" / "submission_feature_lag_mlp_tuned_seed42.csv",
        help="18.8 submission used to build city-isolation candidates.",
    )
    return parser.parse_args()


def _city_hybrid(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    hybrid = baseline.copy()
    city_rows = hybrid["city"].eq(city)
    candidate_city = candidate.loc[candidate["city"].eq(city), TARGET_COLUMN]
    hybrid.loc[city_rows, TARGET_COLUMN] = candidate_city.to_numpy()
    hybrid[TARGET_COLUMN] = hybrid[TARGET_COLUMN].astype(int)
    return hybrid


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, template = load_competition_data(args.data_dir)
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )
    stem = f"feature_lag_mlp_{args.representation}_seed{args.seed}"
    metadata: dict[str, object] = {
        "method": "causal multiscale climate representation",
        "representation": args.representation,
        "seed": args.seed,
        "schedule": schedule.__dict__,
        "cities": {},
    }

    city_predictions = []
    for city in CITIES:
        train_x, train_y = build_representation_training_matrix(
            train, city, args.representation
        )
        test_x = build_representation_test_matrix(
            train, test, city, args.representation
        )
        set_tensorflow_seed(args.seed, clear_session=True)
        result = fit_arrays(
            train_x,
            train_y,
            city,
            TUNED_CONFIGS[city],
            schedule,
            args.verbose,
        )
        raw, cases = predict_cases(result.model, test_x)
        city_predictions.append(prediction_frame(test, city, cases))
        result.model.save(args.output_dir / f"{stem}_{city}.keras")
        result.history.to_csv(
            args.output_dir / f"{stem}_{city}_history.csv", index=False
        )
        metadata["cities"][city] = {  # type: ignore[index]
            "input_dimension": int(train_x.shape[1]),
            "training_rows": int(train_x.shape[0]),
            "final_training_mae": float(result.history["mae"].iloc[-1]),
            "raw_prediction_mean": float(raw.mean()),
            "raw_prediction_max": float(raw.max()),
            "submission_mean": float(cases.mean()),
            "submission_max": int(cases.max()),
        }

    candidate = assemble_submission(template, city_predictions)
    candidate_path = args.output_dir / f"submission_{stem}.csv"
    candidate.to_csv(candidate_path, index=False)

    written = [candidate_path]
    if args.baseline_submission.exists():
        baseline = pd.read_csv(args.baseline_submission)
        if not baseline[template.columns].equals(template):
            # Compare only keys because the template target is intentionally zero.
            key_columns = ["city", "year", "weekofyear"]
            if not baseline[key_columns].equals(template[key_columns]):
                raise ValueError("Baseline submission keys do not match template")
        for city in CITIES:
            hybrid = _city_hybrid(baseline, candidate, city)
            hybrid_path = args.output_dir / f"submission_{stem}_{city}_only.csv"
            hybrid.to_csv(hybrid_path, index=False)
            written.append(hybrid_path)

    metadata_path = args.output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Wrote candidate submissions:")
    for path in written:
        print(f"- {path}")
    print(candidate.groupby("city")[TARGET_COLUMN].agg(["min", "median", "mean", "max"]))


if __name__ == "__main__":
    main()
