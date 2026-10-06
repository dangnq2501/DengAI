"""Train multiscale + PCA winner and export SJ-only hybrid submission."""

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

from feature_lag_mlp.config import CITIES, KEY_COLUMNS, TARGET_COLUMN, TUNED_CONFIGS, TrainingSchedule
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.representations import (
    MULTISCALE_SUMMARIES,
    build_representation_test_matrix,
    build_representation_training_matrix,
)
from feature_lag_mlp.training import assemble_submission, prediction_frame

from research.multiscale_reduction import ReductionSpec, reduce_multiscale
from research.training_backend import fit_city, predict_raw, resolve_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "research" / "artifacts" / "multiscale_reduction_selection.json",
    )
    parser.add_argument(
        "--baseline-submission",
        type=Path,
        default=ROOT / "artifacts" / "submission_feature_lag_mlp_tuned_seed42.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
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


def _spec_from_row(city_entry: dict[str, object]) -> ReductionSpec:
    method = str(city_entry["method"])
    if method == "none":
        return ReductionSpec("baseline_180", "none")
    if method == "global_pca":
        k = int(city_entry["n_components"])  # type: ignore[arg-type]
        return ReductionSpec(f"global_pca_{k}", "global_pca", n_components=k)
    k = int(city_entry["block_components"])  # type: ignore[arg-type]
    return ReductionSpec(f"block_pca_{k}", "block_pca", block_components=k)


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
    if not args.selection.is_file():
        raise FileNotFoundError(
            f"Missing {args.selection}; run research/search_multiscale_reduction.py first."
        )
    if not args.baseline_submission.is_file():
        raise FileNotFoundError(
            f"Missing baseline {args.baseline_submission}; run reproduce_multiscale_best.py first."
        )

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )
    backend = resolve_backend(args.backend)
    train, test, template = load_competition_data(args.data_dir)
    baseline = pd.read_csv(args.baseline_submission)

    city_frames = []
    reduction_states: dict[str, object] = {}
    for city in CITIES:
        spec = _spec_from_row(selection["cities"][city])  # type: ignore[index]
        train_x, train_y = build_representation_training_matrix(
            train, city, MULTISCALE_SUMMARIES
        )
        test_x = build_representation_test_matrix(
            train, test, city, MULTISCALE_SUMMARIES
        )
        train_red, test_red, state = reduce_multiscale(train_x, test_x, spec)
        if state is not None:
            reduction_states[city] = state
        model, _ = fit_city(
            train_red,
            train_y,
            city,
            TUNED_CONFIGS[city],
            schedule,
            args.seed,
            args.verbose,
            backend=backend,
        )
        raw = predict_raw(model, test_red, backend)
        cases = np.maximum(raw, 0).astype(int)
        city_frames.append(prediction_frame(test, city, cases))
        print(
            f"{city}: {spec.name} -> mean={cases.mean():.2f}, max={cases.max()}",
            flush=True,
        )

    candidate = assemble_submission(template, city_frames)
    stem = "feature_lag_mlp_multiscale_pca_seed42"
    candidate_path = args.output_dir / f"submission_{stem}.csv"
    candidate.to_csv(candidate_path, index=False)
    sj_only = _city_hybrid(baseline, candidate, "sj")
    sj_path = args.output_dir / f"submission_{stem}_sj_only.csv"
    sj_only.to_csv(sj_path, index=False)

    metadata = {
        "method": "multiscale + CV-selected PCA",
        "backend": backend,
        "selection": selection,
        "baseline_submission": str(args.baseline_submission),
        "candidate_submission": str(candidate_path),
        "best_submission_sj_only": str(sj_path),
        "reduction_states_saved": list(reduction_states.keys()),
    }
    meta_path = args.output_dir / f"{stem}.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sj_path}")


if __name__ == "__main__":
    main()
