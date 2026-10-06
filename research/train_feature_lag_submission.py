"""Build an improved submission from the tuned feature-lag MLP.

Improvements over ``src/feature_lag_mlp/train.py --profile tuned``:

* multi-seed raw prediction ensemble (default five seeds);
* optional convex blend with the compact seasonal median baseline;
* competition-aligned nearest-integer rounding (leaderboard evaluation uses
  ``floor(x + 0.5)``, not plain truncation).
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

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from feature_lag_mlp.config import CITIES, TARGET_COLUMN, TrainingSchedule
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.training import assemble_submission, prediction_frame

from research.baselines import seasonal_median_for_city
from research.candidates import tuned_candidate
from research.rounding import blend_raw, to_submission_cases
from research.submission_training import train_full_city
from research.training_backend import resolve_backend


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "research" / "artifacts",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=(17, 42, 73, 101, 137),
    )
    parser.add_argument(
        "--sj-mlp-weight",
        type=float,
        default=None,
        help="MLP weight for San Juan; 1.0 is pure MLP. Defaults from blend_search.json if present.",
    )
    parser.add_argument(
        "--iq-mlp-weight",
        type=float,
        default=None,
        help="MLP weight for Iquitos; 1.0 is pure MLP.",
    )
    parser.add_argument(
        "--rounding",
        choices=("nearest", "floor", "truncate"),
        default=None,
        help="Defaults from blend_search.json when available, else nearest.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--blend-config",
        type=Path,
        default=PROJECT_DIR / "research" / "defaults" / "blend_search.json",
        help="Optional output from blend_search.py.",
    )
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


def _city_weight(
    city: str,
    args: argparse.Namespace,
    blend_config: dict[str, object] | None,
) -> float:
    explicit = args.sj_mlp_weight if city == "sj" else args.iq_mlp_weight
    if explicit is not None:
        return float(explicit)
    if blend_config and city in blend_config.get("cities", {}):
        return float(blend_config["cities"][city]["mlp_weight"])  # type: ignore[index]
    return 1.0


def _city_rounding(
    city: str,
    args: argparse.Namespace,
    blend_config: dict[str, object] | None,
) -> str:
    if args.rounding is not None:
        return args.rounding
    if blend_config and city in blend_config.get("cities", {}):
        return str(blend_config["cities"][city]["rounding"])  # type: ignore[index]
    return "nearest"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = TrainingSchedule(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_mode="max",
    )
    blend_config: dict[str, object] | None = None
    if args.blend_config.is_file():
        blend_config = json.loads(args.blend_config.read_text(encoding="utf-8"))

    train, test, template = load_competition_data(args.data_dir)

    backend = resolve_backend(args.backend)
    metadata: dict[str, object] = {
        "method": "tuned feature-lag MLP research submission",
        "backend": backend,
        "seeds": list(args.seeds),
        "cities": {},
    }

    city_predictions = []
    for city in CITIES:
        candidate = tuned_candidate(city)
        raw, runs, _ = train_full_city(
            train,
            test,
            city,
            candidate,
            args.seeds,
            schedule,
            args.verbose,
            backend=backend,
        )
        weight = _city_weight(city, args, blend_config)
        if weight < 1.0:
            seasonal = seasonal_median_for_city(train, test, city)
            raw = blend_raw(raw, seasonal, weight)
        rounding = _city_rounding(city, args, blend_config)
        cases = to_submission_cases(raw, mode=rounding)
        city_predictions.append(prediction_frame(test, city, cases))
        metadata["cities"][city] = {
            "mlp_weight": weight,
            "rounding": rounding,
            "candidate": candidate.name,
            "final_runs": runs,
            "ensemble_raw_mean": float(np.mean(raw)),
            "ensemble_raw_max": float(np.max(raw)),
            "submission_mean": float(np.mean(cases)),
            "submission_max": int(np.max(cases)),
        }

    submission = assemble_submission(template, city_predictions)
    submission_path = args.output_dir / "submission_feature_lag_mlp_research.csv"
    submission.to_csv(submission_path, index=False)
    metadata["submission_sha256"] = _sha256(submission_path)
    metadata_path = args.output_dir / "submission_feature_lag_mlp_research.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {submission_path}")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
