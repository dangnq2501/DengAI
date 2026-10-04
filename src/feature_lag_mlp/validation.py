"""Time-aware validation and error analysis for architecture candidates."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, SHARED_SJ_NORMALIZATION, TARGET_COLUMN, TrainingSchedule
from .representations import (
    RAW_LAGS,
    build_representation_test_matrix,
    build_representation_training_matrix,
)
from .search_space import Candidate
from .training import fit_arrays, predict_cases, set_tensorflow_seed


def temporal_folds(
    train: pd.DataFrame,
    city: str,
    fold_count: int,
    fold_weeks: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Build expanding training prefixes followed by untouched validation years."""
    city_frames = {
        name: train.loc[train["city"].eq(name)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
        for name in ("sj", "iq")
    }
    target = city_frames[city]
    required = (fold_count + 1) * fold_weeks
    if len(target) < required:
        raise ValueError(f"Not enough {city} rows for {fold_count} folds")

    folds = []
    first_start = len(target) - fold_count * fold_weeks
    for fold_index in range(fold_count):
        validation_start = first_start + fold_index * fold_weeks
        validation_end = validation_start + fold_weeks
        fraction = validation_start / len(target)
        prefixes = {}
        for name, frame in city_frames.items():
            end = (
                validation_start
                if name == city
                else max(53, int(len(frame) * fraction))
            )
            prefixes[name] = frame.iloc[:end]
        # Official files place all SJ rows before all IQ rows. This order also
        # preserves the original global interpolation behavior at the boundary.
        fold_train = pd.concat([prefixes["sj"], prefixes["iq"]], ignore_index=True)
        validation = target.iloc[validation_start:validation_end].copy()
        folds.append((fold_index + 1, fold_train, validation))
    return folds


def prepared_folds(
    train: pd.DataFrame,
    city: str,
    fold_count: int,
    fold_weeks: int,
    representation: str = RAW_LAGS,
) -> list[dict[str, Any]]:
    """Materialize matrices once so every candidate sees identical fold data."""
    prepared = []
    city_frames = {
        name: train.loc[train["city"].eq(name)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
        for name in ("sj", "iq")
    }
    for fold, fold_train, validation in temporal_folds(
        train, city, fold_count, fold_weeks
    ):
        train_x, train_y = build_representation_training_matrix(
            fold_train,
            city,
            representation,
            SHARED_SJ_NORMALIZATION,
        )
        validation_features = validation
        if city == "iq":
            # Test preprocessing interpolates the complete SJ-then-IQ table
            # before selecting a city. Add an equivalent SJ block so validation
            # measures that exact pipeline instead of a cleaner approximation.
            iq_frame = city_frames["iq"]
            sj_frame = city_frames["sj"]
            iq_start = int(
                iq_frame.index[
                    iq_frame[DATE_COLUMN].eq(validation[DATE_COLUMN].iloc[0])
                ][0]
            )
            iq_end = iq_start + len(validation)
            sj_start = int(len(sj_frame) * iq_start / len(iq_frame))
            sj_end = max(sj_start + 1, int(len(sj_frame) * iq_end / len(iq_frame)))
            sj_block = sj_frame.iloc[sj_start : min(sj_end, len(sj_frame))]
            validation_features = pd.concat([sj_block, validation], ignore_index=True)

        validation_x = build_representation_test_matrix(
            fold_train,
            validation_features,
            city,
            representation,
            SHARED_SJ_NORMALIZATION,
        )
        prepared.append(
            {
                "fold": fold,
                "train_x": train_x,
                "train_y": train_y,
                "validation_x": validation_x,
                "validation_y": validation[TARGET_COLUMN].to_numpy(float),
                "validation_start": str(validation[DATE_COLUMN].iloc[0].date()),
                "validation_end": str(validation[DATE_COLUMN].iloc[-1].date()),
            }
        )
    return prepared


def fit_candidate(
    city: str,
    candidate: Candidate,
    fold: dict[str, Any],
    seed: int,
    schedule: TrainingSchedule,
    verbose: int,
) -> dict[str, object]:
    """Fit one candidate/fold/seed and return both normal and outbreak MAE."""
    set_tensorflow_seed(seed, clear_session=True)
    result = fit_arrays(
        fold["train_x"],
        fold["train_y"],
        city,
        candidate.config,
        replace(schedule, feature_noise_std=candidate.feature_noise_std),
        verbose,
    )
    _, prediction = predict_cases(result.model, fold["validation_x"])
    target = fold["validation_y"]
    threshold = float(np.quantile(fold["train_y"], 0.90))
    outbreak = target >= threshold
    mae = float(np.mean(np.abs(prediction - target)))
    outbreak_mae = (
        float(np.mean(np.abs(prediction[outbreak] - target[outbreak])))
        if outbreak.any()
        else np.nan
    )
    return {
        "city": city,
        "candidate": candidate.name,
        **asdict(candidate),
        "fold": fold["fold"],
        "seed": seed,
        "validation_start": fold["validation_start"],
        "validation_end": fold["validation_end"],
        "mae": mae,
        "outbreak_mae": outbreak_mae,
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_max": int(np.max(prediction)),
        "final_training_mae": float(result.history["mae"].iloc[-1]),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by mean MAE plus a small across-run stability penalty."""
    group_columns = [
        "city",
        "candidate",
        "hidden_1",
        "hidden_2",
        "dropout_1",
        "dropout_2",
        "learning_rate",
        "linear_skip",
        "l2_strength",
        "feature_noise_std",
    ]
    summary = (
        results.groupby(group_columns, as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            worst_mae=("mae", "max"),
            mean_outbreak_mae=("outbreak_mae", "mean"),
            runs=("mae", "size"),
        )
        .fillna({"std_mae": 0.0})
    )
    summary["selection_score"] = summary["mean_mae"] + 0.10 * summary["std_mae"]
    return summary.sort_values(
        ["selection_score", "worst_mae", "mean_outbreak_mae"]
    ).reset_index(drop=True)
