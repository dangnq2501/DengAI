"""Forecasting-safe validation: expanding annual folds and MAE summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import CITIES, DATE_COLUMN, PREDICTION_LAG, TARGET_COLUMN, TEST_WEEKS, MLPConfig, TrainingSchedule
from .data import city_frame, interpolate_climate
from .features import MULTISCALE, normalization_stats, prediction_matrix, training_matrix, tree_features
from .models import SeluMLP, to_cases, tree_model


@dataclass
class Fold:
    """Train on every week before ``valid``; validate on the next 52 weeks."""

    city: str
    number: int
    train: pd.DataFrame
    valid: pd.DataFrame
    sj_reference: pd.DataFrame  # SJ weeks before the validation year (scaling stats)

    @property
    def label(self) -> str:
        return f"{self.valid[DATE_COLUMN].iloc[0]:%Y-%m} → {self.valid[DATE_COLUMN].iloc[-1]:%Y-%m}"


def interpolated_cities(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {city: interpolate_climate(city_frame(train, city)) for city in CITIES}


def temporal_folds(cities: dict[str, pd.DataFrame], city: str, n_folds: int = 3, fold_weeks: int = 52) -> list[Fold]:
    """The last ``n_folds`` years of a city, each predicted from all earlier weeks."""
    frame = cities[city]
    first = len(frame) - n_folds * fold_weeks
    folds = []
    for number in range(n_folds):
        start = first + number * fold_weeks
        valid = frame.iloc[start : start + fold_weeks]
        sj = cities["sj"]
        sj_reference = sj.loc[sj[DATE_COLUMN] < valid[DATE_COLUMN].iloc[0]]
        folds.append(Fold(city, number + 1, frame.iloc[:start], valid, sj_reference))
    return folds


# ---------------------------------------------------------------------------
# Fold predictors: each returns integer predictions for ``fold.valid``
# ---------------------------------------------------------------------------


def predict_mlp(
    fold: Fold,
    config: MLPConfig,
    representation: str = MULTISCALE,
    seed: int = 42,
    lag: int = PREDICTION_LAG,
    schedule: TrainingSchedule | None = None,
) -> np.ndarray:
    stats = normalization_stats(fold.sj_reference)
    x, y = training_matrix(fold.train, fold.city, stats, representation)
    valid_x = prediction_matrix(fold.train, fold.valid, fold.city, stats, representation, lag)
    model = SeluMLP(fold.city, config, schedule or TrainingSchedule(), seed).fit(x, y)
    return to_cases(model.predict(valid_x))


def predict_tree(fold: Fold, raw_train: pd.DataFrame, n_estimators: int = 300, **params) -> np.ndarray:
    """Trees use their own causal features with NaNs left for median imputation."""
    frame = city_frame(raw_train, fold.city)
    features = tree_features(frame)
    n_train, n_valid = len(fold.train), len(fold.valid)
    model = tree_model(fold.city, n_estimators=n_estimators, **params)
    model.fit(features.iloc[:n_train], frame[TARGET_COLUMN].iloc[:n_train])
    return to_cases(model.predict(features.iloc[n_train : n_train + n_valid]), rounding="round")


# ---------------------------------------------------------------------------
# Running and scoring
# ---------------------------------------------------------------------------


def cross_validate(folds: list[Fold], predict: Callable[[Fold], np.ndarray], **labels) -> pd.DataFrame:
    """Long table of per-week validation predictions for one model."""
    rows = []
    for fold in folds:
        prediction = predict(fold)
        rows.append(
            pd.DataFrame(
                {
                    **labels,
                    "city": fold.city,
                    "fold": fold.number,
                    DATE_COLUMN: fold.valid[DATE_COLUMN].to_numpy(),
                    "weekofyear": fold.valid["weekofyear"].to_numpy(),
                    "actual": fold.valid[TARGET_COLUMN].to_numpy(),
                    "prediction": prediction,
                    # Outbreak weeks: at or above the 90th percentile of the
                    # cases seen in training, so the threshold is not peeked.
                    "outbreak": fold.valid[TARGET_COLUMN].to_numpy()
                    >= np.quantile(fold.train[TARGET_COLUMN], 0.90),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    result["abs_error"] = (result["prediction"] - result["actual"]).abs()
    return result


def fold_scores(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """MAE and outbreak MAE per fold (and per seed, if present)."""
    keys = by + ["city", "fold"] + (["seed"] if "seed" in predictions else [])
    return (
        predictions.groupby(keys)
        .apply(
            lambda g: pd.Series(
                {
                    "mae": g["abs_error"].mean(),
                    "outbreak_mae": g.loc[g["outbreak"], "abs_error"].mean(),
                    "prediction_max": g["prediction"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def summarize(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean ± std over folds (and seeds) per model and city, plus weighted MAE."""
    scores = fold_scores(predictions, by)
    city_summary = (
        scores.groupby(by + ["city"])
        .agg(
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            worst_fold_mae=("mae", "max"),
            outbreak_mae=("outbreak_mae", "mean"),
        )
        .reset_index()
    )
    wide = city_summary.pivot_table(index=by, columns="city", values="mae")
    total = sum(TEST_WEEKS.values())
    wide["weighted"] = sum(wide[c] * TEST_WEEKS[c] / total for c in CITIES if c in wide)
    return city_summary, wide.reset_index().rename_axis(columns=None)
