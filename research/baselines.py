"""Simple baselines that complement the feature-specific lag MLP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_lag_mlp.config import DATE_COLUMN, TARGET_COLUMN


def seasonal_median_predict(
    train_weeks: np.ndarray,
    train_cases: np.ndarray,
    predict_weeks: np.ndarray,
    *,
    window: int = 4,
) -> np.ndarray:
    """Circular week-of-year median using training labels only.

    Matches the compact seasonal baseline in ``leaderboard_notebook/build_notebook.py``.
    """
    train_weeks = np.asarray(train_weeks, dtype=float)
    train_cases = np.asarray(train_cases, dtype=float)
    predict_weeks = np.asarray(predict_weeks, dtype=float)
    fallback = float(np.median(train_cases))
    predictions: list[float] = []
    for week in predict_weeks:
        delta = np.abs((train_weeks - week + 26) % 52 - 26)
        local = train_cases[delta <= window]
        predictions.append(float(np.median(local)) if len(local) else fallback)
    return np.array(predictions, dtype=float)


def seasonal_median_for_city(
    train: pd.DataFrame,
    predict_frame: pd.DataFrame,
    city: str,
) -> np.ndarray:
    """Predict ``total_cases`` for one city's rows in ``predict_frame``."""
    city_train = train.loc[train["city"].eq(city)].sort_values(DATE_COLUMN)
    city_predict = predict_frame.loc[predict_frame["city"].eq(city)].sort_values(
        DATE_COLUMN
    )
    return seasonal_median_predict(
        city_train["weekofyear"].to_numpy(),
        city_train[TARGET_COLUMN].to_numpy(),
        city_predict["weekofyear"].to_numpy(),
    )
