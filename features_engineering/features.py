"""Past-only features that fill gaps in the confirmed tree matrix.

The confirmed matrix already has seasonal harmonics, raw climate, and lags at
1, 2, 4, 8, 12, and 16 weeks. This module keeps that matrix, drops calendar
year (it cannot be extrapolated across the 3- and 5-year test blocks), and
adds:

- lags 6 and 10, covering the San Juan temperature correlation peak
- an 8-week humidity by 10-week temperature interaction
- past-only seasonal anomalies of the main climate drivers
- missingness indicators for NDVI and station measurements
- the mean of earlier seasons at the same week of year, using labels only
  from weeks that have already occurred
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_tree_ensemble import (  # noqa: E402
    DATE_COLUMN,
    NDVI_COLUMNS,
    TARGET_COLUMN,
    WEATHER_MEMORY_COLUMNS,
    make_features,
)

EXTRA_LAGS = (6, 10)
ANOMALY_COLUMNS = (
    "station_avg_temp_c",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "reanalysis_relative_humidity_percent",
)
MISSING_COLUMNS = (
    *NDVI_COLUMNS,
    "station_avg_temp_c",
    "station_min_temp_c",
    "station_max_temp_c",
    "station_precip_mm",
    "station_diur_temp_rng_c",
)
VOLATILITY_COLUMNS = (
    "station_avg_temp_c",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
)


def _past_same_week_mean(values: np.ndarray, week: np.ndarray) -> np.ndarray:
    """Mean of earlier finite observations that share the same week of year."""
    past_mean = np.full(len(values), np.nan)
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for index, (value, week_of_year) in enumerate(zip(values, week)):
        key = int(week_of_year)
        count = counts.get(key, 0)
        if count:
            past_mean[index] = totals[key] / count
        if np.isfinite(value):
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = count + 1
    return past_mean


def _case_climatology(
    week: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Earlier same-week case means. Test rows use the full training history."""
    climatology = np.full(len(week), np.nan)
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for index, is_train in enumerate(train_mask):
        key = int(week[index])
        count = counts.get(key, 0)
        if count:
            climatology[index] = totals[key] / count
        if is_train and np.isfinite(target[index]):
            totals[key] = totals.get(key, 0.0) + float(target[index])
            counts[key] = count + 1
    full_history = {key: totals[key] / counts[key] for key in counts}
    fallback = float(np.nanmean(target[train_mask]))
    for index, is_train in enumerate(train_mask):
        if not is_train:
            climatology[index] = full_history.get(int(week[index]), fallback)
    return climatology


def _shift_lag(values: pd.Series, lag: int) -> np.ndarray:
    return values.astype(float).shift(lag).to_numpy()


def build_features(combined: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Build the improved matrix. `combined` must be sorted by week."""
    if not combined[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Feature rows must be sorted chronologically")
    if len(combined) != len(train_mask):
        raise ValueError("Train mask length does not match the city frame")

    base = make_features(combined.drop(columns=["_is_train"], errors="ignore"))
    base = base.drop(columns=["year"], errors="ignore")
    extra: dict[str, np.ndarray] = {}
    week = combined["weekofyear"].to_numpy()

    for column in MISSING_COLUMNS:
        extra[f"missing__{column}"] = combined[column].isna().to_numpy(float)

    for column in WEATHER_MEMORY_COLUMNS:
        values = combined[column]
        for lag in EXTRA_LAGS:
            extra[f"{column}__lag_{lag}"] = _shift_lag(values, lag)

    for column in VOLATILITY_COLUMNS:
        past = combined[column].astype(float).shift(1)
        extra[f"{column}__past_std_8"] = past.rolling(8).std().to_numpy()

    temperature = combined["station_avg_temp_c"].astype(float)
    humidity = combined["reanalysis_specific_humidity_g_per_kg"].astype(float)
    extra["temp_lag10_x_humidity_lag8"] = _shift_lag(temperature, 10) * _shift_lag(
        humidity, 8
    )
    extra["humidity_above_past_12"] = humidity.to_numpy() - humidity.shift(1).rolling(
        12
    ).mean().to_numpy()

    for column in ANOMALY_COLUMNS:
        observed = combined[column].to_numpy(float)
        extra[f"{column}__seasonal_anomaly"] = observed - _past_same_week_mean(
            observed, week
        )

    target = combined[TARGET_COLUMN].to_numpy(float) if TARGET_COLUMN in combined else (
        np.full(len(combined), np.nan)
    )
    extra["cases_same_week_past_mean"] = _case_climatology(week, target, train_mask)

    features = pd.concat(
        [base, pd.DataFrame(extra, index=combined.index)],
        axis=1,
    )
    features = features.loc[:, ~features.columns.duplicated()].replace(
        [np.inf, -np.inf], np.nan
    )
    return features.astype(float)
