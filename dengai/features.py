"""Feature construction for the tree models and the neural network.

Three representations of the same weekly climate history are provided:

* ``tree_features``   - current climate, seasonal harmonics, sparse lags and
                        trailing means (input of the Random/Extra Trees);
* ``raw_lags``        - each variable's selected contiguous history window,
                        concatenated (461 SJ / 380 IQ inputs);
* ``multiscale``      - 11 causal summaries per variable plus 4 seasonal
                        harmonics (180 inputs). This is the final model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import (
    CONTEXT_WEEKS,
    DATE_COLUMN,
    HISTORY_WEEKS,
    KEY_COLUMNS,
    LAG_WINDOWS,
    MEAN_WINDOWS,
    NDVI_COLUMNS,
    PREDICTION_LAG,
    SEASONAL_NAMES,
    STD_WINDOWS,
    SUMMARY_NAMES,
    TARGET_COLUMN,
    WEATHER_FEATURES,
)

RAW_LAGS = "raw_lags"
MULTISCALE = "multiscale"
RAW_PLUS_MULTISCALE = "raw_plus_multiscale"
REPRESENTATIONS = (RAW_LAGS, MULTISCALE, RAW_PLUS_MULTISCALE)

# ---------------------------------------------------------------------------
# Tree features
# ---------------------------------------------------------------------------

TREE_MEMORY_COLUMNS = [
    "station_avg_temp_c",
    "station_min_temp_c",
    "station_max_temp_c",
    "station_precip_mm",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "precipitation_amt_mm",
    "reanalysis_precip_amt_kg_per_m2",
]
TREE_LAGS = (1, 2, 4, 8, 12, 16)
TREE_MEAN_WINDOWS = (2, 4, 8, 12, 16)


def tree_features(city_frame: pd.DataFrame) -> pd.DataFrame:
    """Past-only tabular features for one chronologically sorted city.

    Missing values are left as NaN; the tree pipeline imputes them with
    training-fold medians.
    """
    if not city_frame[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Rows must be sorted chronologically")
    week = city_frame["weekofyear"].astype(float)
    data: dict[str, pd.Series] = {"year": city_frame["year"].astype(float), "weekofyear": week}
    for harmonic in range(1, 5):
        angle = 2.0 * np.pi * harmonic * week / 52.0
        data[f"week_sin_{harmonic}"] = np.sin(angle)
        data[f"week_cos_{harmonic}"] = np.cos(angle)
    excluded = {*KEY_COLUMNS, DATE_COLUMN, TARGET_COLUMN}
    for column in city_frame.select_dtypes(include="number").columns:
        if column not in excluded:
            data[column] = city_frame[column].astype(float)
    ndvi = city_frame[NDVI_COLUMNS]
    data["ndvi_mean"] = ndvi.mean(axis=1)
    data["ndvi_std"] = ndvi.std(axis=1)
    data["station_temp_range"] = city_frame["station_max_temp_c"] - city_frame["station_min_temp_c"]
    data["reanalysis_temp_range"] = (
        city_frame["reanalysis_max_air_temp_k"] - city_frame["reanalysis_min_air_temp_k"]
    )
    data["temperature_humidity_interaction"] = (
        city_frame["station_avg_temp_c"] * city_frame["reanalysis_relative_humidity_percent"]
    )
    for column in TREE_MEMORY_COLUMNS:
        values = city_frame[column].astype(float)
        for lag in TREE_LAGS:
            data[f"{column}__lag_{lag}"] = values.shift(lag)
        for window in TREE_MEAN_WINDOWS:
            # shift(1) keeps the current week out of its own "past" mean.
            data[f"{column}__past_mean_{window}"] = values.shift(1).rolling(window).mean()
    return pd.DataFrame(data, index=city_frame.index).astype(float)


# ---------------------------------------------------------------------------
# Neural-network histories
# ---------------------------------------------------------------------------


def normalization_stats(sj_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Z-score parameters for every climate variable, from San Juan training rows.

    Both cities are scaled with San Juan statistics ("shared-SJ"); this beat
    city-specific scaling on the leaderboard. Statistics are computed on the
    same padded series used for training (the last 53 weeks prepended), so the
    final year counts twice. Week of year is min-max scaled.
    """
    reference = pd.concat([sj_train.iloc[-CONTEXT_WEEKS:], sj_train], ignore_index=True)
    stats = {
        feature: (
            float(np.nanmean(reference[feature].to_numpy(float))),
            float(np.nanstd(reference[feature].to_numpy(float))),
        )
        for feature in WEATHER_FEATURES
    }
    week = reference["weekofyear"].to_numpy(float)
    stats["weekofyear"] = (float(np.nanmin(week)), float(np.nanmax(week)))
    return stats


def _scaled_columns(frame: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    scaled = {}
    for feature in WEATHER_FEATURES:
        mean, std = stats[feature]
        scaled[feature] = (frame[feature].to_numpy(float) - mean) / max(std, 1e-12)
    low, high = stats["weekofyear"]
    scaled["weekofyear"] = (frame["weekofyear"].to_numpy(float) - low) / max(high - low, 1.0)
    return scaled


def raw_lag_names(city: str) -> list[str]:
    return [
        f"{feature}__t-{lag}"
        for feature, window in LAG_WINDOWS[city].items()
        for lag in range(window - 1, -1, -1)
    ]


def multiscale_names() -> list[str]:
    return [f"{feature}__{name}" for feature in WEATHER_FEATURES for name in SUMMARY_NAMES] + SEASONAL_NAMES


def feature_names(city: str, representation: str) -> list[str]:
    if representation == RAW_LAGS:
        return raw_lag_names(city)
    if representation == MULTISCALE:
        return multiscale_names()
    return raw_lag_names(city) + multiscale_names()


def _raw_lag_block(scaled: dict[str, np.ndarray], city: str, endpoints: np.ndarray) -> np.ndarray:
    blocks = []
    for feature, window in LAG_WINDOWS[city].items():
        windows = sliding_window_view(scaled[feature], window)  # row k ends at k+window-1
        blocks.append(windows[endpoints - window + 1])
    return np.concatenate(blocks, axis=1)


def _multiscale_block(
    scaled: dict[str, np.ndarray], week_numbers: np.ndarray, endpoints: np.ndarray
) -> np.ndarray:
    columns = []
    for feature in WEATHER_FEATURES:
        history = sliding_window_view(scaled[feature], HISTORY_WEEKS)[endpoints - HISTORY_WEEKS + 1]
        means = {w: history[:, -w:].mean(axis=1) for w in MEAN_WINDOWS}
        columns.append(history[:, -1])  # current level
        columns.extend(means[w] for w in MEAN_WINDOWS)  # level at 6 timescales
        columns.extend(history[:, -w:].std(axis=1) for w in STD_WINDOWS)  # volatility
        columns.append(means[4] - means[13])  # last month vs last quarter
        columns.append((history[:, -1] - history[:, -13]) / 12.0)  # 13-week slope
    phase = 2.0 * np.pi * (week_numbers[endpoints] - 1.0) / 52.0
    columns.extend([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    return np.column_stack(columns)


def _history_matrix(
    series: pd.DataFrame,
    city: str,
    stats: dict[str, tuple[float, float]],
    endpoints: np.ndarray,
    representation: str,
) -> np.ndarray:
    scaled = _scaled_columns(series, stats)
    if representation == RAW_LAGS:
        matrix = _raw_lag_block(scaled, city, endpoints)
    elif representation == MULTISCALE:
        matrix = _multiscale_block(scaled, series["weekofyear"].to_numpy(float), endpoints)
    elif representation == RAW_PLUS_MULTISCALE:
        matrix = np.concatenate(
            [
                _raw_lag_block(scaled, city, endpoints),
                _multiscale_block(scaled, series["weekofyear"].to_numpy(float), endpoints),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")
    matrix = matrix.astype("float32")
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite values in history matrix; interpolate first")
    return matrix


def training_matrix(
    train_city: pd.DataFrame,
    city: str,
    stats: dict[str, tuple[float, float]],
    representation: str = MULTISCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every training week with the climate history ending at that week.

    The first 51 weeks lack a full year of history, so the series is padded
    circularly with its own last 53 weeks. This keeps all labels; it borrows
    climate (never cases) from the end of the training period for the first
    year only. Row order is ``[n-2, n-1, 0, 1, ..., n-3]``.
    """
    padded = pd.concat([train_city.iloc[-CONTEXT_WEEKS:], train_city], ignore_index=True)
    endpoints = np.arange(len(train_city)) + HISTORY_WEEKS - 1
    x = _history_matrix(padded, city, stats, endpoints, representation)
    y = padded[TARGET_COLUMN].to_numpy(float)[endpoints].astype("float32")
    return x, y


def prediction_matrix(
    history_city: pd.DataFrame,
    future_city: pd.DataFrame,
    city: str,
    stats: dict[str, tuple[float, float]],
    representation: str = MULTISCALE,
    lag: int = PREDICTION_LAG,
) -> np.ndarray:
    """Inputs for future weeks: week t uses climate history ending at t - ``lag``."""
    if len(history_city) < HISTORY_WEEKS + lag:
        raise ValueError("Not enough history before the first predicted week")
    series = pd.concat([history_city, future_city], ignore_index=True)
    endpoints = len(history_city) + np.arange(len(future_city)) - lag
    return _history_matrix(series, city, stats, endpoints, representation)
