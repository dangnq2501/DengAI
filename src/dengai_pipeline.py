"""Compatibility feature builders used by the forest error-analysis workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from train_tree_ensemble import (
    DATE_COLUMN,
    KEY_COLUMNS,
    RAW_NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_data,
)

LAGGED_CLIMATE_COLUMNS = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "reanalysis_relative_humidity_percent",
    "station_avg_temp_c",
    "precipitation_amt_mm",
]
ROLLING_WINDOWS = {
    "base": (),
    "rolling": (2, 4, 8, 12),
    "long_rolling": (2, 4, 8, 12, 16, 26),
}
CLIMATE_LAGS = (1, 2, 4, 8, 12, 16)
COMPACT_WEATHER_COLUMNS = [
    "station_avg_temp_c",
    "station_min_temp_c",
    "station_max_temp_c",
    "station_precip_mm",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "precipitation_amt_mm",
]
COMPACT_LAGS = (1, 2, 4, 8)
COMPACT_ROLLING_WINDOWS = (2, 4, 8)


def make_features(city_frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Create the original base, rolling, or compact forest feature matrix."""
    if mode not in {*ROLLING_WINDOWS, "compact_weather"}:
        raise ValueError(f"Unknown feature mode: {mode}")
    if not city_frame[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Feature rows must be sorted chronologically")
    if mode == "compact_weather":
        return _make_compact_weather_features(city_frame)

    feature_data: dict[str, pd.Series | np.ndarray] = {}
    week = city_frame["weekofyear"].astype(float)
    for harmonic in range(1, 5):
        angle = 2.0 * np.pi * harmonic * week / 52.0
        feature_data[f"week_sin_{harmonic}"] = np.sin(angle)
        feature_data[f"week_cos_{harmonic}"] = np.cos(angle)
    numeric_columns = [
        column
        for column in city_frame.select_dtypes(include="number").columns
        if column not in RAW_NON_FEATURE_COLUMNS
    ]
    for column in numeric_columns:
        values = city_frame[column].astype(float)
        feature_data[column] = values
        for window in ROLLING_WINDOWS[mode]:
            feature_data[f"{column}__mean_{window}"] = values.rolling(
                window, min_periods=1
            ).mean()
    if ROLLING_WINDOWS[mode]:
        for column in LAGGED_CLIMATE_COLUMNS:
            for lag in CLIMATE_LAGS:
                feature_data[f"{column}__lag_{lag}"] = city_frame[column].shift(lag)
    return pd.DataFrame(feature_data, index=city_frame.index).astype(float)


def _make_compact_weather_features(city_frame: pd.DataFrame) -> pd.DataFrame:
    feature_data: dict[str, pd.Series | np.ndarray] = {}
    week = city_frame["weekofyear"].astype(float)
    feature_data["weekofyear"] = week
    feature_data["week_sin_1"] = np.sin(2.0 * np.pi * week / 52.0)
    feature_data["week_cos_1"] = np.cos(2.0 * np.pi * week / 52.0)
    numeric_columns = [
        column
        for column in city_frame.select_dtypes(include="number").columns
        if column not in RAW_NON_FEATURE_COLUMNS
    ]
    for column in numeric_columns:
        feature_data[column] = city_frame[column].astype(float)
    ndvi = city_frame[["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]]
    feature_data["ndvi_mean"] = ndvi.mean(axis=1)
    feature_data["ndvi_std"] = ndvi.std(axis=1)
    feature_data["station_temp_range"] = (
        city_frame["station_max_temp_c"] - city_frame["station_min_temp_c"]
    )
    for column in COMPACT_WEATHER_COLUMNS:
        values = city_frame[column].astype(float)
        for lag in COMPACT_LAGS:
            feature_data[f"{column}__lag_{lag}"] = values.shift(lag)
        for window in COMPACT_ROLLING_WINDOWS:
            feature_data[f"{column}__past_mean_{window}"] = (
                values.shift(1).rolling(window).mean()
            )
    return pd.DataFrame(feature_data, index=city_frame.index).astype(float)
