"""Alternative causal representations of the same weekly climate history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    LAG_WINDOWS,
    SHARED_SJ_NORMALIZATION,
    TARGET_COLUMN,
    WEATHER_FEATURES,
)
from .data import interpolate_numeric, normalization_parameters
from .features import _flatten_history_rows, _scale_columns


RAW_LAGS = "raw_lags"
MULTISCALE_SUMMARIES = "multiscale_summaries"
RAW_PLUS_SUMMARIES = "raw_plus_summaries"
REPRESENTATIONS = (RAW_LAGS, MULTISCALE_SUMMARIES, RAW_PLUS_SUMMARIES)

MEAN_WINDOWS = (2, 4, 8, 13, 26, 52)
STD_WINDOWS = (4, 13)
SUMMARY_VALUES_PER_WEATHER_FEATURE = 11
SEASONAL_FEATURE_COUNT = 4
SUMMARY_DIMENSION = (
    len(WEATHER_FEATURES) * SUMMARY_VALUES_PER_WEATHER_FEATURE
    + SEASONAL_FEATURE_COUNT
)


def _multiscale_summary_rows(
    combined: pd.DataFrame,
    city: str,
    parameters: dict[str, tuple[float, float]],
    row_count: int,
) -> np.ndarray:
    """Summarize levels, variability, change, trend, and cyclic seasonality."""
    scaled = _scale_columns(combined, parameters)
    week_numbers = combined["weekofyear"].to_numpy(float)
    rows = []
    for output_index in range(row_count):
        endpoint = 51 + output_index
        parts: list[float] = []
        for feature in WEATHER_FEATURES:
            history = scaled[feature][endpoint - 51 : endpoint + 1]
            means = {window: float(np.mean(history[-window:])) for window in MEAN_WINDOWS}
            parts.extend(
                [
                    float(history[-1]),
                    *(means[window] for window in MEAN_WINDOWS),
                    *(float(np.std(history[-window:])) for window in STD_WINDOWS),
                    means[4] - means[13],
                    float((history[-1] - history[-13]) / 12.0),
                ]
            )

        # Unlike three raw week numbers, harmonics encode the December/January
        # boundary continuously and expose annual plus semiannual seasonality.
        phase = 2.0 * np.pi * (week_numbers[endpoint] - 1.0) / 52.0
        parts.extend(
            [np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)]
        )
        rows.append(parts)

    matrix = np.asarray(rows, dtype="float32")
    if matrix.shape != (row_count, SUMMARY_DIMENSION):
        raise AssertionError(f"Unexpected {city} summary shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {city} multiscale summaries")
    return matrix


def _combine_representation(
    combined: pd.DataFrame,
    city: str,
    parameters: dict[str, tuple[float, float]],
    row_count: int,
    representation: str,
) -> np.ndarray:
    if representation == RAW_LAGS:
        return _flatten_history_rows(combined, city, parameters, row_count)

    summaries = _multiscale_summary_rows(combined, city, parameters, row_count)
    if representation == MULTISCALE_SUMMARIES:
        return summaries
    if representation == RAW_PLUS_SUMMARIES:
        raw = _flatten_history_rows(combined, city, parameters, row_count)
        return np.concatenate([raw, summaries], axis=1).astype("float32")
    raise ValueError(f"Unknown representation: {representation}")


def build_representation_training_matrix(
    train: pd.DataFrame,
    city: str,
    representation: str = RAW_LAGS,
    normalization: str = SHARED_SJ_NORMALIZATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one training matrix without changing historical alignment."""
    interpolated = interpolate_numeric(train)
    parameters = normalization_parameters(interpolated, city, normalization)
    city_frame = interpolated.loc[interpolated["city"].eq(city)].reset_index(drop=True)
    combined = pd.concat([city_frame.iloc[-53:], city_frame], ignore_index=True)
    matrix = _combine_representation(
        combined, city, parameters, len(city_frame), representation
    )
    target = combined[TARGET_COLUMN].to_numpy(float)
    labels = target[51 : 51 + len(city_frame)].astype("float32")
    return matrix, labels


def build_representation_test_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    city: str,
    representation: str = RAW_LAGS,
    normalization: str = SHARED_SJ_NORMALIZATION,
) -> np.ndarray:
    """Create a test matrix using the same training context and representation."""
    interpolated_train = interpolate_numeric(train)
    interpolated_test = interpolate_numeric(test)
    parameters = normalization_parameters(interpolated_train, city, normalization)
    train_city = interpolated_train.loc[
        interpolated_train["city"].eq(city)
    ].reset_index(drop=True)
    test_city = interpolated_test.loc[
        interpolated_test["city"].eq(city)
    ].reset_index(drop=True)
    combined = pd.concat([train_city.iloc[-53:], test_city], ignore_index=True)
    return _combine_representation(
        combined, city, parameters, len(test_city), representation
    )
