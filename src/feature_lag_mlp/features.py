"""Stage 2: transform weekly tables into flattened feature-specific histories."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    INPUT_DIMENSIONS,
    LAG_WINDOWS,
    SHARED_SJ_NORMALIZATION,
    TARGET_COLUMN,
    WEATHER_FEATURES,
)
from .data import interpolate_numeric, normalization_parameters


def _scale_columns(
    frame: pd.DataFrame,
    parameters: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    scaled: dict[str, np.ndarray] = {}
    for feature in WEATHER_FEATURES:
        mean, standard_deviation = parameters[feature]
        scaled[feature] = (
            frame[feature].to_numpy(float) - mean
        ) / max(standard_deviation, 1e-12)
    low, high = parameters["weekofyear"]
    scaled["weekofyear"] = (
        frame["weekofyear"].to_numpy(float) - low
    ) / max(high - low, 1.0)
    return scaled


def _flatten_history_rows(
    combined: pd.DataFrame,
    city: str,
    parameters: dict[str, tuple[float, float]],
    row_count: int,
) -> np.ndarray:
    """Concatenate the selected history length of every feature for each row."""
    scaled = _scale_columns(combined, parameters)
    rows: list[np.ndarray] = []
    for output_index in range(row_count):
        # The original generator starts at 52. With 53 prefixed rows, endpoint
        # 51 preserves its confirmed two-row alignment offset.
        endpoint = 51 + output_index
        parts = [
            scaled[feature][endpoint - window + 1 : endpoint + 1]
            for feature, window in LAG_WINDOWS[city].items()
        ]
        rows.append(np.concatenate(parts))

    matrix = np.vstack(rows).astype("float32")
    expected = (row_count, INPUT_DIMENSIONS[city])
    if matrix.shape != expected:
        raise AssertionError(f"Unexpected {city} matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {city} history matrix")
    return matrix


def build_training_matrix(
    train: pd.DataFrame,
    city: str,
    normalization: str = SHARED_SJ_NORMALIZATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for one city using the circular 53-row prefix."""
    interpolated = interpolate_numeric(train)
    parameters = normalization_parameters(interpolated, city, normalization)
    city_frame = interpolated.loc[interpolated["city"].eq(city)].reset_index(drop=True)
    combined = pd.concat([city_frame.iloc[-53:], city_frame], ignore_index=True)
    matrix = _flatten_history_rows(combined, city, parameters, len(city_frame))

    target = combined[TARGET_COLUMN].to_numpy(float)
    labels = target[51 : 51 + len(city_frame)].astype("float32")
    if not np.isfinite(labels).all():
        raise ValueError(f"Non-finite labels for {city}")
    return matrix, labels


def build_test_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    city: str,
    normalization: str = SHARED_SJ_NORMALIZATION,
) -> np.ndarray:
    """Build one city test matrix using the final 53 training rows as context."""
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
    return _flatten_history_rows(combined, city, parameters, len(test_city))

