"""Stage 1: load, validate, interpolate, and calculate normalization values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CITY_SPECIFIC_NORMALIZATION,
    DATE_COLUMN,
    KEY_COLUMNS,
    LAG_WINDOWS,
    SHARED_SJ_NORMALIZATION,
    TARGET_COLUMN,
    WEATHER_FEATURES,
)


def load_competition_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load official data and verify test rows against the submission template."""
    features = pd.read_csv(data_dir / "dengue_features_train.csv")
    labels = pd.read_csv(data_dir / "dengue_labels_train.csv")
    test = pd.read_csv(data_dir / "dengue_features_test.csv")
    template = pd.read_csv(data_dir / "submission_format.csv")
    train = features.merge(labels, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    for frame in (train, test):
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    if not test[KEY_COLUMNS].equals(template[KEY_COLUMNS]):
        raise ValueError("Test keys do not match the submission template")
    if TARGET_COLUMN not in train:
        raise ValueError(f"Training data does not contain {TARGET_COLUMN}")
    return train, test, template


def interpolate_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the confirmed global, in-file-order linear interpolation."""
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].interpolate(
        method="linear", limit_direction="forward"
    )
    return result


def normalization_parameters(
    train: pd.DataFrame,
    city: str,
    normalization: str = SHARED_SJ_NORMALIZATION,
) -> dict[str, tuple[float, float]]:
    """Calculate z-score weather and min-max week parameters.

    ``shared-sj`` intentionally derives the parameters from San Juan and applies
    them to both cities because that is the empirically stronger configuration.
    """
    if city not in LAG_WINDOWS:
        raise ValueError(f"Unknown city: {city}")
    if normalization == SHARED_SJ_NORMALIZATION:
        reference_city = "sj"
    elif normalization == CITY_SPECIFIC_NORMALIZATION:
        reference_city = city
    else:
        raise ValueError(f"Unknown normalization mode: {normalization}")

    city_frame = train.loc[train["city"].eq(reference_city)].reset_index(drop=True)
    reference = pd.concat([city_frame.iloc[-53:], city_frame], ignore_index=True)
    parameters: dict[str, tuple[float, float]] = {}
    for feature in WEATHER_FEATURES:
        values = reference[feature].to_numpy(float)
        parameters[feature] = (float(np.nanmean(values)), float(np.nanstd(values)))
    week = reference["weekofyear"].to_numpy(float)
    parameters["weekofyear"] = (float(np.nanmin(week)), float(np.nanmax(week)))
    return parameters

