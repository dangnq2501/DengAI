"""Load and validate the competition files and split them into X and y."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CITIES, DATE_COLUMN, KEY_COLUMNS, TARGET_COLUMN


def find_data_dir(start: Path | None = None) -> Path:
    """Locate ``data/`` from the project root, ``notebooks/``, or a parent."""
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        for candidate in (folder / "data", folder / "DengAI" / "data"):
            if (candidate / "dengue_features_train.csv").exists():
                return candidate
    raise FileNotFoundError("Could not find the DengAI data directory")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test, template)``; train has features plus the target.

    Rows are kept in file order (all San Juan weeks, then all Iquitos weeks),
    which is chronological within each city.
    """
    features = pd.read_csv(data_dir / "dengue_features_train.csv", parse_dates=[DATE_COLUMN])
    labels = pd.read_csv(data_dir / "dengue_labels_train.csv")
    test = pd.read_csv(data_dir / "dengue_features_test.csv", parse_dates=[DATE_COLUMN])
    template = pd.read_csv(data_dir / "submission_format.csv")

    for name, frame in {"features": features, "labels": labels, "test": test}.items():
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} has duplicate city/year/week keys")
    train = features.merge(labels, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(train) != len(labels) or train[TARGET_COLUMN].isna().any():
        raise ValueError("Feature and label keys do not align")
    if not test[KEY_COLUMNS].equals(template[KEY_COLUMNS]):
        raise ValueError("Test keys do not match the submission template")

    for city in CITIES:
        train_dates = train.loc[train["city"].eq(city), DATE_COLUMN]
        test_dates = test.loc[test["city"].eq(city), DATE_COLUMN]
        if not (train_dates.is_monotonic_increasing and test_dates.is_monotonic_increasing):
            raise ValueError(f"{city} rows are not chronological")
        if (test_dates.iloc[0] - train_dates.iloc[-1]).days != 7:
            raise ValueError(f"{city} test weeks do not follow the training weeks")
    return train, test, template


def split_xy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate inputs X from the weekly case count y."""
    return train.drop(columns=[TARGET_COLUMN]), train[TARGET_COLUMN]


def city_frame(frame: pd.DataFrame, city: str) -> pd.DataFrame:
    """One city's rows in chronological order with a fresh index."""
    return frame.loc[frame["city"].eq(city)].sort_values(DATE_COLUMN).reset_index(drop=True)


def interpolate_climate(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps by linear interpolation along time within one city."""
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns.drop(
        [TARGET_COLUMN], errors="ignore"
    )
    result[numeric] = result[numeric].interpolate(method="linear", limit_direction="forward")
    return result


def make_submission(template: pd.DataFrame, predictions: dict[str, "pd.Series"]) -> pd.DataFrame:
    """Fill the template with non-negative integer predictions per city."""
    submission = template[KEY_COLUMNS].copy()
    submission[TARGET_COLUMN] = -1
    for city, values in predictions.items():
        rows = submission["city"].eq(city)
        if rows.sum() != len(values):
            raise ValueError(f"{city}: {rows.sum()} rows but {len(values)} predictions")
        submission.loc[rows, TARGET_COLUMN] = values
    if (submission[TARGET_COLUMN] < 0).any():
        raise ValueError("Submission is missing predictions")
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission
