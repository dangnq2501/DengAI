"""Train the confirmed DengAI city-specific tree solution.

Iquitos uses Random Forest regression on log-transformed case counts. San Juan
uses Extra Trees with a Poisson split criterion. The script is self-contained:
it validates the data, creates past-only climate features, runs annual temporal
backtests, fits the final models, and writes the competition submission.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

KEY_COLUMNS = ["city", "year", "weekofyear"]
DATE_COLUMN = "week_start_date"
TARGET_COLUMN = "total_cases"
RAW_NON_FEATURE_COLUMNS = {*KEY_COLUMNS, DATE_COLUMN, TARGET_COLUMN}
SUPPORTED_CITIES = {"iq", "sj"}
WEATHER_MEMORY_COLUMNS = [
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
NDVI_COLUMNS = ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]
FEATURE_LAGS = (1, 2, 4, 8, 12, 16)
FEATURE_WINDOWS = (2, 4, 8, 12, 16)
DEFAULT_MODEL_PARAMS = {
    "iq": {"max_features": 0.5, "min_samples_leaf": 5, "max_depth": None},
    "sj": {"max_features": 1.0, "min_samples_leaf": 5, "max_depth": None},
}
TUNING_GRID = {
    "iq": {
        "max_features": (0.4, 0.5, 0.6),
        "min_samples_leaf": (3, 5, 7),
        "max_depth": (None,),
    },
    "sj": {
        "max_features": (0.8, 0.9, 1.0),
        "min_samples_leaf": (3, 5, 7),
        "max_depth": (None, 20),
    },
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "artifacts"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Run the conservative city-specific grid search and write a separate "
            "tuned submission without replacing the confirmed submission."
        ),
    )
    parser.add_argument(
        "--tuning-estimators",
        type=int,
        default=200,
        help="Trees per model during tuning; final fitting uses --n-estimators.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Fit final models without rerunning the annual temporal backtest.",
    )
    return parser.parse_args()


def _read_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    if frame[DATE_COLUMN].isna().any():
        raise ValueError(f"Invalid dates found in {path}")
    return frame


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the four supplied competition CSV files."""
    train_features = _read_features(data_dir / "dengue_features_train.csv")
    labels = pd.read_csv(data_dir / "dengue_labels_train.csv")
    test_features = _read_features(data_dir / "dengue_features_test.csv")
    template = pd.read_csv(data_dir / "submission_format.csv")
    frames = {
        "train features": train_features,
        "labels": labels,
        "test features": test_features,
        "submission template": template,
    }
    for name, frame in frames.items():
        missing = set(KEY_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing key columns: {sorted(missing)}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} contains duplicate city/year/week keys")
    if TARGET_COLUMN not in labels or TARGET_COLUMN not in template:
        raise ValueError(f"Labels and template must contain {TARGET_COLUMN!r}")
    if labels[TARGET_COLUMN].isna().any() or (labels[TARGET_COLUMN] < 0).any():
        raise ValueError("Training targets must be non-negative and complete")
    train = train_features.merge(
        labels, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    if train[TARGET_COLUMN].isna().any() or len(train) != len(labels):
        raise ValueError("Feature and label keys do not align exactly")
    if not template[KEY_COLUMNS].reset_index(drop=True).equals(
        test_features[KEY_COLUMNS].reset_index(drop=True)
    ):
        raise ValueError("Submission template order/keys do not match test features")
    train = train.sort_values(["city", DATE_COLUMN]).reset_index(drop=True)
    unknown = set(train["city"].unique()) - SUPPORTED_CITIES
    if unknown:
        raise ValueError(f"Unsupported cities: {sorted(unknown)}")
    for city in sorted(SUPPORTED_CITIES):
        train_city = train.loc[train["city"].eq(city)]
        test_city = test_features.loc[test_features["city"].eq(city)].sort_values(
            DATE_COLUMN
        )
        if train_city.empty or test_city.empty:
            raise ValueError(f"Missing train or test rows for city {city!r}")
        train_steps = train_city[DATE_COLUMN].diff().dropna().dt.days
        test_steps = test_city[DATE_COLUMN].diff().dropna().dt.days
        if not (train_steps.between(7, 9).all() and test_steps.between(7, 9).all()):
            raise ValueError(f"Weekly dates are not contiguous for city {city!r}")
        gap = test_city[DATE_COLUMN].iloc[0] - train_city[DATE_COLUMN].iloc[-1]
        if gap.days != 7:
            raise ValueError(f"Train and test are not contiguous for city {city!r}")
    return train, test_features, template


def make_features(city_frame: pd.DataFrame) -> pd.DataFrame:
    """Create seasonal, climate, interaction, lag, and trailing features."""
    if not city_frame[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Feature rows must be sorted chronologically")
    week = city_frame["weekofyear"].astype(float)
    data: dict[str, pd.Series | np.ndarray] = {
        "year": city_frame["year"].astype(float),
        "weekofyear": week,
    }
    for harmonic in range(1, 5):
        angle = 2.0 * np.pi * harmonic * week / 52.0
        data[f"week_sin_{harmonic}"] = np.sin(angle)
        data[f"week_cos_{harmonic}"] = np.cos(angle)
    numeric_columns = [
        column
        for column in city_frame.select_dtypes(include="number").columns
        if column not in RAW_NON_FEATURE_COLUMNS
    ]
    for column in numeric_columns:
        data[column] = city_frame[column].astype(float)
    ndvi = city_frame[NDVI_COLUMNS]
    data["ndvi_mean"] = ndvi.mean(axis=1)
    data["ndvi_std"] = ndvi.std(axis=1)
    data["station_temp_range"] = (
        city_frame["station_max_temp_c"] - city_frame["station_min_temp_c"]
    )
    data["reanalysis_temp_range"] = (
        city_frame["reanalysis_max_air_temp_k"]
        - city_frame["reanalysis_min_air_temp_k"]
    )
    data["temperature_humidity_interaction"] = (
        city_frame["station_avg_temp_c"]
        * city_frame["reanalysis_relative_humidity_percent"]
    )
    for column in WEATHER_MEMORY_COLUMNS:
        values = city_frame[column].astype(float)
        for lag in FEATURE_LAGS:
            data[f"{column}__lag_{lag}"] = values.shift(lag)
        for window in FEATURE_WINDOWS:
            data[f"{column}__past_mean_{window}"] = (
                values.shift(1).rolling(window).mean()
            )
    return pd.DataFrame(data, index=city_frame.index).astype(float)


def tree_model(
    city: str,
    random_state: int = 42,
    n_estimators: int = 500,
    model_params: dict[str, Any] | None = None,
) -> Any:
    """Return the confirmed city-specific estimator."""
    params = {**DEFAULT_MODEL_PARAMS[city], **(model_params or {})}
    if city == "iq":
        forest = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion="squared_error",
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            max_depth=params["max_depth"],
            n_jobs=-1,
            random_state=random_state,
        )
        return TransformedTargetRegressor(
            regressor=make_pipeline(SimpleImputer(strategy="median"), forest),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    if city == "sj":
        forest = ExtraTreesRegressor(
            n_estimators=n_estimators,
            criterion="poisson",
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            max_depth=params["max_depth"],
            n_jobs=-1,
            random_state=random_state,
        )
        return make_pipeline(SimpleImputer(strategy="median"), forest)
    raise ValueError(f"Unsupported city: {city}")


def _integer_cases(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.floor(np.maximum(0.0, values) + 0.5).astype(int)


def run_backtests(
    train: pd.DataFrame,
    random_state: int,
    n_estimators: int,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate four and six expanding annual folds without shuffling."""
    scores: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold_count in (4, 6):
        all_actual: list[np.ndarray] = []
        all_prediction: list[np.ndarray] = []
        for city, city_frame in train.groupby("city", sort=True):
            city_frame = city_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
            features = make_features(city_frame)
            targets = city_frame[TARGET_COLUMN].to_numpy(float)
            city_actual: list[np.ndarray] = []
            city_prediction: list[np.ndarray] = []
            starts = range(len(city_frame) - fold_count * 52, len(city_frame), 52)
            for fold, start in enumerate(starts, start=1):
                train_indices = np.arange(start)
                validation_indices = np.arange(start, start + 52)
                model = tree_model(
                    city,
                    random_state,
                    n_estimators,
                    None if model_params is None else model_params[city],
                )
                model.fit(features.iloc[train_indices], targets[train_indices])
                prediction = _integer_cases(
                    model.predict(features.iloc[validation_indices])
                )
                actual = targets[validation_indices].astype(int)
                city_actual.append(actual)
                city_prediction.append(prediction)
                all_actual.append(actual)
                all_prediction.append(prediction)
                if fold_count == 6:
                    rows = city_frame.iloc[validation_indices][
                        KEY_COLUMNS + [DATE_COLUMN]
                    ].copy()
                    rows.insert(0, "fold", fold)
                    rows["actual"] = actual
                    rows["tree_prediction"] = prediction
                    prediction_rows.append(rows)
            actual_all = np.concatenate(city_actual)
            prediction_all = np.concatenate(city_prediction)
            scores.append(
                {
                    "fold_count": fold_count,
                    "city": city,
                    "model": "random_forest_log"
                    if city == "iq"
                    else "extra_trees_poisson",
                    "n_predictions": len(actual_all),
                    "mae": mean_absolute_error(actual_all, prediction_all),
                }
            )
        scores.append(
            {
                "fold_count": fold_count,
                "city": "all",
                "model": "city_specific_tree_ensemble",
                "n_predictions": sum(map(len, all_actual)),
                "mae": mean_absolute_error(
                    np.concatenate(all_actual), np.concatenate(all_prediction)
                ),
            }
        )
    return pd.DataFrame(scores), pd.concat(prediction_rows, ignore_index=True)


def _grid_configs(city: str) -> list[dict[str, Any]]:
    grid = TUNING_GRID[city]
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _evaluate_tuning_config(
    city: str,
    city_frame: pd.DataFrame,
    horizon: int,
    params: dict[str, Any],
    random_state: int,
    n_estimators: int,
) -> dict[str, float]:
    features = make_features(city_frame)
    targets = city_frame[TARGET_COLUMN].to_numpy(float)
    annual_actual: list[np.ndarray] = []
    annual_prediction: list[np.ndarray] = []
    for start in range(len(city_frame) - 4 * 52, len(city_frame), 52):
        train_indices = np.arange(start)
        validation_indices = np.arange(start, start + 52)
        model = tree_model(city, random_state, n_estimators, params)
        model.fit(features.iloc[train_indices], targets[train_indices])
        annual_actual.append(targets[validation_indices].astype(int))
        annual_prediction.append(
            _integer_cases(model.predict(features.iloc[validation_indices]))
        )

    horizon_start = len(city_frame) - horizon
    train_indices = np.arange(horizon_start)
    validation_indices = np.arange(horizon_start, len(city_frame))
    model = tree_model(city, random_state, n_estimators, params)
    model.fit(features.iloc[train_indices], targets[train_indices])
    horizon_actual = targets[validation_indices].astype(int)
    horizon_prediction = _integer_cases(
        model.predict(features.iloc[validation_indices])
    )
    outbreak_threshold = np.quantile(horizon_actual, 0.90)
    outbreak = horizon_actual >= outbreak_threshold
    return {
        "annual_mae": mean_absolute_error(
            np.concatenate(annual_actual), np.concatenate(annual_prediction)
        ),
        "horizon_mae": mean_absolute_error(horizon_actual, horizon_prediction),
        "outbreak_mae": mean_absolute_error(
            horizon_actual[outbreak], horizon_prediction[outbreak]
        ),
        "prediction_mean": float(np.mean(horizon_prediction)),
        "prediction_max": float(np.max(horizon_prediction)),
    }


def tune_hyperparameters(
    train: pd.DataFrame,
    test: pd.DataFrame,
    random_state: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run a guarded local grid search using chronological validation only."""
    rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    for city in sorted(SUPPORTED_CITIES):
        city_frame = (
            train.loc[train["city"].eq(city)]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
        horizon = int(test["city"].eq(city).sum())
        configs = _grid_configs(city)
        baseline = _evaluate_tuning_config(
            city,
            city_frame,
            horizon,
            DEFAULT_MODEL_PARAMS[city],
            random_state,
            n_estimators,
        )
        for trial, params in enumerate(configs, start=1):
            metrics = _evaluate_tuning_config(
                city,
                city_frame,
                horizon,
                params,
                random_state,
                n_estimators,
            )
            amplitude_ok = (
                metrics["prediction_mean"] >= 0.90 * baseline["prediction_mean"]
                and metrics["prediction_mean"] <= 1.10 * baseline["prediction_mean"]
                and metrics["prediction_max"] >= 0.90 * baseline["prediction_max"]
            )
            validation_ok = (
                metrics["annual_mae"] <= 1.03 * baseline["annual_mae"]
                and metrics["horizon_mae"] <= 1.03 * baseline["horizon_mae"]
            )
            relative_score = 0.5 * (
                metrics["annual_mae"] / baseline["annual_mae"]
                + metrics["horizon_mae"] / baseline["horizon_mae"]
            )
            eligible = amplitude_ok and validation_ok
            row = {
                "city": city,
                "trial": trial,
                **params,
                **metrics,
                "baseline_annual_mae": baseline["annual_mae"],
                "baseline_horizon_mae": baseline["horizon_mae"],
                "amplitude_ok": amplitude_ok,
                "validation_ok": validation_ok,
                "eligible": eligible,
                "selection_score": relative_score,
            }
            rows.append(row)
            print(
                f"{city.upper()} {trial:02d}/{len(configs):02d} "
                f"annual={metrics['annual_mae']:.3f} "
                f"horizon={metrics['horizon_mae']:.3f} "
                f"score={relative_score:.4f} eligible={eligible}"
            )
        city_rows = pd.DataFrame(row for row in rows if row["city"] == city)
        winner = city_rows.loc[city_rows["eligible"]].sort_values(
            ["selection_score", "horizon_mae", "annual_mae", "trial"]
        ).iloc[0]
        best_params[city] = {
            "max_features": float(winner["max_features"]),
            "min_samples_leaf": int(winner["min_samples_leaf"]),
            "max_depth": (
                None if pd.isna(winner["max_depth"]) else int(winner["max_depth"])
            ),
        }
    return pd.DataFrame(rows), best_params


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    random_state: int,
    n_estimators: int,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit both city models on all labels and predict the supplied test rows."""
    rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {
        "cities": {},
        "random_state": random_state,
        "n_estimators": n_estimators,
        "model_params": model_params or DEFAULT_MODEL_PARAMS,
    }
    for city in sorted(train["city"].unique()):
        train_city = train.loc[train["city"].eq(city)].copy()
        test_city = test.loc[test["city"].eq(city)].copy()
        train_city["_split"] = "train"
        test_city["_split"] = "test"
        combined = (
            pd.concat([train_city, test_city], ignore_index=True, sort=False)
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
        train_indices = np.flatnonzero(combined["_split"].eq("train").to_numpy())
        test_indices = np.flatnonzero(combined["_split"].eq("test").to_numpy())
        features = make_features(combined)
        targets = combined.iloc[train_indices][TARGET_COLUMN].to_numpy(float)
        model = tree_model(
            city,
            random_state,
            n_estimators,
            None if model_params is None else model_params[city],
        )
        model.fit(features.iloc[train_indices], targets)
        result = combined.iloc[test_indices][KEY_COLUMNS].copy()
        result[TARGET_COLUMN] = _integer_cases(
            model.predict(features.iloc[test_indices])
        )
        rows.append(result)
        artifacts["cities"][city] = {
            "model": model,
            "feature_columns": features.columns.tolist(),
        }
    predictions = pd.concat(rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        predictions, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission, artifacts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, template = load_data(args.data_dir)
    model_params: dict[str, dict[str, Any]] | None = None
    if args.tune:
        tuning_results, model_params = tune_hyperparameters(
            train,
            test,
            args.random_state,
            args.tuning_estimators,
        )
        tuning_results.to_csv(
            args.output_dir / "tree_tuning_results.csv", index=False
        )
        (args.output_dir / "tree_tuning_best_params.json").write_text(
            json.dumps(model_params, indent=2) + "\n", encoding="utf-8"
        )
        print("\nSelected guarded parameters:")
        print(json.dumps(model_params, indent=2))

    if not args.skip_backtest:
        scores, validation_predictions = run_backtests(
            train,
            args.random_state,
            args.n_estimators,
            model_params,
        )
        validation_prefix = "tree_tuned" if args.tune else "tree"
        scores.to_csv(
            args.output_dir / f"{validation_prefix}_validation_scores.csv",
            index=False,
        )
        validation_predictions.to_csv(
            args.output_dir / f"{validation_prefix}_validation_predictions.csv",
            index=False,
        )
        print(scores.to_string(index=False))
    submission, artifacts = fit_final(
        train,
        test,
        template,
        args.random_state,
        args.n_estimators,
        model_params,
    )
    if args.tune:
        submission_name = "submission_tree_tuned.csv"
        model_name = "tree_tuned_models.joblib"
    else:
        submission_name = "submission_tree_ensemble_2.csv"
        model_name = "tree_ensemble_models_2.joblib"
    submission.to_csv(args.output_dir / submission_name, index=False)
    joblib.dump(artifacts, args.output_dir / model_name)
    print("\nFinal prediction distributions:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
