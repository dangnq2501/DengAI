"""Compare San Juan climate anomalies against the supplied Poisson Extra Trees.

Development validation windows end before the latest full-horizon evaluation
window begins. That latest period has already been inspected during earlier
error analysis; treat it as retrospective evaluation, not a fresh holdout.
The default horizon is the number of San Juan test rows (normally 260 weeks).
"""

from __future__ import annotations

import argparse
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
        # "year": city_frame["year"].astype(float),
        # "weekofyear": week,
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
    city: str, random_state: int = 42, n_estimators: int = 500
) -> Any:
    """Return the confirmed city-specific estimator."""
    if city == "iq":
        forest = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion="squared_error",
            max_features=0.5,
            min_samples_leaf=5,
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
            max_features=1.0,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        )
        return make_pipeline(SimpleImputer(strategy="median"), forest)
    raise ValueError(f"Unsupported city: {city}")


def _integer_cases(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.floor(np.maximum(0.0, values) + 0.5).astype(int)


# Each experiment uses exactly the same Extra Trees settings and random seed.
VARIANTS = ("baseline", "anomalies_th", "anomalies_all")
ANOMALY_SOURCES = {
    "temperature": "reanalysis_air_temp_k",
    "humidity": "reanalysis_relative_humidity_percent",
    "rainfall": "precipitation_amt_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sj_experiments"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Default: San Juan test length. Use 52 for a separate annual experiment.")
    parser.add_argument("--min-train-weeks", type=int, default=260)
    parser.add_argument("--max-development-origins", type=int, default=4)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--final-variant", choices=VARIANTS, default=None,
                        help="Optional: fit a reviewed SJ variant on all training rows and write a submission; IQ retains its supplied model.")
    args = parser.parse_args()
    if min(args.n_estimators, args.min_train_weeks, args.max_development_origins) <= 0:
        parser.error("Estimator count, minimum training length, and origin count must be positive")
    if args.horizon is not None and args.horizon <= 0:
        parser.error("--horizon must be positive")
    args.variants = list(dict.fromkeys(["baseline", *args.variants]))
    return args


def weather_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    """Eight previous observations, excluding the current week; no target input."""
    result = {}
    for name, column in ANOMALY_SOURCES.items():
        values = frame[column].astype(float)
        if name == "temperature":
            values = values - 273.15
        result[name] = values.shift(1).rolling(8, min_periods=8).mean()
    return pd.DataFrame(result, index=frame.index)


def fit_climatology(
    frame: pd.DataFrame, summaries: pd.DataFrame, train_size: int
) -> dict[str, Any]:
    """Freeze per-week medians using only the origin's training prefix."""
    historical = summaries.iloc[:train_size].copy()
    historical["weekofyear"] = frame["weekofyear"].iloc[:train_size].to_numpy()
    tables = historical.groupby("weekofyear")[list(ANOMALY_SOURCES)].median()
    fallback = historical[list(ANOMALY_SOURCES)].median()
    if fallback.isna().any():
        raise ValueError("Not enough observed weather to fit the anomaly reference")
    counts = historical.groupby("weekofyear")[list(ANOMALY_SOURCES)].count()
    return {"weekly_medians": tables, "fallback": fallback,
            "weekly_counts": counts, "train_size": train_size, "window": 8}


def experiment_features(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
    climatology: dict[str, Any],
    variant: str,
) -> pd.DataFrame:
    if variant == "baseline":
        return baseline
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    names = ["temperature", "humidity"]
    if variant == "anomalies_all":
        names.append("rainfall")
    result = baseline.copy()
    for name in names:
        reference = frame["weekofyear"].map(climatology["weekly_medians"][name])
        # An unseen week or all-missing seasonal reference uses training-only
        # global climatology. Missing observed summaries remain NaN for the
        # estimator's training-fitted median imputer, just as in the baseline.
        reference = reference.fillna(climatology["fallback"][name])
        result[f"{name}__past_mean_8_anomaly"] = summaries[name] - reference
    return result


def validation_origins(
    n_rows: int, horizon: int, min_train: int, max_development: int
) -> list[tuple[str, int]]:
    latest_start = n_rows - horizon
    latest_development_start = latest_start - horizon
    if latest_development_start < min_train:
        raise ValueError(
            f"Need at least {min_train + 2 * horizon} SJ rows for separate "
            "development and evaluation windows. Reduce --min-train-weeks "
            "or use a separately reported --horizon 52 experiment."
        )
    # Work backward from the last eligible development window in annual steps.
    starts = list(range(latest_development_start, min_train - 1, -52))
    starts = sorted(starts[:max_development])
    assert all(start + horizon <= latest_start for start in starts)
    return [("development", start) for start in starts] + [("latest_evaluation", latest_start)]


def error_metrics(actual: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, float]:
    residual = actual - prediction
    high = actual >= threshold
    return {
        "mae": float(np.abs(residual).mean()),
        "bias": float(residual.mean()),
        "overprediction_mae_contribution": float(np.maximum(-residual, 0).mean()),
        "underprediction_mae_contribution": float(np.maximum(residual, 0).mean()),
        "high_case_mae": float(np.abs(residual[high]).mean()) if high.any() else np.nan,
        "other_week_mae": float(np.abs(residual[~high]).mean()) if (~high).any() else np.nan,
        "high_case_weeks": int(high.sum()),
    }


def run_sj_experiments(
    train: pd.DataFrame, horizon: int, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = train.loc[train.city.eq("sj")].sort_values(DATE_COLUMN).reset_index(drop=True)
    baseline = make_features(frame)
    summaries = weather_summaries(frame)
    targets = frame[TARGET_COLUMN].to_numpy(float)
    origins = validation_origins(len(frame), horizon, args.min_train_weeks,
                                 args.max_development_origins)
    score_rows, prediction_rows, reference_rows = [], [], []
    for origin, (split, start) in enumerate(origins, start=1):
        stop = start + horizon
        reference = fit_climatology(frame, summaries, start)
        cutoff = frame[DATE_COLUMN].iloc[start]
        counts = reference["weekly_counts"].reset_index()
        counts.insert(0, "origin", origin)
        counts.insert(1, "split", split)
        reference_rows.append(counts)
        actual = targets[start:stop].astype(int)
        threshold = float(np.quantile(targets[:start], .90))
        rows = frame.iloc[start:stop][KEY_COLUMNS + [DATE_COLUMN]].copy()
        rows.insert(0, "origin", origin)
        rows["split"] = split
        rows["forecast_origin_date"] = cutoff
        rows["lead_week"] = np.arange(1, horizon + 1)
        rows["actual"] = actual
        rows["high_case_threshold"] = threshold
        print(f"Origin {origin}: {split}, train={start}, validation={horizon}", flush=True)
        for variant in args.variants:
            features = experiment_features(frame, baseline, summaries, reference, variant)
            model = tree_model("sj", args.random_state, args.n_estimators)
            model.fit(features.iloc[:start], targets[:start])
            raw_prediction = np.maximum(0., model.predict(features.iloc[start:stop]))
            prediction = _integer_cases(raw_prediction)
            rows[f"{variant}_prediction"] = prediction
            rows[f"{variant}_prediction_raw"] = raw_prediction
            score_rows.append({
                "city": "sj", "origin": origin, "split": split, "variant": variant,
                "forecast_origin_date": cutoff, "n_train": start,
                "horizon_weeks": horizon, "n_predictions": len(actual),
                "high_case_threshold": threshold,
                "n_features": features.shape[1],
                **error_metrics(actual, prediction, threshold),
            })
            print(f"  {variant}: MAE={score_rows[-1]['mae']:.3f}", flush=True)
        prediction_rows.append(rows)
    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    # All origins have equal horizons. Aggregate forecast pairs, preserving
    # separate split labels; overlapping development windows are not IID folds.
    summary = scores.groupby(["split", "variant"], as_index=False).agg(
        mae=("mae", "mean"), origin_mae_std=("mae", "std"),
        bias=("bias", "mean"),
        overprediction_mae_contribution=("overprediction_mae_contribution", "mean"),
        underprediction_mae_contribution=("underprediction_mae_contribution", "mean"),
        n_origins=("origin", "size"), n_predictions=("n_predictions", "sum"),
    )
    baseline_mae = summary.loc[summary.variant.eq("baseline")].set_index("split")["mae"]
    summary["mae_change_vs_baseline"] = summary.mae - summary.split.map(baseline_mae)
    return scores, predictions, summary, pd.concat(reference_rows, ignore_index=True)


def fit_submission(
    train: pd.DataFrame, test: pd.DataFrame, template: pd.DataFrame,
    variant: str, random_state: int, n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result_rows = []
    artifacts: dict[str, Any] = {"sj_variant": variant, "random_state": random_state,
                                "n_estimators": n_estimators, "cities": {}}
    for city in sorted(SUPPORTED_CITIES):
        historical = train.loc[train.city.eq(city)].sort_values(DATE_COLUMN).copy()
        future = test.loc[test.city.eq(city)].sort_values(DATE_COLUMN).copy()
        if future[DATE_COLUMN].min() <= historical[DATE_COLUMN].max():
            raise ValueError("Test dates must follow training dates")
        combined = pd.concat([historical, future], ignore_index=True, sort=False)
        baseline = make_features(combined)
        reference = None
        features = baseline
        if city == "sj" and variant != "baseline":
            summaries = weather_summaries(combined)
            reference = fit_climatology(combined, summaries, len(historical))
            features = experiment_features(combined, baseline, summaries, reference, variant)
        model = tree_model(city, random_state, n_estimators)
        model.fit(features.iloc[:len(historical)], historical[TARGET_COLUMN].to_numpy(float))
        rows = future[KEY_COLUMNS].copy()
        rows[TARGET_COLUMN] = _integer_cases(model.predict(features.iloc[len(historical):]))
        result_rows.append(rows)
        artifacts["cities"][city] = {"model": model, "feature_columns": features.columns.tolist(),
                                    "climatology": reference}
    predictions = pd.concat(result_rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(predictions, on=KEY_COLUMNS,
                                             how="left", validate="one_to_one")
    if submission[TARGET_COLUMN].isna().any():
        raise ValueError("Some submission keys have no predictions")
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission, artifacts


def main() -> None:
    args = parse_args()
    train, test, template = load_data(args.data_dir)
    horizon = args.horizon or int(test.city.eq("sj").sum())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores, predictions, summary, counts = run_sj_experiments(train, horizon, args)
    scores.to_csv(args.output_dir / "sj_experiment_scores.csv", index=False)
    predictions.to_csv(args.output_dir / "sj_experiment_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "sj_experiment_summary.csv", index=False)
    counts.to_csv(args.output_dir / "sj_climatology_support.csv", index=False)
    config = {**vars(args), "horizon": horizon,
              "validation_note": "Development targets precede latest evaluation targets; development windows can overlap each other. Latest period already inspected in earlier error analysis.",
              "baseline_note": "Single Poisson Extra Trees from supplied script; not the earlier 90/10 robust ensemble."}
    (args.output_dir / "sj_experiment_config.json").write_text(json.dumps(config, default=str, indent=2))
    print("\nComparison (negative MAE change means improvement):")
    print(summary.to_string(index=False))
    if args.final_variant:
        submission, artifacts = fit_submission(train, test, template, args.final_variant,
                                               args.random_state, args.n_estimators)
        submission.to_csv(args.output_dir / f"submission_sj_{args.final_variant}.csv", index=False)
        joblib.dump(artifacts, args.output_dir / f"models_sj_{args.final_variant}.joblib")


if __name__ == "__main__":
    main()
