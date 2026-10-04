"""Search leakage-safe climate feature groups for the confirmed tree models.

This experiment keeps the successful Random Forest / Extra Trees architecture
and changes only its inputs. Candidate groups encode mosquito-temperature
suitability, accumulated moisture, past-only seasonal anomalies, and recent
weather dynamics. Six expanding annual folds, outbreak guards, and prediction
amplitude guards select a city-specific feature set or the original fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from train_tree_ensemble import (
    DATE_COLUMN,
    DEFAULT_MODEL_PARAMS,
    KEY_COLUMNS,
    NDVI_COLUMNS,
    RAW_NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _integer_cases,
    load_data,
    make_features,
    tree_model,
)

FOLD_COUNT = 6
FEATURE_MODES = (
    "baseline",
    "biology",
    "anomaly",
    "dynamics",
    "biology_anomaly",
    "compact_biology",
    "all",
)
BIO_WINDOWS = (4, 8, 12)
ANOMALY_COLUMNS = (
    "station_avg_temp_c",
    "station_precip_mm",
    "precipitation_amt_mm",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
)
DYNAMIC_COLUMNS = (
    "station_avg_temp_c",
    "station_precip_mm",
    "precipitation_amt_mm",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "ndvi_mean",
)
PARAM_OPTIONS = {
    "iq": (
        DEFAULT_MODEL_PARAMS["iq"],
        {"max_features": 0.6, "min_samples_leaf": 7, "max_depth": None},
    ),
    "sj": (
        DEFAULT_MODEL_PARAMS["sj"],
        {"max_features": 0.7, "min_samples_leaf": 10, "max_depth": None},
    ),
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--search-estimators",
        type=int,
        default=200,
        help="Trees per fold during feature search.",
    )
    return parser.parse_args()


def _biological_features(city_frame: pd.DataFrame) -> pd.DataFrame:
    temperature = city_frame["station_avg_temp_c"].astype(float)
    humidity = city_frame["reanalysis_relative_humidity_percent"].astype(float)
    rain = city_frame["precipitation_amt_mm"].astype(float)
    station_rain = city_frame["station_precip_mm"].astype(float)
    specific_humidity = city_frame[
        "reanalysis_specific_humidity_g_per_kg"
    ].astype(float)
    output = pd.DataFrame(index=city_frame.index)
    for optimum in (25.0, 27.0, 29.0):
        output[f"temp_suitability_{optimum:g}"] = np.exp(
            -((temperature - optimum) / 3.0) ** 2
        )
    output["degree_days_above_18"] = np.maximum(temperature - 18.0, 0.0)
    output["heat_above_30"] = np.maximum(temperature - 30.0, 0.0)
    output["rain_log1p"] = np.log1p(np.maximum(rain, 0.0))
    output["station_rain_log1p"] = np.log1p(np.maximum(station_rain, 0.0))
    saturation_pressure = 0.6108 * np.exp(
        17.27 * temperature / (temperature + 237.3)
    )
    output["vapor_pressure_deficit"] = saturation_pressure * (
        1.0 - humidity / 100.0
    )
    output["warm_wet_interaction"] = (
        output["temp_suitability_27"] * output["rain_log1p"]
    )
    output["warm_humid_interaction"] = (
        output["temp_suitability_27"] * specific_humidity
    )
    output["wet_humid_interaction"] = output["rain_log1p"] * humidity
    memory_sources = {
        "rain": rain,
        "station_rain": station_rain,
        "temp_suitability": output["temp_suitability_27"],
        "warm_wet": output["warm_wet_interaction"],
        "vpd": output["vapor_pressure_deficit"],
    }
    for name, values in memory_sources.items():
        for window in BIO_WINDOWS:
            history = values.shift(1).rolling(window, min_periods=2)
            output[f"bio_{name}_past_mean_{window}"] = history.mean()
            if "rain" in name:
                output[f"bio_{name}_past_sum_{window}"] = history.sum()
    return output


def _past_seasonal_anomalies(city_frame: pd.DataFrame) -> pd.DataFrame:
    """Compare weather with previous years at the same week, never future rows."""
    output = pd.DataFrame(index=city_frame.index)
    week = city_frame["weekofyear"]
    for column in ANOMALY_COLUMNS:
        values = city_frame[column].astype(float)
        past_mean = values.groupby(week).transform(
            lambda group: group.shift(1).expanding(min_periods=2).mean()
        )
        past_std = values.groupby(week).transform(
            lambda group: group.shift(1).expanding(min_periods=3).std()
        )
        output[f"{column}__seasonal_anomaly"] = values - past_mean
        output[f"{column}__seasonal_z"] = (values - past_mean) / past_std.clip(
            lower=1e-3
        )
    return output


def _dynamic_features(
    city_frame: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    output = pd.DataFrame(index=city_frame.index)
    source: dict[str, pd.Series] = {
        column: city_frame[column].astype(float)
        for column in DYNAMIC_COLUMNS
        if column != "ndvi_mean"
    }
    source["ndvi_mean"] = city_frame[NDVI_COLUMNS].mean(axis=1)
    for name, values in source.items():
        output[f"{name}__change_1"] = values.diff(1)
        output[f"{name}__change_4"] = values.diff(4)
        for window in (4, 8, 12):
            past = values.shift(1)
            output[f"{name}__past_std_{window}"] = past.rolling(
                window, min_periods=2
            ).std()
            output[f"{name}__ewm_{window}"] = past.ewm(
                span=window, adjust=False, min_periods=2
            ).mean()
    output["rain_change_x_temperature"] = (
        output["precipitation_amt_mm__change_4"]
        * city_frame["station_avg_temp_c"].astype(float)
    )
    return output


def _compact_baseline(baseline: pd.DataFrame) -> pd.DataFrame:
    keep: list[str] = []
    memory_names = {
        "station_avg_temp_c",
        "station_precip_mm",
        "precipitation_amt_mm",
        "reanalysis_relative_humidity_percent",
        "reanalysis_specific_humidity_g_per_kg",
        "reanalysis_dew_point_temp_k",
    }
    for column in baseline.columns:
        if "__" not in column:
            keep.append(column)
            continue
        source, suffix = column.split("__", maxsplit=1)
        if source not in memory_names:
            continue
        if suffix in {
            "lag_2",
            "lag_4",
            "lag_8",
            "lag_12",
            "past_mean_4",
            "past_mean_8",
            "past_mean_12",
        }:
            keep.append(column)
    return baseline[keep].copy()


def make_feature_mode(city_frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode: {mode}")
    baseline = make_features(city_frame)
    biological = _biological_features(city_frame)
    if mode == "baseline":
        return baseline
    if mode == "biology":
        return pd.concat([baseline, biological], axis=1)
    if mode == "anomaly":
        return pd.concat([baseline, _past_seasonal_anomalies(city_frame)], axis=1)
    if mode == "dynamics":
        return pd.concat([baseline, _dynamic_features(city_frame, baseline)], axis=1)
    if mode == "biology_anomaly":
        return pd.concat(
            [baseline, biological, _past_seasonal_anomalies(city_frame)], axis=1
        )
    if mode == "compact_biology":
        return pd.concat([_compact_baseline(baseline), biological], axis=1)
    return pd.concat(
        [
            baseline,
            biological,
            _past_seasonal_anomalies(city_frame),
            _dynamic_features(city_frame, baseline),
        ],
        axis=1,
    )


def _evaluate(
    city: str,
    city_frame: pd.DataFrame,
    features: pd.DataFrame,
    params: dict[str, Any],
    random_state: int,
    n_estimators: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    target = city_frame[TARGET_COLUMN].to_numpy(float)
    starts = list(range(len(city_frame) - FOLD_COUNT * 52, len(city_frame), 52))
    rows: list[pd.DataFrame] = []
    fold_mae: list[float] = []
    for fold, start in enumerate(starts, start=1):
        train_index = np.arange(start)
        validation_index = np.arange(start, start + 52)
        model = tree_model(city, random_state, n_estimators, params)
        model.fit(features.iloc[train_index], target[train_index])
        raw_prediction = model.predict(features.iloc[validation_index])
        prediction = _integer_cases(raw_prediction)
        actual = target[validation_index].astype(int)
        fold_mae.append(mean_absolute_error(actual, prediction))
        fold_rows = city_frame.iloc[validation_index][KEY_COLUMNS + [DATE_COLUMN]].copy()
        fold_rows["fold"] = fold
        fold_rows["actual"] = actual
        fold_rows["raw_prediction"] = raw_prediction
        fold_rows["prediction"] = prediction
        rows.append(fold_rows)
    predictions = pd.concat(rows, ignore_index=True)
    metrics = _prediction_metrics(predictions, features.shape[1])
    return metrics, predictions


def _prediction_metrics(
    predictions: pd.DataFrame, n_features: int
) -> dict[str, float]:
    actual = predictions["actual"].to_numpy(int)
    prediction = predictions["prediction"].to_numpy(int)
    fold_mae = [
        mean_absolute_error(group["actual"], group["prediction"])
        for _, group in predictions.groupby("fold")
    ]
    outbreak_threshold = float(np.quantile(actual, 0.90))
    outbreak = actual >= outbreak_threshold
    metrics = {
        "mae_6fold": mean_absolute_error(actual, prediction),
        "mae_recent_4fold": mean_absolute_error(actual[-4 * 52 :], prediction[-4 * 52 :]),
        "mean_fold_mae": float(np.mean(fold_mae)),
        "std_fold_mae": float(np.std(fold_mae)),
        "robust_score": float(np.mean(fold_mae) + 0.2 * np.std(fold_mae)),
        "outbreak_mae": mean_absolute_error(actual[outbreak], prediction[outbreak]),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_max": float(np.max(prediction)),
        "n_features": int(n_features),
    }
    return metrics


def search_features(
    train: pd.DataFrame,
    random_state: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    selected_predictions: list[pd.DataFrame] = []
    for city in ("iq", "sj"):
        city_frame = (
            train.loc[train["city"].eq(city)]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
        feature_cache = {
            mode: make_feature_mode(city_frame, mode) for mode in FEATURE_MODES
        }
        baseline_metrics, baseline_predictions = _evaluate(
            city,
            city_frame,
            feature_cache["baseline"],
            DEFAULT_MODEL_PARAMS[city],
            random_state,
            n_estimators,
        )
        trial = 0
        prediction_cache: dict[tuple[str, int, float], pd.DataFrame] = {}
        for mode in FEATURE_MODES:
            for param_index, params in enumerate(PARAM_OPTIONS[city]):
                trial += 1
                metrics, predictions = _evaluate(
                    city,
                    city_frame,
                    feature_cache[mode],
                    params,
                    random_state,
                    n_estimators,
                )
                prediction_cache[(mode, param_index, 1.0)] = predictions
                amplitude_ok = (
                    metrics["prediction_mean"]
                    >= 0.90 * baseline_metrics["prediction_mean"]
                    and metrics["prediction_mean"]
                    <= 1.10 * baseline_metrics["prediction_mean"]
                    and metrics["prediction_max"]
                    >= 0.90 * baseline_metrics["prediction_max"]
                )
                outbreak_ok = (
                    metrics["outbreak_mae"]
                    <= 1.03 * baseline_metrics["outbreak_mae"]
                )
                relative_score = (
                    0.45 * metrics["mae_6fold"] / baseline_metrics["mae_6fold"]
                    + 0.35
                    * metrics["mae_recent_4fold"]
                    / baseline_metrics["mae_recent_4fold"]
                    + 0.20
                    * metrics["robust_score"]
                    / baseline_metrics["robust_score"]
                )
                result_rows.append(
                    {
                        "city": city,
                        "trial": trial,
                        "feature_mode": mode,
                        "param_index": param_index,
                        "feature_weight": 1.0,
                        **params,
                        **metrics,
                        "amplitude_ok": amplitude_ok,
                        "outbreak_ok": outbreak_ok,
                        "eligible": amplitude_ok and outbreak_ok,
                        "selection_score": relative_score,
                    }
                )
                print(
                    f"{city.upper()} {trial:02d}/{len(FEATURE_MODES) * len(PARAM_OPTIONS[city]):02d} "
                    f"mode={mode:<18} mae6={metrics['mae_6fold']:.3f} "
                    f"recent4={metrics['mae_recent_4fold']:.3f} "
                    f"eligible={amplitude_ok and outbreak_ok}",
                    flush=True,
                )
        # Some useful compact feature sets underpredict only the largest peaks.
        # Test conservative blends with the confirmed baseline prediction.
        direct_candidates = list(prediction_cache.items())
        for (mode, param_index, _), predictions in direct_candidates:
            if mode == "baseline" and param_index == 0:
                continue
            for weight in (0.25, 0.5, 0.75):
                trial += 1
                blended = baseline_predictions.copy()
                blended["raw_prediction"] = (
                    (1.0 - weight) * baseline_predictions["raw_prediction"]
                    + weight * predictions["raw_prediction"]
                )
                blended["prediction"] = _integer_cases(blended["raw_prediction"])
                metrics = _prediction_metrics(
                    blended, feature_cache[mode].shape[1]
                )
                prediction_cache[(mode, param_index, weight)] = blended
                amplitude_ok = (
                    metrics["prediction_mean"]
                    >= 0.90 * baseline_metrics["prediction_mean"]
                    and metrics["prediction_mean"]
                    <= 1.10 * baseline_metrics["prediction_mean"]
                    and metrics["prediction_max"]
                    >= 0.90 * baseline_metrics["prediction_max"]
                )
                outbreak_ok = (
                    metrics["outbreak_mae"]
                    <= 1.03 * baseline_metrics["outbreak_mae"]
                )
                relative_score = (
                    0.45 * metrics["mae_6fold"] / baseline_metrics["mae_6fold"]
                    + 0.35
                    * metrics["mae_recent_4fold"]
                    / baseline_metrics["mae_recent_4fold"]
                    + 0.20
                    * metrics["robust_score"]
                    / baseline_metrics["robust_score"]
                )
                result_rows.append(
                    {
                        "city": city,
                        "trial": trial,
                        "feature_mode": mode,
                        "param_index": param_index,
                        "feature_weight": weight,
                        **PARAM_OPTIONS[city][param_index],
                        **metrics,
                        "amplitude_ok": amplitude_ok,
                        "outbreak_ok": outbreak_ok,
                        "eligible": amplitude_ok and outbreak_ok,
                        "selection_score": relative_score,
                    }
                )
        city_results = pd.DataFrame(
            row for row in result_rows if row["city"] == city
        )
        winner = city_results.loc[city_results["eligible"]].sort_values(
            ["selection_score", "mae_6fold", "mae_recent_4fold", "trial"]
        ).iloc[0]
        mode = str(winner["feature_mode"])
        param_index = int(winner["param_index"])
        feature_weight = float(winner["feature_weight"])
        selected[city] = {
            "feature_mode": mode,
            "feature_weight": feature_weight,
            "model_params": PARAM_OPTIONS[city][param_index],
        }
        winner_predictions = prediction_cache[
            (mode, param_index, feature_weight)
        ].copy()
        winner_predictions["feature_mode"] = mode
        selected_predictions.append(winner_predictions)
    return pd.DataFrame(result_rows), selected, pd.concat(selected_predictions)


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
    random_state: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {"selection": selected, "cities": {}}
    for city in ("iq", "sj"):
        train_city = train.loc[train["city"].eq(city)].copy()
        test_city = test.loc[test["city"].eq(city)].copy()
        train_city["_split"] = "train"
        test_city["_split"] = "test"
        combined = (
            pd.concat([train_city, test_city], ignore_index=True, sort=False)
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
        train_index = np.flatnonzero(combined["_split"].eq("train").to_numpy())
        test_index = np.flatnonzero(combined["_split"].eq("test").to_numpy())
        choice = selected[city]
        features = make_feature_mode(combined, choice["feature_mode"])
        target = combined.iloc[train_index][TARGET_COLUMN].to_numpy(float)
        model = tree_model(
            city,
            random_state,
            n_estimators,
            choice["model_params"],
        )
        model.fit(features.iloc[train_index], target)
        candidate_raw = model.predict(features.iloc[test_index])
        feature_weight = choice["feature_weight"]
        baseline_model = None
        if feature_weight < 1.0:
            baseline_features = make_feature_mode(combined, "baseline")
            baseline_model = tree_model(
                city,
                random_state,
                n_estimators,
                DEFAULT_MODEL_PARAMS[city],
            )
            baseline_model.fit(baseline_features.iloc[train_index], target)
            raw_prediction = (
                (1.0 - feature_weight)
                * baseline_model.predict(baseline_features.iloc[test_index])
                + feature_weight * candidate_raw
            )
        else:
            raw_prediction = candidate_raw
        result = combined.iloc[test_index][KEY_COLUMNS].copy()
        result[TARGET_COLUMN] = _integer_cases(raw_prediction)
        rows.append(result)
        artifacts["cities"][city] = {
            "model": model,
            "feature_mode": choice["feature_mode"],
            "feature_columns": features.columns.tolist(),
            "feature_weight": feature_weight,
            "baseline_model": baseline_model,
        }
    predictions = pd.concat(rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        predictions, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission, artifacts


def confirm_selected(
    train: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
    random_state: int,
    n_estimators: int,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Recheck search winners using the full final number of trees."""
    confirmed: dict[str, dict[str, Any]] = {}
    score_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for city in ("iq", "sj"):
        city_frame = (
            train.loc[train["city"].eq(city)]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
        baseline_metrics, baseline_predictions = _evaluate(
            city,
            city_frame,
            make_feature_mode(city_frame, "baseline"),
            DEFAULT_MODEL_PARAMS[city],
            random_state,
            n_estimators,
        )
        choice = selected[city]
        if (
            choice["feature_mode"] == "baseline"
            and choice["feature_weight"] == 1.0
            and choice["model_params"] == DEFAULT_MODEL_PARAMS[city]
        ):
            candidate_metrics = baseline_metrics
            candidate_predictions = baseline_predictions
        else:
            direct_metrics, direct_predictions = _evaluate(
                city,
                city_frame,
                make_feature_mode(city_frame, choice["feature_mode"]),
                choice["model_params"],
                random_state,
                n_estimators,
            )
            weight = choice["feature_weight"]
            if weight < 1.0:
                candidate_predictions = baseline_predictions.copy()
                candidate_predictions["raw_prediction"] = (
                    (1.0 - weight) * baseline_predictions["raw_prediction"]
                    + weight * direct_predictions["raw_prediction"]
                )
                candidate_predictions["prediction"] = _integer_cases(
                    candidate_predictions["raw_prediction"]
                )
                candidate_metrics = _prediction_metrics(
                    candidate_predictions, direct_metrics["n_features"]
                )
            else:
                candidate_metrics = direct_metrics
                candidate_predictions = direct_predictions
        amplitude_ok = (
            candidate_metrics["prediction_mean"]
            >= 0.90 * baseline_metrics["prediction_mean"]
            and candidate_metrics["prediction_mean"]
            <= 1.10 * baseline_metrics["prediction_mean"]
            and candidate_metrics["prediction_max"]
            >= 0.90 * baseline_metrics["prediction_max"]
        )
        outbreak_ok = (
            candidate_metrics["outbreak_mae"]
            <= 1.03 * baseline_metrics["outbreak_mae"]
        )
        improvement_ok = (
            candidate_metrics["mae_6fold"] <= baseline_metrics["mae_6fold"]
            and candidate_metrics["robust_score"] <= baseline_metrics["robust_score"]
        )
        if amplitude_ok and outbreak_ok and improvement_ok:
            confirmed[city] = choice
            final_metrics = candidate_metrics
            final_predictions = candidate_predictions
        else:
            confirmed[city] = {
                "feature_mode": "baseline",
                "feature_weight": 1.0,
                "model_params": DEFAULT_MODEL_PARAMS[city],
            }
            final_metrics = baseline_metrics
            final_predictions = baseline_predictions
        score_rows.extend(
            [
                {"city": city, "model": "baseline", **baseline_metrics},
                {
                    "city": city,
                    "model": "selected_confirmed",
                    **final_metrics,
                },
            ]
        )
        final_predictions = final_predictions.copy()
        final_predictions["feature_mode"] = confirmed[city]["feature_mode"]
        final_predictions["feature_weight"] = confirmed[city]["feature_weight"]
        prediction_rows.append(final_predictions)
    return confirmed, pd.DataFrame(score_rows), pd.concat(prediction_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, template = load_data(args.data_dir)
    results, selected, validation = search_features(
        train, args.random_state, args.search_estimators
    )
    results.to_csv(args.output_dir / "feature_search_results.csv", index=False)
    selected, validation_scores, validation = confirm_selected(
        train, selected, args.random_state, args.n_estimators
    )
    validation_scores.to_csv(
        args.output_dir / "feature_validation_scores.csv", index=False
    )
    validation.to_csv(
        args.output_dir / "feature_validation_predictions.csv", index=False
    )
    (args.output_dir / "feature_selected_config.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    submission, artifacts = fit_final(
        train,
        test,
        template,
        selected,
        args.random_state,
        args.n_estimators,
    )
    submission.to_csv(args.output_dir / "submission_tree_features.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "tree_feature_models.joblib")
    print("\nSelected feature configurations:")
    print(json.dumps(selected, indent=2))
    print("\nSubmission distribution:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
