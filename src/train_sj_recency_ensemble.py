"""Tune an SJ recency ensemble on rolling five-year forecast horizons.

The competition requires 260 consecutive SJ predictions. This experiment uses
six expanding origins with matching 260-week validation horizons, compares the
confirmed full-history Extra Trees model with rolling-window and exponentially
time-decayed variants, and searches conservative blends. IQ remains unchanged.
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
    KEY_COLUMNS,
    TARGET_COLUMN,
    _integer_cases,
    load_data,
    make_features,
    tree_model,
)

SJ = "sj"
HORIZON = 260
MINIMUM_TRAIN_WEEKS = 416
ORIGIN_STEP = 52
BLEND_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
RECENCY_CONFIGS: dict[str, dict[str, Any]] = {
    "window_7y": {"window_weeks": 7 * 52, "half_life_years": None},
    "window_10y": {"window_weeks": 10 * 52, "half_life_years": None},
    "window_12y": {"window_weeks": 12 * 52, "half_life_years": None},
    "decay_3y": {"window_weeks": None, "half_life_years": 3.0},
    "decay_5y": {"window_weeks": None, "half_life_years": 5.0},
    "decay_8y": {"window_weeks": None, "half_life_years": 8.0},
    "decay_12y": {"window_weeks": None, "half_life_years": 12.0},
    "window_10y_decay_5y": {"window_weeks": 10 * 52, "half_life_years": 5.0},
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
        default=300,
        help="Trees per model during long-horizon search.",
    )
    return parser.parse_args()


def rolling_origins(n_rows: int) -> list[int]:
    last_origin = n_rows - HORIZON
    if last_origin < MINIMUM_TRAIN_WEEKS:
        raise ValueError("Not enough SJ rows for a 260-week rolling validation")
    return list(range(MINIMUM_TRAIN_WEEKS, last_origin + 1, ORIGIN_STEP))


def recency_training_indices_and_weights(
    origin: int, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray | None]:
    window = config["window_weeks"]
    start = 0 if window is None else max(0, origin - int(window))
    indices = np.arange(start, origin)
    half_life = config["half_life_years"]
    if half_life is None:
        return indices, None
    age_weeks = origin - 1 - indices
    weights = np.power(0.5, age_weeks / (float(half_life) * 52.0))
    return indices, weights


def _fit_sj_tree(
    features: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    sample_weight: np.ndarray | None,
    random_state: int,
    n_estimators: int,
) -> tuple[Any, np.ndarray]:
    model = tree_model(SJ, random_state, n_estimators)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["extratreesregressor__sample_weight"] = sample_weight
    model.fit(features.iloc[train_indices], target[train_indices], **fit_kwargs)
    return model, model.predict(features.iloc[validation_indices])


def generate_long_horizon_predictions(
    train: pd.DataFrame,
    random_state: int,
    n_estimators: int,
) -> pd.DataFrame:
    city = (
        train.loc[train["city"].eq(SJ)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    features = make_features(city)
    target = city[TARGET_COLUMN].to_numpy(float)
    rows: list[pd.DataFrame] = []
    origins = rolling_origins(len(city))
    for origin_number, origin in enumerate(origins, start=1):
        validation_indices = np.arange(origin, origin + HORIZON)
        baseline_indices = np.arange(origin)
        _, baseline_raw = _fit_sj_tree(
            features,
            target,
            baseline_indices,
            validation_indices,
            None,
            random_state,
            n_estimators,
        )
        result = city.iloc[validation_indices][KEY_COLUMNS + [DATE_COLUMN]].copy()
        result["origin"] = origin_number
        result["train_end_date"] = city.iloc[origin - 1][DATE_COLUMN]
        result["actual"] = target[validation_indices].astype(int)
        result["baseline_raw"] = baseline_raw
        for config_name, config in RECENCY_CONFIGS.items():
            train_indices, weights = recency_training_indices_and_weights(
                origin, config
            )
            _, candidate_raw = _fit_sj_tree(
                features,
                target,
                train_indices,
                validation_indices,
                weights,
                random_state,
                n_estimators,
            )
            result[config_name] = candidate_raw
        rows.append(result)
        print(
            f"SJ 260-week origin {origin_number}/{len(origins)} "
            f"train_end={city.iloc[origin - 1][DATE_COLUMN].date()}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True)


def _metrics(frame: pd.DataFrame, raw_prediction: np.ndarray) -> dict[str, Any]:
    prediction = _integer_cases(raw_prediction)
    actual = frame["actual"].to_numpy(int)
    origin_frame = frame[["origin", "actual"]].copy()
    origin_frame["prediction"] = prediction
    origin_mae = np.array(
        [
            mean_absolute_error(group["actual"], group["prediction"])
            for _, group in origin_frame.groupby("origin")
        ]
    )
    recent = frame["origin"].eq(frame["origin"].max()).to_numpy()
    outbreak_threshold = float(np.quantile(actual, 0.90))
    outbreak = actual >= outbreak_threshold
    recent_threshold = float(np.quantile(actual[recent], 0.90))
    recent_outbreak = recent & (actual >= recent_threshold)
    return {
        "mean_origin_mae": float(origin_mae.mean()),
        "std_origin_mae": float(origin_mae.std()),
        "worst_origin_mae": float(origin_mae.max()),
        "recent_origin_mae": mean_absolute_error(
            actual[recent], prediction[recent]
        ),
        "pooled_mae": mean_absolute_error(actual, prediction),
        "outbreak_mae": mean_absolute_error(
            actual[outbreak], prediction[outbreak]
        ),
        "recent_outbreak_mae": mean_absolute_error(
            actual[recent_outbreak], prediction[recent_outbreak]
        ),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_max": float(np.max(prediction)),
        "origins_improved": 0,
        "origin_mae": origin_mae,
    }


def select_recency_blend(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_raw = predictions["baseline_raw"].to_numpy(float)
    baseline = _metrics(predictions, baseline_raw)
    rows: list[dict[str, Any]] = [
        {
            "candidate": "baseline",
            "recency_weight": 0.0,
            **{k: v for k, v in baseline.items() if k != "origin_mae"},
            "amplitude_ok": True,
            "outbreak_ok": True,
            "eligible": True,
            "selection_score": 1.0,
            "improvement_vs_baseline": 0.0,
        }
    ]
    for config_name in RECENCY_CONFIGS:
        candidate_raw = predictions[config_name].to_numpy(float)
        for weight in BLEND_WEIGHTS:
            raw = (1.0 - weight) * baseline_raw + weight * candidate_raw
            metrics = _metrics(predictions, raw)
            metrics["origins_improved"] = int(
                np.sum(metrics["origin_mae"] < baseline["origin_mae"])
            )
            amplitude_ok = (
                metrics["prediction_mean"] >= 0.90 * baseline["prediction_mean"]
                and metrics["prediction_mean"]
                <= 1.10 * baseline["prediction_mean"]
                and metrics["prediction_max"] >= 0.90 * baseline["prediction_max"]
            )
            outbreak_ok = (
                metrics["outbreak_mae"] <= 1.03 * baseline["outbreak_mae"]
                and metrics["recent_outbreak_mae"]
                <= 1.05 * baseline["recent_outbreak_mae"]
            )
            stability_ok = (
                metrics["worst_origin_mae"] <= 1.03 * baseline["worst_origin_mae"]
                and metrics["origins_improved"] >= len(baseline["origin_mae"]) // 2
            )
            score = (
                0.60
                * metrics["mean_origin_mae"]
                / baseline["mean_origin_mae"]
                + 0.30
                * metrics["recent_origin_mae"]
                / baseline["recent_origin_mae"]
                + 0.10
                * metrics["std_origin_mae"]
                / baseline["std_origin_mae"]
            )
            rows.append(
                {
                    "candidate": config_name,
                    "recency_weight": weight,
                    **{k: v for k, v in metrics.items() if k != "origin_mae"},
                    "amplitude_ok": amplitude_ok,
                    "outbreak_ok": outbreak_ok,
                    "stability_ok": stability_ok,
                    "eligible": amplitude_ok and outbreak_ok and stability_ok,
                    "selection_score": score,
                    "improvement_vs_baseline": baseline["mean_origin_mae"]
                    - metrics["mean_origin_mae"],
                }
            )
    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "mean_origin_mae", "recent_origin_mae"]
    ).reset_index(drop=True)
    winner = results.loc[results["eligible"]].iloc[0]
    if not (
        winner["mean_origin_mae"] < baseline["mean_origin_mae"]
        and winner["recent_origin_mae"] <= baseline["recent_origin_mae"]
    ):
        selected = {"candidate": "baseline", "recency_weight": 0.0}
    else:
        selected = {
            "candidate": str(winner["candidate"]),
            "recency_weight": float(winner["recency_weight"]),
        }
    selected["baseline_mean_origin_mae"] = baseline["mean_origin_mae"]
    selected["baseline_recent_origin_mae"] = baseline["recent_origin_mae"]
    return results, selected


def selected_validation_predictions(
    predictions: pd.DataFrame, selected: dict[str, Any]
) -> pd.DataFrame:
    output = predictions[
        KEY_COLUMNS
        + [DATE_COLUMN, "origin", "train_end_date", "actual", "baseline_raw"]
    ].copy()
    if selected["candidate"] == "baseline":
        raw = predictions["baseline_raw"].to_numpy(float)
        output["recency_raw"] = raw
    else:
        recency = predictions[selected["candidate"]].to_numpy(float)
        weight = selected["recency_weight"]
        raw = (
            (1.0 - weight) * predictions["baseline_raw"].to_numpy(float)
            + weight * recency
        )
        output["recency_raw"] = recency
    output["baseline_prediction"] = _integer_cases(output["baseline_raw"])
    output["ensemble_prediction"] = _integer_cases(raw)
    return output


def _fit_final_sj_models(
    train_city: pd.DataFrame,
    test_city: pd.DataFrame,
    selected: dict[str, Any],
    random_state: int,
    n_estimators: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_city = train_city.copy()
    test_city = test_city.copy()
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
    target = combined.iloc[train_indices][TARGET_COLUMN].to_numpy(float)
    baseline, baseline_raw = _fit_sj_tree(
        features,
        target,
        train_indices,
        test_indices,
        None,
        random_state,
        n_estimators,
    )
    artifacts: dict[str, Any] = {"baseline": baseline}
    if selected["candidate"] == "baseline":
        return baseline_raw, artifacts
    config = RECENCY_CONFIGS[selected["candidate"]]
    candidate_indices, weights = recency_training_indices_and_weights(
        len(train_indices), config
    )
    recency_model, recency_raw = _fit_sj_tree(
        features,
        target,
        candidate_indices,
        test_indices,
        weights,
        random_state,
        n_estimators,
    )
    weight = selected["recency_weight"]
    artifacts["recency"] = recency_model
    artifacts["recency_config"] = config
    return (1.0 - weight) * baseline_raw + weight * recency_raw, artifacts


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    selected: dict[str, Any],
    random_state: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {"selection": selected, "cities": {}}
    for city in ("iq", "sj"):
        train_city = train.loc[train["city"].eq(city)].copy()
        test_city = test.loc[test["city"].eq(city)].copy()
        if city == SJ:
            raw, city_artifacts = _fit_final_sj_models(
                train_city,
                test_city,
                selected,
                random_state,
                n_estimators,
            )
            result = test_city.sort_values(DATE_COLUMN)[KEY_COLUMNS].copy()
        else:
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
            target = combined.iloc[train_indices][TARGET_COLUMN].to_numpy(float)
            model = tree_model(city, random_state, n_estimators)
            model.fit(features.iloc[train_indices], target)
            raw = model.predict(features.iloc[test_indices])
            result = combined.iloc[test_indices][KEY_COLUMNS].copy()
            city_artifacts = {"baseline": model}
        result[TARGET_COLUMN] = _integer_cases(raw)
        rows.append(result)
        artifacts["cities"][city] = city_artifacts
    predicted = pd.concat(rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        predicted, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission, artifacts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, template = load_data(args.data_dir)
    predictions = generate_long_horizon_predictions(
        train, args.random_state, args.search_estimators
    )
    search, selected = select_recency_blend(predictions)
    validation = selected_validation_predictions(predictions, selected)
    search.to_csv(args.output_dir / "sj_recency_search_results.csv", index=False)
    validation.to_csv(
        args.output_dir / "sj_recency_validation_predictions.csv", index=False
    )
    (args.output_dir / "sj_recency_selected_config.json").write_text(
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
    submission.to_csv(args.output_dir / "submission_tree_sj_recency.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "tree_sj_recency_models.joblib")
    print("\nSelected SJ recency configuration:")
    print(json.dumps(selected, indent=2))
    print("\nTop long-horizon candidates:")
    print(search.head(12).to_string(index=False))
    print("\nSubmission distribution:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
