"""Nested validation of a STAR-inspired climate-gated forest for San Juan.

The model uses a climate-only outbreak classifier as a smooth transition gate
between normal-regime and outbreak-regime Extra Trees experts. It never uses
lagged case labels at prediction time. Candidate selection is nested: each
260-week outer pseudo-test is unseen while its configuration is selected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

from nested_validate_sj import OUTER_ORIGINS, all_origins, inner_origins
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
STAR_BLEND_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
STAR_CONFIGS: dict[str, dict[str, float]] = {
    "q75_dw25": {"threshold_quantile": 0.75, "opposite_weight": 0.25},
    "q80_dw15": {"threshold_quantile": 0.80, "opposite_weight": 0.15},
    "q85_dw10": {"threshold_quantile": 0.85, "opposite_weight": 0.10},
    "q90_dw10": {"threshold_quantile": 0.90, "opposite_weight": 0.10},
}
GATE_FEATURES = [
    "week_sin_1",
    "week_cos_1",
    "week_sin_2",
    "week_cos_2",
    "station_avg_temp_c",
    "station_avg_temp_c__lag_8",
    "station_avg_temp_c__lag_12",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_specific_humidity_g_per_kg__lag_4",
    "reanalysis_specific_humidity_g_per_kg__lag_8",
    "reanalysis_specific_humidity_g_per_kg__lag_12",
    "reanalysis_relative_humidity_percent",
    "reanalysis_relative_humidity_percent__lag_8",
    "reanalysis_dew_point_temp_k__lag_8",
    "precipitation_amt_mm",
    "precipitation_amt_mm__past_mean_4",
    "precipitation_amt_mm__past_mean_8",
    "station_precip_mm__past_mean_8",
    "temperature_humidity_interaction",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    return parser.parse_args()


def _fit_regressor(
    features: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    weights: np.ndarray | None,
    random_state: int,
    n_estimators: int,
) -> tuple[Any, np.ndarray]:
    model = tree_model(SJ, random_state, n_estimators)
    kwargs: dict[str, Any] = {}
    if weights is not None:
        kwargs["extratreesregressor__sample_weight"] = weights
    model.fit(features.iloc[train_indices], target[train_indices], **kwargs)
    return model, model.predict(features.iloc[prediction_indices])


def _restore_natural_prior(
    balanced_probability: np.ndarray, prevalence: float
) -> np.ndarray:
    """Undo the 50/50 prior induced by balanced classifier weights."""
    probability = np.clip(balanced_probability, 1e-6, 1.0 - 1e-6)
    odds = probability / (1.0 - probability)
    prior_odds = prevalence / max(1.0 - prevalence, 1e-6)
    natural_odds = odds * prior_odds
    return natural_odds / (1.0 + natural_odds)


def fit_star_prediction(
    features: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    config: dict[str, float],
    random_state: int,
    n_estimators: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_target = target[train_indices]
    threshold = float(np.quantile(train_target, config["threshold_quantile"]))
    outbreak = train_target >= threshold
    gate = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=0.7,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    )
    gate.fit(features.iloc[train_indices][GATE_FEATURES], outbreak.astype(int))
    balanced_probability = gate.predict_proba(
        features.iloc[prediction_indices][GATE_FEATURES]
    )[:, 1]
    probability = _restore_natural_prior(
        balanced_probability, float(np.mean(outbreak))
    )
    opposite_weight = config["opposite_weight"]
    normal_weights = np.where(outbreak, opposite_weight, 1.0)
    outbreak_weights = np.where(outbreak, 1.0, opposite_weight)
    normal_model, normal_raw = _fit_regressor(
        features,
        target,
        train_indices,
        prediction_indices,
        normal_weights,
        random_state,
        n_estimators,
    )
    outbreak_model, outbreak_raw = _fit_regressor(
        features,
        target,
        train_indices,
        prediction_indices,
        outbreak_weights,
        random_state + 1,
        n_estimators,
    )
    raw = (1.0 - probability) * normal_raw + probability * outbreak_raw
    return raw, {
        "gate": gate,
        "normal_model": normal_model,
        "outbreak_model": outbreak_model,
        "threshold": threshold,
        "gate_probability_mean": float(np.mean(probability)),
    }


def candidate_names() -> list[str]:
    names = ["baseline"]
    for config_name in STAR_CONFIGS:
        names.extend(
            f"star__{config_name}__w{weight:g}"
            for weight in STAR_BLEND_WEIGHTS
        )
    return names


def generate_predictions(
    train: pd.DataFrame, random_state: int, n_estimators: int
) -> pd.DataFrame:
    city = (
        train.loc[train["city"].eq(SJ)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    features = make_features(city)
    target = city[TARGET_COLUMN].to_numpy(float)
    origins = all_origins(len(city))
    rows: list[pd.DataFrame] = []
    for number, origin in enumerate(origins, start=1):
        train_indices = np.arange(origin)
        validation_indices = np.arange(origin, origin + HORIZON)
        _, baseline_raw = _fit_regressor(
            features,
            target,
            train_indices,
            validation_indices,
            None,
            random_state,
            n_estimators,
        )
        result = city.iloc[validation_indices][KEY_COLUMNS + [DATE_COLUMN]].copy()
        result["origin"] = origin
        result["train_end_date"] = city.iloc[origin - 1][DATE_COLUMN]
        result["actual"] = target[validation_indices].astype(int)
        result["baseline"] = baseline_raw
        for config_name, config in STAR_CONFIGS.items():
            star_raw, details = fit_star_prediction(
                features,
                target,
                train_indices,
                validation_indices,
                config,
                random_state,
                n_estimators,
            )
            result[f"gate_mean__{config_name}"] = details[
                "gate_probability_mean"
            ]
            for weight in STAR_BLEND_WEIGHTS:
                name = f"star__{config_name}__w{weight:g}"
                result[name] = (
                    (1.0 - weight) * baseline_raw + weight * star_raw
                )
        rows.append(result)
        print(
            f"Nested STAR origin {number}/{len(origins)} "
            f"train_end={city.iloc[origin - 1][DATE_COLUMN].date()}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True)


def _metrics(frame: pd.DataFrame, candidate: str) -> dict[str, Any]:
    maes: list[float] = []
    outbreak_maes: list[float] = []
    means: list[float] = []
    maxima: list[float] = []
    for _, group in frame.groupby("origin", sort=True):
        actual = group["actual"].to_numpy(int)
        prediction = _integer_cases(group[candidate])
        outbreak = actual >= np.quantile(actual, 0.90)
        maes.append(mean_absolute_error(actual, prediction))
        outbreak_maes.append(
            mean_absolute_error(actual[outbreak], prediction[outbreak])
        )
        means.append(float(np.mean(prediction)))
        maxima.append(float(np.max(prediction)))
    return {
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "worst_mae": float(np.max(maes)),
        "mean_outbreak_mae": float(np.mean(outbreak_maes)),
        "prediction_mean": float(np.mean(means)),
        "prediction_max": float(np.max(maxima)),
        "origin_mae": np.asarray(maes),
    }


def select_inner(
    predictions: pd.DataFrame, origins: list[int]
) -> tuple[str, pd.DataFrame]:
    frame = predictions.loc[predictions["origin"].isin(origins)]
    baseline = _metrics(frame, "baseline")
    rows: list[dict[str, Any]] = []
    for candidate in candidate_names():
        metrics = _metrics(frame, candidate)
        amplitude_ok = (
            metrics["prediction_mean"] >= 0.90 * baseline["prediction_mean"]
            and metrics["prediction_mean"] <= 1.10 * baseline["prediction_mean"]
            and metrics["prediction_max"] >= 0.90 * baseline["prediction_max"]
        )
        outbreak_ok = (
            metrics["mean_outbreak_mae"] <= baseline["mean_outbreak_mae"]
        )
        origins_improved = int(
            np.sum(metrics["origin_mae"] < baseline["origin_mae"])
        )
        consistency_ok = origins_improved >= math.ceil(len(origins) / 2)
        # Optimize the competition objective (overall MAE) during selection.
        # Outbreak MAE remains visible here and is enforced in aggregate on
        # the untouched outer windows below.
        eligible = candidate == "baseline" or (
            amplitude_ok and consistency_ok
        )
        rows.append(
            {
                "candidate": candidate,
                "inner_origins": len(origins),
                **{k: v for k, v in metrics.items() if k != "origin_mae"},
                "origins_improved": origins_improved,
                "amplitude_ok": amplitude_ok,
                "outbreak_ok": outbreak_ok,
                "consistency_ok": consistency_ok,
                "eligible": eligible,
                "selection_score": metrics["mean_mae"]
                + 0.2 * metrics["std_mae"],
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "mean_mae", "candidate"]
    ).reset_index(drop=True)
    selected = str(results.loc[results["eligible"]].iloc[0]["candidate"])
    return selected, results


def nested_evaluation(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    available = sorted(predictions["origin"].unique().tolist())
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[pd.DataFrame] = []
    for outer_origin in OUTER_ORIGINS:
        origins = inner_origins(outer_origin, available)
        selected, search = select_inner(predictions, origins)
        search.insert(0, "outer_origin", outer_origin)
        search.insert(1, "selected_candidate", selected)
        inner_rows.append(search)
        outer = predictions.loc[predictions["origin"].eq(outer_origin)]
        actual = outer["actual"].to_numpy(int)
        baseline_prediction = _integer_cases(outer["baseline"])
        selected_prediction = _integer_cases(outer[selected])
        outbreak = actual >= np.quantile(actual, 0.90)
        outer_rows.append(
            {
                "outer_origin": outer_origin,
                "train_end_date": outer["train_end_date"].iloc[0],
                "inner_origins": len(origins),
                "selected_candidate": selected,
                "baseline_mae": mean_absolute_error(
                    actual, baseline_prediction
                ),
                "selected_mae": mean_absolute_error(
                    actual, selected_prediction
                ),
                "baseline_outbreak_mae": mean_absolute_error(
                    actual[outbreak], baseline_prediction[outbreak]
                ),
                "selected_outbreak_mae": mean_absolute_error(
                    actual[outbreak], selected_prediction[outbreak]
                ),
            }
        )
    outer = pd.DataFrame(outer_rows)
    outer["mae_improvement"] = outer["baseline_mae"] - outer["selected_mae"]
    outer["outbreak_improvement"] = (
        outer["baseline_outbreak_mae"] - outer["selected_outbreak_mae"]
    )
    development_candidate, final_search = select_inner(predictions, available)
    improved = int((outer["selected_mae"] < outer["baseline_mae"]).sum())
    acceptance = (
        improved >= math.ceil(0.75 * len(outer))
        and outer["selected_mae"].mean() < outer["baseline_mae"].mean()
        and outer["selected_outbreak_mae"].mean()
        <= outer["baseline_outbreak_mae"].mean()
        and outer.iloc[-1]["selected_mae"] <= outer.iloc[-1]["baseline_mae"]
    )
    summary = {
        "mean_baseline_outer_mae": float(outer["baseline_mae"].mean()),
        "mean_selected_outer_mae": float(outer["selected_mae"].mean()),
        "outer_origins_improved": improved,
        "development_candidate": development_candidate,
        "outer_acceptance_ok": bool(acceptance),
        "final_candidate": development_candidate if acceptance else "baseline",
    }
    return outer, pd.concat(inner_rows, ignore_index=True), final_search, summary


def _parse_star_candidate(candidate: str) -> dict[str, Any]:
    if candidate == "baseline":
        return {"family": "baseline"}
    parts = candidate.split("__")
    return {
        "family": "star",
        "config_name": parts[1],
        "weight": float(parts[2][1:]),
    }


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    candidate: str,
    random_state: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specification = _parse_star_candidate(candidate)
    rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {
        "candidate": candidate,
        "specification": specification,
        "cities": {},
    }
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
        train_indices = np.flatnonzero(combined["_split"].eq("train").to_numpy())
        test_indices = np.flatnonzero(combined["_split"].eq("test").to_numpy())
        features = make_features(combined)
        target = combined.iloc[train_indices][TARGET_COLUMN].to_numpy(float)
        if city == "iq":
            model = tree_model(city, random_state, n_estimators)
            model.fit(features.iloc[train_indices], target)
            raw = model.predict(features.iloc[test_indices])
            city_artifacts = {"baseline": model}
        else:
            baseline_model, baseline_raw = _fit_regressor(
                features,
                target,
                train_indices,
                test_indices,
                None,
                random_state,
                n_estimators,
            )
            city_artifacts = {"baseline": baseline_model}
            if specification["family"] == "baseline":
                raw = baseline_raw
            else:
                star_raw, star_artifacts = fit_star_prediction(
                    features,
                    target,
                    train_indices,
                    test_indices,
                    STAR_CONFIGS[specification["config_name"]],
                    random_state,
                    n_estimators,
                )
                weight = specification["weight"]
                raw = (1.0 - weight) * baseline_raw + weight * star_raw
                city_artifacts["star"] = star_artifacts
        result = combined.iloc[test_indices][KEY_COLUMNS].copy()
        result[TARGET_COLUMN] = _integer_cases(raw)
        rows.append(result)
        artifacts["cities"][city] = city_artifacts
    prediction = pd.concat(rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        prediction, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission, artifacts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, template = load_data(args.data_dir)
    predictions = generate_predictions(train, args.random_state, args.n_estimators)
    outer, inner, final_search, summary = nested_evaluation(predictions)
    predictions.to_csv(
        args.output_dir / "sj_star_origin_predictions.csv", index=False
    )
    outer.to_csv(args.output_dir / "sj_star_outer_results.csv", index=False)
    inner.to_csv(args.output_dir / "sj_star_inner_results.csv", index=False)
    final_search.to_csv(
        args.output_dir / "sj_star_final_search.csv", index=False
    )
    (args.output_dir / "sj_star_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    submission, artifacts = fit_final(
        train,
        test,
        template,
        summary["final_candidate"],
        args.random_state,
        args.n_estimators,
    )
    submission.to_csv(args.output_dir / "submission_tree_sj_star.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "tree_sj_star_models.joblib")
    print("\nNested STAR outer results:")
    print(outer.to_string(index=False))
    print("\nNested STAR summary:")
    print(json.dumps(summary, indent=2))
    print("\nFinal development search:")
    print(final_search.head(12).to_string(index=False))
    print("\nSubmission distribution:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
