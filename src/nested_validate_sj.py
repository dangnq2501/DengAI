"""Nested rolling-origin validation for San Juan DengAI models.

Each outer pseudo-test contains 260 future weeks. Model selection for that outer
origin uses only earlier, fully completed 260-week inner blocks. Outer labels do
not influence candidate selection. The fixed candidate registry contains the
confirmed Extra Trees model, recency-weighted/windowed variants, and conservative
across-tree quantile aggregations.
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
from sklearn.metrics import mean_absolute_error

from train_sj_quantile_forest import per_tree_predictions
from train_sj_recency_ensemble import (
    RECENCY_CONFIGS,
    recency_training_indices_and_weights,
)
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
FIRST_ORIGIN = 260
ORIGIN_STEP = 52
OUTER_ORIGINS = (520, 572, 624, 676)
RECENCY_BLEND_WEIGHTS = (0.5, 0.75, 1.0)
QUANTILE_OPTIONS = (
    (0.50, 0.25),
    (0.55, 0.25),
    (0.60, 0.25),
    (0.65, 0.25),
    (0.50, 0.50),
    (0.55, 0.50),
    (0.60, 0.50),
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    return parser.parse_args()


def all_origins(n_rows: int) -> list[int]:
    return list(range(FIRST_ORIGIN, n_rows - HORIZON + 1, ORIGIN_STEP))


def inner_origins(outer_origin: int, available_origins: list[int]) -> list[int]:
    """Return inner origins whose entire validation block precedes outer test."""
    origins = [
        origin
        for origin in available_origins
        if origin + HORIZON <= outer_origin
    ]
    if not origins:
        raise ValueError(f"Outer origin {outer_origin} has no completed inner block")
    return origins


def candidate_names() -> list[str]:
    names = ["baseline"]
    for config_name in RECENCY_CONFIGS:
        names.extend(
            f"recency__{config_name}__w{weight:g}"
            for weight in RECENCY_BLEND_WEIGHTS
        )
    names.extend(
        f"quantile__q{quantile:g}__w{weight:g}"
        for quantile, weight in QUANTILE_OPTIONS
    )
    return names


def _fit_sj(
    features: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    random_state: int,
    n_estimators: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, np.ndarray]:
    model = tree_model(SJ, random_state, n_estimators)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["extratreesregressor__sample_weight"] = sample_weight
    model.fit(features.iloc[train_indices], target[train_indices], **fit_kwargs)
    return model, model.predict(features.iloc[validation_indices])


def generate_origin_predictions(
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
        validation_indices = np.arange(origin, origin + HORIZON)
        baseline_model, baseline_raw = _fit_sj(
            features,
            target,
            np.arange(origin),
            validation_indices,
            random_state,
            n_estimators,
        )
        tree_raw = per_tree_predictions(
            baseline_model, features.iloc[validation_indices]
        )
        result = city.iloc[validation_indices][KEY_COLUMNS + [DATE_COLUMN]].copy()
        result["origin"] = origin
        result["train_end_date"] = city.iloc[origin - 1][DATE_COLUMN]
        result["actual"] = target[validation_indices].astype(int)
        result["baseline"] = baseline_raw
        for quantile, weight in QUANTILE_OPTIONS:
            quantile_raw = np.quantile(tree_raw, quantile, axis=0)
            name = f"quantile__q{quantile:g}__w{weight:g}"
            result[name] = (1.0 - weight) * baseline_raw + weight * quantile_raw
        for config_name, config in RECENCY_CONFIGS.items():
            train_indices, sample_weight = recency_training_indices_and_weights(
                origin, config
            )
            _, recency_raw = _fit_sj(
                features,
                target,
                train_indices,
                validation_indices,
                random_state,
                n_estimators,
                sample_weight,
            )
            for weight in RECENCY_BLEND_WEIGHTS:
                name = f"recency__{config_name}__w{weight:g}"
                result[name] = (
                    (1.0 - weight) * baseline_raw + weight * recency_raw
                )
        rows.append(result)
        print(
            f"Nested SJ origin {number}/{len(origins)} "
            f"train_end={city.iloc[origin - 1][DATE_COLUMN].date()}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True)


def _candidate_metrics(
    predictions: pd.DataFrame, candidate: str
) -> dict[str, Any]:
    origin_mae: list[float] = []
    outbreak_mae: list[float] = []
    prediction_means: list[float] = []
    prediction_maxima: list[float] = []
    for _, group in predictions.groupby("origin", sort=True):
        actual = group["actual"].to_numpy(int)
        prediction = _integer_cases(group[candidate])
        threshold = float(np.quantile(actual, 0.90))
        outbreak = actual >= threshold
        origin_mae.append(mean_absolute_error(actual, prediction))
        outbreak_mae.append(
            mean_absolute_error(actual[outbreak], prediction[outbreak])
        )
        prediction_means.append(float(np.mean(prediction)))
        prediction_maxima.append(float(np.max(prediction)))
    return {
        "mean_mae": float(np.mean(origin_mae)),
        "std_mae": float(np.std(origin_mae)),
        "worst_mae": float(np.max(origin_mae)),
        "mean_outbreak_mae": float(np.mean(outbreak_mae)),
        "prediction_mean": float(np.mean(prediction_means)),
        "prediction_max": float(np.max(prediction_maxima)),
        "origin_mae": np.asarray(origin_mae),
    }


def select_from_inner(
    predictions: pd.DataFrame, origins: list[int]
) -> tuple[str, pd.DataFrame]:
    inner = predictions.loc[predictions["origin"].isin(origins)]
    baseline = _candidate_metrics(inner, "baseline")
    rows: list[dict[str, Any]] = []
    for candidate in candidate_names():
        metrics = _candidate_metrics(inner, candidate)
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
        eligible = candidate == "baseline" or (
            amplitude_ok and outbreak_ok and consistency_ok
        )
        score = metrics["mean_mae"] + 0.20 * metrics["std_mae"]
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
                "selection_score": score,
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "mean_mae", "candidate"]
    ).reset_index(drop=True)
    selected = str(results.loc[results["eligible"]].iloc[0]["candidate"])
    return selected, results


def evaluate_outer(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    available = sorted(predictions["origin"].unique().tolist())
    selection_rows: list[pd.DataFrame] = []
    outer_rows: list[dict[str, Any]] = []
    for outer_origin in OUTER_ORIGINS:
        origins = inner_origins(outer_origin, available)
        selected, inner_results = select_from_inner(predictions, origins)
        inner_results.insert(0, "outer_origin", outer_origin)
        inner_results.insert(1, "selected_candidate", selected)
        selection_rows.append(inner_results)
        outer = predictions.loc[predictions["origin"].eq(outer_origin)]
        actual = outer["actual"].to_numpy(int)
        baseline_prediction = _integer_cases(outer["baseline"])
        selected_prediction = _integer_cases(outer[selected])
        threshold = float(np.quantile(actual, 0.90))
        outbreak = actual >= threshold
        outer_rows.append(
            {
                "outer_origin": outer_origin,
                "train_end_date": outer["train_end_date"].iloc[0],
                "selected_candidate": selected,
                "inner_origins": len(origins),
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
                "baseline_mean": float(np.mean(baseline_prediction)),
                "selected_mean": float(np.mean(selected_prediction)),
                "actual_mean": float(np.mean(actual)),
                "baseline_max": int(np.max(baseline_prediction)),
                "selected_max": int(np.max(selected_prediction)),
                "actual_max": int(np.max(actual)),
            }
        )
    outer_results = pd.DataFrame(outer_rows)
    outer_results["mae_improvement"] = (
        outer_results["baseline_mae"] - outer_results["selected_mae"]
    )
    outer_results["outbreak_improvement"] = (
        outer_results["baseline_outbreak_mae"]
        - outer_results["selected_outbreak_mae"]
    )
    development_candidate, final_search = select_from_inner(
        predictions, available
    )
    outer_origins_improved = int(
        (outer_results["selected_mae"] < outer_results["baseline_mae"]).sum()
    )
    outer_acceptance_ok = (
        outer_origins_improved >= math.ceil(0.75 * len(outer_results))
        and outer_results["selected_mae"].mean()
        < outer_results["baseline_mae"].mean()
        and outer_results["selected_outbreak_mae"].mean()
        <= outer_results["baseline_outbreak_mae"].mean()
        and outer_results.iloc[-1]["selected_mae"]
        <= outer_results.iloc[-1]["baseline_mae"]
    )
    final_candidate = development_candidate if outer_acceptance_ok else "baseline"
    summary = {
        "mean_baseline_outer_mae": float(outer_results["baseline_mae"].mean()),
        "mean_selected_outer_mae": float(outer_results["selected_mae"].mean()),
        "outer_origins_improved": outer_origins_improved,
        "mean_baseline_outbreak_mae": float(
            outer_results["baseline_outbreak_mae"].mean()
        ),
        "mean_selected_outbreak_mae": float(
            outer_results["selected_outbreak_mae"].mean()
        ),
        "development_candidate": development_candidate,
        "outer_acceptance_ok": bool(outer_acceptance_ok),
        "final_candidate": final_candidate,
    }
    return outer_results, pd.concat(selection_rows, ignore_index=True), {
        "summary": summary,
        "final_search": final_search,
    }


def _parse_candidate(candidate: str) -> dict[str, Any]:
    if candidate == "baseline":
        return {"family": "baseline"}
    parts = candidate.split("__")
    if parts[0] == "recency":
        return {
            "family": "recency",
            "config_name": parts[1],
            "weight": float(parts[2][1:]),
        }
    return {
        "family": "quantile",
        "quantile": float(parts[1][1:]),
        "weight": float(parts[2][1:]),
    }


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    final_candidate: str,
    random_state: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specification = _parse_candidate(final_candidate)
    rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {
        "candidate": final_candidate,
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
        baseline_model, baseline_raw = _fit_sj(
            features,
            target,
            train_indices,
            test_indices,
            random_state,
            n_estimators,
        ) if city == SJ else (None, None)
        if city == "iq":
            baseline_model = tree_model(city, random_state, n_estimators)
            baseline_model.fit(features.iloc[train_indices], target)
            raw = baseline_model.predict(features.iloc[test_indices])
        elif specification["family"] == "baseline":
            raw = baseline_raw
        elif specification["family"] == "quantile":
            tree_raw = per_tree_predictions(
                baseline_model, features.iloc[test_indices]
            )
            quantile_raw = np.quantile(
                tree_raw, specification["quantile"], axis=0
            )
            weight = specification["weight"]
            raw = (1.0 - weight) * baseline_raw + weight * quantile_raw
        else:
            config = RECENCY_CONFIGS[specification["config_name"]]
            recency_indices, sample_weight = recency_training_indices_and_weights(
                len(train_indices), config
            )
            recency_model, recency_raw = _fit_sj(
                features,
                target,
                recency_indices,
                test_indices,
                random_state,
                n_estimators,
                sample_weight,
            )
            weight = specification["weight"]
            raw = (1.0 - weight) * baseline_raw + weight * recency_raw
            artifacts["cities"][city] = {"recency": recency_model}
        result = combined.iloc[test_indices][KEY_COLUMNS].copy()
        result[TARGET_COLUMN] = _integer_cases(raw)
        rows.append(result)
        artifacts["cities"].setdefault(city, {})["baseline"] = baseline_model
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
    predictions = generate_origin_predictions(
        train, args.random_state, args.n_estimators
    )
    outer_results, inner_results, evaluation = evaluate_outer(predictions)
    summary = evaluation["summary"]
    final_search = evaluation["final_search"]
    predictions.to_csv(
        args.output_dir / "sj_nested_origin_predictions.csv", index=False
    )
    outer_results.to_csv(
        args.output_dir / "sj_nested_outer_results.csv", index=False
    )
    inner_results.to_csv(
        args.output_dir / "sj_nested_inner_results.csv", index=False
    )
    final_search.to_csv(
        args.output_dir / "sj_nested_final_search.csv", index=False
    )
    (args.output_dir / "sj_nested_summary.json").write_text(
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
    submission.to_csv(args.output_dir / "submission_tree_sj_nested.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "tree_sj_nested_models.joblib")
    print("\nNested outer results:")
    print(outer_results.to_string(index=False))
    print("\nNested summary:")
    print(json.dumps(summary, indent=2))
    print("\nFinal full-development selection:")
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
