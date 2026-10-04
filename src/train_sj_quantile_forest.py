"""Evaluate quantile aggregation for the SJ Extra Trees model.

Standard Extra Trees averages predictions from all trees. Because competition
scoring uses MAE, a conditional median-like aggregation can be more appropriate.
This script searches conservative blends of the forest mean and per-tree
quantiles across both annual and 260-week chronological validation horizons.
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
QUANTILES = tuple(np.round(np.arange(0.25, 0.76, 0.05), 2))
QUANTILE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
LONG_HORIZON = 260


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    return parser.parse_args()


def per_tree_predictions(model: Any, features: pd.DataFrame) -> np.ndarray:
    imputed = model.named_steps["simpleimputer"].transform(features)
    forest = model.named_steps["extratreesregressor"]
    return np.asarray([tree.predict(imputed) for tree in forest.estimators_])


def generate_validation_sets(
    train: pd.DataFrame, random_state: int, n_estimators: int
) -> list[dict[str, Any]]:
    city = (
        train.loc[train["city"].eq(SJ)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    features = make_features(city)
    target = city[TARGET_COLUMN].to_numpy(float)
    definitions: list[tuple[str, int, int]] = []
    definitions.extend(
        ("annual", fold, start)
        for fold, start in enumerate(
            range(len(city) - 6 * 52, len(city), 52), start=1
        )
    )
    definitions.extend(
        ("long", fold, start)
        for fold, start in enumerate(
            range(416, len(city) - LONG_HORIZON + 1, 52), start=1
        )
    )
    sets: list[dict[str, Any]] = []
    for position, (kind, fold, start) in enumerate(definitions, start=1):
        horizon = 52 if kind == "annual" else LONG_HORIZON
        train_indices = np.arange(start)
        validation_indices = np.arange(start, start + horizon)
        model = tree_model(SJ, random_state, n_estimators)
        model.fit(features.iloc[train_indices], target[train_indices])
        tree_prediction = per_tree_predictions(
            model, features.iloc[validation_indices]
        )
        sets.append(
            {
                "kind": kind,
                "fold": fold,
                "keys": city.iloc[validation_indices][
                    KEY_COLUMNS + [DATE_COLUMN]
                ].copy(),
                "actual": target[validation_indices].astype(int),
                "mean_raw": tree_prediction.mean(axis=0),
                "tree_prediction": tree_prediction,
            }
        )
        print(
            f"SJ quantile validation {position}/{len(definitions)} "
            f"kind={kind} fold={fold}",
            flush=True,
        )
    return sets


def _summary(
    validation_sets: list[dict[str, Any]], quantile: float | None, weight: float
) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, float]]] = {"annual": [], "long": []}
    prediction_mean: list[float] = []
    prediction_max: list[float] = []
    for item in validation_sets:
        mean_raw = item["mean_raw"]
        if quantile is None:
            raw = mean_raw
        else:
            quantile_raw = np.quantile(
                item["tree_prediction"], quantile, axis=0
            )
            raw = (1.0 - weight) * mean_raw + weight * quantile_raw
        prediction = _integer_cases(raw)
        actual = item["actual"]
        threshold = float(np.quantile(actual, 0.90))
        outbreak = actual >= threshold
        by_kind[item["kind"]].append(
            {
                "mae": mean_absolute_error(actual, prediction),
                "outbreak_mae": mean_absolute_error(
                    actual[outbreak], prediction[outbreak]
                ),
            }
        )
        prediction_mean.append(float(np.mean(prediction)))
        prediction_max.append(float(np.max(prediction)))
    annual_mae = np.array([row["mae"] for row in by_kind["annual"]])
    long_mae = np.array([row["mae"] for row in by_kind["long"]])
    annual_outbreak = np.array(
        [row["outbreak_mae"] for row in by_kind["annual"]]
    )
    long_outbreak = np.array(
        [row["outbreak_mae"] for row in by_kind["long"]]
    )
    return {
        "annual_mae": float(annual_mae.mean()),
        "annual_std": float(annual_mae.std()),
        "long_mae": float(long_mae.mean()),
        "long_std": float(long_mae.std()),
        "recent_long_mae": float(long_mae[-1]),
        "annual_outbreak_mae": float(annual_outbreak.mean()),
        "long_outbreak_mae": float(long_outbreak.mean()),
        "recent_outbreak_mae": float(long_outbreak[-1]),
        "prediction_mean": float(np.mean(prediction_mean)),
        "prediction_max": float(np.max(prediction_max)),
    }


def select_quantile(
    validation_sets: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = _summary(validation_sets, None, 0.0)
    rows: list[dict[str, Any]] = [
        {
            "quantile": 0.5,
            "quantile_weight": 0.0,
            **baseline,
            "amplitude_ok": True,
            "outbreak_ok": True,
            "eligible": True,
            "selection_score": 1.0,
        }
    ]
    for quantile in QUANTILES:
        for weight in QUANTILE_WEIGHTS:
            metrics = _summary(validation_sets, quantile, weight)
            amplitude_ok = (
                metrics["prediction_mean"] >= 0.90 * baseline["prediction_mean"]
                and metrics["prediction_mean"]
                <= 1.10 * baseline["prediction_mean"]
                and metrics["prediction_max"] >= 0.90 * baseline["prediction_max"]
            )
            outbreak_ok = (
                metrics["annual_outbreak_mae"]
                <= 1.03 * baseline["annual_outbreak_mae"]
                and metrics["long_outbreak_mae"]
                <= 1.03 * baseline["long_outbreak_mae"]
                and metrics["recent_outbreak_mae"]
                <= 1.05 * baseline["recent_outbreak_mae"]
            )
            score = (
                0.45 * metrics["annual_mae"] / baseline["annual_mae"]
                + 0.45 * metrics["long_mae"] / baseline["long_mae"]
                + 0.10 * metrics["annual_std"] / baseline["annual_std"]
            )
            rows.append(
                {
                    "quantile": quantile,
                    "quantile_weight": weight,
                    **metrics,
                    "amplitude_ok": amplitude_ok,
                    "outbreak_ok": outbreak_ok,
                    "eligible": amplitude_ok and outbreak_ok,
                    "selection_score": score,
                }
            )
    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "annual_mae", "long_mae"]
    ).reset_index(drop=True)
    winner = results.loc[results["eligible"]].iloc[0]
    improves = (
        winner["annual_mae"] < baseline["annual_mae"]
        and winner["long_mae"] < baseline["long_mae"]
        and winner["recent_long_mae"] <= baseline["recent_long_mae"]
    )
    if improves:
        selected = {
            "quantile": float(winner["quantile"]),
            "quantile_weight": float(winner["quantile_weight"]),
        }
    else:
        selected = {"quantile": 0.5, "quantile_weight": 0.0}
    selected["baseline_annual_mae"] = baseline["annual_mae"]
    selected["baseline_long_mae"] = baseline["long_mae"]
    return results, selected


def validation_predictions(
    validation_sets: list[dict[str, Any]], selected: dict[str, Any]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in validation_sets:
        quantile_raw = np.quantile(
            item["tree_prediction"], selected["quantile"], axis=0
        )
        weight = selected["quantile_weight"]
        raw = (1.0 - weight) * item["mean_raw"] + weight * quantile_raw
        frame = item["keys"].copy()
        frame["validation_kind"] = item["kind"]
        frame["fold"] = item["fold"]
        frame["actual"] = item["actual"]
        frame["mean_prediction"] = _integer_cases(item["mean_raw"])
        frame["quantile_ensemble_prediction"] = _integer_cases(raw)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


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
        mean_raw = model.predict(features.iloc[test_indices])
        if city == SJ and selected["quantile_weight"] > 0.0:
            tree_raw = per_tree_predictions(model, features.iloc[test_indices])
            quantile_raw = np.quantile(tree_raw, selected["quantile"], axis=0)
            weight = selected["quantile_weight"]
            raw = (1.0 - weight) * mean_raw + weight * quantile_raw
        else:
            raw = mean_raw
        result = combined.iloc[test_indices][KEY_COLUMNS].copy()
        result[TARGET_COLUMN] = _integer_cases(raw)
        rows.append(result)
        artifacts["cities"][city] = {"model": model}
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
    sets = generate_validation_sets(train, args.random_state, args.n_estimators)
    results, selected = select_quantile(sets)
    predictions = validation_predictions(sets, selected)
    results.to_csv(args.output_dir / "sj_quantile_search_results.csv", index=False)
    predictions.to_csv(
        args.output_dir / "sj_quantile_validation_predictions.csv", index=False
    )
    (args.output_dir / "sj_quantile_selected_config.json").write_text(
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
    submission.to_csv(args.output_dir / "submission_tree_sj_quantile.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "tree_sj_quantile_models.joblib")
    print("\nSelected SJ quantile configuration:")
    print(json.dumps(selected, indent=2))
    print("\nTop candidates:")
    print(results.head(12).to_string(index=False))
    print("\nSubmission distribution:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
