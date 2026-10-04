"""Evaluate CatBoost and XGBoost against the confirmed DengAI tree solution.

The search is city-specific and chronological. Each boosting candidate is also
tested as a blend with the confirmed tree model. Amplitude and outbreak guards
prevent selection of deceptively smooth forecasts, and tree-only is always an
available fallback. Existing confirmed artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from train_tree_ensemble import (
    DATE_COLUMN,
    KEY_COLUMNS,
    TARGET_COLUMN,
    _integer_cases,
    load_data,
    make_features,
    tree_model,
)

SEARCH_FOLDS = 4
REPORT_FOLDS = 6
BLEND_WEIGHTS = tuple(np.round(np.arange(0.1, 1.01, 0.1), 2))
BOOSTING_CONFIGS: dict[str, dict[str, Any]] = {
    "xgb_poisson_d2": {"family": "xgboost", "loss": "poisson", "depth": 2},
    "xgb_poisson_d3": {"family": "xgboost", "loss": "poisson", "depth": 3},
    "xgb_mae_d2": {"family": "xgboost", "loss": "mae", "depth": 2},
    "xgb_mae_d3": {"family": "xgboost", "loss": "mae", "depth": 3},
    "cat_poisson_d4": {"family": "catboost", "loss": "poisson", "depth": 4},
    "cat_poisson_d6": {"family": "catboost", "loss": "poisson", "depth": 6},
    "cat_mae_d4": {"family": "catboost", "loss": "mae", "depth": 4},
    "cat_mae_d6": {"family": "catboost", "loss": "mae", "depth": 6},
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Boosting iterations/trees for search and final fitting.",
    )
    return parser.parse_args()


def boosting_model(
    config_name: str, iterations: int, random_state: int
) -> CatBoostRegressor | XGBRegressor:
    config = BOOSTING_CONFIGS[config_name]
    if config["family"] == "xgboost":
        objective = "count:poisson" if config["loss"] == "poisson" else "reg:absoluteerror"
        return XGBRegressor(
            objective=objective,
            n_estimators=iterations,
            learning_rate=0.03,
            max_depth=config["depth"],
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=10.0,
            max_delta_step=1.0 if config["loss"] == "poisson" else 0.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    loss = "Poisson" if config["loss"] == "poisson" else "MAE"
    return CatBoostRegressor(
        loss_function=loss,
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.03,
        depth=config["depth"],
        l2_leaf_reg=10.0,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def generate_search_predictions(
    train: pd.DataFrame,
    random_state: int,
    n_estimators: int,
    iterations: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for city, city_frame in train.groupby("city", sort=True):
        city_frame = city_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
        tree_features = make_features(city_frame)
        boost_features = tree_features.drop(columns=["year"])
        target = city_frame[TARGET_COLUMN].to_numpy(float)
        starts = list(range(len(city_frame) - SEARCH_FOLDS * 52, len(city_frame), 52))
        for fold, start in enumerate(starts, start=1):
            train_index = np.arange(start)
            validation_index = np.arange(start, start + 52)
            fold_rows = city_frame.iloc[validation_index][KEY_COLUMNS + [DATE_COLUMN]].copy()
            fold_rows["fold"] = fold
            fold_rows["actual"] = target[validation_index].astype(int)
            tree = tree_model(city, random_state, n_estimators)
            tree.fit(tree_features.iloc[train_index], target[train_index])
            fold_rows["tree_raw"] = tree.predict(
                tree_features.iloc[validation_index]
            )
            for position, config_name in enumerate(BOOSTING_CONFIGS, start=1):
                model = boosting_model(config_name, iterations, random_state)
                model.fit(boost_features.iloc[train_index], target[train_index])
                fold_rows[config_name] = np.maximum(
                    0.0, model.predict(boost_features.iloc[validation_index])
                )
                print(
                    f"{city.upper()} fold {fold}/{SEARCH_FOLDS} "
                    f"candidate {position}/{len(BOOSTING_CONFIGS)} {config_name}",
                    flush=True,
                )
            rows.append(fold_rows)
    return pd.concat(rows, ignore_index=True)


def _candidate_metrics(
    frame: pd.DataFrame, prediction: np.ndarray, baseline: dict[str, float]
) -> dict[str, Any]:
    actual = frame["actual"].to_numpy(int)
    integer_prediction = _integer_cases(prediction)
    fold_frame = frame[["fold", "actual"]].copy()
    fold_frame["prediction"] = integer_prediction
    fold_mae = [
        mean_absolute_error(group["actual"], group["prediction"])
        for _, group in fold_frame.groupby("fold")
    ]
    outbreak_threshold = float(np.quantile(actual, 0.90))
    outbreak = actual >= outbreak_threshold
    pooled_mae = mean_absolute_error(actual, integer_prediction)
    outbreak_mae = mean_absolute_error(actual[outbreak], integer_prediction[outbreak])
    robust_score = float(np.mean(fold_mae) + 0.2 * np.std(fold_mae))
    prediction_mean = float(np.mean(integer_prediction))
    prediction_max = float(np.max(integer_prediction))
    amplitude_ok = (
        prediction_mean >= 0.90 * baseline["prediction_mean"]
        and prediction_mean <= 1.10 * baseline["prediction_mean"]
        and prediction_max >= 0.90 * baseline["prediction_max"]
    )
    outbreak_ok = outbreak_mae <= 1.03 * baseline["outbreak_mae"]
    return {
        "pooled_mae": pooled_mae,
        "mean_fold_mae": float(np.mean(fold_mae)),
        "std_fold_mae": float(np.std(fold_mae)),
        "robust_score": robust_score,
        "outbreak_mae": outbreak_mae,
        "prediction_mean": prediction_mean,
        "prediction_max": prediction_max,
        "amplitude_ok": amplitude_ok,
        "outbreak_ok": outbreak_ok,
        "eligible": amplitude_ok and outbreak_ok,
    }


def select_city_models(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for city, frame in predictions.groupby("city", sort=True):
        tree_raw = frame["tree_raw"].to_numpy(float)
        baseline_seed = {
            "prediction_mean": 0.0,
            "prediction_max": 0.0,
            "outbreak_mae": 0.0,
        }
        baseline = _candidate_metrics(frame, tree_raw, baseline_seed)
        # Re-evaluate now that amplitude/outbreak reference values are known.
        baseline = _candidate_metrics(frame, tree_raw, baseline)
        rows.append(
            {
                "city": city,
                "candidate": "tree",
                "family": "tree",
                "loss": "confirmed",
                "depth": np.nan,
                "boosting_weight": 0.0,
                **baseline,
                "improvement_vs_tree": 0.0,
            }
        )
        for config_name, config in BOOSTING_CONFIGS.items():
            boost_raw = frame[config_name].to_numpy(float)
            for weight in BLEND_WEIGHTS:
                raw = (1.0 - weight) * tree_raw + weight * boost_raw
                metrics = _candidate_metrics(frame, raw, baseline)
                rows.append(
                    {
                        "city": city,
                        "candidate": config_name,
                        "family": config["family"],
                        "loss": config["loss"],
                        "depth": config["depth"],
                        "boosting_weight": weight,
                        **metrics,
                        "improvement_vs_tree": baseline["pooled_mae"]
                        - metrics["pooled_mae"],
                    }
                )
        city_results = pd.DataFrame(row for row in rows if row["city"] == city)
        winner = city_results.loc[city_results["eligible"]].sort_values(
            ["robust_score", "pooled_mae", "boosting_weight", "candidate"]
        ).iloc[0]
        if not (
            winner["pooled_mae"] < baseline["pooled_mae"]
            and winner["robust_score"] < baseline["robust_score"]
        ):
            selected[city] = {"candidate": "tree", "boosting_weight": 0.0}
        else:
            selected[city] = {
                "candidate": str(winner["candidate"]),
                "boosting_weight": float(winner["boosting_weight"]),
            }
    results = pd.DataFrame(rows).sort_values(
        ["city", "robust_score", "pooled_mae", "candidate", "boosting_weight"]
    )
    return results, selected


def run_selected_backtest(
    train: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
    random_state: int,
    n_estimators: int,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    scores: list[dict[str, Any]] = []
    all_actual: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    for city, city_frame in train.groupby("city", sort=True):
        city_frame = city_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
        tree_features = make_features(city_frame)
        boost_features = tree_features.drop(columns=["year"])
        target = city_frame[TARGET_COLUMN].to_numpy(float)
        city_actual: list[np.ndarray] = []
        city_prediction: list[np.ndarray] = []
        starts = list(range(len(city_frame) - REPORT_FOLDS * 52, len(city_frame), 52))
        choice = selected[city]
        for fold, start in enumerate(starts, start=1):
            train_index = np.arange(start)
            validation_index = np.arange(start, start + 52)
            tree = tree_model(city, random_state, n_estimators)
            tree.fit(tree_features.iloc[train_index], target[train_index])
            tree_raw = tree.predict(tree_features.iloc[validation_index])
            if choice["candidate"] == "tree":
                raw = tree_raw
            else:
                model = boosting_model(choice["candidate"], iterations, random_state)
                model.fit(boost_features.iloc[train_index], target[train_index])
                boost_raw = np.maximum(
                    0.0, model.predict(boost_features.iloc[validation_index])
                )
                weight = choice["boosting_weight"]
                raw = (1.0 - weight) * tree_raw + weight * boost_raw
            prediction = _integer_cases(raw)
            actual = target[validation_index].astype(int)
            fold_rows = city_frame.iloc[validation_index][KEY_COLUMNS + [DATE_COLUMN]].copy()
            fold_rows["fold"] = fold
            fold_rows["actual"] = actual
            fold_rows["tree_prediction"] = _integer_cases(tree_raw)
            fold_rows["boosting_ensemble_prediction"] = prediction
            rows.append(fold_rows)
            city_actual.append(actual)
            city_prediction.append(prediction)
            all_actual.append(actual)
            all_prediction.append(prediction)
        actual_all = np.concatenate(city_actual)
        prediction_all = np.concatenate(city_prediction)
        scores.append(
            {
                "city": city,
                "candidate": choice["candidate"],
                "boosting_weight": choice["boosting_weight"],
                "n_predictions": len(actual_all),
                "mae": mean_absolute_error(actual_all, prediction_all),
            }
        )
    scores.append(
        {
            "city": "all",
            "candidate": "city_specific",
            "boosting_weight": np.nan,
            "n_predictions": sum(map(len, all_actual)),
            "mae": mean_absolute_error(
                np.concatenate(all_actual), np.concatenate(all_prediction)
            ),
        }
    )
    return pd.DataFrame(scores), pd.concat(rows, ignore_index=True)


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
    random_state: int,
    n_estimators: int,
    iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {"selection": selected, "cities": {}}
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
        train_index = np.flatnonzero(combined["_split"].eq("train").to_numpy())
        test_index = np.flatnonzero(combined["_split"].eq("test").to_numpy())
        tree_features = make_features(combined)
        boost_features = tree_features.drop(columns=["year"])
        target = combined.iloc[train_index][TARGET_COLUMN].to_numpy(float)
        tree = tree_model(city, random_state, n_estimators)
        tree.fit(tree_features.iloc[train_index], target)
        tree_raw = tree.predict(tree_features.iloc[test_index])
        choice = selected[city]
        city_artifacts: dict[str, Any] = {"tree": tree}
        if choice["candidate"] == "tree":
            raw = tree_raw
        else:
            boost = boosting_model(choice["candidate"], iterations, random_state)
            boost.fit(boost_features.iloc[train_index], target)
            boost_raw = np.maximum(
                0.0, boost.predict(boost_features.iloc[test_index])
            )
            weight = choice["boosting_weight"]
            raw = (1.0 - weight) * tree_raw + weight * boost_raw
            city_artifacts["boosting"] = boost
        result = combined.iloc[test_index][KEY_COLUMNS].copy()
        result[TARGET_COLUMN] = _integer_cases(raw)
        rows.append(result)
        artifacts["cities"][city] = city_artifacts
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
    search_predictions = generate_search_predictions(
        train, args.random_state, args.n_estimators, args.iterations
    )
    search_results, selected = select_city_models(search_predictions)
    search_results.to_csv(
        args.output_dir / "boosting_search_results.csv", index=False
    )
    (args.output_dir / "boosting_selected_config.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSelected models:")
    print(json.dumps(selected, indent=2))
    scores, validation = run_selected_backtest(
        train,
        selected,
        args.random_state,
        args.n_estimators,
        args.iterations,
    )
    scores.to_csv(args.output_dir / "boosting_validation_scores.csv", index=False)
    validation.to_csv(
        args.output_dir / "boosting_validation_predictions.csv", index=False
    )
    submission, artifacts = fit_final(
        train,
        test,
        template,
        selected,
        args.random_state,
        args.n_estimators,
        args.iterations,
    )
    submission.to_csv(args.output_dir / "submission_tree_boosting.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "tree_boosting_models.joblib")
    print("\nSix-fold validation:")
    print(scores.to_string(index=False))
    print("\nSubmission distribution:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
