"""Build a diversified Random Forest / Extra Trees DengAI ensemble.



Candidate selection uses city-specific forecast horizons matching the supplied

test set (156 weeks for Iquitos and 260 weeks for San Juan).  The most recent

historical origin is excluded from model/ensemble selection and reported as a

chronological holdout, reducing the optimistic bias of the older 52-week

hyperparameter search.

"""



from __future__ import annotations



import argparse

import itertools

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any, Callable



import joblib

import numpy as np

import pandas as pd

from sklearn.compose import TransformedTargetRegressor

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import make_pipeline



from dengai_pipeline import (

    DATE_COLUMN,

    KEY_COLUMNS,

    TARGET_COLUMN,

    load_data,

    make_features,

)

from train_enhanced_ensemble import enhanced_features





@dataclass(frozen=True)

class ForestSpec:

    family: str

    feature_mode: str

    criterion: str

    max_features: float

    min_samples_leaf: int

    max_depth: int | None = None

    log_target: bool = False





CANDIDATES: dict[str, dict[str, ForestSpec]] = {

    "iq": {

        "rf_log_enhanced": ForestSpec(

            "rf", "enhanced", "squared_error", 0.50, 5, log_target=True

        ),

        "rf_log_rolling": ForestSpec(

            "rf", "rolling", "squared_error", 0.65, 5, log_target=True

        ),

        "rf_log_compact": ForestSpec(

            "rf", "compact_weather", "squared_error", 0.75, 5, log_target=True

        ),

        "extra_log_rolling": ForestSpec(

            "extra", "rolling", "squared_error", 0.75, 5, log_target=True

        ),

        "rf_poisson_rolling": ForestSpec(

            "rf", "rolling", "poisson", 0.65, 7

        ),

        "extra_poisson_compact": ForestSpec(

            "extra", "compact_weather", "poisson", 0.80, 7

        ),

        "extra_poisson_compact_regularized": ForestSpec(

            "extra", "compact_weather", "poisson", 0.65, 10, max_depth=16

        ),

        "rf_log_enhanced_regularized": ForestSpec(

            "rf", "enhanced", "squared_error", 0.35, 8, log_target=True

        ),

    },

    "sj": {

        "extra_poisson_enhanced": ForestSpec(

            "extra", "enhanced", "poisson", 1.00, 5

        ),

        "extra_poisson_rolling": ForestSpec(

            "extra", "rolling", "poisson", 0.80, 7

        ),

        "rf_poisson_rolling": ForestSpec(

            "rf", "rolling", "poisson", 0.65, 7

        ),

        "extra_poisson_compact": ForestSpec(

            "extra", "compact_weather", "poisson", 0.80, 7

        ),

        "rf_squared_rolling": ForestSpec(

            "rf", "rolling", "squared_error", 0.55, 10, max_depth=14

        ),

        "extra_squared_base": ForestSpec(

            "extra", "base", "squared_error", 0.80, 10, max_depth=16

        ),

        "extra_poisson_enhanced_regularized": ForestSpec(

            "extra", "enhanced", "poisson", 0.75, 9, max_depth=18

        ),

        "rf_poisson_enhanced": ForestSpec(

            "rf", "enhanced", "poisson", 0.65, 7

        ),

    },

}



# Anchored on the best recent full-horizon forest for each city.  Small weights

# add model diversity without discarding the amplitude of the 22.5-MAE tree

# submission.  Larger secondary weights degraded the chronological holdout.

ROBUST_FOREST_WEIGHTS = {

    "iq": {"extra_poisson_compact": 0.80, "rf_log_enhanced": 0.20},

    "sj": {

        "extra_poisson_enhanced": 0.90,

        "extra_poisson_enhanced_regularized": 0.10,

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

    parser.add_argument("--n-estimators", type=int, default=600)

    parser.add_argument(

        "--max-components",

        type=int,

        default=3,

        choices=(1, 2, 3),

        help="Largest equal-weight forest combination considered.",

    )

    return parser.parse_args()





def feature_builder(mode: str) -> Callable[[pd.DataFrame], pd.DataFrame]:

    if mode == "enhanced":

        return enhanced_features

    return lambda frame: make_features(frame, mode)





def forest_model(

    spec: ForestSpec, random_state: int = 42, n_estimators: int = 600

) -> Any:

    estimator_class = (

        RandomForestRegressor if spec.family == "rf" else ExtraTreesRegressor

    )

    forest = estimator_class(

        n_estimators=n_estimators,

        criterion=spec.criterion,

        max_features=spec.max_features,

        min_samples_leaf=spec.min_samples_leaf,

        max_depth=spec.max_depth,

        n_jobs=-1,

        random_state=random_state,

    )

    pipeline = make_pipeline(SimpleImputer(strategy="median"), forest)

    if not spec.log_target:

        return pipeline

    return TransformedTargetRegressor(

        regressor=pipeline,

        func=np.log1p,

        inverse_func=np.expm1,

    )





def _integer_cases(values: np.ndarray | pd.Series) -> np.ndarray:

    return np.floor(np.maximum(0.0, np.asarray(values, dtype=float)) + 0.5).astype(

        int

    )





def _origins(n_rows: int, horizon: int) -> list[int]:

    first = n_rows - 2 * horizon

    last = n_rows - horizon

    starts = list(range(first, last + 1, 52))

    if not starts or starts[-1] != last:

        starts.append(last)

    if first <= 0:

        raise ValueError("Not enough history for long-horizon validation")

    return starts





def _candidate_combinations(

    names: list[str], max_components: int

) -> list[tuple[str, ...]]:

    return [

        combo

        for size in range(1, max_components + 1)

        for combo in itertools.combinations(names, size)

    ]





def _mean_prediction(

    predictions: dict[str, np.ndarray], components: tuple[str, ...]

) -> np.ndarray:

    return np.mean([predictions[name] for name in components], axis=0)





def _weighted_prediction(

    predictions: dict[str, np.ndarray], weights: dict[str, float]

) -> np.ndarray:

    result = np.zeros_like(next(iter(predictions.values())), dtype=float)

    for name, weight in weights.items():

        result += weight * predictions[name]

    return result





def run_long_horizon_backtest(

    train: pd.DataFrame,

    test: pd.DataFrame,

    random_state: int,

    n_estimators: int,

    max_components: int,

) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, ...]]]:

    score_rows: list[dict[str, Any]] = []

    prediction_rows: list[pd.DataFrame] = []

    selected: dict[str, tuple[str, ...]] = {}



    for city, raw_city_frame in train.groupby("city", sort=True):

        city_frame = raw_city_frame.sort_values(DATE_COLUMN).reset_index(drop=True)

        targets = city_frame[TARGET_COLUMN].to_numpy(float)

        horizon = int((test["city"] == city).sum())

        starts = _origins(len(city_frame), horizon)

        specs = CANDIDATES[city]

        modes = {spec.feature_mode for spec in specs.values()}

        feature_sets = {

            mode: feature_builder(mode)(city_frame) for mode in sorted(modes)

        }

        by_origin: list[dict[str, np.ndarray]] = []

        actual_by_origin: list[np.ndarray] = []

        city_prediction_rows: list[pd.DataFrame] = []



        for origin_number, start in enumerate(starts, start=1):

            train_indices = np.arange(start)

            validation_indices = np.arange(start, start + horizon)

            origin_predictions: dict[str, np.ndarray] = {}

            for offset, (name, spec) in enumerate(specs.items()):

                model = forest_model(

                    spec,

                    random_state=random_state + offset,

                    n_estimators=n_estimators,

                )

                features = feature_sets[spec.feature_mode]

                model.fit(features.iloc[train_indices], targets[train_indices])

                origin_predictions[name] = np.maximum(

                    0.0, model.predict(features.iloc[validation_indices])

                )

            actual = targets[validation_indices].astype(int)

            by_origin.append(origin_predictions)

            actual_by_origin.append(actual)



            rows = city_frame.iloc[validation_indices][

                KEY_COLUMNS + [DATE_COLUMN]

            ].copy()

            rows.insert(0, "origin", origin_number)

            rows["actual"] = actual

            for name, prediction in origin_predictions.items():

                rows[f"{name}_prediction"] = _integer_cases(prediction)

            city_prediction_rows.append(rows)



        # Select only on older origins. The latest origin remains untouched.

        development_actual = np.concatenate(actual_by_origin[:-1])

        combinations = _candidate_combinations(list(specs), max_components)

        development_scores: list[tuple[float, tuple[str, ...]]] = []

        for combination in combinations:

            prediction = np.concatenate(

                [

                    _integer_cases(_mean_prediction(origin, combination))

                    for origin in by_origin[:-1]

                ]

            )

            development_scores.append(

                (mean_absolute_error(development_actual, prediction), combination)

            )

        _, best_combination = min(

            development_scores, key=lambda item: (item[0], len(item[1]), item[1])

        )

        selected[city] = best_combination

        # Use the same unrounded base predictions and final rounding as the
        # score calculation. Each row is paired with its own forecast origin.
        for rows, origin_predictions in zip(city_prediction_rows, by_origin):
            rows["selected_ensemble_prediction"] = _integer_cases(
                _mean_prediction(origin_predictions, best_combination)
            )
            rows["robust_ensemble_prediction"] = _integer_cases(
                _weighted_prediction(
                    origin_predictions, ROBUST_FOREST_WEIGHTS[city]
                )
            )

        prediction_rows.extend(city_prediction_rows)



        for name in specs:

            for split_name, origin_slice in (

                ("development", slice(None, -1)),

                ("chronological_holdout", slice(-1, None)),

                ("all_origins", slice(None)),

            ):

                actual = np.concatenate(actual_by_origin[origin_slice])

                prediction = np.concatenate(

                    [

                        _integer_cases(origin[name])

                        for origin in by_origin[origin_slice]

                    ]

                )

                score_rows.append(

                    {

                        "city": city,

                        "horizon_weeks": horizon,

                        "split": split_name,

                        "model": name,

                        "components": name,

                        "n_predictions": len(actual),

                        "mae": mean_absolute_error(actual, prediction),

                    }

                )



        for split_name, origin_slice in (

            ("development", slice(None, -1)),

            ("chronological_holdout", slice(-1, None)),

            ("all_origins", slice(None)),

        ):

            actual = np.concatenate(actual_by_origin[origin_slice])

            prediction = np.concatenate(

                [

                    _integer_cases(_mean_prediction(origin, best_combination))

                    for origin in by_origin[origin_slice]

                ]

            )

            score_rows.append(

                {

                    "city": city,

                    "horizon_weeks": horizon,

                    "split": split_name,

                    "model": "selected_forest_ensemble",

                    "components": "+".join(best_combination),

                    "n_predictions": len(actual),

                    "mae": mean_absolute_error(actual, prediction),

                }

            )



        robust_weights = ROBUST_FOREST_WEIGHTS[city]

        for split_name, origin_slice in (

            ("development", slice(None, -1)),

            ("chronological_holdout", slice(-1, None)),

            ("all_origins", slice(None)),

        ):

            actual = np.concatenate(actual_by_origin[origin_slice])

            prediction = np.concatenate(

                [

                    _integer_cases(_weighted_prediction(origin, robust_weights))

                    for origin in by_origin[origin_slice]

                ]

            )

            score_rows.append(

                {

                    "city": city,

                    "horizon_weeks": horizon,

                    "split": split_name,

                    "model": "robust_forest_ensemble",

                    "components": "+".join(robust_weights),

                    "n_predictions": len(actual),

                    "mae": mean_absolute_error(actual, prediction),

                }

            )



    return (

        pd.DataFrame(score_rows),

        pd.concat(prediction_rows, ignore_index=True),

        selected,

    )





def fit_final(

    train: pd.DataFrame,

    test: pd.DataFrame,

    template: pd.DataFrame,

    city_weights: dict[str, dict[str, float]],

    random_state: int,

    n_estimators: int,

) -> tuple[pd.DataFrame, dict[str, Any]]:

    rows: list[pd.DataFrame] = []

    artifacts: dict[str, Any] = {

        "city_weights": city_weights,

        "cities": {},

        "random_state": random_state,

        "n_estimators": n_estimators,

    }

    for city in sorted(train["city"].unique()):

        train_city = train.loc[train["city"] == city].copy()

        test_city = test.loc[test["city"] == city].copy()

        train_city["_split"] = "train"

        test_city["_split"] = "test"

        combined = (

            pd.concat([train_city, test_city], ignore_index=True, sort=False)

            .sort_values(DATE_COLUMN)

            .reset_index(drop=True)

        )

        train_indices = np.flatnonzero(combined["_split"].eq("train").to_numpy())

        test_indices = np.flatnonzero(combined["_split"].eq("test").to_numpy())

        targets = combined.iloc[train_indices][TARGET_COLUMN].to_numpy(float)

        predictions: dict[str, np.ndarray] = {}

        fitted: dict[str, Any] = {}

        for name in city_weights[city]:

            spec = CANDIDATES[city][name]

            features = feature_builder(spec.feature_mode)(combined)

            offset = list(CANDIDATES[city]).index(name)

            model = forest_model(

                spec,

                random_state=random_state + offset,

                n_estimators=n_estimators,

            )

            model.fit(features.iloc[train_indices], targets)

            predictions[name] = (

                np.maximum(0.0, model.predict(features.iloc[test_indices]))

            )

            fitted[name] = {

                "model": model,

                "spec": asdict(spec),

                "feature_columns": features.columns.tolist(),

            }

        result = combined.iloc[test_indices][KEY_COLUMNS].copy()

        result[TARGET_COLUMN] = _integer_cases(

            _weighted_prediction(predictions, city_weights[city])

        )

        rows.append(result)

        artifacts["cities"][city] = fitted



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

    scores, validation_predictions, selected = run_long_horizon_backtest(

        train,

        test,

        args.random_state,

        args.n_estimators,

        args.max_components,

    )

    selected_weights = {

        city: {name: 1.0 / len(names) for name in names}

        for city, names in selected.items()

    }

    selected_submission, selected_artifacts = fit_final(

        train,

        test,

        template,

        selected_weights,

        args.random_state,

        args.n_estimators,

    )

    robust_submission, robust_artifacts = fit_final(

        train,

        test,

        template,

        ROBUST_FOREST_WEIGHTS,

        args.random_state,

        args.n_estimators,

    )

    scores.to_csv(args.output_dir / "forest_validation_scores.csv", index=False)

    validation_predictions.to_csv(

        args.output_dir / "forest_validation_predictions.csv", index=False

    )

    selected_submission.to_csv(

        args.output_dir / "submission_forest_long_horizon.csv", index=False

    )

    robust_submission.to_csv(

        args.output_dir / "submission_forest_robust.csv", index=False

    )

    joblib.dump(

        {"selected": selected_artifacts, "robust": robust_artifacts},

        args.output_dir / "forest_ensemble_models.joblib",

    )



    print("Selected components:")

    for city, components in selected.items():

        print(f"  {city}: {', '.join(components)}")

    print("\nSelected ensemble validation:")

    print(

        scores.loc[

            scores["model"].isin(

                ["selected_forest_ensemble", "robust_forest_ensemble"]

            )

        ].to_string(index=False)

    )

    print("\nFinal prediction distributions:")

    print(

        robust_submission.groupby("city")[TARGET_COLUMN]

        .agg(["min", "median", "mean", "max"])

        .round(2)

        .to_string()

    )





if __name__ == "__main__":

    main()
