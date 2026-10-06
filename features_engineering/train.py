"""Compare count models on the improved features with expanding annual folds.

The fold definition matches src/train_tree_ensemble.py: six contiguous 52-week
blocks at the end of each city, trained only on earlier weeks. Predictions are
rounded to integer cases before MAE, which is the competition metric.

A candidate replaces the confirmed tree for a city only when the blend weight
chosen on the first four folds also improves the last two folds.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from features import build_features  # noqa: E402
from train_tree_ensemble import (  # noqa: E402
    DATE_COLUMN,
    KEY_COLUMNS,
    TARGET_COLUMN,
    load_data,
    tree_model,
)

FOLD_COUNT = 6
FOLD_WEEKS = 52
CONFIRMED_TREE = ROOT / "artifacts" / "tree_validation_predictions.csv"
CONFIRMED_SUBMISSION = ROOT / "artifacts" / "submission_tree_ensemble.csv"
OUTPUT_DIR = PACKAGE / "artifacts"


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    objective: str = "mae"
    tweedie_power: float | None = None
    linear_tree: bool = False


SPECS = (
    Spec("climatology", "climatology"),
    Spec("lgbm_mae", "lgbm", "regression_l1"),
    Spec("lgbm_poisson", "lgbm", "poisson"),
    Spec("lgbm_tweedie_1_5", "lgbm", "tweedie", 1.5),
    Spec("lgbm_tweedie_1_8", "lgbm", "tweedie", 1.8),
    Spec("lgbm_residual_mae", "residual_mae"),
    Spec("lgbm_poisson_offset", "poisson_offset"),
    Spec("catboost_mae", "catboost", "MAE"),
    Spec("catboost_poisson", "catboost", "Poisson"),
    Spec("current_tree_new_features", "forest"),
)


def integer_cases(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.floor(np.maximum(0.0, values) + 0.5).astype(int)


def fold_starts(n_rows: int) -> list[int]:
    return list(range(n_rows - FOLD_COUNT * FOLD_WEEKS, n_rows, FOLD_WEEKS))


def _week_offsets(
    week_train: np.ndarray,
    target_train: np.ndarray,
    week_predict: np.ndarray,
) -> np.ndarray:
    climatology = pd.Series(target_train).groupby(week_train).mean()
    fallback = float(np.mean(target_train))
    return (
        pd.Series(week_predict)
        .map(climatology)
        .fillna(fallback)
        .to_numpy(float)
    )


def _lgbm(city: str, spec: Spec) -> LGBMRegressor:
    params: dict[str, float | int | str | bool | None] = dict(
        objective=spec.objective,
        linear_tree=spec.linear_tree,
        n_estimators=250 if spec.linear_tree else 400,
        learning_rate=0.05,
        num_leaves=7 if spec.linear_tree else (15 if city == "iq" else 31),
        min_child_samples=15 if city == "iq" else 25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        force_col_wise=True,
    )
    if spec.tweedie_power is not None:
        params["tweedie_variance_power"] = spec.tweedie_power
    return LGBMRegressor(**params)


def _catboost(spec: Spec) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function=spec.objective,
        iterations=300,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=8,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def _predict_spec(
    spec: Spec,
    city: str,
    features: pd.DataFrame,
    target: np.ndarray,
    week: np.ndarray,
    train_index: np.ndarray,
    predict_index: np.ndarray,
    forest_trees: int,
) -> np.ndarray:
    week_train = week[train_index]
    target_train = target[train_index]
    week_predict = week[predict_index]
    if spec.kind == "climatology":
        return _week_offsets(week_train, target_train, week_predict)

    x_train = features.iloc[train_index]
    x_predict = features.iloc[predict_index]
    if spec.kind == "forest":
        model = tree_model(city, random_state=42, n_estimators=forest_trees)
        model.fit(x_train, target_train)
        return np.maximum(0.0, model.predict(x_predict))

    if spec.kind == "residual_mae":
        offset_train = _week_offsets(week_train, target_train, week_train)
        offset_predict = _week_offsets(week_train, target_train, week_predict)
        model = _lgbm(city, Spec(spec.name, "lgbm", "regression_l1"))
        model.fit(x_train, target_train - offset_train)
        return np.maximum(0.0, offset_predict + model.predict(x_predict))

    if spec.kind == "poisson_offset":
        offset_train = np.maximum(
            _week_offsets(week_train, target_train, week_train), 0.1
        )
        offset_predict = np.maximum(
            _week_offsets(week_train, target_train, week_predict), 0.1
        )
        model = _lgbm(city, Spec(spec.name, "lgbm", "poisson"))
        model.fit(x_train, target_train, init_score=np.log(offset_train))
        raw = model.predict(x_predict, raw_score=True)
        return np.exp(raw + np.log(offset_predict))

    if spec.kind == "lgbm":
        model = _lgbm(city, spec)
        model.fit(x_train, target_train)
        return np.maximum(0.0, model.predict(x_predict))

    if spec.kind == "catboost":
        model = _catboost(spec)
        model.fit(x_train, target_train)
        return np.maximum(0.0, np.asarray(model.predict(x_predict), dtype=float))

    raise ValueError(f"Unsupported model kind: {spec.kind}")


def _prepare_city(
    train: pd.DataFrame,
    test: pd.DataFrame,
    city: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    train_city = train.loc[train["city"].eq(city)].copy()
    test_city = test.loc[test["city"].eq(city)].copy()
    train_city["_is_train"] = True
    test_city["_is_train"] = False
    test_city[TARGET_COLUMN] = np.nan
    combined = (
        pd.concat([train_city, test_city], ignore_index=True, sort=False)
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    train_mask = combined["_is_train"].to_numpy(bool)
    features = build_features(combined, train_mask)
    train_positions = np.flatnonzero(train_mask)
    _check_case_feature(combined, features, train_positions)
    return (
        features,
        combined[TARGET_COLUMN].to_numpy(float),
        combined["weekofyear"].to_numpy(int),
        train_positions,
        combined,
    )


def _check_case_feature(
    combined: pd.DataFrame,
    features: pd.DataFrame,
    train_positions: np.ndarray,
) -> None:
    """The same-week case mean at row i may use only labels from rows before i."""
    week = combined["weekofyear"].to_numpy(int)
    target = combined[TARGET_COLUMN].to_numpy(float)
    column = features["cases_same_week_past_mean"].to_numpy(float)
    start = fold_starts(len(train_positions))[0]
    for local in range(start, start + FOLD_WEEKS):
        index = int(train_positions[local])
        earlier = week[:index] == week[index]
        if not earlier.any():
            if np.isfinite(column[index]):
                raise AssertionError("Same-week case feature is defined without history")
            continue
        expected = float(np.mean(target[:index][earlier]))
        if not np.isclose(column[index], expected):
            raise AssertionError("Same-week case feature uses future or current labels")


def run_city(
    city: str,
    features: pd.DataFrame,
    target: np.ndarray,
    week: np.ndarray,
    train_positions: np.ndarray,
    combined: pd.DataFrame,
    forest_trees: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, float | str | int]] = []
    prediction_rows: list[pd.DataFrame] = []
    starts = fold_starts(len(train_positions))
    for spec in SPECS:
        print(f"  {city} {spec.name}", flush=True)
        for fold, local_start in enumerate(starts, start=1):
            local_train = train_positions[:local_start]
            local_valid = train_positions[local_start : local_start + FOLD_WEEKS]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = _predict_spec(
                    spec,
                    city,
                    features,
                    target,
                    week,
                    local_train,
                    local_valid,
                    forest_trees,
                )
            actual = target[local_valid]
            predicted = integer_cases(raw)
            threshold = float(np.quantile(target[local_train], 0.90))
            outbreak = actual >= threshold
            score_rows.append(
                {
                    "city": city,
                    "model": spec.name,
                    "fold": fold,
                    "n_predictions": int(len(actual)),
                    "mae": mean_absolute_error(actual, predicted),
                    "bias": float(np.mean(predicted - actual)),
                    "outbreak_mae": (
                        mean_absolute_error(actual[outbreak], predicted[outbreak])
                        if outbreak.any()
                        else np.nan
                    ),
                    "prediction_mean": float(np.mean(predicted)),
                    "actual_mean": float(np.mean(actual)),
                }
            )
            rows = combined.iloc[local_valid][
                ["city", "year", "weekofyear", DATE_COLUMN]
            ].copy()
            rows["fold"] = fold
            rows["model"] = spec.name
            rows["actual"] = actual.astype(int)
            rows["prediction"] = predicted
            rows["raw_prediction"] = raw
            prediction_rows.append(rows)
    return pd.DataFrame(score_rows), pd.concat(prediction_rows, ignore_index=True)


def _pooled(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, model), frame in scores.groupby(["city", "model"], sort=False):
        rows.append(
            {
                "city": city,
                "model": model,
                "mae": float(np.average(frame["mae"], weights=frame["n_predictions"])),
                "bias": float(np.average(frame["bias"], weights=frame["n_predictions"])),
                "outbreak_mae": float(np.nanmean(frame["outbreak_mae"])),
            }
        )
    return pd.DataFrame(rows)


def _attach_tree(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    trees = pd.read_csv(CONFIRMED_TREE, parse_dates=[DATE_COLUMN])
    merged = predictions.merge(
        trees[[DATE_COLUMN, "city", "tree_prediction", "fold"]],
        on=[DATE_COLUMN, "city", "fold"],
        how="left",
        validate="many_to_one",
    )
    if merged["tree_prediction"].isna().any() or len(merged) != len(predictions):
        raise RuntimeError("New validation rows do not match the confirmed tree folds")
    return merged


def select_city(
    city: str,
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, float | str]:
    """Choose a blend on folds 1–4, and ship it only if folds 5–6 also improve.

    Every model and every blend weight is eligible. A candidate that wins the
    selection folds but loses the last two folds is discarded, so the search
    can fall through to the next honest improvement, including climatology.
    """
    del scores  # Fold scores are already implied by the prediction rows.
    reference = predictions.loc[
        predictions["city"].eq(city) & predictions["model"].eq("climatology")
    ].sort_values(["fold", DATE_COLUMN])
    actual = reference["actual"].to_numpy(float)
    tree = reference["tree_prediction"].to_numpy(float)
    folds = reference["fold"].to_numpy(int)
    dates = reference[DATE_COLUMN].to_numpy()
    selection = np.isin(folds, [1, 2, 3, 4])
    confirmation = np.isin(folds, [5, 6])
    outbreak = actual >= np.quantile(actual, 0.90)

    def summarize(prediction: np.ndarray) -> dict[str, float]:
        rounded = integer_cases(prediction)
        error = np.abs(rounded - actual)
        return {
            "selection": float(error[selection].mean()),
            "confirmation": float(error[confirmation].mean()),
            "overall": float(error.mean()),
            "outbreak": float(error[outbreak].mean()),
        }

    tree_score = summarize(tree)
    candidates: list[dict[str, float | str]] = []
    for model in predictions.loc[predictions["city"].eq(city), "model"].unique():
        frame = predictions.loc[
            predictions["city"].eq(city) & predictions["model"].eq(model)
        ].sort_values(["fold", DATE_COLUMN])
        if not np.array_equal(frame[DATE_COLUMN].to_numpy(), dates):
            raise RuntimeError(f"{city} {model} validation weeks are misaligned")
        raw = frame["raw_prediction"].to_numpy(float)
        if not np.isfinite(raw).all() or float(np.nanmax(np.abs(raw))) > 1e5:
            continue
        for weight in np.round(np.linspace(0.0, 1.0, 11), 2):
            score = summarize(weight * raw + (1.0 - weight) * tree)
            candidates.append({"model": str(model), "weight": float(weight), **score})

    confirmed = [
        candidate
        for candidate in candidates
        if float(candidate["confirmation"]) <= tree_score["confirmation"] + 1e-9
    ]
    winner = min(
        confirmed,
        key=lambda candidate: (
            float(candidate["selection"]),
            float(candidate["confirmation"]),
            float(candidate["overall"]),
        ),
    )
    weight = float(winner["weight"])
    shipped_model = str(winner["model"]) if weight > 0 else "confirmed_tree"
    return {
        "city": city,
        "selected_on_folds_1_to_4": shipped_model,
        "selection_mae": float(winner["selection"]),
        "blend_weight_on_new_model": weight,
        "confirmation_mae": float(winner["confirmation"]),
        "tree_confirmation_mae": tree_score["confirmation"],
        "tree_selection_mae": tree_score["selection"],
        "shipped_weight_on_new_model": weight,
        "shipped_model": shipped_model,
        "shipped_confirmation_mae": float(winner["confirmation"]),
        "shipped_six_fold_mae": float(winner["overall"]),
        "tree_six_fold_mae": tree_score["overall"],
        "shipped_outbreak_mae": float(winner["outbreak"]),
        "tree_outbreak_mae": tree_score["outbreak"],
    }


def fit_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    decisions: dict[str, dict[str, float | str]],
    forest_trees: int,
) -> pd.DataFrame:
    tree_submission = pd.read_csv(CONFIRMED_SUBMISSION)
    rows: list[pd.DataFrame] = []
    for city, decision in decisions.items():
        features, target, week, train_positions, combined = _prepare_city(
            train, test, city
        )
        test_positions = np.flatnonzero(~combined["_is_train"].to_numpy(bool))
        weight = float(decision["shipped_weight_on_new_model"])
        if weight == 0.0:
            prediction = None
        else:
            spec = next(
                item
                for item in SPECS
                if item.name == decision["selected_on_folds_1_to_4"]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = _predict_spec(
                    spec,
                    city,
                    features,
                    target,
                    week,
                    train_positions,
                    test_positions,
                    forest_trees if spec.kind != "forest" else max(forest_trees, 500),
                )
            prediction = raw
        result = combined.iloc[test_positions][KEY_COLUMNS].copy()
        tree_values = tree_submission.merge(
            result[KEY_COLUMNS], on=KEY_COLUMNS, how="right", validate="one_to_one"
        )[TARGET_COLUMN].to_numpy(float)
        if prediction is None:
            result[TARGET_COLUMN] = integer_cases(tree_values)
        else:
            result[TARGET_COLUMN] = integer_cases(
                weight * prediction + (1.0 - weight) * tree_values
            )
        rows.append(result)
    predictions = pd.concat(rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        predictions, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, template = load_data(ROOT / "data")
    forest_trees = 200
    score_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for city in ("iq", "sj"):
        print(f"\n{city}: building features", flush=True)
        features, target, week, train_positions, combined = _prepare_city(
            train, test, city
        )
        print(f"  {features.shape[1]} features", flush=True)
        scores, predictions = run_city(
            city,
            features,
            target,
            week,
            train_positions,
            combined,
            forest_trees,
        )
        score_frames.append(scores)
        prediction_frames.append(predictions)

    scores = pd.concat(score_frames, ignore_index=True)
    predictions = _attach_tree(pd.concat(prediction_frames, ignore_index=True))
    pooled = _pooled(scores)
    decisions = {
        city: select_city(city, scores, predictions) for city in ("iq", "sj")
    }
    submission = fit_submission(train, test, template, decisions, forest_trees)

    scores.to_csv(OUTPUT_DIR / "validation_fold_scores.csv", index=False)
    pooled.to_csv(OUTPUT_DIR / "validation_pooled_scores.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    (OUTPUT_DIR / "selected_models.json").write_text(
        json.dumps(decisions, indent=2) + "\n", encoding="utf-8"
    )
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    print("\nPooled six-fold MAE")
    print(
        pooled.pivot(index="model", columns="city", values="mae")
        .round(3)
        .sort_values("sj")
        .to_string()
    )
    print("\nShipped decisions")
    print(json.dumps(decisions, indent=2))
    print("\nSubmission")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
