"""Test a negative-binomial model as a guarded SJ Extra Trees ensemble.

The DengAI target is an overdispersed count, so NB2 is a meaningful alternative
to test.  This script does not replace the confirmed tree solution. It selects
the negative-binomial regularization, dispersion scale, and blend weight using
expanding chronological folds; weight zero is always an available fallback.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

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
FOLD_COUNT = 6
L2_GRID = (0.01, 0.1, 1.0, 10.0)
DISPERSION_SCALE_GRID = (0.5, 1.0, 2.0)
BLEND_WEIGHT_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.05), 2))


@dataclass
class NegativeBinomialRegressor:
    """Regularized log-link NB2 regression with method-of-moments dispersion."""

    l2: float = 0.1
    dispersion_scale: float = 1.0
    max_iter: int = 500

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "NegativeBinomialRegressor":
        # A linear log-link year term extrapolates exponentially into the test
        # period. Seasonal harmonics retain time structure without that hazard.
        self.feature_columns_ = [c for c in features.columns if c != "year"]
        matrix = features[self.feature_columns_]
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        x = self.imputer_.fit_transform(matrix)
        x = self.scaler_.fit_transform(x)
        y = np.asarray(target, dtype=float)
        mean = max(float(y.mean()), 1e-6)
        variance = float(y.var())
        moment_alpha = max((variance - mean) / (mean * mean), 1e-4)
        self.alpha_ = max(moment_alpha * self.dispersion_scale, 1e-6)
        size = 1.0 / self.alpha_
        initial = np.zeros(x.shape[1] + 1)
        initial[0] = np.log(mean)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            eta = np.clip(parameters[0] + x @ parameters[1:], -10.0, 10.0)
            mu = np.exp(eta)
            log_likelihood = (
                gammaln(y + size)
                - gammaln(size)
                - gammaln(y + 1.0)
                + size * (np.log(size) - np.log(size + mu))
                + y * (eta - np.log(size + mu))
            )
            residual = ((y + size) * mu / (size + mu)) - y
            penalty = 0.5 * self.l2 * np.dot(parameters[1:], parameters[1:])
            loss = -float(log_likelihood.mean()) + penalty
            gradient = np.empty_like(parameters)
            gradient[0] = residual.mean()
            gradient[1:] = x.T @ residual / len(y) + self.l2 * parameters[1:]
            return loss, gradient

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )
        self.coef_ = result.x[1:]
        self.intercept_ = float(result.x[0])
        self.converged_ = bool(result.success)
        self.optimizer_message_ = str(result.message)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        x = self.imputer_.transform(features[self.feature_columns_])
        x = np.clip(self.scaler_.transform(x), -6.0, 6.0)
        eta = np.clip(self.intercept_ + x @ self.coef_, -10.0, 10.0)
        return np.exp(eta)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    return parser.parse_args()


def _sj_fold_data(train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    city = (
        train.loc[train["city"].eq(SJ)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    starts = list(range(len(city) - FOLD_COUNT * 52, len(city), 52))
    return city, city[TARGET_COLUMN].to_numpy(float), starts


def generate_oof_predictions(
    train: pd.DataFrame,
    random_state: int,
    n_estimators: int,
) -> pd.DataFrame:
    city, target, starts = _sj_fold_data(train)
    features = make_features(city)
    configs = list(itertools.product(L2_GRID, DISPERSION_SCALE_GRID))
    rows: list[pd.DataFrame] = []
    for fold, start in enumerate(starts, start=1):
        train_index = np.arange(start)
        validation_index = np.arange(start, start + 52)
        tree = tree_model(SJ, random_state, n_estimators)
        tree.fit(features.iloc[train_index], target[train_index])
        fold_rows = city.iloc[validation_index][KEY_COLUMNS + [DATE_COLUMN]].copy()
        fold_rows.insert(0, "fold", fold)
        fold_rows["actual"] = target[validation_index].astype(int)
        fold_rows["tree_raw"] = tree.predict(features.iloc[validation_index])
        for l2, scale in configs:
            model = NegativeBinomialRegressor(l2=l2, dispersion_scale=scale)
            model.fit(features.iloc[train_index], target[train_index])
            name = f"nb_l2_{l2:g}_scale_{scale:g}"
            fold_rows[name] = model.predict(features.iloc[validation_index])
        rows.append(fold_rows)
        print(f"SJ chronological fold {fold}/{len(starts)} complete")
    return pd.concat(rows, ignore_index=True)


def select_blend(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    actual = oof["actual"].to_numpy(int)
    tree_raw = oof["tree_raw"].to_numpy(float)
    tree_fold_mae = [
        mean_absolute_error(group["actual"], _integer_cases(group["tree_raw"]))
        for _, group in oof.groupby("fold")
    ]
    baseline_mae = mean_absolute_error(actual, _integer_cases(tree_raw))
    baseline_robust = float(np.mean(tree_fold_mae) + 0.2 * np.std(tree_fold_mae))
    rows: list[dict[str, Any]] = []
    for l2, scale, weight in itertools.product(
        L2_GRID, DISPERSION_SCALE_GRID, BLEND_WEIGHT_GRID
    ):
        column = f"nb_l2_{l2:g}_scale_{scale:g}"
        blended_raw = (1.0 - weight) * tree_raw + weight * oof[column].to_numpy(float)
        prediction = _integer_cases(blended_raw)
        fold_mae = [
            mean_absolute_error(
                group["actual"],
                _integer_cases(
                    (1.0 - weight) * group["tree_raw"] + weight * group[column]
                ),
            )
            for _, group in oof.groupby("fold")
        ]
        pooled_mae = mean_absolute_error(actual, prediction)
        robust_score = float(np.mean(fold_mae) + 0.2 * np.std(fold_mae))
        rows.append(
            {
                "l2": l2,
                "dispersion_scale": scale,
                "nb_weight": weight,
                "pooled_mae": pooled_mae,
                "mean_fold_mae": float(np.mean(fold_mae)),
                "std_fold_mae": float(np.std(fold_mae)),
                "robust_score": robust_score,
                "improvement_vs_tree": baseline_mae - pooled_mae,
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["robust_score", "pooled_mae", "nb_weight", "l2", "dispersion_scale"]
    ).reset_index(drop=True)
    winner = results.iloc[0]
    # Require improvement in both pooled MAE and the stability-aware score.
    if not (
        winner["pooled_mae"] < baseline_mae
        and winner["robust_score"] < baseline_robust
    ):
        selected = {"l2": 0.1, "dispersion_scale": 1.0, "nb_weight": 0.0}
    else:
        selected = {
            "l2": float(winner["l2"]),
            "dispersion_scale": float(winner["dispersion_scale"]),
            "nb_weight": float(winner["nb_weight"]),
        }
    selected["tree_oof_mae"] = float(baseline_mae)
    selected["tree_robust_score"] = baseline_robust
    return results, selected


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    selected: dict[str, float],
    random_state: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions: list[pd.DataFrame] = []
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
        features = make_features(combined)
        target = combined.iloc[train_index][TARGET_COLUMN].to_numpy(float)
        tree = tree_model(city, random_state, n_estimators)
        tree.fit(features.iloc[train_index], target)
        tree_raw = tree.predict(features.iloc[test_index])
        result = combined.iloc[test_index][KEY_COLUMNS].copy()
        city_artifact: dict[str, Any] = {"tree": tree}
        if city == SJ and selected["nb_weight"] > 0.0:
            nb = NegativeBinomialRegressor(
                l2=selected["l2"],
                dispersion_scale=selected["dispersion_scale"],
            )
            nb.fit(features.iloc[train_index], target)
            nb_raw = nb.predict(features.iloc[test_index])
            raw = (1.0 - selected["nb_weight"]) * tree_raw + selected["nb_weight"] * nb_raw
            city_artifact["negative_binomial"] = nb
        else:
            raw = tree_raw
        result[TARGET_COLUMN] = _integer_cases(raw)
        predictions.append(result)
        artifacts["cities"][city] = city_artifact
    predicted = pd.concat(predictions, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        predicted, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission, artifacts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, template = load_data(args.data_dir)
    oof = generate_oof_predictions(train, args.random_state, args.n_estimators)
    search, selected = select_blend(oof)
    selected_column = (
        f"nb_l2_{selected['l2']:g}_scale_{selected['dispersion_scale']:g}"
    )
    oof["nb_prediction"] = _integer_cases(oof[selected_column])
    oof["ensemble_prediction"] = _integer_cases(
        (1.0 - selected["nb_weight"]) * oof["tree_raw"]
        + selected["nb_weight"] * oof[selected_column]
    )
    oof["tree_prediction"] = _integer_cases(oof["tree_raw"])
    keep = KEY_COLUMNS + [
        DATE_COLUMN,
        "fold",
        "actual",
        "tree_prediction",
        "nb_prediction",
        "ensemble_prediction",
    ]
    oof[keep].to_csv(args.output_dir / "sj_nb_validation_predictions.csv", index=False)
    search.to_csv(args.output_dir / "sj_nb_search_results.csv", index=False)
    (args.output_dir / "sj_nb_selected_config.json").write_text(
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
    submission.to_csv(args.output_dir / "submission_tree_sj_nb.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "tree_sj_nb_models.joblib")
    print("\nSelected SJ blend:")
    print(json.dumps(selected, indent=2))
    print("\nTop search results:")
    print(search.head(10).to_string(index=False))
    print("\nSubmission distribution:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
