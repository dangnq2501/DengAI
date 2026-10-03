"""Evaluate an SJ Extra Trees + climate-aware EGARCH-X hybrid.

Extra Trees predicts the conditional case-count level. EGARCH-X is fitted to
chronological tree residuals and models their conditional variance using past
shocks plus weather covariates. Validation may select a volatility-dependent
point correction, a constant median-residual correction, or the unchanged tree
fallback. The confirmed submission is never overwritten.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
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
WARMUP_FOLDS = 2
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
CORRECTION_GRID = tuple(np.round(np.arange(-1.0, 2.01, 0.25), 2))
GARCH_X_COLUMNS = [
    "week_sin_1",
    "week_cos_1",
    "station_avg_temp_c",
    "station_precip_mm",
    "precipitation_amt_mm",
    "reanalysis_relative_humidity_percent",
    "reanalysis_specific_humidity_g_per_kg",
    "temperature_humidity_interaction",
    "station_avg_temp_c__lag_4",
    "precipitation_amt_mm__lag_4",
    "reanalysis_relative_humidity_percent__lag_4",
]


@dataclass
class EgarchX:
    """Gaussian EGARCH(1,1)-X model for residual conditional variance."""

    max_iter: int = 1000
    ridge: float = 0.01

    def _variance_path(
        self,
        residual: np.ndarray,
        x: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        omega, alpha, beta, theta = parameters[:4]
        gamma = parameters[4:]
        log_variance = np.empty(len(residual), dtype=float)
        log_variance[0] = np.log(max(float(np.var(residual)), 1.0))
        for index in range(1, len(residual)):
            previous_sd = np.exp(0.5 * np.clip(log_variance[index - 1], -10, 15))
            shock = residual[index - 1] / max(previous_sd, 1e-6)
            log_variance[index] = (
                omega
                + beta * log_variance[index - 1]
                + alpha * (abs(shock) - SQRT_2_OVER_PI)
                + theta * shock
                + x[index] @ gamma
            )
            log_variance[index] = np.clip(log_variance[index], -10.0, 15.0)
        return np.exp(log_variance)

    def fit(self, residual: np.ndarray, exogenous: pd.DataFrame) -> "EgarchX":
        residual = np.asarray(residual, dtype=float)
        if len(residual) != len(exogenous) or len(residual) < 52:
            raise ValueError("EGARCH-X requires aligned residuals and at least 52 rows")
        self.columns_ = list(exogenous.columns)
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        x = self.scaler_.fit_transform(self.imputer_.fit_transform(exogenous))
        initial_variance = max(float(np.var(residual)), 1.0)
        initial = np.zeros(4 + x.shape[1], dtype=float)
        initial[0] = 0.15 * np.log(initial_variance)
        initial[1] = 0.1
        initial[2] = 0.85

        def objective(parameters: np.ndarray) -> float:
            variance = self._variance_path(residual, x, parameters)
            gaussian_nll = 0.5 * np.mean(
                np.log(variance) + residual * residual / variance
            )
            return float(gaussian_nll + 0.5 * self.ridge * np.dot(parameters[4:], parameters[4:]))

        bounds = [(-10.0, 10.0), (0.0, 2.0), (0.0, 0.995), (-1.0, 1.0)]
        bounds.extend([(-1.5, 1.5)] * x.shape[1])
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-10},
        )
        self.parameters_ = result.x
        self.converged_ = bool(result.success)
        self.optimizer_message_ = str(result.message)
        self.residual_ = residual.copy()
        self.x_ = x
        self.fitted_variance_ = self._variance_path(residual, x, self.parameters_)
        self.typical_sigma_ = float(np.median(np.sqrt(self.fitted_variance_)))
        return self

    def forecast_sigma(self, exogenous: pd.DataFrame) -> np.ndarray:
        x_future = self.scaler_.transform(self.imputer_.transform(exogenous[self.columns_]))
        x_future = np.clip(x_future, -6.0, 6.0)
        omega, alpha, beta, theta = self.parameters_[:4]
        gamma = self.parameters_[4:]
        last_variance = float(self.fitted_variance_[-1])
        last_shock = self.residual_[-1] / max(np.sqrt(last_variance), 1e-6)
        forecast = np.empty(len(x_future), dtype=float)
        for index, row in enumerate(x_future):
            if index == 0:
                shock_size = abs(last_shock) - SQRT_2_OVER_PI
                signed_shock = last_shock
            else:
                # Multi-step forecast integrates unknown future shocks out.
                shock_size = 0.0
                signed_shock = 0.0
            log_variance = (
                omega
                + beta * np.log(max(last_variance, 1e-8))
                + alpha * shock_size
                + theta * signed_shock
                + row @ gamma
            )
            last_variance = float(np.exp(np.clip(log_variance, -10.0, 15.0)))
            forecast[index] = np.sqrt(last_variance)
        return forecast


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    return parser.parse_args()


def arch_lm_test(residual: np.ndarray, lags: int = 4) -> dict[str, float]:
    """Return the elementary Engle ARCH-LM statistic and chi-square p-value."""
    residual = np.asarray(residual, dtype=float)
    dependent = residual[lags:] ** 2
    design = np.column_stack(
        [np.ones(len(dependent))]
        + [residual[lags - lag : -lag] ** 2 for lag in range(1, lags + 1)]
    )
    fitted = design @ np.linalg.lstsq(design, dependent, rcond=None)[0]
    total = np.sum((dependent - dependent.mean()) ** 2)
    r_squared = max(0.0, 1.0 - np.sum((dependent - fitted) ** 2) / total)
    statistic = len(dependent) * r_squared
    return {"lags": lags, "statistic": statistic, "p_value": float(chi2.sf(statistic, lags))}


def sj_oof_predictions(
    train: pd.DataFrame, random_state: int, n_estimators: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    city = (
        train.loc[train["city"].eq(SJ)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    features = make_features(city)
    target = city[TARGET_COLUMN].to_numpy(float)
    starts = list(range(len(city) - FOLD_COUNT * 52, len(city), 52))
    rows: list[pd.DataFrame] = []
    for fold, start in enumerate(starts, start=1):
        train_index = np.arange(start)
        validation_index = np.arange(start, start + 52)
        model = tree_model(SJ, random_state, n_estimators)
        model.fit(features.iloc[train_index], target[train_index])
        result = city.iloc[validation_index][KEY_COLUMNS + [DATE_COLUMN]].copy()
        result["fold"] = fold
        result["actual"] = target[validation_index].astype(int)
        result["tree_raw"] = model.predict(features.iloc[validation_index])
        result["tree_prediction"] = _integer_cases(result["tree_raw"])
        rows.append(result)
        print(f"SJ tree fold {fold}/{FOLD_COUNT} complete")
    return pd.concat(rows, ignore_index=True), features


def sequential_garch_forecasts(
    oof: pd.DataFrame, full_features: pd.DataFrame, train: pd.DataFrame
) -> tuple[pd.DataFrame, list[EgarchX]]:
    city = (
        train.loc[train["city"].eq(SJ)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    feature_keys = city[KEY_COLUMNS].copy()
    feature_keys[GARCH_X_COLUMNS] = full_features[GARCH_X_COLUMNS]
    enriched = oof.merge(feature_keys, on=KEY_COLUMNS, how="left", validate="one_to_one")
    enriched["residual"] = enriched["actual"] - enriched["tree_raw"]
    enriched["garch_sigma"] = np.nan
    enriched["history_bias"] = np.nan
    models: list[EgarchX] = []
    for fold in range(WARMUP_FOLDS + 1, FOLD_COUNT + 1):
        history = enriched.loc[enriched["fold"] < fold]
        validation = enriched.loc[enriched["fold"] == fold]
        model = EgarchX().fit(
            history["residual"].to_numpy(float), history[GARCH_X_COLUMNS]
        )
        sigma = model.forecast_sigma(validation[GARCH_X_COLUMNS])
        enriched.loc[validation.index, "garch_sigma"] = sigma
        enriched.loc[validation.index, "history_bias"] = float(
            np.median(history["residual"])
        )
        enriched.loc[validation.index, "typical_sigma"] = model.typical_sigma_
        models.append(model)
        print(
            f"EGARCH-X fold {fold}/{FOLD_COUNT} complete "
            f"converged={model.converged_}"
        )
    return enriched, models


def select_correction(validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = validation.loc[validation["fold"] > WARMUP_FOLDS].copy()
    actual = scored["actual"].to_numpy(int)
    tree_raw = scored["tree_raw"].to_numpy(float)
    baseline_fold = [
        mean_absolute_error(group["actual"], group["tree_prediction"])
        for _, group in scored.groupby("fold")
    ]
    baseline_mae = mean_absolute_error(actual, _integer_cases(tree_raw))
    baseline_robust = float(np.mean(baseline_fold) + 0.2 * np.std(baseline_fold))
    baseline_prediction = _integer_cases(tree_raw)
    outbreak_threshold = float(np.quantile(actual, 0.90))
    outbreak = actual >= outbreak_threshold
    baseline_outbreak_mae = mean_absolute_error(
        actual[outbreak], baseline_prediction[outbreak]
    )
    baseline_mean = float(np.mean(baseline_prediction))
    baseline_max = float(np.max(baseline_prediction))
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, float]] = [("tree", 0.0), ("bias", 0.0)]
    candidates.extend(("garch_x", value) for value in CORRECTION_GRID)
    for mode, scale in candidates:
        if mode == "tree":
            raw = tree_raw
        elif mode == "bias":
            raw = tree_raw + scored["history_bias"].to_numpy(float)
        else:
            volatility_signal = (
                scored["garch_sigma"] - scored["typical_sigma"]
            ).to_numpy(float)
            raw = tree_raw + scored["history_bias"].to_numpy(float) + scale * volatility_signal
        prediction = _integer_cases(raw)
        temp = scored[["fold", "actual"]].copy()
        temp["prediction"] = prediction
        fold_mae = [
            mean_absolute_error(group["actual"], group["prediction"])
            for _, group in temp.groupby("fold")
        ]
        pooled_mae = mean_absolute_error(actual, prediction)
        robust = float(np.mean(fold_mae) + 0.2 * np.std(fold_mae))
        outbreak_mae = mean_absolute_error(actual[outbreak], prediction[outbreak])
        amplitude_ok = (
            np.mean(prediction) >= 0.90 * baseline_mean
            and np.mean(prediction) <= 1.10 * baseline_mean
            and np.max(prediction) >= 0.90 * baseline_max
        )
        outbreak_ok = outbreak_mae <= 1.03 * baseline_outbreak_mae
        rows.append(
            {
                "mode": mode,
                "volatility_scale": scale,
                "pooled_mae": pooled_mae,
                "mean_fold_mae": float(np.mean(fold_mae)),
                "std_fold_mae": float(np.std(fold_mae)),
                "robust_score": robust,
                "outbreak_mae": outbreak_mae,
                "prediction_mean": float(np.mean(prediction)),
                "prediction_max": float(np.max(prediction)),
                "amplitude_ok": amplitude_ok,
                "outbreak_ok": outbreak_ok,
                "eligible": amplitude_ok and outbreak_ok,
                "improvement_vs_tree": baseline_mae - pooled_mae,
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["robust_score", "pooled_mae", "mode", "volatility_scale"]
    ).reset_index(drop=True)
    winner = results.loc[results["eligible"]].iloc[0]
    if winner["pooled_mae"] < baseline_mae and winner["robust_score"] < baseline_robust:
        mode = str(winner["mode"])
        scale = float(winner["volatility_scale"])
    else:
        mode, scale = "tree", 0.0
    selected: dict[str, Any] = {
        "mode": mode,
        "volatility_scale": scale,
        "tree_oof_mae": float(baseline_mae),
        "tree_robust_score": baseline_robust,
        "tree_outbreak_mae": float(baseline_outbreak_mae),
    }
    return results, selected


def fit_final(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    oof: pd.DataFrame,
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
        train_index = np.flatnonzero(combined["_split"].eq("train").to_numpy())
        test_index = np.flatnonzero(combined["_split"].eq("test").to_numpy())
        features = make_features(combined)
        target = combined.iloc[train_index][TARGET_COLUMN].to_numpy(float)
        tree = tree_model(city, random_state, n_estimators)
        tree.fit(features.iloc[train_index], target)
        raw = tree.predict(features.iloc[test_index])
        city_artifacts: dict[str, Any] = {"tree": tree}
        if city == SJ and selected["mode"] != "tree":
            residual = (oof["actual"] - oof["tree_raw"]).to_numpy(float)
            # OOF rows are the final contiguous 312 training weeks.
            oof_features = make_features(train_city.sort_values(DATE_COLUMN).reset_index(drop=True)).iloc[-len(oof):]
            garch = EgarchX().fit(residual, oof_features[GARCH_X_COLUMNS])
            bias = float(np.median(residual))
            if selected["mode"] == "bias":
                raw = raw + bias
            else:
                sigma = garch.forecast_sigma(features.iloc[test_index][GARCH_X_COLUMNS])
                raw = raw + bias + selected["volatility_scale"] * (
                    sigma - garch.typical_sigma_
                )
            city_artifacts["garch_x"] = garch
            city_artifacts["residual_bias"] = bias
        result = combined.iloc[test_index][KEY_COLUMNS].copy()
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
    oof, sj_features = sj_oof_predictions(train, args.random_state, args.n_estimators)
    residual = oof["actual"] - oof["tree_prediction"]
    diagnostic = arch_lm_test(residual.to_numpy(float), lags=4)
    validation, models = sequential_garch_forecasts(oof, sj_features, train)
    search, selected = select_correction(validation)
    selected["arch_lm"] = diagnostic
    scored = validation.loc[validation["fold"] > WARMUP_FOLDS].copy()
    if selected["mode"] == "tree":
        scored["garch_ensemble_prediction"] = scored["tree_prediction"]
    elif selected["mode"] == "bias":
        scored["garch_ensemble_prediction"] = _integer_cases(
            scored["tree_raw"] + scored["history_bias"]
        )
    else:
        scored["garch_ensemble_prediction"] = _integer_cases(
            scored["tree_raw"]
            + scored["history_bias"]
            + selected["volatility_scale"]
            * (scored["garch_sigma"] - scored["typical_sigma"])
        )
    prediction_columns = KEY_COLUMNS + [
        DATE_COLUMN,
        "fold",
        "actual",
        "tree_prediction",
        "garch_sigma",
        "history_bias",
        "garch_ensemble_prediction",
    ]
    scored[prediction_columns].to_csv(
        args.output_dir / "sj_garchx_validation_predictions.csv", index=False
    )
    search.to_csv(args.output_dir / "sj_garchx_search_results.csv", index=False)
    (args.output_dir / "sj_garchx_selected_config.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    submission, artifacts = fit_final(
        train,
        test,
        template,
        oof,
        selected,
        args.random_state,
        args.n_estimators,
    )
    submission.to_csv(args.output_dir / "submission_tree_sj_garchx.csv", index=False)
    artifacts["sequential_validation_models"] = models
    joblib.dump(artifacts, args.output_dir / "tree_sj_garchx_models.joblib")
    print("\nARCH-LM diagnostic:")
    print(json.dumps(diagnostic, indent=2))
    print("\nSelected correction:")
    print(json.dumps(selected, indent=2))
    print("\nSearch results:")
    print(search.to_string(index=False))
    print("\nSubmission distribution:")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
