"""Reproduce and extend the 12.67-MAE vendor climate-history neural network.

The original notebook used TensorFlow 2.1. This module implements its small
SELU/dropout/RMSprop network in NumPy so it remains runnable in the project's
current Python environment. Four candidates are evaluated independently for
each city:

* vendor_shift2: original climate windows and original two-row output offset;
* vendor_aligned: original climate windows with corrected row alignment;
* enhanced_shift2: original offset plus this project's engineered features;
* enhanced_aligned: corrected alignment plus engineered features.

Every submission is written separately. The confirmed 22.5 tree submission is
never modified.
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
from sklearn.metrics import mean_absolute_error

from train_feature_engineering import make_feature_mode
from train_tree_ensemble import (
    DATE_COLUMN,
    KEY_COLUMNS,
    TARGET_COLUMN,
    load_data,
    make_features,
)


WINDOWS: dict[str, dict[str, int]] = {
    "sj": {
        "reanalysis_dew_point_temp_k": 39,
        "reanalysis_min_air_temp_k": 21,
        "reanalysis_relative_humidity_percent": 34,
        "reanalysis_specific_humidity_g_per_kg": 14,
        "weekofyear": 1,
        "year": 1,
    },
    "iq": {
        "reanalysis_air_temp_k": 10,
        "reanalysis_avg_temp_k": 4,
        "reanalysis_dew_point_temp_k": 6,
        "reanalysis_max_air_temp_k": 41,
        "reanalysis_min_air_temp_k": 40,
        "reanalysis_specific_humidity_g_per_kg": 26,
        "station_max_temp_c": 39,
        "station_min_temp_c": 25,
        "weekofyear": 1,
        "year": 1,
    },
}
HISTORY = 52
CANDIDATES = (
    "vendor_shift2",
    "vendor_aligned",
    "enhanced_shift2",
    "enhanced_aligned",
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--seeds", default="42,137,911")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=200,
        help="Batches per validation epoch; final fits always use 200.",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Only fit final models."
    )
    parser.add_argument(
        "--skip-final", action="store_true", help="Only run chronological validation."
    )
    return parser.parse_args()


def _selu(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0.0, x, alpha * np.expm1(np.clip(x, -30, 0)))


def _selu_gradient(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0.0, 1.0, alpha * np.exp(np.clip(x, -30, 0)))


@dataclass
class SeluMAENetwork:
    hidden_units: int
    epochs: int
    steps_per_epoch: int = 200
    batch_size: int = 16
    learning_rate: float = 0.01
    dropout: float = 0.5
    random_state: int = 42

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SeluMAENetwork":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        second_units = max(1, self.hidden_units // 4)

        def glorot(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.weights_ = [
            glorot(x.shape[1], self.hidden_units),
            np.zeros((1, self.hidden_units)),
            glorot(self.hidden_units, second_units),
            np.zeros((1, second_units)),
            glorot(second_units, 1),
            np.zeros((1, 1)),
        ]
        caches = [np.zeros_like(value) for value in self.weights_]
        rho, epsilon = 0.9, 1e-7
        order = rng.permutation(len(x))
        position = 0
        learning_rate = self.learning_rate
        keep_probability = 1.0 - self.dropout
        for epoch in range(self.epochs):
            for _ in range(self.steps_per_epoch):
                if position >= len(order):
                    order = rng.permutation(len(x))
                    position = 0
                indices = order[position : position + self.batch_size]
                position += len(indices)
                xb, yb = x[indices], y[indices]
                w1, b1, w2, b2, w3, b3 = self.weights_
                z1 = xb @ w1 + b1
                a1 = _selu(z1)
                mask = rng.random(a1.shape) < keep_probability
                dropped = a1 * mask / keep_probability
                z2 = dropped @ w2 + b2
                a2 = _selu(z2)
                prediction = a2 @ w3 + b3
                gradient = np.sign(prediction - yb) / len(indices)
                gradients: list[np.ndarray] = [np.empty(0)] * 6
                gradients[4] = a2.T @ gradient
                gradients[5] = gradient.sum(axis=0, keepdims=True)
                da2 = gradient @ w3.T
                dz2 = da2 * _selu_gradient(z2)
                gradients[2] = dropped.T @ dz2
                gradients[3] = dz2.sum(axis=0, keepdims=True)
                da1 = (dz2 @ w2.T) * mask / keep_probability
                dz1 = da1 * _selu_gradient(z1)
                gradients[0] = xb.T @ dz1
                gradients[1] = dz1.sum(axis=0, keepdims=True)
                for index, grad in enumerate(gradients):
                    grad = np.clip(grad, -100.0, 100.0)
                    caches[index] = rho * caches[index] + (1.0 - rho) * grad**2
                    self.weights_[index] -= (
                        learning_rate * grad / (np.sqrt(caches[index]) + epsilon)
                    )
            # The vendor callback used mode="max" for a decreasing MAE, so it
            # effectively reduced the rate every patience interval.
            if (epoch + 1) % 10 == 0:
                learning_rate = max(learning_rate * 0.8, 1e-6)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2, w3, b3 = self.weights_
        first = _selu(np.asarray(x, dtype=np.float64) @ w1 + b1)
        second = _selu(first @ w2 + b2)
        return (second @ w3 + b3).reshape(-1)


def _interpolate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].interpolate(
        method="linear", limit_direction="both"
    )
    return result


def _enhanced_frame(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = make_features(frame)
    all_features = make_feature_mode(frame, "all")
    added = [column for column in all_features if column not in baseline]
    compact_baseline = [
        column
        for column in baseline
        if column.startswith("week_sin_")
        or column.startswith("week_cos_")
        or column
        in {
            "ndvi_mean",
            "ndvi_std",
            "station_temp_range",
            "reanalysis_temp_range",
            "temperature_humidity_interaction",
        }
    ]
    return all_features[compact_baseline + added].astype(float)


@dataclass
class MatrixBuilder:
    frame: pd.DataFrame
    city: str
    train_end: int
    enhanced: bool
    scale_reference: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.frame = _interpolate(self.frame.reset_index(drop=True))
        reference = (
            self.frame.iloc[: self.train_end]
            if self.scale_reference is None
            else _interpolate(self.scale_reference.reset_index(drop=True))
        )
        self.scaled: dict[str, np.ndarray] = {}
        for column in WINDOWS[self.city]:
            values = self.frame[column].to_numpy(float)
            train_values = reference[column].to_numpy(float)
            if column in {"year", "weekofyear"}:
                low, high = np.min(train_values), np.max(train_values)
                scale = max(high - low, 1.0)
                self.scaled[column] = (values - low) / scale
            else:
                mean = float(np.mean(train_values))
                std = max(float(np.std(train_values)), 1e-6)
                self.scaled[column] = (values - mean) / std
        self.enhanced_values: np.ndarray | None = None
        self.enhanced_median: np.ndarray | None = None
        self.enhanced_scale: np.ndarray | None = None
        if self.enhanced:
            values = _enhanced_frame(self.frame).to_numpy(float)
            train = values[: self.train_end]
            median = np.nanmedian(train, axis=0)
            median = np.where(np.isfinite(median), median, 0.0)
            train = np.where(np.isfinite(train), train, median)
            scale = np.std(train, axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
            self.enhanced_values = np.where(np.isfinite(values), values, median)
            self.enhanced_median = median
            self.enhanced_scale = scale
            self.enhanced_values = (self.enhanced_values - np.mean(train, axis=0)) / scale

    def row(self, endpoint: int, circular_limit: int | None = None) -> np.ndarray:
        parts: list[np.ndarray] = []
        for column, window in WINDOWS[self.city].items():
            indices = np.arange(endpoint - window + 1, endpoint + 1)
            if circular_limit is not None:
                indices %= circular_limit
            elif np.any(indices < 0):
                indices = np.maximum(indices, 0)
            parts.append(self.scaled[column][indices])
        if self.enhanced_values is not None:
            index = endpoint % circular_limit if circular_limit else endpoint
            parts.append(self.enhanced_values[index])
        return np.concatenate(parts)

    def training(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        endpoints = np.concatenate(
            [np.arange(self.train_end - 2, self.train_end), np.arange(self.train_end - 2)]
        )
        x = np.vstack([self.row(int(i), self.train_end) for i in endpoints])
        return x, target[endpoints]

    def prediction(self, start: int, length: int, shift: int) -> np.ndarray:
        endpoints = np.arange(start - shift, start + length - shift)
        return np.vstack([self.row(int(i)) for i in endpoints])


def _network_settings(city: str, steps: int, seed: int) -> SeluMAENetwork:
    return SeluMAENetwork(
        hidden_units=80 if city == "sj" else 100,
        epochs=50 if city == "sj" else 4,
        steps_per_epoch=steps,
        learning_rate=0.01 if city == "sj" else 0.001,
        random_state=seed,
    )


def fit_predict(
    frame: pd.DataFrame,
    city: str,
    train_end: int,
    prediction_length: int,
    candidate: str,
    seeds: list[int],
    steps: int,
    scale_reference: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[SeluMAENetwork]]:
    enhanced = candidate.startswith("enhanced")
    shift = 2 if candidate.endswith("shift2") else 0
    builder = MatrixBuilder(
        frame, city, train_end, enhanced, scale_reference=scale_reference
    )
    target = frame.iloc[:train_end][TARGET_COLUMN].to_numpy(float)
    train_x, train_y = builder.training(target)
    prediction_x = builder.prediction(train_end, prediction_length, shift)
    models: list[SeluMAENetwork] = []
    predictions: list[np.ndarray] = []
    for seed in seeds:
        model = _network_settings(city, steps, seed).fit(train_x, train_y)
        models.append(model)
        predictions.append(model.predict(prediction_x))
    return np.mean(predictions, axis=0), models


def evaluate(
    train: pd.DataFrame, seeds: list[int], folds: int, steps: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for city in ("iq", "sj"):
        city_frame = train.loc[train["city"].eq(city)].sort_values(DATE_COLUMN).reset_index(drop=True)
        starts = list(range(len(city_frame) - folds * 52, len(city_frame), 52))
        for candidate in CANDIDATES:
            fold_maes: list[float] = []
            for fold, start in enumerate(starts, start=1):
                prediction, _ = fit_predict(
                    city_frame.iloc[: start + 52].copy(), city, start, 52,
                    candidate, seeds, steps,
                )
                actual = city_frame.iloc[start : start + 52][TARGET_COLUMN].to_numpy(int)
                integer_prediction = np.maximum(prediction, 0).astype(int)
                mae = mean_absolute_error(actual, integer_prediction)
                fold_maes.append(mae)
                rows.append({
                    "city": city, "candidate": candidate, "fold": fold,
                    "origin": start, "mae": mae,
                    "actual_mean": float(np.mean(actual)),
                    "prediction_mean": float(np.mean(integer_prediction)),
                    "prediction_max": int(np.max(integer_prediction)),
                })
            print(
                f"{city} {candidate}: mean={np.mean(fold_maes):.3f} "
                f"std={np.std(fold_maes):.3f}", flush=True,
            )
    return pd.DataFrame(rows)


def final_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    template: pd.DataFrame,
    candidate: str,
    seeds: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {"candidate": candidate, "cities": {}}
    for city in ("iq", "sj"):
        train_city = train.loc[train["city"].eq(city)].copy()
        test_city = test.loc[test["city"].eq(city)].copy()
        train_city["_split"] = "train"
        test_city["_split"] = "test"
        combined = pd.concat([train_city, test_city], ignore_index=True, sort=False)
        combined = combined.sort_values(DATE_COLUMN).reset_index(drop=True)
        train_end = int(combined["_split"].eq("train").sum())
        prediction, models = fit_predict(
            combined, city, train_end, len(test_city), candidate, seeds, 200
        )
        output = combined.iloc[train_end:][KEY_COLUMNS].copy()
        output[TARGET_COLUMN] = np.maximum(prediction, 0).astype(int)
        rows.append(output)
        artifacts["cities"][city] = models
    values = pd.concat(rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        values, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission, artifacts


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, template = load_data(args.data_dir)
    if not args.skip_validation:
        scores = evaluate(train, seeds, args.folds, args.validation_steps)
        scores.to_csv(args.output_dir / "vendor_nn_validation_scores.csv", index=False)
        summary = (
            scores.groupby(["city", "candidate"])["mae"]
            .agg(["mean", "std", "max"])
            .reset_index()
        )
        summary.to_csv(args.output_dir / "vendor_nn_validation_summary.csv", index=False)
    if not args.skip_final:
        for candidate in CANDIDATES:
            submission, models = final_submission(train, test, template, candidate, seeds)
            submission.to_csv(
                args.output_dir / f"submission_nn_{candidate}.csv", index=False
            )
            joblib.dump(models, args.output_dir / f"nn_{candidate}_models.joblib")
            print(f"\n{candidate} submission distribution:")
            print(
                submission.groupby("city")[TARGET_COLUMN]
                .agg(["min", "median", "mean", "max"])
                .round(2)
                .to_string()
            )
    metadata = {
        "seeds": seeds,
        "folds": args.folds,
        "validation_steps": args.validation_steps,
        "candidates": list(CANDIDATES),
    }
    (args.output_dir / "vendor_nn_run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
