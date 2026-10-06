"""Predictors: seasonal baselines, city-specific trees, and the SELU MLP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .config import TREE_PARAMS, MLPConfig, TrainingSchedule

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def median_baseline(train_y: np.ndarray, n: int) -> np.ndarray:
    """Predict the training median every week."""
    return np.full(n, float(np.median(train_y)))


def seasonal_baseline(train_weeks: np.ndarray, train_y: np.ndarray, future_weeks: np.ndarray) -> np.ndarray:
    """Predict the training median for the same week of the year."""
    profile = pd.Series(train_y).groupby(np.asarray(train_weeks)).median()
    fallback = float(np.median(train_y))
    return pd.Series(future_weeks).map(profile).fillna(fallback).to_numpy(float)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def tree_model(city: str, n_estimators: int = 500, random_state: int = 42, **params: Any):
    """San Juan: Poisson Extra Trees. Iquitos: Random Forest on log1p(cases)."""
    params = {**TREE_PARAMS[city], **params}
    if city == "sj":
        forest = ExtraTreesRegressor(
            n_estimators=n_estimators, criterion="poisson", n_jobs=-1, random_state=random_state, **params
        )
        return make_pipeline(SimpleImputer(strategy="median"), forest)
    forest = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state, **params)
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy="median"), forest),
        func=np.log1p,
        inverse_func=np.expm1,
    )


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805


def _selu(x: np.ndarray) -> np.ndarray:
    return _SELU_SCALE * np.where(x > 0.0, x, _SELU_ALPHA * np.expm1(np.clip(x, -30, 0)))


def _selu_gradient(x: np.ndarray) -> np.ndarray:
    return _SELU_SCALE * np.where(x > 0.0, 1.0, _SELU_ALPHA * np.exp(np.clip(x, -30, 0)))


@dataclass
class SeluMLP:
    """Dense(h1, SELU) -> Dropout -> Dense(h2, SELU) -> Dropout -> Dense(1).

    Trained with RMSprop on the MAE loss (the competition metric), so the
    network predicts case counts directly. The learning rate is multiplied by
    0.8 whenever the epoch training MAE has not improved for 3 (SJ) or
    5 (IQ) epochs. Pure NumPy, so it runs without TensorFlow; it mirrors the
    Keras model that produced the leaderboard submissions but is not
    bit-identical to it.
    """

    city: str
    config: MLPConfig
    schedule: TrainingSchedule = field(default_factory=TrainingSchedule)
    seed: int = 42
    weights_: list[np.ndarray] | None = None
    history_: list[float] = field(default_factory=list)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "SeluMLP":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.float64).reshape(-1, 1)
        rng = np.random.default_rng(self.seed)
        cfg = self.config

        def glorot(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.weights_ = [
            glorot(x.shape[1], cfg.hidden_1),
            np.zeros((1, cfg.hidden_1)),
            glorot(cfg.hidden_1, cfg.hidden_2),
            np.zeros((1, cfg.hidden_2)),
            glorot(cfg.hidden_2, 1),
            np.zeros((1, 1)),
        ]
        caches = [np.zeros_like(w) for w in self.weights_]
        rho, epsilon = 0.9, 1e-7
        learning_rate = cfg.learning_rate
        keep1, keep2 = 1.0 - cfg.dropout_1, 1.0 - cfg.dropout_2
        patience = 3 if self.city == "sj" else 5
        order = rng.permutation(len(x))
        position, best_mae, stale_epochs = 0, np.inf, 0
        self.history_ = []

        for _ in range(self.schedule.epochs):
            batch_errors = []
            for _ in range(self.schedule.steps_per_epoch):
                if position >= len(order):
                    if self.city == "sj":  # SJ reshuffles each pass; IQ keeps one order
                        order = rng.permutation(len(x))
                    position = 0
                index = order[position : position + self.schedule.batch_size]
                position += len(index)
                xb, yb = x[index], y[index]
                w1, b1, w2, b2, w3, b3 = self.weights_

                z1 = xb @ w1 + b1
                a1 = _selu(z1)
                mask1 = rng.random(a1.shape) < keep1
                d1 = a1 * mask1 / keep1
                z2 = d1 @ w2 + b2
                a2 = _selu(z2)
                mask2 = rng.random(a2.shape) < keep2
                d2 = a2 * mask2 / keep2
                prediction = d2 @ w3 + b3
                batch_errors.append(float(np.mean(np.abs(prediction - yb))))

                grad_out = np.sign(prediction - yb) / len(index)
                dz2 = (grad_out @ w3.T) * _selu_gradient(z2) * mask2 / keep2
                dz1 = ((dz2 @ w2.T) * mask1 / keep1) * _selu_gradient(z1)
                gradients = [
                    xb.T @ dz1,
                    dz1.sum(axis=0, keepdims=True),
                    d1.T @ dz2,
                    dz2.sum(axis=0, keepdims=True),
                    d2.T @ grad_out,
                    grad_out.sum(axis=0, keepdims=True),
                ]
                for i, grad in enumerate(gradients):
                    grad = np.clip(grad, -100.0, 100.0)
                    caches[i] = rho * caches[i] + (1.0 - rho) * grad**2
                    self.weights_[i] -= learning_rate * grad / (np.sqrt(caches[i]) + epsilon)

            epoch_mae = float(np.mean(batch_errors))
            self.history_.append(epoch_mae)
            if epoch_mae < best_mae:
                best_mae, stale_epochs = epoch_mae, 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                learning_rate = max(learning_rate * 0.8, 1e-6)
                stale_epochs = 0
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted")
        w1, b1, w2, b2, w3, b3 = self.weights_
        hidden = _selu(_selu(np.asarray(matrix, dtype=np.float64) @ w1 + b1) @ w2 + b2)
        return (hidden @ w3 + b3).reshape(-1)

    @property
    def n_parameters(self) -> int:
        return int(sum(w.size for w in self.weights_ or []))


def to_cases(raw: np.ndarray, rounding: str = "floor") -> np.ndarray:
    """Submission form: non-negative integers.

    The MLP submissions truncated (``floor``); the tree submission rounded to
    the nearest integer (``round``). Both are kept to reproduce each model.
    """
    values = np.maximum(np.asarray(raw, dtype=float), 0.0)
    if rounding == "round":
        values = np.floor(values + 0.5)
    return values.astype(int)
