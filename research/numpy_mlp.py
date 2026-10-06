"""NumPy SELU MLP matching ``feature_lag_mlp`` widths and training schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_lag_mlp.config import ModelConfig, TrainingSchedule


def _selu(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0.0, x, alpha * np.expm1(np.clip(x, -30, 0)))


def _selu_gradient(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0.0, 1.0, alpha * np.exp(np.clip(x, -30, 0)))


@dataclass
class NumpyFeatureLagMlp:
    city: str
    config: ModelConfig
    schedule: TrainingSchedule
    seed: int

    weights_: list[np.ndarray] | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> NumpyFeatureLagMlp:
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.float64).reshape(-1, 1)
        rng = np.random.default_rng(self.seed)
        h1, h2 = self.config.hidden_widths
        d1, d2 = self.config.dropout_rates

        def glorot(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.weights_ = [
            glorot(x.shape[1], h1),
            np.zeros((1, h1)),
            glorot(h1, h2),
            np.zeros((1, h2)),
            glorot(h2, 1),
            np.zeros((1, 1)),
        ]
        caches = [np.zeros_like(value) for value in self.weights_]
        rho, epsilon = 0.9, 1e-7
        learning_rate = self.config.learning_rate
        order = rng.permutation(len(x))
        position = 0
        keep1 = 1.0 - d1
        keep2 = 1.0 - d2
        patience = 3 if self.city == "sj" else 5
        best_mae = np.inf
        stale_epochs = 0

        for epoch in range(self.schedule.epochs):
            epoch_abs_errors: list[float] = []
            for _ in range(self.schedule.steps_per_epoch):
                if position >= len(order):
                    if self.city == "sj":
                        order = rng.permutation(len(x))
                    position = 0
                indices = order[position : position + self.schedule.batch_size]
                position += len(indices)
                xb, yb = x[indices], y[indices]
                w1, b1, w2, b2, w3, b3 = self.weights_
                z1 = xb @ w1 + b1
                a1 = _selu(z1)
                mask1 = rng.random(a1.shape) < keep1
                dropped1 = a1 * mask1 / keep1
                z2 = dropped1 @ w2 + b2
                a2 = _selu(z2)
                mask2 = rng.random(a2.shape) < keep2
                dropped2 = a2 * mask2 / keep2
                prediction = dropped2 @ w3 + b3
                epoch_abs_errors.append(float(np.mean(np.abs(prediction - yb))))
                gradient = np.sign(prediction - yb) / len(indices)
                gradients: list[np.ndarray] = [np.empty(0)] * 6
                gradients[4] = dropped2.T @ gradient
                gradients[5] = gradient.sum(axis=0, keepdims=True)
                da2 = gradient @ w3.T
                dz2 = da2 * _selu_gradient(z2) * mask2 / keep2
                gradients[2] = dropped1.T @ dz2
                gradients[3] = dz2.sum(axis=0, keepdims=True)
                da1 = (dz2 @ w2.T) * mask1 / keep1
                dz1 = da1 * _selu_gradient(z1)
                gradients[0] = xb.T @ dz1
                gradients[1] = dz1.sum(axis=0, keepdims=True)
                for index, grad in enumerate(gradients):
                    grad = np.clip(grad, -100.0, 100.0)
                    caches[index] = rho * caches[index] + (1.0 - rho) * grad**2
                    self.weights_[index] -= (
                        learning_rate * grad / (np.sqrt(caches[index]) + epsilon)
                    )

            epoch_mae = float(np.mean(epoch_abs_errors))
            if epoch_mae < best_mae:
                best_mae = epoch_mae
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                learning_rate = max(learning_rate * 0.8, 1e-6)
                stale_epochs = 0

        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fit")
        w1, b1, w2, b2, w3, b3 = self.weights_
        first = _selu(np.asarray(matrix, dtype=np.float64) @ w1 + b1)
        second = _selu(first @ w2 + b2)
        return (second @ w3 + b3).reshape(-1)
