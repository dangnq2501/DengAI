"""Train and predict with TensorFlow when available, otherwise NumPy."""

from __future__ import annotations

import importlib.util
from typing import Any, Literal

import numpy as np
import pandas as pd

from feature_lag_mlp.config import ModelConfig, TrainingSchedule
from feature_lag_mlp.training import FitResult, fit_arrays as fit_arrays_tf, predict_cases

from research.numpy_mlp import NumpyFeatureLagMlp

Backend = Literal["auto", "tensorflow", "numpy"]


def tensorflow_available() -> bool:
    try:
        if importlib.util.find_spec("tensorflow") is None:
            return False
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def resolve_backend(backend: Backend) -> str:
    if backend == "auto":
        return "tensorflow" if tensorflow_available() else "numpy"
    if backend == "tensorflow" and not tensorflow_available():
        raise RuntimeError(
            "TensorFlow is not importable in this environment; use --backend numpy"
        )
    return backend


def fit_city(
    train_x: np.ndarray,
    train_y: np.ndarray,
    city: str,
    config: ModelConfig,
    schedule: TrainingSchedule,
    seed: int,
    verbose: int,
    backend: Backend = "auto",
) -> tuple[Any, pd.DataFrame]:
    resolved = resolve_backend(backend)
    if resolved == "tensorflow":
        from feature_lag_mlp.training import set_tensorflow_seed

        set_tensorflow_seed(seed, clear_session=True)
        result: FitResult = fit_arrays_tf(
            train_x, train_y, city, config, schedule, verbose
        )
        return result.model, result.history

    model = NumpyFeatureLagMlp(city, config, schedule, seed).fit(train_x, train_y)
    history = pd.DataFrame(
        {
            "epoch": [schedule.epochs],
            "mae": [float(np.mean(np.abs(model.predict(train_x) - train_y)))],
        }
    )
    return model, history


def predict_raw(model: Any, matrix: np.ndarray, backend: str) -> np.ndarray:
    if backend == "tensorflow":
        return model.predict(matrix, verbose=0).reshape(-1)
    return model.predict(matrix)
