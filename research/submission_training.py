"""Full-data training helpers for research submissions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_lag_mlp.features import build_test_matrix, build_training_matrix
from feature_lag_mlp.search_space import Candidate

from research.training_backend import Backend, fit_city, predict_raw, resolve_backend


def train_full_city(
    train: pd.DataFrame,
    test: pd.DataFrame,
    city: str,
    candidate: Candidate,
    seeds: tuple[int, ...],
    schedule: Any,
    verbose: int,
    backend: Backend = "auto",
) -> tuple[np.ndarray, list[dict[str, object]], str]:
    """Fit on all rows and average raw predictions across seeds."""
    resolved = resolve_backend(backend)
    train_x, train_y = build_training_matrix(train, city)
    test_x = build_test_matrix(train, test, city)
    predictions: list[np.ndarray] = []
    runs: list[dict[str, object]] = []
    for seed in seeds:
        print(f"[{city} final] {candidate.name}, seed={seed}, backend={resolved}", flush=True)
        model, history = fit_city(
            train_x,
            train_y,
            city,
            candidate.config,
            schedule,
            seed,
            verbose,
            backend=resolved,
        )
        raw = predict_raw(model, test_x, resolved)
        predictions.append(raw)
        runs.append(
            {
                "seed": seed,
                "backend": resolved,
                "final_training_mae": float(history["mae"].iloc[-1]),
                "prediction_mean": float(np.mean(raw)),
                "prediction_max": float(np.max(raw)),
            }
        )
    return np.mean(predictions, axis=0), runs, resolved
