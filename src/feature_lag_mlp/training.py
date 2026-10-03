"""Stage 4: shared fitting, prediction, and submission helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATE_COLUMN,
    KEY_COLUMNS,
    ModelConfig,
    TARGET_COLUMN,
    TrainingSchedule,
)
from .model import build_model


@dataclass
class FitResult:
    model: Any
    history: pd.DataFrame


def set_tensorflow_seed(seed: int, *, clear_session: bool = False) -> None:
    import tensorflow as tf

    if clear_session:
        tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def fit_arrays(
    train_x: np.ndarray,
    train_y: np.ndarray,
    city: str,
    config: ModelConfig,
    schedule: TrainingSchedule,
    verbose: int,
) -> FitResult:
    """Fit one city using the confirmed repeated-batch schedule."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).cache()
    if city == "sj":
        dataset = dataset.shuffle(500)
    dataset = dataset.batch(schedule.batch_size).repeat()

    model = build_model(city, config)
    callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="mae",
        factor=0.8,
        patience=3 if city == "sj" else 5,
        min_lr=1e-6,
        mode=schedule.lr_mode,
        verbose=verbose,
    )
    history = model.fit(
        dataset,
        epochs=schedule.epochs,
        steps_per_epoch=schedule.steps_per_epoch,
        callbacks=[callback],
        shuffle=False,
        verbose=verbose,
    )
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    return FitResult(model=model, history=history_frame)


def predict_cases(model: Any, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return raw predictions and submission-form non-negative integers."""
    raw = model.predict(matrix, verbose=0).reshape(-1)
    cases = np.maximum(raw, 0).astype(int)
    return raw, cases


def prediction_frame(
    test: pd.DataFrame,
    city: str,
    cases: np.ndarray,
) -> pd.DataFrame:
    city_test = (
        test.loc[test["city"].eq(city)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    if len(city_test) != len(cases):
        raise ValueError(f"{city} has {len(city_test)} rows but {len(cases)} predictions")
    output = city_test[KEY_COLUMNS].copy()
    output[TARGET_COLUMN] = cases
    return output


def assemble_submission(
    template: pd.DataFrame,
    city_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    predictions = pd.concat(city_frames, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        predictions,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if submission[TARGET_COLUMN].isna().any():
        raise ValueError("Submission contains missing predictions")
    if not submission[KEY_COLUMNS].equals(template[KEY_COLUMNS]):
        raise ValueError("Submission row order differs from the template")
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    return submission

