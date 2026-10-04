"""Train a compact residual temporal convolution model for DengAI.

This is a controlled successor to the vendor V10 network.  It retains the
shared San Juan normalization and two-week feature offset that performed well
on the leaderboard, but presents the climate history as a weekly sequence.
The epoch count is selected on each city's final 104 training weeks and the
final model is then retrained on all available rows for that many epochs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from reproduce_vendor_17_57 import (
    WEATHER_COLUMNS,
    _global_interpolate,
    _scale_parameters,
    load_vendor_data,
)
from train_tree_ensemble import DATE_COLUMN, KEY_COLUMNS, TARGET_COLUMN


WINDOW = 42
FORECAST_OFFSET = 2
VALIDATION_WEEKS = 104
CHANNELS = len(WEATHER_COLUMNS) + 2


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--experiment-name",
        default="residual_tcn_v1",
        help="Artifact stem; existing V10 artifacts are never overwritten.",
    )
    return parser.parse_args()


def scaled_weekly_features(
    frame: pd.DataFrame,
    parameters: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Return shared-SJ-normalized weather plus cyclic week coordinates."""
    columns: list[np.ndarray] = []
    for column in WEATHER_COLUMNS:
        mean, std = parameters[column]
        columns.append((frame[column].to_numpy(float) - mean) / max(std, 1e-12))
    angle = 2.0 * np.pi * frame["weekofyear"].to_numpy(float) / 52.0
    columns.extend((np.sin(angle), np.cos(angle)))
    values = np.column_stack(columns)
    # A zero is the shared-SJ mean for weather channels.  This also handles
    # any leading missing observation that forward interpolation cannot fill.
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values.astype("float32")


def lagged_windows(
    values: np.ndarray,
    target_indices: np.ndarray,
    window: int = WINDOW,
    forecast_offset: int = FORECAST_OFFSET,
) -> np.ndarray:
    """Build causal windows ending ``forecast_offset`` rows before targets."""
    if values.ndim != 2:
        raise ValueError("values must have shape (weeks, channels)")
    if window < 1 or forecast_offset < 0:
        raise ValueError("window must be positive and forecast_offset nonnegative")
    output = np.zeros((len(target_indices), window, values.shape[1]), dtype="float32")
    for output_index, target_index in enumerate(target_indices.astype(int)):
        endpoint = target_index - forecast_offset
        if endpoint < 0:
            continue
        start = max(0, endpoint - window + 1)
        history = values[start : endpoint + 1]
        output[output_index, -len(history) :] = history
    return output


def build_city_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    city: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/test weekly tensors with V10's shared normalization."""
    interpolated_train = _global_interpolate(train)
    interpolated_test = _global_interpolate(test)
    parameters = _scale_parameters(interpolated_train)
    train_city = (
        interpolated_train.loc[interpolated_train["city"].eq(city)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    test_city = (
        interpolated_test.loc[interpolated_test["city"].eq(city)]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    train_values = scaled_weekly_features(train_city, parameters)
    test_values = scaled_weekly_features(test_city, parameters)
    train_x = lagged_windows(train_values, np.arange(len(train_city)))
    combined = np.vstack((train_values, test_values))
    test_indices = np.arange(len(train_city), len(train_city) + len(test_city))
    test_x = lagged_windows(combined, test_indices)
    train_y = train_city[TARGET_COLUMN].to_numpy("float32")
    if train_x.shape != (len(train_city), WINDOW, CHANNELS):
        raise AssertionError(f"Unexpected {city} train tensor: {train_x.shape}")
    if test_x.shape != (len(test_city), WINDOW, CHANNELS):
        raise AssertionError(f"Unexpected {city} test tensor: {test_x.shape}")
    if not all(np.isfinite(array).all() for array in (train_x, train_y, test_x)):
        raise ValueError(f"Non-finite values in {city} temporal matrices")
    return train_x, train_y, test_x


def residual_block(inputs: Any, dilation: int, dropout_rate: float) -> Any:
    import tensorflow as tf

    x = tf.keras.layers.Conv1D(
        16, 3, padding="causal", dilation_rate=dilation
    )(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv1D(
        16, 3, padding="causal", dilation_rate=dilation
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Add()([inputs, x])
    return tf.keras.layers.Activation("gelu")(x)


def build_model(target_median: float) -> Any:
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(WINDOW, CHANNELS))
    x = tf.keras.layers.Conv1D(16, 1, padding="same")(inputs)
    for dilation in (1, 2, 4):
        x = residual_block(x, dilation=dilation, dropout_rate=0.15)
    average = tf.keras.layers.GlobalAveragePooling1D()(x)
    maximum = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([average, maximum])
    x = tf.keras.layers.Dense(16, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    median = max(float(target_median), 0.1)
    bias = float(np.log(np.expm1(median)))
    outputs = tf.keras.layers.Dense(
        1,
        activation="softplus",
        bias_initializer=tf.keras.initializers.Constant(bias),
    )(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss="mae",
        metrics=["mae", "mse"],
    )
    return model


def train_city(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    max_epochs: int,
    verbose: int,
) -> tuple[np.ndarray, Any, pd.DataFrame, int, float]:
    import tensorflow as tf

    split = len(train_y) - VALIDATION_WEEKS
    if split <= 0:
        raise ValueError("Not enough rows for temporal validation")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    selection_model = build_model(float(np.median(train_y[:split])))
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=verbose
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            min_delta=0.01,
            restore_best_weights=True,
            verbose=verbose,
        ),
    ]
    selection_history = selection_model.fit(
        train_x[:split],
        train_y[:split],
        validation_data=(train_x[split:], train_y[split:]),
        epochs=max_epochs,
        batch_size=32,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    history = pd.DataFrame(selection_history.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    best_index = int(history["val_loss"].to_numpy().argmin())
    best_epoch = best_index + 1
    best_validation_mae = float(history.loc[best_index, "val_loss"])

    # Use validation only to select training duration, then recover all rows.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed + 10_000)
    final_model = build_model(float(np.median(train_y)))
    final_model.fit(
        train_x,
        train_y,
        epochs=best_epoch,
        batch_size=32,
        shuffle=True,
        verbose=verbose,
    )
    prediction = final_model.predict(test_x, verbose=0).reshape(-1)
    return prediction, final_model, history, best_epoch, best_validation_mae


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import tensorflow as tf

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    train, test, template = load_vendor_data(args.data_dir)
    rows: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {
        "model": "compact residual temporal convolution network",
        "seed": args.seed,
        "tensorflow": tf.__version__,
        "normalization": "vendor-sj-shared",
        "window_weeks": WINDOW,
        "forecast_offset_weeks": FORECAST_OFFSET,
        "weather_channels": WEATHER_COLUMNS,
        "seasonal_channels": ["week_sin", "week_cos"],
        "dilations": [1, 2, 4],
        "residual_channels": 16,
        "residual_dropout": 0.15,
        "head_dropout": 0.25,
        "validation_weeks": VALIDATION_WEEKS,
        "cities": {},
    }
    for city_index, city in enumerate(("sj", "iq")):
        train_x, train_y, test_x = build_city_matrices(train, test, city)
        raw, model, history, best_epoch, validation_mae = train_city(
            train_x,
            train_y,
            test_x,
            seed=args.seed + city_index,
            max_epochs=args.max_epochs,
            verbose=args.verbose,
        )
        city_test = (
            test.loc[test["city"].eq(city)]
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
        output = city_test[KEY_COLUMNS].copy()
        output[TARGET_COLUMN] = np.maximum(raw, 0.0).astype(int)
        rows.append(output)
        model_path = args.output_dir / f"{args.experiment_name}_{city}.keras"
        history_path = args.output_dir / f"{args.experiment_name}_{city}_selection.csv"
        model.save(model_path)
        history.to_csv(history_path, index=False)
        metadata["cities"][city] = {
            "training_shape": list(train_x.shape),
            "test_shape": list(test_x.shape),
            "parameters": int(model.count_params()),
            "selected_epochs": best_epoch,
            "temporal_validation_mae": validation_mae,
            "raw_prediction_mean": float(np.mean(raw)),
            "raw_prediction_max": float(np.max(raw)),
            "submission_mean": float(output[TARGET_COLUMN].mean()),
            "submission_max": int(output[TARGET_COLUMN].max()),
            "model_sha256": _sha256(model_path),
        }

    values = pd.concat(rows, ignore_index=True)
    submission = template[KEY_COLUMNS].merge(
        values, on=KEY_COLUMNS, how="left", validate="one_to_one", sort=False
    )
    if submission[TARGET_COLUMN].isna().any():
        raise ValueError("Submission contains missing predictions")
    submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
    submission_path = args.output_dir / f"submission_{args.experiment_name}.csv"
    submission.to_csv(submission_path, index=False)
    metadata["submission_sha256"] = _sha256(submission_path)
    metadata_path = args.output_dir / f"{args.experiment_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {submission_path}")
    print(
        submission.groupby("city")[TARGET_COLUMN]
        .agg(["min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
