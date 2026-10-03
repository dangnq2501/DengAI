"""Faithful TensorFlow reproduction of the vendor's 12.6779-MAE model.

Run this script with the isolated ``.venv-tf`` environment. It trains the
original shifted climate-history model and an enhanced aligned version using
the project's leakage-safe engineered features. It also writes conservative
25% and 50% enhanced blends; the confirmed tree submission is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from train_tree_ensemble import DATE_COLUMN, KEY_COLUMNS, TARGET_COLUMN, load_data
from train_vendor_history_nn import MatrixBuilder


CANDIDATES = ("vendor_shift2", "enhanced_aligned")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    return parser.parse_args()


def build_model(city: str, input_size: int) -> tf.keras.Model:
    nodes = 80 if city == "sj" else 100
    learning_rate = 0.01 if city == "sj" else 0.001
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(nodes, activation="selu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(nodes // 4, activation="selu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-7,
            centered=False,
        ),
        metrics=["mae", "mse"],
    )
    return model


def train_model(
    city: str,
    x: np.ndarray,
    y: np.ndarray,
    verbose: int,
) -> tuple[tf.keras.Model, dict[str, list[float]]]:
    dataset = tf.data.Dataset.from_tensor_slices(
        (x.astype("float32"), y.astype("float32"))
    )
    dataset = dataset.cache().shuffle(500).batch(16).repeat()
    model = build_model(city, x.shape[1])
    callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="mae",
        factor=0.8,
        patience=10,
        min_lr=1e-6,
        verbose=verbose,
        mode="max",
    )
    history = model.fit(
        dataset,
        epochs=50 if city == "sj" else 4,
        steps_per_epoch=200,
        verbose=verbose,
        callbacks=[callback],
    )
    return model, history.history


def candidate_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: str,
    output_dir: Path,
    verbose: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {}
    # The vendor helper accidentally reused SJ normalization for IQ. Preserve
    # that behavior for the historical feature block in both candidates.
    sj_reference = (
        train.loc[train["city"].eq("sj")]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    for city in ("sj", "iq"):
        train_city = train.loc[train["city"].eq(city)].copy()
        test_city = test.loc[test["city"].eq(city)].copy()
        train_city["_split"] = "train"
        test_city["_split"] = "test"
        combined = (
            pd.concat([train_city, test_city], ignore_index=True, sort=False)
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
        train_end = len(train_city)
        enhanced = candidate.startswith("enhanced")
        shift = 2 if candidate.endswith("shift2") else 0
        builder = MatrixBuilder(
            combined,
            city,
            train_end,
            enhanced,
            scale_reference=sj_reference,
        )
        target = combined.iloc[:train_end][TARGET_COLUMN].to_numpy(float)
        train_x, train_y = builder.training(target)
        test_x = builder.prediction(train_end, len(test_city), shift)
        model, history = train_model(city, train_x, train_y, verbose)
        raw = model.predict(test_x.astype("float32"), verbose=0).reshape(-1)
        result = combined.iloc[train_end:][KEY_COLUMNS].copy()
        # Match the original notebook: truncate positive floats toward zero.
        result[TARGET_COLUMN] = np.maximum(raw, 0).astype(int)
        rows.append(result)
        model.save(output_dir / f"tf_{candidate}_{city}.keras")
        diagnostics[city] = {
            "input_features": int(train_x.shape[1]),
            "final_training_mae": float(history["mae"][-1]),
            "prediction_mean": float(result[TARGET_COLUMN].mean()),
            "prediction_max": int(result[TARGET_COLUMN].max()),
        }
    return pd.concat(rows, ignore_index=True), diagnostics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    train, test, template = load_data(args.data_dir)
    predictions: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, object] = {
        "tensorflow": tf.__version__,
        "seed": args.seed,
        "candidates": {},
    }
    for candidate in CANDIDATES:
        values, candidate_diagnostics = candidate_prediction(
            train, test, candidate, args.output_dir, args.verbose
        )
        submission = template[KEY_COLUMNS].merge(
            values, on=KEY_COLUMNS, how="left", validate="one_to_one"
        )
        submission[TARGET_COLUMN] = submission[TARGET_COLUMN].astype(int)
        submission.to_csv(
            args.output_dir / f"submission_tf_{candidate}.csv", index=False
        )
        predictions[candidate] = submission
        diagnostics["candidates"][candidate] = candidate_diagnostics
        print(f"\n{candidate} distribution:")
        print(
            submission.groupby("city")[TARGET_COLUMN]
            .agg(["min", "median", "mean", "max"])
            .round(2)
            .to_string()
        )
    original = predictions["vendor_shift2"]
    enhanced = predictions["enhanced_aligned"]
    for enhanced_weight in (0.25, 0.5):
        blended = original.copy()
        raw = (
            (1.0 - enhanced_weight) * original[TARGET_COLUMN]
            + enhanced_weight * enhanced[TARGET_COLUMN]
        )
        blended[TARGET_COLUMN] = np.rint(np.maximum(raw, 0)).astype(int)
        label = str(enhanced_weight).replace(".", "")
        blended.to_csv(
            args.output_dir / f"submission_tf_vendor_enhanced_w{label}.csv",
            index=False,
        )
    # The chronological comparison favors the vendor inputs for IQ and the
    # engineered inputs for SJ, so also provide city-specific alternatives.
    sj_mask = original["city"].eq("sj")
    hybrid = original.copy()
    hybrid.loc[sj_mask, TARGET_COLUMN] = enhanced.loc[sj_mask, TARGET_COLUMN]
    hybrid.to_csv(
        args.output_dir / "submission_tf_city_hybrid.csv", index=False
    )
    conservative = original.copy()
    sj_raw = (
        0.75 * original.loc[sj_mask, TARGET_COLUMN]
        + 0.25 * enhanced.loc[sj_mask, TARGET_COLUMN]
    )
    conservative.loc[sj_mask, TARGET_COLUMN] = np.rint(
        np.maximum(sj_raw, 0)
    ).astype(int)
    conservative.to_csv(
        args.output_dir / "submission_tf_city_hybrid_w025.csv", index=False
    )
    (args.output_dir / "vendor_tf_run.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
