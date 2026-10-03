"""Stage 3: define the three-Dense-layer MLP."""

from __future__ import annotations

from typing import Any

from .config import INPUT_DIMENSIONS, ModelConfig


def build_model(city: str, config: ModelConfig) -> Any:
    """Build and compile one city model.

    Dense layers are counted as: hidden 1, hidden 2, and the one-unit output.
    Dropout layers have no trainable weights and are not counted as MLP layers.
    """
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(INPUT_DIMENSIONS[city],)),
            tf.keras.layers.Dense(config.hidden_1, activation="selu"),
            tf.keras.layers.Dropout(config.dropout_1),
            tf.keras.layers.Dense(config.hidden_2, activation="selu"),
            tf.keras.layers.Dropout(config.dropout_2),
            tf.keras.layers.Dense(1),
        ],
        name=f"feature_lag_mlp_{city}",
    )
    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=config.learning_rate,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-7,
            centered=False,
            name="RMSprop",
        ),
        metrics=["mae", "mse"],
    )
    return model

