"""Stage 3: define the three-Dense-layer MLP."""

from __future__ import annotations

from typing import Any

from .config import INPUT_DIMENSIONS, ModelConfig


def build_model(
    city: str,
    config: ModelConfig,
    input_dimension: int | None = None,
) -> Any:
    """Build and compile one city model.

    Dense layers are counted as: hidden 1, hidden 2, and the one-unit output.
    Dropout layers have no trainable weights and are not counted as MLP layers.
    """
    import tensorflow as tf

    model_input_dimension = input_dimension or INPUT_DIMENSIONS[city]

    # Keep the proven baseline construction byte-for-byte equivalent. The
    # functional branch is used only by explicitly selected complexity tests.
    if not config.linear_skip and config.l2_strength == 0.0:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(model_input_dimension,)),
                tf.keras.layers.Dense(config.hidden_1, activation="selu"),
                tf.keras.layers.Dropout(config.dropout_1),
                tf.keras.layers.Dense(config.hidden_2, activation="selu"),
                tf.keras.layers.Dropout(config.dropout_2),
                tf.keras.layers.Dense(1),
            ],
            name=f"feature_lag_mlp_{city}",
        )
    else:
        regularizer = (
            tf.keras.regularizers.l2(config.l2_strength)
            if config.l2_strength > 0
            else None
        )
        inputs = tf.keras.layers.Input(
            shape=(model_input_dimension,), name="lag_history"
        )
        hidden = tf.keras.layers.Dense(
            config.hidden_1,
            activation="selu",
            kernel_regularizer=regularizer,
            name="nonlinear_hidden_1",
        )(inputs)
        hidden = tf.keras.layers.Dropout(config.dropout_1)(hidden)
        hidden = tf.keras.layers.Dense(
            config.hidden_2,
            activation="selu",
            kernel_regularizer=regularizer,
            name="nonlinear_hidden_2",
        )(hidden)
        hidden = tf.keras.layers.Dropout(config.dropout_2)(hidden)
        nonlinear = tf.keras.layers.Dense(
            1, kernel_regularizer=regularizer, name="nonlinear_output"
        )(hidden)
        if config.linear_skip:
            linear = tf.keras.layers.Dense(
                1,
                use_bias=False,
                kernel_regularizer=regularizer,
                name="regularized_linear_skip",
            )(inputs)
            outputs = tf.keras.layers.Add(name="linear_plus_nonlinear")(
                [linear, nonlinear]
            )
        else:
            outputs = nonlinear
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name=f"feature_lag_mlp_{city}_regularized"
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
