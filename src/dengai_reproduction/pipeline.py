from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["city", "year", "weekofyear"]
DATE = "week_start_date"
TARGET = "total_cases"

SEED = 42
EPOCHS = 40
STEPS = 200
BATCH_SIZE = 16
EXPECTED_ROWS = 416

WEATHER_FEATURES = [
    "precipitation_amt_mm",
    "reanalysis_air_temp_k",
    "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k",
    "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2",
    "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm",
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",
    "station_avg_temp_c",
    "station_diur_temp_rng_c",
    "station_max_temp_c",
    "station_min_temp_c",
    "station_precip_mm",
]

MEAN_WINDOWS = (2, 4, 8, 13, 26, 52)
STD_WINDOWS = (4, 13)
SUMMARY_DIMENSION = len(WEATHER_FEATURES) * 11 + 4

IQUITOS_LAG_WINDOWS = {
    "precipitation_amt_mm": 33,
    "reanalysis_air_temp_k": 10,
    "reanalysis_avg_temp_k": 4,
    "reanalysis_dew_point_temp_k": 6,
    "reanalysis_max_air_temp_k": 41,
    "reanalysis_min_air_temp_k": 40,
    "reanalysis_precip_amt_kg_per_m2": 3,
    "reanalysis_relative_humidity_percent": 7,
    "reanalysis_sat_precip_amt_mm": 33,
    "reanalysis_specific_humidity_g_per_kg": 26,
    "reanalysis_tdtr_k": 34,
    "station_avg_temp_c": 40,
    "station_diur_temp_rng_c": 26,
    "station_max_temp_c": 39,
    "station_min_temp_c": 25,
    "station_precip_mm": 10,
    "weekofyear": 3,
}

SELU_ALPHA = 1.6732632423543772
SELU_SCALE = 1.0507009873554805


@dataclass(frozen=True)
class DataBundle:
    """All input tables and fitted normalization parameters."""

    train: pd.DataFrame
    test: pd.DataFrame
    template: pd.DataFrame
    train_interpolated: pd.DataFrame
    test_interpolated: pd.DataFrame
    normalization: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class SubmissionResult:
    """Summary of a completed training and submission run."""

    output_path: Path
    rows: int
    san_juan_rows: int
    iquitos_rows: int
    san_juan_train_mae: float
    iquitos_train_mae: float


def interpolate_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the historical linear interpolation rule to numeric columns."""

    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].interpolate(method="linear", limit_direction="forward")
    return result


def san_juan_normalization(train_interpolated: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Fit the shared San Juan scaling parameters used by the scored pipeline."""

    san_juan = train_interpolated.loc[train_interpolated["city"].eq("sj")].reset_index(drop=True)
    reference = pd.concat([san_juan.iloc[-53:], san_juan], ignore_index=True)
    parameters = {
        feature: (float(np.nanmean(reference[feature])), float(np.nanstd(reference[feature])))
        for feature in WEATHER_FEATURES
    }
    week = reference["weekofyear"].to_numpy(float)
    parameters["weekofyear"] = (float(np.nanmin(week)), float(np.nanmax(week)))
    return parameters


def scale_columns(
    frame: pd.DataFrame,
    parameters: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Scale weather columns by z-score and week-of-year by min-max scaling."""

    scaled = {
        feature: (frame[feature].to_numpy(float) - parameters[feature][0])
        / max(parameters[feature][1], 1e-12)
        for feature in WEATHER_FEATURES
    }
    low, high = parameters["weekofyear"]
    scaled["weekofyear"] = (frame["weekofyear"].to_numpy(float) - low) / max(high - low, 1.0)
    return scaled


def load_data(repo_root: Path) -> DataBundle:
    """Load the competition CSVs and enforce the row-order contract."""

    data_dir = repo_root / "data"
    features = pd.read_csv(data_dir / "dengue_features_train.csv")
    labels = pd.read_csv(data_dir / "dengue_labels_train.csv")
    test = pd.read_csv(data_dir / "dengue_features_test.csv")
    template = pd.read_csv(data_dir / "submission_format.csv")

    train = features.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    for frame in (train, test):
        frame[DATE] = pd.to_datetime(frame[DATE])

    if not test[KEYS].equals(template[KEYS]):
        raise ValueError("Test rows do not match submission_format.csv")

    train_interpolated = interpolate_numeric(train)
    test_interpolated = interpolate_numeric(test)
    missing = int(train_interpolated[WEATHER_FEATURES].isna().sum().sum())
    missing += int(test_interpolated[WEATHER_FEATURES].isna().sum().sum())
    if missing:
        raise ValueError(f"Interpolation left {missing} missing weather values")

    return DataBundle(
        train=train,
        test=test,
        template=template,
        train_interpolated=train_interpolated,
        test_interpolated=test_interpolated,
        normalization=san_juan_normalization(train_interpolated),
    )


def multiscale_rows(
    combined: pd.DataFrame,
    parameters: dict[str, tuple[float, float]],
    row_count: int,
) -> np.ndarray:
    """Build the 180-dimensional multiscale representation for each week."""

    scaled = scale_columns(combined, parameters)
    weeks = combined["weekofyear"].to_numpy(float)
    rows = []

    for row_index in range(row_count):
        end = 51 + row_index
        values: list[float] = []
        for feature in WEATHER_FEATURES:
            history = scaled[feature][end - 51 : end + 1]
            means = {window: float(np.mean(history[-window:])) for window in MEAN_WINDOWS}
            values.extend(
                [
                    float(history[-1]),
                    *(means[window] for window in MEAN_WINDOWS),
                    *(float(np.std(history[-window:])) for window in STD_WINDOWS),
                    means[4] - means[13],
                    float((history[-1] - history[-13]) / 12.0),
                ]
            )

        phase = 2 * np.pi * (weeks[end] - 1.0) / 52.0
        values.extend([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
        rows.append(values)

    matrix = np.asarray(rows, dtype="float32")
    if matrix.shape != (row_count, SUMMARY_DIMENSION) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid multiscale matrix: {matrix.shape}")
    return matrix


def city_frames(
    train_interpolated: pd.DataFrame,
    test_interpolated: pd.DataFrame,
    city: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one city's rows in their original chronological file order."""

    train_city = train_interpolated.loc[train_interpolated["city"].eq(city)].reset_index(drop=True)
    test_city = test_interpolated.loc[test_interpolated["city"].eq(city)].reset_index(drop=True)
    return train_city, test_city


def build_city_features(
    train_interpolated: pd.DataFrame,
    test_interpolated: pd.DataFrame,
    parameters: dict[str, tuple[float, float]],
    city: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build multiscale train/test matrices and aligned targets for one city."""

    train_city, test_city = city_frames(train_interpolated, test_interpolated, city)
    train_with_context = pd.concat([train_city.iloc[-53:], train_city], ignore_index=True)
    test_with_context = pd.concat([train_city.iloc[-53:], test_city], ignore_index=True)
    train_x = multiscale_rows(train_with_context, parameters, len(train_city))
    train_y = train_with_context[TARGET].to_numpy(float)[51 : 51 + len(train_city)].astype("float32")
    test_x = multiscale_rows(test_with_context, parameters, len(test_city))
    return train_x, train_y, test_x


def selu(values: np.ndarray) -> np.ndarray:
    """SELU activation with the constants used by the original experiment."""

    return SELU_SCALE * np.where(values > 0, values, SELU_ALPHA * np.expm1(np.clip(values, -30, 0)))


def selu_derivative(values: np.ndarray) -> np.ndarray:
    """Derivative of :func:`selu`."""

    return SELU_SCALE * np.where(values > 0, 1.0, SELU_ALPHA * np.exp(np.clip(values, -30, 0)))


def predict_mlp(weights: list[np.ndarray], features: np.ndarray) -> np.ndarray:
    """Run inference for the two-hidden-layer NumPy MLP."""

    weight_1, bias_1, weight_2, bias_2, weight_3, bias_3 = weights
    hidden_1 = selu(np.asarray(features, np.float64) @ weight_1 + bias_1)
    hidden_2 = selu(hidden_1 @ weight_2 + bias_2)
    return (hidden_2 @ weight_3 + bias_3).reshape(-1)


def fit_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    city: str,
    hidden_1: int,
    hidden_2: int,
    dropout_1: float,
    dropout_2: float,
    learning_rate: float,
    seed: int = SEED,
    epochs: int = EPOCHS,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Fit the deterministic MLP used for the final San Juan predictions."""

    features = np.asarray(features, np.float64)
    targets = np.asarray(targets, np.float64).reshape(-1, 1)
    random = np.random.default_rng(seed)

    def glorot(inputs: int, outputs: int) -> np.ndarray:
        limit = np.sqrt(6 / (inputs + outputs))
        return random.uniform(-limit, limit, (inputs, outputs))

    weights = [
        glorot(features.shape[1], hidden_1),
        np.zeros((1, hidden_1)),
        glorot(hidden_1, hidden_2),
        np.zeros((1, hidden_2)),
        glorot(hidden_2, 1),
        np.zeros((1, 1)),
    ]
    rms_cache = [np.zeros_like(weight) for weight in weights]
    rho, epsilon = 0.9, 1e-7
    keep_1, keep_2 = 1 - dropout_1, 1 - dropout_2
    patience = 3 if city == "sj" else 5
    order = random.permutation(len(features))
    position, best, stale = 0, np.inf, 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        errors = []
        for _ in range(steps):
            if position >= len(order):
                if city == "sj":
                    order = random.permutation(len(features))
                position = 0

            indices = order[position : position + batch_size]
            position += len(indices)
            batch_x, batch_y = features[indices], targets[indices]
            weight_1, bias_1, weight_2, bias_2, weight_3, bias_3 = weights
            z_1 = batch_x @ weight_1 + bias_1
            activation_1 = selu(z_1)
            mask_1 = random.random(activation_1.shape) < keep_1
            dropped_1 = activation_1 * mask_1 / keep_1
            z_2 = dropped_1 @ weight_2 + bias_2
            activation_2 = selu(z_2)
            mask_2 = random.random(activation_2.shape) < keep_2
            dropped_2 = activation_2 * mask_2 / keep_2
            prediction = dropped_2 @ weight_3 + bias_3
            errors.append(float(np.mean(np.abs(prediction - batch_y))))

            output_gradient = np.sign(prediction - batch_y) / len(indices)
            z_2_gradient = (output_gradient @ weight_3.T) * selu_derivative(z_2) * mask_2 / keep_2
            z_1_gradient = ((z_2_gradient @ weight_2.T) * mask_1 / keep_1) * selu_derivative(z_1)
            gradients = [
                batch_x.T @ z_1_gradient,
                z_1_gradient.sum(0, keepdims=True),
                dropped_1.T @ z_2_gradient,
                z_2_gradient.sum(0, keepdims=True),
                dropped_2.T @ output_gradient,
                output_gradient.sum(0, keepdims=True),
            ]
            for index, gradient in enumerate(gradients):
                gradient = np.clip(gradient, -100.0, 100.0)
                rms_cache[index] = rho * rms_cache[index] + (1 - rho) * gradient**2
                weights[index] -= learning_rate * gradient / (np.sqrt(rms_cache[index]) + epsilon)

        dropout_mae = float(np.mean(errors))
        record = {
            "epoch": float(epoch + 1),
            "train_mae_dropout": dropout_mae,
            "lr": learning_rate,
            "train_mae": float(np.mean(np.abs(predict_mlp(weights, features) - targets.ravel()))),
        }
        if validation is not None:
            validation_prediction = round_nonnegative(predict_mlp(weights, validation[0]))
            record["val_mae"] = float(np.mean(np.abs(validation_prediction - validation[1])))
        history.append(record)

        if dropout_mae < best:
            best, stale = dropout_mae, 0
        else:
            stale += 1
        if stale >= patience:
            learning_rate, stale = max(learning_rate * 0.8, 1e-6), 0

    return weights, pd.DataFrame(history)


def round_nonnegative(predictions: np.ndarray) -> np.ndarray:
    """Clip counts to zero and apply the selected nearest-integer rule."""

    return np.floor(np.maximum(predictions, 0.0) + 0.5).astype(int)


def san_juan_predictions(bundle: DataBundle) -> tuple[np.ndarray, float]:
    """Train the final San Juan model and return its 260 test predictions."""

    train_x, train_y, test_x = build_city_features(
        bundle.train_interpolated,
        bundle.test_interpolated,
        bundle.normalization,
        "sj",
    )
    weights, _ = fit_mlp(
        train_x,
        train_y,
        city="sj",
        hidden_1=100,
        hidden_2=25,
        dropout_1=0.3,
        dropout_2=0.7,
        learning_rate=0.01,
    )
    train_mae = float(np.mean(np.abs(predict_mlp(weights, train_x) - train_y)))
    return round_nonnegative(predict_mlp(weights, test_x)), train_mae


def raw_lag_rows(
    combined: pd.DataFrame,
    parameters: dict[str, tuple[float, float]],
    row_count: int,
) -> np.ndarray:
    """Build the selected 380-dimensional raw-history representation for Iquitos."""

    scaled = scale_columns(combined, parameters)
    rows = [
        np.concatenate(
            [
                scaled[feature][51 + row_index - window + 1 : 51 + row_index + 1]
                for feature, window in IQUITOS_LAG_WINDOWS.items()
            ]
        )
        for row_index in range(row_count)
    ]
    matrix = np.vstack(rows).astype("float32")
    if matrix.shape != (row_count, 380) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid Iquitos raw-lag matrix: {matrix.shape}")
    return matrix


def iquitos_predictions(bundle: DataBundle) -> tuple[np.ndarray, float]:
    """Train the raw-history Iquitos MLP and return predictions and train MAE."""

    train_city, test_city = city_frames(
        bundle.train_interpolated,
        bundle.test_interpolated,
        "iq",
    )
    train_with_context = pd.concat([train_city.iloc[-53:], train_city], ignore_index=True)
    test_with_context = pd.concat([train_city.iloc[-53:], test_city], ignore_index=True)
    train_x = raw_lag_rows(train_with_context, bundle.normalization, len(train_city))
    train_y = train_with_context[TARGET].to_numpy(float)[51 : 51 + len(train_city)]
    test_x = raw_lag_rows(test_with_context, bundle.normalization, len(test_city))

    weights, _ = fit_mlp(
        train_x,
        train_y,
        city="iq",
        hidden_1=70,
        hidden_2=18,
        dropout_1=0.5,
        dropout_2=0.5,
        learning_rate=0.001,
    )
    train_mae = float(np.mean(np.abs(predict_mlp(weights, train_x) - train_y)))
    # The selected Iquitos pipeline used non-negative clipping followed by truncation.
    predictions = np.maximum(predict_mlp(weights, test_x), 0.0).astype(int)
    return predictions, train_mae


def prediction_frame(bundle: DataBundle, city: str, predictions: np.ndarray) -> pd.DataFrame:
    """Attach predictions to city keys in chronological order."""

    city_test = bundle.test.loc[bundle.test["city"].eq(city)].sort_values(DATE).reset_index(drop=True)
    result = city_test[KEYS].copy()
    result[TARGET] = np.asarray(predictions, int)
    return result


def assemble_submission(
    bundle: DataBundle,
    san_juan: np.ndarray,
    iquitos: np.ndarray,
) -> pd.DataFrame:
    """Assemble city predictions in the official submission row order."""

    predictions = pd.concat(
        [prediction_frame(bundle, "sj", san_juan), prediction_frame(bundle, "iq", iquitos)],
        ignore_index=True,
    )
    submission = bundle.template[KEYS].merge(
        predictions, on=KEYS, how="left", validate="one_to_one", sort=False
    )
    if submission[TARGET].isna().any() or not submission[KEYS].equals(bundle.template[KEYS]):
        raise ValueError("Generated submission violates the official row-order contract")
    submission[TARGET] = submission[TARGET].astype(int)
    return submission


def generate_submission(
    repo_root: Path,
    output_path: Path | None = None,
) -> SubmissionResult:
    """Train both city models and write a submission-ready CSV."""

    repo_root = repo_root.resolve()
    if output_path is None:
        output_path = repo_root / "Main" / "outputs" / "submission.csv"
    elif not output_path.is_absolute():
        output_path = repo_root / output_path

    bundle = load_data(repo_root)
    san_juan, san_juan_train_mae = san_juan_predictions(bundle)
    iquitos, iquitos_train_mae = iquitos_predictions(bundle)
    submission = assemble_submission(bundle, san_juan, iquitos)
    if len(submission) != EXPECTED_ROWS or (submission[TARGET] < 0).any():
        raise ValueError("Submission must contain 416 non-negative predictions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return SubmissionResult(
        output_path=output_path,
        rows=len(submission),
        san_juan_rows=int((submission["city"] == "sj").sum()),
        iquitos_rows=int((submission["city"] == "iq").sum()),
        san_juan_train_mae=san_juan_train_mae,
        iquitos_train_mae=iquitos_train_mae,
    )


def validate_san_juan_holdout(repo_root: Path) -> float:
    """Reproduce the historical 52-week San Juan validation MAE of 16.519."""

    bundle = load_data(repo_root.resolve())
    holdout_size = 52
    san_juan = bundle.train_interpolated.loc[
        bundle.train_interpolated["city"].eq("sj")
    ].sort_values(DATE).reset_index(drop=True)
    cut = len(san_juan) - holdout_size

    raw_san_juan = bundle.train.loc[bundle.train["city"].eq("sj")].sort_values(DATE).reset_index(drop=True)
    raw_fold_train = raw_san_juan.iloc[:cut]
    raw_holdout = raw_san_juan.iloc[cut:]
    raw_iquitos = bundle.train.loc[bundle.train["city"].eq("iq")].sort_values(DATE)
    iquitos_end = max(53, int(len(raw_iquitos) * cut / len(san_juan)))
    fold_frame = pd.concat([raw_fold_train, raw_iquitos.iloc[:iquitos_end]], ignore_index=True)

    fold_train = interpolate_numeric(fold_frame)
    fold_test = interpolate_numeric(raw_holdout.drop(columns=[TARGET]))
    fold_parameters = san_juan_normalization(fold_train)
    train_x, train_y, holdout_x = build_city_features(fold_train, fold_test, fold_parameters, "sj")
    holdout_y = raw_holdout[TARGET].to_numpy(float)
    weights, _ = fit_mlp(
        train_x,
        train_y,
        city="sj",
        hidden_1=100,
        hidden_2=25,
        dropout_1=0.3,
        dropout_2=0.7,
        learning_rate=0.01,
        epochs=28,
        steps=120,
        validation=(holdout_x, holdout_y),
    )
    prediction = round_nonnegative(predict_mlp(weights, holdout_x))
    return float(np.mean(np.abs(prediction - holdout_y)))
