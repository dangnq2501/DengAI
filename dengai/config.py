"""Shared constants: column names, climate variables, and model settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

KEY_COLUMNS = ["city", "year", "weekofyear"]
DATE_COLUMN = "week_start_date"
TARGET_COLUMN = "total_cases"
CITIES = ("sj", "iq")
CITY_NAMES = {"sj": "San Juan", "iq": "Iquitos"}

# Weeks in each city's hidden test set. They weight the two city MAEs into one
# "competition-weighted" MAE that mirrors the leaderboard's pooled average.
TEST_WEEKS = {"sj": 260, "iq": 156}

NDVI_COLUMNS = ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]

# The 16 weekly climate measurements used by the neural-network representations.
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

# Contiguous history length (weeks) per variable, selected per city by a
# decision-tree screen of windows 3..51. A window of 40 keeps all 40 weeks
# x[t-39..t], not only lag 40.
LAG_WINDOWS = {
    "sj": {
        "precipitation_amt_mm": 40,
        "reanalysis_air_temp_k": 16,
        "reanalysis_avg_temp_k": 15,
        "reanalysis_dew_point_temp_k": 39,
        "reanalysis_max_air_temp_k": 12,
        "reanalysis_min_air_temp_k": 21,
        "reanalysis_precip_amt_kg_per_m2": 30,
        "reanalysis_relative_humidity_percent": 34,
        "reanalysis_sat_precip_amt_mm": 40,
        "reanalysis_specific_humidity_g_per_kg": 14,
        "reanalysis_tdtr_k": 21,
        "station_avg_temp_c": 41,
        "station_diur_temp_rng_c": 40,
        "station_max_temp_c": 37,
        "station_min_temp_c": 26,
        "station_precip_mm": 32,
        "weekofyear": 3,
    },
    "iq": {
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
    },
}

# Multiscale summary windows (weeks).
MEAN_WINDOWS = (2, 4, 8, 13, 26, 52)
STD_WINDOWS = (4, 13)
SUMMARY_NAMES = (
    ["current"]
    + [f"mean_{w}w" for w in MEAN_WINDOWS]
    + [f"std_{w}w" for w in STD_WINDOWS]
    + ["mean4_minus_mean13", "trend_13w"]
)
SEASONAL_NAMES = ["season_sin_1", "season_cos_1", "season_sin_2", "season_cos_2"]

# History needed per endpoint and the prepended context used to build it.
HISTORY_WEEKS = 52
CONTEXT_WEEKS = 53
# Predictions for week t use climate history that ends at week t - 2. This is
# the alignment of the leaderboard-confirmed model (see notebook 03).
PREDICTION_LAG = 2


@dataclass(frozen=True)
class MLPConfig:
    """The tunable parts of the three-Dense-layer SELU network."""

    hidden_1: int
    hidden_2: int
    dropout_1: float
    dropout_2: float
    learning_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingSchedule:
    """RMSprop on MAE: 40 epochs x 200 mini-batches of 16 rows."""

    epochs: int = 40
    steps_per_epoch: int = 200
    batch_size: int = 16


# Configuration confirmed on the hidden leaderboard (16.6 MAE).
FINAL_MLP_CONFIGS = {
    "sj": MLPConfig(100, 25, 0.30, 0.70, 0.010),
    "iq": MLPConfig(70, 18, 0.50, 0.50, 0.001),
}

TREE_PARAMS = {
    "iq": {"max_features": 0.5, "min_samples_leaf": 5, "max_depth": None},
    "sj": {"max_features": 1.0, "min_samples_leaf": 5, "max_depth": None},
}

# Hidden-test MAE of each submitted stage, as returned by DrivenData.
LEADERBOARD_HISTORY = [
    ("Earlier baseline", 26.7),
    ("City-specific trees", 22.5),
    ("Raw-lag MLP", 19.3),
    ("Raw-lag MLP, dropout 0.3/0.7", 19.1),
    ("Raw-lag MLP, tuned widths", 18.8),
    ("Multiscale MLP (final)", 16.6),
    ("Multiscale MLP, 48→12 (rejected)", 18.1),
]
