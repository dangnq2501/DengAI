"""All fixed feature, architecture, and training configuration.

Keeping constants here makes the scientific assumptions visible without
having to search through training code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


KEY_COLUMNS = ["city", "year", "weekofyear"]
DATE_COLUMN = "week_start_date"
TARGET_COLUMN = "total_cases"
CITIES = ("sj", "iq")

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

# These are the final PPS-labelled windows in the historical experiment. The
# source does not contain the search that produced them, so this project treats
# them as fixed, auditable model configuration rather than recomputed facts.
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

INPUT_DIMENSIONS = {
    city: sum(windows.values()) for city, windows in LAG_WINDOWS.items()
}

SHARED_SJ_NORMALIZATION = "shared-sj"
CITY_SPECIFIC_NORMALIZATION = "city-specific"
NORMALIZATION_CHOICES = (
    SHARED_SJ_NORMALIZATION,
    CITY_SPECIFIC_NORMALIZATION,
)


@dataclass(frozen=True)
class ModelConfig:
    """The only tunable parts of the three-Dense-layer MLP."""

    hidden_1: int
    hidden_2: int
    dropout_1: float
    dropout_2: float
    learning_rate: float

    @property
    def hidden_widths(self) -> tuple[int, int]:
        return (self.hidden_1, self.hidden_2)

    @property
    def dropout_rates(self) -> tuple[float, float]:
        return (self.dropout_1, self.dropout_2)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


REFERENCE_CONFIGS = {
    "sj": ModelConfig(100, 50, 0.5, 0.5, 0.01),
    "iq": ModelConfig(70, 35, 0.5, 0.5, 0.001),
}

IMPROVED_CONFIGS = {
    "sj": ModelConfig(100, 50, 0.3, 0.7, 0.01),
    "iq": REFERENCE_CONFIGS["iq"],
}

TUNED_CONFIGS = {
    "sj": ModelConfig(100, 25, 0.3, 0.7, 0.01),
    "iq": ModelConfig(70, 18, 0.5, 0.5, 0.001),
}

PROFILE_CONFIGS = {
    "reference": REFERENCE_CONFIGS,
    "improved": IMPROVED_CONFIGS,
    "tuned": TUNED_CONFIGS,
}


@dataclass(frozen=True)
class TrainingSchedule:
    """Optimization schedule shared by training and temporal validation."""

    epochs: int = 40
    steps_per_epoch: int = 200
    batch_size: int = 16
    lr_mode: str = "max"

