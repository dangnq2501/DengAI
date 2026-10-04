"""Candidate architectures considered by the small, controlled model search."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ModelConfig


@dataclass(frozen=True)
class Candidate:
    """A named model configuration so experiment tables remain readable."""

    name: str
    hidden_1: int
    hidden_2: int
    dropout_1: float
    dropout_2: float
    learning_rate: float
    linear_skip: bool = False
    l2_strength: float = 0.0
    feature_noise_std: float = 0.0

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            hidden_1=self.hidden_1,
            hidden_2=self.hidden_2,
            dropout_1=self.dropout_1,
            dropout_2=self.dropout_2,
            learning_rate=self.learning_rate,
            linear_skip=self.linear_skip,
            l2_strength=self.l2_strength,
        )

    @property
    def hidden_widths(self) -> tuple[int, int]:
        return self.config.hidden_widths

    @property
    def dropout_rates(self) -> tuple[float, float]:
        return self.config.dropout_rates


def candidate_grid(city: str) -> list[Candidate]:
    """Return the hypothesis-driven grid, with the 19.1 setup as control."""
    if city == "sj":
        values = [
            ("current_19_1", 100, 50, 0.30, 0.70, 0.010),
            ("author_dropout", 100, 50, 0.50, 0.50, 0.010),
            ("width_64_32", 64, 32, 0.30, 0.70, 0.010),
            ("width_80_40", 80, 40, 0.30, 0.70, 0.010),
            ("width_128_64", 128, 64, 0.30, 0.70, 0.010),
            ("width_160_80", 160, 80, 0.30, 0.70, 0.010),
            ("narrow_second", 100, 25, 0.30, 0.70, 0.010),
            ("dropout_20_60", 100, 50, 0.20, 0.60, 0.010),
            ("dropout_25_65", 100, 50, 0.25, 0.65, 0.010),
            ("dropout_35_65", 100, 50, 0.35, 0.65, 0.010),
            ("dropout_40_60", 100, 50, 0.40, 0.60, 0.010),
            ("dropout_40_80", 100, 50, 0.40, 0.80, 0.010),
            ("lr_005", 100, 50, 0.30, 0.70, 0.005),
            ("lr_003", 100, 50, 0.30, 0.70, 0.003),
        ]
    elif city == "iq":
        values = [
            ("current_19_1", 70, 35, 0.50, 0.50, 0.0010),
            ("progressive_dropout", 70, 35, 0.30, 0.70, 0.0010),
            ("width_48_24", 48, 24, 0.50, 0.50, 0.0010),
            ("width_64_32", 64, 32, 0.50, 0.50, 0.0010),
            ("width_96_48", 96, 48, 0.50, 0.50, 0.0010),
            ("width_128_64", 128, 64, 0.50, 0.50, 0.0010),
            ("narrow_second", 70, 18, 0.50, 0.50, 0.0010),
            ("dropout_30_50", 70, 35, 0.30, 0.50, 0.0010),
            ("dropout_40_60", 70, 35, 0.40, 0.60, 0.0010),
            ("dropout_50_70", 70, 35, 0.50, 0.70, 0.0010),
            ("dropout_20_60", 70, 35, 0.20, 0.60, 0.0010),
            ("dropout_60_60", 70, 35, 0.60, 0.60, 0.0010),
            ("lr_0005", 70, 35, 0.50, 0.50, 0.0005),
            ("lr_002", 70, 35, 0.50, 0.50, 0.0020),
        ]
    else:
        raise ValueError(f"Unknown city: {city}")
    return [Candidate(*value) for value in values]


def candidate_from_summary(row: pd.Series) -> Candidate:
    """Rebuild a candidate from a saved experiment-summary row."""
    return Candidate(
        name=str(row["candidate"]),
        hidden_1=int(row["hidden_1"]),
        hidden_2=int(row["hidden_2"]),
        dropout_1=float(row["dropout_1"]),
        dropout_2=float(row["dropout_2"]),
        learning_rate=float(row["learning_rate"]),
        linear_skip=(
            bool(row["linear_skip"])
            if "linear_skip" in row and not pd.isna(row["linear_skip"])
            else False
        ),
        l2_strength=(
            float(row["l2_strength"])
            if "l2_strength" in row and not pd.isna(row["l2_strength"])
            else 0.0
        ),
        feature_noise_std=(
            float(row["feature_noise_std"])
            if "feature_noise_std" in row and not pd.isna(row["feature_noise_std"])
            else 0.0
        ),
    )


def robustness_grid(city: str) -> list[Candidate]:
    """Isolate small complexity and data-augmentation changes from the winner."""
    if city == "sj":
        widths = (100, 25)
        dropout = (0.30, 0.70)
        learning_rate = 0.010
    elif city == "iq":
        widths = (70, 18)
        dropout = (0.50, 0.50)
        learning_rate = 0.001
    else:
        raise ValueError(f"Unknown city: {city}")

    base = (*widths, *dropout, learning_rate)
    return [
        Candidate("tuned_seed42_control", *base),
        Candidate("climate_noise_005", *base, feature_noise_std=0.005),
        Candidate("climate_noise_010", *base, feature_noise_std=0.010),
        Candidate("climate_noise_020", *base, feature_noise_std=0.020),
        Candidate(
            "linear_skip_l2_1e4",
            *base,
            linear_skip=True,
            l2_strength=1e-4,
        ),
        Candidate(
            "linear_skip_noise_010",
            *base,
            linear_skip=True,
            l2_strength=1e-4,
            feature_noise_std=0.010,
        ),
    ]


def multiscale_sj_architecture_grid() -> list[Candidate]:
    """Test capacity and regularization for the 180-value SJ representation.

    The raw-lag winner is retained as the control.  Each other candidate changes
    one design axis at a time, which keeps the result interpretable and limits
    selection noise on this small time series.
    """
    values = [
        ("summary_control", 100, 25, 0.30, 0.70, 0.010),
        ("width_48_12", 48, 12, 0.30, 0.70, 0.010),
        ("width_64_16", 64, 16, 0.30, 0.70, 0.010),
        ("width_80_20", 80, 20, 0.30, 0.70, 0.010),
        ("width_128_32", 128, 32, 0.30, 0.70, 0.010),
        ("dropout_15_45", 100, 25, 0.15, 0.45, 0.010),
        ("dropout_20_50", 100, 25, 0.20, 0.50, 0.010),
        ("dropout_20_60", 100, 25, 0.20, 0.60, 0.010),
        ("dropout_30_60", 100, 25, 0.30, 0.60, 0.010),
        ("lr_005", 100, 25, 0.30, 0.70, 0.005),
    ]
    return [Candidate(*value) for value in values]
