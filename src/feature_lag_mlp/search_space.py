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

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            hidden_1=self.hidden_1,
            hidden_2=self.hidden_2,
            dropout_1=self.dropout_1,
            dropout_2=self.dropout_2,
            learning_rate=self.learning_rate,
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
    )
