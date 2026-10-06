"""Convert continuous model outputs into competition integer case counts."""

from __future__ import annotations

import numpy as np


def to_submission_cases(
    raw: np.ndarray,
    *,
    mode: str = "nearest",
) -> np.ndarray:
    """Map non-negative floats to integer ``total_cases`` predictions."""
    values = np.asarray(raw, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite")
    clipped = np.maximum(values, 0.0)
    if mode == "nearest":
        return np.floor(clipped + 0.5).astype(np.int64)
    if mode == "floor":
        return np.floor(clipped).astype(np.int64)
    if mode == "truncate":
        return clipped.astype(np.int64)
    raise ValueError(f"Unknown rounding mode: {mode!r}")


def blend_raw(
    primary: np.ndarray,
    secondary: np.ndarray,
    weight_primary: float,
) -> np.ndarray:
    """Convex combination of two raw (pre-rounding) prediction vectors."""
    if not 0.0 <= weight_primary <= 1.0:
        raise ValueError("weight_primary must be in [0, 1]")
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    if primary.shape != secondary.shape:
        raise ValueError("Prediction vectors must have the same shape")
    return weight_primary * primary + (1.0 - weight_primary) * secondary
