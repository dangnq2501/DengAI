"""Linear reductions for multiscale climate vectors (PCA, per-variable blocks).

t-SNE and UMAP are intentionally excluded: they do not define a stable,
causal map from new test-week features to coordinates without leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_lag_mlp.representations import (
    SEASONAL_FEATURE_COUNT,
    SUMMARY_VALUES_PER_WEATHER_FEATURE,
    SUMMARY_DIMENSION,
)


@dataclass(frozen=True)
class ReductionSpec:
    name: str
    method: str  # "none" | "global_pca" | "block_pca"
    n_components: int | None = None
    block_components: int | None = None


def _standardize_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _standardize_apply(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (matrix - mean) / std


def _pca_fit_transform(
    train: np.ndarray,
    apply: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean, std = _standardize_fit(train)
    train_z = _standardize_apply(train, mean, std)
    apply_z = _standardize_apply(apply, mean, std)
    n_components = min(n_components, train_z.shape[1], max(1, train_z.shape[0] - 1))
    _, _, vt = np.linalg.svd(train_z, full_matrices=False)
    components = vt[:n_components]
    return train_z @ components.T, apply_z @ components.T, {
        "mean": mean,
        "std": std,
        "components": components,
    }


def _block_pca_fit_transform(
    train: np.ndarray,
    apply: np.ndarray,
    block_components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if train.shape[1] != SUMMARY_DIMENSION:
        raise ValueError(f"Expected {SUMMARY_DIMENSION} multiscale columns")
    weather_dim = SUMMARY_VALUES_PER_WEATHER_FEATURE
    block_count = (SUMMARY_DIMENSION - SEASONAL_FEATURE_COUNT) // weather_dim
    block_components = min(block_components, weather_dim)

    train_parts: list[np.ndarray] = []
    apply_parts: list[np.ndarray] = []
    blocks: list[dict[str, np.ndarray]] = []
    for block_index in range(block_count):
        start = block_index * weather_dim
        end = start + weather_dim
        train_block = train[:, start:end]
        apply_block = apply[:, start:end]
        train_red, apply_red, state = _pca_fit_transform(
            train_block, apply_block, block_components
        )
        train_parts.append(train_red)
        apply_parts.append(apply_red)
        blocks.append(state)

    seasonal_start = block_count * weather_dim
    train_parts.append(train[:, seasonal_start:])
    apply_parts.append(apply[:, seasonal_start:])
    state: dict[str, object] = {"blocks": blocks, "seasonal_start": seasonal_start}
    return np.hstack(train_parts), np.hstack(apply_parts), state


def transform_with_state(matrix: np.ndarray, state: dict[str, object], method: str) -> np.ndarray:
    if method == "global_pca":
        mean = state["mean"]  # type: ignore[index]
        std = state["std"]  # type: ignore[index]
        components = state["components"]  # type: ignore[index]
        z = _standardize_apply(matrix, mean, std)
        return z @ components.T
    if method == "block_pca":
        weather_dim = SUMMARY_VALUES_PER_WEATHER_FEATURE
        blocks = state["blocks"]  # type: ignore[index]
        parts: list[np.ndarray] = []
        for block_index, block_state in enumerate(blocks):
            start = block_index * weather_dim
            end = start + weather_dim
            block = matrix[:, start:end]
            z = _standardize_apply(block, block_state["mean"], block_state["std"])
            parts.append(z @ block_state["components"].T)
        parts.append(matrix[:, state["seasonal_start"] :])  # type: ignore[index]
        return np.hstack(parts)
    raise ValueError(f"Unknown method: {method}")


def reduce_multiscale(
    train: np.ndarray,
    apply: np.ndarray,
    spec: ReductionSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, object] | None]:
    if spec.method == "none":
        return train, apply, None
    if spec.method == "global_pca":
        if spec.n_components is None:
            raise ValueError("global_pca requires n_components")
        train_red, apply_red, state = _pca_fit_transform(
            train, apply, spec.n_components
        )
        state["method"] = "global_pca"
        return train_red, apply_red, state
    if spec.method == "block_pca":
        if spec.block_components is None:
            raise ValueError("block_pca requires block_components")
        train_red, apply_red, state = _block_pca_fit_transform(
            train, apply, spec.block_components
        )
        state["method"] = "block_pca"
        return train_red, apply_red, state
    raise ValueError(f"Unknown reduction method: {spec.method}")


def default_search_grid() -> list[ReductionSpec]:
    specs = [ReductionSpec("baseline_180", "none")]
    for k in (72, 96, 120, 144):
        specs.append(ReductionSpec(f"global_pca_{k}", "global_pca", n_components=k))
    for k in (3, 4, 5, 6):
        dim = 16 * k + SEASONAL_FEATURE_COUNT
        specs.append(
            ReductionSpec(f"block_pca_{k}", "block_pca", block_components=k)
        )
    return specs
