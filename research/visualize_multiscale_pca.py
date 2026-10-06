"""EDA: PCA variance and 2D projection of multiscale training rows (not for inference)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_lag_mlp.config import TARGET_COLUMN
from feature_lag_mlp.data import load_competition_data
from feature_lag_mlp.representations import MULTISCALE_SUMMARIES, build_representation_training_matrix

OUT = ROOT / "research" / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    train, _, _ = load_competition_data(ROOT / "data")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, city in enumerate(("sj", "iq")):
        x, y = build_representation_training_matrix(train, city, MULTISCALE_SUMMARIES)
        x_z = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)
        _, s, vt = np.linalg.svd(x_z, full_matrices=False)
        variance = (s**2) / (s**2).sum()
        cum = np.cumsum(variance)
        ax_var = axes[row, 0]
        ax_var.plot(np.arange(1, len(cum) + 1), cum, marker=".", ms=3)
        ax_var.axhline(0.95, color="gray", ls="--", lw=1)
        ax_var.set_title(f"{city.upper()}: PCA scree")
        ax_var.set_xlabel("components")
        ax_var.set_ylabel("cumulative explained variance")
        pc1 = x_z @ vt[0]
        ax_sc = axes[row, 1]
        ax_sc.scatter(pc1, y, c=y, cmap="viridis", s=10, alpha=0.6)
        ax_sc.set_title(f"{city.upper()}: PC1 vs cases")
        ax_sc.set_xlabel("first PC")
        ax_sc.set_ylabel("total_cases")
    plt.tight_layout()
    path = OUT / "multiscale_pca_eda.png"
    plt.savefig(path, dpi=120)
    print(f"Wrote {path}")
    print(
        "Note: t-SNE/UMAP are for visualization only; this repo uses causal PCA "
        "in search_multiscale_reduction.py for test-time transforms."
    )


if __name__ == "__main__":
    main()
