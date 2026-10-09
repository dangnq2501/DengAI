"""Regenerate report figures for the hash-verified 15.9832 submission."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

labels = [
    "First\nbaseline",
    "Tree\nensemble",
    "Tuned raw\nhistories",
    "Multiscale\nsummaries",
    "Narrow\nmultiscale",
    "Verified\ncity hybrid",
]
scores = np.array([26.7, 22.5, 18.8, 16.6, 18.1, 15.9832])
colors = ["#64748b", "#64748b", "#64748b", "#0f766e", "#b91c1c", "#d9480f"]

fig, ax = plt.subplots(figsize=(13, 6.4))
bars = ax.bar(labels, scores, color=colors, width=.68)
for bar, score in zip(bars, scores):
    label = f"{score:.4f}" if np.isclose(score, 15.9832) else f"{score:.1f}"
    ax.text(bar.get_x() + bar.get_width() / 2, score + .35, label,
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.text(4, 12.2, "REJECTED", ha="center", color="white", fontsize=11, fontweight="bold")
ax.text(5, 11.8, "FINAL", ha="center", color="white", fontsize=11, fontweight="bold")
ax.annotate("multiscale San Juan + raw-history Iquitos\nseed 42, nearest rounding, exact hash",
            xy=(5, 15.9832), xytext=(4.0, 24.2),
            arrowprops={"arrowstyle": "->", "color": "#d9480f", "lw": 1.7},
            ha="center", color="#7c2d12", fontsize=10.5)
ax.set(title="Representation changes produced the largest leaderboard gains",
       ylabel="Public leaderboard MAE — lower is better", ylim=(0, 29))
ax.grid(axis="y", alpha=.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(FIGURES / "06_leaderboard_journey.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Wrote", FIGURES / "06_leaderboard_journey.png")
