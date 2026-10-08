"""Regenerate report figures that changed with the final 15.9 PLS result."""

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
    "Multiscale\n+ 4 PLS",
]
scores = np.array([26.7, 22.5, 18.8, 16.6, 18.1, 15.9])
colors = ["#64748b", "#64748b", "#64748b", "#0f766e", "#b91c1c", "#d9480f"]

fig, ax = plt.subplots(figsize=(13, 6.4))
bars = ax.bar(labels, scores, color=colors, width=.68)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, score + .35, f"{score:.1f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.text(4, 12.2, "REJECTED", ha="center", color="white", fontsize=11, fontweight="bold")
ax.text(5, 11.8, "FINAL", ha="center", color="white", fontsize=11, fontweight="bold")
ax.annotate("small supervised augmentation\nwithout changing the MLP",
            xy=(5, 15.9), xytext=(4.05, 23.8),
            arrowprops={"arrowstyle": "->", "color": "#d9480f", "lw": 1.7},
            ha="center", color="#7c2d12", fontsize=10.5)
ax.set(title="Representation changes produced the largest leaderboard gains",
       ylabel="Hidden-test MAE — lower is better", ylim=(0, 29))
ax.grid(axis="y", alpha=.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(FIGURES / "06_leaderboard_journey.png", dpi=200, bbox_inches="tight")
plt.close(fig)

source = ROOT / "Main" / "outputs" / "pls_components_actual.png"
target = FIGURES / "pls_components_actual.png"
target.write_bytes(source.read_bytes())
print("Wrote", FIGURES / "06_leaderboard_journey.png")
print("Wrote", target)
