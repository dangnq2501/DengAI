"""One plotting style for every notebook figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Validated categorical slots (fixed order): cities first, then models.
CITY_COLORS = {"sj": "#2a78d6", "iq": "#eb6834", "San Juan": "#2a78d6", "Iquitos": "#eb6834"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#8a8984"
ACTUAL_COLOR = "#2b2b2a"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette=SERIES)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.figsize": (11, 4),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c9c8c2",
            "grid.color": "#ecebe6",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "lines.linewidth": 1.6,
            "legend.frameon": False,
        }
    )
