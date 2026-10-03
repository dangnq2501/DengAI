"""Create presentation-ready diagnostics for the feature-lag MLP workflow.

The plots answer four questions:

1. How different are the two city target distributions?
2. How do cases change through time?
3. How strong is short- versus annual-lag target dependence?
4. Which tuning candidate wins on normal and outbreak MAE?
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

# Matplotlib must be configured before it is imported. A temporary cache makes
# the script work in read-only home-directory environments as well as locally.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dengai-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_lag_mlp.config import CITIES, DATE_COLUMN, TARGET_COLUMN
from feature_lag_mlp.data import load_competition_data


CITY_LABELS = {"sj": "San Juan", "iq": "Iquitos"}
CITY_COLORS = {"sj": "#2563eb", "iq": "#ea580c"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--artifact-dir", type=Path, default=project_dir / "artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "artifacts" / "feature_lag_mlp_figures",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 10,
        }
    )


def plot_target_overview(train: pd.DataFrame, output_dir: Path) -> Path:
    """Show chronological cases and city-specific target distributions."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for row, city in enumerate(CITIES):
        city_data = train.loc[train["city"].eq(city)].sort_values(DATE_COLUMN)
        color = CITY_COLORS[city]
        label = CITY_LABELS[city]

        axes[row, 0].plot(
            city_data[DATE_COLUMN], city_data[TARGET_COLUMN], color=color, linewidth=1
        )
        axes[row, 0].set_title(f"{label}: weekly cases over time")
        axes[row, 0].set_ylabel("total cases")

        axes[row, 1].hist(
            city_data[TARGET_COLUMN], bins=30, color=color, alpha=0.85, edgecolor="white"
        )
        axes[row, 1].axvline(
            city_data[TARGET_COLUMN].median(),
            color="#111827",
            linestyle="--",
            label=f"median = {city_data[TARGET_COLUMN].median():.0f}",
        )
        axes[row, 1].set_title(f"{label}: skewed target distribution")
        axes[row, 1].set_xlabel("total cases")
        axes[row, 1].set_ylabel("weeks")
        axes[row, 1].legend(frameon=False)

    path = output_dir / "01_target_distribution_and_timeline.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_target_autocorrelation(train: pd.DataFrame, output_dir: Path) -> Path:
    """Compare persistence from one week through one year."""
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    lags = np.arange(1, 53)
    for city in CITIES:
        target = (
            train.loc[train["city"].eq(city)]
            .sort_values(DATE_COLUMN)[TARGET_COLUMN]
            .reset_index(drop=True)
        )
        correlations = [target.autocorr(int(lag)) for lag in lags]
        axis.plot(
            lags,
            correlations,
            label=CITY_LABELS[city],
            color=CITY_COLORS[city],
            linewidth=2,
        )
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.axvline(52, color="#6b7280", linestyle="--", linewidth=1)
    axis.set(title="Target persistence decays well before one year", xlabel="lag (weeks)", ylabel="correlation")
    axis.legend(frameon=False)
    path = output_dir / "02_target_autocorrelation.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _read_tuning_table(artifact_dir: Path, city: str, suffix: str) -> pd.DataFrame:
    path = artifact_dir / f"feature_lag_mlp_tuning_{city}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run tuning first or pass the correct --artifact-dir."
        )
    return pd.read_csv(path)


def plot_candidate_mae(artifact_dir: Path, output_dir: Path) -> Path:
    """Plot confirmation MAE with seed/fold variation for each finalist."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, city in zip(axes, CITIES):
        summary = _read_tuning_table(
            artifact_dir, city, "confirmation_summary"
        ).sort_values("selection_score", ascending=False)
        colors = [
            "#16a34a" if name == summary.iloc[-1]["candidate"] else "#94a3b8"
            for name in summary["candidate"]
        ]
        axis.barh(
            summary["candidate"],
            summary["mean_mae"],
            xerr=summary["std_mae"],
            color=colors,
            alpha=0.9,
            capsize=3,
        )
        axis.set(
            title=f"{CITY_LABELS[city]} confirmation MAE",
            xlabel="mean MAE ± standard deviation",
            ylabel="candidate",
        )
    path = output_dir / "03_confirmation_candidate_mae.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_outbreak_error(artifact_dir: Path, output_dir: Path) -> Path:
    """Reveal candidates that improve average MAE but miss outbreak peaks."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, city in zip(axes, CITIES):
        runs = _read_tuning_table(artifact_dir, city, "confirmation_runs")
        candidates = runs["candidate"].drop_duplicates().tolist()
        color_map = dict(
            zip(candidates, plt.cm.tab10(np.linspace(0, 1, len(candidates))))
        )
        for candidate, group in runs.groupby("candidate", sort=False):
            axis.scatter(
                group["mae"],
                group["outbreak_mae"],
                label=candidate,
                color=color_map[candidate],
                s=35,
                alpha=0.8,
            )
        axis.set(
            title=f"{CITY_LABELS[city]}: overall vs outbreak error",
            xlabel="overall MAE (lower is better)",
            ylabel="outbreak MAE (lower is better)",
        )
        axis.legend(frameon=False, fontsize=8)
    path = output_dir / "04_overall_vs_outbreak_mae.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    train, _, _ = load_competition_data(args.data_dir)
    paths = [
        plot_target_overview(train, args.output_dir),
        plot_target_autocorrelation(train, args.output_dir),
        plot_candidate_mae(args.artifact_dir, args.output_dir),
        plot_outbreak_error(args.artifact_dir, args.output_dir),
    ]
    print("Wrote figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
