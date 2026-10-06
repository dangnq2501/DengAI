"""Seaborn figures for the DengAI course presentation.

Uses the confirmed six-fold validation file and the same expanding windows
as src/train_tree_ensemble.py. Baselines are fit only on weeks before each
validation year, then rounded to integer cases the same way as the trees.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"

CITY_LABEL = {"iq": "Iquitos", "sj": "San Juan"}
CITY_COLOR = {"iq": "#E69F00", "sj": "#0072B2"}
FOLD_COUNT = 6
WEEKS = 52


def integer_cases(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.floor(np.maximum(0.0, values) + 0.5).astype(int)


def load_training() -> pd.DataFrame:
    features = pd.read_csv(DATA / "dengue_features_train.csv", parse_dates=["week_start_date"])
    labels = pd.read_csv(DATA / "dengue_labels_train.csv")
    frame = features.merge(labels, on=["city", "year", "weekofyear"], validate="one_to_one")
    return frame.sort_values(["city", "week_start_date"]).reset_index(drop=True)


def expanding_baselines(train: pd.DataFrame) -> pd.DataFrame:
    """Historical mean and week-of-year climatology on the last six 52-week folds."""
    pieces: list[pd.DataFrame] = []
    for city, city_frame in train.groupby("city", sort=True):
        city_frame = city_frame.sort_values("week_start_date").reset_index(drop=True)
        target = city_frame["total_cases"].to_numpy(float)
        week = city_frame["weekofyear"].to_numpy()
        mean_prediction = np.full(len(city_frame), np.nan)
        climate_prediction = np.full(len(city_frame), np.nan)
        starts = range(len(city_frame) - FOLD_COUNT * WEEKS, len(city_frame), WEEKS)
        for start in starts:
            history = target[:start]
            week_means = pd.Series(history).groupby(week[:start]).mean()
            stop = start + WEEKS
            history_mean = float(history.mean())
            mean_prediction[start:stop] = history_mean
            mapped = pd.Series(week[start:stop]).map(week_means).astype(float)
            climate_prediction[start:stop] = mapped.fillna(history_mean).to_numpy()
        window = city_frame.iloc[-FOLD_COUNT * WEEKS :].copy()
        window["mean_prediction"] = integer_cases(mean_prediction[-FOLD_COUNT * WEEKS :])
        window["climate_prediction"] = integer_cases(
            climate_prediction[-FOLD_COUNT * WEEKS :]
        )
        pieces.append(window)
    return pd.concat(pieces, ignore_index=True)


def validation_frame(train: pd.DataFrame) -> pd.DataFrame:
    baselines = expanding_baselines(train)
    trees = pd.read_csv(
        ARTIFACTS / "tree_validation_predictions.csv", parse_dates=["week_start_date"]
    )
    merged = trees.merge(
        baselines[
            [
                "city",
                "week_start_date",
                "mean_prediction",
                "climate_prediction",
            ]
        ],
        on=["city", "week_start_date"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(trees):
        raise RuntimeError("Baseline weeks do not match the saved tree validation")
    return merged


def style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
        }
    )


def save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_weekly_cases(train: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=False)
    for axis, city in zip(axes, ("sj", "iq")):
        city_frame = train.loc[train["city"].eq(city)]
        sns.lineplot(
            data=city_frame,
            x="week_start_date",
            y="total_cases",
            color=CITY_COLOR[city],
            linewidth=1.1,
            ax=axis,
        )
        validation_start = city_frame["week_start_date"].iloc[-FOLD_COUNT * WEEKS]
        axis.axvspan(
            validation_start,
            city_frame["week_start_date"].max(),
            color="#000000",
            alpha=0.06,
            label="Six-fold validation window",
        )
        axis.set_title(f"{CITY_LABEL[city]} weekly dengue cases")
        axis.set_xlabel("Week")
        axis.set_ylabel("Reported cases")
        axis.legend(frameon=False, loc="upper right")
    figure.tight_layout()
    save(figure, "01_weekly_cases.png")


def plot_seasonal_profile(train: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for axis, city in zip(axes, ("sj", "iq")):
        city_frame = train.loc[train["city"].eq(city)]
        sns.lineplot(
            data=city_frame,
            x="weekofyear",
            y="total_cases",
            estimator="mean",
            errorbar=("pi", 50),
            color=CITY_COLOR[city],
            linewidth=2,
            ax=axis,
        )
        axis.set_title(f"{CITY_LABEL[city]} seasonal profile")
        axis.set_xlabel("Week of year")
        axis.set_ylabel("Cases (mean and central 50%)")
        axis.set_xlim(1, 53)
    figure.tight_layout()
    save(figure, "02_seasonal_profile.png")


def plot_missingness(train: pd.DataFrame) -> None:
    feature_columns = [
        column
        for column in train.columns
        if column not in {"city", "year", "weekofyear", "week_start_date", "total_cases"}
    ]
    missing = (
        train.groupby("city")[feature_columns]
        .apply(lambda frame: frame.isna().mean())
        .reset_index()
        .melt(id_vars="city", var_name="feature", value_name="missing_fraction")
    )
    missing = missing.loc[missing["missing_fraction"].ge(0.01)].copy()
    missing["city_label"] = missing["city"].map(CITY_LABEL)
    missing["missing_percent"] = 100 * missing["missing_fraction"]
    order = (
        missing.groupby("feature")["missing_percent"].max().sort_values().index.tolist()
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(
        data=missing,
        y="feature",
        x="missing_percent",
        hue="city_label",
        order=order,
        hue_order=["San Juan", "Iquitos"],
        palette={"San Juan": CITY_COLOR["sj"], "Iquitos": CITY_COLOR["iq"]},
        ax=axis,
    )
    axis.set_title("Features missing on at least 1% of training weeks")
    axis.set_xlabel("Missing weeks (%)")
    axis.set_ylabel("")
    axis.legend(title="", frameon=False)
    figure.tight_layout()
    save(figure, "03_missingness.png")


def plot_climate_lags(train: pd.DataFrame) -> None:
    series = {
        "sj": [
            ("station_avg_temp_c", "Station temperature"),
            ("reanalysis_specific_humidity_g_per_kg", "Specific humidity"),
            ("reanalysis_dew_point_temp_k", "Dew-point temperature"),
        ],
        "iq": [
            ("reanalysis_specific_humidity_g_per_kg", "Specific humidity"),
            ("reanalysis_dew_point_temp_k", "Dew-point temperature"),
            ("station_avg_temp_c", "Station temperature"),
        ],
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    lags = list(range(0, 21))
    for axis, city in zip(axes, ("sj", "iq")):
        city_frame = train.loc[train["city"].eq(city)].sort_values("week_start_date")
        rows = []
        for column, label in series[city]:
            for lag in lags:
                rows.append(
                    {
                        "lag_weeks": lag,
                        "correlation": city_frame["total_cases"].corr(
                            city_frame[column].shift(lag)
                        ),
                        "driver": label,
                    }
                )
        sns.lineplot(
            data=pd.DataFrame(rows),
            x="lag_weeks",
            y="correlation",
            hue="driver",
            marker="o",
            markersize=4,
            ax=axis,
        )
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set_title(f"{CITY_LABEL[city]}: climate correlation with cases")
        axis.set_xlabel("Lag of the climate driver (weeks)")
        axis.set_ylabel("Pearson correlation with cases")
        axis.legend(title="", frameon=False, fontsize=9)
    figure.tight_layout()
    save(figure, "04_climate_lags.png")


def plot_validation_forecasts(validation: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=False)
    for axis, city in zip(axes, ("sj", "iq")):
        city_frame = validation.loc[validation["city"].eq(city)].sort_values(
            "week_start_date"
        )
        axis.plot(
            city_frame["week_start_date"],
            city_frame["actual"],
            color="#222222",
            linewidth=1.3,
            label="Actual cases",
        )
        axis.plot(
            city_frame["week_start_date"],
            city_frame["tree_prediction"],
            color="#009E73",
            linewidth=1.2,
            label="Confirmed tree",
        )
        axis.plot(
            city_frame["week_start_date"],
            city_frame["climate_prediction"],
            color="#D55E00",
            linewidth=1.0,
            alpha=0.9,
            label="Week-of-year climatology",
        )
        axis.set_title(f"{CITY_LABEL[city]} validation window")
        axis.set_xlabel("Week")
        axis.set_ylabel("Cases")
        axis.legend(frameon=False, ncol=3, fontsize=9)
    figure.tight_layout()
    save(figure, "05_validation_forecasts.png")


def plot_baseline_comparison(summary: pd.DataFrame) -> None:
    long = summary.melt(
        id_vars="city_label",
        value_vars=["mean_mae", "climate_mae", "tree_mae"],
        var_name="model",
        value_name="mae",
    )
    long["model"] = long["model"].map(
        {
            "mean_mae": "Historical mean",
            "climate_mae": "Week-of-year climatology",
            "tree_mae": "Confirmed tree",
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    sns.barplot(
        data=long,
        x="city_label",
        y="mae",
        hue="model",
        order=["Iquitos", "San Juan"],
        hue_order=[
            "Historical mean",
            "Week-of-year climatology",
            "Confirmed tree",
        ],
        palette=["#999999", "#D55E00", "#009E73"],
        ax=axis,
    )
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f", padding=2, fontsize=9)
    axis.set_title("One-year expanding-fold MAE, same 312 weeks per city")
    axis.set_xlabel("")
    axis.set_ylabel("Mean absolute error (cases)")
    axis.legend(title="", frameon=False)
    axis.set_ylim(0, long["mae"].max() * 1.18)
    figure.tight_layout()
    save(figure, "06_baseline_comparison.png")


def metrics(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, city_frame in validation.groupby("city"):
        actual = city_frame["actual"].to_numpy()
        threshold = np.quantile(actual, 0.90)
        outbreak = actual >= threshold
        rows.append(
            {
                "city": city,
                "city_label": CITY_LABEL[city],
                "mean_mae": np.mean(np.abs(actual - city_frame["mean_prediction"])),
                "climate_mae": np.mean(
                    np.abs(actual - city_frame["climate_prediction"])
                ),
                "tree_mae": np.mean(np.abs(actual - city_frame["tree_prediction"])),
                "tree_bias": np.mean(city_frame["tree_prediction"] - actual),
                "outbreak_mae": np.mean(
                    np.abs(actual[outbreak] - city_frame["tree_prediction"].to_numpy()[outbreak])
                ),
                "endemic_mae": np.mean(
                    np.abs(
                        actual[~outbreak] - city_frame["tree_prediction"].to_numpy()[~outbreak]
                    )
                ),
                "outbreak_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    style()
    train = load_training()
    validation = validation_frame(train)
    summary = metrics(validation)
    plot_weekly_cases(train)
    plot_seasonal_profile(train)
    plot_missingness(train)
    plot_climate_lags(train)
    plot_validation_forecasts(validation)
    plot_baseline_comparison(summary)
    summary.to_csv(FIGURES / "baseline_comparison.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
