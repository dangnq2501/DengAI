# DengAI dengue forecasting

Final model: a multiscale-summary MLP with hidden-test MAE **16.6**
(from 22.5 for the tree ensemble and 26.7 for the first baseline).

## Course notebooks (start here)

The project report is a sequence of five notebooks in [`notebooks/`](notebooks/),
one per grading item of [`project_guidelines.md`](project_guidelines.md):

| Notebook | Rubric item |
| --- | --- |
| [`01_problem_and_data`](notebooks/01_problem_and_data.ipynb) | Problem and motivation; load with pandas, split X / y |
| [`02_exploratory_analysis`](notebooks/02_exploratory_analysis.ipynb) | Seaborn EDA, each finding mapped to a decision |
| [`03_preprocessing_and_features`](notebooks/03_preprocessing_and_features.ipynb) | Encoding, missing values, target, three feature sets |
| [`04_validation_and_model_selection`](notebooks/04_validation_and_model_selection.ipynb) | Temporal CV, metric, baselines, trees, MLP, hyper-parameter search |
| [`05_results_and_conclusions`](notebooks/05_results_and_conclusions.ipynb) | Error analysis, importance, submission, impact, conclusions |

The notebooks import the small [`dengai/`](dengai/) package (data loading,
features, models, validation), which runs without TensorFlow.
[`tests/test_dengai_package.py`](tests/test_dengai_package.py) checks that it
reproduces the original `src/` feature matrices and NumPy MLP exactly.

Open them in Jupyter or VS Code with the `bkk` kernel and run all cells in order
(notebook 05 reads the outputs of 04; 04 takes several minutes on a CPU), or
execute all of them from the command line:

```bash
conda activate bkk
python notebooks/run_all.py            # or e.g. `python notebooks/run_all.py 04 05`
python -m unittest tests.test_dengai_package -v
```

Generated files (CV predictions, selected configuration, submission
`submission_multiscale_mlp.csv`) are written to `notebooks/outputs/`.

The sections below document the original experiment scripts in `src/`,
`research/` and `research2/`. They are kept for reference.

## Tree baseline model (22.5)

- **Iquitos:** `RandomForestRegressor` trained on `log1p(total_cases)`.
- **San Juan:** `ExtraTreesRegressor` using the Poisson split criterion.
- **Features:** seasonal harmonics, raw climate measurements, NDVI summaries,
  temperature interactions, 1–16 week climate lags, and trailing climate means.
- **Validation:** expanding annual folds without shuffled rows.

Case-count lags are intentionally excluded because true prior test labels are
not available through the multi-year forecast horizon.

## Project structure

```text
DengAI/
├── data/                         # Supplied competition data
├── src/train_tree_ensemble.py   # Complete training and prediction pipeline
├── tests/test_pipeline.py       # Data, feature, model, and submission checks
├── artifacts/
│   ├── submission_tree_ensemble.csv
│   ├── tree_validation_scores.csv
│   ├── tree_validation_predictions.csv
│   ├── tree_ensemble_models.joblib
│   ├── submission_tree_tuned.csv
│   ├── tree_tuning_results.csv
│   └── tree_tuning_best_params.json
├── requirements.txt
└── README.md
```

## Setup

From this directory:

```bash
python3 -m venv ../../.venv
../../.venv/bin/python -m pip install -r requirements.txt
```

## Train and validate

```bash
../../.venv/bin/python src/train_tree_ensemble.py
```

For a faster final-model rebuild without temporal backtesting:

```bash
../../.venv/bin/python src/train_tree_ensemble.py --skip-backtest
```

The competition-ready output is:

[`artifacts/submission_tree_ensemble.csv`](artifacts/submission_tree_ensemble.csv)

This confirmed 22.5-MAE file is kept unchanged. A normal rebuild is written to
`submission_tree_ensemble_2.csv` so that it cannot accidentally replace the
confirmed submission.

Expected prediction distributions with the default seed and 500 trees:

| City | Mean | Maximum |
| --- | ---: | ---: |
| Iquitos | 6.14 | 16 |
| San Juan | 32.17 | 99 |

## Conservative hyperparameter tuning

Run the city-specific chronological grid search with:

```bash
../../.venv/bin/python src/train_tree_ensemble.py \
  --tune \
  --tuning-estimators 200 \
  --n-estimators 500
```

The search uses four expanding one-year validation folds plus a recent holdout
whose length matches each city's competition-test horizon. It rejects settings
that materially reduce the validation prediction mean or peak, which helps
avoid selecting the overly smooth San Juan models seen in earlier experiments.
No shuffled K-fold validation is used.

Tuning writes separate files and never replaces the confirmed submission:

- `tree_tuning_results.csv`: every candidate and its validation metrics.
- `tree_tuning_best_params.json`: selected parameters by city.
- `tree_tuned_validation_scores.csv` and
  `tree_tuned_validation_predictions.csv`: full expanding-fold diagnostics.
- `submission_tree_tuned.csv` and `tree_tuned_models.joblib`: final candidate.

Use fewer trees for a quick search, then keep 500 trees for final fitting:

```bash
../../.venv/bin/python src/train_tree_ensemble.py \
  --tune --tuning-estimators 50 --n-estimators 500
```

## San Juan negative-binomial experiment

The SJ target is an overdispersed case count, so a regularized NB2 regression
can be tested as a small addition to Extra Trees. The experiment selects its
regularization, dispersion scale, and blend weight on six expanding annual
folds. A zero NB weight is included as an automatic Extra Trees fallback.

```bash
../../.venv/bin/python src/train_sj_nb_ensemble.py --n-estimators 500
```

This produces `sj_nb_search_results.csv`, `sj_nb_validation_predictions.csv`,
`sj_nb_selected_config.json`, and the separate candidate submission
`submission_tree_sj_nb.csv`. It does not replace the confirmed submission.

## San Juan EGARCH-X experiment

ARCH/GARCH models changing forecast-error variance rather than the nonlinear
case-count mean. The EGARCH-X experiment therefore keeps Extra Trees as the
mean model and tests whether temperature, rainfall, humidity, and past shocks
identify high-risk residual regimes that justify a point-prediction correction.

```bash
../../.venv/bin/python src/train_sj_garchx_ensemble.py --n-estimators 500
```

The first two annual folds supply residual history. The final four folds select
between the unchanged tree, a historical-bias correction, and climate-aware
volatility corrections. Outputs use the `sj_garchx_` prefix and the confirmed
submission is never overwritten.

## CatBoost and XGBoost experiment

The boosting experiment compares shallow CatBoost and XGBoost models using MAE
and Poisson objectives. Every candidate is tested alone and as a city-specific
blend with the confirmed tree. Four chronological annual folds select the
configuration; six folds provide the final report. Outbreak and prediction-
amplitude guards keep over-smoothed candidates from being selected.

```bash
../../.venv/bin/python src/train_boosting_ensemble.py \
  --iterations 500 --n-estimators 500
```

Results are written to `boosting_search_results.csv`,
`boosting_validation_scores.csv`, `boosting_validation_predictions.csv`, and
the separate `submission_tree_boosting.csv` candidate.

## Climate feature-engineering experiment

This experiment retains the confirmed forests and searches leakage-safe feature
groups: temperature suitability, accumulated rain, warm/wet interactions,
vapor-pressure deficit, past-only seasonal anomalies, weather changes, rolling
variability, and exponential weather memory. A compact biological feature set
also tests whether removing redundant climate lags helps.

```bash
../../.venv/bin/python src/train_feature_engineering.py \
  --search-estimators 200 --n-estimators 500
```

Selection combines six-fold MAE, recent four-fold MAE, fold stability, outbreak
MAE, and prediction-amplitude guards. The candidate is written separately as
`submission_tree_features.csv`.

## San Juan recency ensemble

This experiment matches SJ's 260-week competition horizon at six expanding
origins. It compares the confirmed full-history Extra Trees model with 7-, 10-,
and 12-year windows, exponential half-lives of 3–12 years, and conservative
full-history/recency blends. Candidate selection uses SJ MAE only; city weights
are used only when reporting an estimated complete-submission score.

```bash
../../.venv/bin/python src/train_sj_recency_ensemble.py \
  --search-estimators 300 --n-estimators 500
```

The separate candidate is `submission_tree_sj_recency.csv`.

## San Juan quantile Extra Trees

The quantile experiment keeps the confirmed Extra Trees estimator but replaces
part of its across-tree mean with a tuned tree-prediction quantile. This better
matches the competition's MAE objective while preserving the original model as
a zero-weight fallback. Selection must improve both annual and 260-week
validation and pass recent-outbreak guards.

```bash
../../.venv/bin/python src/train_sj_quantile_forest.py --n-estimators 500
```

The separate output is `submission_tree_sj_quantile.csv`.

## Nested SJ validation

The nested evaluator prevents each 260-week outer pseudo-test from influencing
its own model selection. For every outer cutoff, it selects among the original
tree, fixed recency variants, and quantile variants using only fully completed
earlier 260-week blocks. It then evaluates that frozen selection on the outer
period. Candidate selection is based only on SJ; competition city weights are
used only for final score reporting.

```bash
../../.venv/bin/python src/nested_validate_sj.py --n-estimators 500
```

Review `sj_nested_outer_results.csv` before considering the separate
`submission_tree_sj_nested.csv` candidate.

## Nested smooth-transition forest

The STAR-inspired SJ experiment trains normal- and outbreak-regime Extra Trees
experts. A climate-only Extra Trees classifier provides a smooth transition
probability using temperature, humidity, precipitation, seasonality, and their
lags. No previous case labels are required during the forecast horizon. The
same nested 260-week outer acceptance rule determines whether it may replace
the original SJ tree. Inner selection optimizes the competition's overall MAE;
outbreak MAE is reported separately and enforced as an aggregate outer-window
safety check.

```bash
../../.venv/bin/python src/nested_validate_sj_star.py --n-estimators 500
```

The current 500-tree run selected `star__q75_dw25__w1`: it improved three of
four recent outer windows and reduced their mean SJ MAE from 31.21 to 30.02.
Its separate output is `submission_tree_sj_star.csv`; detailed decisions are in
`sj_star_outer_results.csv`, `sj_star_final_search.csv`, and
`sj_star_summary.json`.

### Self-contained feature-specific lag MLP

The canonical implementation is the readable package in
[`src/feature_lag_mlp/`](src/feature_lag_mlp/). The feature-specific lag windows,
preprocessing, model, training, temporal validation, and search space are split
by responsibility; the two top-level scripts are only command-line entry
points.

Start with the complete
[`docs/feature_lag_mlp_workflow.md`](docs/feature_lag_mlp_workflow.md) guide. It
maps each pipeline stage to code, explains the data patterns and historical
alignment/normalization choices, and documents the validation and error
analysis used for tuning. `src/visualize_feature_lag_mlp.py` turns the same raw
data and confirmation tables into four presentation-ready diagnostic figures.

For the chronological reasoning from the 22.5 tree baseline, through the
feature-specific lag search, to the final 16.6 multiscale representation, read
[`docs/from_trees_and_lags_to_multiscale_mlp.md`](docs/from_trees_and_lags_to_multiscale_mlp.md).

Train the confirmed improved configuration:

```bash
.venv-tf/bin/python src/train_feature_lag_mlp.py --profile improved --seed 42
```

Train the reference configuration for comparison:

```bash
.venv-tf/bin/python src/train_feature_lag_mlp.py --profile reference --seed 42
```

Validate preprocessing and matrix shapes without loading TensorFlow:

```bash
.venv-tf/bin/python src/train_feature_lag_mlp.py --prepare-only
```

The profile preserves the feature representation and city-specific
hidden widths. It changes only San Juan dropout from `0.5, 0.5` to `0.3, 0.7`;
Iquitos remains at `0.5, 0.5`.

Tune the same three-Dense-layer model using expanding temporal folds:

```bash
.venv-tf/bin/python src/tune_feature_lag_mlp.py
```

The tuner searches city-specific hidden widths, two dropout rates, and learning
rate. It confirms the leading candidates over three annual folds and three
seeds, always including the 19.1 configuration as the control. Final prediction
now defaults to seed 42: it scored **18.8 MAE**, while averaging seeds 17, 42,
and 73 scored 19.5. The ensemble can still be reproduced explicitly with
`--final-seeds 17,42,73`.

The completed search selected a narrower second hidden layer for both cities:
`100 → 25` for San Juan and `70 → 18` for Iquitos. Competition-weighted
temporal validation improved from 16.827 for the 19.1 control to 15.862. See
`artifacts/feature_lag_mlp_tuning_report.md` for the local and leaderboard
comparisons.

A separate controlled robustness screen tested feature-block climate-noise
augmentation and a regularized linear skip branch. Neither beat the unchanged
18.8 model on temporal MAE, so both remain optional experiments. Run it with
`src/experiment_feature_lag_mlp_robustness.py`; see
`artifacts/feature_lag_mlp_robustness_report.md` for the results.

The stronger representation experiment compresses each climate series into
causal multiscale levels, variability, trends, and cyclic seasonality. It
improves the three-fold, three-seed weighted local MAE from 15.862 to 13.924.
The SJ-only file `submission_feature_lag_mlp_multiscale_summaries_seed42_sj_only.csv`
scored **16.6 hidden-test MAE**, improving the previous 18.8 champion by 2.2.

Retuning the MLP for the compressed 180-value input selected a narrower
`48 → 12` network, but its confirmed local gain is only 0.224 MAE and its
outbreak error is slightly worse. The resulting
`submission_feature_lag_mlp_multiscale_architecture_seed42_sj_only.csv` scored
**18.1 MAE**, confirming that the apparent local gain was validation noise.
Keep the original `100 → 25` architecture for this representation. The
both-city multiscale submission also scored **16.6**, so replacing IQ is
approximately neutral at leaderboard precision. Run the controlled screen with
`src/tune_multiscale_feature_lag_mlp.py`; see
`artifacts/feature_lag_mlp_representation_report.md` for the evidence.

## Tests

```bash
../../.venv/bin/python -m unittest discover -s tests -v
```

The pipeline validates input keys, chronology, past-only feature construction,
model types, and final submission integrity.
