# DengAI dengue forecasting

This repository contains the confirmed city-specific tree solution for the
DengAI competition. It achieved hidden-test MAE **22.5**.

## Model

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

## Vendor 12.67 climate-history neural network

The repository under `vendors/dengai-predicting-disease-spread` records a real
12.6779 leaderboard score. Its final model is a dense SELU network over
city-specific portions of a 52-week climate history, rather than an LSTM. The
reproduction preserves the vendor's variable-specific windows, MAE loss,
dropout, RMSprop schedule, SJ-based IQ scaling, and implicit two-week output
offset. A separate enhanced candidate appends this project's seasonal,
biological, anomaly, and weather-dynamics features with corrected alignment.

TensorFlow is isolated from the main Python 3.14 environment:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-tf
.venv-tf/bin/python -m pip install -r requirements-tf.txt
.venv-tf/bin/python src/train_vendor_history_tf.py --seed 42
```

The run creates six competition files without replacing the confirmed tree:

- `submission_tf_vendor_shift2.csv`: closest TensorFlow reproduction.
- `submission_tf_enhanced_aligned.csv`: full engineered-feature candidate.
- `submission_tf_vendor_enhanced_w025.csv`: conservative 25% enhancement.
- `submission_tf_vendor_enhanced_w05.csv`: equal-weight blend.
- `submission_tf_city_hybrid.csv`: vendor IQ plus fully enhanced SJ.
- `submission_tf_city_hybrid_w025.csv`: vendor IQ plus 25% enhanced SJ.

`train_vendor_history_nn.py` provides a dependency-free NumPy implementation
and chronological comparison of shifted/aligned and vendor/enhanced variants.
The enhanced model wins local recent-fold validation, but its substantially
lower prediction amplitude makes it experimental until leaderboard-tested.

### Self-contained feature-specific lag MLP

The canonical implementation is the readable package in
[`src/feature_lag_mlp/`](src/feature_lag_mlp/). It does not require the local
`vendors/` directory or saved model weights. The feature-specific lag windows,
preprocessing, model, training, temporal validation, and search space are split
by responsibility; the two top-level scripts are only command-line entry
points.

Start with the complete
[`docs/feature_lag_mlp_workflow.md`](docs/feature_lag_mlp_workflow.md) guide. It
maps each pipeline stage to code, explains the data patterns and historical
alignment/normalization choices, and documents the validation and error
analysis used for tuning. `src/visualize_feature_lag_mlp.py` turns the same raw
data and confirmation tables into four presentation-ready diagnostic figures.

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

The improved profile preserves the feature representation and city-specific
hidden widths. It changes only San Juan dropout from `0.5, 0.5` to `0.3, 0.7`;
Iquitos remains at `0.5, 0.5`. The local `vendors/` directory is ignored by
Git and is not needed for this workflow.

Tune the same three-Dense-layer model using expanding temporal folds:

```bash
.venv-tf/bin/python src/tune_feature_lag_mlp.py
```

The tuner searches city-specific hidden widths, two dropout rates, and learning
rate. It confirms the leading candidates over three annual folds and three
seeds, always including the 19.1 configuration as the control. Results use the
`feature_lag_mlp_tuning_` prefix, and the final seed-ensemble candidate is
`submission_feature_lag_mlp_tuned.csv`.

The completed search selected a narrower second hidden layer for both cities:
`100 → 25` for San Juan and `70 → 18` for Iquitos. Competition-weighted
temporal validation improved from 16.827 for the 19.1 control to 15.862. See
`artifacts/feature_lag_mlp_tuning_report.md`; hidden-test improvement still
requires leaderboard evaluation.

### Historical end-to-end reproduction

Retrain the complete V10 process from raw competition data without loading the
vendor's saved model weights:

```bash
.venv-tf/bin/python src/train_vendor_v10_reproduction.py --seed 42
```

The script preserves the original raw-file interpolation, 53-row duplicated
history prefix, SJ-derived scaling for both cities, variable-specific history
trimming, 461/380-dimensional city inputs, SELU/dropout networks, RMSprop
settings, 40 epochs of 200 batches, city-specific shuffling and learning-rate
callbacks, and the original two-row output offset. It writes
`submission_vendor_v10_retrained.csv`, both newly trained `.keras` models,
per-epoch histories, and `vendor_v10_retrained.json`. No H5 weights are read.

### Saved-weight audit only

The vendor did not commit the final 12.67 weights, but it did commit the V10
models associated with its earlier 17-MAE run (the accompanying screenshots
show later scores near 15). Reproduce those models without retraining:

```bash
.venv-tf/bin/python src/reproduce_vendor_17_57.py
```

This command traces the artifacts to commit `be845a3`, loads
`sj_model_17.57MAP.h5` and `iq_model_17.57MAP.h5`, and exactly preserves the V10
461/380-column inputs, raw-file interpolation order, duplicated history rows,
SJ-derived scaling, and two-row submission offset. It writes
`submission_vendor_saved_17_57.csv` and a checksum/shape audit in
`vendor_saved_17_57.json`.

## Tests

```bash
../../.venv/bin/python -m unittest discover -s tests -v
```

The pipeline validates input keys, chronology, past-only feature construction,
model types, and final submission integrity.
