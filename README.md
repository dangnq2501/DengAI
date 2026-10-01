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

## Tests

```bash
../../.venv/bin/python -m unittest discover -s tests -v
```

The pipeline validates input keys, chronology, past-only feature construction,
model types, and final submission integrity.
