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
│   └── tree_ensemble_models.joblib
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

Expected prediction distributions with the default seed and 500 trees:

| City | Mean | Maximum |
| --- | ---: | ---: |
| Iquitos | 6.14 | 16 |
| San Juan | 32.17 | 99 |

## Tests

```bash
../../.venv/bin/python -m unittest discover -s tests -v
```

The pipeline validates input keys, chronology, past-only feature construction,
model types, and final submission integrity.
