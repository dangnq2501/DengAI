# DengAI — exact reproduction of the 15.9832 submission

This repository contains the final code, data, notebook, submission artifact, and report for our AI-7101 DengAI project. The canonical pipeline retrains the San Juan and Iquitos models and regenerates the exact 416-row CSV associated with a **15.9832 public leaderboard MAE**.

## Reproducibility claim

DrivenData does not publish the hidden test labels, so the public MAE cannot be recalculated locally. We therefore use an artifact-level acceptance test:

1. retrain both city models from the provided competition training data;
2. generate all 416 test predictions in the official row order;
3. compare every prediction with the restored scored submission; and
4. require an exact SHA-256 match.



The refactored pipeline produces **0 differing rows out of 416**. It also reproduces the reported 52-week San Juan holdout MAE of **16.519**.

## Final model

The final submission is city-specific because San Juan and Iquitos have different target distributions and respond differently to temporal representations.

| City | Representation | Model | Integer conversion |
|---|---|---|---|
| San Juan (`sj`) | 180 multiscale features: 11 summaries for each of 16 weather variables plus 4 seasonal harmonics | NumPy MLP `180 → 100 → 25 → 1`, SELU, dropout `0.3/0.7`, learning rate `0.01`, seed 42 | clip at zero, then round to nearest |
| Iquitos (`iq`) | 380 feature-specific raw lag values | NumPy MLP `380 → 70 → 18 → 1`, SELU, dropout `0.5/0.5`, learning rate `0.001`, seed 42 | clip at zero, then truncate |

Both models use MAE/L1 optimization, RMSprop-style updates, gradient clipping, 40 epochs, 200 update steps per epoch, and batch size 16.

### Preprocessing contract

- Numeric missing values are filled with the historical linear-interpolation rule used by the scored pipeline.
- The 16 weather variables are z-score normalized using the stored San Juan-based normalization procedure.
- `weekofyear` is min-max scaled.
- San Juan multiscale features summarize 2, 4, 8, 13, 26, and 52-week levels, 4/13-week variability, a short-versus-long gap, a 13-week trend, and annual/semi-annual seasonality.
- Iquitos uses a separately selected contiguous history length for each feature; the lengths sum to 380 inputs.
- The original 53-row context/alignment behavior is preserved intentionally because changing it changes the scored artifact.

PLS was investigated after this submission was created. It is **not** part of the hash-verified 15.9832 pipeline.

## Repository structure

```text
.
├── reproduce_submission.py          # canonical one-command entry point
├── src/dengai_reproduction/
│   ├── __init__.py
│   └── pipeline.py                  # preprocessing, features, MLPs, verification
├── requirements-reproduce.txt       # pinned numerical environment
├── data/                            # official DengAI train/test/template CSVs
├── artifacts/                       # scored reference used by the automated test
├── notebook/final_submission.ipynb      # clean training-to-submission notebook
├── Report/main.pdf
├── SUBMISSION_MANIFEST.txt
└── package_submission.py
```

`reproduce_submission.py` and `src/dengai_reproduction/pipeline.py` are the final implementation. `notebook/final_submission.ipynb` is the course-facing notebook: it trains both models, validates the official format, and writes a CSV ready for upload.

## Environment setup

Python and NumPy versions affect the deterministic training trajectory. Use the pinned environment exactly.

```bash
cd /path/to/DengAI
pip install -r requirements-reproduce.txt
```


## Reproduce the final submission

```bash
.venv-reproduce/bin/python reproduce_submission.py --validate-holdout
```

Generated file:

```text
notebook/outputs/submission.csv
```

The restored scored file is:

```text
artifacts/submission_feature_lag_mlp_best_candidate.csv
```

`cmp` should print nothing and both hashes should equal the expected digest.

## Generate the submission from the final notebook

The final notebook trains both city models from the competition data and creates `notebook/outputs/submission.csv`:

```bash
.venv-reproduce/bin/python -m jupyter execute \
  --inplace \
  --kernel_name=python3 \
  notebook/final_submission.ipynb
```


The archive includes the executable source, pinned requirements, test, competition data, verification artifact, executed final notebook, upload-ready `submission.csv`, final report PDF, and presentation-video link. Local environments, caches, LaTeX temporary files, reproduction-only notebooks, research branches, and unrelated exploratory files are excluded.

## Limitations

- The 15.9832 value is the observed public leaderboard score; hidden labels are not available for local evaluation.
- Exact artifact reproduction requires Python 3.12.x, NumPy 2.4.4, and pandas 3.0.2.
- The preserved context/alignment behavior is required for artifact identity but should receive an additional causal audit before claiming prospective public-health forecasting performance.
- Results cover only San Juan and Iquitos and should not be assumed to transfer to other cities.
