# DengAI leaderboard notebook

Open **[dengai_leaderboard.ipynb](dengai_leaderboard.ipynb)**. It contains the full
pipeline and follows `../project_guidelines.md`: problem motivation, pandas
loading and X/y separation, explained preprocessing, Seaborn exploration,
temporal validation, hyperparameter search, model comparison, error analysis,
conclusions, and presentation notes. Notebook cells include slide metadata.

The notebook aims for a competitive submission; no hidden-test score or rank
is guaranteed. The 22.5 score mentioned in the notebook is the existing parent
README's reported score, not a result obtained by this notebook.

## Executed results

The full notebook executed successfully with all 20 candidates per city.
The frozen selection chose the seasonal median for San Juan and LightGBM
absolute-error regression (7 leaves, minimum 50 rows per leaf) for Iquitos.

| Historical audit | Pooled MAE |
|---|---:|
| City-median baseline | 14.108 |
| Development-selected submission | 13.089 |
| Rebuilt forest reference | 12.911 |

The selected models improved over the simple baseline but **did not beat the
forest reference on the audit**. San Juan outbreak MAE was also materially
worse for the selected seasonal model. This result is preserved, not used to
change the frozen selection. Both submission files are provided; neither has
been uploaded or assigned a verified leaderboard score. These local scores
are not comparable to the parent README's reported hidden-test score.

## Run

Use Python 3.11+ with the packages in `requirements.txt` installed in your
notebook kernel. From this folder:

```bash
python -m pip install -r requirements.txt
```

Open the notebook, select that Python environment, and run all cells in order.
It finds `../data` when launched from this folder, DengAI, or the workspace
root. It does not download data, submit predictions, or alter existing project
artifacts. Re-running replaces generated files under this folder's `artifacts/`.

The default run searches 20 candidates per city across five development folds.
`QUICK = True` reduces models and iterations for a smoke run, changing results.
Do not compare a quick run against the supplied full-run results. Exact package
versions, input hashes, model parameters, and the submission hash are recorded
in `artifacts/run_manifest.json` and `artifacts/requirements-lock.txt`.

## Outputs

- `artifacts/submission.csv`: selected city-specific models, final refit.
- `artifacts/submission_reference.csv`: rebuilt reference forest architecture.
- `artifacts/audit_scores.csv`: final historical audit, including pooled MAE.
- `artifacts/development_ranking.csv`: model-selection scores and fold MAEs.
- `artifacts/selected_models.json`: frozen city-specific components and weights.
- `artifacts/development_predictions.csv`, `audit_predictions.csv`: diagnostics.
- `artifacts/models.joblib`: fitted models, specifications, and feature history.
- `artifacts/figures/`: six figures for the presentation.

Upload `artifacts/submission.csv` yourself to DrivenData to obtain an actual
leaderboard score. The notebook checks all 416 keys, ordering, integer counts,
and nonnegative finite predictions before export.

## Evaluation design

The final 260 San Juan and 156 Iquitos labeled weeks are held back from this
notebook's model selection. Earlier expanding annual folds plus a multi-year
fold guide a bounded search and simple pairwise blends. Development windows
overlap and must not be treated as independent samples. All imputation is
fitted on each training prefix; weather features use current/past inputs only.
No target lags or target encodings enter the features.

The historical audit is retrospective: previous workspace experiments have
already examined these dates. It is not an untouched external test, and a
local MAE cannot be equated with the leaderboard MAE. The selection is frozen
before the audit and is not changed in response to its results.

`build_notebook.py` regenerates the source notebook, clearing saved outputs;
it is for editing/rebuilding, not needed for normal use. The notebook is
self-contained and does not import modeling code from other project folders.

Before the course submission, fill in actual team contributions and the video
presentation URL; those personal deliverables are not generated here.
