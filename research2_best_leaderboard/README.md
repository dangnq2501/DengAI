# Feature-lag MLP ensemble experiment

Run from `DengAI` in the requested conda environment:

```bash
conda run -n bkk python research2/run_experiment.py
conda run -n bkk python -m unittest discover -s research2 -p 'test_*.py' -v
```

This experiment reuses the canonical feature matrices, two-week output offset,
city normalization, improved/tuned architectures, MAE objective, dropout and
40 × 200 optimization schedule. It tests three seeds (42, 137, 2026), mean and
median ensembles, averages of epochs 30/35/40, and architecture mixtures.
Predictions are aggregated before nonnegative integer truncation.

`bkk` uses Python 3.14 and already provides PyTorch. TensorFlow could not be
installed within available disk space. `torch_backend.py` is therefore an
explicit approximation: its random initialization, full-permutation shuffle,
RMSprop epsilon semantics and training metric aggregation differ from Keras.
The control is the improved architecture with seed 42 and epoch 40 **under this
same backend**, not an exact reproduction of the existing leaderboard model.
The original `src/feature_lag_mlp` implementation and submissions are untouched.

Each city's final labeled block (260 SJ weeks, 156 IQ weeks) is withheld before
selection. Three earlier expanding 52-week folds select the lowest average-MAE
recipe independently per city. Only the selected recipe and control are scored
on the long-horizon audit. That result does not change the selected recipe.
Finally both are retrained on all labels to produce separate submissions.
The audit is held out from this search; the inherited architectures may already
have been influenced by those years in earlier work. It is not a pristine
estimate of the entire historical research process or a leaderboard score.

The inherited preprocessing interpolates known covariates, including within
prediction blocks, and preserves historical circular training prefixes. It is
appropriate to compare this competition pipeline, not to claim a strictly
causal operational weather forecast. No audit or test labels enter training
matrices, normalization or ensemble selection.

Outputs in `artifacts/`:

- `manifest.json`: data/source hashes, library versions and run settings.
- `development_scores.csv`, `development_predictions.csv`, `ranking_*.csv`:
  complete selection evidence.
- `selection.json`: frozen per-city choices.
- `audit_scores.csv`, `audit_predictions.csv`, `report.json`: held-out results.
- `submission_candidate.csv`, `submission_control.csv`: 416 competition rows.
- Per-run cached raw predictions and training histories; final `.pt` weights.

Runs resume from completed fits. Changing settings, source or input data requires
a fresh `--output-dir`. A quick integration check can use `--epochs 2 --steps 2
--output-dir research2/smoke`; those scores are not useful for model selection.

## Completed experiment

All 60 fits completed (two architectures × three seeds × five periods × two
cities). Selection chose `tuned_median` for SJ and `tuned_snapshots` for IQ.

| Evaluation | City | Single-model control MAE | Selected ensemble MAE |
| --- | --- | ---: | ---: |
| Development, three annual folds | SJ | 34.737 | 30.827 |
| Development, three annual folds | IQ | 7.917 | 7.391 |
| Held-out 260 weeks | SJ | 27.008 | 27.142 |
| Held-out 156 weeks | IQ | 8.660 | 8.064 |
| Held-out, competition row weights | Both | **20.127** | **19.988** |

The weighted audit improvement is **0.69%**. It comes from IQ; SJ slightly
worsens. This is a candidate for leaderboard testing, not evidence to replace
the confirmed best model. A descriptive paired 13-week circular-block bootstrap
(5,000 replicates, seed 42) gives a 95% interval of **[-0.793, +0.539]** for
candidate-minus-control MAE. The interval includes no improvement, so the small
observed gain is uncertain. Run `python research2/check_results.py` to reproduce
this calculation and check submission keys, count, types and nonnegativity.

The candidate's test predictions average 33.73 SJ cases (maximum 176) and 4.90
IQ cases (maximum 16). Both submission files have exactly 416 rows in template
order. No leaderboard submission was made.

The successful training process started in `bkk` with Python 3.14.3, PyTorch
2.10.0, NumPy 2.4.2 and pandas 3.0.1. Another process changed `bkk` to Python 3.12
while that already-loaded process continued. To rerun in the updated environment,
install `research2/requirements.txt` there first; dependencies were not changed
again while the external installation was running. Three unit tests passed using
the original conda interpreter and retained `bkk` libraries. Saved `.pt` files
contain epoch-40 state dictionaries; checkpoint-ensemble test outputs are saved
in `.npz` files, and regenerating checkpoint models requires rerunning training.
