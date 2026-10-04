# DengAI study notes

Read these documents in order:

1. [`slide_flow_and_figures.md`](slide_flow_and_figures.md) — the recommended
   eight-slide narrative, plot selection, speaker transitions, and appendix.
2. [`tree_preprocessing_explained.md`](tree_preprocessing_explained.md) — why
   causal lags, rolling climate summaries, nonlinear trees, and separate city
   models produced the 22.5 baseline.
3. [`lag_window_selection_explained.md`](lag_window_selection_explained.md) —
   how the vendor used a decision tree to screen history lengths and how those
   windows became the 461/380-value MLP inputs.
4. [`multiscale_16_6_model_explained.md`](multiscale_16_6_model_explained.md) —
   why replacing raw weeks with 180 stable summaries improved hidden MAE from
   18.8 to 16.6.
5. [`from_trees_and_lags_to_multiscale_mlp.md`](from_trees_and_lags_to_multiscale_mlp.md)
   — the complete chronological story in one document, including a short
   presentation script.
6. [`feature_lag_mlp_workflow.md`](feature_lag_mlp_workflow.md) — detailed code,
   validation, reproduction commands, and experiment results.

## Result timeline

| Step | Hidden-test MAE |
|---|---:|
| Earlier baseline | 26.7 |
| City-specific tree ensemble | 22.5 |
| Reproduced raw-history V10 MLP | 19.3 |
| Improved dropout configuration | 19.1 |
| Tuned raw-history MLP, seed 42 | 18.8 |
| **Multiscale-summary MLP** | **16.6** |
| Narrow multiscale architecture | 18.1 — rejected |

The main conclusion is that representation produced the largest reliable gain.
The final model keeps the confirmed MLP and replaces hundreds of correlated
weekly coordinates with causal multiscale descriptions of level, variability,
change, trend, and seasonality.
