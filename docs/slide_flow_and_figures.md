# Slide flow and figure guide

## Recommended core deck: eight slides

The presentation should follow the reasoning process, not the order in which
scripts were written. Each slide should answer one question and end with the
reason for the next experiment.

### Slide 1 — Problem and objective

**Title:** Predict weekly dengue cases from climate history

Include:

- two cities: San Juan and Iquitos;
- competition metric: mean absolute error;
- training sizes: 936 SJ weeks and 520 IQ weeks;
- test sizes: 260 SJ weeks and 156 IQ weeks;
- goal: represent delayed climate effects without using unavailable case lags.

Use no plot or use a small crop from
`artifacts/slide_figures/01_target_scale_and_outbreaks.png`.

Speaker transition:

> Before choosing a model, we examined how different the targets are across
> cities and time.

### Slide 2 — Data pattern and forecasting difficulty

**Title:** SJ has rare high outbreaks; IQ is smaller and more zero-inflated

Use:

```text
artifacts/slide_figures/01_target_scale_and_outbreaks.png
```

Explain:

- SJ has much larger peaks, so smoothing can cause large absolute errors;
- target distributions are skewed, so mean behavior does not describe outbreak
  behavior;
- separate city models are justified.

Source notes: `docs/tree_preprocessing_explained.md`.

### Slide 3 — Seasonality is useful but insufficient

**Title:** Dengue is seasonal, but week of year does not determine the outcome

Use:

```text
artifacts/slide_figures/02_seasonal_case_pattern.png
```

Explain:

- the mean and median change over the year;
- the interquartile band remains wide;
- therefore the model needs both calendar phase and recent climate conditions.

Speaker transition:

> We converted each week into a causal description of season, current climate,
> and previous climate.

### Slide 4 — Tree baseline and first useful representation

**Title:** Past-only lags and rolling climate means reduced MAE to 22.5

Show this compact pipeline rather than a dense plot:

```text
current climate
+ season harmonics
+ lags 1/2/4/8/12/16
+ rolling means 2/4/8/12/16
+ climate interactions
→ SJ Extra Trees / IQ Random Forest
→ 22.5 MAE
```

Explain why it worked:

- trees model nonlinear thresholds;
- lagged features represent delayed effects;
- rolling means represent accumulated exposure;
- separate cities require separate models.

Source notes: `docs/tree_preprocessing_explained.md`.

### Slide 5 — Learning how much history to retain

**Title:** Climate memory differs by feature and city

Use:

```text
artifacts/slide_figures/03_temporal_memory_patterns.png
```

Explain:

- the left panel shows that case counts are temporally dependent;
- the right panel shows different selected history lengths;
- a 40-week window contains all 40 consecutive values, not only lag 40;
- the decision tree was a window-screening tool, not the final predictor.

Mention the limitation: the original 10-fold procedure was not fully
forecasting-safe, so exact values such as 39 versus 40 should not be interpreted
as biological constants.

Source notes: `docs/lag_window_selection_explained.md`.

### Slide 6 — Raw-history MLP and error analysis

**Title:** Raw histories improved MAE to 18.8 but created redundant inputs

Show:

```text
SJ: 461 inputs → Dense 100 → Dense 25 → prediction
IQ: 380 inputs → Dense 70  → Dense 18 → prediction
```

Then use:

```text
artifacts/slide_figures/04_representation_ablation.png
```

Explain:

- adjacent weekly coordinates are highly correlated;
- SJ had 461 inputs for only 936 labels;
- raw plus summaries was worse, proving that adding features was not enough;
- this motivated replacing raw coordinates rather than deepening the model.

### Slide 7 — Final multiscale representation

**Title:** Describe climate state at several timescales instead of copying weeks

Show the formula:

```text
For each of 16 weather variables:
current
+ means over 2/4/8/13/26/52 weeks
+ standard deviations over 4/13 weeks
+ 4-week mean − 13-week mean
+ 13-week trend

16 × 11 + 4 season features = 180 inputs
```

Optionally use:

```text
artifacts/slide_figures/05_sj_prediction_profile.png
```

The prediction plot is supporting evidence, not an accuracy plot: hidden target
values are unavailable. Use it to show that the new representation changes the
outbreak shape and preserves amplitude rather than merely smoothing forecasts.

Source notes: `docs/multiscale_16_6_model_explained.md`.

### Slide 8 — Results and conclusion

**Title:** Representation—not model complexity—produced the largest gain

Use:

```text
artifacts/slide_figures/06_leaderboard_journey.png
```

Key conclusion:

```text
Tree ensemble                 22.5
Raw-history MLP               19.3
Tuned raw-history MLP         18.8
Multiscale-summary MLP        16.6
Narrow multiscale MLP         18.1 — rejected
```

End with:

> The successful model kept the same small MLP and improved the description of
> climate history: level, duration, variability, direction, and seasonality.

## Optional slide 9 — Honest model-selection lesson

**Title:** Local validation is evidence, not proof

Use:

```text
artifacts/slide_figures/07_local_validation_trap.png
```

The `48 → 12` network appeared 0.224 MAE better locally but was 1.5 MAE worse
on the hidden leaderboard. This is a useful error-analysis result: a small win
after searching many candidates can be selection noise, particularly when its
outbreak error is worse.

## Optional appendix slides

Use appendix slides for details that interrupt the main story:

- complete SJ/IQ lag-window table from
  `docs/lag_window_selection_explained.md`;
- shared-SJ normalization and the two-row historical alignment from
  `docs/feature_lag_mlp_workflow.md`;
- validation protocol and outbreak MAE;
- reproduction commands and code map;
- full target statistics from `artifacts/slide_statistics.csv`.

## Which Markdown document to use

| Need | Document |
|---|---|
| Build the slide order | `docs/slide_flow_and_figures.md` |
| Prepare speaker notes for the complete story | `docs/from_trees_and_lags_to_multiscale_mlp.md` |
| Explain the 22.5 tree model | `docs/tree_preprocessing_explained.md` |
| Explain what a selected lag window means | `docs/lag_window_selection_explained.md` |
| Explain the final 180 features and 16.6 result | `docs/multiscale_16_6_model_explained.md` |
| Answer implementation or reproduction questions | `docs/feature_lag_mlp_workflow.md` |

Do not paste whole Markdown sections onto slides. Use one figure, one conclusion,
and at most three short supporting statements per slide. Keep equations and
complete tables in the appendix.

## Reproduce all figures

From the repository root:

```bash
MPLCONFIGDIR=/tmp/dengai-matplotlib \
  ../../.venv/bin/python src/plot_experiment_story.py
```

This writes seven PNG files to `artifacts/slide_figures/` and a city-level
statistics table to `artifacts/slide_statistics.csv`.
