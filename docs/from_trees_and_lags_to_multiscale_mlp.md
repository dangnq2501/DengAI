# From tree features to the 16.6 multiscale MLP

This document explains the reasoning path, not just the final code. The central
lesson is that the largest improvement came from changing **how climate history
is represented**, not from making the prediction model deeper.

## 1. The complete experiment story

| Stage | Representation | Predictor | Hidden-test MAE | Main lesson |
|---|---|---|---:|---|
| Earlier baseline | Earlier feature pipeline | Baseline model | 26.7 | Starting reference point |
| City-specific tree system | Current climate, sparse lags, rolling means, season and interactions | SJ Extra Trees; IQ Random Forest | 22.5 | Past climate and city-specific nonlinear models matter |
| Reproduced V10 | Feature-specific contiguous histories | Three-layer MLP | 19.3 | Different variables need different amounts of memory |
| Improved dropout control | Same raw histories | Three-layer MLP | 19.1 | Regularization matters, especially for SJ |
| Temporally tuned raw-lag model | 461 SJ / 380 IQ raw-history values | Tuned three-layer MLP, seed 42 | 18.8 | A smaller second hidden layer reduces overfitting |
| **Final multiscale model** | **180 stable history summaries** | **Same tuned MLP, seed 42** | **16.6** | **The representation was the main bottleneck** |
| Narrow multiscale model | Same 180 summaries | `48 → 12` MLP | 18.1 | The apparent local architecture gain was selection noise |

The progression can be summarized as:

```mermaid
flowchart LR
    A[Current weekly climate] -->|add past-only signals| B[Tree features<br/>22.5]
    B -->|learn feature-specific memory| C[Raw contiguous lag windows<br/>19.3 to 18.8]
    C -->|compress correlated weeks| D[Multiscale summaries<br/>16.6]
    D -->|make network narrower| E[18.1: rejected]
```

For a forecast endpoint `t`, every successful method follows the same general
idea:

```text
past climate up to t → history representation φ(t) → predictor f(φ(t)) → cases
```

Most of the final gain came from improving `φ(t)`.

## 2. Stage one: why the city-specific tree preprocessing worked

The first successful system did more than place an Extra Trees model on the raw
CSV. `src/train_tree_ensemble.py` created a causal tabular description of each
week.

### Features given to the trees

The feature builder included:

- the current numeric climate measurements;
- four annual Fourier harmonics, giving eight smooth seasonal coordinates;
- mean and standard deviation across the four NDVI locations;
- station and reanalysis temperature ranges;
- a temperature × relative-humidity interaction;
- climate values at lags 1, 2, 4, 8, 12, and 16 weeks;
- trailing climate means over 2, 4, 8, 12, and 16 weeks.

For this model, a lag is one past observation:

```text
lag 4 of rainfall at week t = rainfall at week t - 4
```

A trailing mean is a past interval:

```text
past mean 4 at week t = mean(x[t-1], x[t-2], x[t-3], x[t-4])
```

The rolling means use `shift(1)`, so the current row cannot leak into its past
summary. Previous dengue case counts are intentionally excluded because their
true test values are unavailable during the forecast horizon.

### Why this improved the baseline

The tree system matched several properties of dengue data:

1. **Delayed effects.** Rain, humidity, vegetation, and temperature affect
   mosquito development and transmission over later weeks, not only the current
   week.
2. **Nonlinear thresholds.** Extra Trees and Random Forests can learn rules such
   as “warm and humid after several wet weeks” without assuming a straight-line
   relationship.
3. **Seasonality.** Fourier features make the annual cycle continuous across
   week 52 and week 1.
4. **City differences.** SJ and IQ have different target scales, histories, and
   climate regimes, so they receive separate models.
5. **Small-data regularization.** Ensembles average many trees, while a minimum
   of five samples per leaf prevents extremely specific rules.

The final tree choices were also city-specific:

- SJ: 500-tree `ExtraTreesRegressor`, Poisson splitting;
- IQ: 500-tree `RandomForestRegressor` on `log1p(total_cases)`.

This was a strong, understandable baseline at **22.5 MAE**. Its main limitation
was that it sampled history only at a few fixed locations and averages. Forest
leaves also predict averages of observed targets, which tends to smooth rare SJ
outbreak peaks.

## 3. Stage two: what the vendor's decision-tree lag search did

There are two different uses of trees in this project:

| Tree use | Purpose | Final predictor? |
|---|---|---|
| City-specific Extra Trees / Random Forest | Predict dengue cases directly | Yes, for the 22.5 system |
| V9 `DecisionTreeRegressor` | Score candidate history lengths for one climate feature at a time | No; it was a feature-selection tool |

The V9 procedure considered each city and climate feature independently. For
every candidate history length from 3 through 51 weeks, it:

1. constructed a vector containing that feature's most recent consecutive
   values;
2. fitted a `DecisionTreeRegressor` using only that vector;
3. measured 10-fold negative MAE;
4. displayed a feature-by-window heatmap.

An important terminology detail is that a selected window of 40 does **not**
mean “use only lag 40.” It means:

```text
[x(t-39), x(t-38), ..., x(t-1), x(t)]
```

All 40 consecutive values are retained.

### Selected history lengths

The final V10 dictionaries contain the following window sizes:

| Climate feature | SJ weeks | IQ weeks |
|---|---:|---:|
| precipitation amount | 40 | 33 |
| reanalysis air temperature | 16 | 10 |
| reanalysis average temperature | 15 | 4 |
| reanalysis dew point | 39 | 6 |
| reanalysis maximum temperature | 12 | 41 |
| reanalysis minimum temperature | 21 | 40 |
| reanalysis precipitation | 30 | 3 |
| reanalysis relative humidity | 34 | 7 |
| satellite precipitation | 40 | 33 |
| reanalysis specific humidity | 14 | 26 |
| reanalysis diurnal temperature range | 21 | 34 |
| station average temperature | 41 | 40 |
| station diurnal temperature range | 40 | 26 |
| station maximum temperature | 37 | 39 |
| station minimum temperature | 26 | 25 |
| station precipitation | 32 | 10 |
| week of year | 3 | 3 |

Their sums produce **461 SJ inputs** and **380 IQ inputs**.

### Why this search was useful

It introduced supervised, city-specific memory. A decision tree could detect
nonlinear predictive value without forcing every climate variable to use the
same delay. The very different SJ and IQ selections also warned us that one
global lag rule would be too simple.

### Why it was not a perfect selection method

- Ordinary 10-fold validation was not forecasting-safe: training folds could
  contain dates later than validation dates.
- Adjacent history vectors overlap heavily, so the observations are not
  independent.
- Correlated climate variables can make several window sizes look equivalent.
- The notebook drew a heatmap but did not contain a deterministic `argmax` that
  recreated the final dictionaries; the values appear to have been selected by
  inspection.

Therefore, the selected lags contained a useful scientific idea—different
variables have different memory—but the exact value such as 39 versus 40 weeks
should not be treated as a physical constant.

## 4. Stage three: how the raw-lag MLP used those windows

The V10-style preprocessing performs the following operations:

1. merge labels with features by city, year, and week;
2. linearly interpolate missing numeric values in the confirmed file order;
3. calculate weather normalization statistics from SJ and apply them to both
   cities;
4. retain the selected recent window for each feature;
5. concatenate the windows into one long vector;
6. preserve the historical two-row climate alignment;
7. train a separate MLP for each city.

The shared-SJ normalization and two-row alignment are unusual. We retained them
because controlled leaderboard comparisons showed that changing the
normalization was worse, and the alignment is part of the reproduced behavior.
They are empirical compatibility choices, not universal recommendations.

The tuned raw-history SJ network was:

```text
461 raw history values
→ Dense(100, SELU)
→ Dropout(0.30)
→ Dense(25, SELU)
→ Dropout(0.70)
→ Dense(1)
```

The MLP improves on the forest because it sees each feature's complete selected
trajectory and can learn interactions across climate variables and weeks. The
reproduced V10 scored 19.3, controlled regularization reached 19.1, and the
temporally tuned seed-42 raw-lag model reached **18.8**.

However, 461 raw coordinates are a difficult representation for only 936 SJ
labels:

- neighboring weeks are strongly correlated;
- a one-week phase shift changes many input coordinates;
- interpolated measurement noise is preserved as if it were signal;
- the network must learn averages, variability, direction, and seasonality
  indirectly from limited data;
- the SJ `461 → 100 → 25 → 1` model has 48,751 trainable parameters.

This suggested that the next improvement should simplify the input rather than
add layers.

## 5. Stage four: the final multiscale representation

The final version retains the useful principle of climate memory but removes
the brittle requirement that the MLP interpret every individual week.

For each of the 16 weather variables, it examines the available 52-week history
and computes 11 values:

1. current normalized level;
2. 2-week mean;
3. 4-week mean;
4. 8-week mean;
5. 13-week mean;
6. 26-week mean;
7. 52-week mean;
8. 4-week standard deviation;
9. 13-week standard deviation;
10. 4-week mean minus 13-week mean;
11. change from 13 weeks ago divided by 12.

It then appends annual and semiannual sine/cosine features:

```text
16 weather variables × 11 summaries + 4 seasonal features = 180 inputs
```

For rainfall, for example, the representation tells the network:

```text
current rain
short, monthly, quarterly, half-year, and annual rain levels
recent rain volatility
whether the last month is wetter than the last quarter
whether rainfall is rising or falling
```

The final model keeps the confirmed preprocessing, historical alignment,
architecture, dropout, optimizer, training schedule, and seed. Only the input
representation changes.

### Why the summaries work better

1. **They preserve multiple delay scales.** We do not guess one perfect lag;
   the network receives short-, medium-, and long-term climate state.
2. **They reduce redundancy.** SJ drops from 461 to 180 inputs.
3. **They provide the statistics the MLP previously had to discover.** Level,
   variability, momentum, trend, and season are explicit.
4. **They are more shift-tolerant.** Moving a wet episode by one week changes a
   rolling mean gradually instead of moving information to a completely
   different coordinate.
5. **They regularize the model without destroying outbreak amplitude.** The SJ
   parameter count falls from 48,751 to 20,651, while its test prediction
   maximum rises from 174 to 200 rather than being excessively smoothed.
6. **They encode a better inductive bias.** Dengue responds to accumulated and
   changing environmental conditions, not to an arbitrary identity such as
   “the value exactly 37 weeks ago.”

The ablation result is especially important:

| SJ representation | Inputs | Two-fold mean MAE |
|---|---:|---:|
| Raw selected histories | 461 | 17.856 |
| **Multiscale summaries** | **180** | **12.606** |
| Raw histories + summaries | 641 | 18.433 |

If the summaries helped merely by adding more information, raw plus summaries
would have been strongest. It was the worst. The improvement came from
**discarding redundant coordinates and exposing stable temporal structure**.

Three-fold, three-seed confirmation agreed:

| City | Raw histories | Multiscale summaries |
|---|---:|---:|
| SJ mean MAE | 20.635 | **17.726** |
| IQ mean MAE | 7.908 | **7.588** |
| Competition-weighted mean | 15.862 | **13.924** |

Finally, the leaderboard confirmed the most important result:

- tuned raw-lag MLP: **18.8**;
- multiscale MLP, SJ changed only: **16.6**;
- multiscale MLP, both cities changed: **16.6**.

Because the SJ-only and both-city submissions tie at displayed precision, the
large improvement comes from SJ; the IQ representation change is approximately
neutral.

## 6. Why we rejected the narrower final network

After reducing the input from 461 to 180 values, a local architecture search
preferred a smaller `48 → 12` network by only 0.224 temporal-validation MAE.
Its outbreak MAE was worse, which was already a warning.

The hidden leaderboard resolved the uncertainty:

```text
180 summaries + 100 → 25 MLP = 16.6
180 summaries +  48 → 12 MLP = 18.1
```

The representation survived hidden testing; the small architecture difference
did not. This is why the final decision is to freeze the `100 → 25` SJ model and
continue improving only the representation through controlled ablations.

## 7. What each stage taught us

The tree system was not a failed approach. It established that causal climate
history, nonlinear effects, seasonality, and city-specific models were useful.

The decision-tree window screen was also not the final answer. It established
that climate variables have different memory, but its exact selected lengths
were noisy and high-dimensional.

The raw-lag MLP showed that combining entire trajectories was stronger than
tree leaves, but made the limited neural network learn temporal statistics from
hundreds of correlated coordinates.

The multiscale model kept the useful memory and removed the fragile details. It
gave the MLP a compact description of **level, duration, variability, direction,
and season**, which is why the same basic MLP improved from 18.8 to 16.6.

## 8. Code map

| Question | Implementation |
|---|---|
| How did the 22.5 tree system create lags and rolling features? | `src/train_tree_ensemble.py::make_features` |
| Where are the manually selected V10 window lengths? | `src/feature_lag_mlp/config.py::LAG_WINDOWS` |
| How are raw history vectors concatenated? | `src/feature_lag_mlp/features.py::_flatten_history_rows` |
| How is the three-layer MLP defined? | `src/feature_lag_mlp/model.py::build_model` |
| How are the 180 multiscale inputs calculated? | `src/feature_lag_mlp/representations.py::_multiscale_summary_rows` |
| How were representations compared? | `src/feature_lag_mlp/representation_experiment.py` |
| How is the 16.6 submission reproduced? | `src/train_feature_lag_mlp_representation.py` |

## 9. Short presentation version

> We first built city-specific tree ensembles with seasonal terms, sparse
> climate lags, rolling means, and nonlinear interactions. This reduced MAE to
> 22.5 and proved that climate memory mattered. The vendor workflow then used a
> decision tree as a feature-screening tool to choose a separate contiguous
> history length for every feature and city. Feeding those 461 SJ and 380 IQ
> raw history values to an MLP improved the score to 19.3, and controlled model
> tuning reached 18.8. Error analysis showed that hundreds of adjacent weekly
> values were redundant and forced a small network to learn temporal statistics
> from very little data. We therefore replaced the raw coordinates with 180
> causal multiscale features describing current level, means over 2–52 weeks,
> variability, short-versus-medium change, trend, and seasonality. Keeping the
> same MLP, training process, and seed improved hidden MAE to 16.6. A later
> architecture change worsened MAE to 18.1, confirming that representation—not
> model complexity—produced the main gain.
