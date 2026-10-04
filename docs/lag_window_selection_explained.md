# How feature-specific lag windows were selected and used

## Two tree methods that should not be confused

| Method | Purpose |
|---|---|
| Extra Trees / Random Forest baseline | Directly predict dengue cases; scored 22.5 |
| V9 `DecisionTreeRegressor` | Screen candidate history lengths before training an MLP |

The V9 decision tree was not the final forecasting model. It was a supervised
feature-selection tool.

## The history-length experiment

For each city and climate variable, the vendor considered candidate window
lengths from 3 through 51 weeks. For each candidate `w`, it created a vector:

```text
history_w(t) = [x(t-w+1), ..., x(t-1), x(t)]
```

It then fitted a `DecisionTreeRegressor` using only this one feature's history
and recorded 10-fold negative MAE.

A window length of 40 does not mean only lag 40. It retains all 40 consecutive
values from `t-39` through `t`.

## Final window dictionaries

| Feature | SJ weeks | IQ weeks |
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

The lengths sum to:

```text
SJ input dimension = 461
IQ input dimension = 380
```

The maintained values are stored in
`src/feature_lag_mlp/config.py::LAG_WINDOWS`.

## Why the screening idea was useful

It made climate memory:

- supervised by the target rather than chosen arbitrarily;
- feature-specific rather than forcing one window on every variable;
- city-specific rather than assuming SJ and IQ share the same dynamics;
- nonlinear because a decision tree can recognize thresholds.

The large differences between SJ and IQ also demonstrated that city is more
than a categorical input: the two cities require separate forecasting systems.

## Important weaknesses

The method should not be interpreted as discovering exact biological delays.

1. Ordinary 10-fold validation was not forecasting-safe because some training
   dates could occur after validation dates.
2. Adjacent history examples overlap heavily.
3. Climate measurements are strongly correlated with one another.
4. Several neighboring window lengths can therefore receive similar scores.
5. The notebook produced a heatmap but no deterministic `argmax` that recreated
   the final dictionaries; the final values appear to have been chosen through
   inspection.

The reliable insight is:

> Climate memory matters at different scales.

The unreliable interpretation would be:

> A 39-week window is scientifically correct and a 38- or 40-week window is
> wrong.

## Raw-lag MLP preprocessing

The selected windows were then used by the V10-style pipeline:

1. merge features and labels using city/year/week keys;
2. linearly interpolate missing numeric values;
3. calculate normalization statistics from SJ and apply them to both cities;
4. keep the selected recent window for every feature;
5. concatenate every window into one vector;
6. preserve the reproduced two-row climate alignment;
7. train a separate SELU/dropout MLP for each city.

For SJ:

```text
461 inputs
→ Dense(100, SELU)
→ Dropout(0.30)
→ Dense(25, SELU)
→ Dropout(0.70)
→ Dense(1)
```

This development path produced:

| Version | Hidden MAE |
|---|---:|
| Reproduced V10 | 19.3 |
| Improved dropout configuration | 19.1 |
| Tuned raw-lag MLP, seed 42 | 18.8 |

## Why we moved beyond exact lag coordinates

The raw vector gave the MLP detailed history, but it also gave 461 highly
correlated SJ coordinates for only 936 labels. A small timing shift moved
information between coordinates, and the MLP had to learn means, variability,
and trends indirectly.

The next representation retained multiple history scales while replacing
individual weeks with stable temporal summaries. That final design is explained
in `multiscale_16_6_model_explained.md`.
