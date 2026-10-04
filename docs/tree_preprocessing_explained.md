# Why the tree preprocessing worked

## Goal

The initial problem was that one weekly climate row does not describe dengue
risk adequately. Mosquito development and disease transmission depend on
environmental conditions accumulated over previous weeks.

The tree pipeline therefore transformed each week into a causal description of
current conditions, recent history, season, and climate interactions.

## Feature construction

`src/train_tree_ensemble.py::make_features` creates:

- current numeric weather and vegetation measurements;
- four Fourier harmonics of week of year;
- NDVI mean and spatial standard deviation;
- station and reanalysis temperature ranges;
- station temperature × relative humidity;
- climate values 1, 2, 4, 8, 12, and 16 weeks ago;
- past climate means over 2, 4, 8, 12, and 16 weeks.

For a climate series `x`:

```text
lag_k(t) = x(t-k)

past_mean_w(t) = mean(x(t-1), x(t-2), ..., x(t-w))
```

The rolling mean is shifted by one row before calculation. It therefore uses
only past observations and cannot leak the current week into its own historical
summary.

Previous case-count lags are deliberately excluded. During the competition
test period, the true previous test labels are not available, so using them in
training would create a feature that cannot be reproduced safely at inference.

## Why these features are appropriate

### Delayed climate response

Rainfall can create breeding sites, temperature affects mosquito and viral
development, and humidity affects survival. Their effects can appear several
weeks later. The sparse lags give the model several possible delay scales.

### Accumulated exposure

A single rainy week and a persistently wet month are different conditions.
Trailing means describe sustained exposure instead of only isolated readings.

### Nonlinear interactions

Risk does not necessarily increase linearly with temperature or rainfall. A
tree can learn threshold rules such as:

```text
IF humidity is high
AND recent rainfall is high
AND temperature is in a suitable range
THEN predicted cases increase
```

### Continuous seasonality

Raw week numbers have an artificial discontinuity between week 52 and week 1.
Sine and cosine features place those weeks next to each other on a seasonal
circle and let the model learn several harmonics of the annual pattern.

### Separate city behavior

SJ has more observations and larger outbreaks; IQ is smaller and more
zero-inflated. The final tree models were therefore different:

```text
SJ: ExtraTreesRegressor, Poisson split criterion
IQ: RandomForestRegressor trained through log1p(total_cases)
```

Both use 500 trees and at least five samples per leaf. The leaf constraint and
tree averaging provide useful regularization for the small datasets.

## What the 22.5 result proved

The tree system achieved **22.5 hidden-test MAE**. It established that:

- climate history improves over relying on current measurements;
- nonlinear relationships matter;
- seasonality must be represented explicitly;
- SJ and IQ should be modeled separately;
- a small set of past-only features can generalize to the hidden period.

## Why the tree system was still limited

The history was sampled only at fixed lags and fixed rolling windows. Important
behavior between those positions could be missed. Tree leaves also return
averages of training targets, which tends to smooth rare SJ outbreak peaks.

The next question was therefore:

> How much continuous history should we retain for each climate variable and
> each city?

That question led to the decision-tree lag-window screen described in
`lag_window_selection_explained.md`.
