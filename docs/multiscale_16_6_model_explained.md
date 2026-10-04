# Why the multiscale MLP achieved 16.6 MAE

## Starting point

The best raw-history model used 461 SJ inputs and 380 IQ inputs. It achieved
**18.8 hidden-test MAE** with seed 42.

The main error-analysis finding was that individual adjacent weeks were highly
redundant. The model received detailed sequences but had to discover simple
temporal concepts—average level, variability, recent change, long-term trend,
and season—from a small number of labels.

## Representation design

For each of 16 weather variables, the final feature builder examines the
available 52-week history and calculates:

| Feature | Meaning |
|---|---|
| Current level | Most recent value at the confirmed history endpoint |
| Means over 2, 4, 8, 13, 26, 52 weeks | Climate level at multiple timescales |
| Standard deviation over 4 weeks | Recent instability |
| Standard deviation over 13 weeks | Seasonal-scale instability |
| 4-week mean − 13-week mean | Short-term condition relative to recent baseline |
| `(current − value 13 weeks ago) / 12` | Direction and rate of change |

This is 11 values per weather variable. Four cyclic calendar features are
appended:

```text
sin(annual phase), cos(annual phase)
sin(semiannual phase), cos(semiannual phase)
```

The total dimension is:

```text
16 × 11 + 4 = 180
```

The implementation is
`src/feature_lag_mlp/representations.py::_multiscale_summary_rows`.

## Rainfall example

The raw-lag representation might give the network 40 separate rainfall values:

```text
[rain(t-39), ..., rain(t)]
```

The multiscale representation instead describes the state of that sequence:

```text
current rainfall
2/4/8/13/26/52-week rainfall means
4/13-week rainfall variability
recent month minus recent quarter
13-week rainfall direction
```

This retains short-, medium-, and long-term information without requiring the
MLP to assign a separate meaning to every week.

## What remained unchanged

The experiment deliberately isolated representation. It retained:

- the same 16 weather variables;
- shared SJ normalization;
- the reproduced historical two-row alignment;
- the tuned MLP architecture;
- dropout rates;
- RMSprop and MAE loss;
- training schedule;
- seed 42;
- clipping and integer submission conversion.

For SJ, the final architecture remains:

```text
180 inputs
→ Dense(100, SELU)
→ Dropout(0.30)
→ Dense(25, SELU)
→ Dropout(0.70)
→ Dense(1)
```

Therefore the comparison primarily measures the effect of representation.

## Why it generalizes better

### Lower dimensionality

SJ falls from 461 inputs to 180. With the same `100 → 25` architecture, the
trainable parameter count falls from 48,751 to 20,651.

### Less redundant input

Neighboring weekly values are strongly correlated. Replacing them with a few
summary statistics reduces the number of ways the model can memorize a
historical outbreak.

### Better tolerance to timing shifts

If a wet episode moves one week, many raw coordinates change positions. A
4- or 13-week mean changes gradually, making it more stable across seasons.

### Explicit temporal concepts

The MLP no longer has to learn averaging, variability, momentum, and annual
seasonality from scratch. Those concepts are already present in the input.

### Preserved outbreak amplitude

Compression did not simply smooth every prediction. The full-data SJ prediction
maximum increased from 174 for the 18.8 raw-lag model to 200 for the multiscale
model. This matters because underpredicting SJ outbreaks was a recurring error.

## Ablation evidence

The two-fold SJ representation screen gave:

| Representation | Inputs | Mean MAE | Outbreak MAE |
|---|---:|---:|---:|
| Raw lag histories | 461 | 17.856 | 35.818 |
| **Multiscale summaries** | **180** | **12.606** | **21.182** |
| Raw histories + summaries | 641 | 18.433 | 39.455 |

Raw plus summaries was worse than either intended improvement. This is critical:
the 16.6 approach did not win because it added more features. It won because it
removed fragile coordinates and kept useful temporal structure.

The three-fold, three-seed confirmation was also favorable:

| City | Raw histories | Multiscale summaries |
|---|---:|---:|
| SJ mean MAE | 20.635 | **17.726** |
| IQ mean MAE | 7.908 | **7.588** |
| Competition-weighted mean | 15.862 | **13.924** |

## Hidden leaderboard evidence

| Submission | Hidden MAE | Interpretation |
|---|---:|---|
| Tuned raw-lag MLP | 18.8 | Previous champion |
| Multiscale summaries, SJ only | **16.6** | Representation improves hidden SJ |
| Multiscale summaries, both cities | **16.6** | IQ change approximately neutral |
| Multiscale summaries with `48 → 12` SJ network | 18.1 | Narrower architecture rejected |

Because only SJ changed between the 16.6 control and 18.1 narrow-network
submission, the narrower architecture worsened hidden SJ MAE by approximately:

```text
(18.1 - 16.6) / (260 / 416) ≈ 2.4
```

The local architecture advantage was selection noise. The final decision is to
freeze the `100 → 25` model and focus future experiments on representation.

## Final conclusion

The tree baseline discovered that nonlinear delayed climate information was
useful. The lag-window screen discovered that memory varies by feature and
city. The multiscale representation kept both insights but expressed them as a
compact climate state:

```text
level + duration + variability + recent change + trend + season
```

That change—not a deeper or narrower neural network—produced the reliable
improvement from 18.8 to 16.6.
