# Training on returns instead of price level

## Why this was needed

Historically, the models predicted the absolute `close` level `steps_ahead` bars
ahead. Two runs on real data on 2026-08-12 showed that this framing does not work:

| Run | Data | Model MAPE | Persistence MAPE | Directional accuracy |
| --- | --- | --- | --- | --- |
| 1 | SOL/USDT, 6 years, 210,447 bars | 12.58% | 0.25% | **48.93%** |
| 2 | SOL/USDC, 2.6 years, 91,976 bars | 1.97% | 0.20% | **48.27%** |

MAPE dropped sixfold as the price range narrowed, while directional accuracy did
not move and stayed below 50%, i.e. worse than a coin flip. The absolute metrics
reflected the width of the price range, not predictive power.

The root cause is the problem framing. The entire price level is already present in the
input window: the autocorrelation of close at lag 2 is 0.9999, so persistence ("the price
will not change") gets an R² of about 0.999 for free. The model spent its capacity on a
contest it cannot win and had nothing left for the only thing of value: the direction
and size of the change.

Separately, this broke the category system: the naive forecast scored an "accuracy" of
99.65% and landed in the `95-100%` category without predicting anything.

## What changed

The training target is set in the config:

```yaml
global_config:
  target: log_return_pct   # or level for the previous behavior
  return_scale: 100.0
```

With `log_return_pct`, the target is computed as:

```
y[k] = return_scale * ln(close[t + k + 1] / close[t])
```

where `t` is the last bar of the input window and `k = 0 … steps_ahead-1`.

Why the logarithm: log returns are additive across the horizon and symmetric around zero.
The factor of 100 converts them to percent: a typical 30-minute move becomes about 0.25
instead of 0.0025, so the MSE gradients do not vanish.

Price reconstruction: `price[k] = entry_price * exp(y[k] / return_scale)`.

Implementation: `utils/targets.py`.

## Model metadata

Level and return models are incompatible, so a sidecar file is saved next to each model:

```
<bucket>/<stem>.keras
<bucket>/<stem>_scaler.pkl
<bucket>/<stem>_target.json    ← target, window, horizon, return_scale, hyperparameters
```

### Hyperparameters and comparability

The sidecar stores `lr`, `dropout_rate`, `batch_size`, `units`, `num_layers`:
the parameters whose mismatch makes a comparison of two models invalid.
`epochs` and `patience` are not stored: they do not affect the comparability of
already trained models.

`validation_gate` compares these blocks and warns if the baseline and the candidate
were trained differently:

```
⚠️  baseline and candidate were trained with DIFFERENT hyperparameters:
    lr: baseline 0.001 → candidate 0.00044
    The metric gap reflects the settings as well, not just the data.
```

Without this check, the gate would blame the data for a degradation caused by a change of
settings, i.e. it would give exactly the wrong verdict it exists to prevent.

A block missing on one side **does not count as a mismatch**: older models did not
write it, and treating the unknown as a conflict would devalue the warning with
false positives.

There was a concrete reason for this check: the `lr` key was not read at all for a month
and a half (`Adam()` was created without `learning_rate`), so every model was trained
at 0.001 instead of the 0.00044 set in the config.

**A model without a sidecar is treated as a legacy level model**: before returns were
introduced there was no other option, so older models keep working.

`model_staging.py` moves the sidecar together with the model/scaler pair; for legacy models
its absence does not block staging.

## Running

```bash
scripts/with_gpu.sh .venv/bin/python models/train_gru.py \
  --config configs/config_gpu_sol_returns.yaml \
  --file data/sol_usdc_15m_clean/full_data_base/merged_sol_usdc_15m_clean.csv
```

The config `configs/config_gpu_sol_returns.yaml` differs from `config_gpu_sol.yaml`
only in the `target` key. Everything else is identical on purpose, so that any difference
in metrics comes from the problem framing, not the hyperparameters.

After training, the script prints a comparison with persistence:

```
=== Comparison with naive forecast ===
  persistence: MAPE 0.2419%   accuracy 99.76%
  model:       MAPE 0.2419%   accuracy 99.76%
  model directional accuracy: 48.38%
  ⚠️ Model does NOT beat persistence — a low MAPE proves nothing here.
```

## Scaler leakage fixed along the way

`prep_utils.scale_data()` ran `fit_transform` on the whole dataset, and only then was
the data split. The scaler saw val and test, a direct violation of the project rule
"fit the scaler on train only". As a result, holdout metrics came out more optimistic
than they really were.

`scale_data_train_only()` was added. The row boundary is computed separately because
`split_data` splits sequences, not rows: the last training sequence with index `s`
covers rows `[s, s + window + steps)`.

A side effect that is now visible honestly: val and test values can fall outside
`[0, 1]` if the market moved beyond the training range. This is not a bug; it
reflects what the model will face in production.

The old `scale_data()` is kept: it is correct when only the training portion is
deliberately passed in.

## What the first return-target run showed

```
predicted returns (step 1):
  min −0.00040   max −0.00040   mean −0.00040   std 0.00000
actual:
  min −3.47687   max +5.29655   mean −0.00085   std 0.30415
```

The model output **a single constant** for all 14,712 samples. This is the MSE-optimal
answer when there is no predictable signal: when minimizing squared error on noise, the
best you can do is output the unconditional mean.

Conclusion: at a 30-minute horizon there is no predictable signal in the current
features. This is neither a model defect nor a lack of data: with a correct framing and
92 thousand samples, the network converges to a constant in 33 epochs.

What matters is that this is now **visible immediately**. Previously, the same
"no signal" looked like "98% accuracy, category 95-100%".

## A longer horizon was tested and does not help

The obvious first idea was to lengthen the horizon: 30 minutes is almost pure noise, and
the project plans already assumed 12–24 hour horizons for LP. Instead of hours of
training, the check was done with a linear regression on all 20 features: if there is no
signal even for that, it is too early to discuss architecture.

The **in-sample** picture looked promising: the signal grew monotonically with the horizon.

| Horizon | max \|corr\| | best feature | R² train |
| --- | --- | --- | --- |
| 30 min | 0.0289 | return | 0.0036 |
| 2 hours | 0.0374 | mom_10 | 0.0047 |
| 12 hours | 0.0730 | bb_lower | 0.0131 |
| 24 hours | 0.1025 | low | 0.0196 |
| 48 hours | 0.1470 | low | 0.0307 |

**Out-of-sample**, the picture is reversed:

| Horizon | R² train | R² test | dir.acc test | independent windows |
| --- | --- | --- | --- | --- |
| 30 min | 0.0036 | **−0.0019** | 48.60% | 14,715 |
| 2 hours | 0.0047 | **−0.0106** | 49.36% | 3,678 |
| 12 hours | 0.0131 | **−0.0372** | 48.35% | 612 |
| 24 hours | 0.0196 | **−0.0785** | 49.75% | 305 |
| 48 hours | 0.0307 | **−0.1566** | 48.26% | 152 |

A negative R² means "worse than predicting the mean". Directional accuracy of
48–50% at every horizon is at chance level.

The longer the horizon, the better the in-sample fit and the worse the out-of-sample
result. That is the signature of fitting noise. The reason is visible in the last column:
forward returns on 15-minute bars overlap heavily. At a 48-hour horizon, neighboring
observations share 191 of 192 bars, leaving 152 independent windows rather than 92
thousand. The growing correlation was an artifact of that overlap, not an emerging signal.

**Conclusion: a longer horizon does not help.** This direction is closed; there is no
point in training on 12–24 hours.

Caveat: the check is linear. A nonlinear signal could in principle exist where the
regression cannot see it. But the GRU with attention already collapsed to a constant at
30 minutes, and together these two results are consistent.

## Where to go next

Remaining directions, in order of expected payoff:

1. **Different features.** All 20 current features are derived from the same price
   series, so they carry little independent information. The order book, volume profile,
   funding rates, cross-asset relationships and on-chain metrics carry what OHLCV lacks.
   This is the only direction that changes the input information rather than how it is processed.
2. **Classification instead of regression.** MSE on noisy returns rewards predicting the
   mean, which is exactly what happened. Cross-entropy on balanced classes is harder to
   collapse, but it does not create information out of thin air.
3. **Uncertainty estimation.** p10/p90 intervals are more useful than a point forecast,
   both for choosing the LP range width and for an honest skill test. They do not require
   a signal in the mean.
4. **Accept the negative result.** No edge on 15-minute SOL OHLCV is a legitimate
   conclusion, consistent with the project rule "do not wire in LP logic until an edge
   is proven".

## Limitations

`backtest.py` supports only level models and **stops with an explicit error** on return
models. Running one through the scaler's inverse transform is technically possible, but it
yields plausible-looking numbers and a meaningless trade report. Return support in
backtest is a separate task.

`validation_gate.py` handles both types and picks the transform based on the metadata.

`models/train_gru11.py` has not been updated: nothing calls it and it remains archival.
