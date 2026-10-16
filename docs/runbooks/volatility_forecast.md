# Volatility forecast

The first direction in the project where forecasting actually works.

## Why the direction changed

Price forecasting was closed with a negative result. Three trained models and a linear
regression across five horizons gave a directional accuracy of 48–50% and a negative
out-of-sample R². Details: `return_target_training.md`.

Volatility behaves differently, and this is visible even before any training:

```
autocorrelation of r    at lags 1…192:  −0.03 … −0.01   noise
autocorrelation of |r|  at lags 1…192:   0.31 …  0.10   consistently positive
```

Direction is unpredictable; magnitude clusters.

For LP this is exactly the quantity that matters: the width of the liquidity range is driven by
the size of the move, not by where exactly the price goes.

## Definitions

Realized volatility over a range of bars:

```
RV[a, b) = sqrt( Σ r²[i] ),  i ∈ [a, b),   r[i] = ln(close[i] / close[i-1])
```

The target is `log(RV)` over the next `horizon` bars. The logarithm is used because the RV
distribution is right-skewed; in logs it is close to normal, and MSE stops penalizing
rare spikes more than it should.

**The naive baseline here is not zero.** For price, the naive forecast was "no change";
for volatility it is "same as over the previous horizon bars". That is already a strong forecast.

**HAR** uses past RV at three scales: `horizon`, `2·horizon`, `4·horizon` bars.
The model is well known from the literature; here it is reproduced, not invented.

## Running

```bash
.venv/bin/python controller/volatility_baseline.py \
  --data data/sol_usdc_15m_clean/full_data_base/merged_sol_usdc_15m_clean.csv \
  --horizon 96
```

The horizon is in bars: on 15m, `8` = 2 hours, `48` = 12 hours, `96` = 24 hours.

Exit codes: `0`: HAR beat the naive forecast; `2`: it did not (too early to make the model more
complex, the problem is in the data); `1`: startup error.

The model is four numbers, so it is saved as JSON rather than `.keras`:
`models/saved_models/volatility_har/<stem>_har_h<horizon>.json`.

## Results on the collected data

| Dataset | Horizon | Naive R² | HAR R² | Gain | p10…p90 coverage | Width | Independent windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SOL/USDC 2.6 y | 2 h | 0.0136 | 0.2257 | +0.2121 | 82.2% | 1.91% | 1,838 |
| SOL/USDC 2.6 y | 12 h | 0.0430 | 0.2172 | +0.1741 | 85.1% | 5.29% | 305 |
| SOL/USDC 2.6 y | 24 h | 0.1602 | 0.2283 | +0.0680 | 85.1% | 7.78% | 152 |
| SOL/USDT 6 y | 2 h | 0.1457 | 0.3209 | +0.1751 | 81.3% | 2.27% | 4,208 |
| SOL/USDT 6 y | 12 h | 0.1397 | 0.3222 | +0.1825 | 82.5% | 6.17% | 700 |
| SOL/USDT 6 y | 24 h | 0.1674 | 0.3240 | +0.1566 | 83.1% | 9.03% | 349 |

HAR beats the naive forecast in all six combinations. On the longer dataset, which
has three times as many independent windows, the result is more stable and higher, which
is incompatible with fitting noise.

The absolute R² depends on which slice is held out for testing: with `--test-size 0.32`,
SOL/USDC gives about 0.35 instead of 0.23. It only makes sense to compare
`r2_har` and `r2_naive` from the same run, not values from different runs.

## Technical indicators make it worse

Tested separately: HAR with three numbers gives 0.348, HAR plus all 20 project
indicators gives 0.316. The indicators are derived from the same price series and add overfitting,
not information.

That is why HAR is deliberately kept bare, and **it is the bar for any neural network**: before
adding complexity, a model must beat a linear regression on three numbers. Three GRU
runs on 2026-08-12 showed that this does not happen automatically.

## Range width for LP

`volatility_to_price_band()` converts an RV forecast into bounds:

```
low  = entry · exp(−z · RV)
high = entry · exp(+z · RV)
```

With `z = 1.2816` these are the nominal p10…p90, i.e. 80% coverage under a normal
distribution of returns over the horizon.

Actual coverage on the data is **81–85%, i.e. above nominal**. The range
comes out conservative rather than narrow. The reason is heavy tails: there are more small
and very large moves than the normal distribution predicts, and fewer medium ones,
so the normal interval covers the middle with room to spare.

Practical takeaway for LP: the formula does not underestimate the width, but 15% of
moves still leave the range, and those are large moves. Their cost has to be assessed
separately; coverage alone does not tell you how much a miss costs.

## What this does not solve

Volatility **does not replace a price forecast**. It will not tell you where SOL is going. It
tells you how wide it is reasonable to set the range.

The model has not been tested on data other than SOL or outside the 15-minute timeframe.

The window-overlap limitation still applies: at 24 hours there are only
152–349 independent windows, and the confidence interval around R² is wide.

## Next steps

1. Test HAR on other pairs and timeframes to see whether the result transfers.
2. Estimate the cost of leaving the range, not just its frequency.
3. Only then try a model more complex than HAR, with an honest bar to beat.
