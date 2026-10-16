"""Realized volatility forecasting.

Why this is separate from price forecasting. Price direction on 15-minute bars is
unpredictable: three trained models and a linear regression over five horizons gave
48-50% directional accuracy and a negative out-of-sample R². Volatility behaves
differently: it clusters.

    autocorrelation of r    at lags 1…192:  −0.03 … −0.01   (noise)
    autocorrelation of |r|  at lags 1…192:   0.31 …  0.10   (consistently positive)

Measured on SOL/USDC 15m. A linear regression check gave an out-of-sample R² of about
0.35 across two pairs, two periods and three horizons, the first positive result in
the whole study.

For the LP use case this is exactly the quantity that matters: the width of the
liquidity range is determined by the size of the move, not its direction.

## Definitions

Realized volatility over a range of bars is the square root of the sum of squared
log returns:

    RV[a, b) = sqrt( Σ r²[i] ),  i ∈ [a, b),   r[i] = ln(close[i] / close[i-1])

The target is the log of future RV. Log because the RV distribution is right-skewed;
in logs it is close to normal, and MSE stops penalizing rare spikes more than it
should.

## Naive baseline

For price, the naive forecast was "no change". For volatility it is "same as over
the past h bars". This is a noticeably stronger baseline: on its own it gives R² of
about 0.15-0.21, so it is harder to beat than it looks.

## HAR

Heterogeneous AutoRegressive: past RV on three time scales, h, 2h and 4h bars.
The model is well known from the literature; here it is reproduced, not invented.
Three numbers give R² of about 0.35, roughly twice as good as the naive baseline.

Important observation: adding all 20 of the project's technical indicators to HAR
**worsens** the result (0.348 → 0.316). They are derived from the same price series
and add overfitting rather than information. So HAR is intentionally kept bare.
"""

import numpy as np


LOG_REALIZED_VOL_TARGET = "log_realized_vol"

# HAR scales in units of the horizon: h, 2h, 4h bars back.
DEFAULT_HAR_SCALES = (1, 2, 4)

# Lower bound on RV before taking the log: on a perfectly flat stretch the sum of
# squares is zero, and log would turn into -inf.
MIN_REALIZED_VOL = 1e-12


class VolatilityError(ValueError):
    """Invalid volatility calculation parameters."""


def log_returns(close_prices):
    """Log returns aligned with the bars.

    The first element is zero: the first bar has no predecessor, and dropping it
    would shift all indexing relative to the original series.
    """
    close_prices = np.asarray(close_prices, dtype=np.float64).reshape(-1)
    if close_prices.size == 0:
        return np.empty(0)
    if np.any(close_prices <= 0):
        raise VolatilityError(
            "Log returns require strictly positive prices; "
            f"found {int((close_prices <= 0).sum())} values <= 0"
        )
    returns = np.zeros_like(close_prices)
    returns[1:] = np.diff(np.log(close_prices))
    return returns


def squared_return_cumsum(returns):
    """Prefix sums of r². Allow computing RV for any interval in O(1)."""
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    return np.concatenate([[0.0], np.cumsum(returns**2)])


def realized_volatility(cumsum, start, end):
    """RV over the half-open bar interval [start, end) from prefix sums."""
    start = np.asarray(start)
    end = np.asarray(end)
    if np.any(end < start):
        raise VolatilityError("Interval end cannot be before the start")
    total = np.maximum(cumsum[end] - cumsum[start], MIN_REALIZED_VOL)
    return np.sqrt(total)


def minimum_entry_index(horizon, scales=DEFAULT_HAR_SCALES):
    """First bar with enough history for the longest HAR scale.

    The longest scale looks back max(scales) * horizon bars, so an entry at
    index t requires t + 1 - max(scales) * horizon >= 0.
    """
    return max(scales) * horizon - 1


def entry_indices(total_bars, horizon, scales=DEFAULT_HAR_SCALES):
    """Entry bar indices that have both HAR history and a full horizon ahead."""
    if horizon <= 0:
        raise VolatilityError(f"horizon must be positive, got {horizon}")
    first = minimum_entry_index(horizon, scales)
    last = total_bars - horizon - 1
    if last < first:
        return np.empty(0, dtype=int)
    return np.arange(first, last + 1)


def build_volatility_target(close_prices, horizon, scales=DEFAULT_HAR_SCALES):
    """Target and naive forecast, aligned with the entry bars.

    Returns (y, naive, indices):
      y       - log RV over the next horizon bars after the entry;
      naive   - log RV over the previous horizon bars, i.e. "same as before";
      indices - entry bar indices in the original series.
    """
    returns = log_returns(close_prices)
    cumsum = squared_return_cumsum(returns)
    indices = entry_indices(len(returns), horizon, scales)
    if indices.size == 0:
        return np.empty(0), np.empty(0), indices

    forward = realized_volatility(cumsum, indices + 1, indices + 1 + horizon)
    backward = realized_volatility(cumsum, indices + 1 - horizon, indices + 1)
    return np.log(forward), np.log(backward), indices


def build_har_features(close_prices, horizon, scales=DEFAULT_HAR_SCALES):
    """HAR features: log RV on several scales, aligned with the entry bars.

    Columns follow the order of scales: with (1, 2, 4) and horizon=48 this is RV
    over the past 48, 96 and 192 bars.
    """
    returns = log_returns(close_prices)
    cumsum = squared_return_cumsum(returns)
    indices = entry_indices(len(returns), horizon, scales)
    if indices.size == 0:
        return np.empty((0, len(scales))), indices

    columns = [
        np.log(realized_volatility(cumsum, indices + 1 - scale * horizon, indices + 1))
        for scale in scales
    ]
    return np.column_stack(columns), indices


def fit_har(features, targets):
    """Ordinary least squares with an intercept. Returns a coefficient vector of length k+1.

    Keras is not needed here: the model is four numbers, and it is exactly what any
    more complex model has to beat. It is simpler to store as JSON than as .keras.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(features) != len(targets):
        raise VolatilityError(
            f"Row count mismatch between features and targets: {len(features)} and {len(targets)}"
        )
    if len(features) == 0:
        raise VolatilityError("Cannot fit HAR on an empty sample")
    design = np.hstack([features, np.ones((len(features), 1))])
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return coefficients


def predict_har(features, coefficients):
    features = np.asarray(features, dtype=np.float64)
    design = np.hstack([features, np.ones((len(features), 1))])
    return design @ np.asarray(coefficients, dtype=np.float64).reshape(-1)


def r2_score(targets, predictions):
    """R² relative to the mean. A negative value is worse than predicting the mean."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(targets) != len(predictions):
        raise VolatilityError(
            f"Count mismatch between targets and predictions: {len(targets)} and {len(predictions)}"
        )
    residual = np.sum((targets - predictions) ** 2)
    total = np.sum((targets - targets.mean()) ** 2)
    if total == 0:
        return float("nan")
    return float(1 - residual / total)


def independent_windows(sample_count, horizon):
    """Sample size adjusted for overlap.

    Neighboring observations share horizon-1 of their horizon future bars, so the
    nominal row count greatly overstates statistical power.
    """
    return int(sample_count // horizon)


def volatility_to_price_band(entry_price, log_realized_vol, z=1.2816):
    """From an RV forecast to price range bounds.

    Choosing an LP width needs the range, not a point price. Assuming the return
    over the horizon is normally distributed with zero mean and σ = RV, the
    interval [p_low, p_high] covers 2·Φ(z)−1 of the probability.

    z=1.2816 corresponds to the p10…p90 interval. The normality assumption is known
    to be inaccurate (return tails are heavier), so actual coverage will be below
    nominal; this must be checked on data rather than taken on faith.
    """
    entry_price = np.asarray(entry_price, dtype=np.float64).reshape(-1)
    sigma = np.exp(np.asarray(log_realized_vol, dtype=np.float64).reshape(-1))
    if len(entry_price) != len(sigma):
        raise VolatilityError(
            f"Price count mismatch: {len(entry_price)} prices vs {len(sigma)} forecasts"
        )
    return entry_price * np.exp(-z * sigma), entry_price * np.exp(z * sigma)
