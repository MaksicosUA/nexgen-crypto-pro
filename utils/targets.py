"""Training target: price level or return.

Why this is a separate module. Historically the models predicted the absolute close
level steps_ahead bars ahead. Two runs on real data showed that this framing does
not work:

    SOL/USDT 6 years:   MAPE 12.58%, directional accuracy 48.93%
    SOL/USDC 2.6 years: MAPE  1.97%, directional accuracy 48.27%
    persistence:        MAPE  0.20-0.25%

MAPE changed sixfold along with the width of the price range, while directional
accuracy did not move and stayed below a coin flip. The cause is the problem setup:
the whole price level is already contained in the input window, persistence gets it
for free with R² around 0.999, and the model spends its capacity on a contest it
cannot win.

Here the target is the log return over the horizon, in percent:

    y[k] = 100 * ln(close[t + k + 1] / close[t])

where t is the last bar of the input window, k = 0 … steps_ahead-1.

What this gives:
  * the target is stationary, so a market regime change between train and test
    no longer breaks scaling;
  * persistence becomes "return equals zero", a bar that can now be beaten
    meaningfully;
  * directional accuracy starts to mean exactly what trading signals need;
  * metrics become comparable across pairs, periods and timeframes.

Log rather than a simple ratio: returns are additive over the horizon and symmetric
around zero. The factor of 100 converts to percent, so a typical value is around
0.25 instead of 0.0025 and MSE gradients do not vanish.

The two model types are incompatible with each other, so a metadata file is saved
next to each model. A model without one is treated as a legacy level model.
"""

import json
from pathlib import Path

import numpy as np


LEVEL_TARGET = "level"
LOG_RETURN_TARGET = "log_return_pct"
KNOWN_TARGETS = (LEVEL_TARGET, LOG_RETURN_TARGET)

METADATA_SUFFIX = "_target.json"
DEFAULT_RETURN_SCALE = 100.0


class TargetError(ValueError):
    """Invalid training target or incompatible metadata."""


def metadata_path(model_path):
    """Path to the metadata sidecar next to the model."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}{METADATA_SUFFIX}")


# Hyperparameters whose mismatch makes a model comparison invalid.
# If baseline and candidate were trained with different values, the metric gap
# reflects the settings rather than the data, and the data is what the gate evaluates.
COMPARABLE_HYPERPARAMETERS = (
    "lr",
    "dropout_rate",
    "batch_size",
    "units",
    "num_layers",
)


def extract_hyperparameters(config):
    """Subset of the config that affects whether two models are comparable.

    Written to the sidecar on save: without it there is no way to tell "the model is
    worse because the data is worse" from "the model is worse because it was trained
    differently". That exact case happened in this project: the lr key was not read
    for a month and a half, and every model was trained with a value different from
    the one in the config.
    """
    config = config or {}
    return {
        name: config[name] for name in COMPARABLE_HYPERPARAMETERS if name in config
    }


def compare_hyperparameters(baseline_meta, candidate_meta):
    """List of hyperparameter differences between two models.

    An empty list means the metric comparison is valid. A missing hyperparameters
    block on either side is not a difference but an unknown: older models did not
    write it, and reporting that as a conflict would be a false alarm.
    """
    baseline = (baseline_meta or {}).get("hyperparameters")
    candidate = (candidate_meta or {}).get("hyperparameters")
    if not baseline or not candidate:
        return []

    differences = []
    for name in sorted(set(baseline) | set(candidate)):
        left = baseline.get(name)
        right = candidate.get(name)
        if left != right:
            differences.append({"name": name, "baseline": left, "candidate": right})
    return differences


def write_target_metadata(
    model_path,
    target_kind,
    window_size,
    steps_ahead,
    return_scale=DEFAULT_RETURN_SCALE,
    hyperparameters=None,
    extra=None,
):
    if target_kind not in KNOWN_TARGETS:
        raise TargetError(f"Unknown target: {target_kind!r}. Available: {KNOWN_TARGETS}")
    payload = {
        "target": target_kind,
        "window_size": int(window_size),
        "steps_ahead": int(steps_ahead),
        "return_scale": float(return_scale),
    }
    if hyperparameters:
        payload["hyperparameters"] = dict(hyperparameters)
    if extra:
        payload.update(extra)
    path = metadata_path(model_path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return path


def read_target_metadata(model_path):
    """Model metadata. A missing file means a legacy level model.

    This keeps older models working unchanged: before return targets were added,
    no other option existed.
    """
    path = metadata_path(model_path)
    if not path.is_file():
        return {"target": LEVEL_TARGET, "legacy": True, "return_scale": DEFAULT_RETURN_SCALE}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TargetError(f"Corrupted metadata {path}: {error}") from error

    target = payload.get("target")
    if target not in KNOWN_TARGETS:
        raise TargetError(
            f"Metadata {path} contains an unknown target {target!r}. Available: {KNOWN_TARGETS}"
        )
    payload.setdefault("return_scale", DEFAULT_RETURN_SCALE)
    payload.setdefault("legacy", False)
    return payload


def is_return_model(model_path):
    return read_target_metadata(model_path)["target"] == LOG_RETURN_TARGET


def entry_indices(total_rows, window_size, steps_ahead):
    """Entry bar indices: the last bar of each input window.

    They match the create_sequences slicing: for window i the input ends at
    i + window_size - 1 and the targets start at i + window_size.
    """
    count = total_rows - window_size - steps_ahead + 1
    if count <= 0:
        return np.empty(0, dtype=int)
    return np.arange(count) + window_size - 1


def build_log_return_targets(
    close_prices,
    window_size,
    steps_ahead,
    return_scale=DEFAULT_RETURN_SCALE,
):
    """Return targets and entry prices, aligned with create_sequences.

    Returns (y, entry_prices):
      y             - (n, steps_ahead), log return in percent;
      entry_prices  - (n,), unscaled price of the last bar of the input window.
    """
    close_prices = np.asarray(close_prices, dtype=np.float64).reshape(-1)
    if window_size <= 0 or steps_ahead <= 0:
        raise TargetError(
            f"window_size and steps_ahead must be positive, got {window_size}, {steps_ahead}"
        )
    if return_scale <= 0:
        raise TargetError(f"return_scale must be positive, got {return_scale}")
    if np.any(close_prices <= 0):
        raise TargetError(
            "Log returns require strictly positive prices; "
            f"found {int((close_prices <= 0).sum())} values <= 0"
        )

    indices = entry_indices(len(close_prices), window_size, steps_ahead)
    if indices.size == 0:
        return np.empty((0, steps_ahead)), np.empty(0)

    entry_prices = close_prices[indices]
    targets = np.empty((len(indices), steps_ahead), dtype=np.float64)
    for step in range(steps_ahead):
        future = close_prices[indices + step + 1]
        targets[:, step] = return_scale * np.log(future / entry_prices)
    return targets, entry_prices


def log_returns_to_prices(entry_prices, returns, return_scale=DEFAULT_RETURN_SCALE):
    """Reconstructs absolute prices from returns.

    price[k] = entry_price * exp(return[k] / return_scale)
    """
    entry_prices = np.asarray(entry_prices, dtype=np.float64).reshape(-1)
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    if len(entry_prices) != len(returns):
        raise TargetError(
            f"Price count mismatch: {len(entry_prices)} entry prices vs {len(returns)} returns"
        )
    return entry_prices[:, None] * np.exp(returns / return_scale)


def prices_to_log_returns(entry_prices, prices, return_scale=DEFAULT_RETURN_SCALE):
    """Inverse transform from prices to returns. Needed to compare legacy models."""
    entry_prices = np.asarray(entry_prices, dtype=np.float64).reshape(-1)
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim == 1:
        prices = prices.reshape(-1, 1)
    if np.any(entry_prices <= 0) or np.any(prices <= 0):
        raise TargetError("Converting to returns requires strictly positive prices")
    return return_scale * np.log(prices / entry_prices[:, None])


def zero_return_prediction(entry_prices, steps_ahead):
    """Persistence forecast in return space: identically zero.

    Kept as a separate function so that the "better than doing nothing" bar
    is explicit in the code rather than implied.
    """
    entry_prices = np.asarray(entry_prices).reshape(-1)
    return np.zeros((len(entry_prices), steps_ahead), dtype=np.float64)
