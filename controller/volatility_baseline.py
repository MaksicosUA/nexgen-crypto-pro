"""HAR baseline for realized volatility forecasting.

The project's first model that predicts the size of the price move rather than the
price itself. Intentionally linear and built on three features: it sets the bar that
any more complex model must beat. Three GRU runs on 2026-08-12 showed that a neural
network does not clear this bar automatically.

The model is four numbers, so it is saved as JSON rather than .keras.

Usage:
    .venv/bin/python controller/volatility_baseline.py \\
      --data data/sol_usdc_15m_clean/full_data_base/merged_sol_usdc_15m_clean.csv \\
      --horizon 96

Exit codes:
    0  HAR beat the naive forecast
    2  HAR is no better than naive: too early to add complexity, the problem is in the data
    1  run error
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.volatility import (
    DEFAULT_HAR_SCALES,
    LOG_REALIZED_VOL_TARGET,
    VolatilityError,
    build_har_features,
    build_volatility_target,
    fit_har,
    independent_windows,
    log_returns,
    predict_har,
    r2_score,
    volatility_to_price_band,
)


DEFAULT_OUTPUT = PROJECT_DIR / "models" / "saved_models" / "volatility_har"


def band_coverage(close_prices, indices, horizon, predicted_log_vol, z):
    """Share of cases where the price after horizon bars stayed inside the forecast range.

    The width is derived from the volatility forecast under a normality assumption.
    Return tails are heavier than normal, so actual coverage is usually below
    nominal; this has to be measured rather than taken on faith.
    """
    close_prices = np.asarray(close_prices, dtype=np.float64)
    entry = close_prices[indices]
    future = close_prices[indices + horizon]
    low, high = volatility_to_price_band(entry, predicted_log_vol, z=z)
    inside = (future >= low) & (future <= high)
    relative_width = np.mean((high - low) / entry) * 100
    return float(np.mean(inside) * 100), float(relative_width)


def evaluate(data_path, horizon, test_size, scales=DEFAULT_HAR_SCALES, z=1.2816):
    from backtest import load_numeric_backtest_frame

    frame = load_numeric_backtest_frame(data_path)
    close = frame["close"].to_numpy(dtype=np.float64)

    targets, naive, indices = build_volatility_target(close, horizon, scales)
    features, _ = build_har_features(close, horizon, scales)
    if len(targets) == 0:
        raise VolatilityError(
            f"Not enough data: rows {len(close)}, horizon {horizon}, "
            f"need at least {max(scales) * horizon + horizon} bars"
        )

    split = int(len(targets) * (1 - test_size))
    if split < 2 or len(targets) - split < 2:
        raise VolatilityError(
            f"Too few observations to split: total {len(targets)}, test_size={test_size}"
        )

    coefficients = fit_har(features[:split], targets[:split])
    predicted = predict_har(features[split:], coefficients)
    actual = targets[split:]

    r2_har = r2_score(actual, predicted)
    r2_naive = r2_score(actual, naive[split:])
    coverage, width = band_coverage(
        close, indices[split:], horizon, predicted, z
    )

    return {
        "target": LOG_REALIZED_VOL_TARGET,
        "data": str(data_path),
        "horizon_bars": int(horizon),
        "har_scales": list(scales),
        "test_size": test_size,
        "samples_train": int(split),
        "samples_test": int(len(targets) - split),
        "independent_windows_test": independent_windows(len(targets) - split, horizon),
        "coefficients": [float(value) for value in coefficients],
        "r2_har": r2_har,
        "r2_naive": r2_naive,
        "r2_gain_over_naive": r2_har - r2_naive,
        "beats_naive": bool(r2_har > r2_naive),
        "band_z": z,
        "band_nominal_coverage_pct": float((2 * 0.9 - 1) * 100) if z == 1.2816 else None,
        "band_actual_coverage_pct": coverage,
        "band_mean_width_pct": width,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }


def format_report(report):
    lines = [
        "",
        f"=== HAR volatility baseline, horizon {report['horizon_bars']} bars ===",
        f"  train {report['samples_train']:,} observations, test {report['samples_test']:,}",
        f"  independent windows in test: {report['independent_windows_test']:,}",
        "",
        f"  R² naive ('as before'):  {report['r2_naive']:>8.4f}",
        f"  R² HAR:                  {report['r2_har']:>8.4f}",
        f"  gain:                    {report['r2_gain_over_naive']:>+8.4f}",
    ]
    if report["beats_naive"]:
        lines.append("  ✅ HAR beats the naive forecast")
    else:
        lines.append("  ❌ HAR does NOT beat the naive forecast: too early to add model complexity")

    lines += [
        "",
        f"  forecast range p10…p90 (z={report['band_z']}):",
        f"    mean width:          {report['band_mean_width_pct']:.2f}% of price",
        f"    actual coverage:     {report['band_actual_coverage_pct']:.1f}%"
        f" vs nominal 80.0%",
    ]
    if report["band_actual_coverage_pct"] < 75:
        lines.append(
            "    ⚠️ Coverage is well below nominal: tails are heavier than normal,"
        )
        lines.append(
            "       so for LP the range must be widened beyond the normal assumption."
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HAR baseline for realized volatility forecasting"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--horizon",
        type=int,
        default=96,
        help="Horizon in bars. At 15m: 8=2h, 48=12h, 96=24h (default)",
    )
    parser.add_argument("--test-size", type=float, default=0.16)
    parser.add_argument(
        "--z",
        type=float,
        default=1.2816,
        help="Quantile for the range width; 1.2816 corresponds to p10…p90",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the model and report; defaults to models/saved_models/volatility_har/",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.test_size < 1:
        print(f"test_size must be in (0, 1), got {args.test_size}", file=sys.stderr)
        return 1

    try:
        report = evaluate(args.data, args.horizon, args.test_size, z=args.z)
    except (VolatilityError, FileNotFoundError, ValueError) as error:
        print(f"Failed to build baseline: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(format_report(report))

    if not args.no_save:
        output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.data).stem
        path = output_dir / f"{stem}_har_h{args.horizon}.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nModel and report saved: {path}")

    return 0 if report["beats_naive"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
