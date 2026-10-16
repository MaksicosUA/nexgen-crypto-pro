import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from utils.data_utils import split_data
from utils.metrics import calculate_accuracy, calculate_additional_metrics
from utils.prep_utils import create_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest ML model on historical data")
    parser.add_argument("--data", required=True, help="CSV file to test on")
    parser.add_argument("--model", required=True, help="Model file (.keras/.h5)")
    parser.add_argument("--scaler", required=True, help="Scaler file (.pkl)")
    parser.add_argument("--window", type=int, default=60, help="Window size")
    parser.add_argument("--steps", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--output", default="backtest_result.png", help="Where to save the chart")
    parser.add_argument("--report", default=None, help="Where to save the backtest JSON report")
    parser.add_argument("--start-balance", type=float, default=1000, help="Starting balance in USDT")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission per trade, as a fraction of notional")
    parser.add_argument("--slippage-pct", type=float, default=0.0, help="Execution slippage in percent")
    parser.add_argument(
        "--signal-threshold-pct",
        type=float,
        default=0.2,
        help="Minimum expected forecast edge over the current price, in percent",
    )
    return parser.parse_args()


def require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got: {value}")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def single_shape(shape):
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError(f"Backtest supports only single-input models, got input_shape={shape}")
        return shape[0]
    return shape


def validate_scaler(scaler, feature_columns):
    warnings = []
    if not isinstance(scaler, MinMaxScaler):
        raise ValueError(f"scaler must be sklearn.preprocessing.MinMaxScaler, got: {type(scaler).__name__}")

    required_attrs = ("data_min_", "data_max_", "scale_", "min_", "n_features_in_")
    missing_attrs = [attr for attr in required_attrs if not hasattr(scaler, attr)]
    if missing_attrs:
        raise ValueError(
            "scaler does not look like a fitted sklearn MinMaxScaler; missing attributes: "
            + ", ".join(missing_attrs)
        )

    expected_features = int(scaler.n_features_in_)
    actual_features = len(feature_columns)
    if expected_features != actual_features:
        raise ValueError(
            f"Feature count mismatch: scaler expects {expected_features}, "
            f"CSV after numeric/filter/order yields {actual_features}. "
            "Check the column set and order against training."
        )

    if hasattr(scaler, "feature_names_in_"):
        scaler_columns = list(scaler.feature_names_in_)
        if scaler_columns != list(feature_columns):
            raise ValueError(
                "Feature name/order mismatch between scaler and CSV: "
                f"scaler={scaler_columns}, csv={list(feature_columns)}"
            )
    else:
        warnings.append(
            "scaler has no feature_names_in_; for a legacy model only the feature count "
            "and model shape are strictly checked, exact feature names cannot be recovered"
        )

    invalid_scale_idx = np.where(~np.isfinite(scaler.scale_) | (scaler.scale_ == 0))[0].tolist()
    if invalid_scale_idx:
        invalid_columns = [
            feature_columns[idx] if idx < len(feature_columns) else f"#{idx}"
            for idx in invalid_scale_idx
        ]
        raise ValueError(
            "scaler.scale_ contains zero or non-numeric values for features: "
            + ", ".join(invalid_columns)
        )

    return warnings


def validate_model_shape(model, window_size, feature_count, steps_ahead):
    model_input_shape = single_shape(model.input_shape)
    if len(model_input_shape) != 3:
        raise ValueError(
            "Backtest expects a sequence model with input_shape=(None, window, features), "
            f"got: {model.input_shape}"
        )

    model_window = model_input_shape[1]
    model_features = model_input_shape[2]
    if model_window is not None and int(model_window) != window_size:
        raise ValueError(f"Window mismatch: model expects {model_window}, CLI --window={window_size}")
    if model_features is not None and int(model_features) != feature_count:
        raise ValueError(
            f"Model feature count mismatch: model expects {model_features}, "
            f"backtest prepared {feature_count}"
        )

    output_shape = single_shape(model.output_shape)
    if len(output_shape) < 2:
        raise ValueError(f"Unsupported model output_shape: {model.output_shape}")
    model_steps = output_shape[-1]
    if model_steps is not None and int(model_steps) != steps_ahead:
        raise ValueError(f"Horizon mismatch: model outputs {model_steps} step(s), CLI --steps={steps_ahead}")


def reject_unsupported_target(model_path):
    """Backtest currently supports only price-level models.

    A return model outputs the log change relative to the entry price, not a price.
    Technically it can be run through the scaler's inverse transform — the result is
    plausible-looking numbers and a completely meaningless trade report. So it is better
    to stop explicitly than to compute nonsense.
    """
    from utils.targets import LOG_RETURN_TARGET, read_target_metadata

    meta = read_target_metadata(model_path)
    if meta["target"] == LOG_RETURN_TARGET:
        raise ValueError(
            f"Model {model_path} was trained on returns (target={meta['target']}), "
            "but backtest.py supports only price-level models. Return-model support "
            "in backtest is a separate task; for now use validation_gate.py, "
            "which handles both types."
        )
    return meta


def inverse_scale_target(values, scaler, target_col_idx=0):
    # MinMaxScaler applies X_scaled = X * scale_[i] + min_[i].
    # Therefore X_original = (X_scaled - min_[i]) / scale_[i].
    values = np.asarray(values, dtype=np.float64)
    return (values - scaler.min_[target_col_idx]) / scaler.scale_[target_col_idx]


def write_report(report_path, payload):
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_numeric_backtest_frame(data_path):
    df = pd.read_csv(data_path)
    df.columns = [col.lower() for col in df.columns]

    duplicate_columns = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(
            "Backtest CSV contains duplicate column names after normalization: "
            + ", ".join(duplicate_columns)
        )

    df = df.select_dtypes(include=[np.number])
    if "close" not in df.columns:
        raise ValueError(
            "Backtest CSV must contain a numeric 'close' column; "
            "make sure close values are numeric and the column is not dropped by the select_dtypes filter"
        )

    # Training scripts move close to index 0 before fitting scaler/model.
    # Keep the same feature order here so scaler.transform() receives the same layout.
    df = df.loc[:, ["close"] + [col for col in df.columns if col != "close"]].copy()
    df = df.astype("float32")

    invalid_mask = ~np.isfinite(df.to_numpy(dtype=np.float64))
    if invalid_mask.any():
        invalid_rows = int(np.unique(np.where(invalid_mask)[0]).size)
        raise ValueError(
            f"CSV contains NaN/inf in numeric features after preparation ({invalid_rows} rows). "
            "Clean the data before backtesting to avoid silently distorted metrics."
        )

    return df


def update_trade_stats(balance, balance_peak, max_drawdown, profit, min_profit, max_profit):
    if min_profit is None or profit < min_profit:
        min_profit = profit
    if max_profit is None or profit > max_profit:
        max_profit = profit
    if balance > balance_peak:
        balance_peak = balance
    drawdown = balance_peak - balance
    if drawdown > max_drawdown:
        max_drawdown = drawdown
    return balance_peak, max_drawdown, min_profit, max_profit


def update_drawdown(equity, balance_peak, max_drawdown):
    if equity > balance_peak:
        balance_peak = equity
    drawdown = balance_peak - equity
    if drawdown > max_drawdown:
        max_drawdown = drawdown
    return balance_peak, max_drawdown


def mark_to_market_equity(balance, position_qty, price, commission, slippage_pct):
    if position_qty <= 0:
        return balance
    liquidation_price = price * (1 - slippage_pct / 100)
    liquidation_notional = position_qty * liquidation_price
    return liquidation_notional - liquidation_notional * commission


def run_virtual_trader(
    execution_prices,
    preds_trade,
    initial_balance,
    commission,
    slippage_pct,
    filter_threshold,
):
    balance = initial_balance
    position_qty = 0.0
    entry_price = None
    entry_balance = None
    entry_notional = None
    entry_fee = None
    trade_history = []
    trade_profits = []
    max_drawdown = 0.0
    balance_peak = initial_balance
    min_profit = None
    max_profit = None

    for i in range(len(preds_trade)):
        price_now = float(execution_prices[i])
        pred_now = float(preds_trade[i])
        expected_return_pct = (pred_now / price_now - 1) * 100
        equity = mark_to_market_equity(balance, position_qty, price_now, commission, slippage_pct)
        balance_peak, max_drawdown = update_drawdown(equity, balance_peak, max_drawdown)

        # Signal is based on forecast edge over the executable current close,
        # not on forecast-to-forecast drift.
        if abs(expected_return_pct) < filter_threshold:
            continue

        if expected_return_pct > 0 and position_qty == 0:
            entry_price = price_now * (1 + slippage_pct / 100)
            entry_balance = balance
            position_qty = balance / (entry_price * (1 + commission))
            entry_notional = position_qty * entry_price
            entry_fee = entry_notional * commission
            cash_spent = entry_notional + entry_fee
            balance = entry_balance - cash_spent
            if abs(balance) < 1e-9:
                balance = 0.0
            trade_history.append(
                {
                    "type": "BUY",
                    "price": entry_price,
                    "qty": position_qty,
                    "notional": entry_notional,
                    "fee": entry_fee,
                    "cash_spent": cash_spent,
                    "expected_return_pct": expected_return_pct,
                    "balance": balance,
                    "idx": i,
                }
            )
            equity = mark_to_market_equity(balance, position_qty, price_now, commission, slippage_pct)
            balance_peak, max_drawdown = update_drawdown(equity, balance_peak, max_drawdown)

        elif expected_return_pct < 0 and position_qty > 0:
            exit_price = price_now * (1 - slippage_pct / 100)
            exit_notional = position_qty * exit_price
            exit_fee = exit_notional * commission
            exit_cash = exit_notional - exit_fee
            balance += exit_cash
            profit = exit_cash - (entry_notional + entry_fee)
            trade_history.append(
                {
                    "type": "SELL",
                    "price": exit_price,
                    "qty": position_qty,
                    "notional": exit_notional,
                    "fee": exit_fee,
                    "balance": balance,
                    "profit": profit,
                    "profit_pct": profit / entry_balance * 100,
                    "expected_return_pct": expected_return_pct,
                    "idx": i,
                }
            )
            trade_profits.append(profit)
            balance_peak, max_drawdown, min_profit, max_profit = update_trade_stats(
                balance, balance_peak, max_drawdown, profit, min_profit, max_profit
            )
            position_qty = 0.0
            entry_price = None
            entry_balance = None
            entry_notional = None
            entry_fee = None

    if position_qty > 0:
        final_price = float(execution_prices[-1]) * (1 - slippage_pct / 100)
        exit_notional = position_qty * final_price
        exit_fee = exit_notional * commission
        exit_cash = exit_notional - exit_fee
        balance += exit_cash
        profit = exit_cash - (entry_notional + entry_fee)
        trade_history.append(
            {
                "type": "SELL_END",
                "price": final_price,
                "qty": position_qty,
                "notional": exit_notional,
                "fee": exit_fee,
                "balance": balance,
                "profit": profit,
                "profit_pct": profit / entry_balance * 100,
                "idx": len(execution_prices) - 1,
            }
        )
        trade_profits.append(profit)
        balance_peak, max_drawdown, min_profit, max_profit = update_trade_stats(
            balance, balance_peak, max_drawdown, profit, min_profit, max_profit
        )

    return {
        "balance": balance,
        "trade_history": trade_history,
        "trade_profits": trade_profits,
        "max_drawdown": max_drawdown,
        "min_profit": min_profit,
        "max_profit": max_profit,
    }


def validate_price_series(series_by_name):
    for series_name, series in series_by_name.items():
        if not np.isfinite(series).all():
            raise ValueError(f"{series_name} contains NaN/inf after inverse transform")
        if (series <= 0).any():
            raise ValueError(f"{series_name} contains a price <= 0, backtest cannot safely compute returns")


def main():
    args = parse_args()
    require_positive_int(args.window, "--window")
    require_positive_int(args.steps, "--steps")
    if args.start_balance <= 0:
        raise ValueError(f"--start-balance must be positive, got: {args.start_balance}")
    if args.commission < 0:
        raise ValueError(f"--commission cannot be negative, got: {args.commission}")
    if args.slippage_pct < 0:
        raise ValueError(f"--slippage-pct cannot be negative, got: {args.slippage_pct}")
    if args.signal_threshold_pct < 0:
        raise ValueError(f"--signal-threshold-pct cannot be negative, got: {args.signal_threshold_pct}")

    window_size = args.window
    steps_ahead = args.steps
    target_col_idx = 0

    df = load_numeric_backtest_frame(args.data)
    feature_columns = df.columns.tolist()
    if len(df) < window_size + steps_ahead:
        raise ValueError(
            f"Not enough rows for backtest: rows={len(df)}, "
            f"need at least window + steps = {window_size + steps_ahead}"
        )

    if not args.scaler.endswith(".pkl"):
        raise ValueError("scaler must be a .pkl file (joblib)")

    scaler = joblib.load(args.scaler)
    scaler_warnings = validate_scaler(scaler, feature_columns)
    scaled_data = scaler.transform(df.values)

    X, y = create_sequences(scaled_data, window_size, steps_ahead, target_col_idx=target_col_idx)
    if X.size == 0 or y.size == 0:
        raise ValueError("Failed to build sequence data for backtest; check window/steps and the CSV size")

    _, _, X_test, _, _, y_test = split_data(X, y, val_size=0.0, test_size=0.1)
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError(
            f"Test set is empty after split_data(test_size=0.1): sequences={len(X)}. "
            "Use a larger CSV or reduce window/steps."
        )
    if X_test.ndim != 3:
        raise ValueError(
            "Backtest expects the sequence tensor layout (samples, time, features), "
            f"got X_test.shape={X_test.shape}"
        )
    if X_test.shape[1] != window_size or X_test.shape[2] != len(feature_columns):
        raise ValueError(
            "X_test layout does not match the expected (samples, window, features): "
            f"X_test.shape={X_test.shape}, window={window_size}, features={len(feature_columns)}"
        )

    reject_unsupported_target(args.model)
    model = load_model(args.model)
    validate_model_shape(model, window_size, len(feature_columns), steps_ahead)
    preds = model.predict(X_test)
    if not np.isfinite(preds).all():
        raise ValueError("Model returned NaN/inf in predictions; backtest stopped")
    if preds.shape[0] != y_test.shape[0]:
        raise ValueError(f"Prediction/target count mismatch: preds={preds.shape}, y_test={y_test.shape}")
    if preds.ndim == 1 and steps_ahead != 1:
        raise ValueError(f"Model returned a 1D forecast, but --steps={steps_ahead}")
    if preds.ndim > 1 and preds.shape[1] != steps_ahead:
        raise ValueError(f"Model returned {preds.shape[1]} step(s), expected --steps={steps_ahead}")

    preds_inv = inverse_scale_target(preds, scaler, target_col_idx)
    y_test_inv = inverse_scale_target(y_test, scaler, target_col_idx)
    preds_trade = preds_inv[:, 0] if preds_inv.ndim > 1 else preds_inv.reshape(-1)
    y_test_trade = y_test_inv[:, 0] if y_test_inv.ndim > 1 else y_test_inv.reshape(-1)
    execution_prices = inverse_scale_target(X_test[:, -1, target_col_idx], scaler, target_col_idx).reshape(-1)
    validate_price_series(
        {
            "preds_trade": preds_trade,
            "y_test_trade": y_test_trade,
            "execution_prices": execution_prices,
        }
    )

    # Trading-horizon metrics: virtual trader uses the first forecast step.
    acc, mape = calculate_accuracy(y_test_trade, preds_trade)
    mse, r2, evs = calculate_additional_metrics(y_test_trade, preds_trade)
    print(
        f"\nBacktest Metrics (trading horizon):\n"
        f"  Accuracy: {acc:.2f}%\n"
        f"  MAPE: {mape:.2f}%\n"
        f"  MSE: {mse:.4f}\n"
        f"  R2: {r2:.4f}\n"
    )

    trading_result = run_virtual_trader(
        execution_prices=execution_prices,
        preds_trade=preds_trade,
        initial_balance=args.start_balance,
        commission=args.commission,
        slippage_pct=args.slippage_pct,
        filter_threshold=args.signal_threshold_pct,
    )
    balance = trading_result["balance"]
    trade_history = trading_result["trade_history"]
    trade_profits = trading_result["trade_profits"]
    max_drawdown = trading_result["max_drawdown"]
    min_profit = trading_result["min_profit"]
    max_profit = trading_result["max_profit"]

    net_profit = balance - args.start_balance
    num_trades = len([t for t in trade_history if t["type"].startswith("SELL")])
    avg_profit = float(np.mean(trade_profits)) if trade_profits else 0.0
    total_fees = sum(float(t.get("fee", 0.0)) for t in trade_history)

    print("\n=== Virtual trading result ===")
    print(f"Start: {args.start_balance} USDT")
    print(f"Final: {balance:.2f} USDT")
    print(f"Net profit: {net_profit:.2f} USDT")
    print(f"Number of trades: {num_trades}")
    print(f"Average profit/loss per trade: {avg_profit:.4f} USDT")
    max_profit_display = f"{max_profit:.4f}" if max_profit is not None else "N/A"
    min_profit_display = f"{min_profit:.4f}" if min_profit is not None else "N/A"
    print(f"Max profit per trade: {max_profit_display} USDT")
    print(f"Min profit/loss per trade: {min_profit_display} USDT")
    print(f"Max drawdown: {max_drawdown:.2f} USDT")
    if scaler_warnings:
        print("\nWarnings:")
        for warning in scaler_warnings:
            print(f"  - {warning}")

    report_path = args.report or str(Path(args.output).with_suffix(".json"))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "inputs": {
            "data": args.data,
            "model": args.model,
            "scaler": args.scaler,
            "window": window_size,
            "steps": steps_ahead,
            "start_balance": args.start_balance,
            "commission": args.commission,
            "slippage_pct": args.slippage_pct,
            "signal_threshold_pct": args.signal_threshold_pct,
        },
        "schema": {
            "rows": int(len(df)),
            "sequence_count": int(len(X)),
            "test_sequence_count": int(len(X_test)),
            "feature_count": int(len(feature_columns)),
            "feature_columns": feature_columns,
            "scaler_n_features_in": int(scaler.n_features_in_),
            "model_input_shape": str(model.input_shape),
            "model_output_shape": str(model.output_shape),
        },
        "metrics": {
            "accuracy_pct": float(acc),
            "mape_pct": float(mape),
            "mse": float(mse),
            "r2": float(r2),
            "explained_variance": float(evs),
        },
        "trading": {
            "final_balance": float(balance),
            "net_profit": float(net_profit),
            "num_trades": int(num_trades),
            "avg_profit": float(avg_profit),
            "total_fees": float(total_fees),
            "max_profit": None if max_profit is None else float(max_profit),
            "min_profit": None if min_profit is None else float(min_profit),
            "max_drawdown": float(max_drawdown),
            "trade_history": trade_history,
        },
        "warnings": scaler_warnings,
    }
    write_report(report_path, report)
    print(f"JSON report saved to {report_path}")

    plt.figure(figsize=(15, 6))
    plt.plot(y_test_trade, label="True future close")
    plt.plot(execution_prices, label="Execution close")
    plt.plot(preds_trade, label="Predicted")
    buy_label_used = False
    sell_label_used = False
    for trade in trade_history:
        if trade["type"] == "BUY":
            plt.scatter(
                trade["idx"],
                trade["price"],
                color="g",
                marker="^",
                s=100,
                label="BUY" if not buy_label_used else "",
            )
            buy_label_used = True
        elif trade["type"].startswith("SELL"):
            plt.scatter(
                trade["idx"],
                trade["price"],
                color="r",
                marker="v",
                s=100,
                label="SELL" if not sell_label_used else "",
            )
            sell_label_used = True
    plt.legend()
    plt.title("Backtest: Trades & Price")
    plt.grid(True)
    plt.savefig(args.output)
    plt.close()

    print(f"Trades chart saved to {args.output}")


if __name__ == "__main__":
    main()
