import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from utils.prep_utils import scale_data, create_sequences, train_val_test_split, augment_data
from utils.features import make_features
from utils.data_utils import load_data, split_data, clean_dataframe
from utils.metrics import calculate_accuracy, calculate_additional_metrics
from utils.telegram_utils import send_telegram_message
from utils.plots import plot_training_history, plot_attention_weights, plot_true_vs_pred


parser = argparse.ArgumentParser(description="Backtest ML model on historical data")
parser.add_argument("--data", required=True, help="CSV file to test on")
parser.add_argument("--model", required=True, help="Model file (.keras/.h5)")
parser.add_argument("--scaler", required=True, help="Scaler file (.pkl)")
parser.add_argument("--window", type=int, default=60, help="Window size")
parser.add_argument("--steps", type=int, default=1, help="Prediction horizon")
parser.add_argument("--output", default="backtest_result.png", help="Where to save the chart")
parser.add_argument("--start-balance", type=float, default=1000, help="Starting balance in USDT")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data)
df = df.select_dtypes(include=[np.number])  # numeric columns only
if 'close' in df.columns:
    target_col_idx = df.columns.get_loc('close')
else:
    target_col_idx = 0

if args.scaler.endswith(".pkl"):
    scaler = joblib.load(args.scaler)
    scaled_data = scaler.transform(df.values)
else:
    raise ValueError("scaler must be a .pkl file (joblib)")

window_size = args.window
steps_ahead = args.steps
X, y = create_sequences(scaled_data, window_size, steps_ahead, target_col_idx=target_col_idx)
_, _, X_test, _, _, y_test = split_data(X, y, val_size=0.0, test_size=0.1)

model = load_model(args.model)
preds = model.predict(X_test)
target_min = scaler.data_min_[target_col_idx]
target_max = scaler.data_max_[target_col_idx]
preds_inv = preds * (target_max - target_min) + target_min
y_test_inv = y_test * (target_max - target_min) + target_min
df = pd.read_csv(args.data)
df = clean_dataframe(df, fill_method="drop")

# Model metrics
acc, mape = calculate_accuracy(y_test_inv, preds_inv)
mse, r2, evs = calculate_additional_metrics(y_test_inv, preds_inv)
print(f"\nBacktest Metrics:\n  Accuracy: {acc:.2f}%\n  MAPE: {mape:.2f}%\n  MSE: {mse:.4f}\n  R2: {r2:.4f}\n")

# ==== Virtual trader ====
def to_scalar(val):
    if isinstance(val, (np.ndarray, list)):
        return float(np.array(val).flatten()[0])
    return float(val)

# ==== Virtual trader ====
initial_balance = to_scalar(args.start_balance)
balance = initial_balance
position = 0  # 0 — out of the market, 1 — Long
entry_price = None
commission = 0.0001  # 0.1% commission
trade_history = []
trade_profits = []
max_drawdown = 0
balance_peak = initial_balance
min_profit = None
max_profit = None

filter_threshold = 0.05 # ignore prediction changes below this threshold (noise)

for i in range(1, len(preds_inv)):
    price_now = to_scalar(y_test_inv[i])
    price_prev = to_scalar(y_test_inv[i-1])
    pred_now = to_scalar(preds_inv[i])
    pred_prev = to_scalar(preds_inv[i-1])
    delta_pred = pred_now - pred_prev

    delta_pred_val = to_scalar(delta_pred)
    pred_prev_val = to_scalar(pred_prev)

    if abs(delta_pred_val) / abs(pred_prev_val) * 100 < filter_threshold:
        continue

    # BUY signal
    if delta_pred_val > 0 and position == 0:
        entry_price = price_now
        position = 1
        balance -= entry_price * commission  # entry commission
        balance = to_scalar(balance)
        trade = {"type": "BUY", "price": entry_price, "balance": balance, "idx": i}
        trade_history.append(trade)
    # SELL signal
    elif delta_pred_val < 0 and position == 1:
        exit_price = price_now
        profit = to_scalar(exit_price - entry_price)
        balance += profit
        balance -= exit_price * commission  # exit commission
        balance = to_scalar(balance)
        trade = {
            "type": "SELL",
            "price": exit_price,
            "balance": balance,
            "profit": profit,
            "idx": i
        }
        trade_history.append(trade)
        trade_profits.append(profit)
        # update statistics
        if min_profit is None or profit < to_scalar(min_profit):
            min_profit = profit
        if max_profit is None or profit > to_scalar(max_profit):
            max_profit = profit
        if balance > balance_peak:
            balance_peak = balance
        drawdown = balance_peak - balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        position = 0

# if a position is still open, close it at the last price
if position == 1:
    final_price = to_scalar(y_test_inv[-1])
    profit = to_scalar(final_price - entry_price)
    balance += profit
    balance -= final_price * commission
    balance = to_scalar(balance)
    trade = {
        "type": "SELL_END",
        "price": final_price,
        "balance": balance,
        "profit": profit,
        "idx": len(y_test_inv) - 1
    }
    trade_history.append(trade)
    trade_profits.append(profit)
    if min_profit is None or profit < to_scalar(min_profit):
        min_profit = profit
    if max_profit is None or profit > to_scalar(max_profit):
        max_profit = profit
    if balance > balance_peak:
        balance_peak = balance
    drawdown = balance_peak - balance
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# Compute final metrics
    net_profit = to_scalar(balance - initial_balance)
    num_trades = len([t for t in trade_history if t["type"].startswith("SELL")])
    avg_profit = np.mean(trade_profits) if trade_profits else 0


    # update final variables
    #balance = balance_val
    #balance_peak = balance_peak_val


# === Print statistics ===


def to_float(val):
    net_profit = to_scalar(balance - initial_balance)
    # Generic: extracts a number from an array or returns the value as is
    if isinstance(val, np.ndarray):
        return float(val.flatten()[0])  # take the first element
    return float(val)
print("\n=== Virtual trading results ===")
print(f"Start: {initial_balance} USDT")
print(f"Final: {to_float(balance):.2f} USDT")
print(f"Net profit: {to_float(net_profit):.2f} USDT")
print(f"Number of trades: {num_trades}")
print(f"Average profit/loss per trade: {avg_profit:.4f} USDT")
if max_profit is not None:
    print(f"Max profit per trade: {to_float(max_profit):.4f} USDT")
else:
    print("Max profit per trade: - (no trades)")

if min_profit is not None:
    print(f"Min profit/loss per trade: {to_float(min_profit):.4f} USDT")
else:
    print("Min profit/loss per trade: - (no trades)")


print(f"Max drawdown: {to_float(max_drawdown):.2f} USDT")


# === Price and trades chart ===
plt.figure(figsize=(15, 6))
plt.plot(y_test_inv.flatten(), label='True price')
plt.plot(preds_inv.flatten(), label='Predicted')
for trade in trade_history:
    if trade["type"] == "BUY":
        plt.scatter(trade["idx"], trade["price"], color='g', marker='^', s=100, label='BUY' if trade_history.index(trade) == 0 else "")
    elif trade["type"].startswith("SELL"):
        plt.scatter(trade["idx"], trade["price"], color='r', marker='v', s=100, label='SELL' if trade_history.index(trade) == 1 else "")
plt.legend()
plt.title("Backtest: Trades & Price")
plt.grid(True)
plt.savefig(args.output)
plt.close()

print(f"Trades chart saved to {args.output}")
