import os
import glob
import re
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent)
OUT_REPORT = os.path.join(BASE_DIR, "SUMMARY_REPORT.md")

def extract_metrics_from_log(log_path):
    metrics = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Symbol:"):
                metrics["symbol"] = line.split(":", 1)[1].strip()
            elif line.startswith("MAE:"):
                metrics["mae"] = line.split(":", 1)[1].strip()
            elif line.startswith("RMSE:"):
                metrics["rmse"] = line.split(":", 1)[1].strip()
            elif line.startswith("MSE:"):
                metrics["mse"] = line.split(":", 1)[1].strip()
            elif line.startswith("R2 Score:"):
                metrics["r2"] = line.split(":", 1)[1].strip()
            elif line.startswith("Explained Variance:"):
                metrics["evs"] = line.split(":", 1)[1].strip()
            elif line.startswith("Accuracy:"):
                metrics["accuracy"] = line.split(":", 1)[1].replace("%", "").strip()
            elif line.startswith("MAPE:"):
                metrics["mape"] = line.split(":", 1)[1].replace("%", "").strip()
            elif line.startswith("Category:"):
                metrics["category"] = line.split(":", 1)[1].strip()
            elif line.startswith("Date and time:"):
                metrics["date"] = line.split(":", 1)[1].strip()
            elif "lstm" in os.path.basename(log_path):
                metrics["model"] = "LSTM"
            elif "gru" in os.path.basename(log_path):
                metrics["model"] = "GRU"
    # Fallback: infer the model from the path if not detected above
    if "model" not in metrics:
        if "lstm" in log_path.lower():
            metrics["model"] = "LSTM"
        elif "gru" in log_path.lower():
            metrics["model"] = "GRU"
        else:
            metrics["model"] = "both"
    return metrics

# --- Collect logs ---
log_files = glob.glob(os.path.join(BASE_DIR, "**", "*_log_*.txt"), recursive=True)
all_rows = []
for log_path in log_files:
    m = extract_metrics_from_log(log_path)
    if m.get("symbol"):
        all_rows.append(m)

# --- Sort for readability ---
all_rows.sort(key=lambda x: (x["symbol"], x["model"], x.get("date", "")))

# --- Build the table ---
report = "# LSTM vs GRU Model Comparison on Financial Data\n\n"
report += "## Metrics Summary by Coin\n\n"
report += "| Coin | Model | Category | Date | MAE | RMSE | MSE | R2 Score | Accuracy (%) | MAPE (%) | Explained Variance |\n"
report += "|--------|--------|-----------|---------------------|-------|-------|-------|----------|--------------|----------|--------------------|\n"
for r in all_rows:
    report += (
        f"| {r.get('symbol','')} | {r.get('model','')} | {r.get('category','')} | {r.get('date','')} "
        f"| {r.get('mae','')} | {r.get('rmse','')} | {r.get('mse','')} | {r.get('r2','')} "
        f"| {r.get('accuracy','')} | {r.get('mape','')} | {r.get('evs','')} |\n"
    )

# --- Add plots ---
def rel_path(full_path):
    # Use paths relative to BASE_DIR when embedding images
    return os.path.relpath(full_path, BASE_DIR)

report += "\n## Plots: Prediction vs Actual\n\n"
plot_files = glob.glob(os.path.join(BASE_DIR, "**", "*_prediction_vs_actual_*.png"), recursive=True)
for plot in plot_files:
    caption = os.path.basename(plot)
    report += f"![{caption}]({rel_path(plot)})\n\n"

# --- Attention plots (if any) ---
att_files = glob.glob(os.path.join(BASE_DIR, "**", "*_attention_*.png"), recursive=True)
if att_files:
    report += "\n## Attention Plots\n\n"
    for plot in att_files:
        caption = os.path.basename(plot)
        report += f"![{caption}]({rel_path(plot)})\n\n"

# --- Example experiment parameters ---
report += "\n---\n"
report += "## Experiment Settings\n"
report += "- Window: 30\n"
report += "- Steps ahead: 3\n"
report += "- Batch size: 128\n"
report += "- Epochs: 200\n"
report += "- Dropout: 0.3\n"
report += "- Bidirectional + MultiHeadAttention\n\n"

report += "This report is generated automatically. To refresh it, re-run the script after new training runs.\n"

# --- Save ---
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print(f"✅ Markdown report saved: {OUT_REPORT}")
