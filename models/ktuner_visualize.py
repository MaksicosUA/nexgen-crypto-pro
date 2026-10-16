import yaml
import keras_tuner as kt
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Load the list of coins from config.yaml ===
def get_symbols_from_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    symbols = cfg.get("symbols", {})
    return [sym for sym, enabled in symbols.items() if enabled is True or enabled == "true"]

# === Visualization for a single coin ===
def plot_ktuner_results(project_dir, symbol):
    tuner = kt.BayesianOptimization(
        lambda hp: None,  # dummy, unused
        objective='val_loss',
        max_trials=1,
        directory=project_dir,
        project_name=f'{symbol}_gru'
    )
    tuner.reload()  # load saved results
    trials = tuner.oracle.trials
    trial_data = []
    for trial_id, trial in trials.items():
        val_loss = trial.metrics.get_best_value('val_loss')
        params = trial.hyperparameters.values
        params['val_loss'] = val_loss
        trial_data.append(params)
    df = pd.DataFrame(trial_data)
    # Plot val_loss against each hyperparameter
    for col in df.columns:
        if col != 'val_loss':
            plt.figure()
            plt.scatter(df[col], df['val_loss'])
            plt.xlabel(col)
            plt.ylabel('val_loss')
            plt.title(f'{symbol}: {col} vs val_loss')
            plt.grid(True)
            # Save plots into a folder named after the coin:
            outdir = os.path.join(project_dir, f'{symbol}_gru', 'visualization')
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f'{symbol}_{col}_vs_val_loss.png'))
            plt.close()
    print(f"✅ Visualization created for {symbol}")

if __name__ == "__main__":
    # Project and config paths (relative to the repository root)
    project_dir = os.path.join(BASE_DIR, "ktuner_dir")
    config_path = os.path.join(BASE_DIR, "configs", "config.yaml")

    # Get all enabled coins
    symbol_list = get_symbols_from_config(config_path)
    print(f"🟢 Visualizing coins: {symbol_list}")

    # Build plots for all coins
    for symbol in symbol_list:
        try:
            plot_ktuner_results(project_dir, symbol)
        except Exception as e:
            print(f"⚠️ Failed to build plots for {symbol}: {e}")
