import subprocess
import yaml
import os
import sys

# === Resolve the project root regardless of the launch location ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to models and config
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Models to run one after another (can be extended)
MODELS_TO_RUN = [
    ("train_lstm.py", "LSTM"),
    #("train_gru.py", "GRU"),
    #("train_both.py", "Hybrid")   # Add this to run the ensemble
]

def run_model(script_name, model_tag):
    script_path = os.path.join(MODELS_DIR, script_name)
    print(f"\n==== Starting training for model {model_tag} ====")
    result = subprocess.run(
        [sys.executable, script_path, "--config", CONFIG_PATH],
        cwd=BASE_DIR,  # Working directory (project root)
    )
    if result.returncode != 0:
        print(f"❌ Failed to run {script_name}, code: {result.returncode}")
    else:
        print(f"✅ Training finished for {model_tag}\n")

if __name__ == "__main__":
    print("=== Starting sequential training of all models ===")
    for script, tag in MODELS_TO_RUN:
        run_model(script, tag)
    print("\n=== All models trained! ===")
