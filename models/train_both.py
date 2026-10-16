# 📊 LSTM Training Script for Financial Data (Updated Structure 1.2) (Refactored 1.3)
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
import yaml
import glob
import traceback
import gc
import argparse
import shutil
from datetime import datetime
from itertools import product

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from dotenv import load_dotenv

# === Imports from utils ===
from utils.prep_utils import (
    scale_data,
    scale_data_train_only,
    create_sequences,
    train_val_test_split,
    augment_data,
)
from utils.targets import (
    DEFAULT_RETURN_SCALE,
    LEVEL_TARGET,
    LOG_RETURN_TARGET,
    KNOWN_TARGETS,
    build_log_return_targets,
    extract_hyperparameters,
    log_returns_to_prices,
    write_target_metadata,
    zero_return_prediction,
)
from utils.features import make_features
from utils.data_utils import load_data, split_data
from utils.metrics import calculate_accuracy, calculate_additional_metrics
from utils.telegram_utils import send_telegram_message
from utils.plots import plot_training_history, plot_attention_weights, plot_true_vs_pred


print("GPUs:", tf.config.list_physical_devices('GPU'))

# === 🎲 Set SEED for reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# === ⚡ FP16: enable Mixed Precision (FP16 speed-up) ===
mixed_precision.set_global_policy('mixed_float16')
print(f"🚀 Mixed Precision enabled: {mixed_precision.global_policy()}")

# === 🧹 TensorFlow memory cleanup ===
def clean_memory():
    """Clear Keras/TensorFlow memory and run the garbage collector."""
    tf.keras.backend.clear_session()
    gc.collect()

# # GPU check and setup
gpus = tf.config.experimental.list_physical_devices('GPU')  # for memory growth setup
if gpus:
    print("✅ GPU available")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🔧 Dynamic GPU memory allocation (memory growth) enabled")
    except RuntimeError as e:
        print(f"❌ Failed to set memory growth: {e}")
else:
    print("⚠️ No GPU found, falling back to CPU")

# === Project paths and config ===
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--file', type=str, default=None)
args = parser.parse_args()
CONFIG_PATH = args.config or os.path.join(BASE_DIR, "configs", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    FULL_CONFIG = yaml.safe_load(f)

CONFIG = {
    'num_layers': 2,
    'units': 256,
    'dropout_rate': 0.3,
    'batch_size': 128,
    'epochs': 200,
    'patience_early_stop': 20,
    'patience_reduce_lr': 10,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6
}

CONFIG.update(FULL_CONFIG.get("model_config", {}))
# --- Cast to float ---
for key in ["min_lr", "reduce_lr_factor", "dropout_rate"]:
    if key in CONFIG:
        try:
            CONFIG[key] = float(CONFIG[key])
        except Exception as e:
            print(f"⚠️ Could not convert parameter {key} to float: {CONFIG[key]}, error: {e}")

RETRAIN_EXISTING = FULL_CONFIG.get("global_config", {}).get(
    "retrain_existing",
    FULL_CONFIG.get("model_config", {}).get("retrain_existing", False)
)
CUSTOM_PATHS = FULL_CONFIG.get("paths", {})
ERROR_LOG_PATH = CUSTOM_PATHS.get("error_log", os.path.join(BASE_DIR, "logs", "errors.log"))
TRAINED_LOG_PATH = CUSTOM_PATHS.get("trained_log", os.path.join(BASE_DIR, "logs", "trained_log_hybrid.txt"))
TARGET_KIND = FULL_CONFIG.get("global_config", {}).get("target", LEVEL_TARGET)
if TARGET_KIND not in KNOWN_TARGETS:
    raise ValueError(
        f"❌ Unknown target: target={TARGET_KIND!r}. Available: {KNOWN_TARGETS}"
    )
RETURN_SCALE = float(
    FULL_CONFIG.get("global_config", {}).get("return_scale", DEFAULT_RETURN_SCALE)
)

GLOBAL_STEPS_AHEAD = FULL_CONFIG.get("global_config", {}).get("steps_ahead", 1)
GLOBAL_WINDOW_SIZE = FULL_CONFIG.get("global_config", {}).get("window_size", 30)
LOG_ERRORS = FULL_CONFIG.get("global_config", {}).get("log_errors_separately", True)
VAL_SIZE = FULL_CONFIG.get("global_config", {}).get("val_size", 0.15)
TEST_SIZE = FULL_CONFIG.get("global_config", {}).get("test_size", 0.15)
ERROR_LOG_PATH = os.path.join(BASE_DIR, "logs", "errors.log") if LOG_ERRORS else None

# === Dedicated folder for hybrid (LSTM + GRU) models ===
MODELS_BASE_DIR = os.path.join(BASE_DIR, "models/saved_models", "hybrid_saved_model")
os.makedirs(MODELS_BASE_DIR, exist_ok=True)

# === ✉️ Telegram setup ===
send_telegram_message=send_telegram_message


# === 🔨 Model builder functions ===
def build_lstm(input_shape, num_layers=1, units=64, dropout_rate=0.3, steps_ahead=1, 
               bidirectional=True, use_attention=False):
    """
    Build a stacked LSTM + GRU model (with optional multi-head attention).
    Returns the compiled model.
    """
    attention_weights = {}
    inputs = Input(shape=input_shape)
    x = inputs
    # Add LSTM layer
    lstm_layer = LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
    x = Bidirectional(lstm_layer)(x) if bidirectional else lstm_layer(x)
    x = Dropout(dropout_rate)(x)
    # Add GRU layer
    gru_layer = GRU(units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
    x = Bidirectional(gru_layer)(x) if bidirectional else gru_layer(x)
    x = Dropout(dropout_rate)(x)
    # Optional: multi-head attention
    if use_attention:
        x = LayerNormalization(epsilon=1e-6)(x)
        attention_layer = MultiHeadAttention(num_heads=2, key_dim=units)
        attn_output, attn_scores = attention_layer(x, x, return_attention_scores=True)
        attention_weights['scores'] = attn_scores
        x = Dropout(dropout_rate)(attn_output)
        x = Dense(units, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(steps_ahead, dtype='float32')(x)
        model = Model(inputs, outputs)
        optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam())
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    else:
        outputs = Dense(steps_ahead, dtype='float32')(x)
        model = Model(inputs, outputs)
        optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam())
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Store attention weights (if applicable)
    model.attention_weights = attention_weights
    return model

def build_new_model(input_shape, steps_ahead):
    """
    Build a new hybrid LSTM/GRU model from the global configuration.
    """
    return build_lstm(
        input_shape,
        num_layers=CONFIG.get('num_layers', 1),
        units=CONFIG.get('units', 64),
        dropout_rate=CONFIG.get('dropout_rate', 0.3),
        steps_ahead=steps_ahead,
        bidirectional=True,
        use_attention=True
    )

# === 🔮 Train and predict for a single dataset ===
def train_and_predict(df: pd.DataFrame, symbol: str, steps_ahead: int, window_size: int):
    try:
        send_telegram_message(f"🚀 Hybrid model training started for {symbol} (data rows: {df.shape[0]})")
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        if 'date' in df.columns:
            df.drop('date', axis=1, inplace=True)
        df = df.select_dtypes(include=[np.number])
        target_col = 'close' if 'close' in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
        if target_col and df.columns[0] != target_col:
            cols = [target_col] + [col for col in df.columns if col != target_col]
            df = df.loc[:, cols].copy()
        df = df.astype('float32')
        print("FINAL FEATURES for scaler/model:", df.columns.tolist())
        if len(df) < window_size + steps_ahead:
            print(f"⚠️ Not enough data for {symbol} (rows: {len(df)}) for window={window_size} and horizon={steps_ahead}, skipping...")
            return
        data_values = df.values
        model = None
        scaler = None
        loaded_existing_model = False
        category = None
        category_list = ["95-100%", "75-95%", "50-75%", "25-50%", "0-25%", "below_0%"]
        try:
            for cat in category_list:
                category_path = os.path.join(MODELS_BASE_DIR, cat)
                os.makedirs(category_path, exist_ok=True)
                model_filename = f"{symbol}_hybrid.keras"
                scaler_filename = f"{symbol}_hybrid_scaler.pkl"
                model_path = os.path.join(category_path, model_filename)
                scaler_path = os.path.join(category_path, scaler_filename)
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    print(f"📦 Found existing model for {symbol}: {model_path}")
                    model = load_model(model_path)
                    scaler = joblib.load(scaler_path)
                    loaded_existing_model = True
                    category = cat
                    break
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            model = None
            scaler = None
            loaded_existing_model = False
            category = None

        def fit_fresh_scaler():
            # Fit on training rows only: a full fit_transform over the whole
            # dataset exposed val and test to the scaler, which made holdout metrics
            # look better than they really were.
            scaled, fitted, train_rows = scale_data_train_only(
                data_values, window_size, steps_ahead, VAL_SIZE, TEST_SIZE
            )
            print(
                f"🔒 Scaler fitted on the first {train_rows} of {len(data_values)} rows "
                "(val and test excluded)"
            )
            return scaled, fitted

        if scaler is not None:
            try:
                scaled_data = scaler.transform(data_values)
            except Exception as e:
                print(f"⚠️ Could not apply existing scaler for {symbol}: {e}. A new scaler will be created.")
                model = None
                scaler = None
                loaded_existing_model = False
                category = None
                scaled_data, scaler = fit_fresh_scaler()
        else:
            scaled_data, scaler = fit_fresh_scaler()

        X, y_level = create_sequences(scaled_data, window_size, steps_ahead)
        if X.size == 0 or y_level.size == 0:
            print(f"⚠️ Cannot build sequences for {symbol}, skipping...")
            return

        close_prices = df[target_col].to_numpy(dtype=np.float64)
        returns_target, entry_prices = build_log_return_targets(
            close_prices, window_size, steps_ahead, RETURN_SCALE
        )
        if len(entry_prices) != len(X):
            raise ValueError(
                f"Targets and sequences out of sync: {len(entry_prices)} vs {len(X)}"
            )

        if TARGET_KIND == LOG_RETURN_TARGET:
            y = returns_target
            print(f"🎯 Target: log return in % (scale={RETURN_SCALE})")
        else:
            y = y_level
            print("🎯 Target: absolute price level (legacy)")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, val_size=VAL_SIZE, test_size=TEST_SIZE)
        _, _, entry_test, _, _, _ = split_data(
            entry_prices, entry_prices, val_size=VAL_SIZE, test_size=TEST_SIZE
        )
        if len(X_train) > 0:
            print("X.shape:", X.shape)
            print("y.shape:", y.shape)
            print("y[:20]:", y[:20])
            print("min/max y:", np.min(y), np.max(y))
            print("X_train.shape:", X_train.shape, "y_train.shape:", y_train.shape)
            print("X_val.shape:", X_val.shape, "y_val.shape:", y_val.shape)
            print("X_test.shape:", X_test.shape, "y_test.shape:", y_test.shape)
            X_aug = augment_data(X_train, noise_level=0.01)
            y_aug = y_train.copy()
            X_train = np.concatenate([X_train, X_aug])
            y_train = np.concatenate([y_train, y_aug])

        if model is None:
            print("ℹ️ No model found, creating a new one.")
            model = build_new_model(input_shape=(X_train.shape[1], X_train.shape[2]), steps_ahead=steps_ahead)

        history = None
        should_train = (not loaded_existing_model) or RETRAIN_EXISTING
        if should_train:
            if loaded_existing_model and RETRAIN_EXISTING:
                msg = f"🔄 Fine-tuning existing model for {symbol}..."
                print(msg)
                send_telegram_message(msg)
            elif not loaded_existing_model:
                msg = f"🆕 Training new model for {symbol}..."
                print(msg)
                send_telegram_message(msg)
            else:
                msg = f"🚀 Starting training/fine-tuning for {symbol}..."
                print(msg)
                send_telegram_message(msg)

            print(f"[DEBUG] min_lr type: {type(CONFIG['min_lr'])}, value: {CONFIG['min_lr']}")

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=CONFIG['patience_early_stop'], restore_best_weights=True),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    patience=CONFIG['patience_reduce_lr'],
                    factor=CONFIG['reduce_lr_factor'],
                    min_lr=float(CONFIG['min_lr'])
                )
            ]
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG['epochs'],
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                callbacks=callbacks,
                verbose=1
            )
            print(f"✅ Model training for {symbol} finished. Epochs: {len(history.history['loss'])}")
        else:
            msg = f"⚡ Model for {symbol} loaded without fine-tuning."
            print(msg)
            send_telegram_message(msg)

        # Recompute metrics
        preds = model.predict(X_test)
        if TARGET_KIND == LOG_RETURN_TARGET:
            preds_inverse = log_returns_to_prices(entry_test, preds, RETURN_SCALE)
            y_test_inverse = log_returns_to_prices(entry_test, y_test, RETURN_SCALE)
        else:
            target_min = scaler.data_min_[0]
            target_max = scaler.data_max_[0]
            preds_inverse = preds * (target_max - target_min) + target_min
            y_test_inverse = y_test * (target_max - target_min) + target_min

        persistence_prices = log_returns_to_prices(
            entry_test, zero_return_prediction(entry_test, steps_ahead), RETURN_SCALE
        )
        pers_accuracy, pers_mape = calculate_accuracy(y_test_inverse, persistence_prices)
        model_acc_prev, model_mape_prev = calculate_accuracy(y_test_inverse, preds_inverse)
        direction_true = np.sign(y_test_inverse[:, 0] - entry_test)
        direction_pred = np.sign(preds_inverse[:, 0] - entry_test)
        directional_accuracy = float(np.mean(direction_true == direction_pred) * 100)
        print("\n=== Comparison with naive forecast ===")
        print(f"  persistence: MAPE {pers_mape:.4f}%   accuracy {pers_accuracy:.2f}%")
        print(f"  model:       MAPE {model_mape_prev:.4f}%   accuracy {model_acc_prev:.2f}%")
        print(f"  model directional accuracy: {directional_accuracy:.2f}%")
        if model_mape_prev >= pers_mape:
            print("  ⚠️ Model does NOT beat persistence — a low MAPE proves nothing.")
        print()
        mae = mean_absolute_error(y_test_inverse, preds_inverse)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, preds_inverse))
        mse, r2, evs = calculate_additional_metrics(y_test_inverse, preds_inverse)
        accuracy, mape = calculate_accuracy(y_test_inverse, preds_inverse)
        print(f"[DEBUG] {symbol}: accuracy={accuracy} ({type(accuracy)})")
        try:
            accuracy = float(accuracy)
            if np.isnan(accuracy):
                raise ValueError("accuracy is NaN")
        except Exception as e:
            print(f"⚠️ Invalid accuracy type or value: {accuracy}. Error: {e}")
            send_telegram_message(f"⚠️ Error: invalid accuracy ({accuracy}) for {symbol}. Skipping file.")
            return

        # Category
        if accuracy < 0:
            category = "below_0%"
        elif accuracy < 25:
            category = "0-25%"
        elif accuracy < 50:
            category = "25-50%"
        elif accuracy < 75:
            category = "50-75%"
        elif accuracy < 95:
            category = "75-95%"
        else:
            category = "95-100%"

        category_path = os.path.join(MODELS_BASE_DIR, category)
        os.makedirs(category_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"{symbol}_hybrid.keras"
        scaler_filename = f"{symbol}_hybrid_scaler.pkl"
        model_path = os.path.join(category_path, model_filename)
        scaler_path = os.path.join(category_path, scaler_filename)
        print(f"[DEBUG] Accuracy for {symbol}: {accuracy} (type: {type(accuracy)})")

        # --- Move old artifacts to the archive ---
        category_names = ["95-100%", "75-95%", "50-75%", "25-50%", "0-25%", "below_0%"]
        model_filename = f"{symbol}_hybrid.keras"
        scaler_filename = f"{symbol}_hybrid_scaler.pkl"

        for cat in category_names:
            if cat == category:
                continue

            cat_dir = os.path.join(MODELS_BASE_DIR, cat)
            archive_dir = os.path.join(cat_dir, "archive")
            try:
                os.makedirs(archive_dir, exist_ok=True)
            except Exception as e:
                print(f"❗️ Failed to create archive folder: {archive_dir} | Error: {e}")
                continue

            old_model_path = os.path.join(cat_dir, model_filename)
            archive_model_path = os.path.join(archive_dir, f"{symbol}_hybrid_{timestamp}.keras")
            if os.path.exists(old_model_path):
                try:
                    shutil.move(old_model_path, archive_model_path)
                    print(f"📦 Model {old_model_path} moved to archive: {archive_model_path}")
                except Exception as e:
                    print(f"❗️ Failed to move model: {old_model_path} → {archive_model_path} | {e}")
            else:
                print(f"🔸 No model to move: {old_model_path}")

            old_scaler_path = os.path.join(cat_dir, scaler_filename)
            archive_scaler_path = os.path.join(archive_dir, f"{symbol}_hybrid_scaler_{timestamp}.pkl")
            if os.path.exists(old_scaler_path):
                try:
                    shutil.move(old_scaler_path, archive_scaler_path)
                    print(f"📦 Scaler {old_scaler_path} moved to archive: {archive_scaler_path}")
                except Exception as e:
                    print(f"❗️ Failed to move scaler: {old_scaler_path} → {archive_scaler_path} | {e}")
            else:
                print(f"🔸 No scaler to move: {old_scaler_path}")

            log_files = glob.glob(os.path.join(cat_dir, f"{symbol}_log_*.txt"))
            if log_files:
                for log_file in log_files:
                    try:
                        shutil.move(log_file, os.path.join(archive_dir, os.path.basename(log_file)))
                        print(f"📦 Log {log_file} moved to archive: {archive_dir}")
                    except Exception as e:
                        print(f"❗️ Failed to move log: {log_file} → {archive_dir} | {e}")
            else:
                print(f"🔸 No logs to move from: {cat_dir}")

            plot_files = glob.glob(os.path.join(cat_dir, f"{symbol}_*.png"))
            if plot_files:
                for plot_file in plot_files:
                    try:
                        shutil.move(plot_file, os.path.join(archive_dir, os.path.basename(plot_file)))
                        print(f"📦 Plot {plot_file} moved to archive: {archive_dir}")
                    except Exception as e:
                        print(f"❗️ Failed to move plot: {plot_file} → {archive_dir} | {e}")
            else:
                print(f"🔸 No plots to move from: {cat_dir}")

        # --- Archive the current model in the same category before overwriting ---
        if os.path.exists(model_path):
            same_cat_archive = os.path.join(category_path, "archive")
            os.makedirs(same_cat_archive, exist_ok=True)
            try:
                shutil.move(model_path, os.path.join(same_cat_archive, f"{symbol}_hybrid_{timestamp}.keras"))
                print(f"📦 Current model in the same category archived: {same_cat_archive}")
            except Exception as e:
                print(f"❗️ Failed to archive current model: {e}")
        if os.path.exists(scaler_path):
            same_cat_archive = os.path.join(category_path, "archive")
            os.makedirs(same_cat_archive, exist_ok=True)
            try:
                shutil.move(scaler_path, os.path.join(same_cat_archive, f"{symbol}_hybrid_scaler_{timestamp}.pkl"))
                print(f"📦 Current scaler in the same category archived: {same_cat_archive}")
            except Exception as e:
                print(f"❗️ Failed to archive current scaler: {e}")

        # Save the final model and scaler under a fixed name (active version)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        write_target_metadata(
            model_path,
            TARGET_KIND,
            window_size=window_size,
            steps_ahead=steps_ahead,
            return_scale=RETURN_SCALE,
            hyperparameters=extract_hyperparameters(CONFIG),
            extra={"model_type": "hybrid", "symbol": symbol},
        )
        print(f"🏷️ Training target recorded: {TARGET_KIND}")
        print(f"✅ Final model saved: {model_path}")

        # Telegram msg
        msg = (f"📊 Training finished for {symbol}\n"
               f"MAE: {mae:.4f}\n"
               f"RMSE: {rmse:.4f}\n"
               f"MSE: {mse:.4f}\n"
               f"R2: {r2:.4f}\n"
               f"Accuracy: {accuracy:.2f}%\n"
               f"MAPE: {mape:.2f}%\n"
               f"Category: {category.upper()}\n"
               f"Date: {timestamp}"
               )
        send_telegram_message(msg)

        if history is not None:
            plot_training_history(history, symbol, category_path, timestamp)
            plot_attention_weights(model, X_test, symbol, category_path, timestamp)
            plot_true_vs_pred(y_test_inverse, preds_inverse, symbol, category_path, timestamp)

        log_path = os.path.join(category_path, f"{symbol}_log_{timestamp}.txt")
        with open(log_path, 'w') as log_file:
            log_file.write(f"Date and time: {timestamp}\n")
            log_file.write(f"Symbol: {symbol}\n")
            log_file.write(f"Category: {category}\n")
            log_file.write(f"MAE: {mae:.4f}\n")
            log_file.write(f"RMSE: {rmse:.4f}\n")
            log_file.write(f"Accuracy: {accuracy:.2f}%\n")
            log_file.write(f"MAPE: {mape:.2f}%\n")
            log_file.write(f"MSE: {mse:.4f}\n")
            log_file.write(f"R2 Score: {r2:.4f}\n")
            log_file.write(f"Explained Variance: {evs:.4f}\n")
            if history is not None:
                log_file.write(f"Epochs: {len(history.history['loss'])}\n")
                log_file.write(f"Final Train Loss: {history.history['loss'][-1]:.4f}\n")
                if 'val_loss' in history.history:
                    log_file.write(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}\n")
            else:
                log_file.write("Epochs: 0 (model loaded without training)\n")
            log_file.write(f"Steps ahead: {steps_ahead}\n")
            log_file.write(f"Window size: {window_size}\n")
            log_file.write(f"Model CONFIG: {CONFIG}\n")

        clean_memory()

    except Exception as e:
        error_msg = f"⚠️ Error while processing {symbol}: {e}"
        print(error_msg)
        if ERROR_LOG_PATH:
            with open(ERROR_LOG_PATH, "a") as err_file:
                err_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {symbol}: {traceback.format_exc()}\n")
        send_telegram_message(error_msg)
        clean_memory()

def load_and_train_from_folder(folder_path: str, steps_ahead: int, window_size: int):
    trained_log_path = os.path.join(MODELS_BASE_DIR, "trained_log_hybrid.txt")
    trained_files = []
    if os.path.exists(trained_log_path):
        with open(trained_log_path, "r") as f:
            trained_files = [line.strip() for line in f if line.strip()]
    ignored_dir = os.path.join(MODELS_BASE_DIR, "ignored")
    os.makedirs(ignored_dir, exist_ok=True)
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename in trained_files:
            print(f"📦 {filename} was already used for training — moved to 'ignored'")
            try:
                os.rename(filepath, os.path.join(ignored_dir, filename))
            except Exception as e:
                print(f"⚠️ Could not move {filename} to ignored: {e}")
            continue
        if filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(filepath)
                symbol = os.path.splitext(filename)[0].replace('/', '_').replace('\\', '_').replace(' ', '_')
                print(f"📂 Processing file: {filename} (symbol: {symbol})")
                train_and_predict(df, symbol, steps_ahead=steps_ahead, window_size=window_size)
                with open(trained_log_path, "a") as f:
                    f.write(filename + "\n")
            except Exception as e:
                error_msg = f"⚠️ Error while processing file {filename}: {e}"
                print(error_msg)
                if ERROR_LOG_PATH:
                    with open(ERROR_LOG_PATH, "a") as err_file:
                        err_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FILE {filename}: {traceback.format_exc()}\n")
                send_telegram_message(error_msg)

def smart_train(source_path: str, steps_ahead: int = 1, window_size: int = 30):
    if os.path.isdir(source_path):
        print(f"📁 Found data folder: {source_path}. Processing all CSV files...")
        load_and_train_from_folder(source_path, steps_ahead, window_size)
    elif os.path.isfile(source_path):
        print(f"📄 Found data file: {source_path}. Training a model for this file...")
        df = pd.read_csv(source_path)
        symbol = os.path.splitext(os.path.basename(source_path))[0].replace('/', '_').replace('\\', '_').replace(' ', '_')
        train_and_predict(df, symbol, steps_ahead=steps_ahead, window_size=window_size)
    else:
        msg = f"❌ Specified path not found: {source_path}"
        print(msg)
        send_telegram_message(msg)
        return

DATA_PATH = os.path.join(BASE_DIR, "data", "data_base")
if __name__ == "__main__":
    source_path = args.file or DATA_PATH
    if not os.path.exists(source_path):
        error_message = f"❌ Data path not found: {source_path}"
        print(error_message)
        send_telegram_message(error_message)
    else:
        smart_train(source_path, steps_ahead=GLOBAL_STEPS_AHEAD, window_size=GLOBAL_WINDOW_SIZE)
