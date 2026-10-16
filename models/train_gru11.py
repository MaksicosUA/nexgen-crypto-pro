# 📊 GRU Training Script (Refactored)
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
from tensorflow import keras

import keras_tuner as kt


import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from dotenv import load_dotenv

# === Imports from utils ===
from utils.prep_utils import scale_data, create_sequences, train_val_test_split, augment_data
from utils.features import make_features
from utils.data_utils import load_data, split_data, clean_dataframe
from utils.metrics import calculate_accuracy, calculate_additional_metrics
from utils.telegram_utils import send_telegram_message
from utils.plots import plot_training_history, plot_attention_weights, plot_true_vs_pred

print("GPUs:", tf.config.list_physical_devices('GPU'))

# === SEED for reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# === Mixed Precision (FP16) ===
mixed_precision.set_global_policy('mixed_float16')
print(f"🚀 Mixed Precision enabled: {mixed_precision.global_policy()}")

def clean_memory():
    tf.keras.backend.clear_session()
    gc.collect()

# === GPU Memory Growth ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("✅ GPU available")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🔧 Dynamic memory allocation (memory growth) enabled")
    except RuntimeError as e:
        print(f"❌ Failed to set memory growth: {e}")
else:
    print("⚠️ GPU not found, falling back to CPU")

# === Project paths and config ===
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()
CONFIG_PATH = args.config or os.path.join(BASE_DIR, "configs", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Configuration file not found: {CONFIG_PATH}")
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
            print(f"⚠️ Failed to convert parameter {key} to float: {CONFIG[key]}, error: {e}")

RETRAIN_EXISTING = FULL_CONFIG.get("global_config", {}).get("retrain_existing", False)
CUSTOM_PATHS = FULL_CONFIG.get("paths", {})
ERROR_LOG_PATH = CUSTOM_PATHS.get("error_log", os.path.join(BASE_DIR, "logs", "errors.log"))
TRAINED_LOG_PATH = CUSTOM_PATHS.get("trained_log", os.path.join(BASE_DIR, "logs", "trained_log_gru.txt"))
GLOBAL_STEPS_AHEAD = FULL_CONFIG.get("global_config", {}).get("steps_ahead", 1)
GLOBAL_WINDOW_SIZE = FULL_CONFIG.get("global_config", {}).get("window_size", 30)
LOG_ERRORS = FULL_CONFIG.get("global_config", {}).get("log_errors_separately", True)
VAL_SIZE = FULL_CONFIG.get("global_config", {}).get("val_size", 0.15)
TEST_SIZE = FULL_CONFIG.get("global_config", {}).get("test_size", 0.15)
ERROR_LOG_PATH = os.path.join(BASE_DIR, "logs", "errors.log") if LOG_ERRORS else None

# === Dedicated folder for GRU models ===
MODELS_BASE_DIR = os.path.join(BASE_DIR, "models/saved_models", "gru_saved_model")
os.makedirs(MODELS_BASE_DIR, exist_ok=True)

# === Telegram ===

send_telegram_message=send_telegram_message



def build_gru_tuner(hp, input_shape, steps_ahead):
    model = keras.Sequential()
    num_layers = hp.Int('num_layers', 1, 3, default=CONFIG['num_layers'])
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', 32, 256, step=32, default=CONFIG['units'])
        return_sequences = (i < num_layers - 1)
        if i == 0:
            model.add(GRU(units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(GRU(units=units, return_sequences=return_sequences))
        dropout_rate = hp.Float(f'dropout_{i}', 0.0, 0.4, step=0.1, default=CONFIG['dropout_rate'])
        model.add(Dropout(dropout_rate))
    model.add(Dense(steps_ahead, activation='linear'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', 1e-4, 5e-3, sampling='log', default=1e-3)
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_gru(input_shape, num_layers=1, units=64, dropout_rate=0.3, steps_ahead=1, bidirectional=True, use_attention=True):
    attention_weights = {}
    inputs = Input(shape=input_shape)
    x = inputs
    gru_layer = GRU(units, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
    x = Bidirectional(gru_layer)(x) if bidirectional else gru_layer(x)
    x = Dropout(dropout_rate)(x)
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
    model.attention_weights = attention_weights
    return model

def build_new_model(input_shape, steps_ahead):
    return build_gru(
        input_shape,
        num_layers=CONFIG.get('num_layers', 1),
        units=CONFIG.get('units', 64),
        dropout_rate=CONFIG.get('dropout_rate', 0.3),
        steps_ahead=steps_ahead,
        bidirectional=True,
        use_attention=True
    )
          
def train_and_predict(df: pd.DataFrame, symbol: str, steps_ahead: int, window_size: int, tune=False):
    try:
        send_telegram_message(f"🚀 GRU model training started for {symbol} (data rows: {df.shape[0]})")
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        if 'date' in df.columns:
            df.drop('date', axis=1, inplace=True)
        df = df.select_dtypes(include=[np.number])
        target_col = 'close' if 'close' in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
        if target_col and df.columns[0] != target_col:
            cols = [target_col] + [col for col in df.columns if col != target_col]
            df = df[cols]
        df = df.astype('float32')
        if len(df) < window_size + steps_ahead:
            print(f"⚠️ Not enough data for {symbol} (rows: {len(df)}) for window={window_size} and horizon={steps_ahead}, skipping...")
            return
        data_values = df.values
        scaled_data, scaler = scale_data(data_values)
        X, y = create_sequences(scaled_data, window_size, steps_ahead)
        if X.size == 0 or y.size == 0:
            print(f"⚠️ Unable to build sequences for {symbol}, skipping...")
            return
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
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

            # --- HYPERPARAMETER SEARCH ---
        if tune:
            tuner = kt.BayesianOptimization(
                lambda hp: build_gru_tuner(hp, (X_train.shape[1], X_train.shape[2]), steps_ahead),
                objective='val_loss',
                max_trials=100,
                directory='ktuner_dir',
                project_name=f'{symbol}_gru'
            )
            logdir = f"logs/tuner_{symbol}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_callback = TensorBoard(log_dir=logdir)
            
            tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=40,
                batch_size=CONFIG.get("batch_size", 128),
                callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print(f"🔥 Best hyperparameters for {symbol}:", best_hps.values)
            hps_path = os.path.join(MODELS_BASE_DIR, f"{symbol}_best_hps.json")
            with open(hps_path, "w") as f:
                json.dump(best_hps.values, f, indent=2)
            # Build the best model and train it further
            best_model = tuner.hypermodel.build(best_hps)
            history = best_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG['epochs'],
                batch_size=CONFIG.get("batch_size", 128),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=CONFIG['patience_early_stop'], restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', patience=CONFIG['patience_reduce_lr'],
                                      factor=CONFIG['reduce_lr_factor'], min_lr=CONFIG['min_lr'])
                ],
                verbose=1
            )
            model = best_model
        else:
            model = build_new_model(input_shape=(X_train.shape[1], X_train.shape[2]), steps_ahead=steps_ahead)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG['epochs'],
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=CONFIG['patience_early_stop'], restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', patience=CONFIG['patience_reduce_lr'],
                                      factor=CONFIG['reduce_lr_factor'], min_lr=CONFIG['min_lr'])
                ],
                verbose=1
            )

        # Storage category (keeps the folder structure clean)
        category = None
        preds_for_cat = y_test
        try:
            # Try to load the existing model and scaler
            category_list = ["95-100%", "75-95%", "50-75%", "25-50%", "0-25%", "below_0%"]
            loaded_existing_model = False
            for cat in category_list:
                category_path = os.path.join(MODELS_BASE_DIR, cat)
                os.makedirs(category_path, exist_ok=True)
                model_filename = f"{symbol}_gru.keras"
                scaler_filename = f"{symbol}_gru_scaler.pkl"
                model_path = os.path.join(category_path, model_filename)
                scaler_path = os.path.join(category_path, scaler_filename)
                print("Model path:", model_path)
                print("Expected scaler path:", scaler_path)
                print("Do they exist:", os.path.exists(model_path), os.path.exists(scaler_path))
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    print(f"📦 Found existing model for {symbol}: {model_path}")
                    model = load_model(model_path)
                    scaler = joblib.load(scaler_path)
                    loaded_existing_model = True
                    category = cat
                    break
            if not loaded_existing_model:
                print("ℹ️ Model or scaler missing, creating new ones.")
                model = build_new_model(input_shape=(X_train.shape[1], X_train.shape[2]), steps_ahead=steps_ahead)
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            model = build_new_model(input_shape=(X_train.shape[1], X_train.shape[2]), steps_ahead=steps_ahead)
            loaded_existing_model = False


    # Fine-tune or train from scratch
        history = None
        if not loaded_existing_model or RETRAIN_EXISTING:
    # Print to console and send to Telegram
            if loaded_existing_model and RETRAIN_EXISTING:
                msg = f"🔄 Fine-tuning existing model for {symbol}..."
                print(msg)
                send_telegram_message(msg)
            elif not loaded_existing_model:
                msg = f"🆕 Training a new model for {symbol}..."
                print(msg)
                send_telegram_message(msg)
            else:
                msg = f"🚀 Starting training/fine-tuning for {symbol}..."
                print(msg)
                send_telegram_message(msg)

        callbacks = [
        EarlyStopping(monitor='val_loss', patience=CONFIG['patience_early_stop'], restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=CONFIG['patience_reduce_lr'], factor=CONFIG['reduce_lr_factor'], min_lr=CONFIG['min_lr'])
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
        print("y_train[:10]:", y_train[:10])
        print("y_val[:10]:", y_val[:10])
        print("min/max y_train:", np.min(y_train), np.max(y_train))
        print("min/max y_val:", np.min(y_val), np.max(y_val))

     
        # Recompute metrics
        preds = model.predict(X_test)
        target_min = scaler.data_min_[0]
        target_max = scaler.data_max_[0]
        preds_inverse = preds * (target_max - target_min) + target_min
        y_test_inverse = y_test * (target_max - target_min) + target_min
        mae = mean_absolute_error(y_test_inverse, preds_inverse)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, preds_inverse))
        mse, r2, evs = calculate_additional_metrics(y_test_inverse, preds_inverse)
        accuracy, mape = calculate_accuracy(y_test_inverse, preds_inverse)

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
        model_filename = f"{symbol}_gru.keras"
        scaler_filename = f"{symbol}_gru_scaler.pkl"
        model_path = os.path.join(category_path, model_filename)
        scaler_path = os.path.join(category_path, scaler_filename)


        # --- Move old artifacts to the archive ---
        category_names = ["95-100%", "75-95%", "50-75%", "25-50%", "0-25%", "below_0%"]
        model_filename = f"{symbol}_gru.keras"
        scaler_filename = f"{symbol}_gru_scaler.pkl"
        archive_dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for cat in category_names:
            if cat == category:
                continue

        cat_dir = os.path.join(MODELS_BASE_DIR, cat)
        archive_dir = os.path.join(cat_dir, "archive")
        try:
            os.makedirs(archive_dir, exist_ok=True)
        except Exception as e:
            print(f"❗️ Failed to create archive folder: {archive_dir} | Error: {e}")
            
            # Model
            old_model_path = os.path.join(cat_dir, model_filename)
            archive_model_path = os.path.join(archive_dir, f"{symbol}_gru.keras")
            if os.path.exists(old_model_path):
                try:
                    shutil.move(old_model_path, archive_model_path)
                    print(f"📦 Model {old_model_path} moved to archive: {archive_model_path}")
                except Exception as e:
                    print(f"❗️ Error moving model: {old_model_path} → {archive_model_path} | {e}")
            else:
                print(f"🔸 No model to move: {old_model_path}")


            # Scaler
            old_scaler_path = os.path.join(cat_dir, scaler_filename)
            archive_scaler_path = os.path.join(archive_dir, f"{symbol}_gru_scaler.pkl")
            if os.path.exists(old_scaler_path):
                try:
                    shutil.move(old_scaler_path, archive_scaler_path)
                    print(f"📦 Scaler {old_scaler_path} moved to archive: {archive_scaler_path}")
                except Exception as e:
                    print(f"❗️ Error moving scaler: {old_scaler_path} → {archive_scaler_path} | {e}")
            else:
                print(f"🔸 No scaler to move: {old_scaler_path}")

            # Logs
            log_files = glob.glob(os.path.join(cat_dir, f"{symbol}_log_*.txt"))
            if log_files:
                for log_file in log_files:
                    try:
                        shutil.move(log_file, os.path.join(archive_dir, os.path.basename(log_file)))
                        print(f"📦 Log {log_file} moved to archive: {archive_dir}")
                    except Exception as e:
                        print(f"❗️ Error moving log: {log_file} → {archive_dir} | {e}")
            else:
                print(f"🔸 No logs to move from: {cat_dir}")

            # Plots
            plot_training_history(history, symbol, category_path, timestamp)
            plot_true_vs_pred(y_test_inverse, preds_inverse, symbol, category_path, timestamp)
            plot_attention_weights(model, X_test, symbol, category_path, timestamp)
            
            plot_files = glob.glob(os.path.join(cat_dir, f"{symbol}_*.png"))
            if plot_files:
                for plot_file in plot_files:
                    try:
                        shutil.move(plot_file, os.path.join(archive_dir, os.path.basename(plot_file)))
                        print(f"📦 Plot {plot_file} moved to archive: {archive_dir}")
                    except Exception as e:
                        print(f"❗️ Error moving plot: {plot_file} → {archive_dir} | {e}")
            else:
                print(f"🔸 No plots to move from: {cat_dir}")

        # Save the final model and scaler under the same fixed names (overwrite)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
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

        pred_plot_path = os.path.join(category_path, f"{symbol}_prediction_vs_actual_{timestamp}.png")
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inverse[:, 0], label='True')
        plt.plot(preds_inverse[:, 0], label='Predicted')
        plt.title(f'{symbol}: Forecast vs Actual (step 1 of {steps_ahead})')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(pred_plot_path)
        plt.close()

        plot_attention_weights(model, X_test, symbol, category_path, timestamp)

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
        error_msg = f"⚠️ Error processing {symbol}: {e}"
        print(error_msg)
        if ERROR_LOG_PATH:
            with open(ERROR_LOG_PATH, "a") as err_file:
                err_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {symbol}: {traceback.format_exc()}\n")
        send_telegram_message(error_msg)
        clean_memory()
import glob
import os
import yaml

# --- Get the symbol list from the config ---
def get_symbols_from_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    symbols = cfg.get("symbols", {})
    return [sym for sym, enabled in symbols.items() if enabled is True or enabled == "true"]

# --- Find a matching data file for a symbol ---
import os
import glob

def get_possible_tickers(symbol):
    tickers = []
    # For USDT pairs
    if symbol.endswith('USDT'):
        tickers.append(symbol.replace('USDT', '').upper())
        tickers.append(symbol.replace('USDT', '').lower())
    # Support for other suffixes can be added here
    return tickers

def find_file_for_symbol(data_dir, symbol):
    all_files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.cvs"))
    symbol_lower = symbol.lower()
    tickers = get_possible_tickers(symbol)
    matching = []
    for f in all_files:
        filename = os.path.basename(f).lower()
        if symbol_lower in filename or any(ticker in filename for ticker in tickers):
            matching.append(f)
    print(f"DEBUG: searching {symbol_lower} / {tickers}, found: {[os.path.basename(f) for f in matching]}")
    if matching:
        matching = sorted(matching, key=os.path.getmtime, reverse=True)
        return matching[0]
    else:
        return None

  

# --- smart_train is NO LONGER NEEDED ---


#DATA_PATH = os.path.join(BASE_DIR, "data", "data_base")

if __name__ == "__main__":
    DATA_PATH = os.path.join(BASE_DIR, "data", "data_base")
    CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
    if not os.path.exists(DATA_PATH):
        error_message = f"❌ Data folder not found: {DATA_PATH}"
        print(error_message)
        send_telegram_message(error_message)
    else:
        symbol_list = get_symbols_from_config(CONFIG_PATH)
        print(f"🟢 Symbols to train: {symbol_list}")
        for symbol in symbol_list:
            file_path = find_file_for_symbol(DATA_PATH, symbol)
            if file_path:
                df = pd.read_csv(file_path)
                print(f"🔄 Found file for {symbol}: {os.path.basename(file_path)}")
                train_and_predict(
                    df,
                    symbol,
                    steps_ahead=GLOBAL_STEPS_AHEAD,
                    window_size=GLOBAL_WINDOW_SIZE,
                    tune=True
                )
            else:
                print(f"⚠️ No matching data file for {symbol} in {DATA_PATH}")

                

def find_file_for_symbol(data_dir, symbol):
    pattern = os.path.join(data_dir, f"*{symbol}*.csv")  # or .cvs if the extension is misspelled
    files = glob.glob(pattern)
    print(f"DEBUG: searching by pattern: {pattern}, found: {files}")
    files = [f for f in files if symbol.lower() in os.path.basename(f).lower()]
    if files:
        files = sorted(files, key=os.path.getmtime, reverse=True)
        return files[0]
    else:
        return None

