# 📊 LSTM Training Script with Quality Categorization and Logging
import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
import json
import glob
import traceback
import gc

from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import mixed_precision
from datetime import datetime
print("GPUs:", tf.config.list_physical_devices('GPU'))

# === 🎲 Set SEED for reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# === ⚡ FP16: Enable Mixed Precision (FP16 speedup) ===
mixed_precision.set_global_policy('mixed_float16')
print(f"🚀 Mixed Precision enabled: {mixed_precision.global_policy()}")


# === 🧹 TensorFlow memory cleanup ===
def clean_memory():
    """Clear Keras/TensorFlow memory and run the garbage collector."""
    tf.keras.backend.clear_session()
    gc.collect()

# # GPU check and setup
gpus = tf.config.experimental.list_physical_devices('GPU')  # used for memory growth setup
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


# === 📁 Project paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "training")
os.makedirs(BASE_DIR, exist_ok=True)

# === 🔐 Load model configuration ===
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Configuration file not found: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    FULL_CONFIG = json.load(f)
    RETRAIN_EXISTING = FULL_CONFIG.get("model_config", {}).get("retrain_existing", False)


# Default base configuration, overridden from JSON
CONFIG = {
    'num_layers': 2,
    'units': 256,
    'dropout_rate': 0.3,
    'batch_size': 128,
    'epochs': 200,
    'validation_split': 0.3,      # Not used directly; an explicit split is applied instead
    'patience_early_stop': 20,
    'patience_reduce_lr': 10,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-6
}
CONFIG.update(FULL_CONFIG.get("model_config", {}))

# 💡 Support for retraining flags and custom log paths
RETRAIN_EXISTING = FULL_CONFIG.get("model_config", {}).get("retrain_existing", False)
CUSTOM_PATHS = FULL_CONFIG.get("paths", {})
SEARCH_ROOT = CUSTOM_PATHS.get("models_search_root", BASE_DIR)
ERROR_LOG_PATH = CUSTOM_PATHS.get("error_log", os.path.join(BASE_DIR, "errors.log"))
TRAINED_LOG_PATH = CUSTOM_PATHS.get("trained_log", os.path.join(BASE_DIR, "trained_log.txt"))

# Global parameters from the configuration
ENABLED_EXCHANGES = [k for k, v in FULL_CONFIG.get("exchanges", {}).items() if v]
ENABLED_SYMBOLS = [k for k, v in FULL_CONFIG.get("symbols", {}).items() if v]
GLOBAL_STEPS_AHEAD = FULL_CONFIG.get("global_config", {}).get("steps_ahead", 3)
GLOBAL_WINDOW_SIZE = FULL_CONFIG.get("global_config", {}).get("window_size", 30)
LOG_ERRORS = FULL_CONFIG.get("global_config", {}).get("log_errors_separately", True)
VAL_SIZE = FULL_CONFIG.get("global_config", {}).get("val_size", 0.15)
TEST_SIZE = FULL_CONFIG.get("global_config", {}).get("test_size", 0.15)

# Error log path (if enabled)
ERROR_LOG_PATH = os.path.join(BASE_DIR, "errors.log") if LOG_ERRORS else None

# === ✉️ Telegram setup ===
load_dotenv()  # Load TELEGRAM_TOKEN and CHAT_ID from .env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("❌ TELEGRAM_TOKEN or CHAT_ID not found. Make sure .env contains both values.")

def send_telegram_message(msg: str):
    """Send a text message to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        # Debug output (can be commented out):
        print("Telegram response:", response.json())
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# === 🧮 Data preprocessing functions ===
def augment_data(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Add random noise to the data to simulate new samples.
    Used to augment sequences when the dataset is small.
    """
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise

def split_data(X: np.ndarray, y: np.ndarray, val_size: float = 0.15, test_size: float = 0.15):
    """
    Split data into training, validation and test sets without shuffling (time-ordered).
    Returns a tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    total_samples = len(X)
    # Start index of the test set
    test_start = int(total_samples * (1 - test_size))
    X_temp, X_test = X[:test_start], X[test_start:]
    y_temp, y_test = y[:test_start], y[test_start:]
    # Now split X_temp, y_temp into training and validation
    val_ratio = val_size / (1 - test_size) if (1 - test_size) != 0 else 0
    val_start = int(len(X_temp) * (1 - val_ratio))
    X_train, X_val = X_temp[:val_start], X_temp[val_start:]
    y_train, y_val = y_temp[:val_start], y_temp[val_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(data: np.ndarray):
    """
    Scale data to the [0,1] range with MinMaxScaler.
    Returns the scaled data and the scaler object.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(data: np.ndarray, window_size: int, steps_ahead: int):
    """
    Convert a data series into input sequences X and targets y.
    X: windows of length window_size,
    y: the next steps_ahead values of the target variable (first data column) after each X window.
    """
    X, Y = [], []
    for i in range(len(data) - window_size - steps_ahead + 1):
        X.append(data[i : i + window_size])
        # Target: the first steps_ahead values after the window (first column only, index 0)
        Y.append(data[i + window_size : i + window_size + steps_ahead, 0])
    return np.array(X), np.array(Y)

# === 🔨 Model building functions ===
def build_lstm(input_shape, num_layers=1, units=64, dropout_rate=0.3, steps_ahead=1, 
               bidirectional=True, use_attention=False):
    """
    Build an LSTM/GRU model (with an optional attention mechanism).
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
        from tensorflow.keras.layers import GlobalAveragePooling1D
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
    Build a new LSTM model from the global configuration.
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

# === 🎯 Prediction quality metrics ===
def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate prediction 'accuracy' as a percentage (100 - MAPE).
    Returns a tuple (accuracy, mape); accuracy can be >100 or <0 for very large/negative errors.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero: only consider elements where y_true != 0
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    accuracy = 100 - mape
    return accuracy, mape

# === 📊 Training history visualization ===
def plot_training_history(history, symbol: str, category_path: str, timestamp: str):
    """
    Save the training plot (Train/Val Loss) for the model.
    """
    train_plot_path = os.path.join(category_path, f"{symbol}_train_loss_{timestamp}.png")
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Training history for {symbol}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(train_plot_path)
    plt.close()

# === 📈 Attention visualization (if used) ===
def plot_attention_weights(model, X_test, symbol: str, category_path: str, timestamp: str):
    """
    If the model uses an attention mechanism, save a plot of the average attention weights.
    """
    if hasattr(model, 'attention_weights') and 'scores' in model.attention_weights:
        try:
            # Extract average attention weights along the time axis
            attention_output = Model(inputs=model.input, outputs=model.attention_weights['scores'])
            attention_scores = attention_output.predict(X_test)
            avg_attention = np.mean(attention_scores[0], axis=0)
            plt.figure(figsize=(10, 4))
            plt.plot(avg_attention)
            plt.title(f"Average attention weights for {symbol}")
            plt.xlabel("Time step")
            plt.ylabel("Attention weight")
            plt.grid(True)
            attention_plot_path = os.path.join(category_path, f"{symbol}_attention_{timestamp}.png")
            plt.savefig(attention_plot_path)
            plt.close()
        except Exception as e:
            print(f"⚠️ Attention visualization error for {symbol}: {e}")

# === 🔮 Training and prediction for a single dataset ===
def train_and_predict(df: pd.DataFrame, symbol: str, steps_ahead: int, window_size: int):
    """
    Train a model for the given symbol on df and make a forecast.
    Saves results, the model and logs to the corresponding folders.
    """
    try:
        send_telegram_message(f"🚀 Training started for {symbol} (data rows: {df.shape[0]})")

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

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, val_size=VAL_SIZE, test_size=TEST_SIZE)

        if len(X_train) > 0:
            X_aug = augment_data(X_train, noise_level=0.01)
            y_aug = y_train.copy()
            X_train = np.concatenate([X_train, X_aug])
            y_train = np.concatenate([y_train, y_aug])

        # === 💾 Prepare model file name without date ===
        symbol_tag = symbol
        model_filename = f"{symbol_tag}_lstm.keras"
        scaler_filename = f"{symbol_tag}_scaler.pkl"

        # === The category is determined later from accuracy; save to a temporary folder for now
        temp_model_dir = os.path.join(BASE_DIR, "temp")
        os.makedirs(temp_model_dir, exist_ok=True)
        model_path = os.path.join(temp_model_dir, model_filename)
        scaler_path = os.path.join(temp_model_dir, scaler_filename)

        # === Load the best model (if any) ===
        best_model_dir = os.path.join(BASE_DIR, "best_models")
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_path = os.path.join(best_model_dir, f"{symbol}_best.txt")
        model = None
        loaded_existing_model = False

        if os.path.exists(best_model_path):
            with open(best_model_path, "r") as f:
                best_line = f.read().strip().split(',')
                best_model_name = best_line[0] if best_line else ""
                model_file_pattern = os.path.join(BASE_DIR, '**', best_model_name)
                matched_files = glob.glob(model_file_pattern, recursive=True)
                print(f"[DEBUG] Looking for model: {best_model_name}")
                print(f"[DEBUG] Search pattern: {model_file_pattern}")
                print(f"[DEBUG] Matches found: {len(matched_files)}")

            if matched_files:
                try:
                    print(f"📦 Found existing model for {symbol}: {matched_files[0]}")
                    model = load_model(matched_files[0])
                    scaler_file = matched_files[0].replace('.keras', '_scaler.pkl')
                    if os.path.exists(scaler_file):
                        scaler = joblib.load(scaler_file)
                        loaded_existing_model = True
                    else:
                        print("⚠️ Scaler not found, the model will be trained from scratch")
                        model = None
                except Exception as e:
                    print(f"⚠️ Error loading model: {e}")
                    model = None
            else:
                print("ℹ️ Model file not found, creating a new one")
        else:
            print("ℹ️ No previously saved model, creating a new one")

        if model is None:
            model = build_new_model(input_shape=(X_train.shape[1], X_train.shape[2]), steps_ahead=steps_ahead)

        history = None
        if not loaded_existing_model or RETRAIN_EXISTING:
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
        else:
            print(f"⚡ Model for {symbol} loaded without fine-tuning.")

        # Temporarily save the model before evaluation
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Temporary model saved: {model_path}")

        # Next: metric calculation, categorization, moving the model and logging

       

        # Predict on the test set
        preds = model.predict(X_test)
        # Invert scaling of predictions and targets
        # Use the scaler parameters of the first (target) variable for this
        target_min = scaler.data_min_[0]
        target_max = scaler.data_max_[0]
        preds_inverse = preds * (target_max - target_min) + target_min
        y_test_inverse = y_test * (target_max - target_min) + target_min

        # Compute quality metrics
        mae = mean_absolute_error(y_test_inverse, preds_inverse)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, preds_inverse))
        accuracy, mape = calculate_accuracy(y_test_inverse, preds_inverse)

        # Determine the model quality category from prediction accuracy
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

        # Create the category folder if it does not exist
        category_path = os.path.join(BASE_DIR, category)
        os.makedirs(category_path, exist_ok=True)

        # Training end time (used in logs and file names)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Send a training-complete notification with the results
        msg = (f"📊 Training finished for {symbol}\n"
               f"MAE: {mae:.4f}\n"
               f"RMSE: {rmse:.4f}\n"
               f"Accuracy: {accuracy:.2f}%\n"
               f"MAPE: {mape:.2f}%\n"
               f"Category: {category.upper()}\n"
               f"Date: {timestamp}"
               )
        send_telegram_message(msg)

        # Save the training history plot (if the model was (re)trained in this run)
        if history is not None:
            plot_training_history(history, symbol, category_path, timestamp)

        # Save the "Prediction vs Actual" plot for step 1
        pred_plot_path = os.path.join(category_path, f"{symbol}_prediction_vs_actual_{timestamp}.png")
        plt.figure(figsize=(12, 6))
        # Use the first column (forecast step 1) for clarity
        plt.plot(y_test_inverse[:, 0], label='True')
        plt.plot(preds_inverse[:, 0], label='Predicted')
        plt.title(f'{symbol}: Forecast vs Actual (step 1 of {steps_ahead})')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(pred_plot_path)
        plt.close()

        # Attention visualization (if applicable)
        plot_attention_weights(model, X_test, symbol, category_path, timestamp)

        # Save the model and scaler with the current date
        model_filename = f"{symbol}_lstm_{timestamp}.keras"
        scaler_filename = f"{symbol}_scaler_{timestamp}.pkl"
        model.save(os.path.join(category_path, model_filename))
        joblib.dump(scaler, os.path.join(category_path, scaler_filename))

        # Save the model training log to a text file
        log_path = os.path.join(category_path, f"{symbol}_log_{timestamp}.txt")
        with open(log_path, 'w') as log_file:
            log_file.write(f"Date and time: {timestamp}\n")
            log_file.write(f"Symbol: {symbol}\n")
            log_file.write(f"Category: {category}\n")
            log_file.write(f"MAE: {mae:.4f}\n")
            log_file.write(f"RMSE: {rmse:.4f}\n")
            log_file.write(f"Accuracy: {accuracy:.2f}%\n")
            log_file.write(f"MAPE: {mape:.2f}%\n")
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

        # Update best-model info for the symbol
        update_best = False
        best_recorded_mae = None
        if os.path.exists(best_model_path):
            try:
                with open(best_model_path, "r") as f:
                    best_line = f.read().strip().split(',')
                if len(best_line) >= 2:
                    best_recorded_mae = float(best_line[1])
            except Exception as e:
                print(f"⚠️ Error reading best-model file for {symbol}: {e}")
                # If it could not be read, overwrite the file
                update_best = True
        else:
            update_best = True

        if best_recorded_mae is None or mae < best_recorded_mae:
            update_best = True

        if update_best:
            with open(best_model_path, "w") as f:
                f.write(f"{model_filename},{mae}")
            send_telegram_message(f"🏆 New BEST model for {symbol} with MAE: {mae:.4f}")

        # Free memory after model training
        clean_memory()

    except Exception as e:
        # Log and report errors raised during model training
        error_msg = f"⚠️ Error processing {symbol}: {e}"
        print(error_msg)
        if ERROR_LOG_PATH:
            with open(ERROR_LOG_PATH, "a") as err_file:
                err_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {symbol}: {traceback.format_exc()}\n")
        send_telegram_message(error_msg)
        # Free memory even on error so processing can continue
        clean_memory()

# === 📂 Process all CSV files in a folder ===
def load_and_train_from_folder(folder_path: str, steps_ahead: int, window_size: int):
    """
    Load all CSV files from the given folder and train a model for each file in turn.
    Skips files that have already been trained (listed in trained_log).
    """
    # Helper directories and logs
    trained_log_path = os.path.join(BASE_DIR, "trained_log.txt")
    trained_files = []
    if os.path.exists(trained_log_path):
        with open(trained_log_path, "r") as f:
            trained_files = [line.strip() for line in f if line.strip()]
    ignored_dir = os.path.join(BASE_DIR, "ignored")
    os.makedirs(ignored_dir, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        # Skip if already trained
        if filename in trained_files:
            print(f"📦 {filename} was already trained — moved to 'ignored'")
            try:
                os.rename(filepath, os.path.join(ignored_dir, filename))
            except Exception as e:
                print(f"⚠️ Failed to move {filename} to ignored: {e}")
            continue
        # Process CSV files only
        if filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(filepath)
                # Derive the symbol name from the file name (replace invalid characters with '_')
                symbol = os.path.splitext(filename)[0].replace('/', '_').replace('\\', '_').replace(' ', '_')
                print(f"📂 Processing file: {filename} (symbol: {symbol})")
                # Run training and prediction for this file
                train_and_predict(df, symbol, steps_ahead=steps_ahead, window_size=window_size)
                # Append the file name to the trained log
                with open(trained_log_path, "a") as f:
                    f.write(filename + "\n")
            except Exception as e:
                # Log errors from reading the file or starting training
                error_msg = f"⚠️ Error processing file {filename}: {e}"
                print(error_msg)
                if ERROR_LOG_PATH:
                    with open(ERROR_LOG_PATH, "a") as err_file:
                        err_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FILE {filename}: {traceback.format_exc()}\n")
                send_telegram_message(error_msg)
                # Continue with the next file

# === 🚀 Run training (main entry point) ===
def smart_train(source_path: str, steps_ahead: int = 3, window_size: int = 30):
    """
    Run training: if source_path is a folder, train models for every CSV in it;
    if source_path is a file, train a model for that file.
    """
    if os.path.isdir(source_path):
        print(f"📁 Found data folder: {source_path}. Processing all CSV files...")
        load_and_train_from_folder(source_path, steps_ahead, window_size)
    elif os.path.isfile(source_path):
        print(f"📄 Found data file: {source_path}. Training a model for this file...")
        df = pd.read_csv(source_path)
        symbol = os.path.splitext(os.path.basename(source_path))[0].replace('/', '_').replace('\\', '_').replace(' ', '_')
        train_and_predict(df, symbol, steps_ahead=steps_ahead, window_size=window_size)
    else:
        msg = f"❌ Path not found: {source_path}"
        print(msg)
        send_telegram_message(msg)
        return

# Path to the data folder (pipeline/base_data in the project root)
DATA_PATH = os.path.join(PROJECT_ROOT, "pipeline", "base_data")
if __name__ == "__main__":
    # If the folder is not found, print an error and exit
    if not os.path.exists(DATA_PATH):
        error_message = f"❌ Data folder not found: {DATA_PATH}"
        print(error_message)
        send_telegram_message(error_message)
    else:
        # Run training on the configured path
        smart_train(DATA_PATH, steps_ahead=GLOBAL_STEPS_AHEAD, window_size=GLOBAL_WINDOW_SIZE)
