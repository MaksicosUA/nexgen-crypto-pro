
# 📁 utils/plots.py
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import numpy as np

def plot_training_history(history, symbol: str, category_path: str, timestamp: str):
    train_plot_path = os.path.join(category_path, f"{symbol}_train_loss_{timestamp}.png")
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

def plot_attention_weights(model, X_test, symbol: str, category_path: str, timestamp: str):
    if hasattr(model, 'attention_weights') and 'scores' in model.attention_weights:
        try:
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
            print(f"⚠️ Error visualizing attention for {symbol}: {e}")

def plot_true_vs_pred(y_true, y_pred, symbol, save_dir, timestamp, show=False):
    price_plot_path = os.path.join(save_dir, f"{symbol}_prediction_vs_actual_{timestamp}.png")
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.title(f"{symbol}: Predicted vs Actual")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(price_plot_path)
    if show:
        plt.show()
    plt.close()
