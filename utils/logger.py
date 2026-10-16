# utils/logger.py

import os
from datetime import datetime
import requests

def log_event(
    message,
    log_type="train",
    log_dir="logs",
    symbol=None,
    send_telegram=False,
    telegram_token=None,
    telegram_chat_id=None,
    send_console=True
):
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log to console
    if send_console:
        print(f"[{timestamp}] [{log_type.upper()}]{' ['+symbol+']' if symbol else ''}: {message}")

    # Log to file
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_type}.log"
    filepath = os.path.join(log_dir, log_filename)
    with open(filepath, "a") as f:
        f.write(f"[{timestamp}]{' ['+symbol+']' if symbol else ''}: {message}\n")

    # Log to Telegram
    if send_telegram:
        # Take the token/chat ID from the arguments or the environment
        telegram_token = telegram_token or os.getenv("TELEGRAM_TOKEN")
        telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat_id:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                    data={"chat_id": telegram_chat_id, "text": message}
                )
            except Exception as e:
                print(f"[LOGGER] Failed to send to Telegram: {e}")
