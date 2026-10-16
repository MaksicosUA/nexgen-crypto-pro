import os
import requests
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        print("Telegram response:", response.json())
    except Exception as e:
        print(f"❌ Telegram error: {e}")