#!/usr/bin/env python3
# === crypto_pipeline_v4.3.py (cache-safe revision) ===
import os
import asyncio
import json
import shutil
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta, timezone
import ccxt.async_support as ccxt
import yaml


# --- 1) Resolve the config path ONCE ---
ROOT = Path(__file__).resolve().parent
_cfg_env = os.getenv("PIPELINE_CONFIG")
if _cfg_env:
        cfg_path = Path(_cfg_env)
        if not cfg_path.exists():
            print(f"[WARN] PIPELINE_CONFIG is set, but the file was not found: {cfg_path} → falling back to local pipeline_config.yaml")
            cfg_path = ROOT / "pipeline_config.yaml"
else:
        cfg_path = ROOT / "pipeline_config.yaml"

if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration not found: {cfg_path}")

with open(cfg_path, "r") as _f:
        config = yaml.safe_load(_f) or {}

CONFIG_DIR = cfg_path.resolve().parent

print(f"[DEBUG] Using config: {cfg_path}")
if os.getenv("SESSION_ROOT"):
        print(f"[DEBUG] SESSION_ROOT: {os.getenv('SESSION_ROOT')}")

    # --- output_format (csv|json|both) — safe initialization ---
OUTPUT_ALLOWED = {"csv", "json", "both"}
output_format = str(config.get("output_format", "csv")).lower()
if output_format not in OUTPUT_ALLOWED:
        output_format = "csv"


# IMPORTANT: do NOT read the YAML again below this point. Use the `config` object.
# If another yaml.safe_load(...) appears further down in this file, REMOVE it.


def resolve_config_path_value(value, default: Path) -> Path:
    """Resolve relative config paths from the config file directory."""
    if value in (None, ""):
        return default
    path = Path(value)
    return path if path.is_absolute() else (CONFIG_DIR / path).resolve()


# --- 2) Base paths are bound to the session ---
BASE_PATH = resolve_config_path_value(config.get("base_path"), ROOT)
DATA_PATH = resolve_config_path_value(config.get("data_path"), BASE_PATH / "data")
PLOT_PATH = resolve_config_path_value(config.get("plot_path"), BASE_PATH / "plots")
LOG_FILE  = resolve_config_path_value(config.get("log_file"), BASE_PATH / "crypto_log.txt")
FULL_BASE = BASE_PATH / "full_data_base"
# create directories
for _p in (BASE_PATH, DATA_PATH, PLOT_PATH, FULL_BASE, LOG_FILE.parent):
    _p.mkdir(parents=True, exist_ok=True)



ARCHIVE_PATH = resolve_config_path_value(config.get("archive_path"), BASE_PATH / "archive")
ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
# --- 3) Markets/exchanges/timeframe and history period from the SESSION CONFIG ---
symbols = list(config.get("symbols", ["BTC/USDT"]))
exchanges = list(config.get("exchanges", ["binance"]))
timeframe = str(config.get("timeframe", "15m"))
history_days = int(config.get("history_days", 7))

# --- 4) Merge output paths (CSV/JSON) inside the session ---
# Use merged_file from the config if set, otherwise write to FULL_BASE/merged.csv

MERGED_FILE_BASE = str(resolve_config_path_value(config.get('merged_file'), FULL_BASE / "merged.csv"))
base_name, _ = os.path.splitext(MERGED_FILE_BASE)
merged_json_path = base_name + '.json'
merged_csv_path = MERGED_FILE_BASE  # backward compatibility with older code

# --- 5) Cache is allowed ONLY if the file exists ---
use_cache = bool(config.get('use_cache', False)) and os.path.isfile(merged_json_path)
if config.get('use_cache', False) and not use_cache:
    print("[WARN] use_cache=True, but merged.json is missing — running without cache")

# --- 6) Helper 'since' timestamp from history_days (ms for ccxt) ---
now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
since_ms = now_ms - history_days * 24 * 60 * 60 * 1000
# --- Load .env for sensitive data (tokens, etc.) ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
telegram_enabled = config.get('alerts', {}).get('telegram_enabled', False)

if telegram_enabled:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env file not found: {ENV_PATH}")
    load_dotenv(dotenv_path=ENV_PATH)
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    CHAT_ID = os.getenv("CHAT_ID", "123456")
    EXTRA_CHAT_ID = os.getenv("TELEGRAM_EXTRA_CHAT_ID", "").strip()
else:
    TELEGRAM_TOKEN = None
    CHAT_ID = None
    EXTRA_CHAT_ID = ""


SEND_DATASET_TO_TELEGRAM = os.getenv("PIPELINE_SEND_DATASET", "1").strip().lower() not in {"0", "false", "no", "off"}

# Telegram request timeouts, so that media uploads cannot block pipeline completion for minutes.
TELEGRAM_CONNECT_TIMEOUT = float(config.get('alerts', {}).get('telegram_connect_timeout', 3))
TELEGRAM_READ_TIMEOUT = float(config.get('alerts', {}).get('telegram_read_timeout', 8))

# chat_ids for which Telegram consistently returns a fatal error (chat not found/blocked, etc.).
DISABLED_TELEGRAM_CHAT_IDS: set[str] = set()


def telegram_request_timeout() -> tuple[float, float]:
    return (TELEGRAM_CONNECT_TIMEOUT, TELEGRAM_READ_TIMEOUT)


def is_fatal_telegram_chat_error(status_code: int, response_text: str) -> bool:
    if status_code not in (400, 403):
        return False
    t = (response_text or "").lower()
    fatal_markers = (
        "chat not found",
        "bot was blocked by the user",
        "user is deactivated",
        "forbidden",
    )
    return any(marker in t for marker in fatal_markers)


def get_target_chat_ids() -> list[str]:
    chat_ids: list[str] = []
    for raw_value in (CHAT_ID, EXTRA_CHAT_ID):
        if not raw_value:
            continue
        for chunk in str(raw_value).split(","):
            chat_id = chunk.strip()
            if chat_id and chat_id not in chat_ids:
                chat_ids.append(chat_id)
    return chat_ids

# --- Symbols and exchanges from the configuration ---
SYMBOLS = config.get('symbols', [])
if not SYMBOLS:
    raise ValueError("No trading symbols specified in the configuration.")
# Determine enabled exchanges
exchanges_conf = config.get('exchanges', [])
if isinstance(exchanges_conf, dict):
    ENABLED_EXCHANGES = [name for name, enabled in exchanges_conf.items() if enabled]
elif isinstance(exchanges_conf, list):
    ENABLED_EXCHANGES = exchanges_conf
else:
    raise ValueError("Invalid exchange list format in the config (must be a list or dict).")

# Map exchange names to ccxt classes
ALL_EXCHANGES = {
    'binance': ccxt.binance,
    'bybit': ccxt.bybit,
    'kucoin': ccxt.kucoin,
    'okx': ccxt.okx,
    'gateio': ccxt.gateio,
    'mexc': ccxt.mexc,
    'coinbase': ccxt.coinbase,
    'bitget': ccxt.bitget
}
EXCHANGE_CLASSES = {}
for ex_name in ENABLED_EXCHANGES:
    if ex_name in ALL_EXCHANGES:
        EXCHANGE_CLASSES[ex_name] = ALL_EXCHANGES[ex_name]
    else:
        print(f"⚠️ Unknown exchange '{ex_name}' in the config, skipped.")

# --- Async request parameters ---
BASE_DELAY = config.get('async', {}).get('base_delay', 1.0)
current_delay = BASE_DELAY
MAX_CONCURRENT_REQUESTS = config.get('async', {}).get('max_concurrent_requests', 5)
MAX_RETRIES = config.get('async', {}).get('max_retries', 3)
REQUEST_TIMEOUT = config.get('async', {}).get('request_timeout', 10)

SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# --- Data collection mode and period ---
COLLECTION_MODE = config.get('collection_mode', 'auto')
CUSTOM_START = config.get('custom_start', None)
CUSTOM_END = config.get('custom_end', None)
DELTA_DAYS = config.get('delta_days', 2)
DELTA_HOURS = config.get('delta_hours', 24)
TIMEFRAME = config.get('timeframe', '1h')
LIMIT = config.get('limit', 1000)

# Determine start and end dates
# auto (default): end = now, start = now - history_days (always up to date)
# full: from the coin's first listing on the exchange until now (no limit)
# days/hours: fixed offsets
# custom: manually specified dates
if COLLECTION_MODE == 'auto':
    _auto_days = int(config.get('history_days', 7))
    END_DATE = datetime.now(timezone.utc)
    START_DATE = END_DATE - timedelta(days=_auto_days)
    print(f"[INFO] Auto mode: {START_DATE.date()} → {END_DATE.date()} ({_auto_days} days)")
elif COLLECTION_MODE == 'full':
    START_DATE = datetime(2010, 1, 1, tzinfo=timezone.utc)  # earlier than any listing
    END_DATE = datetime.now(timezone.utc)
    print(f"[INFO] Full mode: collecting the ENTIRE history from {START_DATE.date()} to {END_DATE.date()}")
elif COLLECTION_MODE == 'days':
    END_DATE = datetime.now(timezone.utc)
    START_DATE = END_DATE - timedelta(days=DELTA_DAYS)
elif COLLECTION_MODE == 'hours':
    END_DATE = datetime.now(timezone.utc)
    START_DATE = END_DATE - timedelta(hours=DELTA_HOURS)
elif COLLECTION_MODE == 'custom':
    if not CUSTOM_START or not CUSTOM_END:
        raise ValueError("Custom mode requires 'custom_start' and 'custom_end' to be set in the config.")
    START_DATE = datetime.fromisoformat(CUSTOM_START).replace(tzinfo=timezone.utc)
    END_DATE = datetime.fromisoformat(CUSTOM_END).replace(tzinfo=timezone.utc)
else:
    raise ValueError(f"❌ Invalid COLLECTION_MODE='{COLLECTION_MODE}'. Use one of: auto, full, days, hours, custom.")

# --- Logging setup ---
ERROR_LOG_FILE = os.path.join(os.path.dirname(LOG_FILE), "errors.log")

async def log(msg: str):
    """
    Log a timestamped message to the console and the log file.
    If the message is a warning or an error, also copy it to the error log file.
    """
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{ts} {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a') as f_log:
        f_log.write(full_msg + "\n")
    lower_msg = msg.lower()
    if ('⚠️' in msg or '❌' in msg or '💀' in msg or 'error' in lower_msg or 'exception' in lower_msg):
        with open(ERROR_LOG_FILE, 'a') as f_err:
            f_err.write(full_msg + "\n")

async def send_telegram_message(text: str):
    """Send a text message via the Telegram bot (if enabled)."""
    target_chat_ids = get_target_chat_ids()
    if not telegram_enabled or not TELEGRAM_TOKEN or not target_chat_ids:
        await log("ℹ️ Telegram notifications are disabled, message not sent.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        delivered = 0
        for chat_id in target_chat_ids:
            if chat_id in DISABLED_TELEGRAM_CHAT_IDS:
                continue
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    url,
                    data={'chat_id': chat_id, 'text': text},
                    timeout=telegram_request_timeout(),
                )
            except requests.RequestException as e:
                await log(f"⚠️ Telegram message request failed for {chat_id}: {e}")
                continue
            if response.status_code != 200:
                await log(f"⚠️ Telegram message error for {chat_id} {response.status_code}: {response.text}")
                if is_fatal_telegram_chat_error(response.status_code, response.text):
                    DISABLED_TELEGRAM_CHAT_IDS.add(str(chat_id))
                    await log(f"🚫 Telegram chat_id excluded from delivery: {chat_id}")
            else:
                delivered += 1
        if delivered:
            await log(f"📬 Report sent to Telegram ({delivered} recipients): {text[:50]}...")
    except Exception as e:
        await log(f"❌ Telegram message exception: {e}")

async def send_image(image_path: str, caption: str = ""):
    """Send an image via Telegram (if enabled)."""
    target_chat_ids = get_target_chat_ids()
    if not telegram_enabled or not TELEGRAM_TOKEN or not target_chat_ids:
        await log(f"ℹ️ Telegram is disabled, chart saved locally: {os.path.basename(image_path)}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        delivered = 0
        for chat_id in target_chat_ids:
            if chat_id in DISABLED_TELEGRAM_CHAT_IDS:
                continue
            with open(image_path, 'rb') as photo:
                try:
                    response = await asyncio.to_thread(
                        requests.post,
                        url,
                        data={'chat_id': chat_id, 'caption': caption},
                        files={'photo': photo},
                        timeout=telegram_request_timeout(),
                    )
                except requests.RequestException as e:
                    await log(f"⚠️ Telegram image request failed for {chat_id}: {e}")
                    continue
            if response.status_code != 200:
                await log(f"⚠️ Telegram image error for {chat_id} {response.status_code}: {response.text}")
                if is_fatal_telegram_chat_error(response.status_code, response.text):
                    DISABLED_TELEGRAM_CHAT_IDS.add(str(chat_id))
                    await log(f"🚫 Telegram chat_id excluded from delivery: {chat_id}")
            else:
                delivered += 1
        if delivered:
            await log(f"📷 Chart sent to Telegram ({delivered} recipients): {os.path.basename(image_path)}")
    except Exception as e:
        await log(f"❌ Telegram image exception: {e}")


async def send_document_file(file_path: str, caption: str = ""):
    """Send a document via Telegram to all target chat_ids."""
    target_chat_ids = get_target_chat_ids()
    if not telegram_enabled or not TELEGRAM_TOKEN or not target_chat_ids:
        await log(f"ℹ️ Telegram is disabled, file saved locally: {os.path.basename(file_path)}")
        return
    if not os.path.isfile(file_path):
        await log(f"⚠️ File to send not found: {file_path}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        delivered = 0
        for chat_id in target_chat_ids:
            if chat_id in DISABLED_TELEGRAM_CHAT_IDS:
                continue
            with open(file_path, 'rb') as document:
                try:
                    response = await asyncio.to_thread(
                        requests.post,
                        url,
                        data={'chat_id': chat_id, 'caption': caption},
                        files={'document': document},
                        timeout=telegram_request_timeout(),
                    )
                except requests.RequestException as e:
                    await log(f"⚠️ Telegram document request failed for {chat_id}: {e}")
                    continue
            if response.status_code != 200:
                await log(f"⚠️ Telegram document error for {chat_id} {response.status_code}: {response.text}")
                if is_fatal_telegram_chat_error(response.status_code, response.text):
                    DISABLED_TELEGRAM_CHAT_IDS.add(str(chat_id))
                    await log(f"🚫 Telegram chat_id excluded from delivery: {chat_id}")
            else:
                delivered += 1
        if delivered:
            await log(f"📎 File sent to Telegram ({delivered} recipients): {os.path.basename(file_path)}")
    except Exception as e:
        await log(f"❌ Telegram document exception: {e}")

async def check_internet_connection():
    try:
        await asyncio.to_thread(requests.get, "https://api.binance.com", timeout=5)
        await log("🌐 Internet connection established.")
    except requests.RequestException:
        await log("❌ No internet connection.")
        if telegram_enabled:
            await send_telegram_message("❌ Error: no internet connection.")
        exit(1)

async def check_exchanges_health():
    unavailable = []
    for ex_name, ExClass in EXCHANGE_CLASSES.items():
        try:
            exchange = ExClass()
            await exchange.load_markets()
            await exchange.close()
        except Exception as e:
            unavailable.append(ex_name)
            await log(f"❌ Exchange {ex_name} is unavailable: {e}")
    return unavailable

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df['SMA_14'] = df['close'].rolling(window=14).mean()
    df['EMA_14'] = df['close'].ewm(span=14, adjust=False).mean()
    df['RETURN'] = df['close'].pct_change()
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['BB_MID'] = df['close'].rolling(window=20).mean()
    df['BB_STD'] = df['close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']
    df['STD_14'] = df['close'].rolling(window=14).std()
    df['MOM_10'] = df['close'] - df['close'].shift(10)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    TP = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = TP.rolling(window=20).mean()
    mad = TP.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI_20'] = (TP - sma_tp) / (0.015 * mad)
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['WILLR_14'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-9)
    df = df.dropna().reset_index(drop=True)
    return df

class DataQualityChecker:
    def __init__(self, log_func=log):
        self.log_func = log_func

    async def check_missing_timestamps(self, df: pd.DataFrame, timeframe: str) -> list:
        result = []
        step = pd.Timedelta(timeframe)
        for (symbol, exchange), group in df.groupby(['symbol', 'exchange']):
            group = group.sort_values('timestamp')
            time_diffs = group['timestamp'].diff()
            missing = group[time_diffs > step]
            if not missing.empty:
                for idx, row in missing.iterrows():
                    prev_idx = row.name - 1
                    prev_time = group.iloc[prev_idx]['timestamp'] if prev_idx >= 0 else None
                    gap_msg = (f"⚠️ Data gap: {symbol} on {exchange}: "
                               f"{prev_time} -> {row['timestamp']} "
                               f"(gap {row['timestamp'] - prev_time})")
                    await self.log_func(gap_msg)
                    result.append((symbol, exchange, prev_time, row['timestamp']))
        if not result:
            await self.log_func("✅ No gaps found in the time series.")
        return result

    async def check_outliers(self, df: pd.DataFrame, price_column: str = 'close', threshold: float = 0.15) -> list:
        outlier_points = []
        for (symbol, exchange), group in df.groupby(['symbol', 'exchange']):
            group = group.sort_values('timestamp')
            returns = group[price_column].pct_change()
            outliers = group[(returns.abs() > threshold) & (returns.notna())]
            if not outliers.empty:
                await self.log_func(f"⚠️ Outliers detected in {symbol} on {exchange}:")
                for idx, row in outliers.iterrows():
                    await self.log_func(f"  {row['timestamp']}: price {row[price_column]}")
                    outlier_points.append((symbol, exchange, row['timestamp'], row[price_column]))
            else:
                await self.log_func(f"✅ No outliers found in {symbol} on {exchange}.")
        return outlier_points

async def safe_fetch(exchange, symbol: str, timeframe: str, since: int, limit: int):
    global current_delay
    for attempt in range(MAX_RETRIES):
        try:
            async with SEMAPHORE:
                bars = await asyncio.wait_for(
                    exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit),
                    timeout=REQUEST_TIMEOUT
                )
            current_delay = max(BASE_DELAY, current_delay - 0.1)
            return bars
        except Exception as e:
            await log(f"⚠️ Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            current_delay += 0.5
            await asyncio.sleep(2 * (attempt + 1))
    await log("❌ All data fetch attempts exhausted.")
    return None

async def fetch_data(ex_name: str, symbol: str):
    exchange_class = EXCHANGE_CLASSES.get(ex_name)
    if exchange_class is None:
        return
    exchange = exchange_class()
    all_data = []
    since = exchange.parse8601(START_DATE.isoformat())
    await log(f"🔌 Collecting data from {ex_name} for {symbol}")
    try:
        while True:
            if since >= exchange.parse8601(END_DATE.isoformat()):
                break
            bars = await safe_fetch(exchange, symbol, timeframe=TIMEFRAME, since=since, limit=LIMIT)
            if not bars:
                break
            bars = [bar for bar in bars if bar[0] < exchange.parse8601(END_DATE.isoformat())]
            if not bars:
                break
            all_data.extend(bars)
            since = bars[-1][0] + 1
            await asyncio.sleep(current_delay)
    except Exception as e:
        await log(f"⚠️ Error fetching data for {symbol} @ {ex_name}: {e}")
    finally:
        await exchange.close()
    if all_data:
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['symbol'] = symbol
        df['exchange'] = ex_name
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{ex_name}_{symbol.replace('/', '')}_{TIMEFRAME}_{date_str}.csv"
        df.to_csv(os.path.join(DATA_PATH, filename), index=False)
        await log(f"✅ Data saved: {filename} ({len(df)} candles, {df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    else:
        # Soft warning instead of a crash — the pair may not exist on the exchange
        await log(f"⚠️ No data for {symbol} on {ex_name} for the period {START_DATE.date()} → {END_DATE.date()}. "
                  f"The pair may be unsupported or there are no candles for this period. Continuing...")

async def collect_data():
    tasks = []
    for ex_name in EXCHANGE_CLASSES.keys():
        for symbol in SYMBOLS:
            tasks.append(fetch_data(ex_name, symbol))
    await asyncio.gather(*tasks)

async def clean_and_fix_timeseries(df, timeframe='30m', max_gap_share=0.3, log_func=print):
    step = pd.Timedelta(timeframe)
    result_dfs = []
    total_pairs = 0
    cleaned_pairs = 0

    for (symbol, exchange), group in df.groupby(['symbol', 'exchange']):
        group = group.sort_values('timestamp').reset_index(drop=True)
        total_pairs += 1

        # Drop rows that break chronological order
        mask = group['timestamp'].diff() > pd.Timedelta(0)
        mask.iloc[0] = True
        group = group[mask].reset_index(drop=True)

        if group.shape[0] < 2:
            continue

        full_range = pd.date_range(
            group['timestamp'].iloc[0],
            group['timestamp'].iloc[1] if group.shape[0] == 2 else group['timestamp'].iloc[-1],
            freq=step
        )
        group = group.set_index('timestamp').reindex(full_range).rename_axis('timestamp').reset_index()
        group['symbol'] = symbol
        group['exchange'] = exchange

        n_missing = group['open'].isna().sum()
        n_total = len(group)
        gap_share = n_missing / n_total if n_total else 1

        if gap_share <= max_gap_share:
            log_func(f"✅ [{symbol} @ {exchange}] missing bars found: {n_missing} ({gap_share:.2%}), filling with NaN — time series kept.")
            result_dfs.append(group)
            cleaned_pairs += 1
        else:
            log_func(f"❗ [{symbol} @ {exchange}] too many gaps: {n_missing} ({gap_share:.2%}) — this series will be removed from the final data!")
            continue

    cleaned_df = pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()
    log_func(f"🧹 Summary: {cleaned_pairs}/{total_pairs} pairs kept, {total_pairs - cleaned_pairs} removed for having too many gaps.")
    return cleaned_df

async def patch_missing_from_alternate(df_main, df_alt, key_columns=['timestamp', 'symbol', 'exchange'], value_columns=None):
    if value_columns is None:
        value_columns = [col for col in df_main.columns if col not in key_columns]
    df_main = df_main.set_index(key_columns)
    df_alt = df_alt.set_index(key_columns)
    for col in value_columns:
        df_main[col] = df_main[col].combine_first(df_alt[col])
    df_main = df_main.reset_index()
    return df_main

async def merge_and_analyze():
    await log("🔄 Merging data from CSV files...")
    start_time = datetime.now()
    all_dfs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            file_path = os.path.join(DATA_PATH, file)
            try:
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
                all_dfs.append(df)
            except Exception as e:
                await log(f"❌ Error reading file {file}: {e}")
    if not all_dfs:
        await log("⚠️ No data to merge.")
        if telegram_enabled:
            await send_telegram_message("⚠️ Error: no data to merge.")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged[merged['volume'] > 0]
    merged.drop_duplicates(subset=['timestamp', 'symbol', 'exchange'], inplace=True)
    merged['timestamp'] = pd.to_datetime(merged['timestamp'])
    merged.sort_values(by='timestamp', inplace=True)

    # Clean the time series
    merged = await clean_and_fix_timeseries(
        merged,
        timeframe=TIMEFRAME,
        max_gap_share=0.3,
        log_func=lambda msg: asyncio.create_task(log(msg))
    )

    # Patch NaNs from the alternate exchange
    alt_names = ['kucoin', 'bybit']
    if ('binance' in merged['exchange'].unique() and any(ex in merged['exchange'].unique() for ex in alt_names)):
        await log("🩹 Patching gaps from an alternate exchange (kucoin/bybit)...")
        main_df = merged[merged['exchange'] == 'binance']
        alt_df = merged[merged['exchange'].isin(alt_names)]
        merged_patched = await patch_missing_from_alternate(main_df, alt_df)
        merged = pd.concat([merged_patched, alt_df], ignore_index=True)
    else:
        await log("ℹ️ No alternate exchange found for patching, or the data does not contain both exchanges.")

    checker = DataQualityChecker(log_func=log)
    await checker.check_missing_timestamps(merged, timeframe=TIMEFRAME)
    await checker.check_outliers(merged, price_column='close')

    na_before = merged.isna().sum().sum()
    merged_filled = merged.ffill().bfill()
    na_after = merged_filled.isna().sum().sum()
    await log(f"🔧 NaN before ffill/bfill: {na_before}, after: {na_after}")

    merged_with_indicators = add_indicators(merged_filled)

    # --- Save merged_with_indicators ---
    if output_format in ("csv", "both"):
        merged_with_indicators.to_csv(merged_csv_path, index=False)
        await log(f"📁 Final CSV file saved: {merged_csv_path}")

    if output_format in ("json", "both"):
        merged_with_indicators.to_json(merged_json_path, orient='records', date_format='iso', date_unit='s')
        await log(f"📁 Final JSON file saved: {merged_json_path}")

    await generate_plots_and_summary(merged_with_indicators, len(all_dfs), start_time, cache_used=False)

async def generate_plots_and_summary(df: pd.DataFrame, files_collected: int, start_time: datetime, cache_used: bool = False):
    await log("📊 Generating charts{}...".format(" from cached data" if cache_used else ""))
    plots_built = 0
    pairs = df[['symbol', 'exchange']].drop_duplicates()
    for _, row in pairs.iterrows():
        symbol = row['symbol']
        exchange = row['exchange']
        data = df[(df['symbol'] == symbol) & (df['exchange'] == exchange)]
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(data['timestamp'], data['close'], label=f"{symbol} ({exchange})")
            plt.title(f"{symbol} — {exchange}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_file = f"{symbol.replace('/', '')}_{exchange}.png"
            plot_path = os.path.join(PLOT_PATH, plot_file)
            plt.savefig(plot_path)
            plt.close()
            await send_image(plot_path, caption=f"📈 Chart {symbol} — {exchange}")
            await log(f"📤 Chart saved/sent: {plot_file}")
            plots_built += 1
        except Exception as e:
            await log(f"❌ Error building/sending chart for {symbol} @ {exchange}: {e}")
    end_time = datetime.now()
    duration = end_time - start_time
    header = "✅ Final report:" + (" (cache used)" if cache_used else "")
    summary_lines = [
        header,
        f"- Files collected: {files_collected}",
        f"- Charts built: {plots_built}",
        f"- Total candles in data: {len(df)}",
        f"- Execution time: {str(duration).split('.')[0]}",
        f"- Final delay: {round(current_delay, 2)} s"
    ]
    if 'failed_exchanges' in globals() and failed_exchanges:
        summary_lines.append(f"- Unavailable sources: {', '.join(failed_exchanges)}")
    summary = "\n".join(summary_lines)
    await log("🧾 Results summary:")
    await log(summary)
    if telegram_enabled:
        await send_telegram_message(summary)
        if SEND_DATASET_TO_TELEGRAM:
            if output_format in ("csv", "both") and os.path.isfile(merged_csv_path):
                await send_document_file(merged_csv_path, caption="📄 Final dataset (CSV)")
            if output_format in ("json", "both") and os.path.isfile(merged_json_path):
                await send_document_file(merged_json_path, caption="📄 Final dataset (JSON)")
        else:
            await log("ℹ️ Sending the final dataset to Telegram is disabled for the current role.")

async def clear_old_plots():
    if not os.path.exists(PLOT_PATH):
        return
    for file in os.listdir(PLOT_PATH):
        if file.endswith(".png"):
            try:
                os.remove(os.path.join(PLOT_PATH, file))
                await log(f"🗑️ Removed old chart: {file}")
            except Exception as e:
                await log(f"⚠️ Error deleting file {file}: {e}")

# --- Safe environment preparation when the cache is disabled ---
async def rotate_or_clean_previous_outputs():
    """
    If use_cache=false, guarantee a clean run WITHOUT manual deletion:
    - Move previous merged.* outputs and CSVs from DATA_PATH to the archive (archived by date),
      or delete them if the move fails.
    - Clean the current charts folder.
    """
    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    arc_dir = os.path.join(ARCHIVE_PATH, f"run_{run_tag}")
    os.makedirs(arc_dir, exist_ok=True)

    # 1) Move merged outputs
    for path in [merged_csv_path, merged_json_path]:
        if os.path.isfile(path):
            try:
                shutil.move(path, os.path.join(arc_dir, os.path.basename(path)))
                await log(f"🗃️ Moved previous file: {path} → {arc_dir}")
            except Exception as e:
                await log(f"⚠️ Failed to move {path}: {e}. Trying to delete…")
                try:
                    os.remove(path)
                    await log(f"🧨 Deleted previous file: {path}")
                except Exception as e2:
                    await log(f"❌ Failed to delete {path}: {e2}")

    # 2) Move/clean raw CSVs from DATA_PATH
    if os.path.isdir(DATA_PATH):
        csvs = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
        if csvs:
            arc_data = os.path.join(arc_dir, 'data_csv')
            os.makedirs(arc_data, exist_ok=True)
            for f in csvs:
                src = os.path.join(DATA_PATH, f)
                try:
                    shutil.move(src, os.path.join(arc_data, f))
                except Exception:
                    try:
                        os.remove(src)
                    except Exception as e:
                        await log(f"⚠️ Failed to remove {src}: {e}")
            await log(f"🗃️ Archived raw CSV files: {len(csvs)} → {arc_data}")
        else:
            await log("ℹ️ No raw CSV files to clean — continuing.")

    # 3) Clean today's charts
    await clear_old_plots()

async def main():
    # 1. Check the internet connection
    await check_internet_connection()
    # 2. Check exchange availability
    global failed_exchanges
    failed_exchanges = await check_exchanges_health()
    # 3. Clear old charts (just in case)
    await clear_old_plots()

    cfg_use_cache = bool(config.get('use_cache', True))
    csv_exists = os.path.isfile(merged_csv_path)
    json_exists = os.path.isfile(merged_json_path)

    # If the cache is enabled and the required files exist, use it
    if cfg_use_cache and ((output_format == 'csv' and csv_exists) or
                          (output_format == 'json' and json_exists) or
                          (output_format == 'both' and csv_exists and json_exists)):
        await log("⚡ Found a previously saved file with indicators, using cache.")
        # Load only one source (CSV preferred), then normalize the timestamp
        cached_df = None
        try:
            if output_format in ('csv', 'both') and csv_exists:
                cached_df = pd.read_csv(merged_csv_path, parse_dates=['timestamp'])
            elif output_format in ('json', 'both') and json_exists:
                cached_df = pd.read_json(merged_json_path)
                if not np.issubdtype(cached_df['timestamp'].dtype, np.datetime64):
                    cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'])
        except Exception as e:
            await log(f"⚠️ Failed to load cache ({e}). Falling back to a fresh collection…")
            cached_df = None

        if cached_df is not None and not cached_df.empty:
            cached_df.sort_values(by='timestamp', inplace=True)
            checker = DataQualityChecker(log_func=log)
            await checker.check_missing_timestamps(cached_df, timeframe=TIMEFRAME)
            await checker.check_outliers(cached_df, price_column='close')

            # Save the missing output format
            if output_format in ("csv", "both") and not csv_exists:
                cached_df.to_csv(merged_csv_path, index=False)
                await log(f"📁 Final CSV file saved: {merged_csv_path}")
            if output_format in ("json", "both") and not json_exists:
                cached_df.to_json(merged_json_path, orient='records', date_format='iso', date_unit='s')
                await log(f"📁 Final JSON file saved: {merged_json_path}")

            start_time = datetime.now()
            await generate_plots_and_summary(cached_df, 0, start_time, cache_used=True)
            await log("✅ Pipeline finished.")
            return
        else:
            await log("⚠️ Cache is empty/corrupted — running a fresh collection.")

    # If the cache is disabled, prepare a clean environment (archive previous outputs)
    if not cfg_use_cache:
        await log("🧼 use_cache=false → performing a clean run without manual deletion.")
        await rotate_or_clean_previous_outputs()

    # Fresh collection
    await collect_data()
    await merge_and_analyze()
    await log("✅ Pipeline finished.")

if __name__ == "__main__":
    # Automatically restart the script on failure
    async def runner():
        while True:
            try:
                await main()
                break
            except Exception as e:
                await log(f"💀 Script crashed: {e}, restarting in 5 minutes...")
                if telegram_enabled:
                    await send_telegram_message(f"💀 Script crashed: {e}, restarting in 5 minutes...")
                await asyncio.sleep(300)
    asyncio.run(runner())
