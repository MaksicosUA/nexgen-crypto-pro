import numpy as np
import pandas as pd

def preprocess_data(df, drop_na=True, sort_date_col='date'):
    if sort_date_col in df.columns:
        df = df.sort_values(sort_date_col)
    if drop_na:
        df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def add_lag_features(df, price_col='close', lags=[1,2,3,7,14]):
    for lag in lags:
        df[f"{price_col}_lag{lag}"] = df[price_col].shift(lag)
    return df

def add_technical_indicators(df, price_col='close', volume_col='volume', window=14):
    df['sma'] = df[price_col].rolling(window).mean()
    df['ema'] = df[price_col].ewm(span=window, adjust=False).mean()
    df['return'] = df[price_col].pct_change() * 100
    df['std'] = df[price_col].rolling(window).std()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df[price_col].shift())
    df['low_close'] = np.abs(df['low'] - df[price_col].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window).mean()
    df['bb_upper'] = df['sma'] + (df['std'] * 2)
    df['bb_lower'] = df['sma'] - (df['std'] * 2)
    delta = df[price_col].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    tp = (df['high'] + df['low'] + df[price_col]) / 3
    cci_ma = tp.rolling(window).mean()
    cci_std = tp.rolling(window).std()
    df['cci'] = (tp - cci_ma) / (0.015 * cci_std)
    highest_high = df['high'].rolling(window).max()
    lowest_low = df['low'].rolling(window).min()
    df['williams_r'] = -100 * (highest_high - df[price_col]) / (highest_high - lowest_low + 1e-10)
    return df

def make_features(df, window=14, price_col='close', volume_col='volume'):
    df = preprocess_data(df)
    df = add_technical_indicators(df, price_col, volume_col, window)
    df = add_lag_features(df, price_col, lags=[1,2,3,7,14])
    df = df.dropna()
    exclude = ['date', 'symbol', 'exchange', 'open', 'high', 'low']
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols]
