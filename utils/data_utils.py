import numpy as np
import pandas as pd

from utils.features import (
    add_lag_features,
    add_technical_indicators,
    make_features,
    preprocess_data,
)
from utils.prep_utils import (
    augment_data,
    create_sequences,
    scale_data,
    split_data,
    train_val_test_split,
)


def process_large_csv(filepath, process_fn, chunk_size=100_000):
    """Apply process_fn to CSV chunks and return one concatenated DataFrame."""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunks.append(process_fn(chunk))
    return pd.concat(chunks, ignore_index=True)


def load_data(filepath):
    """Load csv or parquet data by file extension."""
    if str(filepath).endswith(".parquet"):
        return pd.read_parquet(filepath)
    return pd.read_csv(filepath)


def save_bigfile_to_parquet(raw_csv, out_parquet, window=14):
    """Generate features for a large CSV in chunks and save a parquet copy."""
    def pipeline(chunk):
        return make_features(chunk, window=window)

    df_ready = process_large_csv(raw_csv, pipeline)
    df_ready.to_parquet(out_parquet, index=False)
    print(f"Saved parquet: {out_parquet}, shape: {df_ready.shape}")


def clean_dataframe(df, fill_method="drop", fill_value=0):
    """
    Clean NaN and infinite values from a DataFrame.

    fill_method:
    - "drop": remove rows with NaN/inf
    - "fill": replace NaN/inf with fill_value
    - "mean": replace numeric NaN values with column mean
    """
    rows_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    if fill_method == "drop":
        df = df.dropna()
    elif fill_method == "fill":
        df = df.fillna(fill_value)
    elif fill_method == "mean":
        df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x)
    else:
        raise ValueError("fill_method must be 'drop', 'fill' or 'mean'.")

    print(f"Before cleaning: {rows_before} rows, after cleaning: {len(df)} rows")
    if df.isnull().sum().sum() > 0:
        print("⚠️ NaN values remain in the data!")

    return df


if __name__ == "__main__":
    save_bigfile_to_parquet("../data/raw/BTCUSDT.csv", "../data/processed/BTCUSDT_ready.parquet")
