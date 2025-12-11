from typing import Tuple

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_FRACTION


def load_raw_data() -> pd.DataFrame:
    """Load the raw merged CSV."""
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])


def save_raw_data(df: pd.DataFrame) -> None:
    """Save raw merged data."""
    df.to_csv(RAW_DATA_PATH, index=False)


def save_processed_data(df: pd.DataFrame) -> None:
    """Save processed feature data."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def load_processed_data() -> pd.DataFrame:
    """Load processed feature data."""
    return pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])


def time_split(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform a simple time-based train/test split on the processed features.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    y = df[target_col]
    timestamps = df["timestamp"]

    drop_cols = ["timestamp", "city", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]

    order = np.arange(len(df))
    n_train = int(len(df) * TRAIN_FRACTION)
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    ts_test = timestamps.iloc[test_idx]

    return X_train, X_test, y_train, y_test, ts_test
