import json
import random
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple

from src.data.stock_dataset import StockDataset
from sklearn.preprocessing import RobustScaler, StandardScaler

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
from tqdm import tqdm
import pandas as pd
def build_samples(window_size = 60,use_cache = True):
    
    print("Loading CSV...")
    df = pd.read_csv('data/processed/data.csv')
    ticker_data = {}
    samples = []
    horizon = 30
    FEATURE_COLS = [
    # Price Features (3)
    'daily_return',
    'high_low_ratio',

    # MA-Based (4)
    'price_to_MA5',
    'price_to_MA20',
    'price_to_MA60',
    'MA_60_slope',

    # Volatility (3)
    'volatility_20',
    'RSI_14',
    'parkinson_volatility',

    # Critical Features (4)
    'recent_high_20',
    'distance_from_high',
    'low_to_close_ratio',
    'price_position_20',
    'max_drawdown_20',
    'downside_deviation_10',

    # Temporal (3)
    'month_sin',
    'month_cos',
    'is_up_day',

    # Volume Price Index (3) - Highest MI!
    'PVT_cumsum',           # MI = 0.0426 ⭐️⭐️⭐️
    'MOBV',                 # MI = 0.0209 ⭐️⭐️

    # Directional Movement (4)
    'MTM',                  # MI = 0.0127 ⭐️

    # OverBought & OverSold (1)
    'ADTM',                 # MI = 0.0104

    # Energy & Volatility (2)
    'PSY',                  # MI = 0.0085
    'VHF',                  # MI = 0.0088

    # Stochastic (1)
    'K',                    # MI = 0.0083

    # Raw Features
    ]
    for ticker, group in tqdm(df.groupby('ticker'), desc="Processing tickers"):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)

        if n < window_size + horizon:
            continue

        close = group['close'].values.astype(np.float32)
        missing = group["missing_days"].values.astype(np.int8)
        bad = (missing > 0).astype(np.int32)
        bad_cumsum = np.cumsum(bad)

        def has_gap(a, b):
            return bad_cumsum[b] - (bad_cumsum[a - 1] if a > 0 else 0) > 0

        ticker_data[ticker] = group[FEATURE_COLS].values.astype(np.float32)

        dates = group['date'].values
        for i in range(window_size, n - horizon):
            seq_start = i - window_size + 1
            seq_end = i + horizon

            if has_gap(seq_start, seq_end):
                continue

            label = 1 if close[i + horizon] > close[i] else 0

            date = dates[i]
            samples.append((ticker, i, label, date))


    StockDataset.ticker_data = ticker_data
    print(f"✓ Processed {len(samples):,} samples from {len(ticker_data)} tickers")
    return samples, ticker_data


def split_samples_time_based(
        samples: List[Tuple[str, int, int, object]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
):
    samples_sorted = sorted(samples, key=lambda x: x[3])

    n = len(samples_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_samples = samples_sorted[:n_train]
    val_samples = samples_sorted[n_train:n_train + n_val]
    test_samples = samples_sorted[n_train + n_val:]
    return train_samples, val_samples, test_samples


def normalize_ticker_data(ticker_data, train_samples):
    """Normalize features: fit on train, transform on all"""

    # Feature groups
    no_scale = ['is_up_day', 'month_sin', 'month_cos', 'price_position_20']
    robust_cols = [0, 1, 8, 9, 10, 11, 12, 13, 18, 19, 20]  # Indices of robust features
    zscore_cols = [2, 3, 4, 5, 6, 7, 14, 15, 16, 17]  # Indices of zscore features
    standard_cols = [21]  # Index of K

    print("Collecting training data for normalization...")
    train_data = []
    for ticker, i, _, _ in train_samples:
        train_data.append(ticker_data[ticker][i])
    train_data = np.array(train_data)

    print("Fitting scalers on training data...")
    robust_scaler = RobustScaler()
    zscore_scaler = StandardScaler()
    standard_scaler = StandardScaler()

    robust_scaler.fit(train_data[:, robust_cols])
    zscore_scaler.fit(train_data[:, zscore_cols])
    standard_scaler.fit(train_data[:, standard_cols])

    print("Normalizing all ticker data...")
    normalized_ticker_data = {}
    for ticker, data in tqdm(ticker_data.items(), desc="Normalizing"):
        normalized = data.copy()
        normalized[:, robust_cols] = robust_scaler.transform(data[:, robust_cols])
        normalized[:, zscore_cols] = zscore_scaler.transform(data[:, zscore_cols])
        normalized[:, standard_cols] = standard_scaler.transform(data[:, standard_cols])
        normalized_ticker_data[ticker] = normalized


    print("✓ Normalization complete (fit on train, transformed all)")
    return normalized_ticker_data

