import json
import random
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple

from src.data.stock_dataset import StockDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy import stats

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
from tqdm import tqdm
import pandas as pd
def build_samples(window_size = 60):
    
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
    del df
    import gc
    gc.collect()

    # StockDataset.ticker_data = ticker_data
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

    robust_cols = [0, 1, 8, 9, 10, 11, 12, 13, 18, 19, 20]
    zscore_cols = [2, 3, 4, 5, 6, 7, 14, 15, 16, 17]
    standard_cols = [21]  # K

    # ميزات تحتاج Winsorization
    # volatility_20(6), high_low_ratio(1), parkinson_volatility(8), price_to_MA...(2,3,4)
    winsor_params = {
        6: (1, 99), 1: (1, 99), 8: (1, 99),
        2: (0.5, 99), 3: (0.5, 99), 4: (0.5, 99)
    }

    # ميزات Box-Cox (الميزات الموجبة ذات الالتواء العالي)
    boxcox_indices = [6, 1, 8]

    def _process_array_logic(data_array, is_training=False, scalers=None):
        arr = data_array.copy()

        for idx in [18, 19]:
            prev_vals = np.roll(arr[:, idx], 1)
            prev_vals[0] = arr[0, idx]

            denom = np.where(prev_vals == 0, 1e-9, prev_vals)
            pct_change = (arr[:, idx] - prev_vals) / denom
            arr[:, idx] = np.clip(pct_change, -0.5, 0.5)

        # 2. تطبيق Winsorization
        for col_idx, (low, high) in winsor_params.items():
            l_val = np.percentile(arr[:, col_idx], low)
            h_val = np.percentile(arr[:, col_idx], high)
            arr[:, col_idx] = np.clip(arr[:, col_idx], l_val, h_val)

        # 3. تطبيق Box-Cox
        for col_idx in boxcox_indices:
            shift = np.min(arr[:, col_idx])
            shifted_data = arr[:, col_idx] - shift + 1.0
            transformed, _ = stats.boxcox(shifted_data)
            arr[:, col_idx] = transformed

        if not is_training and scalers:
            # 4. Scaling
            arr[:, robust_cols] = scalers['robust'].transform(arr[:, robust_cols])
            arr[:, zscore_cols] = scalers['zscore'].transform(arr[:, zscore_cols])
            arr[:, standard_cols] = scalers['standard'].transform(arr[:, standard_cols])

            # 5. Final Clipping ±10 std
            # استثناء الميزات الزمنية (15, 16, 17) و price_position_20 (12)
            skip_clip = [12, 15, 16, 17]
            for c in range(arr.shape[1]):
                if c not in skip_clip:
                    arr[:, c] = np.clip(arr[:, c], -10, 10)

        return arr

    print("Preparing Training Data for Fitting...")
    raw_train_list = []
    for ticker, i, _, _ in train_samples:
        raw_train_list.append(ticker_data[ticker][i])
    train_data_stack = np.array(raw_train_list)

    processed_train = _process_array_logic(train_data_stack, is_training=True)

    print("Fitting Scalers...")
    scalers = {
        'robust': RobustScaler().fit(processed_train[:, robust_cols]),
        'zscore': StandardScaler().fit(processed_train[:, zscore_cols]),
        'standard': StandardScaler().fit(processed_train[:, standard_cols])
    }

    print("Normalizing all ticker data...")
    normalized_ticker_data = {}
    for ticker, data in tqdm(ticker_data.items(), desc="Normalizing"):
        normalized_ticker_data[ticker] = _process_array_logic(data, is_training=False, scalers=scalers)

    print("✓ Normalization complete (All test-test.py steps applied)")
    return normalized_ticker_data
