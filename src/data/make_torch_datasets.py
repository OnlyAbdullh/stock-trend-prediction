import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.stock_dataset import StockDataset

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
from tqdm import tqdm
import pandas as pd
def get_data_set(window_size = 60):
    print("Loading CSV...")
    df = pd.read_csv('../data/processed/data.csv')

    ticker_data = {}
    samples = []
    horizon = 30
    FEATURE_COLS = []
    for ticker, group in tqdm(df.groupby('ticker'), desc="Processing tickers"):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)

        if n < window_size + horizon:
            continue


        close =  group['close'].values.astype(np.float32)
        missing =  group["missing_days"].values.astype(np.int8)
        bad = (missing > 0).astype(np.int32)
        bad_cumsum = np.cumsum(bad)

        def has_gap(a, b):
            return bad_cumsum[b] - (bad_cumsum[a - 1] if a > 0 else 0) > 0

        ticker_data[ticker] = group[FEATURE_COLS].values.astype(np.float32)

        for i in range(window_size, n - horizon):
            seq_start = i - window_size + 1
            seq_end = i + horizon

            if has_gap(seq_start,seq_end):
                continue

            label = 1 if close[i + horizon] > close[i] else 0

            samples.append((ticker, i, label))

    print(f"✓ Processed {len(samples):,} samples from {len(ticker_data)} tickers")
    return StockDataset(ticker_data, samples, window_size=window_size, horizon=30)