import json
import random
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Dict

from src.data.stock_dataset import StockDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy import stats
from torch.utils.data import DataLoader

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
from tqdm import tqdm
import pandas as pd

mp = {}


def build_samples_from_df(
        df: pd.DataFrame,
        window_size: int = 60,
        horizon: int = 30,
) -> Tuple[List, Dict]:
    ticker_data = {}
    samples = []

    FEATURE_COLS = [
        # Price Features (2)
        'daily_return',  # 0
        'high_low_ratio',  # 1

        # MA-Based (4)
        'price_to_MA5',  # 2
        'price_to_MA20',  # 3
        'price_to_MA60',  # 4
        'MA_60_slope',  # 5

        # Volatility (3)
        'volatility_20',  # 6
        'RSI_14',  # 7
        'parkinson_volatility',  # 8

        # Critical Features (6)
        'recent_high_20',  # 9
        'distance_from_high',  # 10
        'low_to_close_ratio',  # 11
        'price_position_20',  # 12
        'max_drawdown_20',  # 13
        'downside_deviation_10',  # 14

        # Temporal (3)
        'month_sin',  # 15
        'month_cos',  # 16
        'is_up_day',  # 17

        # Volume Price Index (2)
        'PVT_cumsum',  # 18
        'MOBV',  # 19

        # Directional Movement (1)
        'MTM',  # 20

        # OverBought & OverSold (1)
        'ADTM',  # 21

        # Energy & Volatility (2)
        'PSY',  # 22
        'VHF',  # 23

        # Stochastic (1)
        'K',  # 24
    ]

    for i, col in enumerate(FEATURE_COLS):
        mp[col] = i
    for ticker, group in tqdm(df.groupby('ticker'), desc="Building samples"):
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

        # Store feature data
        ticker_data[ticker] = group[FEATURE_COLS].values.astype(np.float32)

        dates = group['date'].values
        for i in range(window_size - 1, n - horizon):
            seq_start = i - window_size + 1
            seq_end = i + horizon

            if has_gap(seq_start, seq_end):
                continue

            label = 1 if close[i + horizon] > close[i] else 0
            date = dates[i]
            samples.append((ticker, i, label, date))

    return samples, ticker_data


def split_dataframe_by_date(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values('date').reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


no_need_scaling = [
    'is_up_day',
    'month_sin',
    'month_cos',
    'price_position_20',
]

robust_scaling_features = [
    'distance_from_high',
    'downside_deviation_10',
    'high_low_ratio',
    'low_to_close_ratio',
    'max_drawdown_20',
    'parkinson_volatility',
    'recent_high_20',
    'volatility_20',
    'VHF',
    'MOBV',
    'PVT_cumsum'
]

zscore_features = [
    'ADTM',
    'daily_return',
    'MA_60_slope',
    'MTM',
    'price_to_MA5',
    'price_to_MA20',
    'price_to_MA60',
    'PSY',
    'RSI_14',
]

standard_scaler_features = [
    'K'
]


def normalize_df(df_train, df_val, df_test, norm_mode="norm1"):
    if norm_mode == 'norm1':
        df_train, df_val, df_test = normalize_df_mode1(df_train, df_val, df_test)
    # elif norm_mode == 'norm2':
    #     normalized_data = normalize_ticker_data_mode2(ticker_data,train_samples)
    # elif norm_mode == 'norm3':
    #     normalized_data = normalize_ticker_data_mode3(ticker_data,train_samples)
    # elif norm_mode == 'norm4':
    #     normalized_data = normalize_ticker_data_mode4(ticker_data,train_samples)
    # else:
    #     normalized_data = normalize_ticker_data_mode5(ticker_data,train_samples)

    return df_train, df_val, df_test


def normalize_df_mode1(df_train, df_val, df_test):
    scalars = {
        "robust": RobustScaler().fit(df_train[robust_scaling_features]),
        "z": StandardScaler().fit(df_train[zscore_features]),
        "std": StandardScaler().fit(df_train[standard_scaler_features]),
    }
    return normalize_df_sc(df_train, scalars), normalize_df_sc(df_val, scalars), normalize_df_sc(df_test, scalars)


def normalize_df_sc(df, scalers):
    df_scaled = df.copy()
    df_scaled[robust_scaling_features] = scalers["robust"].transform(
        df[robust_scaling_features]
    )
    df_scaled[zscore_features] = scalers["z"].transform(
        df[zscore_features]
    )
    df_scaled[standard_scaler_features] = scalers["std"].transform(
        df[standard_scaler_features]
    )
    return df_scaled
