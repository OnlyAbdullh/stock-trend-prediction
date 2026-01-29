import json
import random
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Dict

from src.data.stock_dataset import StockDataset
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler
from scipy import stats
from torch.utils.data import DataLoader

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
from tqdm import tqdm
import pandas as pd

mp = {}

FEATURE_COLS = [
        "daily_return",
        "high_low_ratio",
        "price_to_MA5",
        "price_to_MA20",
        "price_to_MA60",
        "MA_60_slope",
        "volatility_20",
        "RSI_14",
        "parkinson_volatility",
        "recent_high_20",
        "distance_from_high",
        "low_to_close_ratio",
        "price_position_20",
        "max_drawdown_20",
        "downside_deviation_10",
        "month_sin",
        "month_cos",
        "is_up_day",
        "PVT_cumsum",
        "MOBV",
        "MTM",
        "ADTM",
        "PSY",
        "VHF",
        "K",
    ]

def build_samples_from_df(
    df: pd.DataFrame,
    window_size: int = 60,
    horizon: int = 30,
) -> Tuple[List, Dict]:
    ticker_data = {}
    samples = []


    for i, col in enumerate(FEATURE_COLS):
        mp[col] = i
    for ticker, group in tqdm(df.groupby("ticker"), desc="Building samples"):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)

        if n < window_size + horizon:
            continue

        close = group["close"].values.astype(np.float32)
        missing = group["missing_days"].values.astype(np.int8)
        bad = (missing > 0).astype(np.int32)
        bad_cumsum = np.cumsum(bad)

        def has_gap(a, b):
            return bad_cumsum[b] - (bad_cumsum[a - 1] if a > 0 else 0) > 0

        # Store feature data
        ticker_data[ticker] = group[FEATURE_COLS].values.astype(np.float32)

        dates = group["date"].values
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
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df

# Mode 1
no_need_scaling = [
    "is_up_day",
    "month_sin",
    "month_cos",
    "price_position_20",
]

robust_scaling_features = [
    "distance_from_high",
    "downside_deviation_10",
    "high_low_ratio",
    "low_to_close_ratio",
    "max_drawdown_20",
    "parkinson_volatility",
    "recent_high_20",
    "volatility_20",
    "VHF",
    "MOBV",
    "PVT_cumsum",
]

zscore_features = [
    "ADTM",
    "daily_return",
    "MA_60_slope",
    "MTM",
    "price_to_MA5",
    "price_to_MA20",
    "price_to_MA60",
    "PSY",
    "RSI_14",
]

standard_scaler_features = ["K"]

# Mode 2
z_only = ["ADTM", "K", "price_position_20", "RSI_14"]
z_plus_q = [
    "daily_return", "MA_60_slope", "MOBV", "MTM",
    "price_to_MA5", "price_to_MA20", "price_to_MA60",
    "PVT_cumsum", "recent_high_20"
]
robust_only = ["max_drawdown_20", "VHF"]
robust_plus_q = [
    "distance_from_high", "downside_deviation_10",
    "high_low_ratio", "low_to_close_ratio", "parkinson_volatility"
]
robust_plus_z = ["volatility_20"]
max_abs_only = ["PSY"]
no_scaling_mode2 = ["is_up_day", "month_sin", "month_cos"]

def normalize_df(df_train, df_val, df_test, norm_mode="norm1"):
    if norm_mode == "norm1":
        return normalize_df_mode1(df_train, df_val, df_test)
    elif norm_mode == "norm2":
        return normalize_df_mode2(df_train, df_val, df_test)

    return df_train, df_val, df_test


def normalize_df_mode1(df_train, df_val, df_test):
    scalars = {
        "robust": RobustScaler().fit(df_train[robust_scaling_features]),
        "z": StandardScaler().fit(df_train[zscore_features]),
        "std": StandardScaler().fit(df_train[standard_scaler_features]),
    }
    return (
        normalize_df_sc1(df_train, scalars),
        normalize_df_sc1(df_val, scalars),
        normalize_df_sc1(df_test, scalars),
    )


def normalize_df_sc1(df, scalers):
    df_scaled = df.copy()
    df_scaled[robust_scaling_features] = scalers["robust"].transform(
        df[robust_scaling_features]
    )
    df_scaled[zscore_features] = scalers["z"].transform(df[zscore_features])
    df_scaled[standard_scaler_features] = scalers["std"].transform(
        df[standard_scaler_features]
    )
    return df_scaled


def get_q_limits(df, columns):
    """حساب حدود 1% و 99% من بيانات التدريب فقط"""
    limits = {}
    for col in columns:
        limits[col] = (df[col].quantile(0.01), df[col].quantile(0.99))
    return limits


def apply_clipping(df, limits):
    """تطبيق القص بناءً على الحدود المحسوبة مسبقاً"""
    df_clipped = df.copy()
    for col, (low, high) in limits.items():
        if col in df_clipped.columns:
            df_clipped[col] = df_clipped[col].clip(lower=low, upper=high)
    return df_clipped


def normalize_df_mode2(df_train, df_val, df_test):
    # 1. حساب حدود Q (1-99) من التدريب فقط
    q_columns = z_plus_q + robust_plus_q
    q_limits = get_q_limits(df_train, q_columns)

    # 2. تطبيق القص (Clipping)
    train_c = apply_clipping(df_train, q_limits)
    val_c = apply_clipping(df_val, q_limits)
    test_c = apply_clipping(df_test, q_limits)

    # 3. تجهيز الـ Scalers بالاعتماد على بيانات التدريب (بعد القص)
    scalers = {
        "z": StandardScaler().fit(train_c[z_only + z_plus_q]),
        "robust": RobustScaler().fit(train_c[robust_only + robust_plus_q]),
        "max_abs": MaxAbsScaler().fit(train_c[max_abs_only]),
        # حالة خاصة Volatility: Robust ثم Z (سنطبق Robust أولاً للجميع)
        "vol_robust": RobustScaler().fit(train_c[robust_plus_z])
    }

    # حساب Z-score لـ volatility بعد تحويلها بـ Robust
    vol_robust_train = scalers["vol_robust"].transform(train_c[robust_plus_z])
    scalers["vol_z"] = StandardScaler().fit(vol_robust_train)

    return (
        normalize_df_sc2(train_c, scalers),
        normalize_df_sc2(val_c, scalers),
        normalize_df_sc2(test_c, scalers)
    )

def normalize_df_sc2(df_in, scalers):
    df_out = df_in.copy()
    # Z-Score
    df_out[z_only + z_plus_q] = scalers["z"].transform(df_in[z_only + z_plus_q])
    # Robust
    df_out[robust_only + robust_plus_q] = scalers["robust"].transform(df_in[robust_only + robust_plus_q])
    # Max Abs
    df_out[max_abs_only] = scalers["max_abs"].transform(df_in[max_abs_only])
    # Volatility (R + Z)
    vol_r = scalers["vol_robust"].transform(df_in[robust_plus_z])
    df_out[robust_plus_z] = scalers["vol_z"].transform(vol_r)

    return df_out