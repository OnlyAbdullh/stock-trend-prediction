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
mp = {}
def build_samples(window_size = 60):
    
    print("Loading CSV...")
    df = pd.read_csv('data/processed/data.csv')
    ticker_data = {}
    samples = []
    horizon = 30
    FEATURE_COLS = [
    # Price Features (3)
    'daily_return',#0
    'high_low_ratio',# 1

    # MA-Based (4)
    'price_to_MA5',# 2
    'price_to_MA20',# 3
    'price_to_MA60',# 4
    'MA_60_slope',# 5

    # Volatility (3)
    'volatility_20',# 6
    'RSI_14',# 7
    'parkinson_volatility',# 8

    # Critical Features (4)
    'recent_high_20',#9
    'distance_from_high',#10
    'low_to_close_ratio',#11
    'price_position_20',#12
    'max_drawdown_20',#13
    'downside_deviation_10',# 14

    # Temporal (3)
    'month_sin',#15
    'month_cos',#16
    'is_up_day',#17

    # Volume Price Index (3) - Highest MI!
    'PVT_cumsum',   #18        # MI = 0.0426 ⭐️⭐️⭐️
    'MOBV',            #19     # MI = 0.0209 ⭐️⭐️

    # Directional Movement (4)
    'MTM',            #20      # MI = 0.0127 ⭐️

    # OverBought & OverSold (1)
    'ADTM',      #21           # MI = 0.0104

    # Energy & Volatility (2)
    'PSY',          #22        # MI = 0.0085
    'VHF',          #23        # MI = 0.0088

    # Stochastic (1)
    'K',               #24     # MI = 0.0083

    # Raw Features
    ]
    for i,col in enumerate(FEATURE_COLS):
        mp[col] = i
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


def normalize_ticker_data(ticker_data, train_samples, normalization_mode):
    if normalization_mode == 'norm1':
        normalized_data = normalize_ticker_data_mode1(ticker_data,train_samples)
    elif normalization_mode == 'norm2':
        normalized_data = normalize_ticker_data_mode2(ticker_data,train_samples)
    elif normalization_mode == 'norm3':
        normalized_data = normalize_ticker_data_mode3(ticker_data,train_samples)
    elif normalization_mode == 'norm4':
        normalized_data = normalize_ticker_data_mode4(ticker_data,train_samples)
    else:
        normalized_data = normalize_ticker_data_mode5(ticker_data,train_samples)

    return normalized_data

def normalize_ticker_data_mode1(ticker_data, train_samples):
    """Normalize features: fit on train, transform on all"""

    robust_cols = [mp[c] for c in robust_scaling_features]
    zscore_cols = [mp[c] for c in zscore_features]
    standard_cols = [mp[c] for c in standard_scaler_features]

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

    print("✓ Normalization Mode 1 complete (fit on train, transformed all)")
    return normalized_ticker_data

def normalize_ticker_data_mode2(ticker_data, train_samples):
    """
        1. تحويل الميزات التراكمية إلى نسبة تغير.
        2. Winsorization (1%, 99%).
        3. Box-Cox للبيانات الملتوية.
        4. Scaling (Robust, Z-Score, Standard).
        5. القص النهائي عند ±4 انحرافات معيارية.
        """

    # تحديد الفهارس باستخدام القاموس mp
    robust_cols = [mp[c] for c in robust_scaling_features]
    zscore_cols = [mp[c] for c in zscore_features]
    standard_cols = [mp[c] for c in standard_scaler_features]

    # ميزات تحتاج Winsorization عدواني (حسب test-test)
    winsor_map = {
        mp['volatility_20']: (1, 99),
        mp['high_low_ratio']: (1, 99),
        mp['parkinson_volatility']: (1, 99),
        mp['price_to_MA5']: (0.5, 99.5),
        mp['price_to_MA20']: (0.5, 99.5),
        mp['price_to_MA60']: (0.5, 99.5)
    }

    # ميزات تحتاج تحويل Box-Cox
    boxcox_cols = [mp['volatility_20'], mp['high_low_ratio'], mp['parkinson_volatility']]

    # الميزات التراكمية التي سنحولها لنسبة تغير
    cum_cols = [mp['PVT_cumsum'], mp['MOBV']]

    def _apply_advanced_transforms(data_arr):
        """تابع داخلي لمعالجة المصفوفات بنفس المنطق للتدريب والاختبار"""
        arr = data_arr.copy()

        # أ. تحويل Cumulative إلى % Change (معالجة يدوية للمصفوفة)
        for col in cum_cols:
            # حساب التغير: (current - previous) / previous
            shifted = np.roll(arr[:, col], 1)
            shifted[0] = arr[0, col]
            denom = np.where(shifted == 0, 1e-9, shifted)
            pct_change = (arr[:, col] - shifted) / denom
            arr[:, col] = np.clip(pct_change, -0.5, 0.5)

        # ب. Winsorization
        for col, (low, high) in winsor_map.items():
            l_val = np.percentile(arr[:, col], low)
            h_val = np.percentile(arr[:, col], high)
            arr[:, col] = np.clip(arr[:, col], l_val, h_val)

        # ج. Box-Cox (تأكد من أن القيم موجبة)
        for col in boxcox_cols:
            min_val = np.min(arr[:, col])
            shifted_data = arr[:, col] - min_val + 1.0
            # نستخدم stats.boxcox مباشرة (التبسيط المستخدم في test-test)
            transformed, _ = stats.boxcox(shifted_data)
            arr[:, col] = transformed

        return arr

    print("Preparing Training Data for Fitting...")
    raw_train = np.array([ticker_data[t][i] for t, i, _, _ in train_samples])

    # معالجة بيانات التدريب قبل حساب المتوسطات والانحرافات
    processed_train = _apply_advanced_transforms(raw_train)

    print("Fitting Scalers...")
    robust_scaler = RobustScaler().fit(processed_train[:, robust_cols])
    zscore_scaler = StandardScaler().fit(processed_train[:, zscore_cols])
    standard_scaler = StandardScaler().fit(processed_train[:, standard_cols])

    print("Normalizing all ticker data (Mode 2)...")
    normalized_ticker_data = {}

    # ميزات لا يجب قصها نهائياً (مثل الوقت والنسب الثنائية)
    skip_clip_cols = [mp[c] for c in no_need_scaling]

    for ticker, data in tqdm(ticker_data.items(), desc="Normalizing"):
        # 1. تطبيق التحويلات المتقدمة
        norm = _apply_advanced_transforms(data)

        # 2. تطبيق الـ Scalers
        norm[:, robust_cols] = robust_scaler.transform(norm[:, robust_cols])
        norm[:, zscore_cols] = zscore_scaler.transform(norm[:, zscore_cols])
        norm[:, standard_cols] = standard_scaler.transform(norm[:, standard_cols])

        # 3. القص النهائي (Final Clipping) عند ±4 لضمان استقرار الشبكة العصبية
        all_scaled = robust_cols + zscore_cols + standard_cols
        for col in all_scaled:
            if col not in skip_clip_cols:
                norm[:, col] = np.clip(norm[:, col], -4, 4)

        normalized_ticker_data[ticker] = norm

    print("✓ Normalization Mode 2 complete (Advanced Logic Applied)")
    return normalized_ticker_data

def normalize_ticker_data_mode3(ticker_data, train_samples):
    """
           1. تحويل الميزات التراكمية إلى نسبة تغير.
           2. Winsorization (1%, 99%).
           3. Box-Cox للبيانات الملتوية.
           4. Scaling (Robust, Z-Score, Standard).
           """

    # تحديد الفهارس باستخدام القاموس mp
    robust_cols = [mp[c] for c in robust_scaling_features]
    zscore_cols = [mp[c] for c in zscore_features]
    standard_cols = [mp[c] for c in standard_scaler_features]

    # ميزات تحتاج Winsorization عدواني (حسب test-test)
    winsor_map = {
        mp['volatility_20']: (1, 99),
        mp['high_low_ratio']: (1, 99),
        mp['parkinson_volatility']: (1, 99),
        mp['price_to_MA5']: (0.5, 99.5),
        mp['price_to_MA20']: (0.5, 99.5),
        mp['price_to_MA60']: (0.5, 99.5)
    }

    # ميزات تحتاج تحويل Box-Cox
    boxcox_cols = [mp['volatility_20'], mp['high_low_ratio'], mp['parkinson_volatility']]

    # الميزات التراكمية التي سنحولها لنسبة تغير
    cum_cols = [mp['PVT_cumsum'], mp['MOBV']]

    def _apply_advanced_transforms(data_arr):
        """تابع داخلي لمعالجة المصفوفات بنفس المنطق للتدريب والاختبار"""
        arr = data_arr.copy()

        # أ. تحويل Cumulative إلى % Change (معالجة يدوية للمصفوفة)
        for col in cum_cols:
            # حساب التغير: (current - previous) / previous
            shifted = np.roll(arr[:, col], 1)
            shifted[0] = arr[0, col]
            denom = np.where(shifted == 0, 1e-9, shifted)
            pct_change = (arr[:, col] - shifted) / denom
            arr[:, col] = np.clip(pct_change, -0.5, 0.5)

        # ب. Winsorization
        for col, (low, high) in winsor_map.items():
            l_val = np.percentile(arr[:, col], low)
            h_val = np.percentile(arr[:, col], high)
            arr[:, col] = np.clip(arr[:, col], l_val, h_val)

        # ج. Box-Cox (تأكد من أن القيم موجبة)
        for col in boxcox_cols:
            min_val = np.min(arr[:, col])
            shifted_data = arr[:, col] - min_val + 1.0
            # نستخدم stats.boxcox مباشرة (التبسيط المستخدم في test-test)
            transformed, _ = stats.boxcox(shifted_data)
            arr[:, col] = transformed

        return arr

    print("Preparing Training Data for Fitting...")
    raw_train = np.array([ticker_data[t][i] for t, i, _, _ in train_samples])

    # معالجة بيانات التدريب قبل حساب المتوسطات والانحرافات
    processed_train = _apply_advanced_transforms(raw_train)

    print("Fitting Scalers...")
    robust_scaler = RobustScaler().fit(processed_train[:, robust_cols])
    zscore_scaler = StandardScaler().fit(processed_train[:, zscore_cols])
    standard_scaler = StandardScaler().fit(processed_train[:, standard_cols])

    print("Normalizing all ticker data (Mode 3)...")
    normalized_ticker_data = {}

    for ticker, data in tqdm(ticker_data.items(), desc="Normalizing"):
        # 1. تطبيق التحويلات المتقدمة
        norm = _apply_advanced_transforms(data)

        # 2. تطبيق الـ Scalers
        norm[:, robust_cols] = robust_scaler.transform(norm[:, robust_cols])
        norm[:, zscore_cols] = zscore_scaler.transform(norm[:, zscore_cols])
        norm[:, standard_cols] = standard_scaler.transform(norm[:, standard_cols])

    print("✓ Normalization Mode 3 complete (Advanced Logic Applied without clipping)")
    return normalized_ticker_data

def normalize_ticker_data_mode4(ticker_data, train_samples):
    pass

def normalize_ticker_data_mode5(ticker_data, train_samples):
    pass
