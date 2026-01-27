import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("STOCK PRICE PREDICTION - FEATURE ANALYSIS")
print("="*80)
print("\nObjective: Determine optimal sequence length for RNN input")
print("Target: Predict if close price > current price after 30 trading days")
print("="*80)
# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL PREPROCESSING
# ============================================================================

print("\n[1] LOADING DATA...")
# Load the dataset
df = pd.read_csv(r'D:\Stock_trend_project\data\interim\train_clean_after_2010_and_bad_tickers.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
df = df[df['open'] != 0]

# Sort by ticker and date
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

print(f"Total records: {len(df):,}")
print(f"Unique tickers: {df['ticker'].nunique():,}")
print(f"date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display basic statistics
print("\n" + "="*80)
print("DATA OVERVIEW")
print("="*80)
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("[2] ENGINEERING FEATURES...")
print("="*80)

def engineer_features(df):
    """
    Engineer features and keep only selected high-quality features
    in the order they were originally created.
    """
    df = df.copy()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    grouped = df.groupby('ticker')

    # ------------------------------
    # Price Features
    # ------------------------------
    df['daily_return'] = grouped['close'].pct_change()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']

    # ------------------------------
    # Moving Averages
    # ------------------------------
    df['MA_5'] = grouped['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['MA_20'] = grouped['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['MA_60'] = grouped['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())

    # ------------------------------
    # MA-Based Features
    # ------------------------------
    df['price_to_MA5'] = (df['close'] - df['MA_5']) / (df['MA_5'] + 1e-8)
    df['price_to_MA20'] = (df['close'] - df['MA_20']) / (df['MA_20'] + 1e-8)
    df['price_to_MA60'] = (df['close'] - df['MA_60']) / (df['MA_60'] + 1e-8)
    df['MA_60_slope'] = grouped['MA_60'].pct_change(30)

    # ------------------------------
    # Volatility Features
    # ------------------------------
    df['volatility_20'] = grouped['daily_return'].transform(
        lambda x: x.rolling(20, min_periods=1).std()
    )

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = grouped['close'].transform(lambda x: calculate_rsi(x, 14))

    df['parkinson_volatility'] = grouped.apply(
        lambda x: np.sqrt(
            1/(4*np.log(2)) *
            ((np.log(x['high']/(x['low']+1e-8)))**2).rolling(10, min_periods=1).mean()
        )
    ).reset_index(level=0, drop=True)

    # ------------------------------
    # Support/Resistance & Risk
    # ------------------------------
    df['recent_high_20'] = grouped['high'].transform(lambda x: x.rolling(20, min_periods=1).max())
    df['recent_low_20'] = grouped['low'].transform(lambda x: x.rolling(20, min_periods=1).min())
    df['distance_from_high'] = (df['close'] - df['recent_high_20']) / (df['recent_high_20'] + 1e-8)
    df['low_to_close_ratio'] = df['recent_low_20'] / (df['close'] + 1e-8)
    df['price_position_20'] = (
        (df['close'] - df['recent_low_20']) /
        (df['recent_high_20'] - df['recent_low_20'] + 1e-8)
    )

    def max_drawdown(series, window):
        roll_max = series.rolling(window, min_periods=1).max()
        drawdown = (series - roll_max) / (roll_max + 1e-8)
        return drawdown.rolling(window, min_periods=1).min()

    df['max_drawdown_20'] = grouped['close'].transform(lambda x: max_drawdown(x, 20))
    df['downside_deviation_10'] = grouped['daily_return'].transform(
        lambda x: x.where(x < 0, 0).rolling(10, min_periods=1).std()
    )

    # ------------------------------
    # Temporal
    # ------------------------------
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    df['is_up_day'] = (df['daily_return'] > 0).astype(int)

    # ------------------------------
    # Volume Price Index (NEW)
    # ------------------------------
    df['price_change'] = grouped['close'].pct_change()
    df['PVT'] = (df['price_change'] * df['volume']).fillna(0)
    df['PVT_cumsum'] = grouped['PVT'].transform(lambda x: x.cumsum())

    df['MOBV_signal'] = np.where(df['price_change'] > 0, df['volume'],
                                  np.where(df['price_change'] < 0, -df['volume'], 0))
    df['MOBV'] = grouped['MOBV_signal'].transform(lambda x: x.cumsum())

    # ------------------------------
    # Directional Movement
    # ------------------------------
    df['MTM'] = df['close'] - grouped['close'].shift(12)

    # ------------------------------
    # OverBought & OverSold
    # ------------------------------
    df['DTM'] = np.where(df['open'] <= grouped['open'].shift(1),
                         0,
                         np.maximum(df['high'] - df['open'], df['open'] - grouped['open'].shift(1)))
    df['DBM'] = np.where(df['open'] >= grouped['open'].shift(1),
                         0,
                         np.maximum(df['open'] - df['low'], df['open'] - grouped['open'].shift(1)))
    df['DTM_sum'] = grouped['DTM'].transform(lambda x: x.rolling(23, min_periods=1).sum())
    df['DBM_sum'] = grouped['DBM'].transform(lambda x: x.rolling(23, min_periods=1).sum())
    df['ADTM'] = (df['DTM_sum'] - df['DBM_sum']) / (df['DTM_sum'] + df['DBM_sum'] + 1e-8)

    # ------------------------------
    # Energy & Volatility
    # ------------------------------
    df['PSY'] = grouped['is_up_day'].transform(lambda x: x.rolling(12, min_periods=1).mean()) * 100

    df['highest_close'] = grouped['close'].transform(lambda x: x.rolling(28, min_periods=1).max())
    df['lowest_close'] = grouped['close'].transform(lambda x: x.rolling(28, min_periods=1).min())
    df['close_diff_sum'] = grouped['close'].transform(lambda x: x.diff().abs().rolling(28, min_periods=1).sum())
    df['VHF'] = (df['highest_close'] - df['lowest_close']) / (df['close_diff_sum'] + 1e-8)

    # ------------------------------
    # Stochastic
    # ------------------------------
    df['lowest_low_9'] = grouped['low'].transform(lambda x: x.rolling(9, min_periods=1).min())
    df['highest_high_9'] = grouped['high'].transform(lambda x: x.rolling(9, min_periods=1).max())
    df['K'] = ((df['close'] - df['lowest_low_9']) / (df['highest_high_9'] - df['lowest_low_9'] + 1e-8)) * 100

    # ------------------------------
    # Cleanup temporary columns 41 - 16 = 26
    # ------------------------------
    temp_cols = [
        # 'MA_5', 'MA_20', 'MA_60',
        'price_change', 'PVT', 'MOBV_signal',
        'DTM', 'DBM', 'DTM_sum', 'DBM_sum',
        'highest_close', 'lowest_close', 'close_diff_sum',
        'lowest_low_9', 'highest_high_9',
        # 'recent_low_20',
    ]
    df = df.drop(columns=temp_cols, errors='ignore')
    # df = df[ ['ticker', 'date'] + feature_columns_order ]

    return df


# Apply feature engineering
df_features = engineer_features(df)

import gc
del df
gc.collect()
#
print("\n✓ Feature engineering complete!")
print(f"Total features created: 31")
# print(f"Rows with complete features: {df_features.dropna().shape[0]:,}")

df_features = df_features.sort_values(['ticker', 'date'])
# Columns needed to identify rows (NOT features)
id_columns = ['ticker', 'date']

# Target
target_column = ['target']

# Final feature list (28 features)
feature_columns = [
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
# Combine all required columns
model_columns = id_columns +['missing_days'] + ['close'] + feature_columns + target_column

float_cols = df_features.select_dtypes(include=['float64']).columns
df_features[float_cols] = df_features[float_cols].astype(np.float32)


print("Actual columns:", df_features.columns.tolist())
model_cuurent_columns = [c for c in feature_columns if c in df_features.columns]
df_model = df_features.loc[:, model_cuurent_columns]
print(df_model.head())
print("Dataset shape before cleaning:", df_model.shape)
existing_cols = df_features.columns.tolist()

# Keep only valid columns
model_cuurent_columns = [c for c in model_cuurent_columns if c in existing_cols]

print(f"Final model columns ({len(model_cuurent_columns)}): {model_cuurent_columns}")
print("Shape BEFORE cleaning:", df_model.shape)

nan_rows = df_model.isna().any(axis=1).sum()
print("Rows containing at least one NaN:", nan_rows)

nan_percent = (nan_rows / len(df_model)) * 100
print(f"Percentage of dataset that has NaN rows: {nan_percent:.2f}%")

df_clean = df_model.dropna().reset_index(drop=True)

print("Shape AFTER cleaning:", df_clean.shape)
print("Rows removed:", len(df_model) - len(df_clean))
print("Remaining NaN values:", df_clean.isna().sum().sum())

path = r'D:\Stock_trend_project\data\processed\new_stocks_features2.csv'
df_model = df_features[model_cuurent_columns].dropna()
df_model.to_csv(path, index=False, chunksize=100_000)
print("ML dataset exported successfully")

print("Total NaN values:", df_clean.isna().sum().sum())

nan_rows = df_clean.isna().any(axis=1).sum()
print("Rows containing NaN:", nan_rows)

print("Total rows:", len(df_clean))
print("NaN row percentage:", nan_rows / len(df_clean) * 100)