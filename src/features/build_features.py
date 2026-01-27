# Stock Price Prediction - Feature Analysis for RNN Sequence Selection
# Analyzing features to determine optimal sequence length for LSTM/GRU models
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
df = pd.read_csv('C:/Users/LENOVO/Desktop/NN project/stock-trend-prediction/data/interim/train_clean_after_2010_and_bad_tickers.csv')

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
# print("\nMissing values:")
# print(df.isnull().sum())

# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("[2] ENGINEERING FEATURES...")
print("="*80)

def engineer_features(df):
    """
    Create optimized features based on correlation and MI analysis
    Total features: 35 high-quality features (after improvements)
    """
    df = df.copy()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    grouped = df.groupby('ticker')

    print("\nCalculating features:")

    # ========================================================================
    # PRICE FEATURES (5 features - removed intraday_return)
    # ========================================================================
    print("  - Price features...")

    # Daily return: (close - prev_close) / prev_close
    df['daily_return'] = grouped['close'].pct_change()

    # High-Low ratio: (high - low) / close
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']

    # Return over 30 days
    df['close_30d_ago'] = grouped['close'].shift(30)
    df['return_30'] = (df['close'] - df['close_30d_ago']) / (df['close_30d_ago'] + 1e-8)

    # ========================================================================
    # MOVING AVERAGES (3 features)
    # ========================================================================
    print("  - Moving averages (5, 20, 60 days)...")

    df['MA_5'] = grouped['close'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df['MA_20'] = grouped['close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    df['MA_60'] = grouped['close'].transform(
        lambda x: x.rolling(window=60, min_periods=1).mean()
    )

    # ========================================================================
    # MA-BASED FEATURES (4 features)
    # ========================================================================
    print("  - MA-based features...")

    df['price_to_MA5'] = (df['close'] - df['MA_5']) / (df['MA_5'] + 1e-8)
    df['price_to_MA20'] = (df['close'] - df['MA_20']) / (df['MA_20'] + 1e-8)
    df['price_to_MA60'] = (df['close'] - df['MA_60']) / (df['MA_60'] + 1e-8)
    df['MA_60_slope'] = grouped['MA_60'].pct_change(30)

    # ========================================================================
    # VOLATILITY FEATURES (4 features)
    # ========================================================================
    print("  - Volatility features...")

    df['volatility_20'] = grouped['daily_return'].transform(
        lambda x: x.rolling(window=20, min_periods=1).std()
    )

    df['parkinson_volatility'] = grouped.apply(
        lambda x: np.sqrt(
            1/(4*np.log(2)) *
            ((np.log(x['high']/(x['low']+1e-8)))**2).rolling(10, min_periods=1).mean()
        )
    ).reset_index(level=0, drop=True)

    df['downside_deviation_10'] = grouped['daily_return'].transform(
        lambda x: x.where(x < 0, 0).rolling(10, min_periods=1).std()
    )

    # ========================================================================
    # TECHNICAL INDICATORS (1 feature)
    # ========================================================================
    print("  - Technical indicators...")

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = grouped['close'].transform(lambda x: calculate_rsi(x, 14))

    # ========================================================================
    # SUPPORT/RESISTANCE FEATURES (4 features)
    # ========================================================================
    print("  - Support/Resistance levels...")

    df['recent_high_20'] = grouped['high'].transform(
        lambda x: x.rolling(20, min_periods=1).max()
    )
    df['recent_low_20'] = grouped['low'].transform(
        lambda x: x.rolling(20, min_periods=1).min()
    )

    df['distance_from_high'] = (df['close'] - df['recent_high_20']) / (df['recent_high_20'] + 1e-8)
    df['distance_from_low'] = (df['close'] - df['recent_low_20']) / (df['recent_low_20'] + 1e-8)

    # ========================================================================
    # RISK FEATURES (1 feature)
    # ========================================================================
    print("  - Risk features...")

    def max_drawdown(series, window):
        roll_max = series.rolling(window, min_periods=1).max()
        drawdown = (series - roll_max) / (roll_max + 1e-8)
        return drawdown.rolling(window, min_periods=1).min()

    df['max_drawdown_20'] = grouped['close'].transform(lambda x: max_drawdown(x, 20))

    # ========================================================================
    # BOLLINGER BANDS (2 features)
    # ========================================================================
    print("  - Bollinger Bands...")

    df['BB_std'] = grouped['close'].transform(
        lambda x: x.rolling(20, min_periods=1).std()
    )
    df['BB_upper'] = df['MA_20'] + 2 * df['BB_std']
    df['BB_lower'] = df['MA_20'] - 2 * df['BB_std']

    # ========================================================================
    # 🆕 NORMALIZED PRICE FEATURES (4 features)
    # ========================================================================
    print("  - Normalized price features...")

    # Recent high/low as ratio to current price
    df['high_to_close_ratio'] = df['recent_high_20'] / (df['close'] + 1e-8)
    df['low_to_close_ratio'] = df['recent_low_20'] / (df['close'] + 1e-8)

    # Position within 20-day range
    df['price_position_20'] = (
        (df['close'] - df['recent_low_20']) /
        (df['recent_high_20'] - df['recent_low_20'] + 1e-8)
    )

    # ========================================================================
    # 🆕 IMPROVED TEMPORAL FEATURES (6 features)
    # ========================================================================
    print("  - Improved temporal features...")

    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

    df['is_up_day'] = (df['daily_return'] > 0).astype(int)

    # ========================================================================
    # TARGET VARIABLE
    # ========================================================================
    print("  - Target variable...")

    df['close_30d_future'] = grouped['close'].shift(-30)
    df['target'] = (df['close_30d_future'] > df['close']).astype(int)

    # ========================================================================
    # CLEANUP
    # ========================================================================
    temp_columns = ['BB_std', 'close_30d_future']
    df = df.drop(columns=temp_columns, errors='ignore')
    # print(df.head())
    return df


# Apply feature engineering
df_features = engineer_features(df)

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
    'max_drawdown_20',
    'recent_high_20',
    'recent_low_20',
    'downside_deviation_10',

    'high_to_close_ratio',
    'low_to_close_ratio',
    'price_position_20',

    'distance_from_high',
    'distance_from_low',
    'close_30d_ago',

    'MA_5',
    'MA_20',
    'MA_60',

    'price_to_MA5',
    'price_to_MA20',
    'price_to_MA60',
    'MA_60_slope',

    'volatility_20',
    'RSI_14',
    'parkinson_volatility',

    'daily_return',
    'high_low_ratio',
    'return_30',

    'BB_upper',
    'BB_lower',

    'month_sin',
    'month_cos',

    'is_up_day'
]
# Combine all required columns
model_columns = id_columns +['missing_days'] + ['close'] + feature_columns + target_column

float_cols = df_features.select_dtypes(include=['float32']).columns
df_features[float_cols] = df_features[float_cols].astype(np.float32)
df_model = df_features[model_columns].copy()
print(df_model.head())
print("Dataset shape before cleaning:", df_model.shape)

print("Shape BEFORE cleaning:", df_model.shape)

nan_rows = df_model.isna().any(axis=1).sum()
print("Rows containing at least one NaN:", nan_rows)

nan_percent = (nan_rows / len(df_model)) * 100
print(f"Percentage of dataset that has NaN rows: {nan_percent:.2f}%")

df_clean = df_model.dropna().reset_index(drop=True)

print("Shape AFTER cleaning:", df_clean.shape)
print("Rows removed:", len(df_model) - len(df_clean))
print("Remaining NaN values:", df_clean.isna().sum().sum())

path = r'C:/Users/LENOVO/Desktop/NN project/stock-trend-prediction/data/export/new_stocks_features_30d_target.csv'
df_clean.to_csv(path, index=False, chunksize=100_000)
print("ML dataset exported successfully")

print("Total NaN values:", df_clean.isna().sum().sum())

nan_rows = df_clean.isna().any(axis=1).sum()
print("Rows containing NaN:", nan_rows)

print("Total rows:", len(df_clean))
print("NaN row percentage:", nan_rows / len(df_clean) * 100)