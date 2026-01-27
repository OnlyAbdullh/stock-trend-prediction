"""
Script to Export Processed Stock Data with Engineered Features
================================================================
This script exports the data in two formats:
1. FULL: All 28 engineered features
2. TOP20: Top 20 most important features based on MI analysis
"""

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("STOCK PRICE PREDICTION - DATA EXPORT WITH FEATURES")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data path - UPDATE THIS PATH TO YOUR ACTUAL DATA LOCATION
INPUT_DATA_PATH = '../data/interim/train_clean_after_2010_and_bad_tickers.csv'

# Output directory
OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature sets
FINAL_FEATURES_FULL = [
    # Critical (4)
    'max_drawdown_20',
    'recent_high_20',
    'recent_low_20',
    'downside_deviation_10',

    # Price Position (6)
    'close_30d_ago',
    'distance_from_low',
    'low_to_close_ratio',
    'distance_from_high',
    'high_to_close_ratio',
    'is_up_day',

    # Bollinger & MAs (5)
    'BB_upper',
    'BB_lower',
    'MA_60',
    'MA_5',
    'price_position_20',

    # MA-Based (1)
    'MA_20',

    # Temporal (2)
    'month_cos',
    'month_sin',

    # Volatility (3)
    'volatility_20',
    'RSI_14',
    'parkinson_volatility',

    # Price Features (2)
    'high_low_ratio',
    'daily_return',

    # MA-Based (3)
    'MA_60_slope',
    'price_to_MA60',
    'return_30',
    'price_to_MA20',
    'price_to_MA5',
]

FINAL_FEATURES_TOP20 = [
    # Critical (4)
    'max_drawdown_20',
    'recent_high_20',
    'recent_low_20',
    'downside_deviation_10',

    # High Quality (16)
    'close_30d_ago',
    'distance_from_low',
    'low_to_close_ratio',
    'distance_from_high',
    'high_to_close_ratio',
    'is_up_day',
    'BB_upper',
    'BB_lower',
    'MA_60',
    'MA_5',
    'price_position_20',
    'MA_20',
    'month_cos',
    'month_sin',
    'volatility_20',
    'RSI_14',
]


# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_features(df):
    """
    Create optimized features based on correlation and MI analysis
    Total features: 28 high-quality features
    """
    df = df.copy()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    grouped = df.groupby('ticker')

    print("\nCalculating features:")

    # ========================================================================
    # PRICE FEATURES (5 features)
    # ========================================================================
    print("  - Price features...")
    df['daily_return'] = grouped['close'].pct_change()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
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
            1 / (4 * np.log(2)) *
            ((np.log(x['high'] / (x['low'] + 1e-8))) ** 2).rolling(10, min_periods=1).mean()
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
    # NORMALIZED PRICE FEATURES (3 features)
    # ========================================================================
    print("  - Normalized price features...")
    df['high_to_close_ratio'] = df['recent_high_20'] / (df['close'] + 1e-8)
    df['low_to_close_ratio'] = df['recent_low_20'] / (df['close'] + 1e-8)
    df['price_position_20'] = (
            (df['close'] - df['recent_low_20']) /
            (df['recent_high_20'] - df['recent_low_20'] + 1e-8)
    )

    # ========================================================================
    # TEMPORAL FEATURES (3 features)
    # ========================================================================
    print("  - Temporal features...")
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

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n[1] LOADING DATA...")

    # Load the dataset
    df = pd.read_csv(INPUT_DATA_PATH)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Remove zero prices
    df = df[df['open'] != 0]

    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"Total records: {len(df):,}")
    print(f"Unique tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # ========================================================================
    print("\n[2] ENGINEERING FEATURES...")
    # ========================================================================

    df_features = engineer_features(df)

    print("\n✓ Feature engineering complete!")
    print(f"Total features created: 28")

    # ========================================================================
    print("\n[3] PREPARING EXPORT DATA...")
    # ========================================================================

    # Base columns (metadata + original features)
    base_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'target']

    # OPTION 1: Full feature set (28 features)
    print("\n  Processing FULL feature set (28 features)...")
    df_full = df_features[base_columns + FINAL_FEATURES_FULL].copy()
    df_full = df_full.dropna()

    print(f"  - Rows with complete features: {len(df_full):,}")
    print(f"  - Total columns: {len(df_full.columns)}")

    # # OPTION 2: Top 20 features
    # print("\n  Processing TOP20 feature set (20 features)...")
    # df_top20 = df_features[base_columns + FINAL_FEATURES_TOP20].copy()
    # df_top20 = df_top20.dropna()
    #
    # print(f"  - Rows with complete features: {len(df_top20):,}")
    # print(f"  - Total columns: {len(df_top20.columns)}")

    # ========================================================================
    print("\n[4] EXPORTING DATA...")
    # ========================================================================

    # Export FULL dataset
    full_path = os.path.join(OUTPUT_DIR, 'stock_data_with_28_features.csv')
    df_full.to_csv(full_path, index=False)
    print(f"\n✓ Exported FULL dataset: {full_path}")
    print(f"  - Size: {os.path.getsize(full_path) / 1024 ** 2:.2f} MB")

    # # Export TOP20 dataset
    # top20_path = os.path.join(OUTPUT_DIR, 'stock_data_with_20_features.csv')
    # df_top20.to_csv(top20_path, index=False)
    # print(f"\n✓ Exported TOP20 dataset: {top20_path}")
    # print(f"  - Size: {os.path.getsize(top20_path) / 1024 ** 2:.2f} MB")

    # ========================================================================
    print("\n[5] GENERATING SUMMARY REPORT...")
    # ========================================================================

    # # Create summary report
    # summary = {
    #     'Dataset': ['FULL (28 features)', 'TOP20 (20 features)'],
    #     'Total_Rows': [len(df_full), len(df_top20)],
    #     'Total_Columns': [len(df_full.columns), len(df_top20.columns)],
    #     'Feature_Count': [28, 20],
    #     'Tickers': [df_full['ticker'].nunique(), df_top20['ticker'].nunique()],
    #     'Date_Range': [
    #         f"{df_full['date'].min()} to {df_full['date'].max()}",
    #         f"{df_top20['date'].min()} to {df_top20['date'].max()}"
    #     ],
    #     'Target_Distribution': [
    #         f"{df_full['target'].mean():.2%} positive",
    #         f"{df_top20['target'].mean():.2%} positive"
    #     ]
    # }

    # summary_df = pd.DataFrame(summary)
    # summary_path = os.path.join(OUTPUT_DIR, 'export_summary.txt')

    # with open(summary_path, 'w', encoding='utf-8') as f:
    #     f.write("=" * 80 + "\n")
    #     f.write("STOCK DATA EXPORT SUMMARY\n")
    #     f.write("=" * 80 + "\n\n")
    #     f.write(summary_df.to_string(index=False))
    #     f.write("\n\n")
    #     f.write("=" * 80 + "\n")
    #     f.write("FEATURE LISTS\n")
    #     f.write("=" * 80 + "\n\n")
    #     f.write("FULL FEATURE SET (28 features):\n")
    #     f.write("-" * 40 + "\n")
    #     for i, feat in enumerate(FINAL_FEATURES_FULL, 1):
    #         f.write(f"{i:2d}. {feat}\n")
    #     f.write("\n")
    #     f.write("TOP20 FEATURE SET (20 features):\n")
    #     f.write("-" * 40 + "\n")
    #     for i, feat in enumerate(FINAL_FEATURES_TOP20, 1):
    #         f.write(f"{i:2d}. {feat}\n")
    #
    # print(f"\n✓ Generated summary report: {summary_path}")

    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ EXPORT COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print(f"  1. {full_path}")
    # print(f"  2. {top20_path}")
    # print(f"  3. {summary_path}")
    print("\nNext Steps:")
    print("  - Use these datasets to train your LSTM/GRU models")
    print("  - Compare performance between 28 features vs 20 features")
    print("  - Experiment with different sequence lengths (10, 20, 30, 60 days)")
    print("=" * 80)


if __name__ == "__main__":
    main()