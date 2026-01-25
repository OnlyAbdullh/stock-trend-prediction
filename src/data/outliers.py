# src/data/outliers.py

import pandas as pd


def winsorize_returns_per_ticker(
    df: pd.DataFrame,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    min_samples: int = 10,
    return_col: str = "return",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    df = df.copy()

    def winsorize_group(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()

        if return_col not in g.columns:
            return g

        valid = g[return_col].dropna()

        if len(valid) < min_samples:
            return g

        lower_bound = valid.quantile(lower_percentile / 100.0)
        upper_bound = valid.quantile(upper_percentile / 100.0)

        g.loc[g[return_col] < lower_bound, return_col] = lower_bound
        g.loc[g[return_col] > upper_bound, return_col] = upper_bound

        return g


    df_out = (
        df
        .groupby(ticker_col, group_keys=False)
        .apply(winsorize_group)
    )

    return df_out

from sklearn.preprocessing import RobustScaler
import pandas as pd


def apply_robust_scaling_returns(
    df: pd.DataFrame,
    ticker_col: str = "Ticker",
    return_col: str = "Return",
    quantile_range=(5.0, 95.0),
):
    """
    تطبيق Robust Scaling على عمود Return فقط لكل سهم على حدة (بيانات TRAIN).
    """
    df_scaled = df.copy()
    scalers_dict = {}

    if return_col not in df.columns:
        raise ValueError(f"العمود '{return_col}' غير موجود في DataFrame")

    for ticker in df[ticker_col].unique():
        mask = df[ticker_col] == ticker

        scaler = RobustScaler(quantile_range=quantile_range)

        return_values = df.loc[mask, return_col].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(return_values)

        df_scaled.loc[mask, return_col] = scaled_values.flatten()

        scalers_dict[ticker] = scaler

    print(f"✅ تم تطبيق Robust Scaling على عمود '{return_col}' لـ {len(scalers_dict)} سهم")
    print(f"📈 النطاق المئوي المستخدم: {quantile_range}")

    return df_scaled, scalers_dict




