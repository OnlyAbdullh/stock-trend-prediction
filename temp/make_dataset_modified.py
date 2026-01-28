from pathlib import Path
import logging
import click
import pandas as pd
import numpy as np


def verify_columns_and_types(df: pd.DataFrame) -> pd.DataFrame:

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
        .str.lower()
    )
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume", "dividends", "stock_splits"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df[["open", "high", "low", "close", "volume"]] = (
        df[["open", "high", "low", "close", "volume"]]
        .apply(pd.to_numeric, errors="coerce")
    )

    df["ticker"] = df["ticker"].astype(str).str.strip()

    return df

def handle_missing_values(
    df: pd.DataFrame,
    price_cols=("open", "high", "low", "close"),
    max_na_fraction=0.10,
) -> pd.DataFrame:

    price_cols = list(price_cols)

    # 1. remove leading / trailing NaNs per ticker
    row_has_all_prices = ~df[price_cols].isna().any(axis=1)

    first_valid = row_has_all_prices.groupby(df["ticker"]).cummax()
    last_valid = (
        row_has_all_prices.iloc[::-1]
        .groupby(df["ticker"])
        .cummax()
        .iloc[::-1]
    )

    df = df[first_valid & last_valid]

    # 2. drop tickers with too many NaNs
    na_fraction = (
        df.groupby("ticker")[price_cols]
        .apply(lambda x: x.isna().mean().mean())
    )

    bad_tickers = na_fraction[na_fraction > max_na_fraction].index
    df = df[~df["ticker"].isin(bad_tickers)]

    # 3. fill remaining gaps
    df[price_cols] = (
        df.groupby("ticker")[price_cols]
        .ffill()
        .bfill()
    )

    return df

def drop_ticker_date_duplicates(
    df: pd.DataFrame,
    max_duplicates_per_ticker: int = 10
) -> pd.DataFrame:
    dup_counts = (
        df.groupby(["ticker", "date"])
        .size()
        .reset_index(name="n")
    )
    bad_tickers = (
        dup_counts[dup_counts["n"] > 1]
        .groupby("ticker")["n"]
        .sum()
    )
    bad_tickers = bad_tickers[bad_tickers > max_duplicates_per_ticker].index
    df = df[~df["ticker"].isin(bad_tickers)]
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first")
    return df

def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:

    cond_open = df["open"] != 0
    cond_close = df["close"] != 0
    cond_high_low = df["high"] >= df["low"]
    cond_open_range = (df["open"] >= df["low"]) & (df["open"] <= df["high"])
    cond_volume = df["volume"] > 0
    valid_mask = cond_open & cond_close & cond_high_low & cond_open_range & cond_volume

    return df[valid_mask].copy()

def filter_by_start_date(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    return df[df["date"] >= start_date]

def remove_corrupted_tickers_df(
            df,
            price_col="close",
            iqr_factor=1.5,
            threshold=25.0,
    )-> tuple[pd.DataFrame, list[str]]:
    """
    يحسب العوائد + القيم المتطرفة لكل سهم داخلياً،
    ثم يحذف الأسهم التي نسبة القيم المتطرفة فيها تتجاوز threshold٪.
    """
    df["return"] = df.groupby("ticker")[price_col].pct_change()

    q = df.groupby("ticker")["return"].quantile([0.25, 0.75]).unstack()
    iqr = q[0.75] - q[0.25]

    bounds = pd.DataFrame({
        "lower": q[0.25] - iqr_factor * iqr,
        "upper": q[0.75] + iqr_factor * iqr,
    })

    df = df.join(bounds, on="ticker")

    df["return_is_outlier"] = (
            (df["return"] < df["lower"]) |
            (df["return"] > df["upper"])
    ).fillna(False)

    summary = df.groupby("ticker")["return_is_outlier"].mean() * 100
    bad_tickers = summary[summary > threshold].index.tolist()

    df = df[~df["ticker"].isin(bad_tickers)].drop(columns=["lower", "upper"])

    return df, bad_tickers


def run_basic_clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = verify_columns_and_types(df_raw)
    df = df.sort_values(["ticker", "date"], kind="mergesort")
    df = filter_after_2010_df(df)
    df = handle_missing_values(df)
    df = remove_invalid_rows(df)
    df = drop_ticker_date_duplicates(df)
    df = remove_global_gaps(df)
    return df

def filter_after_2010_df(df: pd.DataFrame) -> pd.DataFrame:
    return filter_by_start_date(df, "2010-01-01")

def remove_global_gaps(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(["ticker", "date"]).copy()

    # mark dates that have missing days before them
    df["prev_date"] = df.groupby("ticker")["date"].shift(1)
    df["gap_days"] = (df["date"] - df["prev_date"]).dt.days
    df["missing_days"] = (df["gap_days"] - 1).fillna(0).astype(int)
    gap_ratio_per_date = df[df["missing_days"] >= 1].groupby("date").size() / df.groupby("date").size()
    gap_ratio_per_date = gap_ratio_per_date.dropna()
    global_gap_dates = gap_ratio_per_date[gap_ratio_per_date >= 0.8].index  # index is date here
    df = df.drop(columns=['prev_date', 'gap_days'])
    df.loc[df["date"].isin(global_gap_dates), "missing_days"] = 0
    return df
@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
def main(input_filepath: str):
    logging.basicConfig(level=logging.INFO)

    data_dir = Path("data")
    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_filepath)

    logging.info("Reading raw data from %s", input_path)
    df = pd.read_csv(input_path)

    df = run_basic_clean_df(df)
    df, removed_tickers = remove_corrupted_tickers_df(
        df,
        price_col="close",
        iqr_factor=1.5,
        threshold=25.0,
    )
    final_interim_path = interim_dir / "train_clean_after_2010_and_bad_tickers.csv"
    logging.info(
        "Saving final cleaned data (after removing corrupted tickers) to %s",
        final_interim_path,
    )
    df.to_csv(final_interim_path, index=False)

if __name__ == "__main__":
    main()