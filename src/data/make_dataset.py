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

    price_cols = ["open", "high", "low", "close"]
    for c in price_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip()

    return df

def handle_missing_values(
        df: pd.DataFrame,
        price_cols=("open", "high", "low", "close"),
        max_na_fraction=0.10,
) -> pd.DataFrame:
    price_cols = list(price_cols)

    removed_tickers = 0
    removed_rows = 0

    def process_ticker(g: pd.DataFrame) -> pd.DataFrame:
        nonlocal removed_tickers, removed_rows

        rows_before = len(g)
        g = g.sort_values("date")

        while not g.empty and g[price_cols].iloc[0].isna().any():
            g = g.iloc[1:]

        while not g.empty and g[price_cols].iloc[-1].isna().any():
            g = g.iloc[:-1]

        if g.empty:
            removed_tickers += 1
            removed_rows += rows_before
            return g

        na_fraction = g[price_cols].isna().mean().mean()

        if na_fraction > max_na_fraction:
            removed_tickers += 1
            removed_rows += rows_before
            return g.iloc[0:0]

        g[price_cols] = g[price_cols].ffill().bfill()
        removed_rows += (rows_before - len(g))

        return g

    df_clean = (
        df
        .groupby("ticker", group_keys=False)
        .apply(process_ticker)
    )
    return df_clean # TODO : check this


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
    df: pd.DataFrame,
    price_col: str = "close",
    iqr_factor: float = 1.5,
    threshold: float = 25.0,
) -> tuple[pd.DataFrame, list[str]]:
    """
    يحسب العوائد + القيم المتطرفة لكل سهم داخلياً،
    ثم يحذف الأسهم التي نسبة القيم المتطرفة فيها تتجاوز threshold٪.
    """
    df = df.sort_values(["ticker", "date"])

    df["return"] = (
        df.groupby("ticker")[price_col]
        .pct_change()
    )

    def mark_outliers(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        valid = g["return"].dropna()

        if valid.empty:
            g["return_is_outlier"] = False
            return g

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        g["return_is_outlier"] = (g["return"] < lower) | (g["return"] > upper)
        g.loc[g["return"].isna(), "return_is_outlier"] = False
        return g

    df_marked = (
        df
        .groupby("ticker", group_keys=False)
        .apply(mark_outliers)
    )

    summary = (
        df_marked
        .groupby("ticker")
        .agg(
            n_rows=("return", "count"),
            n_outliers=("return_is_outlier", "sum"),
        )
    )
    summary["outliers_ratio"] = summary["n_outliers"] / summary["n_rows"] * 100

    bad_tickers = summary[summary["outliers_ratio"] > threshold].index.tolist()

    df_cleaned = df_marked[~df_marked["ticker"].isin(bad_tickers)].copy()

    return df_cleaned, bad_tickers


def run_basic_clean_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = verify_columns_and_types(df_raw)
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

    df["prev_date"] = df.groupby("ticker")["date"].shift(1)
    df["gap_days"] = (df["date"] - df["prev_date"]).dt.days

    # gaps = df[df["gap_days"] > MAX_NORMAL_GAP].copy()
    df["missing_days"] = df["gap_days"] - 1
    gaps = df[df["missing_days"] >= 1].copy()
    gap_counts = (
        gaps.groupby("date")
        .size()
        .rename("gap_tickers")
    )
    data_counts = (
        df.groupby("date")
        .size()
        .rename("total_tickers")
    )
    gap_vs_data = (
        gap_counts
        .to_frame()
        .join(data_counts, how="left")
        .fillna(0)
    )

    gap_vs_data["gap_ratio"] = (
            gap_vs_data["gap_tickers"] / gap_vs_data["total_tickers"]
    )
    GLOBAL_THRESHOLD = 0.8

    gap_vs_data["gap_type"] = np.where(
        gap_vs_data["gap_ratio"] >= GLOBAL_THRESHOLD,
        "global",
        "local"
    )
    global_gap_dates = gap_vs_data.index[
        gap_vs_data["gap_type"] == "global"
        ]
    df_no_global_gaps = df[~df["date"].isin(global_gap_dates)].copy()

    local_gaps = gaps[
        ~gaps["date"].isin(global_gap_dates)
    ][["date", "ticker"]]
    logging.info(
        "Saving final cleaned data (after removing corrupted tickers) to %s",
        Path('data/interim/local_gaps.csv')

    )
    # fill missing_days NaNs
    df_no_global_gaps["missing_days"] = df_no_global_gaps["missing_days"].fillna(0).astype(int)

    # drop helper columns
    df_no_global_gaps = df_no_global_gaps.drop(
        columns=["prev_date", "gap_days"],
        errors="ignore"
    )

    local_gaps.to_csv('data/interim/local_gaps.csv', index=False)
    return df_no_global_gaps

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
    # basic clean
    df = run_basic_clean_df(df)
    # basic_clean_path = interim_dir / "train_clean_basic.csv"
    # logging.info("Saving basic cleaned data to %s", basic_clean_path)
    # df.to_csv(basic_clean_path, index=False)
    # # بعد 2010
    # df = filter_after_2010_df(df) # moved to basic_clean_run

    # not necessary now
    # after_2010_path = interim_dir / "train_clean_after_2010.csv"
    # logging.info("Saving data after 2010-01-01 to %s", after_2010_path)
    # df.to_csv(after_2010_path, index=False)
    # إزالة الأسهم الفاسدة
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