from pathlib import Path
import logging
import click
import pandas as pd


def clean_stock_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean basic structure and dtypes of raw stock DataFrame."""
    df = df.copy()
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
    df = df.copy()

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
    return df_clean


def drop_ticker_date_duplicates(
    df: pd.DataFrame,
    max_duplicates_per_ticker: int = 10
) -> pd.DataFrame:
    df = df.copy()
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

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    df_raw = pd.read_csv(input_filepath)
    df_clean = clean_stock_df(df_raw)

    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
