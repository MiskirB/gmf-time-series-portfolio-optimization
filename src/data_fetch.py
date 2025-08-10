"""
Data fetching utilities for GMF project.
"""
from __future__ import annotations
import os
from typing import Dict, Iterable, Optional
import pandas as pd
import yfinance as yf

def fetch_yfinance_data(
    tickers: Iterable[str],
    start: str = "2015-07-01",
    end: str = "2025-07-31",
    interval: str = "1d",
    save_dir: Optional[str] = "data"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for tickers from Yahoo Finance.
    Saves each dataframe to CSV and returns a dict of DataFrames.
    """
    os.makedirs(save_dir, exist_ok=True)
    data = {}
    for tk in tickers:
        df = yf.download(tk, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        # Ensure standard columns exist
        expected = ["Open","High","Low","Close","Adj Close","Volume"]
        for col in expected:
            if col not in df.columns:
                df[col] = pd.NA
        df.index.name = "Date"
        df = df[expected]
        csv_path = os.path.join(save_dir, f"{tk}_{interval}.csv")
        df.to_csv(csv_path)
        data[tk] = df
    return data
