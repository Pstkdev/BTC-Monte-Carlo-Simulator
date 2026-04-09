import pandas as pd
import yfinance as yf
import numpy as np
import math


def fetch_adj_close(ticker: str, start: str, end=None) -> pd.Series:

    df = yf.download(ticker, start=start, end=end, interval="1d", group_by="column")

    if df.empty:
        raise ValueError(f"No data fetched for {ticker}.")

    adj = df["Adj Close"]
    if isinstance(adj, pd.DataFrame):
        if ticker in adj.columns:
            adj = adj[ticker]
        else:
            adj = adj.iloc[:, 0]

    s = adj.dropna()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s


def estimate_mu_sigma(prices: pd.Series, steps_per_year: int) -> tuple[float, float]:
    pass
