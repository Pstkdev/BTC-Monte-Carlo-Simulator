import pandas as pd
import yfinance as yf
import numpy as np
import math


def fetch_adj_close(ticker: str, start: str, end=None) -> pd.Series:

    df = yf.download(ticker, start=start, end=end, interval="1d", group_by="column")

    if df.empty:
        raise ValueError(f"No data fetched for {ticker}.")

    adj = df["Close"]
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
    """
    Estimate annualised drift (mu) and volatility (sigma) from a price series
    using daily log returns.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean() * steps_per_year
    sigma = log_returns.std() * math.sqrt(steps_per_year)
    return mu, sigma


if __name__ == "__main__":
    prices = fetch_adj_close("BTC-USD", start="2022-01-01")
    mu, sigma = estimate_mu_sigma(prices, steps_per_year=365)
    print("mu_annual:", mu)
    print("sigma_annual:", sigma)
