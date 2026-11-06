# data_loader.py - STEP 1: download prices and calculate returns
# Run from the VAR folder:  python src/data_loader.py

import os
import numpy as np
import pandas as pd
import yfinance as yf

from config import PORTFOLIO, PORTFOLIO_VALUE, START_DATE

tickers = list(PORTFOLIO.keys())
weights = pd.Series(PORTFOLIO)

# 1. Download daily closing prices (adjusted for dividends and splits)
print("Downloading prices...")
data = yf.download(tickers, start=START_DATE, auto_adjust=True)
prices = data["Close"]

# 2. Clean: fill small gaps with the previous day's price, drop rows still empty
prices = prices.ffill(limit=3).dropna()

# 3. Daily log returns = ln(today's price / yesterday's price)
returns = np.log(prices / prices.shift(1)).dropna()

# 3b. Remove bad data
# Large-cap stocks, gold and USD almost never move more than 20% in one day.
# A bigger jump is usually a stock split or a Yahoo data error, not a real move.
bad_days = (returns.abs() > 0.20).any(axis=1)
if bad_days.any():
    print("\nSuspicious days removed (data errors or splits):")
    print(returns[bad_days].round(3))
    returns = returns[~bad_days]

# 4. Portfolio daily return = sum of (weight x asset return)
portfolio_returns = (returns * weights).sum(axis=1)

# 5. Save to CSV files in a 'data' folder
os.makedirs("data", exist_ok=True)
prices.to_csv("data/prices.csv")
returns.to_csv("data/returns.csv")
portfolio_returns.to_csv("data/portfolio_returns.csv", header=["portfolio"])

# 6. Print results
print("\nData from", prices.index[0].date(), "to", prices.index[-1].date())
print("Number of trading days:", len(returns))

print("\nAnnual volatility of each asset (%):")
print((returns.std() * np.sqrt(252) * 100).round(2))

print("\nCorrelation between assets:")
print(returns.corr().round(2))

worst_day = portfolio_returns.min()
print("\n--- Portfolio ---")
print("Annual volatility (%):", round(portfolio_returns.std() * np.sqrt(252) * 100, 2))
print("Worst day (%):", round(worst_day * 100, 2), "on", portfolio_returns.idxmin().date())
print("Worst day loss (Rs):", round(worst_day * PORTFOLIO_VALUE))
print("Excess kurtosis:", round(portfolio_returns.kurt(), 2), "(above 0 = fat tails)")
