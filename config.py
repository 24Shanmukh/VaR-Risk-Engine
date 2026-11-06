"""
config.py - one place for all project settings.
Every other file imports from here, so you only change things once.
"""

# ---- Portfolio -------------------------------------------------------------
# Yahoo Finance tickers (.NS = NSE). INR=X is the USD/INR exchange rate.
PORTFOLIO = {
    "RELIANCE.NS":   0.12,   # Reliance Industries
    "HDFCBANK.NS":   0.12,   # HDFC Bank
    "ICICIBANK.NS":  0.10,   # ICICI Bank
    "INFY.NS":       0.10,   # Infosys
    "TCS.NS":        0.10,   # TCS
    "ITC.NS":        0.08,   # ITC
    "LT.NS":         0.08,   # Larsen & Toubro
    "BHARTIARTL.NS": 0.08,   # Bharti Airtel
    "GOLDBEES.NS":   0.12,   # Gold ETF
    "INR=X":         0.10,   # USD held (value moves with USD/INR)
}

PORTFOLIO_VALUE = 10_000_000      # Rs 1 crore notional

# ---- Data ------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = None                   # None = up to today

# ---- Risk settings ---------------------------------------------------------
CONFIDENCE_LEVELS = [0.95, 0.99]
ES_CONFIDENCE = 0.975             # Basel FRTB standard for Expected Shortfall
HORIZONS = [1, 10]                # days
BACKTEST_WINDOW = 250             # ~1 trading year

# ---- Paths -----------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

# Sanity check: weights must add up to 100%
assert abs(sum(PORTFOLIO.values()) - 1.0) < 1e-9, "Weights must sum to 1"
