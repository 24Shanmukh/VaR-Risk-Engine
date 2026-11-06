# backtest.py - STEP 4: Backtesting - would our VaR models have worked?
# Run from the VAR folder:  python src/backtest.py
#
# Idea: every day, calculate 99% VaR using ONLY the past 250 days,
# then check if the next day's real loss was bigger (an "exception").
# A good 99% model should have about 1 exception per 100 days.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

from config import BACKTEST_WINDOW, OUTPUT_DIR

returns = pd.read_csv("data/portfolio_returns.csv", index_col=0, parse_dates=True)["portfolio"]

cl = 0.99
window = BACKTEST_WINDOW          # 250 days
z = norm.ppf(1 - cl)              # -2.326

# 1. Historical VaR: 1st percentile of the past 250 days
hist_var = -returns.rolling(window).quantile(1 - cl).shift(1)

# 2. Parametric VaR: bell curve fitted to the past 250 days
roll_mean = returns.rolling(window).mean()
roll_std = returns.rolling(window).std()
para_var = -(roll_mean + z * roll_std).shift(1)

# 3. EWMA VaR (RiskMetrics): recent days count more than old days
# lambda = 0.94 means each day's weight is 94% of the next day's weight
lam = 0.94
ewma_std = np.sqrt((returns ** 2).ewm(alpha=1 - lam).mean())
ewma_var = -(z * ewma_std).shift(1)

# .shift(1) above = VaR for today uses data only up to yesterday (no peeking!)

models = {"Historical": hist_var, "Parametric": para_var, "EWMA": ewma_var}


# Kupiec test: is the number of exceptions statistically OK?
def kupiec_pvalue(exceptions, days, p=1 - cl):
    x, T = exceptions, days
    if x == 0:
        lr = -2 * T * np.log(1 - p)
    else:
        rate = x / T
        lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p)
                   - (T - x) * np.log(1 - rate) - x * np.log(rate))
    return 1 - chi2.cdf(lr, df=1)


# Basel traffic light (for 250 days at 99%)
def basel_zone(exceptions):
    if exceptions <= 4:
        return "Green"
    elif exceptions <= 9:
        return "Yellow"
    return "Red"


# 4. Test every model
rows = []
for name, var in models.items():
    test = pd.DataFrame({"actual": returns, "var": var}).dropna()
    exceptions = test["actual"] < -test["var"]          # loss bigger than VaR

    days = len(test)
    n_exc = exceptions.sum()
    last_250 = exceptions.iloc[-250:].sum()
    pval = kupiec_pvalue(n_exc, days)

    rows.append({
        "Model": name,
        "Days tested": days,
        "Exceptions": n_exc,
        "Expected": round(days * (1 - cl)),
        "Exception rate %": round(n_exc / days * 100, 2),
        "Kupiec p-value": round(pval, 3),
        "Kupiec result": "Pass" if pval > 0.05 else "Fail",
        "Last 250 days": last_250,
        "Basel zone": basel_zone(last_250),
    })

table = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("\nBacktest of 1-day 99% VaR")
print(table.to_string(index=False))

os.makedirs(OUTPUT_DIR, exist_ok=True)
table.to_csv(f"{OUTPUT_DIR}/backtest.csv", index=False)

# 5. Chart: real returns vs each model's VaR line, exceptions marked in red
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for ax, (name, var) in zip(axes, models.items()):
    test = pd.DataFrame({"actual": returns, "var": var}).dropna()
    exc = test[test["actual"] < -test["var"]]
    ax.plot(test.index, test["actual"] * 100, color="gray", linewidth=0.5, label="Daily return")
    ax.plot(test.index, -test["var"] * 100, color="blue", linewidth=1, label="-VaR 99%")
    ax.scatter(exc.index, exc["actual"] * 100, color="red", s=12, label=f"Exceptions ({len(exc)})")
    ax.set_title(f"{name} VaR")
    ax.set_ylabel("Return (%)")
    ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/backtest_chart.png", dpi=150)
plt.show()

print(f"\nSaved table and chart in '{OUTPUT_DIR}/'")