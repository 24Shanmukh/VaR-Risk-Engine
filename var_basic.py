# var_basic.py - STEP 2: Historical VaR and Parametric (Normal) VaR
# Run from the VAR folder:  python src/var_basic.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from config import PORTFOLIO_VALUE, CONFIDENCE_LEVELS, HORIZONS, OUTPUT_DIR

# 1. Load the portfolio daily returns saved in Step 1
returns = pd.read_csv("data/portfolio_returns.csv", index_col=0, parse_dates=True)["portfolio"]

mean = returns.mean()
std = returns.std()

results = []

for cl in CONFIDENCE_LEVELS:          # 0.95 and 0.99
    for days in HORIZONS:             # 1 and 10

        # 2. HISTORICAL VaR
        # Sort all past daily returns and pick the worst 5% (or 1%) cut-off point
        hist_1day = -np.percentile(returns, (1 - cl) * 100)

        # 3. PARAMETRIC (NORMAL) VaR
        # Assume returns follow a bell curve: VaR = -(mean + z * std)
        z = norm.ppf(1 - cl)          # z = -1.645 for 95%, -2.326 for 99%
        para_1day = -(mean + z * std)

        # 4. Scale from 1 day to N days with the square-root-of-time rule
        hist_var = hist_1day * np.sqrt(days)
        para_var = para_1day * np.sqrt(days)

        results.append({
            "Confidence": f"{int(cl * 100)}%",
            "Days": days,
            "Historical VaR %": round(hist_var * 100, 2),
            "Historical VaR Rs": round(hist_var * PORTFOLIO_VALUE),
            "Parametric VaR %": round(para_var * 100, 2),
            "Parametric VaR Rs": round(para_var * PORTFOLIO_VALUE),
        })

table = pd.DataFrame(results)

# 5. Print and save the results
pd.set_option("display.width", 140)
print("\nValue at Risk on a Rs 1 crore portfolio")
print(table.to_string(index=False))

os.makedirs(OUTPUT_DIR, exist_ok=True)
table.to_csv(f"{OUTPUT_DIR}/var_basic.csv", index=False)

# 6. Chart: histogram of daily returns with the 99% VaR lines
hist_99 = -np.percentile(returns, 1)
para_99 = -(mean + norm.ppf(0.01) * std)

plt.figure(figsize=(10, 5))
plt.hist(returns * 100, bins=100, color="lightgray", edgecolor="gray")
plt.axvline(-hist_99 * 100, color="red", label=f"Historical 99% VaR: {hist_99*100:.2f}%")
plt.axvline(-para_99 * 100, color="blue", linestyle="--", label=f"Parametric 99% VaR: {para_99*100:.2f}%")
plt.title("Portfolio daily returns and 1-day 99% VaR")
plt.xlabel("Daily return (%)")
plt.ylabel("Number of days")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/var_histogram.png", dpi=150)
plt.show()

print(f"\nSaved table and chart in '{OUTPUT_DIR}/'")