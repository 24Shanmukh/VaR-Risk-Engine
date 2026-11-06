# risk_decomposition.py - STEP 5: Which asset adds the most risk?
# Run from the VAR folder:  python src/risk_decomposition.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from config import PORTFOLIO, PORTFOLIO_VALUE, OUTPUT_DIR

returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
weights = pd.Series(PORTFOLIO)[returns.columns]

cl = 0.99
z = -norm.ppf(1 - cl)                 # 2.326

cov = returns.cov()                   # covariance matrix (volatility + correlation)

# 1. Portfolio volatility = sqrt(w' x Cov x w)
port_std = np.sqrt(weights @ cov @ weights)
port_var = z * port_std               # diversified 99% VaR (in %)

# 2. Marginal VaR: how much VaR changes if we add a little more of this asset
marginal_var = z * (cov @ weights) / port_std

# 3. Component VaR: each asset's share of the total VaR (they add up exactly to the total)
component_var = weights * marginal_var

# 4. Standalone VaR: the VaR of each holding on its own (ignoring the others)
standalone_var = z * weights * returns.std()

table = pd.DataFrame({
    "Weight %": (weights * 100).round(1),
    "Standalone VaR (Rs)": (standalone_var * PORTFOLIO_VALUE).round(0),
    "Component VaR (Rs)": (component_var * PORTFOLIO_VALUE).round(0),
    "Share of risk %": (component_var / port_var * 100).round(1),
}).sort_values("Share of risk %", ascending=False)

undiversified = standalone_var.sum()

pd.set_option("display.width", 140)
print("\nRisk contribution of each asset (1-day 99% parametric VaR)")
print(table.to_string())
print("\nUndiversified VaR (sum of standalone):", round(undiversified * PORTFOLIO_VALUE), "Rs")
print("Diversified VaR (whole portfolio)     :", round(port_var * PORTFOLIO_VALUE), "Rs")
print("Diversification benefit              :", round((undiversified - port_var) * PORTFOLIO_VALUE), "Rs",
      f"({(1 - port_var / undiversified) * 100:.1f}% less risk)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
table.to_csv(f"{OUTPUT_DIR}/risk_decomposition.csv")

# 5. Chart: weight vs share of risk for each asset
chart = table[["Weight %", "Share of risk %"]].sort_values("Share of risk %")
chart.plot(kind="barh", figsize=(10, 6), color=["lightgray", "crimson"])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Weight vs share of portfolio risk (99% VaR)")
plt.xlabel("%")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/risk_decomposition.png", dpi=150)
plt.show()

print(f"\nSaved table and chart in '{OUTPUT_DIR}/'")