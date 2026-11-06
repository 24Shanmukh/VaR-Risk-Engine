# var_advanced.py - STEP 3: Expected Shortfall and Monte Carlo VaR
# Run from the VAR folder:  python src/var_advanced.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from config import PORTFOLIO, PORTFOLIO_VALUE, ES_CONFIDENCE, OUTPUT_DIR

# 1. Load data from Step 1
asset_returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
port_returns = pd.read_csv("data/portfolio_returns.csv", index_col=0, parse_dates=True)["portfolio"]

weights = pd.Series(PORTFOLIO)[asset_returns.columns]   # same order as the columns


# Helper: VaR and Expected Shortfall from any list of returns
def var_and_es(returns, cl):
    cutoff = np.percentile(returns, (1 - cl) * 100)      # the VaR point
    var = -cutoff
    es = -returns[returns <= cutoff].mean()              # average of the days beyond VaR
    return var, es


# 2. HISTORICAL - real past days
hist = port_returns.values

# 3. PARAMETRIC (NORMAL) - formula for a bell curve
mean, std = port_returns.mean(), port_returns.std()

def normal_var_es(cl):
    z = norm.ppf(1 - cl)
    var = -(mean + z * std)
    es = -(mean - std * norm.pdf(z) / (1 - cl))
    return var, es

# 4. MONTE CARLO - simulate 10,000 possible days
np.random.seed(42)            # same random numbers every run, so results repeat
n_sims = 10000

asset_mean = asset_returns.mean().values
asset_cov = asset_returns.cov().values      # volatilities + correlations of all assets

# 4a. Normal Monte Carlo: random days drawn from a multivariate bell curve
sim_normal = np.random.multivariate_normal(asset_mean, asset_cov, n_sims)
mc_normal = sim_normal @ weights.values     # portfolio return for each simulated day

# 4b. Student-t Monte Carlo: same correlations, but fat tails (df = 4)
df = 4
shock = np.random.multivariate_normal(np.zeros(len(weights)), asset_cov, n_sims)
scale = np.sqrt(np.random.chisquare(df, n_sims) / (df - 2))   # makes some days extreme
sim_t = asset_mean + shock / scale[:, None]
mc_t = sim_t @ weights.values

# 5. Results table
methods = {
    "Historical": lambda cl: var_and_es(hist, cl),
    "Parametric (Normal)": normal_var_es,
    "Monte Carlo (Normal)": lambda cl: var_and_es(mc_normal, cl),
    "Monte Carlo (Student-t)": lambda cl: var_and_es(mc_t, cl),
}

rows = []
for name, calc in methods.items():
    var95, _ = calc(0.95)
    var99, _ = calc(0.99)
    _, es975 = calc(ES_CONFIDENCE)
    rows.append({
        "Method": name,
        "VaR 95% (%)": round(var95 * 100, 2),
        "VaR 99% (%)": round(var99 * 100, 2),
        "ES 97.5% (%)": round(es975 * 100, 2),
        "VaR 99% (Rs)": round(var99 * PORTFOLIO_VALUE),
        "ES 97.5% (Rs)": round(es975 * PORTFOLIO_VALUE),
    })

table = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("\n1-day VaR and Expected Shortfall on a Rs 1 crore portfolio")
print(table.to_string(index=False))

os.makedirs(OUTPUT_DIR, exist_ok=True)
table.to_csv(f"{OUTPUT_DIR}/var_advanced.csv", index=False)

# 6. Chart: historical returns with VaR 99% and ES 97.5%
var99, _ = var_and_es(hist, 0.99)
_, es975 = var_and_es(hist, ES_CONFIDENCE)

plt.figure(figsize=(10, 5))
plt.hist(hist * 100, bins=100, color="lightgray", edgecolor="gray")
plt.axvline(-var99 * 100, color="red", label=f"Historical VaR 99%: {var99*100:.2f}%")
plt.axvline(-es975 * 100, color="purple", linestyle="--", label=f"Historical ES 97.5%: {es975*100:.2f}%")
plt.title("VaR shows where the tail starts; ES shows the average loss inside it")
plt.xlabel("Daily return (%)")
plt.ylabel("Number of days")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/var_es_chart.png", dpi=150)
plt.show()

print(f"\nSaved table and chart in '{OUTPUT_DIR}/'")