# stress_test.py - STEP 6: Stress testing - what happens in a crisis?
# Run from the VAR folder:  python src/stress_test.py
#
# VaR covers "normal bad days". Stress tests ask: "what if a real crisis happens?"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PORTFOLIO, PORTFOLIO_VALUE, OUTPUT_DIR

returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
port_returns = pd.read_csv("data/portfolio_returns.csv", index_col=0, parse_dates=True)["portfolio"]
weights = pd.Series(PORTFOLIO)[returns.columns]

# VaR numbers to compare against (historical, 99%)
var_1day = -np.percentile(port_returns, 1)
var_10day = var_1day * np.sqrt(10)

results = []

# ---------------------------------------------------------------------------
# PART A: HISTORICAL SCENARIOS - replay real Indian market crises
# (start and end dates are approximate market peak -> bottom)
# ---------------------------------------------------------------------------
historical_scenarios = {
    "Demonetisation (Nov-Dec 2016)":   ("2016-11-08", "2016-12-26"),
    "IL&FS crisis (Aug-Oct 2018)":     ("2018-08-28", "2018-10-26"),
    "COVID crash (Feb-Mar 2020)":      ("2020-02-20", "2020-03-23"),
    "Russia-Ukraine (Jan-Mar 2022)":   ("2022-01-18", "2022-03-07"),
    "Election result day (4 Jun 2024)": ("2024-06-04", "2024-06-04"),
}

for name, (start, end) in historical_scenarios.items():
    period = returns.loc[start:end]
    if len(period) == 0:
        continue                                   # dates not in our data
    asset_change = np.exp(period.sum()) - 1        # total % change of each asset
    portfolio_change = (weights * asset_change).sum()
    results.append({"Scenario": name, "Type": "Historical",
                    "Days": len(period), "Portfolio change %": portfolio_change * 100})

# ---------------------------------------------------------------------------
# PART B: HYPOTHETICAL SCENARIOS - "what if" shocks we choose ourselves
# ---------------------------------------------------------------------------
stocks = [t for t in weights.index if t not in ("GOLDBEES.NS", "INR=X")]
banks = ["HDFCBANK.NS", "ICICIBANK.NS"]
it = ["INFY.NS", "TCS.NS"]

hypothetical_scenarios = {
    # shock = % move for each asset (anything not listed = 0%)
    "Market crash: stocks -20%, gold +5%, USD +3%":
        {**{s: -0.20 for s in stocks}, "GOLDBEES.NS": 0.05, "INR=X": 0.03},
    "Banking crisis: banks -30%, other stocks -10%":
        {**{s: -0.10 for s in stocks}, **{b: -0.30 for b in banks}},
    "IT sell-off: Infosys & TCS -25%":
        {i: -0.25 for i in it},
    "Rupee crash: USD +10%, stocks -10%":
        {**{s: -0.10 for s in stocks}, "INR=X": 0.10},
    "Everything falls: stocks -15%, gold -10%, USD -5%":
        {**{s: -0.15 for s in stocks}, "GOLDBEES.NS": -0.10, "INR=X": -0.05},
}

for name, shock in hypothetical_scenarios.items():
    asset_change = pd.Series(shock).reindex(weights.index, fill_value=0)
    portfolio_change = (weights * asset_change).sum()
    results.append({"Scenario": name, "Type": "Hypothetical",
                    "Days": "-", "Portfolio change %": portfolio_change * 100})

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
table = pd.DataFrame(results)
table["P&L (Rs)"] = (table["Portfolio change %"] / 100 * PORTFOLIO_VALUE).round(0)
table["Loss / 10-day VaR"] = (-table["Portfolio change %"] / 100 / var_10day).round(1)
table["Portfolio change %"] = table["Portfolio change %"].round(2)

pd.set_option("display.width", 160)
print("\nStress test results on a Rs 1 crore portfolio")
print(table.to_string(index=False))
print(f"\nFor comparison: 1-day 99% VaR = Rs {var_1day * PORTFOLIO_VALUE:,.0f}, "
      f"10-day 99% VaR = Rs {var_10day * PORTFOLIO_VALUE:,.0f}")
print("'Loss / 10-day VaR' above 1 means the crisis loss was bigger than VaR predicted.")

# Worst 5 single days in our data
worst = port_returns.nsmallest(5)
print("\nWorst 5 days for this portfolio:")
for date, r in worst.items():
    print(f"  {date.date()}   {r * 100:6.2f}%   Rs {r * PORTFOLIO_VALUE:,.0f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
table.to_csv(f"{OUTPUT_DIR}/stress_test.csv", index=False)

# Chart: P&L of each scenario with the VaR lines
chart = table.sort_values("P&L (Rs)")
colors = ["crimson" if t == "Historical" else "darkorange" for t in chart["Type"]]
plt.figure(figsize=(11, 6))
plt.barh(chart["Scenario"], chart["P&L (Rs)"] / 100000, color=colors)
plt.axvline(-var_1day * PORTFOLIO_VALUE / 100000, color="blue", linestyle="--", label="1-day 99% VaR")
plt.axvline(-var_10day * PORTFOLIO_VALUE / 100000, color="navy", label="10-day 99% VaR")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Stress test P&L (red = historical, orange = hypothetical)")
plt.xlabel("Profit / loss (Rs lakh)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/stress_test.png", dpi=150)
plt.show()

print(f"\nSaved table and chart in '{OUTPUT_DIR}/'")