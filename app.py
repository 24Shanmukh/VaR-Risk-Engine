# app.py - STEP 7: Interactive VaR dashboard
# Run from the VAR folder:  streamlit run src/app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

from config import PORTFOLIO, PORTFOLIO_VALUE

st.set_page_config(page_title="VaR Risk Dashboard", layout="wide")
st.title("Portfolio Value at Risk Dashboard")
st.caption("8 NSE large caps + Gold ETF + USD | daily data from 2015")

# 1. Load the asset returns from Step 1
returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)

# 2. Sidebar: the user's choices
st.sidebar.header("Settings")
value_lakh = st.sidebar.number_input("Portfolio value (Rs lakh)", 1, 10000, int(PORTFOLIO_VALUE / 100000))
value = value_lakh * 100000
cl = st.sidebar.selectbox("Confidence level", [0.95, 0.99], index=1, format_func=lambda x: f"{x:.0%}")
days = st.sidebar.selectbox("Horizon (days)", [1, 10])

st.sidebar.subheader("Weights (%)")
raw = {}
for ticker, w in PORTFOLIO.items():
    raw[ticker] = st.sidebar.slider(ticker, 0, 50, int(w * 100))

total = sum(raw.values())
if total == 0:
    st.error("Set at least one weight above 0.")
    st.stop()
weights = pd.Series(raw)[returns.columns] / total          # rescale so they add to 100%
st.sidebar.caption(f"Weights total {total}% - rescaled to 100%.")

# 3. Portfolio returns with the chosen weights
port = (returns * weights).sum(axis=1)
scale = np.sqrt(days)

# 4. VaR and ES with three methods
cutoff = np.percentile(port, (1 - cl) * 100)
hist_var = -cutoff * scale
hist_es = -port[port <= cutoff].mean() * scale

z = norm.ppf(1 - cl)
para_var = -(port.mean() + z * port.std()) * scale
para_es = -(port.mean() - port.std() * norm.pdf(z) / (1 - cl)) * scale

np.random.seed(42)
sims = np.random.multivariate_normal(returns.mean(), returns.cov(), 10000) @ weights.values
mc_cut = np.percentile(sims, (1 - cl) * 100)
mc_var = -mc_cut * scale
mc_es = -sims[sims <= mc_cut].mean() * scale

# 5. Headline numbers
st.subheader(f"{days}-day {cl:.0%} risk on Rs {value_lakh:,} lakh")
c1, c2, c3 = st.columns(3)
c1.metric(f"Historical VaR ({hist_var:.2%})", f"Rs {hist_var * value / 100000:.2f} lakh")
c2.metric(f"Parametric VaR ({para_var:.2%})", f"Rs {para_var * value / 100000:.2f} lakh")
c3.metric(f"Monte Carlo VaR ({mc_var:.2%})", f"Rs {mc_var * value / 100000:.2f} lakh")

table = pd.DataFrame({
    "Method": ["Historical", "Parametric (Normal)", "Monte Carlo (Normal)"],
    "VaR (Rs)": [hist_var * value, para_var * value, mc_var * value],
    "Expected Shortfall (Rs)": [hist_es * value, para_es * value, mc_es * value],
}).round(0)
st.dataframe(table, hide_index=True, width="stretch")

# 6. Charts
left, right = st.columns(2)

with left:
    st.markdown("**Distribution of daily returns**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(port * 100, bins=100, color="lightgray", edgecolor="gray")
    ax.axvline(cutoff * 100, color="red", label=f"Historical 1-day VaR {-cutoff:.2%}")
    ax.set_xlabel("Daily return (%)")
    ax.set_ylabel("Number of days")
    ax.legend()
    st.pyplot(fig)

with right:
    st.markdown("**Share of risk by asset**")
    cov = returns.cov()
    port_std = np.sqrt(weights @ cov @ weights)
    contrib = (weights * (cov @ weights) / port_std**2 * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(contrib.index, contrib.values, color="crimson")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% of portfolio risk")
    st.pyplot(fig)

# 7. EWMA VaR over time with exceptions
st.markdown("**1-day EWMA VaR over time (red dots = losses bigger than VaR)**")
ewma_std = np.sqrt((port ** 2).ewm(alpha=0.06).mean()).shift(1)
ewma_var = -(z * ewma_std)
exceptions = port[port < -ewma_var]

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(port.index, port * 100, color="gray", linewidth=0.5, label="Daily return")
ax.plot(ewma_var.index, -ewma_var * 100, color="blue", linewidth=1, label=f"-VaR {cl:.0%}")
ax.scatter(exceptions.index, exceptions * 100, color="red", s=10, label=f"Exceptions ({len(exceptions)})")
ax.set_ylabel("Return (%)")
ax.legend(loc="lower left")
st.pyplot(fig)

st.caption("Built in Python with pandas, NumPy, SciPy and Streamlit. Data: Yahoo Finance.")