import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from src.calibration import fetch_adj_close, estimate_mu_sigma
from src.btc_simulation import BTCMonteCarlo


st.set_page_config(page_title="BTC Monte Carlo Simulator", layout="wide")
st.title("Bitcoin Monte Carlo Simulator")
st.caption("GBM Monte Carlo simulation for BTC-USD")


# ---- helpers ----
@st.cache_data(show_spinner=False)
def fetch_btc_history_bounds():
    """
    Download as much BTC-USD history as possible once, and return (first_date, last_date).
    Cached to avoid repeated downloads.
    """
    prices = fetch_adj_close("BTC-USD", start="2010-01-01")
    first_date = prices.index.min()
    last_date = prices.index.max()
    return first_date, last_date


def max_available_years(first_date: pd.Timestamp, last_date: pd.Timestamp) -> int:
    days = (last_date - first_date).days
    return max(1, int(days // 365.25))


# ---- Sidebar ----
st.sidebar.header("Simulation parameters")

first_date, last_date = fetch_btc_history_bounds()
max_years = max_available_years(first_date, last_date)

st.sidebar.markdown(
    f"**BTC-USD history available:**<br>" f"{first_date.date()} → {last_date.date()}" f" (~{max_years} years)",
    unsafe_allow_html=True,
)

lookback_years = st.sidebar.number_input(
    "Historical window (years)",
    min_value=1,
    value=min(4, max_years),
    step=1,
    help=(
        "Used for calibration: estimate volatility and average return from historical BTC data. "
        "Longer window includes more market regimes (bull + bear) which can increase volatility."
    ),
)

if lookback_years > max_years:
    st.sidebar.error(f"Max available lookback is {max_years} years based on BTC-USD data.")
    st.stop()

years = st.sidebar.slider(
    "Simulation horizon (years)",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="Number of years to simulate BTC price paths into the future.",
)

num_simulations = st.sidebar.slider(
    "Number of simulations",
    min_value=200,
    max_value=20000,
    value=5000,
    step=200,
    help="Number of Monte Carlo paths to simulate. More paths = more accurate estimates.",
)

threshold_price = st.sidebar.number_input(
    f"Target price ($)",
    min_value=1.0,
    value=100000.0,
    step=1000.0,
    help="Compute the chance that BTC ends ABOVE this value at the end of the simulation horizon.",
)


# ---- Calibration ----
@st.cache_data(show_spinner=False)
def calibrate_btc(lookback_years: int, last_date: pd.Timestamp):
    start_date = last_date - pd.Timedelta(days=int(lookback_years * 365.25))
    prices = fetch_adj_close("BTC-USD", start=str(start_date.date()))
    mu, sigma = estimate_mu_sigma(prices, steps_per_year=365)
    last_price = float(prices.iloc[-1])
    return last_price, float(mu), float(sigma), prices


with st.spinner("Fetching BTC-USD and calibrating μ/σ..."):
    start_price, mu_cal, sigma_cal, prices_hist = calibrate_btc(int(lookback_years), last_date)

# ---- Advanced (overrides + seed) ----
with st.sidebar.expander("Advanced parameters", expanded=False):
    st.caption("Calibration (annualised)")
    st.write(f"μ (drift): {mu_cal:.3f}")
    st.write(f"σ (vol): {sigma_cal:.3f}")

    override_mu = st.checkbox(
        "Override μ (annual drift)",
        value=False,
        help="Model’s average return estimated from history. Higher μ pushes paths upward.",
    )
    mu_override = st.number_input("μ annual", value=mu_cal, step=0.01, format="%.4f", disabled=not override_mu)

    override_sigma = st.checkbox("Override σ (annual vol)", value=False, help="Annualised volatility of log-returns.")
    sigma_override = st.number_input("σ annual", value=sigma_cal, step=0.01, format="%.4f", disabled=not override_sigma)

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
        help="Same seed = same simulated paths.",
    )

# Apply overrides AFTER the expander inputs exist
mu = float(mu_override) if override_mu else float(mu_cal)
sigma = float(sigma_override) if override_sigma else float(sigma_cal)

# ---- Run simulation ----
sim = BTCMonteCarlo(
    start_price=float(start_price),
    mu=float(mu),
    sigma=float(sigma),
    years=int(years),
    num_simulations=int(num_simulations),
    seed=int(seed),
)

paths = sim.simulate_paths()
q = sim.compute_quantiles(paths, qs=(0.1, 0.5, 0.9))

# simple time axis in years
num_steps = q.shape[1] - 1
t_years = np.linspace(0, years, num_steps + 1)

df_q = pd.DataFrame({"t": t_years, "P10": q[0], "P50": q[1], "P90": q[2]})

final_prices = paths[:, -1]
p10_final = float(np.quantile(final_prices, 0.10))
p50_final = float(np.quantile(final_prices, 0.50))
p90_final = float(np.quantile(final_prices, 0.90))
prob_over = float((final_prices > threshold_price).mean()) * 100.0

# ---- Layout ----
col1, col2, col3, col4 = st.columns(4)

col1.metric("Start price (BTC-USD)", f"${start_price:,.0f}")
col2.metric("Horizon", f"{years} year(s)")
col3.metric("Simulations", f"{num_simulations:,}")
col4.metric("Chance BTC ends above target", f"{prob_over:.1f}%")

col1.metric("Bear case", f"${p10_final:,.0f}")
col2.metric("Base case", f"${p50_final:,.0f}")
col3.metric("Bull case", f"${p90_final:,.0f}")
col4.metric("σ used (annual)", f"{sigma:.3f}")

st.divider()

# ---- Charts ----

# ---- Spaghetti plot settings ----
st.subheader("Sample simulated paths")
n_show = st.slider("Paths to display", min_value=20, max_value=300, value=80, step=10)

rng_vis = np.random.default_rng(seed)
idx = rng_vis.choice(paths.shape[0], size=min(n_show, paths.shape[0]), replace=False)

t_years = np.linspace(0, years, paths.shape[1])  # paths already includes t=0 column

df_paths = pd.DataFrame(paths[idx], columns=t_years)
df_paths["path_id"] = [f"path_{i}" for i in idx]  # keep identity

df_long = df_paths.melt(id_vars="path_id", var_name="t", value_name="price")
df_long["t"] = df_long["t"].astype(float)

fig_paths = px.line(
    df_long,
    x="t",
    y="price",
    line_group="path_id",
    color="path_id",
    title=f"{len(idx)} simulated BTC paths",
)
fig_paths.update_traces(line=dict(width=1), opacity=0.70)
fig_paths.update_layout(showlegend=True, xaxis_title="Years", yaxis_title="Price (USD)")

# --- Overlay quantiles on top of spaghetti ---
fig_paths.add_scatter(
    x=df_q["t"],
    y=df_q["P10"],
    mode="lines",
    name="P10",
    line=dict(width=3),
)
fig_paths.add_scatter(
    x=df_q["t"],
    y=df_q["P50"],
    mode="lines",
    name="P50 (median)",
    line=dict(width=4),
)
fig_paths.add_scatter(
    x=df_q["t"],
    y=df_q["P90"],
    mode="lines",
    name="P90",
    line=dict(width=3),
)

st.plotly_chart(fig_paths, width="stretch")

st.subheader("Charts")
fig_fan = px.line(df_q, x="t", y=["P10", "P50", "P90"], title="BTC price quantiles over time")
fig_fan.update_layout(xaxis_title="Years", yaxis_title="Price (USD)")
st.plotly_chart(fig_fan, width="stretch")

df_final = pd.DataFrame({"final_price": final_prices})


cap = float(np.quantile(final_prices, 0.95))  # P95
df_hist = df_final[df_final["final_price"] <= cap]

BIN_SIZE = 25_000

fig_hist = px.histogram(
    df_hist,
    x="final_price",
    title=f"Final price distribution (capped at P95 = {cap:,.0f})",
)
fig_hist.update_traces(xbins=dict(start=0, end=cap, size=BIN_SIZE))
fig_hist.update_xaxes(title="Final price (USD)", range=[0, cap])
fig_hist.update_yaxes(title="Count")
st.plotly_chart(fig_hist, width="stretch")

st.caption("Note: chart is capped at the 95th percentile for readability.")

st.divider()

# ---- Data ----
st.subheader("Quantiles table (preview)")
st.dataframe(df_q, width="stretch")
