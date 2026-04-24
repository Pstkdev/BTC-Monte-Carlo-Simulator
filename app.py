import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from src.calibration import fetch_adj_close, estimate_mu_sigma
from src.btc_simulation import BTCMonteCarlo

st.set_page_config(
    page_title="BTC Monte Carlo Simulator",
    page_icon="assets/btc.png",
    layout="wide",
)
st.title("Bitcoin Monte Carlo Simulator")
st.caption("Geometric Brownian Motion (GBM) Monte Carlo simulation for BTC-USD")


# ------------ helpers ------------ #
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


@st.cache_data(show_spinner=False)
def build_viz_long_df(paths: np.ndarray, idx: tuple[int, ...], years: int) -> pd.DataFrame:
    """
    Build a long DataFrame for Plotly from a subset of simulated paths.
    Cached because this is expensive.
    """
    t_years = np.linspace(0, years, paths.shape[1])
    df_paths = pd.DataFrame(paths[list(idx)], columns=t_years)
    df_paths["path_id"] = [f"path_{i}" for i in idx]
    df_long = df_paths.melt(id_vars="path_id", var_name="t", value_name="price")
    df_long["t"] = df_long["t"].astype(float)
    return df_long


# ------------ Sidebar ------------ #
st.sidebar.image("assets/bitcoin.png", width=250)
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

n_show = st.sidebar.slider(
    "Number of simulated paths to show",
    min_value=20,
    max_value=300,
    value=100,
    step=10,
    help="Number of simulated paths to show in the spaghetti plot.",
)

threshold_price = st.sidebar.number_input(
    f"Target price ($)",
    min_value=1.0,
    value=100000.0,
    step=1000.0,
    help="Compute the chance that BTC ends ABOVE this value at the end of the simulation horizon.",
)


# ---- Chart options ---- #
st.sidebar.divider()
st.sidebar.subheader("Chart options")
log_y = st.sidebar.checkbox(
    "Log scale (Y axis)",
    value=False,
    help="Better readability when some simulated paths explode. Log scale compresses extreme values.",
)


# ------------ BTC Data Calibration ------------ #
@st.cache_data(show_spinner=False)
def calibrate_btc(lookback_years: int, last_date: pd.Timestamp):
    start_date = last_date - pd.Timedelta(days=int(lookback_years * 365.25))
    prices = fetch_adj_close("BTC-USD", start=str(start_date.date()))
    mu, sigma = estimate_mu_sigma(prices, steps_per_year=365)
    last_price = float(prices.iloc[-1])
    return last_price, float(mu), float(sigma), prices


with st.spinner("Fetching BTC-USD and calibrating μ/σ..."):
    start_price, mu_cal, sigma_cal, prices_hist = calibrate_btc(int(lookback_years), last_date)


# ------------ Advanced parameters ------------ #
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

# Apply overrides after the expander inputs exist
mu = float(mu_override) if override_mu else float(mu_cal)
sigma = float(sigma_override) if override_sigma else float(sigma_cal)


# ------------ RUN SIMULATION ------------ #
@st.cache_data(show_spinner=False)
def run_mc(start_price, mu, sigma, years, num_simulations, seed):
    sim = BTCMonteCarlo(start_price, mu, sigma, years, num_simulations, seed)
    paths = sim.simulate_paths()
    q = sim.compute_quantiles(paths, qs=(0.1, 0.5, 0.9))
    return paths, q


with st.spinner("Running simulations..."):
    paths, q = run_mc(float(start_price), float(mu), float(sigma), int(years), int(num_simulations), int(seed))

# simple time axis in years
num_steps = q.shape[1] - 1
t_years = np.linspace(0, years, num_steps + 1)

df_q = pd.DataFrame({"t": t_years, "P10": q[0], "P50": q[1], "P90": q[2]})

final_prices = paths[:, -1]
# get percentiles and probability of ending above threshold
p10_final = float(np.quantile(final_prices, 0.10))
p50_final = float(np.quantile(final_prices, 0.50))
p90_final = float(np.quantile(final_prices, 0.90))
prob_over = float((final_prices > threshold_price).mean()) * 100.0


# ------------ Metrics ------------ #

col1, col2, col3, col4 = st.columns(4)

col1.metric("Start price (BTC-USD)", f"${start_price:,.0f}")
col2.metric("Horizon", f"{years} year(s)")
col3.metric("Simulations", f"{num_simulations:,}")
col4.metric(f"Chance BTC ends above ${threshold_price:,.0f}", f"{prob_over:.1f}%")

col1.metric("Bear case", f"${p10_final:,.0f}")
col2.metric("Base case", f"${p50_final:,.0f}")
col3.metric("Bull case", f"${p90_final:,.0f}")
col4.metric("Annual volatility (σ)", f"{sigma*100:.1f}%")

st.divider()


# ------------ Charts ------------ #

# Graph 1: Spaghetti plot settings
st.subheader("Simulated price paths")

# random subset of paths for visualization
rng_vis = np.random.default_rng(seed)
idx = rng_vis.choice(paths.shape[0], size=min(n_show, paths.shape[0]), replace=False)
idx_tuple = tuple(int(i) for i in idx)

df_long = build_viz_long_df(paths, idx_tuple, int(years))

# Build long dataframe
df_long = build_viz_long_df(paths, idx, int(years))

# ---- Graph 1 ----

fig_paths = px.line(
    df_long,
    x="t",
    y="price",
    line_group="path_id",
    color="path_id",
)
fig_paths.update_traces(line=dict(width=1))
fig_paths.update_layout(showlegend=True, xaxis_title="Years", yaxis_title="BTC Price (USD)")
fig_paths.update_yaxes(type="log" if log_y else "linear")

st.plotly_chart(fig_paths, width="stretch")
st.caption(
    "Note: number of paths shown is limited for readability. Use the slider in the sidebar to adjust. "
    "You can hide a path by clicking its id in the legend."
    " If paths explode, enable Log scale in the sidebar."
)

# ---- Graph 2 ----
st.subheader("Percentile curves (P10 / P50 / P90)")

fig_overlay = px.line(
    df_long,
    x="t",
    y="price",
    line_group="path_id",
    title="Percentile curves over simulated paths",
)

# make paths grey + transparent
fig_overlay.update_traces(line=dict(width=1, color="rgba(160,160,160,0.25)"))
fig_overlay.update_layout(showlegend=True, xaxis_title="Years", yaxis_title="BTC Price (USD)")

# add quantiles on top
fig_overlay.add_scatter(
    x=df_q["t"], y=df_q["P10"], mode="lines", name="P10 (Bear)", line=dict(width=3, color="#FF3C3C")
)
fig_overlay.add_scatter(
    x=df_q["t"], y=df_q["P50"], mode="lines", name="P50 (Base)", line=dict(width=4, color="#FF9845")
)
fig_overlay.add_scatter(
    x=df_q["t"], y=df_q["P90"], mode="lines", name="P90 (Bull)", line=dict(width=3, color="#3DFF57")
)

fig_overlay.update_yaxes(type="log" if log_y else "linear")

st.plotly_chart(fig_overlay, width="stretch")

st.caption(
    "P10 can be interpreted as a pessimistic scenario, P50 as a base case, and P90 as an optimistic scenario. "
    "If paths explode, enable Log scale in the sidebar."
)

# ---- Graph 3 ----

df_final = pd.DataFrame({"final_price": final_prices})
cap = float(np.quantile(final_prices, 0.95))  # P95
df_hist = df_final[df_final["final_price"] <= cap]

BIN_SIZE = 25_000  # adjust bin size for readability based on the range of final prices

fig_hist = px.histogram(
    df_hist,
    x="final_price",
    title="Final price distribution",
)
fig_hist.update_traces(
    marker_color="#F7931A",
    marker_line_color="rgba(0,0,0,0.25)",
    marker_line_width=1,
)
fig_hist.update_traces(xbins=dict(start=0, end=cap, size=BIN_SIZE))
fig_hist.update_xaxes(title="Final price (USD)", range=[0, cap])
fig_hist.update_yaxes(title="Count")

st.plotly_chart(fig_hist, width="stretch")
st.caption("Note: chart is capped at the 95th percentile for readability.")

st.divider()

# ---- Data ----
st.subheader("Bitcoin price scenario summary")
st.dataframe(df_q, width="stretch")
st.caption(
    "Summary of P10 (bear), P50 (base) and P90 (bull) market scenarios across the simulation horizon. "
    "Can be downloaded as a CSV file using the button in the top-right corner of the table."
)
