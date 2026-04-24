# Bitcoin Monte Carlo Simulator

![Python](https://img.shields.io/badge/Python-3.12-orange?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple?logo=plotly&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-Historical%20Data-lightgrey)

A **Monte Carlo** simulator for **Bitcoin (BTC-USD)** using a **Geometric Brownian Motion (GBM)** price model.

This project includes a **Streamlit** interface: users can calibrate parameters from historical data (average return μ and volatility σ), run thousands of simulations, explore scenario percentiles (P10/P50/P90), visualise distributions and sample trajectories.

---

## Disclaimer

**This simulator is designed for educational and illustrative purposes only.**

**It does not constitute investment advice and does not guarantee any future performance.**  
**All projections are based on simplified assumptions that may differ significantly from real-world outcomes.**

**Investing involves risk, including the potential loss of capital.**

---

## Table of Contents

- [Why this project?](#why-this-project)
- [Features](#features)
- [Model overview](#model-overview)
- [Limitations](#limitations)
- [Installation](#installation)
- [How to use](#how-to-use)

---

## Why this project?

This project started as a **personal learning and investing tool**.

I wanted:
- A simple and transparent Monte Carlo simulator for BTC.
- A practical way to understand **GBM**, **log returns** and **volatility calibration**.
- An interactive UI to test scenarios quickly (horizon, target price, number of simulations, etc.).

---

## Features

### Historical calibration (BTC-USD)
- Downloads BTC-USD historical prices using `yfinance`.
- Estimates parameters from log-returns:
  - **μ (drift)**: average annualised log-return (can be overridden)
  - **σ (volatility)**: annualised volatility (can be overridden)
- User selects a **lookback window** (in years) to control how much history is used.

### Monte Carlo simulation (GBM)
- Simulates **N paths** of BTC price evolution with GBM.
- Output includes:
  - Scenario percentiles: **P10 / P50 / P90**
  - Distribution of final prices
  - Probability that final price ends above a user-defined **target**

### Streamlit UI
- Parameter controls (horizon, simulations, lookback, target price)
- Advanced panel (μ/σ overrides, seed)
- Visualisations:
  - Sample simulated paths
  - Percentile curves overlay
  - Histogram of final prices
- Optional **log scale** for better readability when extreme paths appear.

---

## Model overview

### Geometric Brownian Motion (GBM)
GBM assumes the **log price** follows a random walk with drift:

- Drift term (trend): **μ**
- Random shocks: Gaussian noise scaled by **σ**

At each step:
- The model draws a random normal variable `Z ~ N(0,1)`
- Price evolves multiplicatively (hence `exp(cumsum(...))`)

This is a standard baseline model used in quantitative finance (also related to Black–Scholes assumptions).

### Percentiles (P10 / P50 / P90)
At each time step:
- **P10**: 10% of simulated paths are below this value (bear-ish scenario)
- **P50**: median scenario
- **P90**: 90% of paths are below this value (bull-ish scenario)

---

## Limitations

This project is designed for **scenario exploration**, not exact forecasting.

- GBM is simplified and does **not** model:
  - sudden crashes / jumps
  - regime changes (bull/bear regimes)
  - and other complicated stuff
- Calibration is sensitive to the lookback window.
- No fees, no taxes, inflation rate etc.
- BTC-USD historical data quality depends on `yfinance`.

---

## Installation

### Option A — Local (Python venv)

1. Clone the repository and go into the project directory:
    ```bash
    git clone https://github.com/Pstkdev/BTC-Monte-Carlo-Simulator.git
    cd BTC-Monte-Carlo-Simulator/
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the app (Streamlit) from the project root:
    ```bash
    streamlit run app.py
    ```

Then open http://localhost:8501

### Option B — Docker Compose

1. Clone the repository and go into the project directory:
    ```bash
    git clone https://github.com/Pstkdev/BTC-Monte-Carlo-Simulator.git
    cd BTC-Monte-Carlo-Simulator/
    ```

2. Build and run:
    ```bash
    docker compose up --build
    ```

Then open http://localhost:8501

## How to use

In the sidebar you can set:

- **Historical window (years):** how much BTC history is used to estimate average return μ and volatility σ  
- **Simulation horizon:** number of years simulated into the future
- **Number of simulations:** number of Monte Carlo paths
- **Target price:** used to compute `P(final price > target)`
- **Chart options:** log-scale for readability  

### Advanced parameters

- **Override μ (drift)**
- **Override σ (volatility)**
- **Random seed**

The app displays:

- Summary metrics (bear / base / bull, probability above target, volatility σ)  
- Simulated paths + percentile curves  
- Final price distribution  
- A percentile table (can be downloaded as CSV)
