
from __future__ import annotations

import numpy as np
import pandas as pd
try:
    import streamlit as _streamlit
    _cache_data = _streamlit.cache_data
except Exception:
    def _cache_data(**_kwargs):
        def deco(func):
            return func
        return deco

import plotly.express as px
try:
    import streamlit as _streamlit
    _cache_data = _streamlit.cache_data
except Exception:
    def _cache_data(**_kwargs):
        def deco(func):
            return func
        return deco

import plotly.graph_objects as go

from dashboard_components import alert_box, fmt_money, fmt_pct, metric_card, source_caption
from dashboard_data_layer import current_holdings, numeric, read_price_cache
from dashboard_theme import apply_plotly_layout


@_cache_data(show_spinner=False)
def _simulate_paths(seed: int, returns_key: tuple[float, ...], horizon: int, sims: int, initial: float, model: str) -> np.ndarray:
    returns = np.asarray(returns_key, dtype=float)
    rng = np.random.default_rng(seed)
    if model == "bootstrap historical":
        idx = rng.integers(0, len(returns), size=(horizon, sims))
        draws = returns[idx]
    else:
        mu = np.nanmean(returns)
        sigma = np.nanstd(returns)
        if model == "Student-t":
            draws = mu + sigma * rng.standard_t(df=5, size=(horizon, sims)) / np.sqrt(5 / 3)
        else:
            draws = rng.normal(mu, sigma, size=(horizon, sims))
    return initial * np.cumprod(1 + draws, axis=0)


def render_monte_carlo_lab(st, data: dict[str, pd.DataFrame]) -> dict[str, object]:
    st.markdown("#### Monte Carlo Lab")
    horizon = int(st.selectbox("Horizon sessions", [20, 60, 126, 252], index=1, key="qlab_mc_horizon"))
    sims = int(st.slider("Simulations", 250, 5000, 1000, 250, key="qlab_mc_sims"))
    model = st.selectbox("Return model", ["Gaussian", "Student-t", "bootstrap historical"], key="qlab_mc_model")
    holdings = current_holdings(data, "Official Forward Paper")
    tickers = holdings[~holdings.get("ticker", "").astype(str).str.upper().eq("CASH")]["ticker"].astype(str).str.upper().tolist() if not holdings.empty and "ticker" in holdings.columns else []
    if not tickers:
        alert_box(st, "Monte Carlo unavailable: no official holdings.", "warning")
        return {"surface": "monte_carlo", "status": "warning", "detail": "no holdings"}
    prices = read_price_cache(tickers, lookback=280)
    returns = prices.drop(columns=["date"], errors="ignore").pct_change().dropna(how="all") if not prices.empty else pd.DataFrame()
    if returns.empty:
        alert_box(st, "Monte Carlo unavailable: price history missing.", "warning")
        return {"surface": "monte_carlo", "status": "warning", "detail": "missing returns"}
    non_cash = holdings[~holdings.get("ticker", "").astype(str).str.upper().eq("CASH")] if "ticker" in holdings.columns else holdings
    weights = numeric(non_cash.get("paper_position_weight", pd.Series(dtype=float))).dropna().values
    weights = weights[:returns.shape[1]] if len(weights) else np.repeat(1 / returns.shape[1], returns.shape[1])
    if weights.sum() > 0:
        weights = weights / weights.sum()
    port = returns.iloc[:, :len(weights)].fillna(0).values @ weights
    initial = float(numeric(holdings.get("gross_portfolio_value", pd.Series([100000]))).dropna().iloc[-1]) if "gross_portfolio_value" in holdings.columns and numeric(holdings.get("gross_portfolio_value", pd.Series(dtype=float))).notna().any() else 100000.0
    paths = _simulate_paths(114, tuple(np.round(port.astype(float), 10)), horizon, sims, initial, model)
    terminal = paths[-1]
    fig = go.Figure()
    sample = paths[:, : min(80, sims)]
    for i in range(sample.shape[1]):
        fig.add_trace(go.Scatter(y=sample[:, i], mode="lines", line={"width": 0.7, "color": "rgba(255,122,0,0.13)"}, showlegend=False))
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    for label, row in zip(["p05", "p25", "median", "p75", "p95"], pct):
        fig.add_trace(go.Scatter(y=row, mode="lines", name=label))
    apply_plotly_layout(fig, "Monte Carlo Path Fan")
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")
    hist = px.histogram(pd.DataFrame({"terminal_value": terminal}), x="terminal_value", nbins=50, title="Terminal Value Distribution")
    apply_plotly_layout(hist, "Terminal Value Distribution")
    hist.update_layout(height=360)
    st.plotly_chart(hist, width="stretch")
    ret = terminal / initial - 1
    var95 = np.percentile(ret, 5)
    cvar95 = ret[ret <= var95].mean() if np.any(ret <= var95) else np.nan
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Expected Terminal", fmt_money(np.mean(terminal)))
    with cols[1]: metric_card(st, "Probability Loss", fmt_pct(np.mean(terminal < initial)))
    with cols[2]: metric_card(st, "VaR 95", fmt_pct(var95))
    with cols[3]: metric_card(st, "CVaR 95", fmt_pct(cvar95))
    alert_box(st, "Simulation outcomes depend on historical distributional assumptions.", "info")
    source_caption(st, "yahoo_ohlcv_price_cache + official holdings", "diagnostic simulation")
    return {"surface": "monte_carlo", "status": "available", "detail": f"horizon={horizon}, sims={sims}, model={model}"}
