from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard_components import source_caption
from dashboard_data_layer import current_holdings, numeric, read_price_cache
from dashboard_theme import AMBER, RED, apply_plotly_layout


def _surface(st, z: pd.DataFrame, title: str, source: str, status: str) -> None:
    if z.empty or z.shape[0] < 2 or z.shape[1] < 2:
        st.warning("Insufficient data for this surface.")
        source_caption(st, source, status)
        return
    fig = go.Figure(data=[go.Surface(z=z.values, x=list(z.columns), y=list(z.index), colorscale="Viridis")])
    fig.update_layout(scene={"xaxis_title": "Ticker", "yaxis_title": "Ticker", "zaxis_title": title.split()[0]})
    fig = apply_plotly_layout(fig, title)
    st.plotly_chart(fig, width="stretch")
    source_caption(st, source, status)


def _current_tickers(data: dict[str, pd.DataFrame]) -> list[str]:
    holdings = current_holdings(data, "Official Forward Paper")
    if holdings.empty or "ticker" not in holdings.columns:
        return []
    tickers = holdings[~holdings["ticker"].astype(str).str.upper().eq("CASH")]["ticker"].astype(str).str.upper().unique().tolist()
    return tickers


def render_correlation_surface(st, data: dict[str, pd.DataFrame]) -> None:
    tickers = _current_tickers(data)
    prices = read_price_cache(tickers, lookback=260)
    if prices.empty:
        st.warning("Insufficient data for this surface.")
        source_caption(st, "yahoo_ohlcv_price_cache", "official diagnostic")
        return
    returns = prices.drop(columns=["date"], errors="ignore").pct_change().dropna(how="all")
    corr = returns.corr().fillna(0)
    _surface(st, corr, "Correlation Surface", "yahoo_ohlcv_price_cache adjusted close, current official holdings", "official diagnostic")


def render_covariance_surface(st, data: dict[str, pd.DataFrame]) -> None:
    lookback = st.select_slider("Covariance lookback", options=[60, 126, 252], value=126)
    tickers = _current_tickers(data)
    prices = read_price_cache(tickers, lookback=lookback + 5)
    if prices.empty:
        st.warning("Insufficient data for this surface.")
        source_caption(st, "yahoo_ohlcv_price_cache", "official diagnostic")
        return
    returns = prices.drop(columns=["date"], errors="ignore").pct_change().tail(lookback).dropna(how="all")
    cov = returns.cov().fillna(0) * 252
    _surface(st, cov, "Covariance Surface", "yahoo_ohlcv_price_cache adjusted close, annualized covariance", "official diagnostic")


def render_vol_target_surface(st, data: dict[str, pd.DataFrame]) -> None:
    vol = np.linspace(0.05, 0.9, 42)
    target = np.linspace(0.12, 0.30, 36)
    x, y = np.meshgrid(vol, target)
    exposure = np.minimum(np.maximum(y / x, 0.40), 0.60)
    fresh = data.get("vol_fresh", pd.DataFrame())
    dual_cap = 0.60
    cur_vol = np.nan
    cur_target = 0.22
    cur_exposure = np.nan
    if not fresh.empty:
        row = fresh.sort_values("date").tail(1).iloc[0]
        dual_cap = float(row.get("dual_trend_cap", 0.60) or 0.60)
        cur_vol = float(row.get("estimated_portfolio_vol", np.nan))
        cur_target = float(row.get("target_vol", 0.22) or 0.22)
        cur_exposure = float(row.get("final_exposure", np.nan))
    exposure = np.minimum(exposure, dual_cap)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=exposure, colorscale="Teal")])
    if not np.isnan(cur_vol) and not np.isnan(cur_exposure):
        fig.add_trace(go.Scatter3d(x=[cur_vol], y=[cur_target], z=[cur_exposure], mode="markers+text", text=["current"], marker={"size": 7, "color": AMBER}))
    fig.update_layout(scene={"xaxis_title": "Estimated portfolio volatility", "yaxis_title": "Target volatility", "zaxis_title": "Exposure"})
    fig = apply_plotly_layout(fig, "Volatility Target Surface")
    st.plotly_chart(fig, width="stretch")
    source_caption(st, "growth_volatility_targeting_fresh.csv", "official diagnostic")


def render_parameter_surface(st, data: dict[str, pd.DataFrame]) -> None:
    df = data.get("parameter_stability", pd.DataFrame())
    if df.empty:
        st.warning("Insufficient data for this surface.")
        source_caption(st, "parameter_stability_map.csv", "research diagnostic")
        return
    metric = st.selectbox("Parameter surface metric", [c for c in ["Sharpe", "CAGR", "max_drawdown", "Calmar"] if c in df.columns], index=0)
    subset = df.copy()
    if "min_exposure" in subset.columns:
        subset = subset[subset["min_exposure"].astype(str).eq("0.4") | subset["min_exposure"].astype(str).eq("40") | subset["min_exposure"].astype(str).str.lower().eq("0.40")]
        if subset.empty:
            subset = df.copy()
    if "vol_lookback_days" in subset.columns:
        subset = subset[numeric(subset["vol_lookback_days"]).eq(60)] if numeric(subset["vol_lookback_days"]).eq(60).any() else subset
    pivot = subset.pivot_table(index="target_vol", columns="exposure_cap", values=metric, aggfunc="mean").sort_index().sort_index(axis=1)
    if pivot.empty:
        st.warning("Insufficient data for this surface.")
        return
    fig = go.Figure(data=[go.Surface(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Cividis")])
    if "Sharpe" in metric or metric in ["CAGR", "Calmar"]:
        marker_z = pivot.loc[pivot.index[np.abs(pivot.index - 0.22).argmin()], pivot.columns[np.abs(pivot.columns - 0.60).argmin()]]
        fig.add_trace(go.Scatter3d(x=[0.60], y=[0.22], z=[marker_z], mode="markers+text", text=["active 22/60"], marker={"size": 8, "color": RED}))
    fig.update_layout(scene={"xaxis_title": "Exposure cap", "yaxis_title": "Target volatility", "zaxis_title": metric})
    fig = apply_plotly_layout(fig, f"Parameter Stability Surface — {metric}")
    st.plotly_chart(fig, width="stretch")
    source_caption(st, "parameter_stability_map.csv", "research diagnostic — not used by official allocation")


def render_hmm_3d(st, data: dict[str, pd.DataFrame]) -> None:
    df = data.get("hmm_oos", pd.DataFrame())
    if df.empty or not {"test_risk_off_rate", "future_volatility_corr_proxy", "test_state_switch_rate"}.issubset(df.columns):
        st.warning("Insufficient data for this surface.")
        source_caption(st, "hmm_out_of_sample_results.csv", "research diagnostic")
        return
    work = df.copy()
    fig = go.Figure(data=[go.Scatter3d(
        x=numeric(work["test_risk_off_rate"]),
        y=numeric(work["future_volatility_corr_proxy"]),
        z=numeric(work["test_state_switch_rate"]),
        mode="markers",
        marker={"size": 5, "color": numeric(work.get("n_states", pd.Series(4, index=work.index))), "colorscale": "Turbo", "showscale": True},
        text=work.get("fold", pd.Series(index=work.index)).astype(str),
    )])
    fig.update_layout(scene={"xaxis_title": "Risk-off probability", "yaxis_title": "Future vol corr proxy", "zaxis_title": "State switch rate"})
    fig = apply_plotly_layout(fig, "HMM Regime 3D — diagnostic only")
    st.plotly_chart(fig, width="stretch")
    source_caption(st, "hmm_out_of_sample_results.csv", "research diagnostic — not used by Growth official allocation")


def render_feature_relationship(st, data: dict[str, pd.DataFrame]) -> None:
    df = data.get("current_features", pd.DataFrame())
    needed = {"raw_target_return_exact", "realized_vol_60d"}
    if df.empty or not needed.issubset(df.columns):
        st.warning("Insufficient data for this surface.")
        source_caption(st, "current_growth_features.csv", "official diagnostic")
        return
    z_col = "final_growth_weight" if "final_growth_weight" in df.columns else "raw_target_rank" if "raw_target_rank" in df.columns else None
    if z_col is None:
        st.warning("Insufficient data for this surface.")
        return
    work = df.dropna(subset=["raw_target_return_exact", "realized_vol_60d", z_col]).copy()
    if work.empty:
        st.warning("Insufficient data for this surface.")
        return
    fig = go.Figure(data=[go.Scatter3d(
        x=numeric(work["raw_target_return_exact"]),
        y=numeric(work["realized_vol_60d"]),
        z=numeric(work[z_col]),
        mode="markers+text",
        text=work.get("ticker", pd.Series(index=work.index)).astype(str),
        marker={"size": 6, "color": numeric(work[z_col]), "colorscale": "Viridis", "showscale": True},
    )])
    fig.update_layout(scene={"xaxis_title": "raw_target_return_exact", "yaxis_title": "60D realized volatility", "zaxis_title": z_col})
    fig = apply_plotly_layout(fig, "Feature Relationship 3D")
    st.plotly_chart(fig, width="stretch")
    source_caption(st, "current_growth_features.csv", "official diagnostic")


def render_black_litterman_surface(st, data: dict[str, pd.DataFrame]) -> None:
    st.warning("Insufficient data for this surface.")
    st.caption("Black-Litterman research data not found. Research diagnostic — not used by Growth official allocation.")


def render_efficient_frontier(st, data: dict[str, pd.DataFrame]) -> None:
    tickers = _current_tickers(data)
    prices = read_price_cache(tickers, lookback=260)
    if prices.empty or len(tickers) < 2:
        st.warning("Insufficient data for this surface.")
        source_caption(st, "yahoo_ohlcv_price_cache", "official diagnostic")
        return
    returns = prices.drop(columns=["date"], errors="ignore").pct_change().dropna()
    if returns.empty:
        st.warning("Insufficient data for this surface.")
        return
    rng = np.random.default_rng(7)
    points = []
    for _ in range(350):
        w = rng.dirichlet(np.ones(returns.shape[1]))
        mu = float(np.dot(returns.mean() * 252, w))
        vol = float(np.sqrt(w @ (returns.cov() * 252).values @ w))
        conc = float((w ** 2).sum())
        points.append((vol, mu, conc))
    arr = np.array(points)
    fig = go.Figure(data=[go.Scatter3d(x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="markers", marker={"size": 3, "color": arr[:, 1] / np.maximum(arr[:, 0], 1e-9), "colorscale": "Viridis", "showscale": True})])
    fig.update_layout(scene={"xaxis_title": "Volatility", "yaxis_title": "Expected return proxy", "zaxis_title": "Concentration HHI"})
    fig = apply_plotly_layout(fig, "Efficient Frontier 3D — diagnostic proxy")
    st.plotly_chart(fig, width="stretch")
    source_caption(st, "yahoo_ohlcv_price_cache random portfolios", "diagnostic proxy — not used by official allocation")


def render_quant_lab(st, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Quant Lab 3D")
    st.caption("Interactive research diagnostics. These surfaces do not alter official Growth allocation.")
    section = st.selectbox(
        "3D surface",
        [
            "Correlation Surface",
            "Covariance Surface",
            "Black-Litterman Surface",
            "Efficient Frontier 3D",
            "Volatility Target Surface",
            "Parameter Stability Surface",
            "HMM Regime 3D",
            "Feature Relationship 3D",
        ],
    )
    if section == "Correlation Surface":
        render_correlation_surface(st, data)
    elif section == "Covariance Surface":
        render_covariance_surface(st, data)
    elif section == "Black-Litterman Surface":
        render_black_litterman_surface(st, data)
    elif section == "Efficient Frontier 3D":
        render_efficient_frontier(st, data)
    elif section == "Volatility Target Surface":
        render_vol_target_surface(st, data)
    elif section == "Parameter Stability Surface":
        render_parameter_surface(st, data)
    elif section == "HMM Regime 3D":
        render_hmm_3d(st, data)
    else:
        render_feature_relationship(st, data)
