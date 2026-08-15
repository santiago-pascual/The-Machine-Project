
from __future__ import annotations

from dataclasses import dataclass

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

from dashboard_components import (
    alert_box,
    fmt_num,
    fmt_pct,
    metric_card,
    source_caption,
)
from dashboard_data_layer import current_holdings, latest, numeric, read_price_cache
from dashboard_theme import (
    AMBER,
    BRIGHT_ORANGE,
    CYAN,
    GREEN,
    ORANGE,
    RED,
    apply_plotly_layout,
)

COLORWAY = [ORANGE, BRIGHT_ORANGE, AMBER, CYAN, GREEN, RED]

@dataclass
class SurfaceResult:
    name: str
    status: str
    source: str
    detail: str


def _chart(st, fig: go.Figure, title: str, height: int = 560) -> None:
    apply_plotly_layout(fig, title=title)
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def _warn(st, title: str, source: str, detail: str) -> SurfaceResult:
    alert_box(st, f"{title} unavailable: {detail}", "warning")
    source_caption(st, source, "diagnostic")
    return SurfaceResult(title, "warning", source, detail)


def current_tickers(data: dict[str, pd.DataFrame], include_cash: bool = False) -> list[str]:
    holdings = current_holdings(data, "Official Forward Paper")
    if holdings.empty or "ticker" not in holdings.columns:
        return []
    tickers = holdings["ticker"].astype(str).str.upper().unique().tolist()
    if not include_cash:
        tickers = [t for t in tickers if t != "CASH"]
    return tickers


@_cache_data(show_spinner=False)
def _cached_price_cache(tickers_key: tuple[str, ...], lookback: int) -> pd.DataFrame:
    return read_price_cache(list(tickers_key), lookback=lookback + 5)


def _price_returns(tickers: list[str], lookback: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = _cached_price_cache(tuple(sorted(tickers)), lookback)
    if prices.empty:
        return prices, pd.DataFrame()
    returns = prices.drop(columns=["date"], errors="ignore").pct_change().tail(lookback).dropna(how="all")
    return prices, returns


def _current_weights(data: dict[str, pd.DataFrame], tickers: list[str]) -> np.ndarray:
    holdings = current_holdings(data, "Official Forward Paper")
    if holdings.empty:
        return np.repeat(1 / max(len(tickers), 1), len(tickers))
    weight_col = "paper_position_weight" if "paper_position_weight" in holdings.columns else "weight" if "weight" in holdings.columns else None
    if weight_col is None:
        return np.repeat(1 / max(len(tickers), 1), len(tickers))
    mapping = holdings.assign(ticker=holdings["ticker"].astype(str).str.upper()).set_index("ticker")[weight_col].to_dict()
    w = np.array([float(pd.to_numeric(pd.Series([mapping.get(t, np.nan)]), errors="coerce").iloc[0]) for t in tickers], dtype=float)
    if np.nansum(w) <= 0:
        return np.repeat(1 / max(len(tickers), 1), len(tickers))
    return np.nan_to_num(w / np.nansum(w))


def render_efficient_frontier(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Efficient Frontier 3D")
    tickers = current_tickers(data)
    sims = int(st.slider("Simulated portfolios", 150, 1500, 450, 50, key="qlab_frontier_sims"))
    include_cash = st.checkbox("Include cash diagnostic point", value=True, key="qlab_frontier_cash")
    if len(tickers) < 2:
        return _warn(st, "Efficient Frontier 3D", "yahoo_ohlcv_price_cache", "need at least two official holdings")
    prices, returns = _price_returns(tickers, 252)
    if returns.empty or returns.shape[1] < 2:
        return _warn(st, "Efficient Frontier 3D", "yahoo_ohlcv_price_cache", "insufficient aligned returns")
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    rng = np.random.default_rng(114)
    rows = []
    for _ in range(sims):
        w = rng.dirichlet(np.ones(len(mu)))
        ret = float(np.dot(mu.values, w))
        vol = float(np.sqrt(max(w @ cov.values @ w, 0)))
        hhi = float(np.square(w).sum())
        rows.append({"volatility": vol, "expected_return_proxy": ret, "hhi": hhi, "type": "feasible diagnostic"})
    ew = np.repeat(1 / len(mu), len(mu))
    cur = _current_weights(data, list(mu.index))
    for label, w in [("equal weight", ew), ("current official weights", cur)]:
        rows.append({"volatility": float(np.sqrt(max(w @ cov.values @ w, 0))), "expected_return_proxy": float(np.dot(mu.values, w)), "hhi": float(np.square(w).sum()), "type": label})
    if include_cash:
        rows.append({"volatility": 0.0, "expected_return_proxy": 0.0, "hhi": 0.0, "type": "cash-heavy reference"})
    df = pd.DataFrame(rows)
    fig = px.scatter_3d(df, x="volatility", y="expected_return_proxy", z="hhi", color="type", color_discrete_sequence=COLORWAY, opacity=0.72)
    _chart(st, fig, "Efficient Frontier 3D — optimizer diagnostic only")
    source_caption(st, "yahoo_ohlcv_price_cache + official holdings", "diagnostic only")
    return SurfaceResult("efficient_frontier", "available", "yahoo_ohlcv_price_cache", f"{len(df)} portfolios")


def render_correlation_surface(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Correlation Surface")
    lookback = int(st.selectbox("Correlation lookback", [20, 60, 126, 252], index=1, key="qlab_corr_lookback"))
    scope = st.selectbox("Universe", ["official holdings only", "top candidates"], index=0, key="qlab_corr_scope")
    absolute = st.checkbox("Absolute correlation", value=False, key="qlab_corr_abs")
    tickers = current_tickers(data)
    if scope == "top candidates":
        cf = latest(data.get("current_features", pd.DataFrame()))
        if not cf.empty and "ticker" in cf.columns:
            ranked = cf.sort_values("raw_target_rank") if "raw_target_rank" in cf.columns else cf
            tickers = ranked["ticker"].astype(str).str.upper().head(12).tolist()
    prices, returns = _price_returns(tickers, lookback)
    if returns.empty or returns.shape[1] < 2:
        return _warn(st, "Correlation Surface", "yahoo_ohlcv_price_cache", "insufficient aligned returns")
    corr = returns.corr().fillna(0)
    if absolute:
        corr = corr.abs()
    fig = go.Figure(data=[go.Surface(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale="Oranges")])
    fig.update_layout(scene={"xaxis_title": "Ticker", "yaxis_title": "Ticker", "zaxis_title": "Correlation"})
    _chart(st, fig, "Correlation Surface")
    hfig = px.imshow(corr, color_continuous_scale="Oranges", aspect="auto", title="2D Correlation Heatmap")
    _chart(st, hfig, "2D Correlation Heatmap", 430)
    vals = corr.where(~np.eye(len(corr), dtype=bool)).stack().dropna()
    cols = st.columns(3)
    with cols[0]: metric_card(st, "Avg Pairwise Corr", fmt_num(vals.mean(), 3))
    with cols[1]: metric_card(st, "Highest Pair", fmt_num(vals.max(), 3))
    with cols[2]: metric_card(st, "Lowest Pair", fmt_num(vals.min(), 3))
    source_caption(st, "yahoo_ohlcv_price_cache", f"official diagnostic · obs {len(returns)}")
    return SurfaceResult("correlation_surface", "available", "yahoo_ohlcv_price_cache", f"lookback={lookback}, obs={len(returns)}")


def render_covariance_surface(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Covariance Surface")
    lookback = int(st.selectbox("Covariance lookback", [20, 60, 126, 252], index=2, key="qlab_cov_lookback"))
    method = st.selectbox("Covariance method", ["historical", "diagonal shrinkage", "Ledoit-Wolf if available"], key="qlab_cov_method")
    annualize = st.checkbox("Annualized", value=True, key="qlab_cov_ann")
    tickers = current_tickers(data)
    _, returns = _price_returns(tickers, lookback)
    if returns.empty or returns.shape[1] < 2:
        return _warn(st, "Covariance Surface", "yahoo_ohlcv_price_cache", "insufficient aligned returns")
    cov = returns.cov().fillna(0)
    if method == "diagonal shrinkage":
        cov = 0.75 * cov + 0.25 * pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
    elif method == "Ledoit-Wolf if available":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.dropna().values)
            cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        except Exception:
            alert_box(st, "Ledoit-Wolf unavailable; showing historical covariance fallback.", "warning")
    if annualize:
        cov = cov * 252
    vals = np.linalg.eigvalsh(cov.values) if cov.shape[0] else np.array([])
    cond = float(np.nanmax(vals) / max(np.nanmin(vals[vals > 1e-12]) if np.any(vals > 1e-12) else np.nan, 1e-12)) if vals.size else np.nan
    eff_rank = float(np.exp(-np.sum((vals / vals.sum()) * np.log(np.maximum(vals / vals.sum(), 1e-12))))) if vals.size and vals.sum() > 0 else np.nan
    fig = go.Figure(data=[go.Surface(z=cov.values, x=list(cov.columns), y=list(cov.index), colorscale="Cividis")])
    fig.update_layout(scene={"xaxis_title": "Ticker", "yaxis_title": "Ticker", "zaxis_title": "Covariance"})
    _chart(st, fig, "Covariance Surface")
    cols = st.columns(3)
    with cols[0]: metric_card(st, "Condition Number", fmt_num(cond, 2))
    with cols[1]: metric_card(st, "Effective Rank", fmt_num(eff_rank, 2))
    with cols[2]: metric_card(st, "Method", method, "Growth official uses equal-weight allocation; covariance shown diagnostic")
    source_caption(st, "yahoo_ohlcv_price_cache", "diagnostic covariance")
    return SurfaceResult("covariance_surface", "available", "yahoo_ohlcv_price_cache", f"method={method}, lookback={lookback}")


def render_volatility_target_surface(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Volatility Target Surface")
    dual_cap = float(st.selectbox("Dual trend cap", [0.60, 0.40, 0.25], index=0, format_func=lambda x: f"{x:.0%}", key="qlab_vol_dual"))
    vol = np.linspace(0.05, 0.90, 70)
    target = np.linspace(0.12, 0.30, 60)
    x, y = np.meshgrid(vol, target)
    unclipped = y / x
    exact = np.minimum(np.minimum(np.maximum(unclipped, 0.40), 0.60), dual_cap)
    fresh = latest(data.get("vol_fresh", pd.DataFrame()))
    cur_vol = cur_target = cur_exposure = np.nan
    if not fresh.empty:
        row = fresh.iloc[-1]
        cur_vol = float(row.get("estimated_portfolio_vol", np.nan))
        cur_target = float(row.get("target_vol", 0.22))
        cur_exposure = float(row.get("final_exposure", np.nan))
    left, right = st.columns(2)
    with left:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=exact, colorscale="Oranges", showscale=True)])
        if np.isfinite(cur_vol) and np.isfinite(cur_exposure):
            fig.add_trace(go.Scatter3d(x=[cur_vol], y=[cur_target], z=[cur_exposure], mode="markers+text", text=["current"], marker={"size": 8, "color": CYAN}))
        fig.update_layout(scene={"xaxis_title": "Estimated portfolio vol", "yaxis_title": "Target vol", "zaxis_title": "Final exposure"})
        _chart(st, fig, "Exact Policy Surface", 520)
    with right:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=unclipped, colorscale="Viridis", showscale=True)])
        fig.update_layout(scene={"xaxis_title": "Estimated portfolio vol", "yaxis_title": "Target vol", "zaxis_title": "Unclipped exposure"})
        _chart(st, fig, "Unclipped Vol Target Surface", 520)
    reason = "floor region" if np.isfinite(cur_exposure) and abs(cur_exposure - 0.40) < 1e-6 else "cap/dual-trend/continuous region"
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Current Est. Vol", fmt_pct(cur_vol))
    with cols[1]: metric_card(st, "Target Vol", fmt_pct(cur_target))
    with cols[2]: metric_card(st, "Final Exposure", fmt_pct(cur_exposure))
    with cols[3]: metric_card(st, "Binding Region", reason)
    source_caption(st, "growth_volatility_targeting_fresh.csv", "official diagnostic")
    return SurfaceResult("volatility_target_surface", "available", "growth_volatility_targeting_fresh.csv", reason)


def render_parameter_stability_surface(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Parameter Stability Surface")
    df = data.get("parameter_stability", pd.DataFrame())
    if df.empty:
        df = data.get("parameter_stability_map", pd.DataFrame())
    if df.empty:
        return _warn(st, "Parameter Stability Surface", "parameter_stability_map.csv", "Phase 92 output missing")
    metric_options = [c for c in ["Sharpe", "sharpe", "CAGR", "cagr", "max_drawdown", "Max DD", "Sortino", "sortino", "Calmar", "calmar", "average_exposure"] if c in df.columns]
    if not metric_options or not {"target_vol", "exposure_cap"}.issubset(df.columns):
        return _warn(st, "Parameter Stability Surface", "parameter_stability_map.csv", "required target_vol/exposure_cap/metric columns missing")
    metric = st.selectbox("Metric", metric_options, key="qlab_param_metric")
    work = df.copy()
    for col, value in [("min_exposure", 0.40), ("vol_lookback_days", 60)]:
        if col in work.columns and numeric(work[col]).notna().any():
            sub = work[np.isclose(numeric(work[col]), value)]
            if not sub.empty:
                work = sub
    pivot = work.pivot_table(index="target_vol", columns="exposure_cap", values=metric, aggfunc="mean").sort_index().sort_index(axis=1)
    if pivot.empty:
        return _warn(st, "Parameter Stability Surface", "parameter_stability_map.csv", "empty pivot")
    fig = go.Figure(data=[go.Surface(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Cividis")])
    try:
        y = pivot.index[np.abs(pivot.index.astype(float) - 0.22).argmin()]
        xval = pivot.columns[np.abs(pivot.columns.astype(float) - 0.60).argmin()]
        fig.add_trace(go.Scatter3d(x=[xval], y=[y], z=[pivot.loc[y, xval]], mode="markers+text", text=["active 22/60"], marker={"size": 8, "color": RED}))
    except Exception:
        pass
    fig.update_layout(scene={"xaxis_title": "Exposure cap", "yaxis_title": "Target volatility", "zaxis_title": metric})
    _chart(st, fig, f"Parameter Stability Surface — {metric}")
    source_caption(st, "parameter_stability_map.csv", "research diagnostic — no retuning")
    return SurfaceResult("parameter_stability_surface", "available", "parameter_stability_map.csv", metric)


def render_black_litterman_lab(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Black–Litterman Lab")
    candidates = ["black_litterman_results", "black_litterman_diagnostics"]
    available = [k for k in candidates if not data.get(k, pd.DataFrame()).empty]
    if not available:
        alert_box(st, "Black–Litterman data not found. Research diagnostic only — not used by Growth official equal-weight allocation.", "warning")
        st.dataframe(pd.DataFrame({"missing_input": ["equilibrium prior", "view confidence grid", "tau grid", "posterior returns/weights"], "required_for": ["posterior return surface", "posterior return surface", "posterior return surface", "posterior weight surface"]}), width="stretch", hide_index=True)
        return SurfaceResult("black_litterman_lab", "warning", "black_litterman_*", "missing BL diagnostic grids")
    return SurfaceResult("black_litterman_lab", "available", ",".join(available), "source detected")


def render_hmm_regime_space(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### HMM Regime Space")
    df = data.get("hmm_oos", pd.DataFrame())
    if df.empty:
        df = data.get("hmm_model_comparison", pd.DataFrame())
    if df.empty:
        return _warn(st, "HMM Regime Space", "hmm_out_of_sample_results.csv", "HMM diagnostics missing")
    cols = [c for c in ["test_risk_off_rate", "future_volatility_corr_proxy", "test_state_switch_rate", "log_likelihood", "bic", "aic"] if c in df.columns]
    if len(cols) < 3:
        return _warn(st, "HMM Regime Space", "hmm_out_of_sample_results.csv", "not enough numeric HMM columns")
    x, y, z = cols[:3]
    color = "n_states" if "n_states" in df.columns else cols[0]
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, color_continuous_scale="Turbo", title="HMM Regime Diagnostic Space")
    _chart(st, fig, "HMM Regime Diagnostic Space")
    alert_box(st, "4-state HMM is diagnostic only and remains redundant with dual trend until separate portfolio evidence passes.", "info")
    source_caption(st, "hmm_out_of_sample_results.csv / hmm_model_comparison.csv", "research diagnostic")
    return SurfaceResult("hmm_regime_space", "available", "hmm diagnostics", f"axes={x},{y},{z}")


def render_feature_space_3d(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Feature Space 3D")
    df = latest(data.get("current_features", pd.DataFrame()))
    if df.empty:
        return _warn(st, "Feature Space 3D", "current_growth_features.csv", "current feature file missing")
    numeric_cols = [c for c in df.columns if numeric(df[c]).notna().sum() >= 5]
    preferred = [c for c in ["raw_target_return_exact", "realized_vol_60d", "rank_percentile", "raw_target_rank", "median_60d_dollar_volume", "market_cap", "return_20d"] if c in numeric_cols]
    options = preferred + [c for c in numeric_cols if c not in preferred]
    if len(options) < 3:
        return _warn(st, "Feature Space 3D", "current_growth_features.csv", "need at least three numeric features")
    x = st.selectbox("X feature", options, index=0, key="qlab_feat_x")
    y = st.selectbox("Y feature", options, index=min(1, len(options)-1), key="qlab_feat_y")
    z = st.selectbox("Z feature", options, index=min(2, len(options)-1), key="qlab_feat_z")
    color_options = [c for c in ["raw_target_selected", "sector", "holding_quality_classification", "passed_tradability_filter", "quality_pass"] if c in df.columns]
    color = st.selectbox("Color by", color_options if color_options else [x], key="qlab_feat_color")
    work = df.dropna(subset=[x, y, z]).copy()
    if work.empty:
        return _warn(st, "Feature Space 3D", "current_growth_features.csv", "selected axes have no rows")
    fig = px.scatter_3d(work, x=x, y=y, z=z, color=color, hover_name="ticker" if "ticker" in work.columns else None, color_discrete_sequence=COLORWAY)
    _chart(st, fig, "Feature Space 3D")
    source_caption(st, "current_growth_features.csv", "official diagnostic")
    return SurfaceResult("feature_space_3d", "available", "current_growth_features.csv", f"axes={x},{y},{z}")


def render_stress_cube(st, data: dict[str, pd.DataFrame]) -> SurfaceResult:
    st.markdown("#### Stress Cube")
    tickers = current_tickers(data)
    holdings = current_holdings(data, "Official Forward Paper")
    if not tickers or holdings.empty:
        return _warn(st, "Stress Cube", "growth_official_paper_state.csv", "official holdings missing")
    beta = 1.0
    if "beta_to_spy" in holdings.columns:
        vals = numeric(holdings["beta_to_spy"]).dropna()
        if not vals.empty:
            beta = float(vals.mean())
    spy = np.linspace(-0.25, 0.15, 50)
    qqq = np.linspace(-0.35, 0.20, 50)
    x, y = np.meshgrid(spy, qqq)
    non_cash = holdings[~holdings.get("ticker", "").astype(str).str.upper().eq("CASH")] if "ticker" in holdings.columns else holdings
    exposure = numeric(non_cash.get("paper_position_weight", pd.Series(dtype=float))).sum()
    z = float(exposure if pd.notna(exposure) else 0.4) * (0.55 * beta * x + 0.45 * beta * y)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="RdYlGn")])
    fig.update_layout(scene={"xaxis_title": "SPY shock", "yaxis_title": "QQQ shock", "zaxis_title": "Estimated portfolio return"})
    _chart(st, fig, "Stress Cube — diagnostic linear beta proxy")
    source_caption(st, "growth_official_paper_state.csv + beta proxy", "diagnostic only")
    return SurfaceResult("stress_cube", "available", "official holdings", f"exposure={exposure}")
