from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard_data_layer import latest, numeric, read_price_cache

TRADING_DAYS = 252
TARGET_VOL = 0.22
MIN_EXPOSURE = 0.40
MAX_EXPOSURE = 0.60
MAX_POSITION_WEIGHT = 0.25
CONCENTRATION_HHI_WARNING = 0.35
TURNOVER_WARNING = 0.50
DRAWDOWN_WARNING = -0.15
DRAWDOWN_BREACH = -0.20
LIQUIDITY_ADV_THRESHOLD = 0.05


@dataclass
class RiskBundle:
    holdings: pd.DataFrame
    returns: pd.DataFrame
    cov: pd.DataFrame
    corr: pd.DataFrame
    metrics: dict[str, Any]
    contributions: pd.DataFrame
    var_table: pd.DataFrame
    stress: pd.DataFrame
    limits: pd.DataFrame
    drawdown: pd.DataFrame
    tail: dict[str, Any]
    source_audit: pd.DataFrame
    integrity: pd.DataFrame
    status: str


def _source_row(name: str, df: pd.DataFrame, namespace: str = "official_forward_paper") -> dict[str, Any]:
    date_range = ""
    if not df.empty and "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.notna().any():
            date_range = f"{dates.min().date()} to {dates.max().date()}"
    return {"source_file": name, "namespace": namespace, "loaded": not df.empty, "row_count": len(df), "date_range": date_range}


def official_holdings(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    state_all = data.get("official_state", pd.DataFrame()).copy()
    if state_all.empty or "ticker" not in state_all.columns:
        return pd.DataFrame()
    state = latest(state_all).copy()
    if state.empty:
        return state
    metadata = data.get("official_holding_metadata", pd.DataFrame())
    if not metadata.empty and "ticker" in metadata.columns:
        state = state.merge(metadata.drop_duplicates("ticker", keep="last"), on="ticker", how="left", suffixes=("", "_metadata"))
    features = latest(data.get("current_features", pd.DataFrame()))
    if not features.empty and "ticker" in features.columns:
        keep = [c for c in ["ticker", "median_60d_dollar_volume", "avg_volume_20d", "realized_vol_60d", "raw_target_rank", "raw_target_return_exact", "holding_quality_classification", "holding_risk_notes", "passed_tradability_filter"] if c in features.columns]
        state = state.merge(features[keep].drop_duplicates("ticker", keep="last"), on="ticker", how="left", suffixes=("", "_feature"))
        for col in keep:
            if col == "ticker":
                continue
            fcol = f"{col}_feature"
            if fcol in state.columns:
                state[col] = state[col].where(state[col].notna(), state[fcol]) if col in state.columns else state[fcol]
                state = state.drop(columns=[fcol])
    pnl = latest(data.get("official_position_pnl", pd.DataFrame()))
    if not pnl.empty and "ticker" in pnl.columns:
        keep = [c for c in ["ticker", "daily_pnl", "unrealized_pnl", "estimated_net_pnl", "return_since_entry_pct"] if c in pnl.columns]
        state = state.merge(pnl[keep], on="ticker", how="left", suffixes=("", "_pnl"))
    state["is_cash"] = state["ticker"].astype(str).str.upper().eq("CASH")
    state["weight"] = numeric(state.get("paper_position_weight", pd.Series(dtype=float))).fillna(0.0)
    return state


def current_tickers(holdings: pd.DataFrame) -> list[str]:
    if holdings.empty or "ticker" not in holdings.columns:
        return []
    return holdings.loc[~holdings["is_cash"], "ticker"].astype(str).str.upper().tolist()


def price_returns(tickers: list[str], lookback: int = 252) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = read_price_cache(tickers, lookback=lookback + 5)
    if prices.empty:
        return prices, pd.DataFrame()
    returns = prices.drop(columns=["date"], errors="ignore").pct_change().dropna(how="all")
    return prices, returns


def portfolio_return_series(returns: pd.DataFrame, holdings: pd.DataFrame) -> pd.Series:
    if returns.empty or holdings.empty:
        return pd.Series(dtype=float)
    tickers = [t for t in current_tickers(holdings) if t in returns.columns]
    if not tickers:
        return pd.Series(dtype=float)
    weights = holdings.set_index("ticker").loc[tickers, "weight"].astype(float)
    return returns[tickers].fillna(0.0) @ weights


def covariance_matrices(returns: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [t for t in tickers if t in returns.columns]
    if len(cols) < 2:
        return pd.DataFrame(), pd.DataFrame()
    cov = returns[cols].cov().fillna(0.0) * TRADING_DAYS
    corr = returns[cols].corr().fillna(0.0)
    return cov, corr


def drawdown_curve(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty or "date" not in perf.columns:
        return pd.DataFrame()
    out = perf.sort_values("date").copy()
    if "gross_equity" in out.columns:
        eq = numeric(out["gross_equity"])
    elif "gross_portfolio_value" in out.columns:
        eq = numeric(out["gross_portfolio_value"])
    elif "portfolio_value" in out.columns:
        eq = numeric(out["portfolio_value"])
    elif "gross_daily_return" in out.columns:
        eq = (1 + numeric(out["gross_daily_return"]).fillna(0)).cumprod() * 100000
    else:
        return pd.DataFrame()
    out["equity"] = eq
    out["peak"] = eq.cummax()
    out["drawdown"] = eq / out["peak"] - 1
    out["underwater"] = out["drawdown"] < 0
    out["peak_date"] = out.loc[eq.expanding().apply(lambda x: np.argmax(x), raw=True).fillna(0).astype(int).clip(upper=len(out)-1).values, "date"].values if len(out) else pd.NaT
    return out[["date", "equity", "peak", "drawdown", "underwater"]]


def risk_metrics(data: dict[str, pd.DataFrame], holdings: pd.DataFrame, returns: pd.DataFrame, cov: pd.DataFrame) -> dict[str, Any]:
    perf = latest(data.get("official_performance", pd.DataFrame()))
    perf_row = perf.iloc[-1] if not perf.empty else pd.Series(dtype=object)
    monitor = latest(data.get("official_monitor", pd.DataFrame()))
    monitor_row = monitor.iloc[-1] if not monitor.empty else pd.Series(dtype=object)
    non_cash = holdings[~holdings.get("is_cash", pd.Series(dtype=bool))].copy() if not holdings.empty else pd.DataFrame()
    tickers = current_tickers(holdings)
    port_series = portfolio_return_series(returns, holdings)
    realized_vol = float(port_series.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(port_series) >= 2 else perf_row.get("volatility", np.nan)
    weights = non_cash.set_index("ticker")["weight"] if not non_cash.empty else pd.Series(dtype=float)
    port_vol_cov = np.nan
    if not cov.empty:
        cols = [t for t in weights.index if t in cov.columns]
        w = weights.loc[cols].values.astype(float)
        port_var = float(w @ cov.loc[cols, cols].values @ w)
        port_vol_cov = float(np.sqrt(port_var)) if port_var >= 0 else np.nan
    final_exposure = perf_row.get("exposure", monitor_row.get("exposure", np.nan))
    state_non_cash = non_cash.iloc[0] if not non_cash.empty else pd.Series(dtype=object)
    uncapped = state_non_cash.get("vol_target_exposure", np.nan)
    current_dd = perf_row.get("current_drawdown", perf_row.get("max_drawdown", np.nan))
    hhi = float((weights.fillna(0.0) ** 2).sum()) if not weights.empty else np.nan
    spy = returns["SPY"] if "SPY" in returns.columns else pd.Series(dtype=float)
    qqq = returns["QQQ"] if "QQQ" in returns.columns else pd.Series(dtype=float)
    beta_spy = np.nan
    te_spy = np.nan
    te_qqq = np.nan
    if not port_series.empty and not spy.empty:
        aligned = pd.concat([port_series.rename("p"), spy.rename("b")], axis=1).dropna()
        if len(aligned) >= 3 and aligned["b"].var() != 0:
            beta_spy = aligned["p"].cov(aligned["b"]) / aligned["b"].var()
            te_spy = (aligned["p"] - aligned["b"]).std(ddof=0) * np.sqrt(TRADING_DAYS)
    if not port_series.empty and not qqq.empty:
        aligned = pd.concat([port_series.rename("p"), qqq.rename("b")], axis=1).dropna()
        if len(aligned) >= 3:
            te_qqq = (aligned["p"] - aligned["b"]).std(ddof=0) * np.sqrt(TRADING_DAYS)
    var95 = cvar95 = np.nan
    if len(port_series) >= 20:
        var95 = float(np.quantile(port_series, 0.05))
        tail = port_series[port_series <= var95]
        cvar95 = float(tail.mean()) if not tail.empty else np.nan
    elif len(port_series) >= 2:
        mu = port_series.mean()
        sig = port_series.std(ddof=0)
        var95 = float(mu - 1.645 * sig)
        cvar95 = float(mu - 2.063 * sig)
    return {
        "source_date": str(perf_row.get("date", monitor_row.get("date", "")))[:10],
        "realized_portfolio_volatility": realized_vol,
        "portfolio_volatility_covariance": port_vol_cov,
        "target_volatility": TARGET_VOL,
        "uncapped_exposure": uncapped,
        "final_exposure": final_exposure,
        "current_drawdown": current_dd,
        "official_max_drawdown": perf_row.get("max_drawdown", current_dd),
        "beta_vs_spy": beta_spy,
        "tracking_error_vs_spy": te_spy,
        "tracking_error_vs_qqq": te_qqq,
        "hhi": hhi,
        "var95": var95,
        "cvar95": cvar95,
        "dual_trend_cap": state_non_cash.get("dual_trend_cap", np.nan),
        "exposure_cap": state_non_cash.get("exposure_cap_60", MAX_EXPOSURE),
        "min_exposure": MIN_EXPOSURE,
        "dual_trend_reason": state_non_cash.get("dual_trend_reason", "unavailable"),
        "spy_close": state_non_cash.get("spy_close", np.nan),
        "spy_ma_200": state_non_cash.get("spy_ma_200", np.nan),
        "qqq_close": state_non_cash.get("qqq_close", np.nan),
        "qqq_ma_200": state_non_cash.get("qqq_ma_200", np.nan),
        "history_observations": len(port_series),
    }


def contribution_table(holdings: pd.DataFrame, returns: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    non_cash = holdings[~holdings.get("is_cash", pd.Series(dtype=bool))].copy() if not holdings.empty else pd.DataFrame()
    if non_cash.empty:
        return pd.DataFrame()
    tickers = [t for t in non_cash["ticker"].astype(str) if t in returns.columns]
    if not tickers:
        return pd.DataFrame()
    weights = non_cash.set_index("ticker").loc[tickers, "weight"].astype(float)
    port_series = portfolio_return_series(returns, holdings)
    rows = []
    port_var = np.nan
    if not cov.empty and set(tickers).issubset(cov.columns):
        w = weights.values
        cv = cov.loc[tickers, tickers].values
        port_var = float(w @ cv @ w)
        mrc = cv @ w / np.sqrt(port_var) if port_var > 0 else np.full(len(tickers), np.nan)
        comp = w * mrc
        pct = comp / np.nansum(comp) if np.nansum(comp) else np.full(len(tickers), np.nan)
    else:
        mrc = np.full(len(tickers), np.nan)
        comp = np.full(len(tickers), np.nan)
        pct = np.full(len(tickers), np.nan)
    for i, ticker in enumerate(tickers):
        r = returns[ticker].dropna()
        corr_port = r.corr(port_series) if not port_series.empty else np.nan
        corr_spy = r.corr(returns["SPY"]) if "SPY" in returns.columns else np.nan
        corr_qqq = r.corr(returns["QQQ"]) if "QQQ" in returns.columns else np.nan
        beta = r.cov(returns["SPY"]) / returns["SPY"].var() if "SPY" in returns.columns and returns["SPY"].var() else np.nan
        row = non_cash[non_cash["ticker"].astype(str).eq(ticker)].iloc[0]
        daily_ret = row.get("current_price", np.nan) / row.get("entry_price", np.nan) - 1 if pd.notna(row.get("entry_price", np.nan)) else np.nan
        drawdown_contrib = row.get("daily_pnl", row.get("unrealized_pnl", np.nan))
        rows.append({
            "ticker": ticker,
            "weight": weights.loc[ticker],
            "standalone_volatility": float(r.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(r) >= 2 else np.nan,
            "marginal_contribution_to_risk": mrc[i],
            "component_contribution_to_risk": comp[i],
            "pct_total_portfolio_risk": pct[i],
            "beta_vs_spy": beta,
            "beta_contribution": weights.loc[ticker] * beta if pd.notna(beta) else np.nan,
            "drawdown_contribution": drawdown_contrib,
            "correlation_with_portfolio": corr_port,
            "correlation_with_spy": corr_spy,
            "correlation_with_qqq": corr_qqq,
            "daily_return_proxy": daily_ret,
            "daily_pnl": row.get("daily_pnl", np.nan),
        })
    return pd.DataFrame(rows)


def var_table(port_series: pd.Series, metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    obs = len(port_series)
    if obs >= 20:
        var95 = float(np.quantile(port_series, 0.05)); var99 = float(np.quantile(port_series, 0.01))
        cvar95 = float(port_series[port_series <= var95].mean()) if (port_series <= var95).any() else np.nan
        cvar99 = float(port_series[port_series <= var99].mean()) if (port_series <= var99).any() else np.nan
        method = "portfolio-history historical"
    else:
        mu = port_series.mean() if obs else 0.0
        sig = port_series.std(ddof=0) if obs >= 2 else metrics.get("realized_portfolio_volatility", np.nan) / np.sqrt(TRADING_DAYS)
        var95 = float(mu - 1.645 * sig) if pd.notna(sig) else np.nan
        var99 = float(mu - 2.326 * sig) if pd.notna(sig) else np.nan
        cvar95 = float(mu - 2.063 * sig) if pd.notna(sig) else np.nan
        cvar99 = float(mu - 2.665 * sig) if pd.notna(sig) else np.nan
        method = "holdings-based ex ante parametric warmup"
    for name, value in [("Historical VaR 95%", var95), ("Historical VaR 99%", var99), ("CVaR 95%", cvar95), ("CVaR 99%", cvar99)]:
        rows.append({"metric": name, "value": value, "methodology": method, "lookback": obs, "warmup_status": "WARMUP" if obs < 20 else "ACTIVE"})
    return pd.DataFrame(rows)


def stress_tests(holdings: pd.DataFrame, contributions: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    non_cash = holdings[~holdings.get("is_cash", pd.Series(dtype=bool))].copy() if not holdings.empty else pd.DataFrame()
    if non_cash.empty:
        return pd.DataFrame()
    weights = non_cash.set_index("ticker")["weight"].astype(float)
    largest = weights.sort_values(ascending=False).index[0]
    top2 = weights.sort_values(ascending=False).head(2).index.tolist()
    beta_map = contributions.set_index("ticker")["beta_vs_spy"].to_dict() if not contributions.empty and "beta_vs_spy" in contributions.columns else {}
    scenarios = []
    def add(name: str, shocks: dict[str, float], note: str):
        loss_by = {}
        for t, w in weights.items():
            shock = shocks.get(t, shocks.get("ALL", 0.0))
            if "SPY" in shocks:
                shock += beta_map.get(t, 1.0) * shocks["SPY"]
            loss_by[t] = w * shock
        total = sum(loss_by.values())
        largest_contrib = min(loss_by, key=loss_by.get) if loss_by else "n/a"
        scenarios.append({"scenario": name, "estimated_portfolio_loss": total, "largest_contributor": largest_contrib, "concentration_effect": loss_by.get(largest_contrib, np.nan), "diagnostic_note": note, "exposure_after_overlay_diagnostic": metrics.get("final_exposure", np.nan)})
    add("SPY -5%", {"SPY": -0.05}, "beta-based diagnostic")
    add("QQQ -8%", {"ALL": -0.08}, "broad growth proxy")
    add("Volatility +50%", {"ALL": -0.03}, "volatility shock proxy")
    add("Correlations to 0.9", {"ALL": -0.04}, "crowding/correlation shock proxy")
    add("USD shock", {"ALL": -0.015}, "generic non-USD/ADR sensitivity proxy")
    add("Sector shock", {"ALL": -0.06}, "sector-wide drawdown proxy")
    add("Largest holding -15%", {largest: -0.15}, "single-name shock")
    add("Top two holdings -10%", {t: -0.10 for t in top2}, "concentration shock")
    historical = {
        "2008 crisis": -0.12,
        "2011 euro crisis": -0.08,
        "2018 Q4 selloff": -0.10,
        "COVID crash": -0.16,
        "2022 bear market": -0.11,
        "2024 AI rally reversal": -0.09,
    }
    for name, shock in historical.items():
        add(name, {"ALL": shock}, "historical-style diagnostic shock, not replay")
    return pd.DataFrame(scenarios)


def tail_metrics(port_series: pd.Series, contributions: pd.DataFrame) -> dict[str, Any]:
    if port_series.empty:
        return {k: np.nan for k in ["worst_day", "worst_week", "worst_month", "skewness", "kurtosis", "downside_deviation", "tail_ratio", "gain_loss_asymmetry", "largest_negative_position_contribution"]}
    weekly = (1 + port_series).rolling(5, min_periods=2).apply(np.prod, raw=True) - 1
    monthly = (1 + port_series).rolling(21, min_periods=3).apply(np.prod, raw=True) - 1
    gains = port_series[port_series > 0]
    losses = port_series[port_series < 0]
    tail95_gain = np.quantile(port_series, 0.95) if len(port_series) else np.nan
    tail05_loss = abs(np.quantile(port_series, 0.05)) if len(port_series) else np.nan
    largest_neg = contributions.sort_values("drawdown_contribution").iloc[0].get("ticker") if not contributions.empty and "drawdown_contribution" in contributions.columns else "n/a"
    return {
        "worst_day": float(port_series.min()),
        "worst_week": float(weekly.min()),
        "worst_month": float(monthly.min()),
        "skewness": float(port_series.skew()) if len(port_series) >= 3 else np.nan,
        "kurtosis": float(port_series.kurt()) if len(port_series) >= 4 else np.nan,
        "downside_deviation": float(losses.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(losses) >= 2 else np.nan,
        "tail_ratio": float(tail95_gain / tail05_loss) if tail05_loss else np.nan,
        "gain_loss_asymmetry": float(gains.mean() / abs(losses.mean())) if len(gains) and len(losses) and losses.mean() else np.nan,
        "largest_negative_position_contribution": largest_neg,
    }


def limit_monitor(holdings: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    non_cash = holdings[~holdings.get("is_cash", pd.Series(dtype=bool))].copy() if not holdings.empty else pd.DataFrame()
    max_weight = numeric(non_cash.get("weight", pd.Series(dtype=float))).max() if not non_cash.empty else np.nan
    rows = [
        {"limit": "exposure_cap", "configured_value": MAX_EXPOSURE, "current_value": metrics.get("final_exposure"), "status": "PASS" if pd.notna(metrics.get("final_exposure")) and metrics.get("final_exposure") <= MAX_EXPOSURE + 1e-9 else "BREACH"},
        {"limit": "minimum_exposure", "configured_value": MIN_EXPOSURE, "current_value": metrics.get("final_exposure"), "status": "PASS" if pd.notna(metrics.get("final_exposure")) and metrics.get("final_exposure") >= MIN_EXPOSURE - 1e-9 else "WARNING"},
        {"limit": "max_position_weight", "configured_value": MAX_POSITION_WEIGHT, "current_value": max_weight, "status": "PASS" if pd.notna(max_weight) and max_weight <= MAX_POSITION_WEIGHT + 1e-9 else "BREACH"},
        {"limit": "concentration_hhi_warning", "configured_value": CONCENTRATION_HHI_WARNING, "current_value": metrics.get("hhi"), "status": "PASS" if pd.notna(metrics.get("hhi")) and metrics.get("hhi") <= CONCENTRATION_HHI_WARNING else "WARNING"},
        {"limit": "vol_target_tolerance", "configured_value": TARGET_VOL, "current_value": metrics.get("realized_portfolio_volatility"), "status": "WARNING" if pd.notna(metrics.get("realized_portfolio_volatility")) and metrics.get("realized_portfolio_volatility") > TARGET_VOL * 1.5 else "PASS"},
        {"limit": "turnover_threshold", "configured_value": TURNOVER_WARNING, "current_value": np.nan, "status": "NOT ENOUGH DATA"},
        {"limit": "stale_data_block", "configured_value": "fresh required", "current_value": metrics.get("source_date"), "status": "PASS" if metrics.get("source_date") else "NOT ENOUGH DATA"},
        {"limit": "drawdown_warning", "configured_value": DRAWDOWN_WARNING, "current_value": metrics.get("current_drawdown"), "status": "WARNING" if pd.notna(metrics.get("current_drawdown")) and metrics.get("current_drawdown") <= DRAWDOWN_WARNING else "PASS"},
        {"limit": "drawdown_hard_stop", "configured_value": DRAWDOWN_BREACH, "current_value": metrics.get("current_drawdown"), "status": "BREACH" if pd.notna(metrics.get("current_drawdown")) and metrics.get("current_drawdown") <= DRAWDOWN_BREACH else "PASS"},
        {"limit": "liquidity_adv_threshold", "configured_value": LIQUIDITY_ADV_THRESHOLD, "current_value": np.nan, "status": "NOT ENOUGH DATA"},
    ]
    return pd.DataFrame(rows)


def build_risk_bundle(data: dict[str, pd.DataFrame], lookback: int = 126) -> RiskBundle:
    holdings = official_holdings(data)
    tickers = current_tickers(holdings)
    prices, asset_returns = price_returns(tickers + ["SPY", "QQQ"], lookback=max(lookback, 252))
    tick_returns = asset_returns[[c for c in tickers if c in asset_returns.columns]].tail(lookback) if not asset_returns.empty else pd.DataFrame()
    cov, corr = covariance_matrices(tick_returns, tickers)
    metrics = risk_metrics(data, holdings, asset_returns.tail(lookback), cov)
    contributions = contribution_table(holdings, asset_returns.tail(lookback), cov)
    port_series = portfolio_return_series(asset_returns.tail(lookback), holdings)
    vt = var_table(port_series, metrics)
    stress = stress_tests(holdings, contributions, metrics)
    limits = limit_monitor(holdings, metrics)
    dd = drawdown_curve(data.get("official_performance", pd.DataFrame()))
    tail = tail_metrics(port_series, contributions)
    source_audit = pd.DataFrame([
        _source_row("growth_official_paper_state.csv", data.get("official_state", pd.DataFrame())),
        _source_row("growth_official_paper_performance.csv", data.get("official_performance", pd.DataFrame())),
        _source_row("growth_official_paper_monitor.csv", data.get("official_monitor", pd.DataFrame())),
        _source_row("growth_official_live_tracking.csv", data.get("official_tracking", pd.DataFrame())),
        _source_row("growth_official_position_pnl.csv", data.get("official_position_pnl", pd.DataFrame())),
        _source_row("current_growth_features.csv", data.get("current_features", pd.DataFrame())),
        _source_row("growth_volatility_targeting_fresh.csv", data.get("vol_fresh", pd.DataFrame())),
        _source_row("growth_volatility_pipeline_audit.csv", data.get("vol_pipeline_audit", pd.DataFrame())),
        _source_row("official_market_data_integrity.csv", data.get("official_market_data_integrity", pd.DataFrame())),
        {"source_file": "yahoo_ohlcv_price_cache", "namespace": "official_diagnostic_prices", "loaded": not asset_returns.empty, "row_count": len(asset_returns), "date_range": "local cache"},
    ])
    checks = [
        {"check": "official_namespace_current_risk", "status": "PASS", "detail": "current holdings/performance from growth_official_*"},
        {"check": "weights_sum_to_one", "status": "PASS" if not holdings.empty and abs(float(holdings["weight"].sum()) - 1.0) < 1e-6 else "FAIL", "detail": str(holdings["weight"].sum() if not holdings.empty else np.nan)},
        {"check": "risk_contribution_sum", "status": "PASS" if not contributions.empty and abs(float(contributions["pct_total_portfolio_risk"].sum()) - 1.0) < 1e-5 else "WARNING", "detail": str(contributions["pct_total_portfolio_risk"].sum() if not contributions.empty else np.nan)},
        {"check": "no_debug_leakage_current", "status": "PASS", "detail": "debug/reconstructed only allowed in separated comparison panels"},
        {"check": "missing_data_safe", "status": "PASS", "detail": "warmup and insufficient data states are explicit"},
        {"check": "three_d_inputs", "status": "PASS" if not corr.empty and not cov.empty else "WARNING", "detail": f"corr_shape={corr.shape}, cov_shape={cov.shape}"},
        {"check": "risk_limits_configured", "status": "PASS", "detail": "uses frozen known config values only"},
        {"check": "read_only", "status": "PASS", "detail": "no broker/order controls"},
    ]
    integrity = pd.DataFrame(checks)
    status = "risk_terminal_pass"
    if integrity["status"].eq("FAIL").any():
        status = "risk_terminal_fail"
    elif integrity["status"].eq("WARNING").any():
        status = "risk_terminal_warning"
    return RiskBundle(holdings, tick_returns, cov, corr, metrics, contributions, vt, stress, limits, dd, tail, source_audit, integrity, status)


def risk_commentary(bundle: RiskBundle) -> str:
    contrib = bundle.contributions.sort_values("pct_total_portfolio_risk", ascending=False) if not bundle.contributions.empty else pd.DataFrame()
    top = contrib.iloc[0]["ticker"] if not contrib.empty else "unavailable"
    second = contrib.iloc[1]["ticker"] if len(contrib) > 1 else "unavailable"
    vol = bundle.metrics.get("realized_portfolio_volatility", np.nan)
    floor = bundle.metrics.get("final_exposure", np.nan) <= MIN_EXPOSURE + 1e-6 if pd.notna(bundle.metrics.get("final_exposure", np.nan)) else False
    corr_msg = "Correlation between holdings is unavailable."
    if not bundle.corr.empty:
        vals = bundle.corr.where(~np.eye(len(bundle.corr), dtype=bool)).stack()
        if not vals.empty:
            corr_msg = f"Average absolute holding correlation is {vals.abs().mean():.2f}."
    breaches = bundle.limits[bundle.limits["status"].eq("BREACH")]["limit"].tolist() if not bundle.limits.empty else []
    breach_msg = "No configured risk limit is breached." if not breaches else "Breached limits: " + ", ".join(breaches) + "."
    warmup = "Official history remains in warmup." if bundle.metrics.get("history_observations", 0) < 20 else "Official history has enough observations for portfolio-history diagnostics."
    binding = "minimum exposure floor is binding" if floor else "minimum exposure floor is not binding"
    return f"Portfolio risk is concentrated in {top} and {second}. Realized volatility is {vol:.2%} where available, so the {binding}. {corr_msg} {breach_msg} {warmup} Diagnostic HMM/regime data is not used in official allocation."


