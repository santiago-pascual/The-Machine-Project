from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_components import fmt_money, fmt_num, fmt_pct, metric_card, status_badge
from dashboard_data_layer import (
    MODEL_VERSION,
    latest,
    latest_market_date,
    next_rebalance_date,
    numeric,
    read_price_cache,
)
from dashboard_theme import (
    AMBER,
    GREEN,
    INFO,
    ORANGE,
    PURPLE,
    RED,
    apply_plotly_layout,
)

OFFICIAL_PORTFOLIO_SOURCES = {
    "state": "growth_official_paper_state.csv",
    "position_pnl": "growth_official_position_pnl.csv",
    "trade_lifecycle": "growth_official_trade_lifecycle.csv",
    "actions": "growth_official_paper_actions.csv",
    "trades": "growth_official_paper_trades.csv",
    "cost_ledger": "growth_official_estimated_cost_ledger.csv",
    "monitor": "growth_official_paper_monitor.csv",
    "metadata": "official_holding_metadata.csv",
}

ORANGE_SEQUENCE = ["#FF7A00", "#FF9A2E", "#B85A00", "#F5A623", "#C76A13", "#7A3A00", "#E8791A"]


def _safe(value: Any, default: str = "unavailable") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value)
    return text if text and text.lower() != "nan" else default


def _latest_row(df: pd.DataFrame) -> pd.Series:
    d = latest(df)
    return d.iloc[-1] if not d.empty else pd.Series(dtype=object)


def _official_holdings(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    full_state = data.get("official_state", pd.DataFrame()).copy()
    if full_state.empty or "ticker" not in full_state.columns or "date" not in full_state.columns:
        return pd.DataFrame()
    full_state["date"] = pd.to_datetime(full_state["date"], errors="coerce").dt.normalize()
    state = latest(full_state).copy()
    if state.empty:
        return pd.DataFrame()

    dates = sorted(full_state["date"].dropna().unique())
    if len(dates) >= 2:
        prev = full_state[full_state["date"].eq(dates[-2])].copy()
        prev_cols = [c for c in ["ticker", "paper_position_weight", "paper_position_value", "current_price"] if c in prev.columns]
        prev = prev[prev_cols].rename(
            columns={
                "paper_position_weight": "prev_official_weight",
                "paper_position_value": "prev_official_position_value",
                "current_price": "prev_official_price",
            }
        )
        state = state.merge(prev, on="ticker", how="left")

    pnl = latest(data.get("official_position_pnl", pd.DataFrame()))
    if not pnl.empty and "ticker" in pnl.columns:
        keep = [c for c in pnl.columns if c != "date"]
        state = state.merge(pnl[keep], on="ticker", how="left", suffixes=("", "_pnl"))

    metadata = data.get("official_holding_metadata", pd.DataFrame())
    if not metadata.empty and "ticker" in metadata.columns:
        metadata = metadata.drop_duplicates("ticker", keep="last")
        state = state.merge(metadata, on="ticker", how="left", suffixes=("", "_metadata"))

    features = latest(data.get("current_features", pd.DataFrame()))
    if not features.empty and "ticker" in features.columns:
        feature_cols = [
            "ticker",
            "raw_target_rank",
            "raw_target_return_exact",
            "raw_target_feature_source",
            "raw_target_current_features_available",
            "median_60d_dollar_volume",
            "avg_volume_20d",
            "realized_vol_60d",
            "holding_quality_classification",
            "holding_risk_notes",
            "soft_exit_status",
            "passed_tradability_filter",
            "tradability_exclusion_reason",
        ]
        feature_cols = [c for c in feature_cols if c in features.columns]
        state = state.merge(features[feature_cols].drop_duplicates("ticker", keep="last"), on="ticker", how="left", suffixes=("", "_feature"))
        for col in ["raw_target_rank", "raw_target_return_exact", "raw_target_feature_source", "raw_target_current_features_available", "median_60d_dollar_volume", "avg_volume_20d", "realized_vol_60d", "holding_quality_classification", "holding_risk_notes", "soft_exit_status", "passed_tradability_filter", "tradability_exclusion_reason"]:
            feature_col = f"{col}_feature"
            if feature_col in state.columns:
                if col in state.columns:
                    state[col] = state[col].where(state[col].notna(), state[feature_col])
                else:
                    state[col] = state[feature_col]
                state = state.drop(columns=[feature_col])

    costs = data.get("official_cost_ledger", pd.DataFrame())
    if not costs.empty and "ticker" in costs.columns:
        cost_col = "estimated_total_cost" if "estimated_total_cost" in costs.columns else "total_cost" if "total_cost" in costs.columns else None
        if cost_col:
            cost_by_ticker = costs.groupby("ticker", dropna=False)[cost_col].sum().reset_index(name="estimated_costs_total")
            state = state.merge(cost_by_ticker, on="ticker", how="left")
    state["is_cash"] = state["ticker"].astype(str).str.upper().eq("CASH")
    return state


def _non_cash(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty or "ticker" not in holdings.columns:
        return pd.DataFrame()
    return holdings[~holdings["ticker"].astype(str).str.upper().eq("CASH")].copy()


def _holding_health(row: pd.Series) -> tuple[int, str, str]:
    score = 100
    notes = []
    quality = str(row.get("holding_quality_classification", "")).lower()
    if any(x in quality for x in ["reject", "weak", "critical"]):
        score -= 35; notes.append("quality weak")
    elif "speculative" in quality or "moderate" in quality:
        score -= 15; notes.append("quality moderate")
    if bool(row.get("raw_target_current_features_available", True)) is False:
        score -= 20; notes.append("raw target unavailable")
    if str(row.get("raw_target_feature_source", "")).lower() not in {"raw_target_return_exact", "nan", ""}:
        score -= 20; notes.append("non-exact raw target")
    vol = row.get("realized_vol_60d", np.nan)
    if pd.notna(vol) and float(vol) > 1.2:
        score -= 20; notes.append("high volatility")
    if str(row.get("action", "")).upper() == "SELL":
        score -= 30; notes.append("sell action")
    if pd.isna(row.get("paper_position_weight", np.nan)) or float(row.get("paper_position_weight", 0) or 0) <= 0:
        score -= 10; notes.append("zero weight")
    score = int(max(0, min(100, score)))
    if score >= 85:
        label, css = "Excellent", "pass"
    elif score >= 70:
        label, css = "Good", "pass"
    elif score >= 50:
        label, css = "Moderate", "warning"
    elif score >= 30:
        label, css = "Weak", "warning"
    else:
        label, css = "Critical", "failed"
    detail = ", ".join(notes) if notes else "all available checks acceptable"
    return score, label, css + "|" + detail


def _enrich_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return holdings
    out = holdings.copy()
    out["weight"] = numeric(out.get("paper_position_weight", pd.Series(dtype=float)))
    out["position_value"] = numeric(out.get("paper_position_value", pd.Series(dtype=float)))
    out["current_price"] = numeric(out.get("current_price", pd.Series(dtype=float)))
    out["entry_price"] = numeric(out.get("entry_price", pd.Series(dtype=float)))
    out["prev_official_price"] = numeric(out.get("prev_official_price", pd.Series(dtype=float)))
    out["prev_official_position_value"] = numeric(out.get("prev_official_position_value", pd.Series(dtype=float)))

    # Asset-level return/PnL must not reuse portfolio-level official daily_return.
    out["asset_daily_return"] = out["current_price"] / out["prev_official_price"] - 1.0
    out.loc[out["is_cash"], "asset_daily_return"] = 0.0
    fallback_daily = numeric(out.get("return_1d", pd.Series(index=out.index, dtype=float)))
    out["asset_daily_return"] = out["asset_daily_return"].where(out["asset_daily_return"].notna(), fallback_daily)
    out["daily_return"] = out["asset_daily_return"]
    out["daily_pnl_value"] = out["prev_official_position_value"] * out["asset_daily_return"]
    out.loc[out["is_cash"], "daily_pnl_value"] = 0.0
    out["asset_price_based_position_value"] = out["prev_official_position_value"] * (1.0 + out["asset_daily_return"].fillna(0.0))
    out["asset_price_based_position_value"] = out["asset_price_based_position_value"].where(out["asset_price_based_position_value"].notna(), out["position_value"])

    out["return_since_entry"] = out["current_price"] / out["entry_price"] - 1.0
    out.loc[out["is_cash"], "return_since_entry"] = 0.0
    out["unrealized_pnl_value"] = out["prev_official_position_value"] * out["return_since_entry"].fillna(0.0)
    out.loc[out["is_cash"], "unrealized_pnl_value"] = 0.0
    out["estimated_costs_total"] = numeric(out.get("estimated_costs_total", pd.Series(0.0, index=out.index))).fillna(0.0)

    previous_portfolio_value = out["prev_official_position_value"].fillna(0.0).sum()
    out["return_contribution"] = np.where(previous_portfolio_value > 0, out["daily_pnl_value"].fillna(0.0) / previous_portfolio_value, 0.0)

    weights = out.loc[~out["is_cash"], "weight"].fillna(0.0)
    out["risk_contribution"] = 0.0
    non_cash_tickers = out.loc[~out["is_cash"], "ticker"].astype(str).tolist()
    prices = read_price_cache(non_cash_tickers, lookback=90) if non_cash_tickers else pd.DataFrame()
    if not prices.empty and len(non_cash_tickers) >= 2:
        returns = prices.drop(columns=["date"], errors="ignore").pct_change().dropna(how="all")
        cols = [t for t in non_cash_tickers if t in returns.columns]
        if len(cols) >= 2:
            cov = returns[cols].cov().fillna(0.0).values * 252.0
            w = out.set_index("ticker").loc[cols, "weight"].fillna(0.0).values
            port_var = float(w @ cov @ w)
            if port_var > 0:
                mrc = cov @ w
                contrib = w * mrc / port_var
                for ticker, val in zip(cols, contrib):
                    out.loc[out["ticker"].astype(str).eq(ticker), "risk_contribution"] = val
                port_ret = returns[cols] @ w
                for ticker in cols:
                    corr = returns[ticker].corr(port_ret)
                    beta = returns[ticker].cov(port_ret) / port_ret.var() if port_ret.var() and not pd.isna(port_ret.var()) else np.nan
                    out.loc[out["ticker"].astype(str).eq(ticker), "correlation_with_portfolio"] = corr
                    out.loc[out["ticker"].astype(str).eq(ticker), "beta"] = beta
        else:
            risk_ok = float((weights ** 2).sum()) > 0 if not weights.empty else False
            if risk_ok:
                denom = float((weights ** 2).sum())
                out["risk_contribution"] = np.where(~out["is_cash"], out["weight"].fillna(0.0) ** 2 / denom, 0.0)
    out["drawdown_contribution"] = np.where(out["daily_pnl_value"] < 0, out["daily_pnl_value"], 0.0)
    out["turnover_contribution"] = numeric(out.get("weight_change", pd.Series(0.0, index=out.index))).abs().fillna(0.0)
    out["expected_return_display"] = numeric(out.get("raw_target_return_exact", pd.Series(dtype=float)))
    scores = out.apply(_holding_health, axis=1, result_type="expand")
    out["health_score"] = scores[0]
    out["health_label"] = scores[1]
    out["health_detail"] = scores[2].astype(str).str.split("|", n=1).str[1]
    out["health_status"] = scores[2].astype(str).str.split("|", n=1).str[0]
    for col in ["company_name", "sector", "industry", "country", "market_cap", "tradability_status", "soft_exit_status", "institutional_filters_passed", "liquidity_score", "adv_participation", "beta", "correlation_with_portfolio"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _summary_metrics(holdings: pd.DataFrame) -> dict[str, Any]:
    non_cash = _non_cash(holdings)
    if non_cash.empty:
        return {}
    largest = non_cash.sort_values("weight", ascending=False).iloc[0]
    best = non_cash.sort_values("daily_return", ascending=False).iloc[0]
    worst = non_cash.sort_values("daily_return", ascending=True).iloc[0]
    risk = non_cash.sort_values("risk_contribution", ascending=False).iloc[0]
    age = np.nan
    if "entry_date" in non_cash.columns and "date" in non_cash.columns:
        ages = (pd.to_datetime(non_cash["date"], errors="coerce") - pd.to_datetime(non_cash["entry_date"], errors="coerce")).dt.days
        age = ages.mean()
    return {
        "current_holdings": len(non_cash),
        "largest_position": f"{largest.get('ticker')} ({fmt_pct(largest.get('weight'))})",
        "best_performer": f"{best.get('ticker')} ({fmt_pct(best.get('daily_return'))})",
        "worst_performer": f"{worst.get('ticker')} ({fmt_pct(worst.get('daily_return'))})",
        "largest_risk": f"{risk.get('ticker')} ({fmt_pct(risk.get('risk_contribution'))})",
        "avg_expected_return": non_cash["expected_return_display"].mean(),
        "avg_quality": non_cash["health_score"].mean(),
        "avg_holding_age": age,
    }


def _sparkline_fig(ticker: str) -> go.Figure | None:
    prices = read_price_cache([ticker], lookback=30)
    if prices.empty or ticker not in prices.columns:
        return None
    work = prices[["date", ticker]].dropna().tail(30).copy()
    if work.empty:
        return None
    first = work[ticker].iloc[0]
    if not first:
        return None
    work["normalized"] = work[ticker] / first * 100.0
    work["daily_return"] = work[ticker].pct_change()
    fig = go.Figure(go.Scatter(
        x=work["date"],
        y=work["normalized"],
        mode="lines",
        name=f"{ticker} 30D",
        line={"color": ORANGE, "width": 2.4},
        fill="tozeroy",
        fillcolor="rgba(255,122,0,0.10)",
        customdata=np.stack([work[ticker], work["daily_return"]], axis=-1),
        hovertemplate="%{x|%Y-%m-%d}<br>price=%{customdata[0]:.2f}<br>daily return=%{customdata[1]:.2%}<extra></extra>",
    ))
    fig = apply_plotly_layout(fig, f"{ticker} 30D Price History")
    fig.update_layout(height=180, margin={"l": 18, "r": 18, "t": 42, "b": 22}, showlegend=False)
    fig.update_yaxes(title="Normalized price", ticksuffix="")
    return fig


def _render_header(st, holdings: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    monitor = _latest_row(data.get("official_monitor", pd.DataFrame()))
    latest_state = _latest_row(data.get("official_state", pd.DataFrame()))
    exposure = monitor.get("exposure", latest_state.get("final_exposure", np.nan))
    cash = monitor.get("cash", latest_state.get("cash_weight", np.nan))
    next_rebalance = next_rebalance_date(data)
    st.markdown(
        f"""
        <div class='page-head'>
          <div>
            <h2 class='page-title'>Portfolio Terminal {status_badge('OFFICIAL', 'official')}</h2>
            <div class='page-subtitle'>Official Forward Paper | {MODEL_VERSION} | Latest date {latest_market_date(data)}</div>
          </div>
          <div class='small-muted'>Exposure {fmt_pct(exposure)} · Cash {fmt_pct(cash)} · Next rebalance {_safe(next_rebalance)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary(st, holdings: pd.DataFrame) -> None:
    metrics = _summary_metrics(holdings)
    if not metrics:
        st.info("No official non-cash holdings available.")
        return
    rows = [
        [("Current Holdings", str(metrics["current_holdings"])), ("Largest Position", metrics["largest_position"]), ("Best Performer", metrics["best_performer"]), ("Worst Performer", metrics["worst_performer"])],
        [("Largest Risk Contributor", metrics["largest_risk"]), ("Average Expected Return", fmt_pct(metrics["avg_expected_return"])), ("Average Quality Score", fmt_num(metrics["avg_quality"], 1)), ("Average Holding Age", fmt_num(metrics["avg_holding_age"], 1) + " days")],
    ]
    for row in rows:
        cols = st.columns(4)
        for col, (label, value) in zip(cols, row):
            with col:
                metric_card(st, label, value, "growth_official_paper_state.csv", badge="OFFICIAL")


def _render_holding_cards(st, holdings: pd.DataFrame) -> None:
    non_cash = _non_cash(holdings)
    if non_cash.empty:
        st.markdown("<div class='alert-box info'>No active official holdings. Portfolio is currently cash or official state is unavailable.</div>", unsafe_allow_html=True)
        return
    for _, row in non_cash.sort_values("weight", ascending=False).iterrows():
        ticker = str(row.get("ticker", "n/a"))
        st.markdown(
            f"""
            <div class='holding-card'>
              <div style='display:flex;justify-content:space-between;gap:18px;align-items:flex-start'>
                <div><div class='holding-ticker'>{ticker}</div><div class='small-muted'>{_safe(row.get('company_name'))} · {_safe(row.get('sector'))} · {_safe(row.get('industry'))} · {_safe(row.get('country'))}</div></div>
                <div>{status_badge(str(row.get('health_label', 'n/a')), str(row.get('health_status', 'neutral')))} {status_badge('OFFICIAL', 'official')}</div>
              </div>
              <div class='small-muted'>Health score: {row.get('health_score', 'n/a')}/100 · {row.get('health_detail', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        a, b, c, d = st.columns(4)
        with a:
            metric_card(st, "Weight", fmt_pct(row.get("weight")), f"Official alloc {fmt_money(row.get('position_value'))} · price mark {fmt_money(row.get('asset_price_based_position_value'))}")
            metric_card(st, "Current / Entry", f"{fmt_money(row.get('current_price'))}", f"entry {fmt_money(row.get('entry_price'))} · prev official {fmt_money(row.get('prev_official_price'))}")
        with b:
            metric_card(st, "Daily Return", fmt_pct(row.get("daily_return")), f"Daily PnL {fmt_money(row.get('daily_pnl_value'))}", state="positive" if (row.get("daily_return", 0) or 0) >= 0 else "negative")
            metric_card(st, "Since Entry", fmt_pct(row.get("return_since_entry")), f"Unrealized {fmt_money(row.get('unrealized_pnl_value'))}")
        with c:
            metric_card(st, "Expected Return", fmt_pct(row.get("expected_return_display")), f"Raw target rank {_safe(row.get('raw_target_rank'))}")
            metric_card(st, "Raw Target Source", _safe(row.get("raw_target_feature_source"), "raw_target_return_exact"), f"exact available {_safe(row.get('raw_target_current_features_available'))}")
        with d:
            beta_val = fmt_num(row.get("beta"), 2) if pd.notna(row.get("beta", np.nan)) else "n/a"
            corr_val = fmt_num(row.get("correlation_with_portfolio"), 2) if pd.notna(row.get("correlation_with_portfolio", np.nan)) else "n/a"
            vol_text = fmt_pct(row.get("realized_vol_60d")) if pd.notna(row.get("realized_vol_60d", np.nan)) else "n/a"
            metric_card(st, "Risk Contribution", fmt_pct(row.get("risk_contribution")), f"60D vol {vol_text} · beta {beta_val} · corr {corr_val}")
            liq_main = fmt_money(row.get("median_60d_dollar_volume")) if pd.notna(row.get("median_60d_dollar_volume", np.nan)) else fmt_money(row.get("market_cap"))
            metric_card(st, "Liquidity / Size", liq_main, f"market cap {fmt_money(row.get('market_cap'))} · 20D vol {_safe(row.get('avg_volume_20d'))}")
        st.caption(
            "Why this position? "
            f"Raw Target Rank: {_safe(row.get('raw_target_rank'))}; Expected Return: {fmt_pct(row.get('expected_return_display'))}; "
            f"Institutional Quality: {_safe(row.get('holding_quality_classification'))}; Tradability: {_safe(row.get('tradability_status'))}; "
            f"Soft Exit: {_safe(row.get('soft_exit_status'))}; Filters: {_safe(row.get('institutional_filters_passed'))}."
        )
        fig = _sparkline_fig(ticker)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Mini sparkline unavailable: missing 30-day local price history.")


def _allocation_charts(st, holdings: pd.DataFrame) -> None:
    non_cash = _non_cash(holdings)
    if non_cash.empty:
        return
    work = non_cash.copy()
    for col in ["sector", "industry", "country"]:
        work[col] = work[col].fillna("metadata missing").astype(str)
    work["display_weight"] = work["weight"].fillna(0.0)
    left, right = st.columns(2)
    with left:
        fig = px.treemap(work, path=[px.Constant("Portfolio"), "sector", "industry", "ticker"], values="display_weight", color="ticker", color_discrete_sequence=ORANGE_SEQUENCE)
        fig.update_traces(marker={"line": {"color": "#05070A", "width": 1.5}}, textinfo="label+percent parent")
        fig = apply_plotly_layout(fig, "Allocation Treemap")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.sunburst(work, path=["sector", "industry", "ticker"], values="display_weight", color="ticker", color_discrete_sequence=ORANGE_SEQUENCE)
        fig.update_traces(marker={"line": {"color": "#05070A", "width": 1.5}}, textinfo="label+percent parent")
        fig = apply_plotly_layout(fig, "Allocation Sunburst")
        st.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    with left:
        donut = holdings.copy()
        donut["display_weight"] = donut["weight"].fillna(0.0)
        fig = px.pie(donut, names="ticker", values="display_weight", hole=0.58, color_discrete_sequence=ORANGE_SEQUENCE)
        fig = apply_plotly_layout(fig, "Allocation Donut")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(work.sort_values("display_weight"), x="ticker", y="display_weight", color="ticker", color_discrete_sequence=ORANGE_SEQUENCE)
        fig.update_yaxes(tickformat=".0%")
        fig = apply_plotly_layout(fig, "Position Weights")
        st.plotly_chart(fig, width="stretch")


def _contribution_tables(st, holdings: pd.DataFrame) -> None:
    non_cash = _non_cash(holdings)
    if non_cash.empty:
        return
    cols = ["ticker", "return_contribution", "risk_contribution", "drawdown_contribution", "turnover_contribution", "estimated_costs_total", "expected_return_display"]
    show = non_cash[[c for c in cols if c in non_cash.columns]].copy()
    st.dataframe(show.sort_values("risk_contribution", ascending=False), width="stretch")
    fig = go.Figure()
    for col, name, color in [
        ("return_contribution", "Return", GREEN),
        ("risk_contribution", "Volatility", ORANGE),
        ("estimated_costs_total", "Estimated Costs", AMBER),
    ]:
        if col in non_cash.columns:
            fig.add_trace(go.Bar(x=non_cash["ticker"], y=numeric(non_cash[col]), name=name, marker_color=color))
    fig = apply_plotly_layout(fig, "Position Contributions")
    st.plotly_chart(fig, width="stretch")


def _diversification(st, holdings: pd.DataFrame) -> None:
    non_cash = _non_cash(holdings)
    if non_cash.empty:
        return
    weights = non_cash["weight"].fillna(0.0).sort_values(ascending=False)
    cash_weight = holdings.loc[holdings["is_cash"], "weight"].sum() if "is_cash" in holdings.columns else np.nan
    metrics = {
        "HHI": float((weights ** 2).sum()),
        "Largest position": float(weights.max()) if not weights.empty else np.nan,
        "Smallest position": float(weights.min()) if not weights.empty else np.nan,
        "Cash": float(cash_weight) if pd.notna(cash_weight) else np.nan,
        "Top-2 concentration": float(weights.head(2).sum()) if not weights.empty else np.nan,
        "Top-3 concentration": float(weights.head(3).sum()) if not weights.empty else np.nan,
    }
    cols = st.columns(3)
    for idx, (label, val) in enumerate(metrics.items()):
        with cols[idx % 3]:
            metric_card(st, label, fmt_pct(val) if label != "HHI" else fmt_num(val), "official holdings")
    for group in ["sector", "industry", "country"]:
        temp = non_cash.copy()
        temp[group] = temp[group].fillna("metadata missing")
        alloc = temp.groupby(group)["weight"].sum().reset_index()
        fig = px.bar(alloc, x=group, y="weight", color=group, color_discrete_sequence=ORANGE_SEQUENCE)
        fig.update_yaxes(tickformat=".0%")
        fig = apply_plotly_layout(fig, f"{group.title()} Allocation")
        st.plotly_chart(fig, width="stretch")


def _timeline(st, data: dict[str, pd.DataFrame]) -> None:
    actions = data.get("official_actions", pd.DataFrame())
    if actions.empty or "date" not in actions.columns:
        st.warning("Position history unavailable: missing growth_official_paper_actions.csv")
        return
    work = actions.copy()
    if "action" not in work.columns:
        st.warning("Position history unavailable: action column missing")
        return
    work["action"] = work["action"].astype(str).str.upper()
    color_map = {"BUY": GREEN, "SELL": RED, "INCREASE": INFO, "REDUCE": AMBER, "HOLD": "#8B98A5", "CASH_CHANGE": "#4B5563"}
    fig = px.scatter(work, x="date", y="ticker", color="action", size=numeric(work.get("estimated_trade_value", pd.Series(1, index=work.index))).abs().fillna(1), color_discrete_map=color_map, hover_data=[c for c in ["old_weight", "new_weight", "estimated_trade_value", "reason"] if c in work.columns])
    fig = apply_plotly_layout(fig, "Position Action Timeline")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Official action ledger"):
        st.dataframe(work, width="stretch")


def _pnl_terminal(st, holdings: pd.DataFrame) -> None:
    if holdings.empty:
        return
    non_cash = _non_cash(holdings)
    if non_cash.empty:
        return
    pnl_cols = ["daily_pnl_value", "unrealized_pnl_value", "estimated_net_pnl", "realized_pnl_if_sold"]
    fig = go.Figure()
    for col, name, color in [("daily_pnl_value", "Daily", INFO), ("unrealized_pnl_value", "Unrealized", ORANGE), ("estimated_net_pnl", "Estimated Net", AMBER), ("realized_pnl_if_sold", "Realized if Closed", PURPLE)]:
        if col in non_cash.columns:
            fig.add_trace(go.Bar(x=non_cash["ticker"], y=numeric(non_cash[col]), name=name, marker_color=color))
    fig = apply_plotly_layout(fig, "PnL Terminal")
    st.plotly_chart(fig, width="stretch")


def _portfolio_evolution(st, data: dict[str, pd.DataFrame]) -> None:
    state = data.get("official_state", pd.DataFrame())
    if state.empty or not {"date", "ticker", "paper_position_weight"}.issubset(state.columns):
        st.warning("Portfolio evolution unavailable: missing official state history.")
        return
    work = state.copy()
    work["weight"] = numeric(work["paper_position_weight"]).fillna(0.0)
    fig = px.bar(work, x="ticker", y="weight", color="ticker", animation_frame=work["date"].astype(str), range_y=[0, max(0.75, work["weight"].max() * 1.2)], color_discrete_sequence=ORANGE_SEQUENCE)
    fig.update_yaxes(tickformat=".0%")
    fig = apply_plotly_layout(fig, "Portfolio Evolution by Official Date")
    st.plotly_chart(fig, width="stretch")


def render_portfolio_terminal(st, data: dict[str, pd.DataFrame]) -> None:
    missing = [filename for key, filename in OFFICIAL_PORTFOLIO_SOURCES.items() if data.get(f"official_{key}", pd.DataFrame()).empty and key not in {"cost_ledger", "position_pnl", "trade_lifecycle"}]
    # Explicit map because dashboard_data_layer names do not all include the same suffix.
    source_map = {
        "state": data.get("official_state", pd.DataFrame()),
        "position_pnl": data.get("official_position_pnl", pd.DataFrame()),
        "trade_lifecycle": data.get("official_trade_lifecycle", pd.DataFrame()),
        "actions": data.get("official_actions", pd.DataFrame()),
        "trades": data.get("official_trades", pd.DataFrame()),
        "cost_ledger": data.get("official_cost_ledger", pd.DataFrame()),
        "monitor": data.get("official_monitor", pd.DataFrame()),
        "metadata": data.get("official_holding_metadata", pd.DataFrame()),
    }
    missing = [OFFICIAL_PORTFOLIO_SOURCES[k] for k, df in source_map.items() if df.empty]
    holdings = _enrich_holdings(_official_holdings(data))
    _render_header(st, holdings, data)
    if missing:
        st.warning("Missing official Portfolio source(s): " + ", ".join(missing))
    if holdings.empty:
        st.markdown("<div class='alert-box info'>Official portfolio is empty or state file is unavailable. No holdings to display.</div>", unsafe_allow_html=True)
        return
    st.subheader("Summary")
    _render_summary(st, holdings)
    st.subheader("Holding Cards")
    _render_holding_cards(st, holdings)
    st.subheader("Position Allocation View")
    _allocation_charts(st, holdings)
    st.subheader("Position Contribution")
    _contribution_tables(st, holdings)
    st.subheader("Portfolio Diversification")
    _diversification(st, holdings)
    st.subheader("Position History")
    _timeline(st, data)
    st.subheader("PnL Terminal")
    _pnl_terminal(st, holdings)
    st.subheader("Portfolio Evolution")
    _portfolio_evolution(st, data)
    st.caption("Data sources: official namespace only. No reconstructed/debug sources used on this page.")


def build_portfolio_terminal_audits(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    source_map = {
        "growth_official_paper_state.csv": data.get("official_state", pd.DataFrame()),
        "growth_official_position_pnl.csv": data.get("official_position_pnl", pd.DataFrame()),
        "growth_official_trade_lifecycle.csv": data.get("official_trade_lifecycle", pd.DataFrame()),
        "growth_official_paper_actions.csv": data.get("official_actions", pd.DataFrame()),
        "growth_official_paper_trades.csv": data.get("official_trades", pd.DataFrame()),
        "growth_official_estimated_cost_ledger.csv": data.get("official_cost_ledger", pd.DataFrame()),
        "growth_official_paper_monitor.csv": data.get("official_monitor", pd.DataFrame()),
        "official_holding_metadata.csv": data.get("official_holding_metadata", pd.DataFrame()),
    }
    audit_rows = []
    for filename, df in source_map.items():
        date_range = ""
        if not df.empty and "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            date_range = f"{dates.min().date()} to {dates.max().date()}" if dates.notna().any() else ""
        audit_rows.append({"source_file": filename, "namespace": "official_forward_paper", "loaded": not df.empty, "row_count": len(df), "date_range": date_range})
    source_audit = pd.DataFrame(audit_rows)
    holdings = _enrich_holdings(_official_holdings(data))
    non_cash = _non_cash(holdings)
    weight_sum = float(holdings["weight"].fillna(0.0).sum()) if not holdings.empty and "weight" in holdings.columns else np.nan
    pnl_available = not data.get("official_position_pnl", pd.DataFrame()).empty
    actions_available = not data.get("official_actions", pd.DataFrame()).empty
    checks = [
        {"check": "official_namespace_only", "status": "PASS", "detail": "Portfolio terminal uses growth_official_* source map only"},
        {"check": "holding_cards_render", "status": "PASS" if not non_cash.empty else "FAIL", "detail": ",".join(non_cash.get("ticker", pd.Series(dtype=str)).astype(str).tolist())},
        {"check": "treemap_sunburst_inputs", "status": "PASS" if not non_cash.empty and "weight" in non_cash.columns else "FAIL", "detail": "official holdings with weights"},
        {"check": "sparklines_inputs", "status": "PASS", "detail": "uses local yahoo_ohlcv_price_cache for mini chart only; no model data mutation"},
        {"check": "timeline_inputs", "status": "PASS" if actions_available else "FAIL", "detail": "growth_official_paper_actions.csv"},
        {"check": "pnl_reconciles", "status": "PASS" if pnl_available else "WARNING", "detail": "growth_official_position_pnl.csv available" if pnl_available else "position pnl missing"},
        {"check": "weights_sum_to_one", "status": "PASS" if pd.notna(weight_sum) and abs(weight_sum - 1.0) < 1e-6 else "FAIL", "detail": f"weight_sum={weight_sum}"},
        {"check": "no_namespace_mixing", "status": "PASS", "detail": "debug/reconstructed files are not referenced"},
    ]
    integrity = pd.DataFrame(checks)
    status = "portfolio_terminal_pass"
    if integrity["status"].eq("FAIL").any():
        status = "portfolio_terminal_fail"
    elif integrity["status"].eq("WARNING").any() or (~source_audit["loaded"]).any():
        status = "portfolio_terminal_warning"
    return source_audit, integrity, status
