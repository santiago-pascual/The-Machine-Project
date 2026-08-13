from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard_mission_control import render_mission_control
from dashboard_alert_engine import render_alert_center
from dashboard_notifications import render_live_status_bar
from dashboard_quant_lab import render_quant_lab
from dashboard_components import (
    action_counts,
    bar_chart,
    drawdown_frame,
    fmt_money,
    fmt_num,
    fmt_pct,
    fmt_pct_points,
    heatmap,
    latest_value,
    line_chart,
    metric_card,
    monthly_return_table,
    source_caption,
)
from dashboard_executive import build_executive_audits, render_executive_terminal
from dashboard_execution import render_execution_terminal
from dashboard_governance import render_governance_terminal
from dashboard_historical_replay import render_historical_replay
from dashboard_portfolio import build_portfolio_terminal_audits, render_portfolio_terminal
from dashboard_risk import render_risk_terminal
from dashboard_research import render_research_terminal
from dashboard_report_generator import render_report_generator
from dashboard_alpha_attribution import render_alpha_attribution
from dashboard_decision_engine import render_decision_engine
from dashboard_data_layer import (
    MODEL_NAME,
    MODEL_VERSION,
    VARIANT,
    benchmark_curve_for_scope,
    current_holdings,
    equity_from_performance,
    get_scope_namespace,
    latest,
    latest_market_date,
    next_rebalance_date,
    load_all,
    numeric,
    official_start_date,
    read_price_cache,
    scope_data,
)
from dashboard_theme import CSS, apply_plotly_layout
from dashboard_layout import hero as layout_hero, page_header as layout_page_header, sidebar as layout_sidebar


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def render_header(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    official_status = latest(data.get("official_monitor", pd.DataFrame()))
    integrity = "unavailable"
    next_rebalance = next_rebalance_date(data)
    if not official_status.empty:
        row = official_status.iloc[-1]
        integrity = str(row.get("integrity_status", "unavailable"))
    st.markdown(
        f"""
        <div class='hero'>
          <h1>Growth Champion Final</h1>
          <p>{MODEL_VERSION} · {VARIANT} · read-only institutional dashboard · real capital blocked</p>
          <p>Scope: <b>{scope}</b> · Namespace: <b>{get_scope_namespace(scope)}</b> · Latest market date: <b>{latest_market_date(data)}</b> · Integrity: <b>{integrity}</b> · Next rebalance: <b>{next_rebalance}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(st, data: dict[str, pd.DataFrame]) -> tuple[str, str]:
    st.sidebar.markdown("### Growth Control Room")
    scope = st.sidebar.selectbox("Data scope", ["Official Forward Paper", "Historical 2008 Backtest"], index=0)
    nav = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "Portfolio",
            "Rebalance Ledger",
            "Performance",
            "Risk",
            "Costs & Capacity",
            "Live Validation",
            "Market Data",
            "Quant Lab 3D",
            "Diagnostics",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model: {MODEL_NAME}")
    st.sidebar.caption(f"Version: {MODEL_VERSION}")
    st.sidebar.caption(f"Latest market date: {latest_market_date(data)}")
    st.sidebar.caption(f"Official start: {official_start_date(data)}")
    if scope != "Official Forward Paper":
        st.sidebar.warning("Historical 2008 backtest scope. Reconstructed stress data; never mixed with official forward paper metrics.")
    return scope, nav


def _latest_perf(data: dict[str, pd.DataFrame], scope: str) -> pd.Series:
    perf = scope_data(data, scope).get("performance", pd.DataFrame())
    row = latest(perf)
    return row.iloc[-1] if not row.empty else pd.Series(dtype=object)


def render_executive(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    row = _latest_perf(data, scope)
    curve = benchmark_curve_for_scope(data, scope)
    latest_curve = curve.iloc[-1] if not curve.empty else pd.Series(dtype=object)
    cols = st.columns(5)
    with cols[0]: metric_card(st, "Gross Portfolio Value", fmt_money(row.get("gross_portfolio_value", row.get("portfolio_value", np.nan))), "official gross" if scope == "Official Forward Paper" else scope)
    with cols[1]: metric_card(st, "Estimated Net Value", fmt_money(row.get("estimated_net_portfolio_value", np.nan)), "cost-adjusted estimate")
    with cols[2]: metric_card(st, "Gross Cum Return", fmt_pct(row.get("gross_cumulative_return", np.nan)), "paper gross")
    with cols[3]: metric_card(st, "Estimated Net Cum", fmt_pct(row.get("estimated_net_cumulative_return", np.nan)), "paper net estimate")
    with cols[4]: metric_card(st, "Current Drawdown", fmt_pct(row.get("current_drawdown", row.get("max_drawdown", np.nan))), "gross equity")
    cols = st.columns(5)
    with cols[0]: metric_card(st, "SPY Cumulative", fmt_pct_points(latest_curve.get("SPY", np.nan)), "same official dates")
    with cols[1]: metric_card(st, "QQQ Cumulative", fmt_pct_points(latest_curve.get("QQQ", np.nan)), "same official dates")
    with cols[2]: metric_card(st, "Exposure", fmt_pct(row.get("exposure", np.nan)), "final exposure")
    with cols[3]: metric_card(st, "Cash", fmt_pct(row.get("cash_weight", row.get("cash", np.nan))), "cash weight")
    reb = latest(data.get("official_daily_status", pd.DataFrame()))
    with cols[4]: metric_card(st, "Next Rebalance", next_rebalance_date(data), "5-session cadence")

    st.markdown("#### Growth vs SPY vs QQQ")
    if scope == "Official Forward Paper":
        st.caption(f"Official Forward Paper — start date: {official_start_date(data)}")
    line_chart(st, curve, "date", [c for c in ["Growth Gross", "Growth Estimated Net", "Growth Champion Final", "SPY", "QQQ"] if c in curve.columns], "Growth gross/net vs SPY/QQQ — cumulative return %")
    source_caption(st, "growth_official_benchmark_equity.csv" if scope == "Official Forward Paper" else "selected scope benchmark files", get_scope_namespace(scope))

    scoped = scope_data(data, scope)
    perf = equity_from_performance(scoped.get("performance", pd.DataFrame()))
    if not perf.empty:
        line_cols = [c for c in ["exposure", "cash_weight"] if c in scoped.get("performance", pd.DataFrame()).columns]
        if line_cols:
            line_chart(st, scoped["performance"], "date", line_cols, "Portfolio Exposure / Cash Through Time")
        dd = drawdown_frame(scoped["performance"])
        line_chart(st, dd, "date", "drawdown", "Rolling Drawdown")
    actions = scoped.get("actions", pd.DataFrame())
    counts = action_counts(actions)
    if not counts.empty:
        bar_chart(st, counts, "date", "count", "Recent Actions Timeline", color="action")


def render_portfolio(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    holdings = current_holdings(data, scope)
    st.subheader("Current Portfolio")
    if holdings.empty:
        st.warning("Current portfolio unavailable for selected scope.")
        return
    non_cash = holdings[~holdings.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")].copy()
    if non_cash.empty:
        st.info("No non-cash holdings.")
    for _, row in non_cash.iterrows():
        cols = st.columns([1.2, 1, 1, 1, 1, 1])
        ticker = str(row.get("ticker", "n/a"))
        with cols[0]: st.markdown(f"<div class='holding-card'><div class='holding-ticker'>{ticker}</div><span class='badge'>{row.get('holding_quality_classification', row.get('action', 'holding'))}</span><div class='small-muted'>{row.get('holding_risk_notes', '')}</div></div>", unsafe_allow_html=True)
        with cols[1]: metric_card(st, "Weight", fmt_pct(row.get("paper_position_weight", row.get("weight", np.nan))))
        with cols[2]: metric_card(st, "Position Value", fmt_money(row.get("paper_position_value", row.get("position_value", np.nan))))
        with cols[3]: metric_card(st, "Current Price", fmt_money(row.get("current_price", np.nan)))
        with cols[4]: metric_card(st, "Day PnL", fmt_money(row.get("position_pnl_today", np.nan)), fmt_pct(row.get("day_return_pct", np.nan)))
        with cols[5]: metric_card(st, "Unrealized PnL", fmt_money(row.get("position_unrealized_pnl", np.nan)), fmt_pct(row.get("unrealized_return", np.nan)))
    show_cols = [c for c in ["date", "ticker", "action", "paper_position_weight", "paper_position_value", "entry_price", "current_price", "day_return_pct", "position_pnl_today", "unrealized_return", "position_unrealized_pnl", "raw_target_rank", "raw_target_return_exact", "holding_quality_classification", "holding_risk_notes"] if c in holdings.columns]
    with st.expander("Detailed holdings table"):
        st.dataframe(holdings[show_cols] if show_cols else holdings, width="stretch")
    source_caption(st, "growth_official_paper_state.csv + growth_official_position_pnl.csv" if scope == "Official Forward Paper" else "selected scope state", get_scope_namespace(scope))


def render_rebalance(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    scoped = scope_data(data, scope)
    actions = scoped.get("actions", pd.DataFrame())
    report = scoped.get("rebalance", pd.DataFrame())
    st.subheader("Rebalance Ledger")
    if actions.empty:
        st.warning("Rebalance ledger unavailable.")
        return
    latest_actions = latest(actions)
    latest_report = latest(report)
    cols = st.columns(5)
    with cols[0]: metric_card(st, "Latest Date", str(latest_actions.iloc[-1].get("date", "n/a"))[:10])
    with cols[1]: metric_card(st, "Turnover", fmt_pct(latest_report.iloc[-1].get("turnover", np.nan)) if not latest_report.empty else "n/a")
    with cols[2]: metric_card(st, "BUY", str(int((latest_actions.get("action", pd.Series()).astype(str).str.upper() == "BUY").sum())))
    with cols[3]: metric_card(st, "SELL", str(int((latest_actions.get("action", pd.Series()).astype(str).str.upper() == "SELL").sum())))
    with cols[4]: metric_card(st, "Recon", str(latest_report.iloc[-1].get("reconciliation_passed", "n/a")) if not latest_report.empty else "n/a")
    counts = action_counts(actions)
    bar_chart(st, counts, "date", "count", "Actions by Day", color="action")
    if not report.empty and "turnover" in report.columns:
        line_chart(st, report, "date", "turnover", "Turnover by Day")
    with st.expander("Daily action table"):
        st.dataframe(actions, width="stretch")
    with st.expander("Daily rebalance summary"):
        st.dataframe(report, width="stretch")


def render_performance(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    scoped = scope_data(data, scope)
    perf = scoped.get("performance", pd.DataFrame())
    curve = benchmark_curve_for_scope(data, scope)
    st.subheader("Performance")
    line_chart(st, curve, "date", [c for c in ["Growth Gross", "Growth Estimated Net", "Growth Champion Final", "SPY", "QQQ"] if c in curve.columns], "Growth gross/net vs SPY/QQQ")
    eq = equity_from_performance(perf)
    line_chart(st, eq, "date", [c for c in ["Gross Equity", "Estimated Net Equity"] if c in eq.columns], "Gross vs Estimated-Net Equity")
    if not perf.empty and "gross_daily_return" in perf.columns:
        out = perf.sort_values("date").copy()
        out["rolling_return"] = (1 + numeric(out["gross_daily_return"]).fillna(0)).rolling(20, min_periods=2).apply(np.prod, raw=True) - 1
        out["rolling_sharpe"] = numeric(out["gross_daily_return"]).rolling(20, min_periods=2).mean() / numeric(out["gross_daily_return"]).rolling(20, min_periods=2).std(ddof=0).replace(0, np.nan) * np.sqrt(252)
        line_chart(st, out, "date", ["rolling_return", "rolling_sharpe"], "Rolling Return / Sharpe")
    dd = drawdown_frame(perf)
    line_chart(st, dd, "date", "drawdown", "Drawdown")
    monthly = monthly_return_table(perf)
    if not monthly.empty:
        heatmap(st, monthly, "Monthly Returns Heatmap")
    pnl = data.get("official_position_pnl", pd.DataFrame())
    if scope == "Official Forward Paper" and not pnl.empty and "position_pnl_today" in pnl.columns:
        bar_chart(st, pnl, "ticker", "position_pnl_today", "Realized Daily PnL by Holding")


def render_risk(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    st.subheader("Risk")
    scoped = scope_data(data, scope)
    perf = scoped.get("performance", pd.DataFrame())
    vol = data.get("vol_fresh", pd.DataFrame())
    if not vol.empty:
        cols = [c for c in ["estimated_portfolio_vol", "target_vol", "uncapped_exposure", "final_exposure", "dual_trend_cap", "exposure_cap_60"] if c in vol.columns]
        line_chart(st, vol, "date", cols, "Volatility Targeting: Estimated Vol / Exposure Stack")
        source_caption(st, "growth_volatility_targeting_fresh.csv", "official diagnostic")
    holdings = current_holdings(data, scope)
    non_cash = holdings[~holdings.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")] if not holdings.empty else pd.DataFrame()
    weights = numeric(non_cash.get("paper_position_weight", pd.Series(dtype=float))) if not non_cash.empty else pd.Series(dtype=float)
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Concentration HHI", fmt_num(float((weights ** 2).sum()) if not weights.empty else np.nan))
    with cols[1]: metric_card(st, "Top Weight", fmt_pct(float(weights.max()) if not weights.empty else np.nan))
    with cols[2]: metric_card(st, "Current DD", fmt_pct(latest_value(drawdown_frame(perf), "drawdown")))
    with cols[3]: metric_card(st, "Position Count", str(len(non_cash)))
    tickers = non_cash.get("ticker", pd.Series(dtype=str)).astype(str).tolist() if not non_cash.empty else []
    prices = read_price_cache(tickers, lookback=126)
    if not prices.empty:
        corr = prices.drop(columns=["date"], errors="ignore").pct_change().corr()
        heatmap(st, corr, "Current Holdings Correlation Heatmap")
    dd = drawdown_frame(perf)
    line_chart(st, dd, "date", "drawdown", "Drawdown Duration / Underwater Curve")
    hmm = data.get("hmm_comparison", pd.DataFrame())
    if not hmm.empty:
        st.caption("HMM 4-state regime is diagnostic only; it does not alter official allocation.")
        st.dataframe(hmm.tail(3), width="stretch")


def render_costs(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    st.subheader("Costs & Capacity")
    ledger = data.get("official_cost_ledger", pd.DataFrame()) if scope == "Official Forward Paper" else data.get("advanced_costs", pd.DataFrame())
    if ledger.empty:
        st.warning("Cost ledger unavailable for selected scope.")
    else:
        cols = st.columns(4)
        with cols[0]: metric_card(st, "Cumulative Costs", fmt_money(numeric(ledger.get("estimated_total_cost", ledger.get("total_cost", pd.Series(dtype=float)))).sum()))
        with cols[1]: metric_card(st, "Orders", str(len(ledger)))
        with cols[2]: metric_card(st, "Avg Cost / Order", fmt_money(numeric(ledger.get("estimated_total_cost", ledger.get("total_cost", pd.Series(dtype=float)))).mean()))
        with cols[3]: metric_card(st, "Official Cost Mode", "Reporting Only" if scope == "Official Forward Paper" else "Research")
        date_cost = ledger.copy()
        cost_col = "daily_estimated_cost" if "daily_estimated_cost" in date_cost.columns else "total_cost" if "total_cost" in date_cost.columns else None
        if cost_col:
            daily = date_cost.groupby("date", dropna=False)[cost_col].sum().reset_index(name="cost") if "date" in date_cost.columns else pd.DataFrame()
            bar_chart(st, daily, "date", "cost", "Costs by Date")
        if "ticker" in ledger.columns and cost_col:
            ticker_cost = ledger.groupby("ticker")[cost_col].sum().reset_index(name="cost").sort_values("cost", ascending=False)
            bar_chart(st, ticker_cost, "ticker", "cost", "Costs by Ticker")
        with st.expander("Official cost ledger / execution detail"):
            st.dataframe(ledger, width="stretch")
    perf = equity_from_performance(scope_data(data, scope).get("performance", pd.DataFrame()))
    line_chart(st, perf, "date", [c for c in ["Gross Equity", "Estimated Net Equity"] if c in perf.columns], "Gross vs Estimated-Net Equity")
    cap = data.get("capacity", pd.DataFrame())
    if cap.empty:
        cap = data.get("growth_capacity", pd.DataFrame())
    if not cap.empty:
        st.subheader("Capacity / ADV Participation")
        st.dataframe(cap, width="stretch")
    adv = data.get("advanced_costs", pd.DataFrame())
    if not adv.empty and "scenario" in adv.columns and "total_cost" in adv.columns:
        scenario = adv.groupby("scenario")["total_cost"].sum().reset_index()
        bar_chart(st, scenario, "scenario", "total_cost", "Advanced Execution Cost Scenarios")


def render_live(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    st.subheader("Live Validation")
    health = latest(data.get("official_monitor", pd.DataFrame())) if scope == "Official Forward Paper" else latest(data.get("debug_monitor", pd.DataFrame()))
    row = health.iloc[-1] if not health.empty else pd.Series(dtype=object)
    cols = st.columns(5)
    with cols[0]: metric_card(st, "Status", str(row.get("governance_status", "WARMUP")))
    with cols[1]: metric_card(st, "Data Status", str(row.get("data_status", "n/a")))
    with cols[2]: metric_card(st, "Integrity", str(row.get("integrity_status", "n/a")))
    with cols[3]: metric_card(st, "Promotion", str(row.get("promotion_status", "real_capital_blocked")))
    with cols[4]: metric_card(st, "Risk Flags", str(row.get("risk_flags", "n/a")))
    tracking = data.get("official_tracking", pd.DataFrame())
    if not tracking.empty:
        st.dataframe(tracking.tail(10), width="stretch")
    gates = data.get("official_integrity", pd.DataFrame())
    if not gates.empty:
        st.dataframe(gates.tail(10), width="stretch")


def render_market_data(st, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Market Data")
    cols = st.columns(4)
    gov = latest(data.get("official_market_data_governance", pd.DataFrame()))
    row = gov.iloc[-1] if not gov.empty else pd.Series(dtype=object)
    with cols[0]: metric_card(st, "Primary Source", "Yahoo/yfinance")
    with cols[1]: metric_card(st, "Secondary Status", str(row.get("classification", row.get("governance", "single_source_warning"))))
    with cols[2]: metric_card(st, "Latest Date", latest_market_date(data))
    with cols[3]: metric_card(st, "Real Capital", "Blocked", "requires reliable second source")
    for key, title in [("official_market_data_integrity", "Official Market Data Integrity"), ("secondary_provider_status", "Secondary Provider Status"), ("multi_source_price_audit", "Multi-Source Price Audit"), ("market_data_governance", "Market Data Governance")]:
        df = data.get(key, pd.DataFrame())
        if not df.empty:
            with st.expander(title):
                st.dataframe(df, width="stretch")


def render_diagnostics(st, data: dict[str, pd.DataFrame], diag: pd.DataFrame, scope: str) -> None:
    st.subheader("Diagnostics")
    st.caption("Data loading and namespace audit. This page is read-only.")
    st.dataframe(diag, width="stretch")
    for key in ["benchmark_chart_audit", "benchmark_chart_reconciliation", "official_integrity", "official_daily_status", "official_version_history"]:
        df = data.get(key, pd.DataFrame())
        if not df.empty:
            with st.expander(key):
                st.dataframe(df, width="stretch")


def render_execution(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    left, right = st.columns([1, 1])
    with left:
        render_rebalance(st, data, scope)
    with right:
        render_costs(st, data, scope)


def render_research(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    st.subheader("Research Context")
    final = data.get("final_results", pd.DataFrame())
    recon = data.get("reconstructed_results", pd.DataFrame())
    stress = data.get("final_stress", pd.DataFrame())
    param = data.get("parameter_stability", pd.DataFrame())
    hmm = data.get("hmm_comparison", pd.DataFrame())
    if not final.empty:
        with st.expander("Growth final selection results"):
            st.dataframe(final, width="stretch")
    if not recon.empty:
        with st.expander("Reconstructed long-horizon stress"):
            st.dataframe(recon, width="stretch")
    if not stress.empty:
        bar_chart(st, stress, "stress_period", "max_drawdown", "Stress Period Drawdown", color="window_start")
        with st.expander("Stress period table"):
            st.dataframe(stress, width="stretch")
    if not param.empty:
        st.caption("Parameter stability is research-only; active configuration is frozen elsewhere.")
        with st.expander("Parameter stability map"):
            st.dataframe(param, width="stretch")
    if not hmm.empty:
        st.caption("HMM is diagnostic only; it does not alter holdings, rankings, exposure or weights.")
        with st.expander("HMM model comparison"):
            st.dataframe(hmm, width="stretch")


def render_governance(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    render_live(st, data, scope)
    st.markdown("---")
    render_market_data(st, data)


def render_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Growth Champion Final", page_icon="", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    data, diag = load_all()
    scope, nav = layout_sidebar(st, data)
    layout_hero(st, data, scope)
    render_live_status_bar(st, data)
    layout_page_header(st, nav, scope, data)
    if nav == "Mission Control":
        render_mission_control(st, data)
    elif nav == "Alert Center":
        render_alert_center(st, data)
    elif nav == "Executive":
        render_executive_terminal(st, data)
    elif nav == "Portfolio":
        render_portfolio_terminal(st, data)
    elif nav == "Decision Engine":
        render_decision_engine(st, data)
    elif nav == "Performance":
        render_performance(st, data, scope)
    elif nav == "Risk":
        render_risk_terminal(st, data)
    elif nav == "Execution":
        render_execution_terminal(st, data)
    elif nav == "Research":
        render_research_terminal(st, data)
    elif nav == "Alpha Attribution":
        render_alpha_attribution(st, data)
    elif nav == "Governance":
        render_governance_terminal(st, data)
    elif nav == "Reports":
        render_report_generator(st, data)
    elif nav == "Historical Replay":
        render_historical_replay(st, data)
    elif nav == "Quant Lab 3D":
        render_quant_lab(st, data)
    else:
        render_diagnostics(st, data, diag, scope)


def fallback_cli() -> None:
    data, diag = load_all()
    print("===== GROWTH CHAMPION FINAL DASHBOARD =====")
    print(f"Streamlit installed: {has_module('streamlit')}")
    if not has_module("streamlit"):
        print("Install with: pip install streamlit plotly")
    print(f"Official start: {official_start_date(data)}")
    print(f"Latest market date: {latest_market_date(data)}")
    print(diag[["name", "exists", "rows"]].to_string(index=False))


def main() -> None:
    if has_module("streamlit") and any("streamlit" in arg.lower() for arg in sys.argv):
        render_streamlit()
    elif has_module("streamlit"):
        render_streamlit()
    else:
        fallback_cli()


if __name__ == "__main__":
    main()
