
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_components import (
    alert_box,
    fmt_money,
    fmt_num,
    fmt_pct,
    metric_card,
    section_header,
    source_caption,
)
from dashboard_execution_calculations import build_execution_bundle
from dashboard_theme import (
    AMBER,
    BRIGHT_ORANGE,
    CHART_COLORS,
    CYAN,
    GREEN,
    INFO,
    MUTED_ORANGE,
    ORANGE,
    RED,
    apply_plotly_layout,
)

EXEC_COLORS = [ORANGE, BRIGHT_ORANGE, AMBER, MUTED_ORANGE, INFO, CYAN, GREEN, RED]
ACTION_COLORS = {
    "BUY": GREEN,
    "SELL": RED,
    "INCREASE": CYAN,
    "REDUCE": AMBER,
    "HOLD": "#6B7280",
    "CASH_CHANGE": INFO,
}


def _arrow_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(200).tolist()
            type_names = {type(v).__name__ for v in sample}
            if len(type_names) > 1 or any(t in type_names for t in ("str", "dict", "list", "tuple")):
                out[col] = series.map(lambda v: "" if pd.isna(v) else str(v))
    return out


def _safe_df(st, df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    if df.empty:
        st.info("No data available for this section.")
        return
    view = df.copy()
    if columns:
        existing = [c for c in columns if c in view.columns]
        view = view[existing] if existing else view
    kwargs = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(_arrow_safe_frame(view), **kwargs)


def _money(x: Any) -> str:
    return fmt_money(x)


def _pct(x: Any) -> str:
    return fmt_pct(x)


def _num(x: Any) -> str:
    return fmt_num(x)


def _header(st, kpis: dict[str, Any], status: str) -> None:
    section_header(
        st,
        "Execution Terminal",
        "Official Forward Paper execution diagnostics. Broker/orders disabled. Costs are estimated unless explicitly marked as live fills.",
        status,
    )
    cols = st.columns(4)
    with cols[0]:
        metric_card(st, "Latest Signal Date", str(kpis.get("latest_signal_date", "unavailable")), "growth_official_paper_performance.csv", badge="official")
    with cols[1]:
        metric_card(st, "Economic Application", str(kpis.get("economic_application_date", "unavailable")), "t+1 execution lag", badge="read only")
    with cols[2]:
        metric_card(st, "Last Rebalance", str(kpis.get("last_rebalance_date", "unavailable")), f"next {kpis.get('next_rebalance_date', 'unavailable')}", badge="official")
    with cols[3]:
        rec = str(kpis.get("reconciliation_status", "unavailable"))
        metric_card(st, "Reconciliation", rec, "broker/orders disabled", state="neutral" if rec.lower() in {"true", "pass"} else "warning", badge="no orders")


def _kpi_cards(st, bundle) -> None:
    k = bundle.kpis
    rows = [
        ("Last Rebalance Turnover", _pct(k.get("last_rebalance_turnover")), "growth_official_paper_rebalance_report.csv"),
        ("Cumulative Turnover", _pct(k.get("cumulative_turnover")), "official cumulative"),
        ("Costed Orders", str(k.get("costed_order_count", 0)), "BUY/SELL/INCREASE/REDUCE"),
        ("Last Rebalance Cost", _money(k.get("last_rebalance_estimated_cost")), "estimated execution"),
        ("Cumulative Cost", _money(k.get("cumulative_estimated_cost")), "reporting-only cost ledger"),
        ("Gross Portfolio Value", _money(k.get("gross_portfolio_value")), "gross accounting"),
        ("Estimated Net Value", _money(k.get("estimated_net_portfolio_value")), "gross less estimated costs"),
        ("Cost Drag", _money(k.get("cost_drag_since_start")), "since official start"),
        ("Avg Cost / Order", _money(k.get("average_cost_per_order")), "estimated"),
        ("Highest-Cost Ticker", str(k.get("highest_cost_ticker", "n/a")), "by cumulative estimated cost"),
        ("Max ADV Participation", _pct(k.get("current_adv_participation")), "current/action blotter"),
        ("Capacity Status", str(k.get("capacity_status", "unavailable")), "operational capacity report"),
    ]
    for i in range(0, len(rows), 4):
        cols = st.columns(4)
        for col, (label, value, note) in zip(cols, rows[i:i+4]):
            with col:
                metric_card(st, label, value, note, badge="official")


def _order_blotter(st, blotter: pd.DataFrame) -> None:
    cols = [
        "signal date", "economic application date", "ticker", "company", "action", "old weight", "new weight", "weight change",
        "estimated_trade_value", "reference price", "synthetic_quantity", "commission", "spread_cost", "slippage", "market_impact",
        "estimated_total_cost", "ADV", "participation_rate", "execution_status", "reconciliation_status",
    ]
    st.markdown("### Order Blotter")
    st.caption("Estimated execution quality — no live broker fills.")
    _safe_df(st, blotter, cols, height=420)


def _cost_charts(st, bundle) -> None:
    st.markdown("### Execution Cost Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        if not bundle.cost_components.empty:
            fig = px.bar(bundle.cost_components, x="component", y="estimated_cost", color="component", color_discrete_sequence=EXEC_COLORS)
            st.plotly_chart(apply_plotly_layout(fig, "Cost by component"), width="stretch")
        else:
            st.warning("Cost component chart unavailable.")
        if not bundle.cost_by_action.empty:
            fig = px.bar(bundle.cost_by_action, x="action", y="estimated_total_cost", color="action", color_discrete_map=ACTION_COLORS)
            st.plotly_chart(apply_plotly_layout(fig, "Cost by action type"), width="stretch")
    with c2:
        if not bundle.cost_by_ticker.empty:
            fig = px.bar(bundle.cost_by_ticker, x="ticker", y="estimated_total_cost", color="ticker", color_discrete_sequence=EXEC_COLORS)
            st.plotly_chart(apply_plotly_layout(fig, "Cost by ticker"), width="stretch")
        else:
            st.warning("Cost by ticker unavailable.")
        if not bundle.cost_by_date.empty:
            fig = px.line(bundle.cost_by_date, x="date", y="cumulative_estimated_cost", markers=True)
            st.plotly_chart(apply_plotly_layout(fig, "Cumulative estimated cost drag"), width="stretch")
    with st.expander("Cost ledger details"):
        _safe_df(st, bundle.cost_by_date)


def _gross_vs_net(st, equity: pd.DataFrame) -> None:
    st.markdown("### Gross vs Estimated Net")
    if equity.empty or "date" not in equity.columns:
        st.warning("Gross/net chart unavailable: missing official performance ledger.")
        return
    fig = go.Figure()
    if "gross_equity_display" in equity.columns:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["gross_equity_display"], mode="lines+markers", name="Gross equity", line=dict(color=CHART_COLORS["growth"])))
    if "estimated_net_equity_display" in equity.columns:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["estimated_net_equity_display"], mode="lines+markers", name="Estimated net equity", line=dict(color=CHART_COLORS["growth_net"])))
    if "cumulative_cost_drag" in equity.columns:
        fig.add_trace(go.Bar(x=equity["date"], y=equity["cumulative_cost_drag"], name="Cumulative cost drag", marker_color=AMBER, yaxis="y2", opacity=0.35))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cost drag"))
    st.plotly_chart(apply_plotly_layout(fig, "Official gross vs estimated net equity"), width="stretch")
    _safe_df(st, equity, ["date", "gross_equity_display", "estimated_net_equity_display", "cumulative_cost_drag", "estimated_execution_cost"])


def _turnover(st, turnover: pd.DataFrame) -> None:
    st.markdown("### Turnover Analysis")
    if turnover.empty:
        st.warning("Turnover unavailable.")
        return
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(turnover, x="date", y="turnover", color="session_type", color_discrete_sequence=EXEC_COLORS)
        st.plotly_chart(apply_plotly_layout(fig, "Turnover by session"), width="stretch")
    with c2:
        fig = px.scatter(turnover, x="turnover", y="gross_daily_return", color="session_type", size=turnover["turnover"].abs() + 0.001, color_discrete_sequence=EXEC_COLORS)
        st.plotly_chart(apply_plotly_layout(fig, "Turnover vs portfolio return"), width="stretch")
    _safe_df(st, turnover)


def _capacity(st, capacity: pd.DataFrame, kpis: dict[str, Any]) -> None:
    st.markdown("### Capacity and Liquidity")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(st, "Current Portfolio Size", _money(kpis.get("gross_portfolio_value")), "official gross value")
    with c2:
        metric_card(st, "Max ADV Participation", _pct(kpis.get("current_adv_participation")), "current orders")
    with c3:
        metric_card(st, "Capacity Status", str(kpis.get("capacity_status", "unavailable")), "safe/caution/capacity_limited")
    if capacity.empty:
        st.warning("Capacity table unavailable.")
        return
    fig = px.scatter(capacity, x="capital", y="max_participation", color="capacity_status", facet_col="participation_limit" if "participation_limit" in capacity.columns else None, color_discrete_sequence=EXEC_COLORS)
    st.plotly_chart(apply_plotly_layout(fig, "Capacity scenarios by participation limit"), width="stretch")
    _safe_df(st, capacity, height=360)


def _quality_and_lifecycle(st, bundle) -> None:
    st.markdown("### Execution Quality")
    alert_box(st, "Estimated execution quality — no live broker fills.", "info")
    _safe_df(st, bundle.quality)
    st.markdown("### Trade Lifecycle")
    _safe_df(st, bundle.lifecycle, height=360)
    if not bundle.lifecycle.empty and "ticker" in bundle.lifecycle.columns:
        value_col = "estimated_net_pnl" if "estimated_net_pnl" in bundle.lifecycle.columns else "unrealized_pnl"
        if value_col in bundle.lifecycle.columns:
            fig = px.bar(bundle.lifecycle, x="ticker", y=value_col, color="current_status" if "current_status" in bundle.lifecycle.columns else None, color_discrete_sequence=EXEC_COLORS)
            st.plotly_chart(apply_plotly_layout(fig, "Realized/unrealized contribution"), width="stretch")


def _reconciliation(st, bundle) -> None:
    st.markdown("### Reconciliation Control")
    statuses = bundle.reconciliation["status"].astype(str).value_counts().to_dict() if not bundle.reconciliation.empty else {}
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(st, "PASS", str(statuses.get("PASS", 0)), "controls")
    with c2:
        metric_card(st, "WARNING", str(statuses.get("WARNING", 0)), "controls")
    with c3:
        metric_card(st, "FAIL", str(statuses.get("FAIL", 0)), "controls")
    _safe_df(st, bundle.reconciliation)
    with st.expander("Execution terminal integrity"):
        _safe_df(st, bundle.integrity)
        _safe_df(st, bundle.source_audit)


def render_execution_terminal(st, data: dict[str, pd.DataFrame]) -> None:
    bundle = build_execution_bundle(data)
    _header(st, bundle.kpis, bundle.status)
    source_caption(st, "growth_official_* execution namespace", bundle.status)
    _kpi_cards(st, bundle)
    alert_box(st, bundle.commentary, "info")

    tabs = st.tabs([
        "Rebalance Summary",
        "Order Blotter",
        "Costs",
        "Gross vs Net",
        "Turnover",
        "Capacity",
        "Quality & Lifecycle",
        "Reconciliation",
        "Sources",
    ])
    with tabs[0]:
        _safe_df(st, bundle.rebalance_summary, height=360)
    with tabs[1]:
        _order_blotter(st, bundle.order_blotter)
    with tabs[2]:
        _cost_charts(st, bundle)
    with tabs[3]:
        _gross_vs_net(st, bundle.equity)
    with tabs[4]:
        _turnover(st, bundle.turnover)
    with tabs[5]:
        _capacity(st, bundle.capacity, bundle.kpis)
    with tabs[6]:
        _quality_and_lifecycle(st, bundle)
    with tabs[7]:
        _reconciliation(st, bundle)
    with tabs[8]:
        _safe_df(st, bundle.source_audit)
        _safe_df(st, bundle.integrity)
