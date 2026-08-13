
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_components import alert_box, fmt_money, fmt_num, fmt_pct, metric_card, section_header, source_caption, status_badge
from dashboard_data_layer import MODEL_VERSION, VARIANT, latest_market_date, next_rebalance_date, numeric
from dashboard_system_health import build_mission_control
from dashboard_theme import AMBER, BRIGHT_ORANGE, CHART_COLORS, GREEN, ORANGE, RED, apply_plotly_layout

STATUS_COLORS = {"PASS": GREEN, "WARNING": AMBER, "FAIL": RED, "BLOCKED": RED, "RUNNING": BRIGHT_ORANGE}


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif out[col].dtype == object:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    return out


def _plot(st, fig, height: int = 420, title: str | None = None) -> None:
    existing_title = getattr(getattr(fig, "layout", None), "title", None)
    existing_text = getattr(existing_title, "text", None) if existing_title is not None else None
    apply_plotly_layout(fig, title or existing_text or "")
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def _status_banner(st, bundle: dict[str, Any]) -> None:
    overall = bundle["overall_status"]
    health = bundle["health_pct"]
    color = {"GREEN": GREEN, "AMBER": AMBER, "RED": RED}.get(overall, AMBER)
    st.markdown(
        f"""
        <div class='hero' style='border-color:{color}; box-shadow: inset 5px 0 0 {color}, 0 18px 55px rgba(0,0,0,.30);'>
          <h1>Mission Control</h1>
          <p>Bloomberg + NASA style read-only monitoring center · {MODEL_VERSION} · {VARIANT}</p>
          <p>{status_badge(overall, overall)} <span class='small-muted'>Overall Health: <b>{health:.1f}%</b> · Latest market date: <b>{latest_market_date(bundle.get('data', {}))}</b> · Real capital blocked</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _live_grid(st, grid: pd.DataFrame) -> None:
    section_header(st, "Live Status Grid", "Current module status from official diagnostic files.")
    if grid.empty:
        alert_box(st, "Module grid unavailable.", "warning")
        return
    cols = st.columns(4)
    for i, (_, row) in enumerate(grid.iterrows()):
        with cols[i % 4]:
            metric_card(st, str(row.get("module", "module")), str(row.get("status", "WARNING")), f"latest {row.get('latest_date','')} · {row.get('source_file','')}", badge=str(row.get("status", "WARNING")))
    with st.expander("Full module status table"):
        st.dataframe(_safe_df(grid), width="stretch", hide_index=True)


def _timeline(st, timeline: pd.DataFrame) -> None:
    section_header(st, "Pipeline Execution Timeline", "Read-only file-modification timeline; no pipeline execution occurs here.")
    if timeline.empty:
        alert_box(st, "Pipeline timeline unavailable.", "warning")
        return
    fig = px.bar(timeline, x="step", y="rows", color="status", color_discrete_map=STATUS_COLORS, text="status", title="Pipeline Stages and Rows Available")
    _plot(st, fig, 390, "Pipeline Stages and Rows Available")
    st.dataframe(_safe_df(timeline), width="stretch", hide_index=True)


def _execution_perf(st, runtime: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    section_header(st, "Execution Performance", "Derived from files and row counts; duration is unavailable unless logged by the runner.")
    vals = {str(r.get("metric")): r.get("value") for _, r in runtime.iterrows()} if not runtime.empty else {}
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Pipeline Duration", str(vals.get("total_pipeline_duration", "n/a")))
    with cols[1]: metric_card(st, "Slowest Module Proxy", str(vals.get("slowest_module", "n/a")), "largest row count")
    with cols[2]: metric_card(st, "CSV Files", str(vals.get("csv_files_generated", "n/a")))
    with cols[3]: metric_card(st, "Rows Processed", str(vals.get("rows_processed", "n/a")))
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Holdings", str(vals.get("holdings", "n/a")))
    with cols[1]: metric_card(st, "Trades/Actions", str(vals.get("trades", "n/a")))
    with cols[2]: metric_card(st, "Market Rows", str(len(data.get("official_market_data_integrity", pd.DataFrame()))))
    with cols[3]: metric_card(st, "Forecast Rows", str(len(data.get("current_raw_target_features", pd.DataFrame()))))


def _market_portfolio_governance(st, bundle: dict[str, Any], data: dict[str, pd.DataFrame]) -> None:
    left, mid, right = st.columns(3)
    with left:
        section_header(st, "Market Status")
        md = data.get("official_market_data_integrity", pd.DataFrame())
        gov = data.get("official_market_data_governance", pd.DataFrame())
        latest_md = md.tail(1).iloc[-1] if not md.empty else pd.Series(dtype=object)
        latest_gov = gov.tail(1).iloc[-1] if not gov.empty else pd.Series(dtype=object)
        metric_card(st, "Latest Market Date", latest_market_date(data))
        metric_card(st, "Primary Source", "Yahoo/yfinance")
        metric_card(st, "Secondary Source", str(latest_gov.get("classification", latest_gov.get("governance", "single_source_warning"))))
        metric_card(st, "Data Quality", str(latest_md.get("status", latest_md.get("data_confidence", "available"))))
    with mid:
        section_header(st, "Portfolio Snapshot")
        perf = bundle["performance"].iloc[-1] if not bundle["performance"].empty else pd.Series(dtype=object)
        state = bundle["state"]
        holdings = state[~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")]["ticker"].astype(str).tolist() if not state.empty and "ticker" in state.columns else []
        metric_card(st, "Portfolio Value", fmt_money(perf.get("gross_portfolio_value", perf.get("portfolio_value", np.nan))))
        metric_card(st, "Net Value", fmt_money(perf.get("estimated_net_portfolio_value", np.nan)))
        metric_card(st, "Exposure / Cash", f"{fmt_pct(perf.get('exposure', np.nan))} / {fmt_pct(perf.get('cash_weight', perf.get('cash', np.nan)))}")
        metric_card(st, "Current Holdings", ", ".join(holdings) if holdings else "none")
    with right:
        section_header(st, "Governance Snapshot")
        mon = bundle["monitor"].iloc[-1] if not bundle["monitor"].empty else pd.Series(dtype=object)
        metric_card(st, "Technical Integrity", str(mon.get("integrity_status", "unavailable")))
        metric_card(st, "Accounting", str(mon.get("accounting_status", "unavailable")))
        metric_card(st, "Research/PBO", str(mon.get("research_status", mon.get("risk_flags", "unavailable"))))
        metric_card(st, "Real Capital", str(mon.get("promotion_status", "real_capital_blocked")))


def _alerts_incidents(st, bundle: dict[str, Any]) -> None:
    section_header(st, "Incident and Alert Center", "Persistent warnings, fixes, backups and blocking conditions.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Alerts")
        alerts = bundle["alerts"]
        if alerts.empty:
            alert_box(st, "No active alerts found in loaded diagnostics.", "success")
        else:
            st.dataframe(_safe_df(alerts), width="stretch", hide_index=True)
    with right:
        st.markdown("#### Incidents / Repairs")
        incidents = bundle["incidents"]
        if incidents.empty:
            alert_box(st, "No incident registry rows available.", "info")
        else:
            st.dataframe(_safe_df(incidents), width="stretch", hide_index=True)


def _next_actions_summary(st, bundle: dict[str, Any]) -> None:
    section_header(st, "Next Actions and Executive Summary", "Deterministic summary. No recommendation engine.")
    cols = st.columns(5)
    with cols[0]: metric_card(st, "Next Rebalance", next_rebalance_date(bundle.get("data", {})))
    with cols[1]: metric_card(st, "Forward History", "collecting")
    with cols[2]: metric_card(st, "Market", "fresh")
    with cols[3]: metric_card(st, "Governance", "blocked by research/paper gates")
    with cols[4]: metric_card(st, "Accounting", "healthy" if "FAIL" not in str(bundle["grid"].to_string()) else "check required")
    alert_box(st, bundle["summary"], "info")


def _system_map_resources_checklist(st, bundle: dict[str, Any]) -> None:
    section_header(st, "System Map")
    grid = bundle["grid"]
    if not grid.empty:
        nodes = grid["module"].astype(str).tolist()
        colors = [STATUS_COLORS.get(s, AMBER) for s in grid["status"].astype(str)]
        x = list(range(len(nodes)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[0]*len(nodes), mode="markers+text", marker=dict(size=34, color=colors), text=nodes, textposition="bottom center", hovertext=grid["status"].astype(str), hoverinfo="text"))
        for i in range(len(nodes)-1):
            fig.add_shape(type="line", x0=i, y0=0, x1=i+1, y1=0, line=dict(color="rgba(255,138,42,0.35)", width=2))
        fig.update_yaxes(visible=False); fig.update_xaxes(visible=False)
        _plot(st, fig, 260, "System Dependency Map")
    left, right = st.columns(2)
    with left:
        st.markdown("#### System Resources")
        st.dataframe(_safe_df(bundle["resources"]), width="stretch", hide_index=True)
    with right:
        st.markdown("#### Daily Checklist")
        checklist = bundle["grid"][["module", "status", "latest_date"]].copy() if not bundle["grid"].empty else pd.DataFrame()
        st.dataframe(_safe_df(checklist), width="stretch", hide_index=True)


def render_mission_control(st, data: dict[str, pd.DataFrame]) -> None:
    bundle = build_mission_control(data)
    bundle["data"] = data
    _status_banner(st, bundle)
    _live_grid(st, bundle["grid"])
    _timeline(st, bundle["timeline"])
    _execution_perf(st, bundle["runtime"], data)
    _market_portfolio_governance(st, bundle, data)
    _alerts_incidents(st, bundle)
    _next_actions_summary(st, bundle)
    _system_map_resources_checklist(st, bundle)
    source_caption(st, "official diagnostic CSVs + dashboard_system_health.py", "read-only")
