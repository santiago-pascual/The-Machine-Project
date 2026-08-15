
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_components import (
    alert_box,
    metric_card,
    section_header,
    source_caption,
)
from dashboard_governance_calculations import build_governance_bundle
from dashboard_theme import (
    AMBER,
    BRIGHT_ORANGE,
    CYAN,
    GREEN,
    INFO,
    MUTED_ORANGE,
    ORANGE,
    RED,
    apply_plotly_layout,
)

GOV_COLORS = [ORANGE, BRIGHT_ORANGE, AMBER, MUTED_ORANGE, INFO, CYAN, GREEN, RED]
STATUS_COLORS = {"PASS": GREEN, "WARNING": AMBER, "BLOCKED": RED, "FAIL": RED, "WARMUP": INFO, "NOT AVAILABLE": "#6B7280"}


def _arrow_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_object_dtype(s):
            sample = s.dropna().head(200).tolist()
            types = {type(v).__name__ for v in sample}
            if len(types) > 1 or any(t in types for t in ("str", "dict", "list", "tuple")):
                out[col] = s.map(lambda v: "" if pd.isna(v) else str(v))
    return out


def _safe_df(st, df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    if df.empty:
        st.info("No data available for this section.")
        return
    view = df.copy()
    if columns:
        keep = [c for c in columns if c in view.columns]
        view = view[keep] if keep else view
    kwargs = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(_arrow_safe_frame(view), **kwargs)


def _header(st, bundle) -> None:
    k = bundle.kpis
    section_header(st, "Governance Terminal", "Model integrity, operational readiness, data quality, accounting, research gates and incident management. Read-only.", bundle.status)
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Active Version", str(k.get("active_model_version")), str(k.get("lifecycle_classification")), badge="frozen")
    with cols[1]: metric_card(st, "Official Date", str(k.get("latest_official_date")), f"last run {k.get('latest_successful_run')}", badge="official")
    with cols[2]: metric_card(st, "Integrity / Market", f"{k.get('integrity_status')} / {k.get('market_data_status')}", "official governance", badge="paper")
    with cols[3]: metric_card(st, "Real Capital", str(k.get("real_capital_status")), str(k.get("broker_orders_status")), state="warning", badge="blocked")


def _scoreboard(st, bundle) -> None:
    st.markdown("### Overall Governance Scoreboard")
    df = bundle.scoreboard.copy()
    if not df.empty:
        counts = df["status"].value_counts().reset_index(); counts.columns=["status","count"]
        fig = px.bar(counts, x="status", y="count", color="status", color_discrete_map=STATUS_COLORS)
        st.plotly_chart(apply_plotly_layout(fig, "Control status distribution"), width="stretch")
    _safe_df(st, df, height=460)


def _real_capital(st, bundle) -> None:
    st.markdown("### Real Capital Readiness")
    gates = bundle.real_capital_gates.copy()
    passed = int(gates.get("passed", pd.Series(dtype=bool)).fillna(False).sum()) if not gates.empty else 0
    total = len(gates)
    c1,c2,c3 = st.columns(3)
    with c1: metric_card(st,"Passed Gates", f"{passed}/{total}", "configured governance gates")
    with c2: metric_card(st,"Classification", "REAL CAPITAL BLOCKED", "no promotion automation", state="warning")
    with c3: metric_card(st,"Broker Orders", "DISABLED", "read-only dashboard", state="neutral")
    _safe_df(st, gates, height=420)


def _lifecycle_version(st, bundle) -> None:
    st.markdown("### Model Lifecycle")
    life=bundle.lifecycle
    if not life.empty:
        fig=go.Figure(go.Scatter(x=list(range(len(life))), y=[1]*len(life), mode="lines+markers+text", text=life["stage"], textposition="top center", marker=dict(size=[22 if x else 14 for x in life["current"]], color=[ORANGE if x else "#6B7280" for x in life["current"]])))
        fig.update_yaxes(visible=False)
        st.plotly_chart(apply_plotly_layout(fig,"Lifecycle timeline"), width="stretch")
    _safe_df(st, life)
    st.markdown("### Version Control and Freeze")
    _safe_df(st, bundle.version, height=360)


def _pipeline(st, bundle) -> None:
    st.markdown("### Daily Pipeline Status")
    pipe=bundle.pipeline
    if not pipe.empty:
        fig=go.Figure()
        xs=list(range(len(pipe)))
        fig.add_trace(go.Scatter(x=xs,y=[1]*len(pipe),mode="lines",line=dict(color="rgba(255,122,0,.35)",width=3),showlegend=False))
        fig.add_trace(go.Scatter(x=xs,y=[1]*len(pipe),mode="markers+text",text=pipe["node"],textposition="top center",marker=dict(size=22,color=[STATUS_COLORS.get(s,"#6B7280") for s in pipe["status"]]),showlegend=False))
        fig.update_yaxes(visible=False)
        fig.update_xaxes(visible=False)
        st.plotly_chart(apply_plotly_layout(fig,"Market Data -> Integrity pipeline"), width="stretch")
    _safe_df(st, pipe, height=420)


def _freshness(st, bundle) -> None:
    st.markdown("### Data Freshness and Provenance")
    _safe_df(st, bundle.freshness)
    alert_box(st,"Primary provider is Yahoo/cache. Secondary provider absence blocks real capital but does not block official paper when Yahoo is fresh.","warning")


def _incidents_backups(st, bundle) -> None:
    st.markdown("### Incident and Repair Log")
    _safe_df(st, bundle.incidents, height=420)
    st.markdown("### Backup and Replay Control")
    _safe_df(st, bundle.backups, height=260)


def _accounting_research(st, bundle) -> None:
    st.markdown("### Accounting Governance")
    _safe_df(st, bundle.accounting)
    st.markdown("### Research Governance")
    _safe_df(st, bundle.research)
    alert_box(st,"Strong DSR and strong OOS do not override elevated CSCV PBO automatically; all gates remain visible.","warning")


def _operational_warnings(st, bundle) -> None:
    st.markdown("### Operational Risk")
    _safe_df(st, bundle.operational_risk)
    st.markdown("### Persistent Governance Warnings")
    _safe_df(st, bundle.warnings)
    alert_box(st,bundle.commentary,"warning")


def _history_sources(st, bundle) -> None:
    st.markdown("### Governance History")
    hist=bundle.history
    if not hist.empty and "date" in hist.columns and "current_status" in hist.columns:
        fig=px.scatter(hist,x="date",y="current_status",color="promotion_status" if "promotion_status" in hist.columns else None,color_discrete_sequence=GOV_COLORS)
        st.plotly_chart(apply_plotly_layout(fig,"Governance status through time"), width="stretch")
    _safe_df(st,hist,height=340)
    st.markdown("### Source Audit")
    _safe_df(st,bundle.source_audit,height=420)
    st.markdown("### Terminal Integrity")
    _safe_df(st,bundle.integrity)
    st.markdown("### Gate Reconciliation")
    _safe_df(st,bundle.gate_reconciliation)


def render_governance_terminal(st, data: dict[str, pd.DataFrame]) -> None:
    bundle=build_governance_bundle(data)
    _header(st,bundle)
    source_caption(st,"governance, official paper, accounting, market data and research governance CSVs",bundle.status)
    _operational_warnings(st,bundle)
    tabs=st.tabs(["Scoreboard","Real Capital","Lifecycle & Freeze","Pipeline","Freshness","Incidents & Backups","Accounting & Research","Sources"])
    with tabs[0]: _scoreboard(st,bundle)
    with tabs[1]: _real_capital(st,bundle)
    with tabs[2]: _lifecycle_version(st,bundle)
    with tabs[3]: _pipeline(st,bundle)
    with tabs[4]: _freshness(st,bundle)
    with tabs[5]: _incidents_backups(st,bundle)
    with tabs[6]: _accounting_research(st,bundle)
    with tabs[7]: _history_sources(st,bundle)
