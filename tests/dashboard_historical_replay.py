
from __future__ import annotations

import numpy as np
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
    status_badge,
)
from dashboard_historical_loader import load_replay_data, nearest_replay_date
from dashboard_replay_engine import (
    build_snapshot,
    compare_snapshots,
    performance_evolution,
    validate_replay,
)
from dashboard_theme import (
    AMBER,
    BRIGHT_ORANGE,
    GREEN,
    RED,
    apply_plotly_layout,
)

COLOR_MAP = {"BUY": GREEN, "INCREASE": BRIGHT_ORANGE, "REDUCE": AMBER, "SELL": RED, "HOLD": "#7A8490", "CASH_CHANGE": "#4B5563"}


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


def _chart(st, fig: go.Figure, title: str, height: int = 420) -> None:
    apply_plotly_layout(fig, title=title)
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def _ss_get(st, key: str, default=None):
    try:
        return st.session_state.get(key, default)
    except AttributeError:
        return getattr(st.session_state, key, default)


def _ss_set(st, key: str, value) -> None:
    try:
        st.session_state[key] = value
    except TypeError:
        setattr(st.session_state, key, value)


def _init_state(st, dates: list[pd.Timestamp]) -> int:
    if _ss_get(st, "historical_replay_idx") is None:
        _ss_set(st, "historical_replay_idx", len(dates) - 1 if dates else 0)
    idx = max(0, min(int(_ss_get(st, "historical_replay_idx", 0)), max(len(dates) - 1, 0)))
    _ss_set(st, "historical_replay_idx", idx)
    return idx


def _time_machine(st, dates: list[pd.Timestamp]) -> pd.Timestamp | None:
    if not dates:
        alert_box(st, "Historical replay unavailable: no official replay dates found.", "warning")
        return None
    idx = _init_state(st, dates)
    section_header(st, "Global Time Machine", "Official Historical State Reconstruction")
    cols = st.columns([1, 1, 1, 1, 1, 1.4])
    with cols[0]:
        if st.button("Previous", key="replay_prev"):
            _ss_set(st, "historical_replay_idx", max(0, idx - 1))
    with cols[1]:
        if st.button("Next", key="replay_next"):
            _ss_set(st, "historical_replay_idx", min(len(dates) - 1, idx + 1))
    with cols[2]:
        if st.button("Play", key="replay_play"):
            _ss_set(st, "historical_replay_play", True)
    with cols[3]:
        if st.button("Pause", key="replay_pause"):
            _ss_set(st, "historical_replay_play", False)
    with cols[4]:
        speed = st.selectbox("Speed", ["1x", "2x", "5x", "20x"], index=0, key="replay_speed")
    with cols[5]:
        mode = st.selectbox("Jump mode", ["exact replay date", "calendar", "month", "year"], index=0, key="replay_jump_mode")
    if _ss_get(st, "historical_replay_play", False):
        step = {"1x": 1, "2x": 1, "5x": 1, "20x": 1}.get(speed, 1)
        _ss_set(st, "historical_replay_idx", min(len(dates) - 1, int(_ss_get(st, "historical_replay_idx", 0)) + step))
    idx = st.slider("Replay timeline", min_value=0, max_value=len(dates) - 1, value=int(_ss_get(st, "historical_replay_idx", 0)), format="%d", key="replay_slider")
    _ss_set(st, "historical_replay_idx", idx)
    selected = dates[idx]
    if mode == "calendar":
        requested = st.date_input("Calendar selector", selected.date(), key="replay_calendar")
        selected = nearest_replay_date(dates, requested) or selected
        st.caption(f"Nearest official replay date: {selected.date()}")
    elif mode == "month":
        months = sorted({d.strftime("%Y-%m") for d in dates})
        month = st.selectbox("Month", months, index=months.index(selected.strftime("%Y-%m")) if selected.strftime("%Y-%m") in months else len(months)-1, key="replay_month")
        selected = [d for d in dates if d.strftime("%Y-%m") == month][-1]
    elif mode == "year":
        years = sorted({d.year for d in dates})
        year = st.selectbox("Year", years, index=years.index(selected.year), key="replay_year")
        selected = [d for d in dates if d.year == year][-1]
    events = {"COVID Crash": "2020-03-23", "2022 Bear Market": "2022-10-12", "AI Rally": "2024-05-01", "Official Start": str(dates[0].date())}
    jump = st.selectbox("Historical event jump", ["none"] + list(events.keys()), key="replay_event_jump")
    if jump != "none":
        selected = nearest_replay_date(dates, events[jump]) or selected
        st.caption(f"Event mapped to official replay date: {selected.date()}")
    return selected


def _snapshot_cards(st, snap) -> None:
    section_header(st, "Official Snapshot", "Values are read from official historical paper files for selected date.")
    p, r, g = snap.performance, snap.risk, snap.governance
    cols = st.columns(5)
    with cols[0]: metric_card(st, "Portfolio Value", fmt_money(p.get("gross_portfolio_value", p.get("portfolio_value", np.nan))))
    with cols[1]: metric_card(st, "Net Portfolio Value", fmt_money(p.get("estimated_net_portfolio_value", np.nan)))
    with cols[2]: metric_card(st, "Cash", fmt_pct(r.get("cash", p.get("cash_weight", np.nan))))
    with cols[3]: metric_card(st, "Exposure", fmt_pct(r.get("exposure", p.get("exposure", np.nan))))
    with cols[4]: metric_card(st, "Drawdown", fmt_pct(r.get("drawdown", p.get("current_drawdown", np.nan))))
    cols = st.columns(5)
    with cols[0]: metric_card(st, "Gross Return", fmt_pct(p.get("gross_daily_return", p.get("daily_return", np.nan))))
    with cols[1]: metric_card(st, "Net Return", fmt_pct(p.get("estimated_net_daily_return", np.nan)))
    with cols[2]: metric_card(st, "Volatility", fmt_pct(r.get("volatility", np.nan)))
    with cols[3]: metric_card(st, "HHI", fmt_num(r.get("hhi", np.nan), 3))
    with cols[4]: metric_card(st, "Governance", str(g.get("paper_status", "unavailable")), f"integrity {g.get('integrity_status', 'unavailable')}")


def _holdings(st, snap) -> None:
    section_header(st, "Historical Holdings", "Position state for the selected official date.")
    h = snap.holdings.copy()
    if h.empty:
        alert_box(st, "Historical holdings unavailable.", "warning")
        return
    cols = ["ticker", "action", "paper_position_weight", "entry_date", "days_held", "daily_pnl", "unrealized_pnl", "raw_target_return_exact", "pct_total_portfolio_risk", "sector", "industry", "country", "market_cap", "holding_quality_classification"]
    if "entry_date" in h.columns and "date" in h.columns:
        h["days_held"] = (pd.to_datetime(h["date"], errors="coerce") - pd.to_datetime(h["entry_date"], errors="coerce")).dt.days
    elif "entry_date" not in h.columns:
        h["entry_date"] = "historical entry unavailable"
        h["days_held"] = np.nan
    st.dataframe(_safe_df(h[[c for c in cols if c in h.columns]]), width="stretch", hide_index=True)


def _rebalance(st, snap) -> None:
    section_header(st, "Rebalance Comparison", "Before/after available only on stored official rebalance dates.")
    if snap.rebalance.empty:
        alert_box(st, "Selected day is not a stored rebalance report date or report unavailable.", "info")
        return
    st.dataframe(_safe_df(snap.rebalance), width="stretch", hide_index=True)
    acts = snap.actions.copy()
    if not acts.empty:
        counts = acts["action"].astype(str).value_counts().reset_index()
        counts.columns = ["action", "count"]
        fig = px.bar(counts, x="action", y="count", color="action", color_discrete_map=COLOR_MAP, title="Rebalance Action Counts")
        _chart(st, fig, "Rebalance Action Counts", 330)


def _decision_replay(st, snap) -> None:
    section_header(st, "Decision Replay", "Historical funnel uses stored official history only.")
    st.dataframe(_safe_df(snap.decision_funnel), width="stretch", hide_index=True)
    fig = px.funnel(snap.decision_funnel.fillna({"count": 0}), y="stage", x="count", title="Historical Decision Funnel")
    _chart(st, fig, "Historical Decision Funnel", 360)
    alert_box(st, "If historical feature snapshots are unavailable, the replay shows stored selected holdings only and does not fabricate universe/ranking counts.", "warning")


def _risk_execution_governance(st, snap) -> None:
    tabs = st.tabs(["Historical Risk", "Historical Execution", "Historical Governance", "Historical Research", "Market Regime"])
    with tabs[0]:
        st.dataframe(_safe_df(pd.DataFrame([snap.risk])), width="stretch", hide_index=True)
    with tabs[1]:
        st.dataframe(_safe_df(pd.DataFrame([snap.execution])), width="stretch", hide_index=True)
        if not snap.actions.empty:
            st.dataframe(_safe_df(snap.actions), width="stretch", hide_index=True)
    with tabs[2]:
        st.dataframe(_safe_df(pd.DataFrame([snap.governance])), width="stretch", hide_index=True)
    with tabs[3]:
        st.dataframe(_safe_df(pd.DataFrame([snap.research])), width="stretch", hide_index=True)
    with tabs[4]:
        st.dataframe(_safe_df(pd.DataFrame([snap.regime])), width="stretch", hide_index=True)


def _performance(st, replay, snap) -> None:
    section_header(st, "Performance Evolution", "Synchronized official series up to selected date.")
    perf = performance_evolution(replay, snap.date)
    if perf.empty:
        alert_box(st, "Performance evolution unavailable.", "warning")
        return
    fig = go.Figure()
    if "gross_equity" in perf.columns:
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["gross_equity"], mode="lines+markers", name="Portfolio Gross"))
    if "estimated_net_equity" in perf.columns:
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["estimated_net_equity"], mode="lines+markers", name="Portfolio Net"))
    for col in ["SPY_cumulative_pct", "QQQ_cumulative_pct"]:
        if col in perf.columns:
            base = snap.performance.get("gross_equity", snap.performance.get("portfolio_value", 100000))
            fig.add_trace(go.Scatter(x=perf["date"], y=100000 * (1 + pd.to_numeric(perf[col], errors="coerce") / 100), mode="lines+markers", name=col.replace("_cumulative_pct", "")))
    _chart(st, fig, "Portfolio vs Benchmark Replay", 430)
    metrics = [c for c in ["cash_weight", "exposure", "current_drawdown", "volatility", "turnover"] if c in perf.columns]
    if metrics:
        m = perf.melt(id_vars="date", value_vars=metrics, var_name="metric", value_name="value")
        fig = px.line(m, x="date", y="value", color="metric", markers=True, title="Cash / Exposure / Risk Evolution")
        _chart(st, fig, "Cash / Exposure / Risk Evolution", 390)


def _comparison(st, replay, dates: list[pd.Timestamp]) -> None:
    section_header(st, "Date Comparison Mode", "Official snapshot A vs B.")
    if len(dates) < 2:
        alert_box(st, "Need at least two official dates for comparison.", "info")
        return
    cols = st.columns(2)
    with cols[0]:
        da = st.selectbox("Date A", dates, index=0, format_func=lambda x: str(x.date()), key="replay_date_a")
    with cols[1]:
        db = st.selectbox("Date B", dates, index=len(dates)-1, format_func=lambda x: str(x.date()), key="replay_date_b")
    a, b = build_snapshot(replay, da), build_snapshot(replay, db)
    st.dataframe(_safe_df(compare_snapshots(a, b)), width="stretch", hide_index=True)
    left = set(a.holdings.get("ticker", pd.Series(dtype=str)).astype(str)) - {"CASH"}
    right = set(b.holdings.get("ticker", pd.Series(dtype=str)).astype(str)) - {"CASH"}
    st.dataframe(pd.DataFrame({"added": [", ".join(sorted(right-left))], "removed": [", ".join(sorted(left-right))], "unchanged": [", ".join(sorted(left & right))]}), width="stretch", hide_index=True)


def render_historical_replay(st, data: dict[str, pd.DataFrame]) -> None:
    replay = load_replay_data()
    section_header(st, "Historical Replay", "Official Historical State Reconstruction", badge="READ ONLY")
    source_caption(st, "official forward paper namespace", "official only")
    selected_date = _time_machine(st, replay.dates)
    if selected_date is None:
        return
    snap = build_snapshot(replay, selected_date)
    if snap.warnings:
        alert_box(st, "; ".join(snap.warnings), "warning")
    _snapshot_cards(st, snap)
    _holdings(st, snap)
    _rebalance(st, snap)
    _decision_replay(st, snap)
    _risk_execution_governance(st, snap)
    _performance(st, replay, snap)
    _comparison(st, replay, replay.dates)
    with st.expander("Replay Source Audit"):
        st.dataframe(_safe_df(replay.source_audit), width="stretch", hide_index=True)
        validation, status = validate_replay(replay)
        st.dataframe(_safe_df(validation), width="stretch", hide_index=True)
        st.caption(f"Replay validation status: {status_badge(status, status)}", unsafe_allow_html=True)
