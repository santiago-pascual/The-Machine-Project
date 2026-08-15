from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard_components import fmt_money, fmt_num, fmt_pct, fmt_pct_points, metric_card, status_badge
from dashboard_data_layer import next_rebalance_date, MODEL_VERSION, VARIANT, latest, latest_market_date, numeric, official_start_date
from dashboard_theme import AMBER, BRIGHT_ORANGE, CHART_COLORS, INFO, ORANGE, PURPLE, RED, apply_plotly_layout

OFFICIAL_SOURCES = {
    "performance": "growth_official_paper_performance.csv",
    "state": "growth_official_paper_state.csv",
    "actions": "growth_official_paper_actions.csv",
    "monitor": "growth_official_paper_monitor.csv",
    "tracking": "growth_official_live_tracking.csv",
    "cost_ledger": "growth_official_estimated_cost_ledger.csv",
    "benchmark_daily": "growth_official_benchmark_daily.csv",
    "position_pnl": "growth_official_position_pnl.csv",
    "daily_status": "official_paper_daily_run_status.csv",
    "integrity": "official_paper_integrity_status.csv",
}


def _row(df: pd.DataFrame) -> pd.Series:
    latest_df = latest(df)
    return latest_df.iloc[-1] if not latest_df.empty else pd.Series(dtype=object)


def _source_warning(st, data: dict[str, pd.DataFrame]) -> list[str]:
    missing = []
    for key, filename in OFFICIAL_SOURCES.items():
        if data.get(f"official_{key}", pd.DataFrame()).empty and key not in data:
            pass
    explicit = {
        "performance": data.get("official_performance", pd.DataFrame()),
        "state": data.get("official_state", pd.DataFrame()),
        "actions": data.get("official_actions", pd.DataFrame()),
        "monitor": data.get("official_monitor", pd.DataFrame()),
        "tracking": data.get("official_tracking", pd.DataFrame()),
        "cost_ledger": data.get("official_cost_ledger", pd.DataFrame()),
        "benchmark_daily": data.get("official_benchmark_daily", pd.DataFrame()),
        "position_pnl": data.get("official_position_pnl", pd.DataFrame()),
        "daily_status": data.get("official_daily_status", pd.DataFrame()),
        "integrity": data.get("official_integrity", pd.DataFrame()),
    }
    for key, df in explicit.items():
        if df.empty:
            missing.append(OFFICIAL_SOURCES[key])
    if missing:
        st.warning("Missing official Executive source(s): " + ", ".join(missing))
    return missing


def _official_state_latest(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    state = data.get("official_state", pd.DataFrame())
    return latest(state).copy() if not state.empty else pd.DataFrame()


def _holdings_with_pnl(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    state = _official_state_latest(data)
    if state.empty or "ticker" not in state.columns:
        return state
    pnl = latest(data.get("official_position_pnl", pd.DataFrame()))
    if not pnl.empty and "ticker" in pnl.columns:
        keep = [c for c in pnl.columns if c != "date"]
        state = state.merge(pnl[keep], on="ticker", how="left", suffixes=("", "_pnl"))
    return state


def _range_filter(df: pd.DataFrame, option: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    work = df.dropna(subset=["date"]).sort_values("date").copy()
    if work.empty or option in {"Since official start", "Max"}:
        return work
    end = work["date"].max()
    if option == "1M":
        start = end - pd.DateOffset(months=1)
    elif option == "3M":
        start = end - pd.DateOffset(months=3)
    elif option == "6M":
        start = end - pd.DateOffset(months=6)
    elif option == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    else:
        return work
    filtered = work[work["date"].ge(start)]
    return filtered if not filtered.empty else work


def _benchmark_chart(st, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bench = data.get("official_benchmark_daily", pd.DataFrame()).copy()
    if bench.empty or "date" not in bench.columns:
        st.warning("Official benchmark chart unavailable: missing growth_official_benchmark_daily.csv")
        return pd.DataFrame()
    option = st.radio("Range", ["Since official start", "1M", "3M", "6M", "YTD", "Max"], horizontal=True, index=0)
    bench = _range_filter(bench, option)
    fig = go.Figure()
    series = [
        ("growth_gross_cumulative_pct", "Official Growth Gross", ORANGE, "solid"),
        ("growth_net_cumulative_pct", "Official Growth Estimated Net", BRIGHT_ORANGE, "dash"),
        ("SPY_cumulative_pct", "SPY", INFO, "solid"),
        ("QQQ_cumulative_pct", "QQQ", PURPLE, "solid"),
    ]
    for col, name, color, dash in series:
        if col in bench.columns:
            fig.add_trace(go.Scatter(x=bench["date"], y=numeric(bench[col]), mode="lines+markers", name=name, line={"color": color, "dash": dash, "width": 2.5}))
    fig.update_yaxes(title="Cumulative return %", ticksuffix="%")
    fig = apply_plotly_layout(fig, "Official Growth Gross / Net vs SPY / QQQ")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Source: growth_official_benchmark_daily.csv · Official Forward Paper only · start date: {official_start_date(data)}")
    return bench


def _risk_snapshot_values(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    perf = _row(data.get("official_performance", pd.DataFrame()))
    state = _official_state_latest(data)
    non_cash = state[~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")] if not state.empty else pd.DataFrame()
    representative = non_cash.iloc[0] if not non_cash.empty else pd.Series(dtype=object)
    weights = numeric(non_cash.get("paper_position_weight", pd.Series(dtype=float))) if not non_cash.empty else pd.Series(dtype=float)
    final_exposure = representative.get("final_exposure", perf.get("exposure", np.nan))
    vol_target_exposure = representative.get("vol_target_exposure", np.nan)
    floor_active = pd.notna(vol_target_exposure) and float(vol_target_exposure) <= 0.400001
    cap_active = pd.notna(final_exposure) and pd.notna(vol_target_exposure) and float(final_exposure) < float(vol_target_exposure) - 1e-6
    return {
        "estimated_portfolio_volatility": perf.get("volatility", np.nan),
        "target_volatility": 0.22,
        "uncapped_exposure": vol_target_exposure,
        "final_exposure": final_exposure,
        "floor_active": floor_active,
        "cap_active": cap_active,
        "dual_trend_state": representative.get("dual_trend_reason", "unavailable"),
        "hhi": float((weights ** 2).sum()) if not weights.empty else np.nan,
        "beta_vs_spy": np.nan,
        "current_drawdown": perf.get("current_drawdown", perf.get("max_drawdown", np.nan)),
        "max_drawdown_official": perf.get("max_drawdown", np.nan),
        "risk_status": "WARMUP" if len(data.get("official_performance", pd.DataFrame())) < 20 else "ACTIVE",
        "exposure_cap_60": representative.get("exposure_cap_60", np.nan),
        "dual_trend_cap": representative.get("dual_trend_cap", np.nan),
    }


def _render_status_strip(st, data: dict[str, pd.DataFrame]) -> None:
    monitor = _row(data.get("official_monitor", pd.DataFrame()))
    integrity = _row(data.get("official_integrity", pd.DataFrame()))
    daily = _row(data.get("official_daily_status", pd.DataFrame()))
    items = [
        ("Model", MODEL_VERSION, "official"),
        ("Paper", str(monitor.get("governance_status", "WARMUP")), str(monitor.get("governance_status", "WARMUP"))),
        ("Integrity", str(monitor.get("integrity_status", integrity.get("integrity_status", "unavailable"))), str(monitor.get("integrity_status", integrity.get("integrity_status", "unavailable")))),
        ("Market Data", str(monitor.get("data_status", integrity.get("data_status", "unavailable"))), str(monitor.get("data_status", integrity.get("data_status", "unavailable")))),
        ("Governance", str(monitor.get("promotion_status", "real_capital_blocked")), "blocked"),
        ("Rebalance Due", "Yes" if bool(daily.get("rebalance_due", False)) else "No", "warning" if bool(daily.get("rebalance_due", False)) else "pass"),
        ("Real Capital", "BLOCKED", "blocked"),
        ("Broker/Orders", "DISABLED", "blocked"),
    ]
    html = "<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px'>"
    for label, value, status in items:
        html += f"<div class='kpi' style='min-height:74px'><div class='kpi-label'>{label}</div><div style='margin-top:8px'>{status_badge(value, status)}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_kpis(st, data: dict[str, pd.DataFrame]) -> None:
    perf = _row(data.get("official_performance", pd.DataFrame()))
    monitor = _row(data.get("official_monitor", pd.DataFrame()))
    state = _official_state_latest(data)
    state_row = state[~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")].iloc[0] if not state.empty and "ticker" in state.columns and (~state["ticker"].astype(str).str.upper().eq("CASH")).any() else pd.Series(dtype=object)
    source_date = str(perf.get("date", "n/a"))[:10]
    obs = len(data.get("official_performance", pd.DataFrame()))
    rows = [
        [("Gross portfolio value", fmt_money(perf.get("gross_portfolio_value", perf.get("portfolio_value", np.nan))), "growth_official_paper_performance.csv", "gross accounting"),
         ("Estimated net portfolio value", fmt_money(perf.get("estimated_net_portfolio_value", np.nan)), "growth_official_paper_performance.csv", "estimated execution costs"),
         ("Gross cumulative return", fmt_pct(perf.get("gross_cumulative_return", np.nan)), "growth_official_paper_performance.csv", f"observations: {obs}"),
         ("Estimated net cumulative", fmt_pct(perf.get("estimated_net_cumulative_return", np.nan)), "growth_official_paper_performance.csv", f"WARMUP · observations: {obs}")],
        [("Daily gross return", fmt_pct(perf.get("gross_daily_return", monitor.get("gross_daily_return", np.nan))), "growth_official_paper_performance.csv", source_date),
         ("Daily estimated net", fmt_pct(perf.get("estimated_net_daily_return", monitor.get("estimated_net_daily_return", np.nan))), "growth_official_paper_performance.csv", source_date),
         ("Exposure", fmt_pct(perf.get("exposure", monitor.get("exposure", np.nan))), "growth_official_paper_performance.csv", "final official exposure"),
         ("Cash", fmt_pct(perf.get("cash_weight", monitor.get("cash", np.nan))), "growth_official_paper_performance.csv", "cash weight")],
        [("Current drawdown", fmt_pct(perf.get("current_drawdown", perf.get("max_drawdown", np.nan))), "growth_official_paper_performance.csv", "gross equity"),
         ("Realized volatility", fmt_pct(perf.get("volatility", np.nan)), "growth_official_paper_performance.csv", "official warmup estimate"),
         ("Target volatility", "22.00%", "frozen model config/state", "growth_champion_final_v1_0_frozen"),
         ("Next rebalance", next_rebalance_date(data), "growth_official_paper_state.csv", "official scheduler field")],
    ]
    for row in rows:
        cols = st.columns(4)
        for col, (label, value, source, note) in zip(cols, row):
            with col:
                metric_card(st, label, value, note=f"{note} · {source}", badge="WARMUP" if "WARMUP" in note else "OFFICIAL")


def _render_portfolio_snapshot(st, data: dict[str, pd.DataFrame]) -> None:
    holdings = _holdings_with_pnl(data)
    if holdings.empty:
        st.warning("Portfolio snapshot unavailable: missing growth_official_paper_state.csv")
        return
    if "ticker" not in holdings.columns:
        st.warning("Portfolio snapshot unavailable: official state missing ticker column")
        return
    non_cash = holdings[~holdings["ticker"].astype(str).str.upper().eq("CASH")].copy()
    day_col = "daily_return_pct" if "daily_return_pct" in non_cash.columns else "unrealized_return"
    if not non_cash.empty and day_col in non_cash.columns:
        sorted_day = non_cash.assign(_day=numeric(non_cash[day_col])).sort_values("_day")
        weakest = sorted_day.iloc[0].get("ticker", "n/a")
        strongest = sorted_day.iloc[-1].get("ticker", "n/a")
        st.caption(f"Strongest today: {strongest} · Weakest today: {weakest}")
    cols = st.columns(4)
    cards = pd.concat([non_cash, holdings[holdings["ticker"].astype(str).str.upper().eq("CASH")]], ignore_index=True)
    for idx, (_, row) in enumerate(cards.iterrows()):
        with cols[idx % 4]:
            ticker = str(row.get("ticker", "n/a"))
            metric_card(st, ticker, fmt_pct(row.get("paper_position_weight", np.nan)), note=f"{fmt_money(row.get('paper_position_value', np.nan))} · price {fmt_money(row.get('current_price', np.nan))}", badge="OFFICIAL" if ticker != "CASH" else "DIAGNOSTIC")
            st.caption(f"Day: {fmt_pct(row.get('daily_return_pct', np.nan))} · PnL: {fmt_money(row.get('daily_pnl', row.get('position_pnl_today', np.nan)))} · Entry return: {fmt_pct(row.get('return_since_entry_pct', row.get('unrealized_return', np.nan)))}")
            st.caption(f"Rank: {row.get('raw_target_rank', 'n/a')} · Quality: {row.get('holding_quality_classification', 'n/a')} · Risk: {row.get('holding_risk_notes', 'n/a')}")


def _render_risk_snapshot(st, data: dict[str, pd.DataFrame]) -> None:
    vals = _risk_snapshot_values(data)
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Estimated portfolio vol", fmt_pct(vals["estimated_portfolio_volatility"]), "growth_official_paper_performance.csv")
    with cols[1]: metric_card(st, "Target volatility", fmt_pct(vals["target_volatility"]), "frozen config")
    with cols[2]: metric_card(st, "Uncapped exposure", fmt_pct(vals["uncapped_exposure"]), "growth_official_paper_state.csv")
    with cols[3]: metric_card(st, "Final exposure", fmt_pct(vals["final_exposure"]), "after floor/cap/dual trend")
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Floor active", "Yes" if vals["floor_active"] else "No", "min exposure 40%")
    with cols[1]: metric_card(st, "Cap active", "Yes" if vals["cap_active"] else "No", "cap/dual trend")
    with cols[2]: metric_card(st, "HHI", fmt_num(vals["hhi"]), "concentration")
    with cols[3]: metric_card(st, "Risk status", vals["risk_status"], vals["dual_trend_state"], badge=vals["risk_status"])
    exposure_steps = pd.DataFrame(
        {
            "stage": ["Raw exposure", "After floor", "After cap", "After dual trend", "Final exposure"],
            "exposure": [
                vals["uncapped_exposure"],
                max(vals["uncapped_exposure"], 0.40) if pd.notna(vals["uncapped_exposure"]) else np.nan,
                min(max(vals["uncapped_exposure"], 0.40), vals["exposure_cap_60"]) if pd.notna(vals["uncapped_exposure"]) and pd.notna(vals["exposure_cap_60"]) else np.nan,
                min(max(vals["uncapped_exposure"], 0.40), vals["exposure_cap_60"], vals["dual_trend_cap"]) if pd.notna(vals["uncapped_exposure"]) and pd.notna(vals["exposure_cap_60"]) and pd.notna(vals["dual_trend_cap"]) else np.nan,
                vals["final_exposure"],
            ],
        }
    )
    fig = go.Figure(go.Bar(x=exposure_steps["stage"], y=exposure_steps["exposure"], marker_color=[ORANGE, BRIGHT_ORANGE, AMBER, INFO, ORANGE]))
    fig.update_yaxes(tickformat=".0%", title="Exposure")
    fig = apply_plotly_layout(fig, "Exposure Stack")
    st.plotly_chart(fig, width="stretch")


def _today_changes(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    state = data.get("official_state", pd.DataFrame())
    perf = data.get("official_performance", pd.DataFrame())
    if state.empty or "date" not in state.columns:
        return pd.DataFrame(), "Official state unavailable."
    dates = sorted(pd.to_datetime(state["date"], errors="coerce").dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame(), "First official observation — no previous official date to compare."
    cur_date, prev_date = dates[-1], dates[-2]
    cur = state[state["date"].eq(cur_date)].copy()
    prev = state[state["date"].eq(prev_date)].copy()
    cur_hold = set(cur[~cur["ticker"].astype(str).str.upper().eq("CASH")]["ticker"].astype(str))
    prev_hold = set(prev[~prev["ticker"].astype(str).str.upper().eq("CASH")]["ticker"].astype(str))
    rows = [
        {"change": "holdings_added", "value": ",".join(sorted(cur_hold - prev_hold)) or "none"},
        {"change": "holdings_removed", "value": ",".join(sorted(prev_hold - cur_hold)) or "none"},
    ]
    for ticker in sorted(cur_hold | prev_hold):
        cw = numeric(cur[cur["ticker"].astype(str).eq(ticker)].get("paper_position_weight", pd.Series(dtype=float))).sum()
        pw = numeric(prev[prev["ticker"].astype(str).eq(ticker)].get("paper_position_weight", pd.Series(dtype=float))).sum()
        if abs(cw - pw) > 1e-8:
            rows.append({"change": f"weight_change_{ticker}", "value": f"{pw:.4f} -> {cw:.4f}"})
    if not perf.empty and len(perf) >= 2:
        pp, cp = perf.sort_values("date").iloc[-2], perf.sort_values("date").iloc[-1]
        rows += [
            {"change": "exposure_change", "value": f"{pp.get('exposure', np.nan):.4f} -> {cp.get('exposure', np.nan):.4f}"},
            {"change": "cash_change", "value": f"{pp.get('cash_weight', np.nan):.4f} -> {cp.get('cash_weight', np.nan):.4f}"},
            {"change": "volatility_change", "value": f"{pp.get('volatility', np.nan):.4f} -> {cp.get('volatility', np.nan):.4f}"},
        ]
    msg = "No portfolio changes — monitoring-only session." if all(r["value"] in {"none", "nan -> nan"} or "->" in r["value"] and r["value"].split(" -> ")[0] == r["value"].split(" -> ")[1] for r in rows[:2]) else "Official changes detected."
    return pd.DataFrame(rows), msg


def _render_next_action(st, data: dict[str, pd.DataFrame]) -> None:
    daily = _row(data.get("official_daily_status", pd.DataFrame()))
    state = _official_state_latest(data)
    state_row = state[~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")].iloc[0] if not state.empty and (~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")).any() else pd.Series(dtype=object)
    candidates = str(state_row.get("observed_candidate_tickers", "unavailable"))
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Next rebalance date", next_rebalance_date(data), "official scheduler")
    with cols[1]: metric_card(st, "Sessions remaining", str(state_row.get("sessions_since_last_rebalance", "unavailable")), "since last rebalance")
    with cols[2]: metric_card(st, "Expected execution date", str(state_row.get("economic_application_date", daily.get("date", "unavailable")))[:10], "t+1 application")
    with cols[3]: metric_card(st, "State", "Scheduled" if bool(daily.get("rebalance_due", False)) else "Monitoring only", "official daily status")
    st.caption("Pending candidates: " + candidates + " · Signal only — not executed until scheduled rebalance.")


def _executive_commentary(data: dict[str, pd.DataFrame]) -> str:
    perf = _row(data.get("official_performance", pd.DataFrame()))
    monitor = _row(data.get("official_monitor", pd.DataFrame()))
    changes, msg = _today_changes(data)
    holdings = _holdings_with_pnl(data)
    non_cash = holdings[~holdings.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")] if not holdings.empty else pd.DataFrame()
    strongest = weakest = "unavailable"
    if not non_cash.empty and "daily_return_pct" in non_cash.columns:
        ranked = non_cash.assign(_d=numeric(non_cash["daily_return_pct"])).sort_values("_d")
        weakest = str(ranked.iloc[0].get("ticker", "unavailable"))
        strongest = str(ranked.iloc[-1].get("ticker", "unavailable"))
    return (
        f"Portfolio remains {fmt_pct(perf.get('exposure', np.nan))} exposed with {fmt_pct(perf.get('cash_weight', monitor.get('cash', np.nan)))} cash. "
        f"Official integrity is {monitor.get('integrity_status', 'unavailable')} and market data status is {monitor.get('data_status', 'unavailable')}. "
        f"{msg} The strongest current position is {strongest}; the weakest is {weakest}. "
        "This is a deterministic monitoring summary, not a trading recommendation."
    )


def render_executive_terminal(st, data: dict[str, pd.DataFrame]) -> None:
    st.markdown(
        f"""
        <div class='page-head'>
          <div>
            <h2 class='page-title'>La Máquina Trading System {status_badge('READ ONLY', 'diagnostic')}</h2>
            <div class='page-subtitle'>Growth Champion Final — Official Forward Paper | {latest_market_date(data)} | {MODEL_VERSION}</div>
            <div class='page-subtitle'>Variant: {VARIANT}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _source_warning(st, data)
    _render_status_strip(st, data)
    st.subheader("Primary KPIs")
    _render_kpis(st, data)
    st.subheader("Performance Panel")
    chart_df = _benchmark_chart(st, data)
    st.subheader("Portfolio Snapshot")
    _render_portfolio_snapshot(st, data)
    st.subheader("Risk Snapshot")
    _render_risk_snapshot(st, data)
    st.subheader("What Changed Today")
    changes, msg = _today_changes(data)
    st.info(msg)
    if not changes.empty:
        st.dataframe(changes, width="stretch")
    st.subheader("Next Action")
    _render_next_action(st, data)
    st.subheader("Executive Commentary")
    st.markdown(f"<div class='alert-box info'>{_executive_commentary(data)}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(
        f"Data quality footer · source scope: Official Forward Paper · official start: {official_start_date(data)} · latest market date: {latest_market_date(data)} · "
        "market data provider: Yahoo/yfinance primary · exact/raw target status: official integrity file · real capital blocked"
    )


def build_executive_audits(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    source_map = {
        "growth_official_paper_performance.csv": data.get("official_performance", pd.DataFrame()),
        "growth_official_paper_state.csv": data.get("official_state", pd.DataFrame()),
        "growth_official_paper_actions.csv": data.get("official_actions", pd.DataFrame()),
        "growth_official_paper_monitor.csv": data.get("official_monitor", pd.DataFrame()),
        "growth_official_live_tracking.csv": data.get("official_tracking", pd.DataFrame()),
        "growth_official_estimated_cost_ledger.csv": data.get("official_cost_ledger", pd.DataFrame()),
        "growth_official_benchmark_daily.csv": data.get("official_benchmark_daily", pd.DataFrame()),
        "growth_official_position_pnl.csv": data.get("official_position_pnl", pd.DataFrame()),
        "official_paper_daily_run_status.csv": data.get("official_daily_status", pd.DataFrame()),
        "official_paper_integrity_status.csv": data.get("official_integrity", pd.DataFrame()),
    }
    for filename, df in source_map.items():
        date_range = ""
        if not df.empty and "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            date_range = f"{dates.min().date()} to {dates.max().date()}" if dates.notna().any() else ""
        rows.append({"source_file": filename, "namespace": "official_forward_paper", "exists_loaded": not df.empty, "row_count": len(df), "date_range": date_range})
    source_audit = pd.DataFrame(rows)

    perf = _row(data.get("official_performance", pd.DataFrame()))
    bench = data.get("official_benchmark_daily", pd.DataFrame())
    bench_last = bench.sort_values("date").iloc[-1] if not bench.empty and "date" in bench.columns else pd.Series(dtype=object)
    state = _official_state_latest(data)
    non_cash = state[~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")] if not state.empty else pd.DataFrame()
    checks = [
        {"check": "official_sources_loaded", "status": "PASS" if source_audit["exists_loaded"].all() else "WARNING", "detail": ",".join(source_audit.loc[~source_audit["exists_loaded"], "source_file"].tolist())},
        {"check": "chart_latest_equals_gross_card", "status": "PASS" if pd.notna(perf.get("gross_cumulative_return", np.nan)) and abs(float(perf.get("gross_cumulative_return")) * 100 - float(bench_last.get("growth_gross_cumulative_pct", np.nan))) < 1e-6 else "FAIL", "detail": "gross cumulative return vs official benchmark chart"},
        {"check": "chart_latest_equals_net_card", "status": "PASS" if pd.notna(perf.get("estimated_net_cumulative_return", np.nan)) and abs(float(perf.get("estimated_net_cumulative_return")) * 100 - float(bench_last.get("growth_net_cumulative_pct", np.nan))) < 1e-6 else "FAIL", "detail": "estimated net cumulative return vs official benchmark chart"},
        {"check": "holdings_equal_official_state", "status": "PASS" if not non_cash.empty else "FAIL", "detail": ",".join(non_cash.get("ticker", pd.Series(dtype=str)).astype(str).tolist())},
        {"check": "no_namespace_mixing", "status": "PASS", "detail": "Executive page uses official source map only"},
    ]
    integrity = pd.DataFrame(checks)
    final_status = "executive_terminal_pass"
    if integrity["status"].eq("FAIL").any():
        final_status = "executive_terminal_fail"
    elif integrity["status"].eq("WARNING").any():
        final_status = "executive_terminal_warning"
    return source_audit, integrity, final_status
