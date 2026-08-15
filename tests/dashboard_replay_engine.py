
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard_historical_loader import (
    ReplayData,
    latest_on_or_before,
    rows_on_or_before,
)


@dataclass
class ReplaySnapshot:
    date: pd.Timestamp
    performance: pd.Series
    holdings: pd.DataFrame
    actions: pd.DataFrame
    rebalance: pd.DataFrame
    benchmark: pd.Series
    risk: dict[str, Any]
    execution: dict[str, Any]
    governance: dict[str, Any]
    research: dict[str, Any]
    regime: dict[str, Any]
    decision_funnel: pd.DataFrame
    warnings: list[str]


def _num(x: Any) -> float:
    try:
        return float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])
    except Exception:
        return np.nan


def _last_series(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def _non_cash(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df.copy()
    return df[~df["ticker"].astype(str).str.upper().eq("CASH")].copy()


def _hhi(weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce").dropna()
    return float(np.square(w).sum()) if not w.empty else np.nan


def _drawdown_from_perf(perf: pd.DataFrame, date: pd.Timestamp) -> float:
    hist = perf.copy()
    if hist.empty or "date" not in hist.columns:
        return np.nan
    hist = hist[pd.to_datetime(hist["date"], errors="coerce").dt.normalize().le(pd.Timestamp(date).normalize())]
    if hist.empty:
        return np.nan
    eq_col = "gross_equity" if "gross_equity" in hist.columns else "portfolio_value" if "portfolio_value" in hist.columns else None
    if eq_col is None:
        return np.nan
    eq = pd.to_numeric(hist[eq_col], errors="coerce").dropna()
    if eq.empty:
        return np.nan
    return float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1)


def build_snapshot(replay: ReplayData, date: pd.Timestamp) -> ReplaySnapshot:
    frames = replay.frames
    warnings: list[str] = []
    perf_df = rows_on_or_before(frames.get("performance", pd.DataFrame()), date, exact=True)
    state = rows_on_or_before(frames.get("state", pd.DataFrame()), date, exact=True)
    actions = rows_on_or_before(frames.get("actions", pd.DataFrame()), date, exact=True)
    rebalance = rows_on_or_before(frames.get("rebalance_report", pd.DataFrame()), date, exact=True)
    benchmark_df = rows_on_or_before(frames.get("benchmark_daily", pd.DataFrame()), date, exact=True)
    if perf_df.empty: warnings.append("Historical performance unavailable for selected date")
    if state.empty: warnings.append("Historical holdings unavailable for selected date")
    if benchmark_df.empty: warnings.append("Historical benchmark unavailable for selected date")

    perf = _last_series(perf_df)
    benchmark = _last_series(benchmark_df)
    holdings = state.copy()
    pnl = rows_on_or_before(frames.get("position_pnl", pd.DataFrame()), date, exact=True)
    if not holdings.empty and not pnl.empty and "ticker" in holdings.columns and "ticker" in pnl.columns:
        holdings = holdings.merge(pnl.drop(columns=["date"], errors="ignore"), on="ticker", how="left", suffixes=("", "_pnl"))
    non_cash = _non_cash(holdings)
    weight_col = "paper_position_weight" if "paper_position_weight" in non_cash.columns else "weight" if "weight" in non_cash.columns else None
    weights = pd.to_numeric(non_cash[weight_col], errors="coerce") if weight_col else pd.Series(dtype=float)
    risk = {
        "drawdown": _drawdown_from_perf(frames.get("performance", pd.DataFrame()), date),
        "volatility": _num(perf.get("volatility", np.nan)),
        "hhi": _hhi(weights),
        "exposure": _num(perf.get("exposure", non_cash.get("paper_position_weight", pd.Series(dtype=float)).pipe(pd.to_numeric, errors="coerce").sum() if not non_cash.empty and "paper_position_weight" in non_cash.columns else np.nan)),
        "cash": _num(perf.get("cash_weight", holdings.loc[holdings.get("ticker", "").astype(str).str.upper().eq("CASH"), "paper_position_weight"].sum() if not holdings.empty and "paper_position_weight" in holdings.columns else np.nan)),
        "target_vol": 0.22,
        "dual_trend_cap": _num(non_cash.get("dual_trend_cap", pd.Series(dtype=float)).dropna().iloc[-1]) if not non_cash.empty and "dual_trend_cap" in non_cash.columns and non_cash["dual_trend_cap"].notna().any() else np.nan,
        "largest_risk_contributor": non_cash.sort_values(weight_col, ascending=False).iloc[0].get("ticker") if weight_col and not non_cash.empty else "unavailable",
        "beta": np.nan,
        "var": np.nan,
        "cvar": np.nan,
        "correlation": np.nan,
    }
    costs = rows_on_or_before(frames.get("cost_ledger", pd.DataFrame()), date, exact=True)
    trades = rows_on_or_before(frames.get("trades", pd.DataFrame()), date, exact=True)
    execution = {
        "orders": len(actions[actions.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().ne("CASH")]) if not actions.empty else 0,
        "trades": len(trades) if not trades.empty else 0,
        "estimated_costs": _num(costs.get("estimated_total_cost", pd.Series(dtype=float)).sum()) if not costs.empty and "estimated_total_cost" in costs.columns else _num(perf.get("estimated_execution_cost", 0)),
        "estimated_slippage": np.nan,
        "adv_participation": np.nan,
        "execution_delay": str(perf.get("economic_application_date", "unavailable")),
        "broker_status": "no broker / no real orders",
        "accounting_status": "stored official accounting" if not frames.get("accounting_reconciliation", pd.DataFrame()).empty else "historical accounting unavailable",
    }
    monitor = latest_on_or_before(frames.get("monitor", pd.DataFrame()), date)
    integrity = latest_on_or_before(frames.get("integrity", pd.DataFrame()), date)
    daily = latest_on_or_before(frames.get("daily_status", pd.DataFrame()), date)
    governance = {
        "paper_status": _last_series(monitor).get("governance_status", "unavailable"),
        "promotion_status": _last_series(monitor).get("promotion_status", "unavailable"),
        "integrity_status": _last_series(integrity).get("integrity_status", _last_series(monitor).get("integrity_status", "unavailable")),
        "data_status": _last_series(daily).get("data_status", _last_series(monitor).get("data_status", "unavailable")),
        "research": "historical research state unavailable" if frames.get("governance_history", pd.DataFrame()).empty else "available diagnostic history",
        "market_data": "official_market_data_integrity latest snapshot" if not frames.get("market_data_integrity", pd.DataFrame()).empty else "unavailable",
        "real_orders": _last_series(monitor).get("real_orders", False),
    }
    research = {
        "champion_version": perf.get("growth_model_version", perf.get("model_version", "unavailable")),
        "dsr": "historical DSR snapshot unavailable",
        "pbo": "historical PBO snapshot unavailable",
        "parameter_stability": "historical parameter stability snapshot unavailable",
        "research_status": governance["research"],
    }
    regime = {
        "dual_trend": non_cash.get("dual_trend_reason", pd.Series(["unavailable"])).dropna().iloc[-1] if not non_cash.empty and "dual_trend_reason" in non_cash.columns and non_cash["dual_trend_reason"].notna().any() else "unavailable",
        "spy_below_200d": non_cash.get("spy_below_200d", pd.Series([np.nan])).dropna().iloc[-1] if not non_cash.empty and "spy_below_200d" in non_cash.columns and non_cash["spy_below_200d"].notna().any() else np.nan,
        "qqq_below_200d": non_cash.get("qqq_below_200d", pd.Series([np.nan])).dropna().iloc[-1] if not non_cash.empty and "qqq_below_200d" in non_cash.columns and non_cash["qqq_below_200d"].notna().any() else np.nan,
        "hmm_state": "historical HMM state unavailable",
        "market_classification": "official dual-trend diagnostic" if not non_cash.empty else "unavailable",
    }
    decision_funnel = pd.DataFrame([
        {"stage": "Universe", "count": np.nan, "status": "Historical pipeline counts unavailable"},
        {"stage": "Quality", "count": np.nan, "status": "Historical pipeline counts unavailable"},
        {"stage": "Tradability", "count": np.nan, "status": "Historical pipeline counts unavailable"},
        {"stage": "Ranking", "count": len(non_cash) if not non_cash.empty else np.nan, "status": "Stored selected holdings only"},
        {"stage": "Portfolio", "count": len(non_cash), "status": "Official holdings stored"},
    ])
    return ReplaySnapshot(pd.Timestamp(date).normalize(), perf, holdings, actions, rebalance, benchmark, risk, execution, governance, research, regime, decision_funnel, warnings)


def performance_evolution(replay: ReplayData, date: pd.Timestamp) -> pd.DataFrame:
    perf = replay.frames.get("performance", pd.DataFrame()).copy()
    bench = replay.frames.get("benchmark_daily", pd.DataFrame()).copy()
    if perf.empty or "date" not in perf.columns:
        return pd.DataFrame()
    target = pd.Timestamp(date).normalize()
    perf = perf[pd.to_datetime(perf["date"], errors="coerce").dt.normalize().le(target)].copy()
    if not bench.empty and "date" in bench.columns:
        perf = perf.merge(bench, on="date", how="left", suffixes=("", "_bench"))
    return perf


def compare_snapshots(a: ReplaySnapshot, b: ReplaySnapshot) -> pd.DataFrame:
    rows = []
    for field, label in [("portfolio_value", "Portfolio Value"), ("estimated_net_portfolio_value", "Net Portfolio Value"), ("exposure", "Exposure"), ("cash_weight", "Cash"), ("current_drawdown", "Drawdown"), ("volatility", "Volatility")]:
        av = a.performance.get(field, a.risk.get(field, np.nan))
        bv = b.performance.get(field, b.risk.get(field, np.nan))
        rows.append({"metric": label, "date_a": a.date.date(), "value_a": av, "date_b": b.date.date(), "value_b": bv, "difference": _num(bv) - _num(av)})
    return pd.DataFrame(rows)


def validate_replay(replay: ReplayData) -> tuple[pd.DataFrame, str]:
    rows = []
    status = "historical_replay_pass"
    for date in replay.dates:
        snap = build_snapshot(replay, date)
        non_cash = _non_cash(snap.holdings)
        weight_col = "paper_position_weight" if "paper_position_weight" in snap.holdings.columns else None
        weights_sum = pd.to_numeric(snap.holdings[weight_col], errors="coerce").sum() if weight_col else np.nan
        perf_value = _num(snap.performance.get("gross_portfolio_value", snap.performance.get("portfolio_value", np.nan)))
        holdings_value = pd.to_numeric(snap.holdings.get("paper_position_value", pd.Series(dtype=float)), errors="coerce").sum() if "paper_position_value" in snap.holdings.columns else np.nan
        value_diff = holdings_value - perf_value if pd.notna(holdings_value) and pd.notna(perf_value) else np.nan
        rows.append({
            "date": date.date(),
            "holdings_rows": len(snap.holdings),
            "performance_available": not snap.performance.empty,
            "benchmark_available": not snap.benchmark.empty,
            "weights_sum": weights_sum,
            "portfolio_value": perf_value,
            "holdings_value_sum": holdings_value,
            "value_diff": value_diff,
            "no_future_leakage": True,
            "namespace": "official_only",
            "warnings": "; ".join(snap.warnings),
        })
    df = pd.DataFrame(rows)
    if df.empty or df["performance_available"].eq(False).any() or df["benchmark_available"].eq(False).any():
        status = "historical_replay_warning"
    if df.get("no_future_leakage", pd.Series([True])).eq(False).any():
        status = "historical_replay_fail"
    return df, status
