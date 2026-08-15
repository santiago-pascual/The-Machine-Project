from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as _st

    _cache_data = _st.cache_data
except Exception:

    def _cache_data(**_kwargs):
        def deco(func):
            return func

        return deco


CORE_REPLAY_DATE_KEYS = {"performance", "state", "actions", "benchmark_daily", "monitor"}

OFFICIAL_REPLAY_SOURCES: dict[str, str] = {
    "performance": "growth_official_paper_performance.csv",
    "state": "growth_official_paper_state.csv",
    "actions": "growth_official_paper_actions.csv",
    "trades": "growth_official_paper_trades.csv",
    "trade_lifecycle": "growth_official_trade_lifecycle.csv",
    "benchmark_daily": "growth_official_benchmark_daily.csv",
    "benchmark_equity": "growth_official_benchmark_equity.csv",
    "rebalance_report": "growth_official_paper_rebalance_report.csv",
    "cost_ledger": "growth_official_estimated_cost_ledger.csv",
    "position_pnl": "growth_official_position_pnl.csv",
    "realized_pnl": "growth_official_realized_pnl.csv",
    "monitor": "growth_official_paper_monitor.csv",
    "daily_status": "official_paper_daily_run_status.csv",
    "integrity": "official_paper_integrity_status.csv",
    "version_history": "official_paper_version_history.csv",
    "market_data_integrity": "official_market_data_integrity.csv",
    "accounting_reconciliation": "official_accounting_reconciliation.csv",
    "accounting_audit": "official_forward_accounting_audit.csv",
    "cost_duplication": "official_cost_duplication_audit.csv",
    "execution_lag": "official_execution_lag_audit.csv",
    "data_quality": "official_paper_data_quality.csv",
    "scheduler": "growth_rebalance_schedule.csv",
    "pipeline_history": "growth_pipeline_stage_validation.csv",
    "governance_history": "growth_paper_governance_history.csv",
}

EXPECTED_BUT_OPTIONAL = [
    "growth_official_daily_portfolio.csv",
    "growth_official_position_history.csv",
    "growth_official_portfolio_equity.csv",
    "growth_official_accounting.csv",
    "growth_official_governance.csv",
    "growth_official_pipeline_history.csv",
    "growth_official_scheduler.csv",
    "growth_rebalance_history.csv",
]

DATE_COLUMNS = ["date", "signal_date", "economic_application_date", "run_time", "timestamp", "latest_date", "expected_date"]


@dataclass
class ReplayData:
    frames: dict[str, pd.DataFrame]
    source_audit: pd.DataFrame
    dates: list[pd.Timestamp]
    missing_sources: list[str]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
            if col != "run_time" and col != "timestamp":
                out[col] = out[col].dt.normalize()
    return out


@_cache_data(show_spinner=False)
def _read_csv_cached(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return _normalize(pd.read_csv(p))
    except Exception:
        return pd.DataFrame()


def load_replay_data() -> ReplayData:
    frames: dict[str, pd.DataFrame] = {}
    rows = []
    dates: set[pd.Timestamp] = set()
    missing: list[str] = []
    for key, path in OFFICIAL_REPLAY_SOURCES.items():
        df = _read_csv_cached(path)
        frames[key] = df
        exists = Path(path).exists()
        if not exists:
            missing.append(path)
        date_min = date_max = ""
        if not df.empty and "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            if d.notna().any():
                date_min = str(d.min().date())
                date_max = str(d.max().date())
                if key in CORE_REPLAY_DATE_KEYS:
                    for val in d.dropna().dt.normalize().unique():
                        dates.add(pd.Timestamp(val))
        rows.append(
            {
                "source_key": key,
                "source_file": path,
                "exists": exists,
                "loaded": not df.empty,
                "row_count": len(df),
                "date_min": date_min,
                "date_max": date_max,
                "scope": "official_forward_history",
                "namespace": "official_only",
            }
        )
    for path in EXPECTED_BUT_OPTIONAL:
        if not Path(path).exists():
            missing.append(path)
            rows.append(
                {
                    "source_key": "optional_expected",
                    "source_file": path,
                    "exists": False,
                    "loaded": False,
                    "row_count": 0,
                    "date_min": "",
                    "date_max": "",
                    "scope": "official_forward_history",
                    "namespace": "missing_optional",
                }
            )
    return ReplayData(frames=frames, source_audit=pd.DataFrame(rows), dates=sorted(dates), missing_sources=missing)


def rows_on_or_before(df: pd.DataFrame, date: pd.Timestamp, exact: bool = True) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    target = pd.Timestamp(date).normalize()
    if exact:
        return df[d.eq(target)].copy()
    prev = d[d.le(target)]
    if prev.empty:
        return pd.DataFrame()
    return df[d.eq(prev.max())].copy()


def latest_on_or_before(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    return rows_on_or_before(df, date, exact=False)


def nearest_replay_date(dates: list[pd.Timestamp], requested: Any) -> pd.Timestamp | None:
    if not dates:
        return None
    target = pd.Timestamp(requested).normalize()
    arr = pd.Series(dates).sort_values()
    le = arr[arr.le(target)]
    return pd.Timestamp(le.iloc[-1]) if not le.empty else pd.Timestamp(arr.iloc[0])
