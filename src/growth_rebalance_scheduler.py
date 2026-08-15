from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_FILE = "growth_candidate_paper_config.json"
SCHEDULE_FILE = "growth_rebalance_schedule.csv"
SEMANTICS_AUDIT_FILE = "growth_rebalance_semantics_audit.csv"
SEMANTICS_REPORT_FILE = "growth_rebalance_semantics_report.txt"
SPY_CACHE = Path("yahoo_ohlcv_price_cache") / "SPY.csv"
RECON_DAILY_FILE = "reconstructed_growth_long_horizon_daily_returns.csv"
FORECAST_HISTORY_FILE = "forecast_history.csv"

DEFAULT_ANCHOR = "2023-01-04"
DEFAULT_FREQUENCY = 5
DEFAULT_EXECUTION_LAG = "close_t_signal_apply_next_session_return"


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _date_series_from_csv(path: str | Path, col: str = "Date") -> pd.Series:
    df = _read_csv(path)
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    if col not in df.columns:
        col = "date" if "date" in df.columns else df.columns[0]
    dates = pd.to_datetime(df[col], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    return dates.reset_index(drop=True)


def audit_backtest_semantics() -> tuple[pd.DataFrame, str]:
    recon = _read_csv(RECON_DAILY_FILE)
    anchor = DEFAULT_ANCHOR
    if not recon.empty and {"window_start", "date"}.issubset(recon.columns):
        sub = recon[recon["window_start"].astype(str).eq("2022-01-03")].copy()
        if not sub.empty:
            anchor = pd.to_datetime(sub["date"], errors="coerce").dropna().min().strftime("%Y-%m-%d")
    rows = [
        {
            "item": "rebalance_frequency",
            "value": "every 5 trading sessions",
            "evidence": "STEP_DAYS = 5 and dates = px.index[LOOKBACK::STEP_DAYS]",
        },
        {
            "item": "scheduling_method",
            "value": "rolling session count",
            "evidence": "No fixed weekday; schedule advances by market-data row index.",
        },
        {"item": "anchor_date", "value": anchor, "evidence": "First 2022+ reconstructed decision date after 252-session lookback."},
        {
            "item": "signal_timing",
            "value": "signals use close/history through decision date t",
            "evidence": "hist = px.iloc[:loc + 1]; raw targets generated from truncated history.",
        },
        {
            "item": "execution_lag",
            "value": DEFAULT_EXECUTION_LAG,
            "evidence": "entry_date = px.index[loc + 1]; exit_date = loc + STEP_DAYS + 1.",
        },
        {
            "item": "holding_period",
            "value": "target holdings frozen between rebalance dates",
            "evidence": "Loop only creates weights on decision dates; no daily soft-exit path exists.",
        },
        {
            "item": "soft_exit_timing",
            "value": "rebalance dates only",
            "evidence": "soft_keep is evaluated inside the decision_date loop only.",
        },
        {
            "item": "volatility_target_timing",
            "value": "rebalance dates only",
            "evidence": "_target_exposure(prior_returns, previous_exposure) is called inside the decision_date loop.",
        },
        {
            "item": "dual_trend_timing",
            "value": "rebalance dates only in v3 overlay",
            "evidence": "Crisis overlay is applied to scheduled reconstructed rows, not intra-period daily holdings.",
        },
        {
            "item": "turnover_cost_date",
            "value": "decision/rebalance date",
            "evidence": "turnover is stored on the decision_date row from current weights vs prior_weights.",
        },
    ]
    audit = pd.DataFrame(rows)
    report = "===== GROWTH REBALANCE SEMANTICS AUDIT =====\n" + "\n".join(f"{r['item']}: {r['value']} ({r['evidence']})" for r in rows)
    audit.to_csv(SEMANTICS_AUDIT_FILE, index=False)
    Path(SEMANTICS_REPORT_FILE).write_text(report + "\n", encoding="utf-8")
    return audit, report


def load_config() -> dict[str, object]:
    default = {
        "rebalance_frequency_sessions": DEFAULT_FREQUENCY,
        "rebalance_mode": "exact_backtest_parity",
        "rebalance_anchor_date": DEFAULT_ANCHOR,
        "signal_execution_lag": DEFAULT_EXECUTION_LAG,
        "allow_unscheduled_rebalance": False,
    }
    p = Path(CONFIG_FILE)
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                default.update(loaded)
        except Exception:
            pass
    return default


def update_config_with_semantics() -> dict[str, object]:
    audit, _ = audit_backtest_semantics()
    cfg = load_config()
    anchor_rows = audit[audit["item"].eq("anchor_date")]
    anchor = str(anchor_rows.iloc[0]["value"]) if not anchor_rows.empty else DEFAULT_ANCHOR
    cfg.update(
        {
            "rebalance_frequency_sessions": DEFAULT_FREQUENCY,
            "rebalance_mode": "exact_backtest_parity",
            "rebalance_anchor_date": anchor,
            "signal_execution_lag": DEFAULT_EXECUTION_LAG,
            "allow_unscheduled_rebalance": False,
        }
    )
    p = Path(CONFIG_FILE)
    existing = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(cfg)
    p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return existing


def _market_sessions(anchor: pd.Timestamp, latest: pd.Timestamp | None = None) -> pd.Series:
    spy = _date_series_from_csv(SPY_CACHE, "Date")
    if spy.empty:
        recon = _date_series_from_csv(RECON_DAILY_FILE, "date")
        spy = recon
    if latest is None:
        fh = _date_series_from_csv(FORECAST_HISTORY_FILE, "date")
        latest = fh.max() if not fh.empty else (spy.max() if not spy.empty else anchor)
    sessions = spy[(spy >= anchor) & (spy <= latest)].drop_duplicates().sort_values().reset_index(drop=True)
    return sessions


def build_rebalance_schedule(latest_date: str | pd.Timestamp | None = None, write: bool = True) -> pd.DataFrame:
    cfg = load_config()
    anchor = pd.Timestamp(str(cfg.get("rebalance_anchor_date", DEFAULT_ANCHOR))).normalize()
    freq = int(cfg.get("rebalance_frequency_sessions", DEFAULT_FREQUENCY))
    latest = pd.Timestamp(latest_date).normalize() if latest_date is not None else None
    sessions = _market_sessions(anchor, latest)
    rows = []
    rebalance_dates = []
    for idx, dt in enumerate(sessions):
        due = idx % freq == 0
        if due:
            rebalance_dates.append(dt)
        prev = rebalance_dates[-1] if rebalance_dates else pd.NaT
        next_idx = idx + (freq - idx % freq if idx % freq else freq)
        next_dt = sessions.iloc[next_idx] if next_idx < len(sessions) else pd.NaT
        rows.append(
            {
                "market_date": dt.strftime("%Y-%m-%d"),
                "trading_session_number": idx,
                "rebalance_due": bool(due),
                "rebalance_sequence_number": int(idx // freq) if due else np.nan,
                "previous_rebalance_date": prev.strftime("%Y-%m-%d") if pd.notna(prev) else "",
                "next_rebalance_date": next_dt.strftime("%Y-%m-%d") if pd.notna(next_dt) else "",
                "anchor_date": anchor.strftime("%Y-%m-%d"),
                "scheduling_method": "exact_backtest_parity_session_count",
                "rebalance_frequency_sessions": freq,
                "sessions_since_last_rebalance": int(idx % freq),
            }
        )
    schedule = pd.DataFrame(rows)
    if write:
        schedule.to_csv(SCHEDULE_FILE, index=False)
    return schedule


def scheduler_status(date: str | pd.Timestamp) -> dict[str, object]:
    dt = pd.Timestamp(date).normalize()
    schedule = build_rebalance_schedule(latest_date=dt, write=True)
    if schedule.empty:
        return {"date": dt.strftime("%Y-%m-%d"), "rebalance_due": False, "reason": "schedule_unavailable"}
    row = schedule[schedule["market_date"].astype(str).eq(dt.strftime("%Y-%m-%d"))]
    if row.empty:
        prior = schedule[pd.to_datetime(schedule["market_date"]) < dt]
        latest = prior.iloc[-1] if not prior.empty else schedule.iloc[-1]
        return {
            "date": dt.strftime("%Y-%m-%d"),
            "rebalance_due": False,
            "reason": "date_not_in_market_session_calendar",
            "previous_rebalance_date": latest.get("previous_rebalance_date", ""),
            "next_rebalance_date": latest.get("next_rebalance_date", ""),
            "sessions_since_last_rebalance": latest.get("sessions_since_last_rebalance", np.nan),
        }
    return row.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-date", default=None)
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()
    audit, report = audit_backtest_semantics()
    if args.update_config:
        update_config_with_semantics()
    schedule = build_rebalance_schedule(args.latest_date, write=True)
    print(report)
    print("\n===== GROWTH REBALANCE SCHEDULE =====")
    print(schedule.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
