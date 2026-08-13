from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from growth_rebalance_scheduler import build_rebalance_schedule
from current_growth_feature_generation import generate_current_growth_features
from growth_candidate_paper_trading import run_growth_candidate_paper_trading

BACKUP_FILES = [
    "growth_candidate_paper_state.csv",
    "growth_candidate_paper_trades.csv",
    "growth_candidate_paper_performance.csv",
    "growth_candidate_paper_monitor.csv",
    "growth_candidate_action_signals.csv",
    "growth_candidate_rebalance_report.csv",
    "growth_paper_governance_report.csv",
    "growth_paper_governance_history.csv",
    "growth_paper_monthly_report.csv",
    "growth_live_tracking.csv",
    "growth_live_health.csv",
    "growth_live_drift.csv",
    "growth_live_tracking_governance.csv",
    "benchmark_daily_returns.csv",
    "benchmark_equity_curves.csv",
    "current_raw_target_features.csv",
    "current_growth_features.csv",
    "current_growth_candidate_allocation.csv",
]
PAPER_FILES = BACKUP_FILES[:]
OUT_AUDIT = "growth_five_session_replay_audit.csv"
OUT_CHANGES = "growth_five_session_replay_changes.csv"
OUT_SUMMARY = "growth_five_session_replay_summary.txt"


def read(path: str) -> pd.DataFrame:
    p=Path(path)
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()


def latest_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if df.empty or "date" not in df.columns: return None
    d=pd.to_datetime(df["date"], errors="coerce").dropna()
    return d.min().normalize() if not d.empty else None


def backup_files() -> Path:
    folder=Path("paper_history_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(exist_ok=True)
    for f in BACKUP_FILES:
        p=Path(f)
        if p.exists(): shutil.copy2(p, folder / p.name)
    return folder


def replay_dates() -> list[pd.Timestamp]:
    state=read("growth_candidate_paper_state.csv")
    perf=read("growth_candidate_paper_performance.csv")
    starts=[x for x in [latest_date(state), latest_date(perf)] if x is not None]
    first=min(starts) if starts else pd.Timestamp("2026-06-18")
    fh=read("forecast_history.csv")
    if fh.empty or "date" not in fh.columns: return []
    fh_dates=pd.to_datetime(fh["date"], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    schedule=build_rebalance_schedule(fh_dates.max(), write=True)
    sched=schedule.copy()
    sched["market_date"]=pd.to_datetime(sched["market_date"], errors="coerce").dt.normalize()
    prior_due=sched[(sched["market_date"] <= first) & (sched["rebalance_due"].astype(bool))]
    start=prior_due["market_date"].max() if not prior_due.empty else first
    return [d for d in fh_dates if d >= start]


def holdings_for(path: str) -> dict[str, str]:
    df=read(path)
    if df.empty or "date" not in df.columns: return {}
    df["date"]=pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out={}
    for d,g in df.groupby("date"):
        if "ticker" in g.columns:
            out[d]=",".join(g[g["ticker"].astype(str).ne("CASH")]["ticker"].astype(str).tolist())
    return out


def main() -> None:
    old_perf=read("growth_candidate_paper_performance.csv")
    old_state=read("growth_candidate_paper_state.csv")
    old_holdings=holdings_for("growth_candidate_paper_state.csv")
    backup=backup_files()
    dates=replay_dates()
    for f in PAPER_FILES:
        p=Path(f)
        if p.exists(): p.unlink()
    audit=[]
    for dt in dates:
        ds=dt.strftime("%Y-%m-%d")
        try:
            generate_current_growth_features(overwrite_same_day=True, allow_stale_growth_volatility=False, as_of_date=ds)
            result=run_growth_candidate_paper_trading(overwrite_same_day=True, allow_proxy_fallback=False)
            perf=read("growth_candidate_paper_performance.csv")
            latest=perf[pd.to_datetime(perf["date"], errors="coerce").dt.strftime("%Y-%m-%d").eq(ds)].tail(1)
            audit.append({"date":ds,"status":"ok","rebalance_due":result.get("rebalance_due", ""),"monitoring_only":result.get("monitoring_only", ""),"portfolio_value": latest.iloc[-1].get("portfolio_value", "") if not latest.empty else "","daily_return": latest.iloc[-1].get("daily_return", "") if not latest.empty else ""})
        except Exception as exc:
            audit.append({"date":ds,"status":"failed","error":str(exc)})
            raise
    # Regenerate derived monitors if scripts are available.
    for script in ["growth_paper_governance.py", "growth_live_tracking_monitor.py", "benchmark_daily_series_export.py"]:
        if Path(script).exists():
            subprocess.run([sys.executable, script], check=False)
    new_perf=read("growth_candidate_paper_performance.csv")
    new_state=read("growth_candidate_paper_state.csv")
    new_holdings=holdings_for("growth_candidate_paper_state.csv")
    audit_df=pd.DataFrame(audit)
    audit_df.to_csv(OUT_AUDIT,index=False)
    changes=[]
    if not old_perf.empty and not new_perf.empty:
        old=old_perf.copy(); new=new_perf.copy()
        old["date"]=pd.to_datetime(old["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        new["date"]=pd.to_datetime(new["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        old=old.drop_duplicates("date", keep="last").set_index("date")
        new=new.drop_duplicates("date", keep="last").set_index("date")
        for d in sorted(set(old.index)|set(new.index)):
            changes.append({
                "date":d,
                "old_portfolio_value": old.loc[d,"portfolio_value"] if d in old.index and "portfolio_value" in old.columns else "",
                "new_portfolio_value": new.loc[d,"portfolio_value"] if d in new.index and "portfolio_value" in new.columns else "",
                "old_daily_return": old.loc[d,"daily_return"] if d in old.index and "daily_return" in old.columns else "",
                "new_daily_return": new.loc[d,"daily_return"] if d in new.index and "daily_return" in new.columns else "",
                "old_holdings": old_holdings.get(d,""),
                "new_holdings": new_holdings.get(d,""),
                "changed": old_holdings.get(d,"") != new_holdings.get(d,"") or (d not in old.index) or (d not in new.index),
            })
    pd.DataFrame(changes).to_csv(OUT_CHANGES,index=False)
    final_perf=new_perf.tail(1).to_dict("records")[0] if not new_perf.empty else {}
    summary=(
        "===== FIVE SESSION GROWTH PAPER REPLAY SUMMARY =====\n"
        f"backup_folder: {backup}\n"
        f"replay_dates: {len(dates)}\n"
        f"first_replay_date: {dates[0].strftime('%Y-%m-%d') if dates else ''}\n"
        f"last_replay_date: {dates[-1].strftime('%Y-%m-%d') if dates else ''}\n"
        f"old_perf_rows: {len(old_perf)}\nnew_perf_rows: {len(new_perf)}\n"
        f"final_portfolio_value: {final_perf.get('portfolio_value','')}\n"
        f"final_cumulative_return: {final_perf.get('cumulative_return','')}\n"
        f"final_exposure: {final_perf.get('exposure','')}\n"
        f"final_cash: {final_perf.get('cash_weight','')}\n"
    )
    Path(OUT_SUMMARY).write_text(summary, encoding="utf-8")
    print(summary)

if __name__ == "__main__":
    main()
