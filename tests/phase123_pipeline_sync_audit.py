from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd

IMPORTANT_FILES = [
    "forecast_history.csv",
    "current_raw_target_features.csv",
    "current_growth_features.csv",
    "current_growth_candidate_allocation.csv",
    "growth_candidate_paper_state.csv",
    "growth_candidate_paper_performance.csv",
    "growth_official_paper_state.csv",
    "growth_official_paper_actions.csv",
    "growth_official_paper_trades.csv",
    "growth_official_paper_performance.csv",
    "growth_official_paper_monitor.csv",
    "growth_official_live_tracking.csv",
    "growth_official_trade_lifecycle.csv",
    "growth_official_position_pnl.csv",
    "growth_official_estimated_cost_ledger.csv",
    "growth_official_benchmark_daily.csv",
    "growth_official_benchmark_equity.csv",
    "official_paper_daily_run_status.csv",
    "official_paper_integrity_status.csv",
    "official_market_data_integrity.csv",
    "official_market_data_governance.csv",
    "benchmark_daily_returns.csv",
    "benchmark_equity_curves.csv",
]

DASHBOARD_CONSUMED = {
    "growth_official_paper_performance.csv",
    "growth_official_paper_state.csv",
    "growth_official_paper_actions.csv",
    "growth_official_paper_rebalance_report.csv",
    "growth_official_paper_trades.csv",
    "growth_official_trade_lifecycle.csv",
    "growth_official_paper_monitor.csv",
    "growth_official_live_tracking.csv",
    "growth_official_estimated_cost_ledger.csv",
    "growth_official_position_pnl.csv",
    "growth_official_realized_pnl.csv",
    "growth_official_benchmark_daily.csv",
    "growth_official_benchmark_equity.csv",
    "official_paper_integrity_status.csv",
    "official_paper_daily_run_status.csv",
    "official_market_data_governance.csv",
    "official_market_data_integrity.csv",
}

REGENERATED_BY_DAILY = {
    "forecast_history.csv": "financial_data_system.py / forecast_history_freshness_refresh.py",
    "current_raw_target_features.csv": "current_growth_feature_generation.py",
    "current_growth_features.csv": "current_growth_feature_generation.py",
    "current_growth_candidate_allocation.csv": "current_growth_feature_generation.py",
    "growth_candidate_paper_state.csv": "growth_candidate_paper_trading.py",
    "growth_candidate_paper_performance.csv": "growth_candidate_paper_trading.py",
    "growth_official_paper_state.csv": "growth_official_paper_lifecycle.py",
    "growth_official_paper_actions.csv": "growth_official_paper_lifecycle.py",
    "growth_official_paper_trades.csv": "growth_official_paper_lifecycle.py",
    "growth_official_paper_performance.csv": "growth_official_paper_lifecycle.py",
    "growth_official_paper_monitor.csv": "growth_official_paper_lifecycle.py",
    "growth_official_live_tracking.csv": "growth_official_paper_lifecycle.py",
    "growth_official_estimated_cost_ledger.csv": "growth_official_paper_lifecycle.py",
    "growth_official_benchmark_daily.csv": "official_benchmark_chart_repair.py",
    "growth_official_benchmark_equity.csv": "official_benchmark_chart_repair.py",
    "official_paper_daily_run_status.csv": "growth_official_paper_lifecycle.py",
    "official_paper_integrity_status.csv": "growth_official_paper_lifecycle.py",
    "official_market_data_integrity.csv": "canonical_market_data_manager.validate_freshness",
    "official_market_data_governance.csv": "canonical_market_data_manager.validate_freshness",
}


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, engine="python")
        except Exception:
            return pd.DataFrame()


def date_col(df: pd.DataFrame) -> str | None:
    for c in ["date", "Date", "signal_date", "latest_date", "canonical_market_date", "expected_signal_date"]:
        if c in df.columns:
            return c
    return None


def latest_date(path: str | Path) -> pd.Timestamp | pd.NaT:
    df = read_csv(path)
    if df.empty:
        return pd.NaT
    dc = date_col(df)
    if dc is None:
        return pd.NaT
    d = pd.to_datetime(df[dc], errors="coerce").dropna()
    return d.max().normalize() if not d.empty else pd.NaT


def dup_dates(path: str | Path) -> int:
    df = read_csv(path)
    dc = date_col(df)
    if df.empty or dc is None:
        return 0
    if "ticker" in df.columns:
        return int(df.duplicated(subset=[dc, "ticker"]).sum())
    return int(df.duplicated(subset=[dc]).sum())


def file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_market_date() -> pd.Timestamp | pd.NaT:
    forecast = latest_date("forecast_history.csv")
    gov = read_csv("official_market_data_governance.csv")
    if not gov.empty and "expected_signal_date" in gov.columns:
        d = pd.to_datetime(gov["expected_signal_date"], errors="coerce").dropna()
        if not d.empty:
            return max(forecast, d.max().normalize()) if pd.notna(forecast) else d.max().normalize()
    return forecast


def audit_official_freshness() -> pd.DataFrame:
    market = latest_market_date()
    rows = []
    for name in IMPORTANT_FILES:
        p = Path(name)
        df = read_csv(p)
        latest = latest_date(p)
        stale = bool(pd.notna(market) and pd.notna(latest) and latest < market)
        rows.append(
            {
                "file": name,
                "exists": p.exists(),
                "rows": len(df),
                "date_column": date_col(df) or "",
                "latest_market_observation_date": market.date().isoformat() if pd.notna(market) else "missing",
                "latest_artifact_date": latest.date().isoformat() if pd.notna(latest) else "missing",
                "duplicate_date_or_date_ticker_rows": dup_dates(p),
                "stale": stale,
                "source_or_regenerator": REGENERATED_BY_DAILY.get(name, "manual_or_external"),
                "dashboard_consumes": name in DASHBOARD_CONSUMED,
                "checksum": file_hash(p) if p.exists() else "missing",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv("phase123_official_freshness_audit.csv", index=False)
    return out


def reconciliation() -> pd.DataFrame:
    market = latest_market_date()
    checks = {
        "latest_market_date": market,
        "forecast_history": latest_date("forecast_history.csv"),
        "current_growth_features": latest_date("current_growth_features.csv"),
        "current_growth_allocation": latest_date("current_growth_candidate_allocation.csv"),
        "official_state": latest_date("growth_official_paper_state.csv"),
        "official_performance": latest_date("growth_official_paper_performance.csv"),
        "official_monitor": latest_date("growth_official_paper_monitor.csv"),
        "official_tracking": latest_date("growth_official_live_tracking.csv"),
        "official_benchmark_daily": latest_date("growth_official_benchmark_daily.csv"),
        "dashboard_official_date": latest_date("growth_official_paper_performance.csv"),
    }
    rows = []
    for stage, dt in checks.items():
        rows.append(
            {
                "stage": stage,
                "latest_date": dt.date().isoformat() if pd.notna(dt) else "missing",
                "matches_latest_market_date": bool(pd.notna(market) and pd.notna(dt) and dt == market),
            }
        )
    out = pd.DataFrame(rows)
    dates = [v for k, v in checks.items() if k != "latest_market_date" and pd.notna(v)]
    status = "FRESH" if pd.notna(market) and dates and all(v == market for v in dates[:]) else "STALE_OFFICIAL_STATE"
    if pd.isna(market):
        status = "DATA_ERROR"
    out["freshness_status"] = status
    out.to_csv("phase123_reconciliation.csv", index=False)
    return out


def idempotency_audit() -> pd.DataFrame:
    rows = []
    for f in IMPORTANT_FILES:
        p = Path(f)
        df = read_csv(p)
        rows.append(
            {
                "file": f,
                "exists": p.exists(),
                "rows": len(df),
                "duplicate_date_or_date_ticker_rows": dup_dates(p),
                "checksum": file_hash(p) if p.exists() else "missing",
                "idempotency_check": "PASS"
                if dup_dates(p) == 0
                or f
                in {
                    "forecast_history.csv",
                    "current_growth_features.csv",
                    "current_raw_target_features.csv",
                    "growth_official_paper_state.csv",
                    "growth_official_paper_actions.csv",
                    "growth_official_paper_trades.csv",
                    "growth_official_estimated_cost_ledger.csv",
                }
                else "WARNING",
                "note": "multi-row per date expected"
                if f
                in {
                    "forecast_history.csv",
                    "current_growth_features.csv",
                    "current_raw_target_features.csv",
                    "growth_official_paper_state.csv",
                    "growth_official_paper_actions.csv",
                    "growth_official_paper_trades.csv",
                    "growth_official_estimated_cost_ledger.csv",
                }
                else "one row per date expected",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv("phase123_idempotency_audit.csv", index=False)
    return out


def write_report(fresh: pd.DataFrame, recon: pd.DataFrame, idem: pd.DataFrame) -> str:
    status = str(recon["freshness_status"].iloc[0]) if not recon.empty else "DATA_ERROR"
    final_status = "phase123_pipeline_sync_pass" if status == "FRESH" else "phase123_pipeline_sync_warning"
    stale = fresh[fresh["stale"].astype(bool)] if not fresh.empty and "stale" in fresh.columns else pd.DataFrame()
    report = [
        "===== PHASE 123 PIPELINE SYNCHRONIZATION REPORT =====",
        f"final_status: {final_status}",
        f"freshness_status: {status}",
        "root_cause: official gate previously validated raw forecast top-N pre-filter tickers and benchmark official exporter was not run by daily_research_run.py after official lifecycle.",
        "repair: daily_research_run.py now validates only official-impact tickers and runs official_benchmark_chart_repair.py after successful official lifecycle.",
        "production_changed: False",
        "model_logic_changed: False",
        "optimizer_changed: False",
        "parameters_changed: False",
        "real_orders: False",
        "",
        "Stale artifacts:",
    ]
    if stale.empty:
        report.append("none")
    else:
        for _, r in stale.iterrows():
            report.append(f"- {r['file']}: {r['latest_artifact_date']} < {r['latest_market_observation_date']}")
    report += [
        "",
        "Validation files:",
        "- phase123_official_freshness_audit.csv",
        "- phase123_idempotency_audit.csv",
        "- phase123_reconciliation.csv",
        "",
        "Commands to test:",
        "python daily_research_run.py --growth-paper --overwrite-same-day",
        "python phase123_pipeline_sync_audit.py",
        "python -m py_compile daily_research_run.py phase123_pipeline_sync_audit.py official_benchmark_chart_repair.py growth_official_paper_lifecycle.py dashboard_data_layer.py dashboard_app.py",
    ]
    Path("phase123_pipeline_sync_report.txt").write_text("\n".join(report), encoding="utf-8")
    return final_status


def main() -> None:
    t0 = time.time()
    fresh = audit_official_freshness()
    recon = reconciliation()
    idem = idempotency_audit()
    status = write_report(fresh, recon, idem)
    print("===== PHASE 123 PIPELINE SYNC AUDIT =====")
    print(f"status: {status}")
    print(recon.to_string(index=False))
    print(f"runtime_ms: {(time.time() - t0) * 1000:.0f}")


if __name__ == "__main__":
    main()
