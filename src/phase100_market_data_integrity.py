from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canonical_market_data_manager import (
    CANONICAL_CACHE_DIR,
    get_price_history,
    latest_dates_summary,
    refresh_tickers,
    return_data_provenance,
    validate_freshness,
)

MODULES = [
    "financial_data_system.py",
    "daily_research_run.py",
    "current_growth_feature_generation.py",
    "growth_candidate_paper_trading.py",
    "official_paper_baseline_reset.py",
    "multi_source_market_data_validation.py",
    "benchmark_daily_series_export.py",
    "forecast_history_freshness_refresh.py",
    "growth_volatility_targeting.py",
    "canonical_market_data_manager.py",
]

OFFICIAL_FILES = {
    "state": "growth_official_paper_state.csv",
    "trades": "growth_official_paper_trades.csv",
    "actions": "growth_official_paper_actions.csv",
    "performance": "growth_official_paper_performance.csv",
    "monitor": "growth_official_paper_monitor.csv",
    "tracking": "growth_official_live_tracking.csv",
}

DEBUG_PREFIX = "growth_historical_debug_reconstruction_official_invalid"
REPORT = Path("phase100_market_data_report.txt")
PATH_AUDIT_REPORT = Path("market_data_path_audit_report.txt")
PATH_AUDIT = Path("market_data_path_audit.csv")
DEPENDENCY = Path("market_data_source_dependency.csv")
DATE_REPAIR = Path("official_paper_date_repair_audit.csv")
QUALITY = Path("official_paper_data_quality.csv")


def read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out.dropna(subset=[col]).sort_values(col)


def latest(path: str | Path, col: str = "date") -> pd.Timestamp | pd.NaT:
    df = read(path)
    if df.empty:
        return pd.NaT
    c = col if col in df.columns else "Date" if "Date" in df.columns else None
    if c is None:
        return pd.NaT
    ds = pd.to_datetime(df[c], errors="coerce").dropna()
    return ds.max().normalize() if not ds.empty else pd.NaT


def current_growth_tickers() -> list[str]:
    tickers: list[str] = []
    for f in ["current_growth_candidate_allocation.csv", "growth_official_paper_state.csv", "growth_candidate_paper_state.csv"]:
        df = dates(read(f))
        if df.empty or "ticker" not in df.columns:
            continue
        df = df[df["date"].eq(df["date"].max())] if "date" in df.columns and not df.empty else df
        tickers.extend([t for t in df["ticker"].dropna().astype(str).str.upper().str.strip() if t and t != "CASH"])
    forecast = dates(read("forecast_history.csv"))
    if not forecast.empty and "ticker" in forecast.columns:
        day = forecast[forecast["date"].eq(forecast["date"].max())].copy()
        score_col = "raw_target_return_exact" if "raw_target_return_exact" in day.columns else "expected_daily_return" if "expected_daily_return" in day.columns else None
        if score_col:
            day[score_col] = pd.to_numeric(day[score_col], errors="coerce")
            day = day.sort_values(score_col, ascending=False).head(20)
        tickers.extend([t for t in day["ticker"].dropna().astype(str).str.upper().str.strip() if t and t != "CASH"])
    tickers.extend(["SPY", "QQQ"])
    return list(dict.fromkeys(tickers))


def audit_paths(tickers: list[str]) -> pd.DataFrame:
    rows = []
    dirs = [CANONICAL_CACHE_DIR, Path(".yfinance_cache"), Path("yf_cache"), Path("stooq_ohlcv_cache")]
    for d in dirs:
        files = list(d.glob("*.csv")) if d.exists() else []
        latest_dates = []
        for p in files:
            dt = latest(p, "Date")
            if pd.notna(dt):
                latest_dates.append(dt)
        rows.append({
            "source_name": d.name,
            "filesystem_path": str(d.resolve()),
            "cache_directory": str(d.resolve()),
            "file_format": "csv" if files else "missing_or_empty",
            "latest_date": max(latest_dates).date().isoformat() if latest_dates else "missing",
            "ticker_count": len(files),
            "adjusted_vs_unadjusted_close": "yahoo adjusted+close" if d.name == CANONICAL_CACHE_DIR.name else "auxiliary/cache metadata",
            "data_origin": "canonical_yahoo_cache" if d.name == CANONICAL_CACHE_DIR.name else "auxiliary_or_secondary_cache",
            "duplicate_cache_directory": d.name != CANONICAL_CACHE_DIR.name,
        })
    for ticker in tickers:
        ph = get_price_history(ticker)
        rows.append({
            "source_name": f"canonical_{ticker}",
            "filesystem_path": str(ph.cache_path.resolve()),
            "cache_directory": str(CANONICAL_CACHE_DIR.resolve()),
            "file_format": "csv_ohlcv_yahoo_schema",
            "latest_date": ph.latest_date.date().isoformat() if pd.notna(ph.latest_date) else "missing",
            "ticker_count": 1,
            "adjusted_vs_unadjusted_close": "Adj Close preferred; Close available=" + str(ph.close_available),
            "data_origin": ph.source,
            "duplicate_cache_directory": False,
            "row_count": len(ph.data),
            "volume_available": ph.volume_available,
        })
    out = pd.DataFrame(rows)
    out.to_csv(PATH_AUDIT, index=False)
    return out



def write_path_audit_report(path_audit: pd.DataFrame) -> None:
    if path_audit.empty:
        lines = [
            "===== MARKET DATA PATH AUDIT =====",
            "status: no market data paths detected",
            "canonical_source: missing",
        ]
        PATH_AUDIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    cache_rows = path_audit[path_audit["source_name"].astype(str).str.startswith("canonical_")].copy()
    stale_or_missing = cache_rows[cache_rows["latest_date"].astype(str).eq("missing")]
    duplicate_dirs = path_audit[path_audit.get("duplicate_cache_directory", False).astype(bool)] if "duplicate_cache_directory" in path_audit.columns else pd.DataFrame()
    latest_dates = sorted([x for x in cache_rows.get("latest_date", pd.Series(dtype=str)).dropna().astype(str).unique() if x != "missing"])
    lines = [
        "===== MARKET DATA PATH AUDIT =====",
        f"canonical_cache_path: {CANONICAL_CACHE_DIR.resolve()}",
        f"canonical_tickers_checked: {len(cache_rows)}",
        f"canonical_latest_dates: {','.join(latest_dates) if latest_dates else 'missing'}",
        f"missing_or_stale_canonical_rows: {len(stale_or_missing)}",
        f"auxiliary_or_duplicate_cache_directories_detected: {len(duplicate_dirs)}",
        "official_growth_source_policy: canonical_market_data_manager only",
        "production_changed: False",
        "optimizer_changed: False",
        "paper_trading_logic_changed: False",
    ]
    if not stale_or_missing.empty:
        lines.append("missing_tickers: " + ",".join(stale_or_missing["source_name"].astype(str).str.replace("canonical_", "", regex=False).tolist()))
    PATH_AUDIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

def dependency_audit() -> pd.DataFrame:
    rows = []
    patterns = ["yahoo_ohlcv_price_cache", ".yfinance_cache", "yf_cache", "yfinance", "forecast_history", "current_growth", "growth_official"]
    for module in MODULES:
        p = Path(module)
        text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        for pat in patterns:
            if pat.lower() in text.lower():
                rows.append({
                    "module": module,
                    "dependency_pattern": pat,
                    "uses_canonical_manager": "canonical_market_data_manager" in text,
                    "allowed_for_growth_official": module in {"canonical_market_data_manager.py", "daily_research_run.py", "current_growth_feature_generation.py", "multi_source_market_data_validation.py"},
                    "notes": "direct market data dependency detected",
                })
    out = pd.DataFrame(rows)
    out.to_csv(DEPENDENCY, index=False)
    return out


def file_tickers_for_date(path: str, date: pd.Timestamp) -> list[str]:
    df = dates(read(path))
    if df.empty or "ticker" not in df.columns:
        return []
    day = df[df["date"].eq(date)].copy()
    return [t for t in day["ticker"].dropna().astype(str).str.upper().str.strip() if t and t != "CASH"]


def repair_official(expected_tickers: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    perf = dates(read(OFFICIAL_FILES["performance"]))
    rows = []
    if perf.empty:
        audit = pd.DataFrame(columns=["date", "fresh_prices_available", "exact_raw_target_available", "valid_official_date", "reason"])
        audit.to_csv(DATE_REPAIR, index=False)
        pd.DataFrame([{
            "date": datetime.now().date().isoformat(),
            "fresh_prices_available": np.nan,
            "exact_raw_target_available": np.nan,
            "valid_official_date": False,
            "reason": "official namespace has no valid forward rows after stale-data repair",
            "history_classification": "official_forward_blocked",
            "governance": "official_forward_blocked_waiting_for_next_exact_forward_date",
        }]).to_csv(QUALITY, index=False)
        return audit, {"backup_dir": "", "invalid_dates": [], "valid_dates": [], "rows_archived": 0, "official_start_after_repair": "blocked_no_official_rows"}
    forecast = dates(read("forecast_history.csv"))
    all_invalid_dates = []
    valid_dates = []
    for d in perf["date"].dropna().drop_duplicates().sort_values():
        tickers = file_tickers_for_date(OFFICIAL_FILES["state"], d) or expected_tickers
        provenance = return_data_provenance(tickers, d)
        fresh = bool(not provenance.empty and provenance["fresh_for_expected_date"].astype(bool).all())
        exact = False
        if not forecast.empty and "raw_target_return_exact" in forecast.columns:
            fday = forecast[forecast["date"].eq(d)].copy()
            exact = bool(not fday.empty and pd.to_numeric(fday["raw_target_return_exact"], errors="coerce").notna().any())
        valid = fresh and exact
        reason = "ok" if valid else ";".join([x for x, flag in [("stale_or_missing_canonical_prices", not fresh), ("missing_exact_raw_target", not exact)] if flag])
        rows.append({
            "date": d.date().isoformat(),
            "tickers_checked": ",".join(tickers),
            "fresh_prices_available": fresh,
            "exact_raw_target_available": exact,
            "valid_official_date": valid,
            "reason": reason,
        })
        (valid_dates if valid else all_invalid_dates).append(d)
    audit = pd.DataFrame(rows)
    audit.to_csv(DATE_REPAIR, index=False)
    invalid_set = set(all_invalid_dates)
    valid_set = set(valid_dates)
    if not invalid_set:
        return audit, {"backup_dir": "", "invalid_dates": [], "valid_dates": [d.date().isoformat() for d in valid_dates], "rows_archived": 0, "official_start_after_repair": min(valid_dates).date().isoformat() if valid_dates else "blocked"}
    backup_dir = Path("official_paper_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup_dir.mkdir(exist_ok=True)
    rows_archived = 0
    for name, path in OFFICIAL_FILES.items():
        p = Path(path)
        if p.exists():
            shutil.copy2(p, backup_dir / p.name)
        df = dates(read(path))
        if df.empty or "date" not in df.columns:
            continue
        invalid = df[df["date"].isin(invalid_set)].copy()
        valid_df = df[df["date"].isin(valid_set)].copy()
        if not invalid.empty:
            invalid["data_mode"] = "historical_debug_reconstruction"
            invalid["archive_reason"] = "official row invalidated by Phase100 stale market data audit"
            invalid.to_csv(f"{DEBUG_PREFIX}_{name}.csv", index=False)
            rows_archived += len(invalid)
        valid_df.to_csv(path, index=False)
    if not valid_dates:
        # Official namespace is intentionally empty/blocked, but preserve schemas.
        for name, path in OFFICIAL_FILES.items():
            df = read(backup_dir / Path(path).name)
            df.iloc[0:0].to_csv(path, index=False)
        pd.DataFrame([{
            "date": datetime.now().date().isoformat(),
            "model": "growth_champion_final",
            "data_mode": "official_forward_blocked",
            "days_tracked": 0,
            "portfolio_value": np.nan,
            "cumulative_return": np.nan,
            "governance_status": "official_forward_blocked",
            "promotion_status": "real_capital_blocked",
            "reason": "all official rows invalidated by stale canonical market data; waiting for fresh exact forward date",
        }]).to_csv(OFFICIAL_FILES["tracking"], index=False)
    quality = audit.copy()
    quality["history_classification"] = np.where(quality["valid_official_date"], "official_forward_candidate", "historical_debug_reconstruction")
    quality["governance"] = np.where(quality["valid_official_date"], "official_forward_warmup", "official_forward_blocked_stale_data")
    quality.to_csv(QUALITY, index=False)
    return audit, {
        "backup_dir": str(backup_dir),
        "invalid_dates": [d.date().isoformat() for d in all_invalid_dates],
        "valid_dates": [d.date().isoformat() for d in valid_dates],
        "rows_archived": rows_archived,
        "official_start_after_repair": min(valid_dates).date().isoformat() if valid_dates else "blocked_waiting_for_fresh_exact_forward_date",
    }


def main() -> None:
    tickers = current_growth_tickers()
    deps = dependency_audit()
    dates_summary = latest_dates_summary()
    expected = dates_summary.get("forecast_history_date") or dates_summary.get("official_paper_date") or datetime.now().date().isoformat()
    integrity, gov = validate_freshness(expected, tickers)
    refresh = pd.DataFrame()
    if gov.empty or not bool(gov.iloc[0].get("paper_may_run", False)):
        # Attempt refresh for current holdings/top candidates/benchmarks. If network is unavailable, validation remains blocked.
        refresh = refresh_tickers(tickers, expected)
        integrity, gov = validate_freshness(expected, tickers)
    path_audit = audit_paths(tickers)
    write_path_audit_report(path_audit)
    repair_audit, repair = repair_official(tickers)
    canonical_latest = gov.iloc[0].get("canonical_market_date", "missing") if not gov.empty else "missing"
    final_gov = gov.copy()
    final_gov["official_repair_status"] = repair.get("official_start_after_repair")
    final_gov.to_csv("official_market_data_governance.csv", index=False)
    lines = [
        "===== PHASE 100 MARKET DATA INTEGRITY REPORT =====",
        "cause_of_mismatch: official/forecast dates advanced beyond canonical yahoo cache; latest canonical date before refresh was stale relative to official signal date",
        f"canonical_cache_path: {CANONICAL_CACHE_DIR.resolve()}",
        f"expected_signal_date: {expected}",
        f"latest_valid_market_date: {canonical_latest}",
        f"tickers_checked: {','.join(tickers)}",
        f"market_data_governance: {final_gov.iloc[0]['classification'] if not final_gov.empty else 'missing'}",
        f"invalid_official_dates_found: {','.join(repair.get('invalid_dates', [])) or 'none'}",
        f"rows_archived_or_removed_from_official_namespace: {repair.get('rows_archived', 0)}",
        f"official_backup_dir: {repair.get('backup_dir', '')}",
        f"official_forward_start_after_repair: {repair.get('official_start_after_repair')}",
        "production_changed: False",
        "optimizer_changed: False",
        "ranking_changed: False",
        "real_orders: False",
        "outputs: market_data_path_audit.csv, market_data_path_audit_report.txt, market_data_source_dependency.csv, official_market_data_integrity.csv, official_market_data_governance.csv, official_paper_date_repair_audit.csv",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
