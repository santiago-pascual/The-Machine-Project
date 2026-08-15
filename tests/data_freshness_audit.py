
from __future__ import annotations

from pathlib import Path

import pandas as pd

AUDIT_OUT = Path("data_freshness_audit.csv")
SUMMARY_OUT = Path("data_freshness_summary.txt")
CACHE_DIR = Path("yahoo_ohlcv_price_cache")

STAGES = [
    ("forecast_history", Path("forecast_history.csv"), "date"),
    ("current_raw_target_features", Path("current_raw_target_features.csv"), "date"),
    ("current_growth_features", Path("current_growth_features.csv"), "date"),
    ("current_growth_candidate_allocation", Path("current_growth_candidate_allocation.csv"), "date"),
    ("growth_candidate_paper_state", Path("growth_candidate_paper_state.csv"), "date"),
    ("growth_candidate_paper_performance", Path("growth_candidate_paper_performance.csv"), "date"),
    ("benchmark_daily_returns", Path("benchmark_daily_returns.csv"), "date"),
    ("growth_live_tracking", Path("growth_live_tracking.csv"), "date"),
]


def _last_business_day(today: pd.Timestamp) -> pd.Timestamp:
    d = today.normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _latest_csv_date(path: Path, date_col: str = "date") -> tuple[pd.Timestamp | pd.NaT, int, str]:
    if not path.exists():
        return pd.NaT, 0, "missing_file"
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return pd.NaT, 0, f"read_error: {exc}"
    if df.empty:
        return pd.NaT, 0, "empty_file"
    if date_col not in df.columns:
        return pd.NaT, len(df), f"missing_date_column:{date_col}"
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return pd.NaT, len(df), "no_parseable_dates"
    return dates.max().normalize(), len(df), "ok"


def _cache_dates() -> dict[str, object]:
    result = {
        "latest_cache_date_any": pd.NaT,
        "latest_cache_date_spy": pd.NaT,
        "latest_cache_date_qqq": pd.NaT,
        "cache_file_count": 0,
        "cache_status": "missing_cache_dir",
    }
    if not CACHE_DIR.exists():
        return result
    files = list(CACHE_DIR.glob("*.csv"))
    result["cache_file_count"] = len(files)
    latest_dates = []
    for path in files:
        latest, _, status = _latest_csv_date(path, "Date")
        if pd.isna(latest):
            latest, _, status = _latest_csv_date(path, "date")
        if pd.notna(latest):
            latest_dates.append(latest)
        if path.stem.upper() == "SPY":
            result["latest_cache_date_spy"] = latest
        if path.stem.upper() == "QQQ":
            result["latest_cache_date_qqq"] = latest
    result["latest_cache_date_any"] = max(latest_dates) if latest_dates else pd.NaT
    result["cache_status"] = "ok" if latest_dates else "no_parseable_cache_dates"
    return result


def _script_download_info() -> dict[str, str]:
    info = {
        "financial_data_system_download_request": "unknown",
        "benchmark_download_request": "unknown",
    }
    fds = Path("financial_data_system.py")
    if fds.exists():
        text = fds.read_text(encoding="utf-8", errors="ignore")
        if "end_date = datetime.today()" in text:
            info["financial_data_system_download_request"] = "uses datetime.today() as yfinance end date for day-based downloads; period mode also used when configured"
        if "yf.download" in text:
            info["financial_data_system_yfinance"] = "present"
    bench = Path("benchmark_daily_series_export.py")
    if bench.exists():
        text = bench.read_text(encoding="utf-8", errors="ignore")
        if "end + pd.Timedelta(days=5)" in text:
            info["benchmark_download_request"] = "downloads through last paper date + 5 calendar days; Yahoo end is exclusive"
    return info


def run_audit() -> tuple[pd.DataFrame, str]:
    today = pd.Timestamp.today().normalize()
    expected = _last_business_day(today)
    cache = _cache_dates()
    script_info = _script_download_info()
    rows = []

    rows.append({
        "stage": "financial_data_system_market_download/cache",
        "file": "financial_data_system.py / yahoo_ohlcv_price_cache",
        "expected_latest_market_date": expected.date().isoformat(),
        "actual_latest_date": cache["latest_cache_date_any"].date().isoformat() if pd.notna(cache["latest_cache_date_any"]) else "",
        "stale": bool(pd.isna(cache["latest_cache_date_any"]) or cache["latest_cache_date_any"] < expected),
        "rows": cache["cache_file_count"],
        "status": cache["cache_status"],
        "notes": script_info.get("financial_data_system_download_request", "unknown"),
    })
    rows.append({
        "stage": "yahoo_cache_SPY",
        "file": str(CACHE_DIR / "SPY.csv"),
        "expected_latest_market_date": expected.date().isoformat(),
        "actual_latest_date": cache["latest_cache_date_spy"].date().isoformat() if pd.notna(cache["latest_cache_date_spy"]) else "",
        "stale": bool(pd.isna(cache["latest_cache_date_spy"]) or cache["latest_cache_date_spy"] < expected),
        "rows": "",
        "status": "ok" if pd.notna(cache["latest_cache_date_spy"]) else "missing_or_unparseable",
        "notes": script_info.get("benchmark_download_request", "unknown"),
    })
    rows.append({
        "stage": "yahoo_cache_QQQ",
        "file": str(CACHE_DIR / "QQQ.csv"),
        "expected_latest_market_date": expected.date().isoformat(),
        "actual_latest_date": cache["latest_cache_date_qqq"].date().isoformat() if pd.notna(cache["latest_cache_date_qqq"]) else "",
        "stale": bool(pd.isna(cache["latest_cache_date_qqq"]) or cache["latest_cache_date_qqq"] < expected),
        "rows": "",
        "status": "ok" if pd.notna(cache["latest_cache_date_qqq"]) else "missing_or_unparseable",
        "notes": script_info.get("benchmark_download_request", "unknown"),
    })

    stage_dates = {}
    for stage, path, col in STAGES:
        latest, row_count, status = _latest_csv_date(path, col)
        stage_dates[stage] = latest
        rows.append({
            "stage": stage,
            "file": str(path),
            "expected_latest_market_date": expected.date().isoformat(),
            "actual_latest_date": latest.date().isoformat() if pd.notna(latest) else "",
            "stale": bool(pd.isna(latest) or latest < expected),
            "rows": row_count,
            "status": status,
            "notes": "",
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_OUT, index=False)

    stale_rows = audit[audit["stale"].astype(bool)].copy()
    blocking_stage = "none"
    reason = "all audited stages are fresh"
    forecast_latest = stage_dates.get("forecast_history", pd.NaT)
    growth_latest = stage_dates.get("current_growth_features", pd.NaT)
    allocation_latest = stage_dates.get("current_growth_candidate_allocation", pd.NaT)
    paper_latest = stage_dates.get("growth_candidate_paper_state", pd.NaT)
    if pd.isna(forecast_latest) or forecast_latest < expected:
        blocking_stage = "forecast_history.csv"
        reason = "current_growth_feature_generation reads the latest date from forecast_history.csv; if forecast_history is stale, growth features/allocation/paper state cannot advance."
    elif pd.isna(growth_latest) or growth_latest < forecast_latest:
        blocking_stage = "current_growth_feature_generation.py / current_growth_features.csv"
        reason = "forecast_history is fresher than growth features, so current growth feature generation did not advance."
    elif pd.isna(allocation_latest) or allocation_latest < growth_latest:
        blocking_stage = "current_growth_candidate_allocation.csv"
        reason = "growth allocation is older than growth features."
    elif pd.isna(paper_latest) or paper_latest < allocation_latest:
        blocking_stage = "growth_candidate_paper_trading.py / growth_candidate_paper_state.csv"
        reason = "paper state is older than current growth allocation."
    elif not stale_rows.empty:
        blocking_stage = str(stale_rows.iloc[0]["stage"])
        reason = "one or more downstream/benchmark/live tracking files are stale, but core paper allocation appears internally aligned."

    lines = []
    lines.append("===== DATA FRESHNESS AUDIT =====")
    lines.append(f"audit_run_date: {today.date().isoformat()}")
    lines.append(f"expected_latest_market_date: {expected.date().isoformat()}")
    lines.append("")
    lines.append("Stage freshness:")
    for _, row in audit.iterrows():
        lines.append(f"- {row['stage']}: actual={row['actual_latest_date'] or 'missing'} expected={row['expected_latest_market_date']} stale={row['stale']} status={row['status']}")
    lines.append("")
    lines.append(f"Blocking stage: {blocking_stage}")
    lines.append(f"Explanation: {reason}")
    lines.append("")
    lines.append("Important note: if the audit is run before Yahoo has published today's close, the expected latest market date may be same-day while Yahoo/cache legitimately remains on the prior completed close.")
    summary = "\n".join(lines)
    SUMMARY_OUT.write_text(summary, encoding="utf-8")
    return audit, summary


if __name__ == "__main__":
    audit, summary = run_audit()
    print(summary)
    print(f"Saved: {AUDIT_OUT.resolve()}")
    print(f"Saved: {SUMMARY_OUT.resolve()}")
