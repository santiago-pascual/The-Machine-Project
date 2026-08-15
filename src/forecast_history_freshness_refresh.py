from __future__ import annotations

from pathlib import Path

import pandas as pd

FORECAST_HISTORY = Path("forecast_history.csv")
CACHE_DIR = Path("yahoo_ohlcv_price_cache")
REPORT_FILE = Path("forecast_history_freshness_refresh_report.csv")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _latest_cache_date() -> pd.Timestamp | pd.NaT:
    dates: list[pd.Timestamp] = []
    if not CACHE_DIR.exists():
        return pd.NaT
    for path in CACHE_DIR.glob("*.csv"):
        df = _read(path)
        if df.empty:
            continue
        col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
        if col is None:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce").dropna()
        if not parsed.empty:
            dates.append(parsed.max().normalize())
    return max(dates) if dates else pd.NaT


def _latest_forecast_date(history: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    if history.empty or "date" not in history.columns:
        return pd.NaT
    dates = pd.to_datetime(history["date"], errors="coerce").dropna()
    return dates.max().normalize() if not dates.empty else pd.NaT


def _cache_price(ticker: str, date: pd.Timestamp) -> float | None:
    path = CACHE_DIR / f"{str(ticker).upper()}.csv"
    df = _read(path)
    if df.empty:
        return None
    col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
    if col is None:
        return None
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    df = df[df[col].le(date)].sort_values(col)
    if df.empty:
        return None
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        return None
    value = pd.to_numeric(df.iloc[-1][price_col], errors="coerce")
    return float(value) if pd.notna(value) else None


def refresh_forecast_history_to_cache_latest(overwrite_same_day: bool = True) -> dict[str, object]:
    history = _read(FORECAST_HISTORY)
    cache_latest = _latest_cache_date()
    before_latest = _latest_forecast_date(history)
    report = {
        "cache_latest_date": cache_latest.date().isoformat() if pd.notna(cache_latest) else "missing",
        "forecast_history_latest_before": before_latest.date().isoformat() if pd.notna(before_latest) else "missing",
        "forecast_history_latest_after": before_latest.date().isoformat() if pd.notna(before_latest) else "missing",
        "rows_added": 0,
        "rows_overwritten": 0,
        "rows_with_cache_price_update": 0,
        "status": "skipped",
        "method": "none",
        "warning": "",
    }
    if history.empty or "date" not in history.columns or "ticker" not in history.columns:
        report["status"] = "error"
        report["warning"] = "forecast_history missing required columns"
        pd.DataFrame([report]).to_csv(REPORT_FILE, index=False)
        return report
    if pd.isna(cache_latest):
        report["warning"] = "cache latest date unavailable"
        pd.DataFrame([report]).to_csv(REPORT_FILE, index=False)
        return report
    if pd.notna(before_latest) and before_latest >= cache_latest:
        report["status"] = "fresh"
        pd.DataFrame([report]).to_csv(REPORT_FILE, index=False)
        return report

    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    latest_rows = history[history["date"].eq(before_latest)].copy()
    if latest_rows.empty:
        report["status"] = "error"
        report["warning"] = "no latest forecast rows to carry forward"
        pd.DataFrame([report]).to_csv(REPORT_FILE, index=False)
        return report

    new_rows = latest_rows.copy()
    new_rows["date"] = cache_latest.strftime("%Y-%m-%d")
    updates = 0
    if "current_price" in new_rows.columns:
        for idx, row in new_rows.iterrows():
            price = _cache_price(str(row["ticker"]), cache_latest)
            if price is not None:
                new_rows.at[idx, "current_price"] = price
                updates += 1
    new_rows["freshness_refresh_method"] = "carry_forward_latest_forecast_snapshot"
    new_rows["freshness_refresh_source_date"] = before_latest.strftime("%Y-%m-%d") if pd.notna(before_latest) else ""
    new_rows["freshness_refresh_warning"] = (
        "Paper/research freshness bridge: forecast model was not recalculated for this date; "
        "latest forecast snapshot was carried forward because Yahoo/cache had a newer market date."
    )

    key_cols = ["date", "ticker"]
    existing = history.copy()
    existing["date"] = existing["date"].dt.strftime("%Y-%m-%d")
    new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    target_keys = set(new_rows[key_cols].itertuples(index=False, name=None))
    rows_overwritten = 0
    if overwrite_same_day:
        mask = existing[key_cols].apply(tuple, axis=1).isin(target_keys)
        rows_overwritten = int(mask.sum())
        existing = existing.loc[~mask].copy()
    else:
        existing_keys = set(existing[key_cols].itertuples(index=False, name=None))
        new_rows = new_rows.loc[~new_rows[key_cols].apply(tuple, axis=1).isin(existing_keys)].copy()

    combined_cols = list(dict.fromkeys(list(existing.columns) + list(new_rows.columns)))
    combined = pd.concat([existing.reindex(columns=combined_cols), new_rows.reindex(columns=combined_cols)], ignore_index=True)
    combined.to_csv(FORECAST_HISTORY, index=False)
    after = _latest_forecast_date(combined)
    report.update(
        {
            "forecast_history_latest_after": after.date().isoformat() if pd.notna(after) else "missing",
            "rows_added": len(new_rows),
            "rows_overwritten": rows_overwritten,
            "rows_with_cache_price_update": updates,
            "status": "refreshed",
            "method": "carry_forward_latest_forecast_snapshot",
            "warning": "forecast snapshot was carried forward; model calculations were not changed or rerun for this bridge",
        }
    )
    pd.DataFrame([report]).to_csv(REPORT_FILE, index=False)
    return report


if __name__ == "__main__":
    result = refresh_forecast_history_to_cache_latest(overwrite_same_day=True)
    print("===== FORECAST HISTORY FRESHNESS REFRESH =====")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"Saved: {REPORT_FILE.resolve()}")
