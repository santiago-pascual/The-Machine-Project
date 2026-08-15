from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_CACHE_DIR = Path("yahoo_ohlcv_price_cache")
PROVENANCE_FILE = Path("canonical_market_data_provenance.csv")
INTEGRITY_FILE = Path("official_market_data_integrity.csv")
GOVERNANCE_FILE = Path("official_market_data_governance.csv")
TRADING_SESSION_TOLERANCE = 0


@dataclass
class PriceHistory:
    ticker: str
    data: pd.DataFrame
    cache_path: Path
    source: str
    latest_date: pd.Timestamp | pd.NaT
    adjusted_close_available: bool
    close_available: bool
    volume_available: bool
    error: str = ""


def _read(path: str | Path) -> pd.DataFrame:
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


def _date_col(df: pd.DataFrame) -> str | None:
    for col in ["Date", "date", "Datetime", "timestamp"]:
        if col in df.columns:
            return col
    return None


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    dc = _date_col(out)
    if dc is None:
        return pd.DataFrame()
    out["Date"] = pd.to_datetime(out[dc], errors="coerce").dt.tz_localize(None).dt.normalize()
    rename = {}
    for src, dst in [
        ("Open", "Open"),
        ("High", "High"),
        ("Low", "Low"),
        ("Close", "Close"),
        ("Adj Close", "Adj Close"),
        ("Volume", "Volume"),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("adj_close", "Adj Close"),
        ("volume", "Volume"),
    ]:
        if src in out.columns and dst not in rename.values():
            rename[src] = dst
    out = out.rename(columns=rename)
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    out = out[keep].dropna(subset=["Date"]).copy()
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values("Date").drop_duplicates("Date", keep="last")
    return out


def _latest_date_in_file(path: str | Path, date_col: str = "date") -> pd.Timestamp | pd.NaT:
    df = _read(path)
    if df.empty:
        return pd.NaT
    col = date_col if date_col in df.columns else ("Date" if "Date" in df.columns else None)
    if col is None:
        return pd.NaT
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    return dates.max().normalize() if not dates.empty else pd.NaT


def _trading_gap(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    lo, hi = sorted([pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()])
    return float(max(0, len(pd.bdate_range(lo, hi)) - 1))


def _download_yfinance(ticker: str, start: pd.Timestamp | None, end: pd.Timestamp, timeout_retries: int = 2) -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"yfinance_import_failed: {exc}"
    try:
        yf_cache = Path(".yfinance_cache")
        yf_cache.mkdir(exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(yf_cache.resolve()))
    except Exception:
        pass
    last_error = ""
    for attempt in range(max(1, timeout_retries + 1)):
        try:
            kwargs = {"progress": False, "auto_adjust": False}
            if start is None:
                kwargs["period"] = "2y"
            else:
                kwargs["start"] = start.strftime("%Y-%m-%d")
                kwargs["end"] = (end + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            df = yf.download(ticker, **kwargs)
            if df is None or df.empty:
                last_error = "empty_yfinance_response"
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.reset_index()
                norm = _normalize_ohlcv(df)
                if not norm.empty:
                    return norm, "yfinance_live"
                last_error = "unexpected_yfinance_schema"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempt < timeout_retries:
            time.sleep(min(2**attempt, 5))
    return pd.DataFrame(), last_error


def get_price_history(ticker: str, end_date: str | pd.Timestamp | None = None) -> PriceHistory:
    ticker = str(ticker).upper().strip()
    cache_path = CANONICAL_CACHE_DIR / f"{ticker}.csv"
    raw = _read(cache_path)
    data = _normalize_ohlcv(raw)
    if end_date is not None and not data.empty:
        data = data[data["Date"].le(pd.Timestamp(end_date).normalize())].copy()
    latest = data["Date"].max().normalize() if not data.empty else pd.NaT
    return PriceHistory(
        ticker=ticker,
        data=data,
        cache_path=cache_path,
        source="canonical_yahoo_cache" if cache_path.exists() else "missing_cache",
        latest_date=latest,
        adjusted_close_available="Adj Close" in data.columns and data["Adj Close"].notna().any() if not data.empty else False,
        close_available="Close" in data.columns and data["Close"].notna().any() if not data.empty else False,
        volume_available="Volume" in data.columns and data["Volume"].notna().any() if not data.empty else False,
    )


def get_latest_market_date(tickers: Iterable[str] | None = None) -> pd.Timestamp | pd.NaT:
    if tickers is None:
        if not CANONICAL_CACHE_DIR.exists():
            return pd.NaT
        tickers = [p.stem for p in CANONICAL_CACHE_DIR.glob("*.csv")]
    dates = []
    for ticker in tickers:
        ph = get_price_history(str(ticker))
        if pd.notna(ph.latest_date):
            dates.append(ph.latest_date)
    return min(dates) if dates else pd.NaT


def refresh_tickers(tickers: Iterable[str], expected_date: str | pd.Timestamp, retries: int | None = None) -> pd.DataFrame:
    CANONICAL_CACHE_DIR.mkdir(exist_ok=True)
    expected = pd.Timestamp(expected_date).normalize()
    retries = int(os.getenv("CANONICAL_MARKET_DATA_RETRIES", str(retries if retries is not None else 2)) or 2)
    rows = []
    for ticker_raw in tickers:
        ticker = str(ticker_raw).upper().strip()
        if not ticker or ticker == "CASH":
            continue
        before = get_price_history(ticker)
        start = None
        if pd.notna(before.latest_date):
            start = before.latest_date - pd.Timedelta(days=10)
        downloaded, status = _download_yfinance(ticker, start, expected, retries)
        combined = before.data.copy()
        if not downloaded.empty:
            combined = pd.concat([combined, downloaded], ignore_index=True) if not combined.empty else downloaded
            combined = _normalize_ohlcv(combined)
            combined.to_csv(CANONICAL_CACHE_DIR / f"{ticker}.csv", index=False)
        after = get_price_history(ticker)
        rows.append(
            {
                "ticker": ticker,
                "cache_path": str((CANONICAL_CACHE_DIR / f"{ticker}.csv").resolve()),
                "expected_date": expected.date().isoformat(),
                "latest_before": before.latest_date.date().isoformat() if pd.notna(before.latest_date) else "missing",
                "latest_after": after.latest_date.date().isoformat() if pd.notna(after.latest_date) else "missing",
                "refresh_attempted": True,
                "refresh_success": pd.notna(after.latest_date) and after.latest_date >= expected,
                "download_status": status,
                "rows_before": len(before.data),
                "rows_after": len(after.data),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(PROVENANCE_FILE, index=False)
    return out


def return_data_provenance(tickers: Iterable[str], expected_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    expected = pd.Timestamp(expected_date).normalize() if expected_date is not None and pd.notna(expected_date) else pd.NaT
    for ticker_raw in tickers:
        ticker = str(ticker_raw).upper().strip()
        if not ticker or ticker == "CASH":
            continue
        ph = get_price_history(ticker)
        rows.append(
            {
                "ticker": ticker,
                "canonical_cache_dir": str(CANONICAL_CACHE_DIR.resolve()),
                "cache_path": str(ph.cache_path.resolve()),
                "exists": ph.cache_path.exists(),
                "source": ph.source,
                "file_format": "csv_ohlcv_yahoo_schema",
                "latest_date": ph.latest_date.date().isoformat() if pd.notna(ph.latest_date) else "missing",
                "expected_date": expected.date().isoformat() if pd.notna(expected) else "",
                "row_count": len(ph.data),
                "adjusted_close_available": ph.adjusted_close_available,
                "close_available": ph.close_available,
                "volume_available": ph.volume_available,
                "fresh_for_expected_date": bool(pd.notna(expected) and pd.notna(ph.latest_date) and ph.latest_date >= expected),
                "trading_session_gap": _trading_gap(ph.latest_date, expected) if pd.notna(expected) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_freshness(expected_date: str | pd.Timestamp, tickers: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = pd.Timestamp(expected_date).normalize()
    integrity = return_data_provenance(tickers, expected)
    if integrity.empty:
        classification = "STALE_DATA_BLOCKED"
        reason = "no canonical market data rows"
        canonical_latest = "missing"
        stale = True
    else:
        integrity["is_fresh"] = integrity["fresh_for_expected_date"].astype(bool)
        stale_rows = integrity[~integrity["is_fresh"]]
        stale = not stale_rows.empty
        canonical_latest_ts = pd.to_datetime(integrity["latest_date"], errors="coerce").min()
        canonical_latest = canonical_latest_ts.date().isoformat() if pd.notna(canonical_latest_ts) else "missing"
        classification = "STALE_DATA_BLOCKED" if stale else "SINGLE_SOURCE_FRESH"
        reason = (
            "stale canonical Yahoo cache for: " + ",".join(stale_rows["ticker"].astype(str).tolist())
            if stale
            else "canonical Yahoo cache fresh; no secondary confirmation required for paper"
        )
    integrity.to_csv(INTEGRITY_FILE, index=False)
    governance = pd.DataFrame(
        [
            {
                "expected_signal_date": expected.date().isoformat(),
                "canonical_market_date": canonical_latest,
                "classification": classification,
                "paper_may_run": not stale,
                "block_new_rebalance": stale,
                "do_not_advance_official_paper": stale,
                "real_capital_blocked": True,
                "reason": reason,
                "canonical_cache_dir": str(CANONICAL_CACHE_DIR.resolve()),
                "production_changed": False,
                "paper_logic_changed": False,
            }
        ]
    )
    governance.to_csv(GOVERNANCE_FILE, index=False)
    return integrity, governance


def latest_dates_summary() -> dict[str, str]:
    files = {
        "forecast_history_date": "forecast_history.csv",
        "raw_target_date": "current_raw_target_features.csv",
        "growth_features_date": "current_growth_features.csv",
        "allocation_signal_date": "current_growth_candidate_allocation.csv",
        "official_paper_date": "growth_official_paper_performance.csv",
    }
    out = {k: (_latest_date_in_file(v).date().isoformat() if pd.notna(_latest_date_in_file(v)) else "missing") for k, v in files.items()}
    return out


if __name__ == "__main__":
    # Lightweight CLI diagnostic.
    tickers = []
    for f in ["current_growth_candidate_allocation.csv", "growth_official_paper_state.csv"]:
        df = _read(f)
        if not df.empty and "ticker" in df.columns:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df[df["date"].eq(df["date"].max())]
            tickers.extend([t for t in df["ticker"].dropna().astype(str).str.upper() if t != "CASH"])
    tickers.extend(["SPY", "QQQ"])
    tickers = list(dict.fromkeys(tickers))
    expected = latest_dates_summary().get("forecast_history_date", "") or pd.Timestamp.today().date().isoformat()
    validate_freshness(expected, tickers)
    print("===== CANONICAL MARKET DATA MANAGER =====")
    print(f"canonical_cache_dir: {CANONICAL_CACHE_DIR.resolve()}")
    print(f"expected_date: {expected}")
    print(f"tickers: {','.join(tickers)}")
    print(f"latest_market_date: {get_latest_market_date(tickers)}")
