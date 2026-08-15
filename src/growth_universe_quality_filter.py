from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRICE_CACHE_DIR = Path("yahoo_ohlcv_price_cache")
BLACKLIST_FILE = Path("growth_universe_blacklist.csv")
REPORT_FILE = Path("growth_universe_quality_report.csv")
EXCLUSIONS_FILE = Path("growth_universe_exclusions.csv")
CURRENT_FILE = Path("current_growth_universe_quality.csv")
ALLOWLIST_FILE = Path("growth_institutional_allowlist.csv")
TRADABILITY_REPORT_FILE = Path("growth_tradability_filter_report.csv")
TRADABILITY_EXCLUSIONS_FILE = Path("growth_tradability_exclusions.csv")
MARKET_CAP_CACHE_FILE = Path("market_cap_cache.csv")

MIN_PRICE = 5.0
MIN_MEDIAN_20D_DOLLAR_VOLUME = 20_000_000.0
MIN_HISTORY_DAYS = 252
MAX_1D_ABS_MOVE_60D = 0.80
MAX_20D_REALIZED_VOL = 1.50
MAX_MISSING_OHLCV_RATIO = 0.05
YAHOO_LOOKBACK_YEARS = 3

INSTITUTIONAL_MIN_MEDIAN_60D_DOLLAR_VOLUME = 50_000_000.0
INSTITUTIONAL_MIN_MARKET_CAP = 2_000_000_000.0
INSTITUTIONAL_MIN_HISTORY_DAYS = 504
INSTITUTIONAL_MAX_60D_REALIZED_VOL = 1.20
INSTITUTIONAL_MIN_AVG_20D_VOLUME = 500_000.0


def ensure_allowlist_template() -> set[str] | None:
    if not ALLOWLIST_FILE.exists():
        pd.DataFrame(columns=["ticker", "notes", "active"]).to_csv(ALLOWLIST_FILE, index=False)
        return None
    try:
        df = pd.read_csv(ALLOWLIST_FILE)
    except Exception:
        return None
    if "ticker" not in df.columns:
        return None
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].ne("") & df["ticker"].ne("NAN")]
    if df.empty:
        return None
    if "active" in df.columns:
        active = df["active"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df = df[active]
    return set(df["ticker"].dropna().astype(str)) if not df.empty else None


def _market_cap_for_ticker(ticker: str) -> float:
    if not MARKET_CAP_CACHE_FILE.exists():
        return np.nan
    try:
        caps = pd.read_csv(MARKET_CAP_CACHE_FILE)
    except Exception:
        return np.nan
    if not {"ticker", "market_cap"}.issubset(caps.columns):
        return np.nan
    row = caps[caps["ticker"].astype(str).str.upper().eq(ticker)]
    if row.empty:
        return np.nan
    return float(pd.to_numeric(row.iloc[-1]["market_cap"], errors="coerce"))


def _is_suspicious_symbol(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    # Avoid broad suffix guesses: common tickers like MSTR, SNOW, UBER, and PBR
    # legitimately end in R/W and must not be treated as warrants/rights.
    explicit_markers = (".WS", "-WS", "/WS", ".WT", "-WT", "/WT", ".W", "-W", "/W", ".U", "-U", "/U", ".R", "-R", "/R")
    if any(marker in t for marker in explicit_markers):
        return True
    # Compact warrant/unit/right symbols are usually longer suffix variants; keep this
    # narrow so ordinary 3-5 letter equity symbols are not falsely excluded.
    return len(t) >= 6 and (t.endswith(("WS", "WT")))


def ensure_blacklist() -> pd.DataFrame:
    if BLACKLIST_FILE.exists():
        df = pd.read_csv(BLACKLIST_FILE)
    else:
        df = pd.DataFrame(
            [
                {
                    "ticker": "AFJK",
                    "reason": "manual blacklist: unstable/low-quality growth paper ticker",
                    "date_added": pd.Timestamp.today().date().isoformat(),
                },
                {
                    "ticker": "AIXI",
                    "reason": "manual blacklist: unstable/low-quality growth paper ticker",
                    "date_added": pd.Timestamp.today().date().isoformat(),
                },
            ]
        )
        df.to_csv(BLACKLIST_FILE, index=False)
    if "ticker" not in df.columns:
        df["ticker"] = []
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if "reason" not in df.columns:
        df["reason"] = "manual blacklist"
    if "date_added" not in df.columns:
        df["date_added"] = ""
    return df.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else ("Datetime" if "Datetime" in df.columns else None))
    if date_col is None:
        return pd.DataFrame()
    out = df.copy()
    out["Date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=["Date"]).sort_values("Date")
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _download_yahoo_ohlcv(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return pd.DataFrame()
    try:
        yf_cache = Path(".yfinance_cache")
        yf_cache.mkdir(exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(yf_cache))
    except Exception:
        pass
    start = (as_of_date - pd.DateOffset(years=YAHOO_LOOKBACK_YEARS)).strftime("%Y-%m-%d")
    end = (as_of_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    except Exception:
        return pd.DataFrame()
    return _standardize_ohlcv(df.reset_index() if df is not None and not df.empty else pd.DataFrame())


def _download_yahoo_ohlcv_batch(tickers: list[str], as_of_date: pd.Timestamp) -> dict[str, bool]:
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return {}
    result = {ticker: False for ticker in tickers}
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return result
    try:
        yf_cache = Path(".yfinance_cache")
        yf_cache.mkdir(exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(yf_cache))
    except Exception:
        pass
    start = (as_of_date - pd.DateOffset(years=YAHOO_LOOKBACK_YEARS)).strftime("%Y-%m-%d")
    end = (as_of_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False, threads=False, group_by="ticker")
    except Exception:
        return result
    if data is None or data.empty:
        return result
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker].reset_index()
            else:
                # yfinance returns a flat frame for a single valid ticker.
                df = data.reset_index() if len(tickers) == 1 else pd.DataFrame()
            clean = _standardize_ohlcv(df)
            if clean.empty:
                continue
            _cache_ohlcv(ticker, clean)
            result[ticker] = True
        except Exception:
            result[ticker] = False
    return result


def _cache_ohlcv(ticker: str, downloaded: pd.DataFrame) -> None:
    if downloaded.empty:
        return
    path = PRICE_CACHE_DIR / f"{ticker.upper()}.csv"
    PRICE_CACHE_DIR.mkdir(exist_ok=True)
    existing = _standardize_ohlcv(pd.read_csv(path)) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, downloaded], ignore_index=True) if not existing.empty else downloaded
    combined = _standardize_ohlcv(combined).drop_duplicates("Date", keep="last").sort_values("Date")
    combined.to_csv(path, index=False)


def _read_cached_ohlcv(ticker: str) -> pd.DataFrame:
    path = PRICE_CACHE_DIR / f"{ticker.upper()}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df = _standardize_ohlcv(df)
    if not df.empty:
        df["date"] = df["Date"]
    return df


def _quality_for_ticker(
    ticker: str,
    as_of_date: pd.Timestamp,
    blacklist: pd.DataFrame,
    allowlist: set[str] | None = None,
    yahoo_attempted: bool = False,
    yahoo_success: bool = False,
) -> dict[str, object]:
    ticker = str(ticker).strip().upper()
    reasons: list[str] = []
    warnings: list[str] = []
    blacklisted = blacklist[blacklist["ticker"].eq(ticker)]
    if not blacklisted.empty:
        reasons.append(str(blacklisted.iloc[-1].get("reason", "manual blacklist")))

    local_ohlcv_available = False
    yahoo_fetch_attempted = False
    yahoo_fetch_success = False
    ohlcv = _read_cached_ohlcv(ticker)
    hist = ohlcv[ohlcv["date"].le(as_of_date)].copy() if not ohlcv.empty else pd.DataFrame()
    local_ohlcv_available = not hist.empty
    yahoo_fetch_attempted = bool(yahoo_attempted)
    yahoo_fetch_success = bool(yahoo_success)
    latest_price = np.nan
    median_20d_dollar_volume = np.nan
    history_days = len(hist)
    max_abs_move_60d = np.nan
    realized_vol_20d = np.nan
    missing_ratio = np.nan
    suspicious_spike = False
    median_60d_dollar_volume = np.nan
    avg_volume_20d = np.nan
    realized_vol_60d = np.nan
    market_cap = _market_cap_for_ticker(ticker)
    tradability_reasons: list[str] = []
    allowlist_active = allowlist is not None
    if allowlist_active and ticker not in allowlist:
        tradability_reasons.append("not in growth institutional allowlist")
    if _is_suspicious_symbol(ticker):
        tradability_reasons.append("SPAC/warrant/rights/unit-like ticker symbol")

    if hist.empty:
        reasons.append("missing OHLCV data")
    else:
        price_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
        latest_price = float(hist[price_col].dropna().iloc[-1]) if not hist[price_col].dropna().empty else np.nan
        if not np.isfinite(latest_price):
            reasons.append("missing latest price")
        elif latest_price < MIN_PRICE:
            reasons.append(f"price below ${MIN_PRICE:g}")

        if "Volume" not in hist.columns or hist["Volume"].dropna().empty:
            reasons.append("missing volume data")
        else:
            dollar_volume = hist[price_col] * hist["Volume"]
            median_20d_dollar_volume = float(dollar_volume.tail(20).median()) if not dollar_volume.tail(20).dropna().empty else np.nan
            median_60d_dollar_volume = float(dollar_volume.tail(60).median()) if not dollar_volume.tail(60).dropna().empty else np.nan
            avg_volume_20d = float(hist["Volume"].tail(20).mean()) if not hist["Volume"].tail(20).dropna().empty else np.nan
            if not np.isfinite(median_20d_dollar_volume) or median_20d_dollar_volume < MIN_MEDIAN_20D_DOLLAR_VOLUME:
                reasons.append(f"median 20D dollar volume below ${MIN_MEDIAN_20D_DOLLAR_VOLUME:,.0f}")
            if not np.isfinite(median_60d_dollar_volume) or median_60d_dollar_volume < INSTITUTIONAL_MIN_MEDIAN_60D_DOLLAR_VOLUME:
                tradability_reasons.append(f"median 60D dollar volume below ${INSTITUTIONAL_MIN_MEDIAN_60D_DOLLAR_VOLUME:,.0f}")
            if not np.isfinite(avg_volume_20d) or avg_volume_20d < INSTITUTIONAL_MIN_AVG_20D_VOLUME:
                tradability_reasons.append(f"20D average volume below {INSTITUTIONAL_MIN_AVG_20D_VOLUME:,.0f} shares")

        if np.isfinite(latest_price) and latest_price < MIN_PRICE:
            tradability_reasons.append(f"current price below ${MIN_PRICE:g}")
        if np.isfinite(market_cap) and market_cap < INSTITUTIONAL_MIN_MARKET_CAP:
            tradability_reasons.append(f"market cap below ${INSTITUTIONAL_MIN_MARKET_CAP:,.0f}")
        if history_days < MIN_HISTORY_DAYS:
            reasons.append(f"less than {MIN_HISTORY_DAYS} trading days of history")
        if history_days < INSTITUTIONAL_MIN_HISTORY_DAYS:
            tradability_reasons.append("less than 2 years of trading history")

        prices = hist[price_col].dropna()
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        if len(returns.dropna()) >= 2:
            max_abs_move_60d = float(returns.tail(60).abs().max())
            if np.isfinite(max_abs_move_60d) and max_abs_move_60d > MAX_1D_ABS_MOVE_60D:
                reasons.append("1D absolute move above 80% in last 60D")
            realized_vol_20d = float(returns.tail(20).std(ddof=0) * np.sqrt(252)) if len(returns.dropna()) >= 20 else np.nan
            realized_vol_60d = float(returns.tail(60).std(ddof=0) * np.sqrt(252)) if len(returns.dropna()) >= 60 else np.nan
            if np.isfinite(realized_vol_20d) and realized_vol_20d > MAX_20D_REALIZED_VOL:
                reasons.append("20D realized volatility above 150% annualized")
            if np.isfinite(realized_vol_60d) and realized_vol_60d > INSTITUTIONAL_MAX_60D_REALIZED_VOL:
                tradability_reasons.append("60D annualized volatility above 120%")
            if returns.tail(252).abs().max(skipna=True) > 0.90:
                tradability_reasons.append("reverse split/extreme discontinuity suspected")
            suspicious_spike = bool(np.isfinite(max_abs_move_60d) and max_abs_move_60d > 0.50)
            if suspicious_spike:
                reasons.append("suspicious spike/gap behavior")

        required_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
        if required_cols:
            missing_ratio = float(hist[required_cols].isna().any(axis=1).mean())
            if missing_ratio > MAX_MISSING_OHLCV_RATIO:
                reasons.append("missing OHLCV data above 5%")
        else:
            reasons.append("missing OHLCV columns")

    passed_tradability_filter = len(tradability_reasons) == 0
    final_reasons = list(dict.fromkeys(reasons + tradability_reasons))
    passed = len(final_reasons) == 0
    return {
        "date": as_of_date.date().isoformat(),
        "ticker": ticker,
        "quality_pass": passed,
        "passed_tradability_filter": passed_tradability_filter,
        "tradability_exclusion_reason": "; ".join(dict.fromkeys(tradability_reasons)),
        "exclusion_reason": "; ".join(final_reasons),
        "final_exclusion_reason": "; ".join(final_reasons),
        "warning": "; ".join(warnings),
        "local_ohlcv_available": local_ohlcv_available,
        "yahoo_fetch_attempted": yahoo_fetch_attempted,
        "yahoo_fetch_success": yahoo_fetch_success,
        "latest_price": latest_price,
        "median_20d_dollar_volume": median_20d_dollar_volume,
        "median_60d_dollar_volume": median_60d_dollar_volume,
        "avg_volume_20d": avg_volume_20d,
        "market_cap": market_cap,
        "history_days": history_days,
        "trading_history_days": history_days,
        "max_abs_1d_move_60d": max_abs_move_60d,
        "realized_vol_20d_ann": realized_vol_20d,
        "realized_vol_60d": realized_vol_60d,
        "missing_ohlcv_ratio": missing_ratio,
        "suspicious_spike_gap": suspicious_spike,
    }


def apply_growth_universe_quality_filter(
    candidates: pd.DataFrame, as_of_date: pd.Timestamp, yahoo_fetch_tickers: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty or "ticker" not in candidates.columns:
        return candidates.copy(), pd.DataFrame(), pd.DataFrame()
    blacklist = ensure_blacklist()
    allowlist = ensure_allowlist_template()
    tickers = candidates["ticker"].astype(str).str.strip().str.upper().dropna().unique().tolist()
    blacklisted = set(blacklist["ticker"].astype(str))
    fetch_set = {str(t).strip().upper() for t in (yahoo_fetch_tickers or []) if str(t).strip()}
    missing_local = []
    for ticker in tickers:
        if ticker in blacklisted or (fetch_set and ticker not in fetch_set):
            continue
        local = _read_cached_ohlcv(ticker)
        hist = local[local["date"].le(as_of_date)].copy() if not local.empty else pd.DataFrame()
        if hist.empty:
            missing_local.append(ticker)
    yahoo_results = _download_yahoo_ohlcv_batch(missing_local, as_of_date) if missing_local else {}
    quality = pd.DataFrame(
        [
            _quality_for_ticker(ticker, as_of_date, blacklist, allowlist, ticker in yahoo_results, yahoo_results.get(ticker, False))
            for ticker in tickers
        ]
    )
    output = candidates.copy()
    output["ticker"] = output["ticker"].astype(str).str.strip().str.upper()
    merge_cols = [
        "ticker",
        "quality_pass",
        "passed_tradability_filter",
        "tradability_exclusion_reason",
        "exclusion_reason",
        "final_exclusion_reason",
        "local_ohlcv_available",
        "yahoo_fetch_attempted",
        "yahoo_fetch_success",
        "median_60d_dollar_volume",
        "market_cap",
        "trading_history_days",
        "realized_vol_60d",
    ]
    output = output.merge(quality[[c for c in merge_cols if c in quality.columns]], on="ticker", how="left")
    output["quality_pass"] = output["quality_pass"].fillna(False).astype(bool)
    output["exclusion_reason"] = output["exclusion_reason"].fillna("quality data unavailable")
    exclusions = (
        output[~output["quality_pass"]][["date", "ticker", "exclusion_reason"]].copy()
        if "date" in output.columns
        else output[~output["quality_pass"]][["ticker", "exclusion_reason"]].copy()
    )
    tradability_exclusions = (
        quality[~quality.get("passed_tradability_filter", pd.Series(False, index=quality.index)).fillna(False)].copy()
        if not quality.empty
        else pd.DataFrame()
    )
    quality.to_csv(CURRENT_FILE, index=False)
    quality.to_csv(REPORT_FILE, index=False)
    quality.to_csv(TRADABILITY_REPORT_FILE, index=False)
    exclusions.to_csv(EXCLUSIONS_FILE, index=False)
    tradability_exclusions.to_csv(TRADABILITY_EXCLUSIONS_FILE, index=False)
    return output, quality, exclusions


def main() -> None:
    features = pd.read_csv("current_growth_features.csv") if Path("current_growth_features.csv").exists() else pd.DataFrame()
    if features.empty:
        raise SystemExit("current_growth_features.csv not found or empty")
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
    as_of = features["date"].max()
    _, quality, exclusions = apply_growth_universe_quality_filter(features, as_of)
    print("===== GROWTH UNIVERSE QUALITY FILTER =====")
    print(f"date: {as_of.date().isoformat()}")
    print(f"tickers checked: {len(quality)}")
    print(f"passed: {int(quality['quality_pass'].sum()) if not quality.empty else 0}")
    print(f"excluded: {len(exclusions)}")
    if not exclusions.empty:
        print(exclusions[["ticker", "exclusion_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
