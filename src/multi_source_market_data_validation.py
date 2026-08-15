from __future__ import annotations

import argparse
import io
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

YAHOO_CACHE = Path("yahoo_ohlcv_price_cache")
STOOQ_CACHE = Path("stooq_ohlcv_cache")
SECONDARY_CACHE = Path("secondary_market_data_cache")
PRICE_DIFF_THRESHOLD = 0.005
VOLUME_DIFF_THRESHOLD = 0.20
DATE_MISMATCH_TRADING_SESSIONS = 1
YAHOO_FRESHNESS_TRADING_SESSIONS = 1

OUTPUT_PROVIDER_STATUS = "secondary_provider_status.csv"
OUTPUT_PRICE_AUDIT = "multi_source_price_audit.csv"
OUTPUT_CORP_AUDIT = "corporate_action_audit.csv"
OUTPUT_GOVERNANCE = "market_data_governance.csv"
OUTPUT_CANONICAL = "canonical_price_history.csv"


@dataclass
class ProviderConfig:
    provider: str
    api_key: str
    timeout: int
    retries: int
    configured: bool
    reason: str


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def normalize_date(df: pd.DataFrame, source_col: str = "date") -> pd.DataFrame:
    if df.empty or source_col not in df.columns:
        return df
    out = df.copy()
    out[source_col] = pd.to_datetime(out[source_col], errors="coerce").dt.normalize()
    return out.dropna(subset=[source_col]).sort_values(source_col)


def selected_tickers() -> list[str]:
    tickers: list[str] = []
    for path in ["current_growth_candidate_allocation.csv", "growth_official_paper_state.csv", "growth_candidate_paper_state.csv"]:
        df = read_csv(path)
        if df.empty or "ticker" not in df.columns:
            continue
        if "date" in df.columns:
            df = normalize_date(df)
            if not df.empty:
                df = df[df["date"].eq(df["date"].max())]
        tickers.extend([t for t in df["ticker"].dropna().astype(str).str.upper().tolist() if t and t != "CASH"])
    seen: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.append(t)
    return seen


def expected_market_date(end: str) -> pd.Timestamp:
    candidates: list[pd.Timestamp] = []
    for path in [
        "growth_official_paper_performance.csv",
        "growth_candidate_paper_performance.csv",
        "current_growth_candidate_allocation.csv",
        "forecast_history.csv",
    ]:
        df = read_csv(path)
        if df.empty or "date" not in df.columns:
            continue
        tmp = normalize_date(df)
        if not tmp.empty:
            candidates.append(tmp["date"].max())
    if candidates:
        return max(candidates).normalize()
    return pd.Timestamp(end).normalize()


def choose_provider() -> ProviderConfig:
    requested = os.getenv("SECONDARY_DATA_PROVIDER", "").strip().lower()
    timeout = env_int("SECONDARY_DATA_TIMEOUT", 20)
    retries = env_int("SECONDARY_DATA_RETRIES", 2)
    keys = {
        "polygon": os.getenv("POLYGON_API_KEY") or os.getenv("SECONDARY_DATA_API_KEY") or "",
        "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("SECONDARY_DATA_API_KEY") or "",
        "nasdaq_data_link": os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("SECONDARY_DATA_API_KEY") or "",
        "broker": os.getenv("BROKER_MARKET_DATA_API_KEY") or os.getenv("SECONDARY_DATA_API_KEY") or "",
        "stooq": "optional_no_key",
    }
    priority = ["polygon", "alpha_vantage", "nasdaq_data_link", "broker"]
    if requested:
        if requested in {"alpha", "alphavantage", "alpha-vantage"}:
            requested = "alpha_vantage"
        if requested in {"nasdaq", "nasdaq_data_link", "nasdaq-data-link"}:
            requested = "nasdaq_data_link"
        if requested in {"broker", "broker_api", "broker-market-data"}:
            requested = "broker"
        if requested == "stooq":
            return ProviderConfig("stooq", "", timeout, retries, True, "explicit_stooq_fallback_requested")
        api_key = keys.get(requested, "")
        if api_key:
            return ProviderConfig(requested, api_key, timeout, retries, True, "explicit_provider_configured")
        return ProviderConfig(requested, "", timeout, retries, False, f"missing_api_key_for_{requested}")
    for provider in priority:
        if keys[provider]:
            return ProviderConfig(provider, keys[provider], timeout, retries, True, "auto_selected_from_available_credentials")
    if os.getenv("SECONDARY_DATA_USE_STOOQ", "0").strip() in {"1", "true", "yes"}:
        return ProviderConfig("stooq", "", timeout, retries, True, "stooq_optional_fallback_enabled")
    return ProviderConfig("none", "", timeout, retries, False, "no_secondary_provider_credentials")


def url_read(url: str, timeout: int, retries: int) -> tuple[str, str]:
    last_err = ""
    for attempt in range(max(1, retries + 1)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace"), "ok"
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 5))
    return "", last_err or "request_failed"


def load_yahoo(ticker: str) -> pd.DataFrame:
    df = read_csv(YAHOO_CACHE / f"{ticker.upper()}.csv")
    if df.empty:
        return pd.DataFrame()
    date_col = "Date" if "Date" in df.columns else "date" if "date" in df.columns else ""
    if not date_col:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    close = df["Close"] if "Close" in df.columns else df.get("close", np.nan)
    adj = df["Adj Close"] if "Adj Close" in df.columns else df.get("adj_close", close)
    vol = df["Volume"] if "Volume" in df.columns else df.get("volume", np.nan)
    out = pd.DataFrame({"date": df["date"], "yahoo_close": close, "yahoo_adj_close": adj, "yahoo_volume": vol})
    return normalize_date(out).dropna(subset=["yahoo_close"], how="all")


def parse_polygon(raw: str) -> pd.DataFrame:
    import json

    js = json.loads(raw)
    rows = js.get("results", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df.get("t"), unit="ms", errors="coerce").dt.normalize(),
            "secondary_close": pd.to_numeric(df.get("c"), errors="coerce"),
            "secondary_adj_close": pd.to_numeric(df.get("c"), errors="coerce"),
            "secondary_volume": pd.to_numeric(df.get("v"), errors="coerce"),
        }
    )
    return normalize_date(out)


def parse_alpha_vantage(raw: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import json

    js = json.loads(raw)
    key = "Time Series (Daily)"
    if key not in js:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    corp = []
    for date, vals in js[key].items():
        close = vals.get("4. close")
        adj = vals.get("5. adjusted close", close)
        volume = vals.get("6. volume")
        dividend = vals.get("7. dividend amount", "0")
        split = vals.get("8. split coefficient", "1")
        rows.append({"date": date, "secondary_close": close, "secondary_adj_close": adj, "secondary_volume": volume})
        try:
            if abs(float(dividend)) > 0 or abs(float(split) - 1.0) > 1e-9:
                corp.append(
                    {
                        "date": date,
                        "event_type": "alpha_vantage_adjustment",
                        "dividend": dividend,
                        "split_coefficient": split,
                        "resolved": True,
                    }
                )
        except Exception:
            pass
    df = pd.DataFrame(rows)
    for c in ["secondary_close", "secondary_adj_close", "secondary_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    corp_df = pd.DataFrame(corp)
    if not corp_df.empty:
        corp_df["date"] = pd.to_datetime(corp_df["date"], errors="coerce").dt.normalize()
    return normalize_date(df), corp_df


def stooq_symbol(ticker: str) -> str:
    aliases = {"BRK.B": "brk-b.us", "BRK-B": "brk-b.us"}
    return aliases.get(ticker.upper(), f"{ticker.lower().replace('.', '-')}.us")


def parse_stooq(raw: str) -> pd.DataFrame:
    if raw.lstrip().lower().startswith(("<!doctype", "<html")):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce").dt.normalize(),
            "secondary_close": pd.to_numeric(df.get("Close"), errors="coerce"),
            "secondary_adj_close": pd.to_numeric(df.get("Close"), errors="coerce"),
            "secondary_volume": pd.to_numeric(df.get("Volume"), errors="coerce"),
        }
    )
    return normalize_date(out)


def fetch_secondary(
    ticker: str, start: str, end: str, cfg: ProviderConfig, force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    status: dict[str, Any] = {
        "ticker": ticker,
        "provider": cfg.provider,
        "provider_configured": cfg.configured,
        "provider_reason": cfg.reason,
        "fetch_attempted": False,
        "fetch_success": False,
        "fetch_error": "",
    }
    if not cfg.configured or cfg.provider == "none":
        status["fetch_error"] = cfg.reason
        return pd.DataFrame(), pd.DataFrame(), status
    SECONDARY_CACHE.mkdir(exist_ok=True)
    cache = SECONDARY_CACHE / f"{cfg.provider}_{ticker.upper()}.csv"
    corp_cache = SECONDARY_CACHE / f"{cfg.provider}_{ticker.upper()}_corporate.csv"
    if cache.exists() and not force:
        df = read_csv(cache)
        corp = read_csv(corp_cache)
        status.update(
            {"fetch_attempted": False, "fetch_success": not df.empty, "fetch_error": "cache_used" if not df.empty else "empty_cache"}
        )
        return normalize_date(df), normalize_date(corp), status
    status["fetch_attempted"] = True
    raw = ""
    err = ""
    if cfg.provider == "polygon":
        url = f"https://api.polygon.io/v2/aggs/ticker/{urllib.parse.quote(ticker)}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={urllib.parse.quote(cfg.api_key)}"
        raw, err = url_read(url, cfg.timeout, cfg.retries)
        df = parse_polygon(raw) if err == "ok" else pd.DataFrame()
        corp = pd.DataFrame()
    elif cfg.provider == "alpha_vantage":
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={urllib.parse.quote(ticker)}&outputsize=full&apikey={urllib.parse.quote(cfg.api_key)}"
        raw, err = url_read(url, cfg.timeout, cfg.retries)
        df, corp = parse_alpha_vantage(raw) if err == "ok" else (pd.DataFrame(), pd.DataFrame())
    elif cfg.provider == "stooq":
        d1 = pd.Timestamp(start).strftime("%Y%m%d")
        d2 = pd.Timestamp(end).strftime("%Y%m%d")
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&d1={d1}&d2={d2}&i=d"
        raw, err = url_read(url, cfg.timeout, cfg.retries)
        df = parse_stooq(raw) if err == "ok" else pd.DataFrame()
        corp = pd.DataFrame()
    elif cfg.provider in {"nasdaq_data_link", "broker"}:
        df = pd.DataFrame()
        corp = pd.DataFrame()
        err = f"{cfg.provider}_adapter_not_configured_for_generic_equity_ohlcv"
    else:
        df = pd.DataFrame()
        corp = pd.DataFrame()
        err = f"unsupported_provider_{cfg.provider}"
    if not df.empty:
        df.to_csv(cache, index=False)
    if not corp.empty:
        corp.to_csv(corp_cache, index=False)
    status.update({"fetch_success": not df.empty, "fetch_error": "" if not df.empty else err or "empty_response"})
    return df, corp, status


def pct_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs() / pd.to_numeric(b, errors="coerce").abs().replace(
        0, np.nan
    )


def trading_day_gap(a: Any, b: Any) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    lo, hi = sorted([pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()])
    return float(len(pd.bdate_range(lo, hi)) - 1)


def validate_ticker(
    ticker: str, start: str, end: str, cfg: ProviderConfig, force_fetch: bool = False, expected_date: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    yahoo = load_yahoo(ticker)
    secondary, corp, provider_status = fetch_secondary(ticker, start, end, cfg, force_fetch)
    if yahoo.empty:
        audit = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "date": "",
                    "primary_source": "Yahoo/yfinance_cache",
                    "secondary_provider": cfg.provider,
                    "yahoo_available": False,
                    "secondary_available": not secondary.empty,
                    "classification": "primary_data_blocked",
                    "paper_may_continue": False,
                    "block_new_rebalance": True,
                    "real_capital_blocked": True,
                    "reason": "primary_yahoo_missing",
                }
            ]
        )
        return audit, corp, pd.DataFrame(), provider_status
    yahoo = yahoo[yahoo["date"].ge(pd.Timestamp(start).normalize()) & yahoo["date"].le(pd.Timestamp(end).normalize())].copy()
    latest_yahoo = yahoo["date"].max() if not yahoo.empty else pd.NaT
    latest_secondary = secondary["date"].max() if not secondary.empty else pd.NaT
    expected = expected_date.normalize() if expected_date is not None and pd.notna(expected_date) else pd.Timestamp(end).normalize()
    yahoo_gap = trading_day_gap(latest_yahoo, expected)
    yahoo_stale = bool(pd.notna(yahoo_gap) and yahoo_gap > YAHOO_FRESHNESS_TRADING_SESSIONS)
    missing_secondary = secondary.empty or not provider_status.get("fetch_success", False)
    if missing_secondary:
        latest = yahoo.tail(1).copy()
        latest["ticker"] = ticker
        latest["primary_source"] = "Yahoo/yfinance_cache"
        latest["secondary_provider"] = cfg.provider
        latest["secondary_available"] = False
        latest["provider_reason"] = provider_status.get("fetch_error") or cfg.reason
        latest["expected_market_date"] = expected.date().isoformat() if pd.notna(expected) else ""
        latest["latest_yahoo_date"] = latest_yahoo.date().isoformat() if pd.notna(latest_yahoo) else ""
        latest["latest_secondary_date"] = ""
        latest["yahoo_freshness_gap_sessions"] = yahoo_gap
        latest["yahoo_stale"] = yahoo_stale
        latest["close_diff_pct"] = np.nan
        latest["adj_close_diff_pct"] = np.nan
        latest["volume_diff_pct"] = np.nan
        latest["price_difference_gt_0_5pct"] = False
        latest["volume_difference_gt_20pct"] = False
        latest["date_mismatch_gt_1_session"] = False
        latest["corporate_action_unresolved"] = False
        latest["classification"] = "primary_data_blocked" if yahoo_stale else "single_source_warning"
        latest["paper_may_continue"] = not yahoo_stale
        latest["block_new_rebalance"] = bool(yahoo_stale)
        latest["real_capital_blocked"] = True
        latest["reason"] = np.where(
            yahoo_stale,
            "Yahoo primary cache stale versus expected market date",
            "Yahoo fresh but reliable secondary source unavailable; no false discrepancy marked",
        )
        canonical = latest[["date", "ticker", "yahoo_adj_close", "yahoo_close", "yahoo_volume", "classification"]].copy()
        canonical["canonical_close"] = canonical["yahoo_adj_close"]
        canonical["canonical_source"] = "yahoo_adj_close_single_source_warning"
        return latest, corp, canonical, provider_status
    merged = yahoo.merge(secondary, on="date", how="outer", indicator=True).sort_values("date")
    merged["ticker"] = ticker
    merged["primary_source"] = "Yahoo/yfinance_cache"
    merged["secondary_provider"] = cfg.provider
    merged["yahoo_available"] = merged["yahoo_close"].notna()
    merged["secondary_available"] = merged["secondary_close"].notna()
    merged["close_diff_pct"] = pct_diff(merged["yahoo_close"], merged["secondary_close"])
    merged["adj_close_diff_pct"] = pct_diff(merged["yahoo_adj_close"], merged["secondary_adj_close"])
    merged["volume_diff_pct"] = pct_diff(merged["yahoo_volume"], merged["secondary_volume"])
    merged["price_difference_gt_0_5pct"] = (merged["close_diff_pct"] > PRICE_DIFF_THRESHOLD) | (
        merged["adj_close_diff_pct"] > PRICE_DIFF_THRESHOLD
    )
    merged["volume_difference_gt_20pct"] = merged["volume_diff_pct"] > VOLUME_DIFF_THRESHOLD
    date_gap = trading_day_gap(latest_yahoo, latest_secondary)
    yahoo_expected_gap = trading_day_gap(latest_yahoo, expected)
    yahoo_stale = bool(pd.notna(yahoo_expected_gap) and yahoo_expected_gap > YAHOO_FRESHNESS_TRADING_SESSIONS)
    date_mismatch = bool(pd.notna(date_gap) and date_gap > DATE_MISMATCH_TRADING_SESSIONS)
    unresolved_corp = False
    if not corp.empty:
        corp["ticker"] = ticker
        if "resolved" not in corp.columns:
            corp["resolved"] = False
        unresolved_corp = bool((~corp["resolved"].fillna(False).astype(bool)).any())
    material_conflict = bool(
        merged["price_difference_gt_0_5pct"].fillna(False).any()
        or merged["volume_difference_gt_20pct"].fillna(False).any()
        or date_mismatch
        or unresolved_corp
    )
    if yahoo_stale:
        classification = "primary_data_blocked"
    else:
        classification = "data_conflict_blocked" if material_conflict else "multi_source_confirmed"
    merged["expected_market_date"] = expected.date().isoformat() if pd.notna(expected) else ""
    merged["latest_yahoo_date"] = latest_yahoo.date().isoformat() if pd.notna(latest_yahoo) else ""
    merged["latest_secondary_date"] = latest_secondary.date().isoformat() if pd.notna(latest_secondary) else ""
    merged["yahoo_freshness_gap_sessions"] = yahoo_expected_gap
    merged["yahoo_stale"] = yahoo_stale
    merged["date_mismatch_trading_sessions"] = date_gap
    merged["date_mismatch_gt_1_session"] = date_mismatch
    merged["corporate_action_unresolved"] = unresolved_corp
    merged["classification"] = classification
    merged["paper_may_continue"] = classification not in {"data_conflict_blocked", "primary_data_blocked"}
    merged["block_new_rebalance"] = classification in {"data_conflict_blocked", "primary_data_blocked"}
    merged["real_capital_blocked"] = classification != "multi_source_confirmed"
    merged["reason"] = np.where(
        merged["price_difference_gt_0_5pct"].fillna(False),
        "material_adjusted_close_or_close_difference",
        np.where(
            merged["volume_difference_gt_20pct"].fillna(False),
            "material_volume_difference",
            np.where(merged["yahoo_stale"], "Yahoo primary cache stale versus expected market date", "ok"),
        ),
    )
    canonical = merged[merged["yahoo_available"]].copy()
    canonical["canonical_close"] = canonical["yahoo_adj_close"]
    canonical["canonical_source"] = np.where(
        classification == "data_conflict_blocked", "blocked_conflict_no_silent_merge", "yahoo_adj_close_multi_source_confirmed"
    )
    if classification == "data_conflict_blocked":
        canonical["canonical_close"] = np.nan
    return merged, corp, canonical, provider_status


def governance(
    audit: pd.DataFrame, corp: pd.DataFrame, provider_status: pd.DataFrame, tickers: list[str], cfg: ProviderConfig
) -> pd.DataFrame:
    if audit.empty:
        classification = "primary_data_blocked"
        reason = "no audit rows"
        latest_yahoo = ""
        latest_secondary = ""
    else:
        latest = audit.copy()
        if "date" in latest.columns:
            latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
            latest = latest.sort_values("date").groupby("ticker", as_index=False).tail(1)
        classes = set(latest.get("classification", pd.Series(dtype=str)).dropna().astype(str))
        if "data_conflict_blocked" in classes or "primary_data_blocked" in classes:
            classification = "data_conflict_blocked" if "data_conflict_blocked" in classes else "primary_data_blocked"
        elif "single_source_warning" in classes:
            classification = "single_source_warning"
        elif classes == {"multi_source_confirmed"}:
            classification = "multi_source_confirmed"
        else:
            classification = "single_source_warning"
        reason = "; ".join(sorted(set(latest.get("reason", pd.Series(dtype=str)).dropna().astype(str))))
        latest_yahoo = ",".join(sorted(set(latest.get("latest_yahoo_date", pd.Series(dtype=str)).dropna().astype(str))))
        latest_secondary = ",".join(sorted(set(latest.get("latest_secondary_date", pd.Series(dtype=str)).dropna().astype(str))))
    return pd.DataFrame(
        [
            {
                "validation_scope": "growth_paper_market_data_validation_only",
                "classification": classification,
                "secondary_provider": cfg.provider,
                "provider_configured": cfg.configured,
                "tickers_checked": len(tickers),
                "tickers": ",".join(tickers),
                "latest_yahoo_dates": latest_yahoo,
                "latest_secondary_dates": latest_secondary,
                "paper_may_continue": classification in {"single_source_warning", "multi_source_confirmed"},
                "block_new_rebalance": classification == "data_conflict_blocked" or classification == "primary_data_blocked",
                "retain_existing_holdings_until_review": classification == "data_conflict_blocked",
                "real_capital_blocked": classification != "multi_source_confirmed",
                "promotion_status": "blocked_without_reliable_second_source"
                if classification != "multi_source_confirmed"
                else "market_data_gate_passed_for_review_only",
                "corporate_action_events": 0 if corp.empty else len(corp),
                "reason": reason or cfg.reason,
                "production_changed": False,
                "paper_logic_changed": False,
                "allocation_logic_changed": False,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliable secondary market-data validation for Growth paper. Governance only.")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers. Default uses current selected holdings/allocation.")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--force-fetch", action="store_true")
    args = parser.parse_args()
    cfg = choose_provider()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else selected_tickers()
    if not tickers:
        tickers = ["SPY", "QQQ"]
    audits: list[pd.DataFrame] = []
    corps: list[pd.DataFrame] = []
    canonicals: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    expected_date = expected_market_date(args.end)
    for ticker in tickers:
        audit, corp, canonical, status = validate_ticker(ticker, args.start, args.end, cfg, args.force_fetch, expected_date)
        audits.append(audit)
        if not corp.empty:
            corps.append(corp)
        if not canonical.empty:
            canonicals.append(canonical)
        statuses.append(status)
    audit_df = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
    corp_df = (
        pd.concat(corps, ignore_index=True, sort=False)
        if corps
        else pd.DataFrame(columns=["date", "ticker", "event_type", "resolved", "reason"])
    )
    canonical_df = pd.concat(canonicals, ignore_index=True, sort=False) if canonicals else pd.DataFrame()
    status_df = pd.DataFrame(statuses)
    gov = governance(audit_df, corp_df, status_df, tickers, cfg)
    status_df.to_csv(OUTPUT_PROVIDER_STATUS, index=False)
    audit_df.to_csv(OUTPUT_PRICE_AUDIT, index=False)
    corp_df.to_csv(OUTPUT_CORP_AUDIT, index=False)
    canonical_df.to_csv(OUTPUT_CANONICAL, index=False)
    gov.to_csv(OUTPUT_GOVERNANCE, index=False)
    print("===== RELIABLE SECONDARY MARKET DATA INTEGRATION =====")
    print(f"secondary_provider: {cfg.provider}")
    print(f"provider_configured: {cfg.configured}")
    print(f"expected_market_date: {expected_date.date().isoformat() if pd.notna(expected_date) else ''}")
    print(f"tickers_checked: {len(tickers)}")
    print(f"classification: {gov.iloc[0]['classification']}")
    print(f"paper_may_continue: {gov.iloc[0]['paper_may_continue']}")
    print(f"block_new_rebalance: {gov.iloc[0]['block_new_rebalance']}")
    print(f"real_capital_blocked: {gov.iloc[0]['real_capital_blocked']}")
    print("outputs: secondary_provider_status.csv, multi_source_price_audit.csv, corporate_action_audit.csv, market_data_governance.csv")


if __name__ == "__main__":
    main()
