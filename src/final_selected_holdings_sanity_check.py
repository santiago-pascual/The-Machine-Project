
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRICE_CACHE_DIR = Path("yahoo_ohlcv_price_cache")
AUDIT_FILE = Path("final_selected_holdings_audit.csv")
REPLACEMENTS_FILE = Path("final_selected_holdings_replacements.csv")
METADATA_CACHE_FILE = Path("ticker_metadata_cache.csv")
SANITY_BLACKLIST_FILE = Path("growth_holdings_sanity_blacklist.csv")

CRYPTO_KEYWORDS = ("BITCOIN", "CRYPTO", "BLOCKCHAIN", "MINER", "MINING", "DIGITAL ASSET")
SPAC_KEYWORDS = ("SPAC", "ACQUISITION CORP", "BLANK CHECK", "SHELL")
BIOTECH_KEYWORDS = ("BIOTECH", "BIOPHARMA", "THERAPEUTICS", "PHARMA", "CLINICAL", "ONCOLOGY")
DISTRESSED_KEYWORDS = ("DISTRESSED", "BANKRUPT", "RESTRUCTUR", "PENNY")
ADR_KEYWORDS = ("ADR", "ADS", "AMERICAN DEPOSITARY")


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _num(value, default=np.nan) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if isinstance(out, pd.Series):
            out = out.iloc[0] if not out.empty else np.nan
        return float(out) if pd.notna(out) else default
    except Exception:
        return default


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    date_col = "Date" if "Date" in out.columns else ("date" if "date" in out.columns else out.columns[0])
    out["Date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=["Date"]).sort_values("Date")
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _read_ohlcv(ticker: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    path = PRICE_CACHE_DIR / f"{str(ticker).upper()}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = _standardize_ohlcv(pd.read_csv(path))
    except Exception:
        return pd.DataFrame()
    return df[df["Date"].le(as_of_date)].copy() if not df.empty else df


def _metadata(ticker: str) -> dict[str, object]:
    ticker = str(ticker).upper().strip()
    meta = {
        "company_name": "",
        "sector": "",
        "industry": "",
        "exchange": "",
        "market_cap": np.nan,
    }
    df = _read_csv(METADATA_CACHE_FILE)
    if not df.empty and "ticker" in df.columns:
        row = df[df["ticker"].astype(str).str.upper().eq(ticker)]
        if not row.empty:
            latest = row.iloc[-1]
            for col in meta:
                if col in row.columns:
                    meta[col] = latest.get(col, meta[col])
    return meta


def _sanity_blacklist() -> dict[str, str]:
    if not SANITY_BLACKLIST_FILE.exists():
        pd.DataFrame(columns=["ticker", "reason", "date_added"]).to_csv(SANITY_BLACKLIST_FILE, index=False)
        return {}
    df = _read_csv(SANITY_BLACKLIST_FILE)
    if df.empty or "ticker" not in df.columns:
        return {}
    if "reason" not in df.columns:
        df["reason"] = "manual sanity blacklist"
    return dict(zip(df["ticker"].astype(str).str.upper(), df["reason"].astype(str)))


def _risk_flags(ticker: str, meta: dict[str, object], row: pd.Series, ohlcv: pd.DataFrame) -> list[str]:
    text = " ".join(str(meta.get(k, "")) for k in ["company_name", "sector", "industry", "exchange"]).upper()
    flags: list[str] = []
    if any(k in text for k in CRYPTO_KEYWORDS):
        flags.append("crypto-linked")
    if any(k in text for k in SPAC_KEYWORDS) or any(marker in str(ticker).upper() for marker in [".WS", "-WS", "/WS", ".WT", "-WT", "/WT", ".U", "-U", "/U", ".R", "-R", "/R"]):
        flags.append("SPAC/warrant/shell-linked")
    if any(k in text for k in BIOTECH_KEYWORDS):
        flags.append("biotech/binary-risk")
    if any(k in text for k in ADR_KEYWORDS):
        flags.append("ADR")
    if any(k in text for k in DISTRESSED_KEYWORDS):
        flags.append("distressed")

    price = _num(row.get("current_price", row.get("latest_price", np.nan)))
    med_dv = _num(row.get("median_60d_dollar_volume", np.nan))
    vol = _num(row.get("realized_vol_60d", np.nan))
    history = _num(row.get("trading_history_days", np.nan))
    market_cap = _num(meta.get("market_cap", row.get("market_cap", np.nan)))
    if np.isfinite(price) and price < 10:
        flags.append("low-price")
    if np.isfinite(med_dv) and med_dv < 100_000_000:
        flags.append("below-100M-median-dollar-volume")
    if np.isfinite(market_cap) and market_cap < 2_000_000_000:
        flags.append("micro/small-cap")
    if np.isfinite(vol) and vol > 0.80:
        flags.append("high-volatility")
    if np.isfinite(history) and history < 756:
        flags.append("short-history")

    if not ohlcv.empty:
        price_col = "Adj Close" if "Adj Close" in ohlcv.columns else "Close"
        returns = pd.to_numeric(ohlcv[price_col], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan)
        if returns.tail(252).abs().max(skipna=True) > 0.90:
            flags.append("extreme-discontinuity")
    return list(dict.fromkeys(flags))


def _holding_metrics(ticker: str, as_of_date: pd.Timestamp, source_row: pd.Series) -> dict[str, object]:
    ohlcv = _read_ohlcv(ticker, as_of_date)
    price_col = "Adj Close" if not ohlcv.empty and "Adj Close" in ohlcv.columns else "Close"
    prices = pd.to_numeric(ohlcv[price_col], errors="coerce").dropna() if not ohlcv.empty and price_col in ohlcv.columns else pd.Series(dtype=float)
    volumes = pd.to_numeric(ohlcv.get("Volume", pd.Series(dtype=float)), errors="coerce") if not ohlcv.empty else pd.Series(dtype=float)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan) if not prices.empty else pd.Series(dtype=float)
    current_price = _num(source_row.get("current_price", np.nan))
    if not np.isfinite(current_price) and not prices.empty:
        current_price = float(prices.iloc[-1])
    med_60_dv = _num(source_row.get("median_60d_dollar_volume", np.nan))
    if not np.isfinite(med_60_dv) and not prices.empty and not volumes.empty:
        aligned = pd.concat([prices, volumes], axis=1).dropna()
        if not aligned.empty:
            med_60_dv = float((aligned.iloc[:, 0] * aligned.iloc[:, 1]).tail(60).median())
    avg_20_vol = np.nan
    if not volumes.empty:
        avg_20_vol = float(volumes.tail(20).mean()) if not volumes.tail(20).dropna().empty else np.nan
    realized_vol_60d = _num(source_row.get("realized_vol_60d", np.nan))
    if not np.isfinite(realized_vol_60d) and len(returns.dropna()) >= 60:
        realized_vol_60d = float(returns.tail(60).std(ddof=0) * np.sqrt(252))
    history_days = _num(source_row.get("trading_history_days", np.nan))
    if not np.isfinite(history_days):
        history_days = float(len(ohlcv))
    def ret_n(n: int) -> float:
        if len(prices) <= n:
            return np.nan
        return float(prices.iloc[-1] / prices.iloc[-n-1] - 1.0)
    return {
        "current_price": current_price,
        "median_60d_dollar_volume": med_60_dv,
        "avg_volume_20d": avg_20_vol,
        "realized_vol_60d": realized_vol_60d,
        "trading_history_days": int(history_days) if np.isfinite(history_days) else np.nan,
        "return_1d": ret_n(1),
        "return_5d": ret_n(5),
        "return_20d": ret_n(20),
    }


def _classify(flags: list[str], source_row: pd.Series, blacklist: dict[str, str], ticker: str) -> tuple[str, str]:
    reasons: list[str] = []
    if ticker in blacklist:
        reasons.append(f"manual sanity blacklist: {blacklist[ticker]}")
    if str(source_row.get("quality_pass", "true")).lower() not in ["true", "1", "yes"]:
        reasons.append("failed upstream quality/tradability filter")
    if any(f in flags for f in ["SPAC/warrant/shell-linked", "distressed", "extreme-discontinuity"]):
        reasons.append("structural/binary tradability risk")
    if ticker in blacklist or reasons:
        return "reject_from_growth_universe", "; ".join(reasons)
    if flags:
        return "speculative_but_tradable", "; ".join(flags)
    return "institutional_quality", "no major holding-level risk flags detected"


def audit_and_filter_selected_holdings(features_df: pd.DataFrame, selected_tickers: list[str], as_of_date: pd.Timestamp, max_positions: int) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    if features_df.empty or "ticker" not in features_df.columns:
        return selected_tickers, pd.DataFrame(), pd.DataFrame()
    df = features_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    selected = list(dict.fromkeys(str(t).upper().strip() for t in selected_tickers if str(t).strip()))
    blacklist = _sanity_blacklist()
    audit_rows: list[dict[str, object]] = []

    for ticker in selected:
        source = df[df["ticker"].eq(ticker)].iloc[-1] if not df[df["ticker"].eq(ticker)].empty else pd.Series(dtype=object)
        meta = _metadata(ticker)
        metrics = _holding_metrics(ticker, as_of_date, source)
        row_for_flags = pd.concat([source, pd.Series(metrics)])
        ohlcv = _read_ohlcv(ticker, as_of_date)
        flags = _risk_flags(ticker, meta, row_for_flags, ohlcv)
        classification, notes = _classify(flags, source, blacklist, ticker)
        audit_rows.append({
            "date": as_of_date.date().isoformat(),
            "ticker": ticker,
            **meta,
            **metrics,
            "is_crypto_linked": "crypto-linked" in flags,
            "is_spac_linked": "SPAC/warrant/shell-linked" in flags,
            "is_biotech_binary_risk": "biotech/binary-risk" in flags,
            "is_adr": "ADR" in flags,
            "is_microcap_or_low_float": "micro/small-cap" in flags or "low-price" in flags,
            "is_distressed": "distressed" in flags,
            "holding_quality_classification": classification,
            "holding_risk_notes": notes,
        })

    audit = pd.DataFrame(audit_rows)
    rejected = set(audit.loc[audit["holding_quality_classification"].eq("reject_from_growth_universe"), "ticker"].astype(str)) if not audit.empty else set()
    final = [t for t in selected if t not in rejected]
    replacements: list[dict[str, object]] = []

    if rejected:
        candidates = df.copy()
        if "quality_pass" in candidates.columns:
            candidates = candidates[candidates["quality_pass"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if "raw_target_return" in candidates.columns:
            candidates = candidates[pd.to_numeric(candidates["raw_target_return"], errors="coerce") > 0]
            candidates = candidates.sort_values(["raw_target_return", "signal_strength" if "signal_strength" in candidates.columns else "raw_target_return"], ascending=False)
        for _, cand in candidates.iterrows():
            ticker = str(cand.get("ticker", "")).upper().strip()
            if not ticker or ticker in final or ticker in rejected:
                continue
            meta = _metadata(ticker)
            metrics = _holding_metrics(ticker, as_of_date, cand)
            row_for_flags = pd.concat([cand, pd.Series(metrics)])
            flags = _risk_flags(ticker, meta, row_for_flags, _read_ohlcv(ticker, as_of_date))
            classification, notes = _classify(flags, cand, blacklist, ticker)
            audit_rows.append({
                "date": as_of_date.date().isoformat(),
                "ticker": ticker,
                **meta,
                **metrics,
                "is_crypto_linked": "crypto-linked" in flags,
                "is_spac_linked": "SPAC/warrant/shell-linked" in flags,
                "is_biotech_binary_risk": "biotech/binary-risk" in flags,
                "is_adr": "ADR" in flags,
                "is_microcap_or_low_float": "micro/small-cap" in flags or "low-price" in flags,
                "is_distressed": "distressed" in flags,
                "holding_quality_classification": classification,
                "holding_risk_notes": notes,
            })
            if classification != "reject_from_growth_universe":
                replaced = sorted(rejected)[len(replacements)] if len(replacements) < len(rejected) else ""
                final.append(ticker)
                replacements.append({
                    "date": as_of_date.date().isoformat(),
                    "rejected_ticker": replaced,
                    "replacement_ticker": ticker,
                    "replacement_classification": classification,
                    "replacement_reason": notes,
                })
            if len(final) >= min(max_positions, len(selected)):
                break

    audit = pd.DataFrame(audit_rows).drop_duplicates(["date", "ticker"], keep="first") if audit_rows else audit
    replacements_df = pd.DataFrame(replacements, columns=["date", "rejected_ticker", "replacement_ticker", "replacement_classification", "replacement_reason"])
    audit.to_csv(AUDIT_FILE, index=False)
    replacements_df.to_csv(REPLACEMENTS_FILE, index=False)
    return final[:max_positions], audit, replacements_df


def main() -> None:
    allocation = _read_csv("current_growth_candidate_allocation.csv")
    features = _read_csv("current_growth_features.csv")
    if allocation.empty or features.empty:
        raise SystemExit("current growth files missing")
    allocation["date"] = pd.to_datetime(allocation["date"], errors="coerce").dt.normalize()
    as_of = allocation["date"].max()
    selected = allocation[allocation["date"].eq(as_of)]["ticker"].astype(str).tolist()
    final, audit, replacements = audit_and_filter_selected_holdings(features, selected, as_of, len(selected))
    print("===== FINAL SELECTED HOLDINGS SANITY CHECK =====")
    print(f"date: {as_of.date().isoformat()}")
    print(f"original selected holdings: {', '.join(selected)}")
    print(f"final selected holdings: {', '.join(final)}")
    if not audit.empty:
        cols = [c for c in ["ticker", "holding_quality_classification", "holding_risk_notes", "current_price", "median_60d_dollar_volume", "avg_volume_20d", "realized_vol_60d", "trading_history_days"] if c in audit.columns]
        print(audit[cols].to_string(index=False))
    if not replacements.empty:
        print("replacements:")
        print(replacements.to_string(index=False))
    print(f"Saved: {AUDIT_FILE.resolve()}")
    print(f"Saved: {REPLACEMENTS_FILE.resolve()}")


if __name__ == "__main__":
    main()
