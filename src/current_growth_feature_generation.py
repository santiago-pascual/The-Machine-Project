from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_market_data_manager import get_price_history
from final_selected_holdings_sanity_check import audit_and_filter_selected_holdings
from growth_universe_quality_filter import apply_growth_universe_quality_filter

FORECAST_HISTORY_FILE = "forecast_history.csv"
GROWTH_DAILY_FILE = "growth_volatility_targeting_daily_returns.csv"
GROWTH_STATE_FILE = "growth_candidate_paper_state.csv"
CONFIG_FILE = "growth_candidate_paper_config.json"
PRICE_CACHE_DIR = Path("yahoo_ohlcv_price_cache")

RAW_FEATURES_FILE = "current_raw_target_features.csv"
GROWTH_FEATURES_FILE = "current_growth_features.csv"
GROWTH_ALLOCATION_FILE = "current_growth_candidate_allocation.csv"
FRESH_VOL_FILE = "growth_volatility_targeting_fresh.csv"
VOL_PIPELINE_AUDIT_FILE = "growth_volatility_pipeline_audit.csv"

CANDIDATE_NAME = "growth_candidate_v1"
CANDIDATE_VARIANT = "soft_exit_rule_vol_target_22pct"
MAX_POSITIONS = 4
BASE_POSITIONS = 2
TARGET_VOL = 0.22
MIN_EXPOSURE = 0.40
MAX_EXPOSURE = 1.00
MAX_EXPOSURE_CHANGE = 0.15
VOL_LOOKBACK_DAYS = 60


def _growth_paper_config() -> dict[str, object]:
    default = {
        "active_growth_paper_model": "growth_champion_final",
        "active_variant": "growth_v1_exposure_cap_60_dual_trend_filter",
        "legacy_uncapped_model": "growth_v1_uncapped",
        "volatility_target": TARGET_VOL,
        "exposure_cap": 0.60,
        "paper_only": True,
    }
    path = Path(CONFIG_FILE)
    if not path.exists():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    default.update(loaded if isinstance(loaded, dict) else {})
    return default


def _configured_exposure_cap() -> float:
    cfg = _growth_paper_config()
    cap = float(cfg.get("exposure_cap", 0.60))
    return float(np.clip(cap, 0.0, 1.0))


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"])


def _load_cached_price(ticker: str) -> pd.Series:
    history = get_price_history(ticker)
    df = history.data
    if df.empty or "Date" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if col is None:
        return pd.Series(dtype=float, name=ticker)
    prices = pd.to_numeric(df[col], errors="coerce").dropna()
    prices.index = pd.to_datetime(df.loc[prices.index, "Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return prices.rename(ticker)


def _benchmark_trend_state(ticker: str, date: pd.Timestamp) -> dict[str, object]:
    prices = _load_cached_price(ticker)
    hist = prices[prices.index <= date].copy() if not prices.empty else pd.Series(dtype=float)
    if hist.empty:
        return {
            f"{ticker.lower()}_close": np.nan,
            f"{ticker.lower()}_ma_200": np.nan,
            f"{ticker.lower()}_below_200d": False,
            f"{ticker.lower()}_trend_price_date": "",
        }
    close = float(hist.iloc[-1])
    ma_200 = hist.rolling(200, min_periods=150).mean().iloc[-1]
    below = bool(close < ma_200) if pd.notna(ma_200) else False
    return {
        f"{ticker.lower()}_close": close,
        f"{ticker.lower()}_ma_200": float(ma_200) if pd.notna(ma_200) else np.nan,
        f"{ticker.lower()}_below_200d": below,
        f"{ticker.lower()}_trend_price_date": hist.index[-1].strftime("%Y-%m-%d"),
    }


def _dual_trend_filter(date: pd.Timestamp) -> dict[str, object]:
    spy = _benchmark_trend_state("SPY", date)
    qqq = _benchmark_trend_state("QQQ", date)
    spy_below = bool(spy.get("spy_below_200d", False))
    qqq_below = bool(qqq.get("qqq_below_200d", False))
    if spy_below and qqq_below:
        cap = 0.25
        reason = "SPY and QQQ below 200D MA"
    elif spy_below or qqq_below:
        cap = 0.40
        reason = "one of SPY/QQQ below 200D MA"
    else:
        cap = 0.60
        reason = "SPY and QQQ above 200D MA"
    return {**spy, **qqq, "dual_trend_cap": cap, "dual_trend_reason": reason}


def _latest_forecast(as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    forecast = _dates(_read_csv(FORECAST_HISTORY_FILE))
    if forecast.empty:
        return forecast
    if as_of_date is not None:
        target_date = pd.Timestamp(as_of_date).normalize()
        forecast = forecast[forecast["date"].dt.normalize().le(target_date)].copy()
        if forecast.empty:
            return forecast
    latest_date = forecast["date"].max()
    latest = forecast[forecast["date"].eq(latest_date)].copy()
    for col in [
        "current_price",
        "target_price",
        "expected_daily_return",
        "expected_total_return",
        "final_weight_percent",
        "signal_strength",
        "quality_score",
        "target_confidence",
        "raw_target_return_exact",
        "raw_expected_daily_return_exact",
        "raw_target_price_exact",
        "time_to_target",
        "signal_strength_adjustment_value",
        "final_expected_return_after_adjustments",
    ]:
        if col in latest.columns:
            latest[col] = _num(latest[col])
    return latest


def _prior_growth_tickers() -> set[str]:
    state = _dates(_read_csv(GROWTH_STATE_FILE))
    if state.empty:
        return set()
    latest = state[state["date"].eq(state["date"].max())].copy()
    latest = latest[latest["ticker"].astype(str).ne("CASH")]
    if "paper_position_weight" in latest.columns:
        latest = latest[_num(latest["paper_position_weight"]).fillna(0.0) > 0]
    return set(latest["ticker"].astype(str))


def _previous_exposure() -> float:
    state = _dates(_read_csv(GROWTH_STATE_FILE))
    if state.empty or "paper_position_weight" not in state.columns:
        return 1.0
    latest = state[state["date"].eq(state["date"].max())].copy()
    non_cash = latest[latest["ticker"].astype(str).ne("CASH")]
    exposure = float(_num(non_cash["paper_position_weight"]).fillna(0.0).sum())
    return float(np.clip(exposure, MIN_EXPOSURE, MAX_EXPOSURE)) if np.isfinite(exposure) and exposure > 0 else 1.0


def _legacy_rolling_growth_vol() -> float:
    daily = _dates(_read_csv(GROWTH_DAILY_FILE))
    if daily.empty:
        return np.nan
    selector = (
        daily.get("vol_target_variant", daily.get("variant", pd.Series(index=daily.index, dtype=str))).astype(str).eq(CANDIDATE_VARIANT)
    )
    daily = daily[selector].copy().sort_values("date")
    if daily.empty:
        return np.nan
    returns = _num(daily.get("return", daily.get("vol_target_return", pd.Series(index=daily.index, dtype=float)))).dropna()
    if len(returns) < 4:
        return np.nan
    dates = daily["date"].dropna().sort_values()
    step = np.median(dates.diff().dt.days.dropna()) if len(dates) > 1 else 7.0
    ppy = 365.25 / step if np.isfinite(step) and step > 0 else 52.0
    return float(returns.tail(12).std(ddof=0) * np.sqrt(ppy))


def _current_snapshot_prices(date: pd.Timestamp) -> dict[str, float]:
    forecast = _dates(_read_csv(FORECAST_HISTORY_FILE))
    if forecast.empty or "ticker" not in forecast.columns or "current_price" not in forecast.columns:
        return {}
    day = forecast[forecast["date"].eq(pd.Timestamp(date).normalize())].copy()
    if day.empty:
        return {}
    day["ticker"] = day["ticker"].astype(str).str.upper().str.strip()
    day["current_price"] = _num(day["current_price"])
    return dict(zip(day["ticker"], day["current_price"]))


def _fresh_portfolio_volatility(selected_tickers: list[str], date: pd.Timestamp) -> dict[str, object]:
    tickers = [str(t).strip().upper() for t in selected_tickers if str(t).strip()]
    snapshot_prices = _current_snapshot_prices(date)
    price_series = []
    source_dates = []
    missing = []
    used_snapshot = []
    for ticker in tickers:
        prices = _load_cached_price(ticker)
        hist = prices[prices.index <= date].dropna() if not prices.empty else pd.Series(dtype=float)
        snap_price = snapshot_prices.get(ticker, np.nan)
        if np.isfinite(snap_price) and snap_price > 0 and (hist.empty or hist.index[-1].normalize() < date.normalize()):
            hist = pd.concat([hist, pd.Series([float(snap_price)], index=[date.normalize()], name=ticker)]).sort_index()
            hist = hist[~hist.index.duplicated(keep="last")]
            used_snapshot.append(ticker)
        if len(hist) < VOL_LOOKBACK_DAYS + 1 or hist.index[-1].normalize() < date.normalize():
            missing.append(ticker)
            continue
        source_dates.append(hist.index[-1])
        price_series.append(hist.rename(ticker))
    if missing:
        return {
            "ok": False,
            "estimated_portfolio_vol": np.nan,
            "source_date": min(source_dates).strftime("%Y-%m-%d") if source_dates else "",
            "missing_tickers": ",".join(missing),
            "sample_size": 0,
            "used_current_snapshot_prices": ",".join(used_snapshot),
        }
    if not price_series:
        return {
            "ok": False,
            "estimated_portfolio_vol": np.nan,
            "source_date": "",
            "missing_tickers": ",".join(missing),
            "sample_size": 0,
            "used_current_snapshot_prices": ",".join(used_snapshot),
        }
    prices = pd.concat(price_series, axis=1, sort=True).sort_index().ffill().dropna(how="all")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    aligned = returns.dropna(axis=0, how="any").tail(VOL_LOOKBACK_DAYS)
    if len(aligned) < max(20, min(VOL_LOOKBACK_DAYS, 40)):
        return {
            "ok": False,
            "estimated_portfolio_vol": np.nan,
            "source_date": min(source_dates).strftime("%Y-%m-%d") if source_dates else "",
            "missing_tickers": ",".join(missing),
            "sample_size": len(aligned),
            "used_current_snapshot_prices": ",".join(used_snapshot),
        }
    portfolio_returns = aligned.mean(axis=1)
    vol = float(portfolio_returns.std(ddof=0) * np.sqrt(252))
    source_date = min(source_dates).strftime("%Y-%m-%d") if source_dates else ""
    return {
        "ok": np.isfinite(vol) and vol > 0 and source_date >= date.strftime("%Y-%m-%d"),
        "estimated_portfolio_vol": vol,
        "source_date": source_date,
        "missing_tickers": ",".join(missing),
        "sample_size": len(portfolio_returns),
        "used_current_snapshot_prices": ",".join(used_snapshot),
    }


def _append_or_replace_date(path: str | Path, row: pd.DataFrame, date: str) -> None:
    existing = _read_csv(path)
    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"].astype(str).ne(str(date))]
    out = pd.concat([existing, row], ignore_index=True) if not existing.empty else row
    out.to_csv(path, index=False)


def _legacy_vol_target_exposure() -> tuple[float, float, float]:
    rolling_vol = _legacy_rolling_growth_vol()
    previous = _previous_exposure()
    raw_unfloored = previous if not np.isfinite(rolling_vol) or rolling_vol <= 0 else TARGET_VOL / rolling_vol
    raw = float(np.clip(raw_unfloored, MIN_EXPOSURE, MAX_EXPOSURE))
    change = float(np.clip(raw - previous, -MAX_EXPOSURE_CHANGE, MAX_EXPOSURE_CHANGE))
    exposure = float(np.clip(previous + change, MIN_EXPOSURE, MAX_EXPOSURE))
    return exposure, raw, rolling_vol


def _vol_target_exposure(
    selected_tickers: list[str] | None = None, date: pd.Timestamp | None = None, allow_stale: bool = False
) -> tuple[float, float, float, dict[str, object]]:
    selected_tickers = selected_tickers or []
    date = pd.Timestamp(date).normalize() if date is not None else pd.Timestamp.today().normalize()
    fresh = _fresh_portfolio_volatility(selected_tickers, date)
    if not fresh.get("ok", False):
        if not allow_stale:
            raise ValueError("Fresh growth volatility could not be computed. Use --allow-stale-growth-volatility to override.")
        exposure, raw, rolling_vol = _legacy_vol_target_exposure()
        meta = {
            **fresh,
            "volatility_source": "stale_growth_volatility_targeting_daily_returns",
            "is_fresh": False,
            "used_stale_fallback": True,
        }
        return exposure, raw, rolling_vol, meta
    rolling_vol = float(fresh["estimated_portfolio_vol"])
    raw_unfloored = TARGET_VOL / rolling_vol
    raw = float(np.clip(raw_unfloored, MIN_EXPOSURE, MAX_EXPOSURE))
    previous = _previous_exposure()
    change = float(np.clip(raw - previous, -MAX_EXPOSURE_CHANGE, MAX_EXPOSURE_CHANGE))
    exposure = float(np.clip(previous + change, MIN_EXPOSURE, MAX_EXPOSURE))
    meta = {
        **fresh,
        "volatility_source": "fresh_selected_holdings_ohlcv",
        "is_fresh": str(fresh.get("source_date", "")) >= date.strftime("%Y-%m-%d"),
        "used_stale_fallback": False,
        "raw_unfloored_exposure": raw_unfloored,
    }
    return exposure, raw, rolling_vol, meta


def generate_current_growth_features(
    overwrite_same_day: bool = True, allow_stale_growth_volatility: bool = False, as_of_date: str | pd.Timestamp | None = None
) -> dict[str, object]:
    latest = _latest_forecast(as_of_date=as_of_date)
    if latest.empty:
        raise ValueError("forecast_history.csv is missing or empty.")
    date = pd.Timestamp(latest["date"].max())
    prior = _prior_growth_tickers()
    df = latest.copy()
    if "ticker" not in df.columns:
        raise ValueError("forecast_history.csv missing ticker column.")
    exact_raw_target = _num(df.get("raw_target_return_exact", pd.Series(index=df.index, dtype=float)))
    exact_available = bool(exact_raw_target.notna().any())
    proxy_raw_target = np.where(
        (_num(df.get("current_price", pd.Series(index=df.index, dtype=float))) > 0)
        & _num(df.get("target_price", pd.Series(index=df.index, dtype=float))).notna(),
        _num(df["target_price"]) / _num(df["current_price"]) - 1.0,
        _num(df.get("expected_total_return", pd.Series(index=df.index, dtype=float))),
    )
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["raw_target_return"] = exact_raw_target if exact_available else proxy_raw_target
    df["raw_target_return"] = _num(df["raw_target_return"])
    df["raw_target_feature_source"] = "raw_target_return_exact" if exact_available else "target_implied_proxy"
    df["exact_raw_target_available"] = exact_available

    pre_quality_positive = df[df["raw_target_return"] > 0].sort_values("raw_target_return", ascending=False)
    pre_quality_base = pre_quality_positive.head(BASE_POSITIONS)["ticker"].astype(str).tolist()
    pre_quality_soft = df[df["ticker"].astype(str).isin(prior) & (df["raw_target_return"] > 0)]["ticker"].astype(str).tolist()
    selected_before_quality = list(dict.fromkeys(pre_quality_base + pre_quality_soft))[:MAX_POSITIONS]

    yahoo_fetch_candidates = list(
        dict.fromkeys(selected_before_quality + pre_quality_positive.head(20)["ticker"].astype(str).tolist() + list(prior))
    )
    df, quality_report, quality_exclusions = apply_growth_universe_quality_filter(df, date, yahoo_fetch_tickers=yahoo_fetch_candidates)
    eligible = df["quality_pass"].fillna(False).astype(bool)
    df["raw_target_rank"] = np.nan
    df["raw_target_rank_pct"] = np.nan
    df.loc[eligible, "raw_target_rank"] = df.loc[eligible, "raw_target_return"].rank(ascending=False, method="first")
    df.loc[eligible, "raw_target_rank_pct"] = df.loc[eligible, "raw_target_return"].rank(ascending=False, pct=True)
    df["raw_target_selected"] = False
    positive = df[eligible & (df["raw_target_return"] > 0)].sort_values("raw_target_return", ascending=False)
    base_tickers = positive.head(BASE_POSITIONS)["ticker"].astype(str).tolist()
    soft_keep = df[eligible & df["ticker"].astype(str).isin(prior) & (df["raw_target_return"] > 0)]["ticker"].astype(str).tolist()
    selected_tickers = list(dict.fromkeys(base_tickers + soft_keep))[:MAX_POSITIONS]
    selected_tickers_after_quality = selected_tickers.copy()
    selected_tickers, holdings_audit, holdings_replacements = audit_and_filter_selected_holdings(df, selected_tickers, date, MAX_POSITIONS)
    if not holdings_audit.empty and "ticker" in holdings_audit.columns:
        audit_cols = [
            "ticker",
            "holding_quality_classification",
            "holding_risk_notes",
            "company_name",
            "sector",
            "industry",
            "exchange",
            "avg_volume_20d",
            "return_1d",
            "return_5d",
            "return_20d",
            "is_crypto_linked",
            "is_spac_linked",
            "is_biotech_binary_risk",
            "is_adr",
            "is_microcap_or_low_float",
            "is_distressed",
        ]
        df = df.merge(holdings_audit[[c for c in audit_cols if c in holdings_audit.columns]], on="ticker", how="left")
    else:
        df["holding_quality_classification"] = "not_a_final_holding"
        df["holding_risk_notes"] = ""
    if "holding_quality_classification" in df.columns:
        df["holding_quality_classification"] = df["holding_quality_classification"].fillna("not_a_final_holding")
    if "holding_risk_notes" in df.columns:
        df["holding_risk_notes"] = df["holding_risk_notes"].fillna("")
    df.loc[df["ticker"].astype(str).isin(selected_tickers), "raw_target_selected"] = True
    df["prior_position_status"] = np.where(df["ticker"].astype(str).isin(prior), "prior_position", "new_or_unheld")
    df["soft_exit_status"] = np.where(
        ~eligible & df["ticker"].astype(str).isin(prior),
        "blocked_by_quality_filter",
        np.where(
            df["ticker"].astype(str).isin(prior) & (df["raw_target_return"] > 0),
            "retained_positive_raw_target",
            np.where(df["ticker"].astype(str).isin(prior), "exit_nonpositive_raw_target", "not_prior_position"),
        ),
    )
    exposure, raw_exposure, rolling_vol, vol_meta = _vol_target_exposure(selected_tickers, date, allow_stale=allow_stale_growth_volatility)
    exposure_cap = _configured_exposure_cap()
    dual_trend = _dual_trend_filter(date)
    dual_trend_cap = float(dual_trend["dual_trend_cap"])
    uncapped_exposure = exposure
    final_exposure = float(np.clip(min(uncapped_exposure, exposure_cap, dual_trend_cap), 0.0, 1.0))
    cash = 1.0 - final_exposure
    vol_row = pd.DataFrame(
        [
            {
                "date": date.date().isoformat(),
                "selected_tickers": ",".join(selected_tickers),
                "volatility_source": vol_meta.get("volatility_source", ""),
                "volatility_source_date": vol_meta.get("source_date", ""),
                "estimated_portfolio_vol": rolling_vol,
                "target_vol": TARGET_VOL,
                "uncapped_exposure": vol_meta.get("raw_unfloored_exposure", raw_exposure),
                "min_exposure": MIN_EXPOSURE,
                "exposure_cap_60": exposure_cap,
                "dual_trend_cap": dual_trend_cap,
                "final_exposure": final_exposure,
                "is_fresh": vol_meta.get("is_fresh", False),
                "sample_size": vol_meta.get("sample_size", 0),
                "missing_tickers": vol_meta.get("missing_tickers", ""),
                "used_stale_fallback": vol_meta.get("used_stale_fallback", False),
                "used_current_snapshot_prices": vol_meta.get("used_current_snapshot_prices", ""),
            }
        ]
    )
    _append_or_replace_date(FRESH_VOL_FILE, vol_row, date.date().isoformat())
    _append_or_replace_date(VOL_PIPELINE_AUDIT_FILE, vol_row, date.date().isoformat())
    final_weight = final_exposure / len(selected_tickers) if selected_tickers else 0.0
    df["vol_target_exposure"] = uncapped_exposure
    df["volatility_target_exposure"] = final_exposure
    df["uncapped_volatility_target_exposure"] = uncapped_exposure
    df["raw_volatility_target_exposure"] = raw_exposure
    df["exposure_cap"] = exposure_cap
    df["exposure_cap_60"] = exposure_cap
    df["dual_trend_cap"] = dual_trend_cap
    df["final_exposure"] = final_exposure
    for key, value in dual_trend.items():
        df[key] = value
    df["rolling_volatility_used"] = rolling_vol
    df["volatility_source"] = vol_meta.get("volatility_source", "")
    df["volatility_source_date"] = vol_meta.get("source_date", "")
    df["volatility_is_fresh"] = vol_meta.get("is_fresh", False)
    df["final_growth_weight"] = np.where(df["raw_target_selected"], final_weight, 0.0)
    df["cash_weight"] = cash
    cfg = _growth_paper_config()
    df["data_source"] = "current_growth_feature_generation"
    df["growth_paper_model"] = str(cfg.get("active_growth_paper_model", "growth_champion_final"))
    df["growth_paper_variant"] = str(cfg.get("active_variant", "growth_v1_exposure_cap_60_dual_trend_filter"))
    df["fallback_reason"] = (
        "exact raw pre-signal target captured from expected_returns_model diagnostics"
        if exact_available
        else "raw pre-signal target component unavailable in forecast_history; used target_price/current_price proxy"
    )
    df["exact_growth_features_available"] = exact_available

    raw_cols = [
        "date",
        "ticker",
        "current_price",
        "target_price",
        "raw_target_return",
        "raw_target_return_exact",
        "raw_expected_daily_return_exact",
        "raw_target_price_exact",
        "time_to_target",
        "signal_strength_adjustment_value",
        "final_expected_return_after_adjustments",
        "raw_target_rank",
        "raw_target_rank_pct",
        "raw_target_feature_source",
        "exact_raw_target_available",
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
        "holding_quality_classification",
        "holding_risk_notes",
        "company_name",
        "sector",
        "industry",
        "exchange",
        "avg_volume_20d",
        "return_1d",
        "return_5d",
        "return_20d",
        "is_crypto_linked",
        "is_spac_linked",
        "is_biotech_binary_risk",
        "is_adr",
        "is_microcap_or_low_float",
        "is_distressed",
        "realized_vol_60d",
        "fallback_reason",
    ]
    growth_cols = [
        "date",
        "ticker",
        "raw_target_return",
        "raw_target_return_exact",
        "raw_expected_daily_return_exact",
        "raw_target_price_exact",
        "time_to_target",
        "signal_strength_adjustment_value",
        "final_expected_return_after_adjustments",
        "raw_target_rank",
        "raw_target_feature_source",
        "exact_raw_target_available",
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
        "holding_quality_classification",
        "holding_risk_notes",
        "company_name",
        "sector",
        "industry",
        "exchange",
        "avg_volume_20d",
        "return_1d",
        "return_5d",
        "return_20d",
        "is_crypto_linked",
        "is_spac_linked",
        "is_biotech_binary_risk",
        "is_adr",
        "is_microcap_or_low_float",
        "is_distressed",
        "realized_vol_60d",
        "raw_target_selected",
        "soft_exit_status",
        "prior_position_status",
        "vol_target_exposure",
        "volatility_target_exposure",
        "uncapped_volatility_target_exposure",
        "raw_volatility_target_exposure",
        "exposure_cap",
        "exposure_cap_60",
        "dual_trend_cap",
        "final_exposure",
        "rolling_volatility_used",
        "volatility_source",
        "volatility_source_date",
        "volatility_is_fresh",
        "spy_close",
        "spy_ma_200",
        "qqq_close",
        "qqq_ma_200",
        "spy_below_200d",
        "qqq_below_200d",
        "dual_trend_reason",
        "rolling_volatility_used",
        "volatility_source",
        "volatility_source_date",
        "volatility_is_fresh",
        "final_growth_weight",
        "cash_weight",
        "data_source",
        "growth_paper_model",
        "growth_paper_variant",
        "exact_growth_features_available",
        "fallback_reason",
    ]
    allocation = df[df["raw_target_selected"]].copy()
    allocation_cols = [
        "date",
        "ticker",
        "current_price",
        "raw_target_return",
        "raw_target_return_exact",
        "raw_expected_daily_return_exact",
        "raw_target_price_exact",
        "time_to_target",
        "signal_strength_adjustment_value",
        "final_expected_return_after_adjustments",
        "raw_target_rank",
        "raw_target_feature_source",
        "exact_raw_target_available",
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
        "holding_quality_classification",
        "holding_risk_notes",
        "company_name",
        "sector",
        "industry",
        "exchange",
        "avg_volume_20d",
        "return_1d",
        "return_5d",
        "return_20d",
        "is_crypto_linked",
        "is_spac_linked",
        "is_biotech_binary_risk",
        "is_adr",
        "is_microcap_or_low_float",
        "is_distressed",
        "realized_vol_60d",
        "raw_target_selected",
        "soft_exit_status",
        "prior_position_status",
        "vol_target_exposure",
        "volatility_target_exposure",
        "uncapped_volatility_target_exposure",
        "exposure_cap",
        "exposure_cap_60",
        "dual_trend_cap",
        "final_exposure",
        "rolling_volatility_used",
        "volatility_source",
        "volatility_source_date",
        "volatility_is_fresh",
        "spy_close",
        "spy_ma_200",
        "qqq_close",
        "qqq_ma_200",
        "spy_below_200d",
        "qqq_below_200d",
        "dual_trend_reason",
        "final_growth_weight",
        "cash_weight",
        "data_source",
        "growth_paper_model",
        "growth_paper_variant",
        "exact_growth_features_available",
        "fallback_reason",
    ]
    df[[c for c in raw_cols if c in df.columns]].to_csv(RAW_FEATURES_FILE, index=False)
    df[[c for c in growth_cols if c in df.columns]].to_csv(GROWTH_FEATURES_FILE, index=False)
    allocation[[c for c in allocation_cols if c in allocation.columns]].to_csv(GROWTH_ALLOCATION_FILE, index=False)

    print("\n===== CURRENT GROWTH FEATURE GENERATION =====")
    print(f"date: {date.date().isoformat()}")
    print(f"exact growth features available: {exact_available}")
    print(f"raw target feature source: {df['raw_target_feature_source'].iloc[0]}")
    print(f"selected tickers before quality filter: {', '.join(selected_before_quality)}")
    if not quality_exclusions.empty:
        print("excluded tickers and reasons:")
        print(quality_exclusions[["ticker", "exclusion_reason"]].to_string(index=False))
    else:
        print("excluded tickers and reasons: none")
    print(f"selected tickers after quality filter: {', '.join(selected_tickers_after_quality)}")
    if not holdings_audit.empty:
        print("final selected holdings sanity check:")
        show_cols = [c for c in ["ticker", "holding_quality_classification", "holding_risk_notes"] if c in holdings_audit.columns]
        print(holdings_audit[show_cols].to_string(index=False))
    if not holdings_replacements.empty:
        print("sanity replacements:")
        print(holdings_replacements.to_string(index=False))
    print(f"selected tickers: {', '.join(selected_tickers)}")
    print(f"growth paper model: {_growth_paper_config().get('active_growth_paper_model', 'growth_champion_final')!s}")
    print(f"exposure cap 60: {exposure_cap:.6f}")
    print(f"SPY close / 200D MA / below: {dual_trend['spy_close']:.4f} / {dual_trend['spy_ma_200']:.4f} / {dual_trend['spy_below_200d']}")
    print(f"QQQ close / 200D MA / below: {dual_trend['qqq_close']:.4f} / {dual_trend['qqq_ma_200']:.4f} / {dual_trend['qqq_below_200d']}")
    print(f"dual trend cap: {dual_trend_cap:.6f} ({dual_trend['dual_trend_reason']})")
    print(f"volatility source: {vol_meta.get('volatility_source', '')}")
    print(f"volatility source date: {vol_meta.get('source_date', '')}")
    print(f"estimated portfolio volatility: {rolling_vol:.6f}")
    print(f"volatility target exposure uncapped: {uncapped_exposure:.6f}")
    print(f"final exposure: {final_exposure:.6f}")
    print(f"cash: {cash:.6f}")
    print("raw target ranks:")
    print(
        df[df["ticker"].astype(str).isin(selected_tickers)][
            ["ticker", "raw_target_return", "raw_target_rank", "soft_exit_status", "final_growth_weight"]
        ].to_string(index=False)
    )
    print(f"fallback reason: {df['fallback_reason'].iloc[0]}")
    print(f"Saved: {Path(RAW_FEATURES_FILE).resolve()}")
    print(f"Saved: {Path(GROWTH_FEATURES_FILE).resolve()}")
    print(f"Saved: {Path(GROWTH_ALLOCATION_FILE).resolve()}")
    return {
        "date": date.date().isoformat(),
        "selected_tickers_before_quality_filter": selected_before_quality,
        "selected_tickers_after_quality_filter": selected_tickers_after_quality,
        "selected_tickers": selected_tickers,
        "rejected_holdings": holdings_audit.loc[
            holdings_audit["holding_quality_classification"].eq("reject_from_growth_universe"), "ticker"
        ]
        .astype(str)
        .tolist()
        if not holdings_audit.empty and "holding_quality_classification" in holdings_audit.columns
        else [],
        "replacement_tickers": holdings_replacements["replacement_ticker"].astype(str).tolist()
        if not holdings_replacements.empty and "replacement_ticker" in holdings_replacements.columns
        else [],
        "excluded_tickers": quality_exclusions["ticker"].astype(str).tolist()
        if not quality_exclusions.empty and "ticker" in quality_exclusions.columns
        else [],
        "exact_growth_features_available": exact_available,
        "raw_target_feature_source": str(df["raw_target_feature_source"].iloc[0]),
        "volatility_source": vol_meta.get("volatility_source", ""),
        "volatility_source_date": vol_meta.get("source_date", ""),
        "volatility_is_fresh": vol_meta.get("is_fresh", False),
        "vol_target_exposure": uncapped_exposure,
        "volatility_target_exposure": final_exposure,
        "uncapped_volatility_target_exposure": uncapped_exposure,
        "exposure_cap": exposure_cap,
        "exposure_cap_60": exposure_cap,
        "dual_trend_cap": dual_trend_cap,
        "final_exposure": final_exposure,
        "spy_below_200d": dual_trend["spy_below_200d"],
        "qqq_below_200d": dual_trend["qqq_below_200d"],
        "cash_weight": cash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate current Growth Candidate v1 features for paper trading.")
    parser.add_argument("--overwrite-same-day", action="store_true")
    parser.add_argument("--allow-stale-growth-volatility", action="store_true")
    parser.add_argument("--as-of-date", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_current_growth_features(
        overwrite_same_day=args.overwrite_same_day,
        allow_stale_growth_volatility=args.allow_stale_growth_volatility,
        as_of_date=args.as_of_date,
    )
