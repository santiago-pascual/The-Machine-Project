from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import yfinance as yf


def _safe_series(series: pd.Series) -> pd.Series:
    return pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).ffill().bfill()


def _normalized_slope(series: pd.Series, lookback: int = 5) -> float:
    s = _safe_series(series).dropna()
    if len(s) <= lookback:
        return 0.5
    last = float(s.iloc[-1])
    if abs(last) < 1e-12:
        return 0.5
    raw = float((s.iloc[-1] - s.iloc[-1 - lookback]) / abs(last))
    return float(np.clip((np.tanh(raw * 80.0) + 1.0) / 2.0, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _prox(distance: float, k: float) -> float:
    return float(np.exp(-k * max(distance, 0.0)))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    d = float(den)
    if not np.isfinite(d) or abs(d) < 1e-12:
        return float(default)
    return float(num / d)


def _consecutive_true_tail(mask: pd.Series) -> int:
    values = pd.Series(mask, dtype=bool).to_numpy()
    count = 0
    for value in values[::-1]:
        if not value:
            break
        count += 1
    return count


def _wma(series: pd.Series, window: int) -> pd.Series:
    s = _safe_series(series)
    weights = np.arange(1, window + 1, dtype=float)
    return s.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


@contextmanager
def _temporary_disable_proxies():
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    previous = {k: os.environ.get(k) for k in proxy_keys}
    try:
        for k in proxy_keys:
            os.environ.pop(k, None)
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _download_spy_close(period: str = "2y") -> pd.Series:
    try:
        with _temporary_disable_proxies():
            data = yf.download(
                "SPY",
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=20,
            )
    except Exception:
        return pd.Series(dtype=float)
    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.empty or close.shape[1] == 0:
            return pd.Series(dtype=float)
        close = close.iloc[:, 0]
    return _safe_series(close).dropna()


def compute_daily_timing(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for ticker in prices_df.columns:
        price_series = _safe_series(prices_df[ticker]).dropna()
        if price_series.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "price": 0.0,
                    "ema21": 0.0,
                    "ema30": 0.0,
                    "ema150": 0.0,
                    "ema200": 0.0,
                    "entry_score": 0.0,
                    "entry_valid": False,
                    "trend_score": 0.0,
                    "structure_score": 0.0,
                    "short_pullback_score": 0.0,
                    "long_pullback_score": 0.0,
                    "extension_penalty": 1.0,
                    "ema_timing_score": 0.0,
                    "timing_action": "no_buy",
                    "timing_reason": "missing_price_data",
                }
            )
            continue

        ema21 = price_series.ewm(span=21, adjust=False).mean()
        ema30 = price_series.ewm(span=30, adjust=False).mean()
        ema150 = price_series.ewm(span=150, adjust=False).mean()
        ema200 = price_series.ewm(span=200, adjust=False).mean()

        price = float(price_series.iloc[-1])
        v_ema21 = float(ema21.iloc[-1])
        v_ema30 = float(ema30.iloc[-1])
        v_ema150 = float(ema150.iloc[-1])
        v_ema200 = float(ema200.iloc[-1])

        price_vs_ema21 = _safe_div(price - v_ema21, v_ema21, default=0.0)
        price_vs_ema30 = _safe_div(price - v_ema30, v_ema30, default=0.0)
        ema_spread = _safe_div(v_ema21 - v_ema30, v_ema30, default=0.0)
        short_trend_bullish = bool((price_vs_ema21 > 0) and (price_vs_ema30 > 0) and (ema_spread > 0))

        slope_ema21 = _normalized_slope(ema21, lookback=5)
        slope_ema150 = _normalized_slope(ema150, lookback=8)

        trend_raw = (
            0.40 * (2.0 * _sigmoid(price_vs_ema21 * 8.0) - 1.0)
            + 0.30 * (2.0 * _sigmoid(price_vs_ema30 * 8.0) - 1.0)
            + 0.20 * (2.0 * _sigmoid(ema_spread * 14.0) - 1.0)
            + 0.10 * (2.0 * slope_ema21 - 1.0)
        )
        trend_score = float(np.clip((np.tanh(trend_raw * 1.4) + 1.0) / 2.0, 0.0, 1.0))

        ema30_vs_ema150 = _safe_div(v_ema30 - v_ema150, v_ema150, default=0.0)
        ema150_vs_ema200 = _safe_div(v_ema150 - v_ema200, v_ema200, default=0.0)
        price_vs_ema150 = _safe_div(price - v_ema150, v_ema150, default=0.0)
        price_vs_ema200 = _safe_div(price - v_ema200, v_ema200, default=0.0)
        short_bull_days = _consecutive_true_tail(ema21 > ema30)
        recent_return_40 = float(price_series.iloc[-1] / price_series.iloc[-41] - 1.0) if len(price_series) > 40 else 0.0
        recent_cross_gate = 1.0 - _sigmoid((short_bull_days - 35.0) * 0.22)
        long_recovery_gate = _sigmoid((0.08 - max(price_vs_ema150, price_vs_ema200)) * 18.0)
        sustained_rally_penalty = 1.0 - _sigmoid((recent_return_40 - 0.18) * 18.0)
        early_reversal_intensity = (
            _sigmoid(ema_spread * 18.0)
            * _sigmoid((0.04 - abs(ema30_vs_ema150)) * 18.0)
            * recent_cross_gate
            * long_recovery_gate
            * sustained_rally_penalty
        )
        structure_raw = (
            0.35 * (2.0 * _sigmoid(price_vs_ema150 * 7.0) - 1.0)
            + 0.25 * (2.0 * _sigmoid(price_vs_ema200 * 7.0) - 1.0)
            + 0.20 * (2.0 * _sigmoid(ema150_vs_ema200 * 16.0) - 1.0)
            + 0.10 * (2.0 * slope_ema150 - 1.0)
            + 0.10 * (2.0 * early_reversal_intensity - 1.0)
        )
        structure_score = float(np.clip((np.tanh(structure_raw * 1.3) + 1.0) / 2.0, 0.0, 1.0))

        dist_21 = _safe_div(price - v_ema21, v_ema21, default=0.0)
        dist_30 = _safe_div(price - v_ema30, v_ema30, default=0.0)
        dist_150 = _safe_div(price - v_ema150, v_ema150, default=0.0)
        dist_200 = _safe_div(price - v_ema200, v_ema200, default=0.0)

        pullback21 = np.exp(-22.0 * abs(dist_21))
        pullback30 = np.exp(-22.0 * abs(dist_30))
        short_zone = _sigmoid((0.03 - min(abs(dist_21), abs(dist_30))) * 110.0)
        short_trend_gate = _sigmoid((trend_score - 0.45) * 10.0)
        short_pullback_score = float(np.clip(short_trend_gate * short_zone * max(pullback21, pullback30), 0.0, 1.0))

        pullback150 = np.exp(-16.0 * abs(dist_150))
        pullback200 = np.exp(-16.0 * abs(dist_200))
        long_zone = _sigmoid((0.06 - min(abs(dist_150), abs(dist_200))) * 80.0)
        early_gate = _sigmoid((early_reversal_intensity - 0.35) * 9.0)
        long_pullback_score = float(np.clip(long_zone * max(pullback150, pullback200) * (0.4 + 0.6 * early_gate), 0.0, 1.0))

        overextension = max(dist_21, dist_30, 0.0)
        extension_penalty = float(np.clip(_sigmoid((overextension - 0.07) * 35.0), 0.0, 0.90))

        ema_bull_seq = _sigmoid(ema_spread * 18.0) * _sigmoid(ema30_vs_ema150 * 12.0) * _sigmoid(ema150_vs_ema200 * 15.0)
        ema_bear_seq = _sigmoid((-ema_spread) * 18.0)
        if ema_bull_seq > 0.70:
            structure_state = "strong_trend"
        elif early_reversal_intensity > 0.55 and short_bull_days <= 45 and recent_return_40 < 0.25:
            structure_state = "early_reversal"
        elif ema_bear_seq > 0.60:
            structure_state = "bearish_structure"
        else:
            structure_state = "intermediate_structure"

        ema_timing_score = (
            0.32 * trend_score
            + 0.30 * structure_score
            + 0.25 * short_pullback_score
            + 0.13 * long_pullback_score
        )
        ema_timing_score *= (1.0 - 0.40 * extension_penalty)
        ema_timing_score += 0.10 * early_reversal_intensity
        ema_timing_score = float(np.clip(ema_timing_score, 0.0, 1.0))

        entry_score = float(np.clip(0.55 * trend_score + 0.45 * structure_score, 0.0, 1.0))
        entry_valid = bool(entry_score >= 0.45)

        if entry_score < 0.30:
            timing_action = "no_buy"
            timing_reason = "low_entry_score"
        elif ema_bear_seq > 0.70 and trend_score < 0.35:
            timing_action = "exit_structure_break"
            timing_reason = "bearish_structure_persistent"
        elif trend_score < 0.40:
            timing_action = "reduce"
            timing_reason = "trend_weakening"
        elif short_pullback_score > 0.55:
            timing_action = "add_on_short_pullback"
            timing_reason = "short_pullback_edge"
        elif long_pullback_score > 0.55:
            timing_action = "add_on_long_support"
            timing_reason = "long_pullback_edge"
        else:
            timing_action = "hold"
            timing_reason = structure_state

        rows.append(
            {
                "ticker": ticker,
                "price": price,
                "ema21": v_ema21,
                "ema30": v_ema30,
                "ema150": v_ema150,
                "ema200": v_ema200,
                "entry_score": entry_score,
                "entry_valid": entry_valid,
                "trend_score": trend_score,
                "structure_score": structure_score,
                "short_pullback_score": float(np.clip(short_pullback_score, 0.0, 1.0)),
                "long_pullback_score": float(np.clip(long_pullback_score, 0.0, 1.0)),
                "extension_penalty": float(np.clip(extension_penalty, 0.0, 1.0)),
                "ema_timing_score": ema_timing_score,
                "timing_action": timing_action,
                "timing_reason": timing_reason,
            }
        )

    timing_df = pd.DataFrame(rows).set_index("ticker")
    return timing_df


def compute_weekly_timing(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    for ticker in prices_df.columns:
        s = _safe_series(prices_df[ticker]).dropna()
        if s.empty:
            rows.append({"ticker": ticker, "weekly_timing_score": 0.0, "weekly_entry_valid": False})
            continue
        weekly = s.resample("W-FRI").last().dropna()
        if weekly.empty:
            rows.append({"ticker": ticker, "weekly_timing_score": 0.0, "weekly_entry_valid": False})
            continue
        wma30 = _wma(weekly, 30)
        wma50 = _wma(weekly, 50)
        sma200 = weekly.rolling(200).mean()
        price = float(weekly.iloc[-1])
        v_wma30 = float(wma30.iloc[-1]) if not np.isnan(wma30.iloc[-1]) else price
        v_wma50 = float(wma50.iloc[-1]) if not np.isnan(wma50.iloc[-1]) else price
        v_sma200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else price
        entry_valid = bool(price > v_wma30 and price > v_wma50)
        structure = float(price > v_sma200)
        slope = _normalized_slope(wma30.fillna(method="ffill").fillna(method="bfill"), lookback=3)
        weekly_timing_score = float(np.clip(0.45 * float(entry_valid) + 0.35 * structure + 0.20 * slope, 0.0, 1.0))
        rows.append(
            {
                "ticker": ticker,
                "weekly_timing_score": weekly_timing_score,
                "weekly_entry_valid": entry_valid,
                "wma30": v_wma30,
                "wma50": v_wma50,
                "sma200": v_sma200,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def compute_asset_timing(prices_df: pd.DataFrame, timing_mode: str = "daily") -> pd.DataFrame:
    mode = str(timing_mode).lower()
    if mode == "weekly":
        return compute_weekly_timing(prices_df)
    return compute_daily_timing(prices_df)


def apply_timing_to_expected_returns(
    adjusted_expected_returns: pd.Series,
    timing_df: pd.DataFrame,
) -> pd.Series:
    expected = pd.Series(adjusted_expected_returns, dtype=float).copy()
    if expected.empty:
        return expected
    aligned = timing_df.reindex(expected.index)
    timing_score = aligned.get("ema_timing_score", pd.Series(0.5, index=expected.index)).fillna(0.5)
    entry_score = aligned.get("entry_score", pd.Series(0.5, index=expected.index)).fillna(0.5)
    structure_score = aligned.get("structure_score", pd.Series(0.5, index=expected.index)).fillna(0.5)

    timing_multiplier = 0.5 + timing_score
    entry_multiplier = 0.30 + 0.70 * np.power(entry_score.clip(0.0, 1.0), 1.15)
    structure_multiplier = 0.50 + 0.50 * structure_score.clip(0.0, 1.0)
    timed = expected * timing_multiplier * entry_multiplier * structure_multiplier
    return timed.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_spy_ema_regime(timing_mode: str = "daily") -> dict[str, float | str]:
    spy_close = _download_spy_close(period="3y")
    if spy_close.empty:
        return {
            "spy_price": 0.0,
            "spy_ema21": 0.0,
            "spy_ema30": 0.0,
            "spy_ema150": 0.0,
            "spy_ema200": 0.0,
            "macro_ema_score": 0.5,
            "spy_macro_regime": "neutral",
        }

    mode = str(timing_mode).lower()
    if mode == "weekly":
        series = spy_close.resample("W-FRI").last().dropna()
    else:
        series = spy_close

    ema21 = series.ewm(span=21, adjust=False).mean()
    ema30 = series.ewm(span=30, adjust=False).mean()
    ema150 = series.ewm(span=150, adjust=False).mean()
    ema200 = series.ewm(span=200, adjust=False).mean()

    price = float(series.iloc[-1])
    v21 = float(ema21.iloc[-1])
    v30 = float(ema30.iloc[-1])
    v150 = float(ema150.iloc[-1])
    v200 = float(ema200.iloc[-1])

    p21 = _safe_div(price - v21, v21, default=0.0)
    p30 = _safe_div(price - v30, v30, default=0.0)
    p150 = _safe_div(price - v150, v150, default=0.0)
    p200 = _safe_div(price - v200, v200, default=0.0)
    e2130 = _safe_div(v21 - v30, v30, default=0.0)
    e30150 = _safe_div(v30 - v150, v150, default=0.0)
    e150200 = _safe_div(v150 - v200, v200, default=0.0)

    short_score = (
        0.40 * _sigmoid(p21 * 10.0)
        + 0.30 * _sigmoid(p30 * 10.0)
        + 0.30 * _sigmoid(e2130 * 18.0)
    )
    long_score = (
        0.30 * _sigmoid(p150 * 7.0)
        + 0.30 * _sigmoid(p200 * 7.0)
        + 0.20 * _sigmoid(e30150 * 12.0)
        + 0.20 * _sigmoid(e150200 * 12.0)
    )
    macro_ema_score = float(np.clip(0.6 * short_score + 0.4 * long_score, 0.0, 1.0))
    if macro_ema_score >= 0.67:
        spy_macro_regime = "bullish"
    elif macro_ema_score >= 0.4:
        spy_macro_regime = "neutral"
    else:
        spy_macro_regime = "bearish"

    return {
        "spy_price": price,
        "spy_ema21": v21,
        "spy_ema30": v30,
        "spy_ema150": v150,
        "spy_ema200": v200,
        "macro_ema_score": float(macro_ema_score),
        "spy_macro_regime": spy_macro_regime,
    }
