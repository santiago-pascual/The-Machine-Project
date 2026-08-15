from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_series(values: pd.Series, fallback: float = 0.5) -> pd.Series:
    clean_values = values.replace([np.inf, -np.inf], np.nan).astype(float)
    if clean_values.dropna().empty:
        return pd.Series(fallback, index=clean_values.index, dtype=float)

    min_value = float(clean_values.min(skipna=True))
    max_value = float(clean_values.max(skipna=True))
    if max_value - min_value < 1e-12:
        return pd.Series(fallback, index=clean_values.index, dtype=float)

    normalized = (clean_values - min_value) / (max_value - min_value)
    return normalized.fillna(fallback).astype(float)


def _compute_trend_slope(ema20: pd.DataFrame, lookback: int = 5) -> pd.Series:
    slope = (ema20.iloc[-1] - ema20.iloc[-1 - lookback]) / max(lookback, 1)
    normalized_slope = np.tanh((slope / ema20.iloc[-1].replace(0, np.nan)).fillna(0.0) * 50.0)
    return normalized_slope.fillna(0.0).astype(float)


def generate_targets_advanced(prices_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Institutional-grade probabilistic target system.
    Returns a dictionary of Series indexed by ticker.
    """
    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    close = prices_df.ffill().copy()
    close = close.replace([np.inf, -np.inf], np.nan)
    close = close.dropna(axis=1, how="all")
    if close.empty:
        raise ValueError("No valid price columns after cleaning.")

    close = close.ffill().bfill()
    tickers = close.columns

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

    short_vol = returns.rolling(10, min_periods=5).std().iloc[-1]
    mid_vol = returns.rolling(20, min_periods=10).std().iloc[-1]
    long_vol = returns.rolling(60, min_periods=20).std().iloc[-1]
    atr_vol = close.diff().abs().rolling(14, min_periods=7).mean().iloc[-1] / close.iloc[-1].replace(0, np.nan)

    short_vol = short_vol.fillna(mid_vol).fillna(0.0)
    mid_vol = mid_vol.fillna(short_vol).fillna(0.0)
    long_vol = long_vol.fillna(mid_vol).fillna(0.0)
    atr_vol = atr_vol.replace([np.inf, -np.inf], np.nan).fillna(mid_vol).fillna(0.0)

    blended_vol = 0.4 * short_vol + 0.4 * mid_vol + 0.2 * long_vol

    momentum_20 = close.iloc[-1] / close.shift(20).iloc[-1] - 1
    momentum_60 = close.iloc[-1] / close.shift(60).iloc[-1] - 1
    momentum_20 = momentum_20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    momentum_60 = momentum_60.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()

    trend_slope = _compute_trend_slope(ema20, lookback=5)
    trend_alignment = pd.Series(0.0, index=tickers, dtype=float)
    bullish_mask = (ema20.iloc[-1] > ema50.iloc[-1]) & (ema50.iloc[-1] > ema100.iloc[-1])
    bearish_mask = (ema20.iloc[-1] < ema50.iloc[-1]) & (ema50.iloc[-1] < ema100.iloc[-1])
    trend_alignment.loc[bullish_mask] = 1.0
    trend_alignment.loc[bearish_mask] = -1.0

    trend_strength_raw = 0.6 * trend_slope + 0.4 * trend_alignment
    trend_strength = _normalize_series(trend_strength_raw).fillna(0.5)
    signal_strength = _normalize_series(0.5 * momentum_20 + 0.5 * trend_strength_raw).fillna(0.5)

    expected_move = 0.35 * blended_vol + 0.25 * atr_vol + 0.25 * momentum_20 + 0.15 * trend_strength
    expected_move = expected_move * (0.5 + signal_strength)
    expected_move = expected_move.clip(-0.2, 0.3)

    current_price = close.iloc[-1].replace(0, np.nan).fillna(close.iloc[-1].mean())
    upper_1sigma = current_price * (1 + expected_move)
    upper_2sigma = current_price * (1 + 2 * blended_vol)
    lower_1sigma = current_price * (1 - expected_move)

    target_price = pd.Series(index=tickers, dtype=float)
    high_conf = signal_strength > 0.75
    mid_conf = (signal_strength > 0.5) & ~high_conf
    low_conf = ~high_conf & ~mid_conf

    target_price.loc[high_conf] = upper_2sigma.loc[high_conf]
    target_price.loc[mid_conf] = upper_1sigma.loc[mid_conf]
    target_price.loc[low_conf] = current_price.loc[low_conf] * (1 + 0.5 * expected_move.loc[low_conf])

    max_allowed_move = 3 * blended_vol
    unrealistic_mask = expected_move > max_allowed_move
    adjusted_expected_move = expected_move.copy()
    adjusted_expected_move.loc[unrealistic_mask] = max_allowed_move.loc[unrealistic_mask]
    target_validity = pd.Series("valid", index=tickers, dtype=object)
    target_validity.loc[unrealistic_mask] = "unrealistic"

    trend_consistency = (returns.tail(20) > 0).mean().fillna(0.5)
    volatility_variance = returns.rolling(20, min_periods=10).std().tail(20).var().iloc[-1]
    volatility_stability_raw = 1 / (1 + volatility_variance.fillna(volatility_variance.median()).clip(lower=0))
    volatility_stability = _normalize_series(volatility_stability_raw).fillna(0.5)

    target_confidence = 0.4 * signal_strength + 0.3 * trend_consistency + 0.3 * volatility_stability
    target_confidence = target_confidence.clip(0.0, 1.0).fillna(0.5)

    target_price = target_price.replace([np.inf, -np.inf], np.nan).fillna(current_price)
    upper_1sigma = upper_1sigma.replace([np.inf, -np.inf], np.nan).fillna(current_price)
    upper_2sigma = upper_2sigma.replace([np.inf, -np.inf], np.nan).fillna(current_price)
    lower_1sigma = lower_1sigma.replace([np.inf, -np.inf], np.nan).fillna(current_price)
    adjusted_expected_move = adjusted_expected_move.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return {
        "target_price": target_price.astype(float),
        "expected_move": adjusted_expected_move.astype(float),
        "target_confidence": target_confidence.astype(float),
        "target_validity": target_validity,
        "upper_1sigma": upper_1sigma.astype(float),
        "upper_2sigma": upper_2sigma.astype(float),
        "lower_1sigma": lower_1sigma.astype(float),
    }
