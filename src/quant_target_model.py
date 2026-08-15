from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_features import (
    compute_asset_quant_features,
    kalman_local_level,
)

TRADING_DAYS_PER_YEAR = 252
EPS = 1e-12


def _clean_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    clean = prices_df.replace([np.inf, -np.inf], np.nan).ffill().dropna(axis=1, how="all")
    return clean.ffill().bfill()


def _safe_log_returns(series: pd.Series) -> pd.Series:
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return np.log(clean / clean.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


def _clip_target(target: float, current_price: float, max_move: float) -> float:
    lower = current_price * (1.0 - max_move)
    upper = current_price * (1.0 + max_move)
    return float(np.clip(target, lower, upper))


def _ou_half_life_log_price(series: pd.Series, lookback: int = 126) -> float:
    log_prices = np.log(pd.Series(series, dtype=float).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    log_prices = log_prices.tail(min(lookback, len(log_prices)))
    if len(log_prices) < 30:
        return float(TRADING_DAYS_PER_YEAR)

    demeaned = log_prices - float(log_prices.mean())
    lagged = demeaned.shift(1).dropna()
    delta = demeaned.diff().dropna().reindex(lagged.index)
    x = lagged.to_numpy(dtype=float)
    y = delta.to_numpy(dtype=float)
    denom = float(np.dot(x, x))
    if denom <= EPS:
        return float(TRADING_DAYS_PER_YEAR)

    beta = float(np.dot(x, y) / denom)
    if beta >= 0:
        return float(TRADING_DAYS_PER_YEAR)
    return float(np.clip(-np.log(2.0) / beta, 1.0, TRADING_DAYS_PER_YEAR))


def generate_quant_targets(
    prices_df: pd.DataFrame,
    old_target_prices: pd.Series | dict[str, float],
    regime_type: str = "neutral",
    horizon_days: int = 20,
    blend_weight: float = 0.15,
) -> dict[str, pd.Series]:
    """
    Quantitative target framework.

    The GBM median target uses S_T = S_0 * exp((mu - 0.5*sigma^2) * h).
    This is a conservative median/log-price projection, not the arithmetic
    expectation E[S_T] = S_0 * exp(mu * h). The arithmetic GBM target is kept
    separately as a diagnostic-only series.

    Default usage is diagnostic: downstream code decides whether to blend.
    """
    close = _clean_prices(prices_df)
    if close.empty:
        raise ValueError("prices_df cannot be empty for quant target generation.")

    old_target = pd.Series(old_target_prices, dtype=float).reindex(close.columns)
    current = close.iloc[-1].astype(float)
    quant_features = compute_asset_quant_features(close).reindex(close.columns)

    gbm_median_targets: dict[str, float] = {}
    gbm_expected_targets: dict[str, float] = {}
    kalman_targets: dict[str, float] = {}
    ou_targets: dict[str, float] = {}
    quant_targets: dict[str, float] = {}
    confidences: dict[str, float] = {}
    selected_methods: dict[str, str] = {}

    regime_multiplier = {
        "risk_on": 1.05,
        "neutral": 0.90,
        "risk_off": 0.70,
        "high_volatility": 0.80,
        "low_volatility": 1.00,
    }.get(str(regime_type), 0.90)

    h = max(1, int(horizon_days))
    for ticker in close.columns:
        series = close[ticker].dropna()
        spot = float(current.loc[ticker])
        if len(series) < 30 or spot <= 0:
            gbm_median_targets[ticker] = spot
            gbm_expected_targets[ticker] = spot
            kalman_targets[ticker] = spot
            ou_targets[ticker] = spot
            quant_targets[ticker] = spot
            confidences[ticker] = 0.0
            selected_methods[ticker] = "insufficient_data"
            continue

        returns = _safe_log_returns(series)
        mu = float(returns.tail(60).mean()) if len(returns) else 0.0
        garch_vol = (
            float(quant_features.loc[ticker].get("garch_volatility", returns.std()))
            if ticker in quant_features.index
            else float(returns.std())
        )
        egarch_vol = (
            float(quant_features.loc[ticker].get("egarch_volatility", returns.std()))
            if ticker in quant_features.index
            else float(returns.std())
        )
        sigma = max(float(returns.tail(60).std()) if len(returns) else 0.0, garch_vol, egarch_vol, EPS)

        gbm_drift = mu - 0.5 * sigma**2
        gbm_median_target = spot * float(np.exp(gbm_drift * h))
        gbm_expected_target = spot * float(np.exp(mu * h))

        kalman = kalman_local_level(series)
        if len(kalman) >= 6:
            kalman_slope = float((kalman.iloc[-1] - kalman.iloc[-6]) / 5.0)
        else:
            kalman_slope = 0.0
        kalman_target = spot + kalman_slope * h

        log_prices = np.log(series.replace(0, np.nan)).dropna()
        long_mean_log_price = float(log_prices.tail(min(126, len(log_prices))).mean())
        half_life = float(_ou_half_life_log_price(series))
        kappa = np.log(2.0) / max(half_life, 1.0)
        ou_expected_log = long_mean_log_price + (np.log(spot) - long_mean_log_price) * np.exp(-kappa * h)
        ou_target = float(np.exp(ou_expected_log))

        entropy = float(quant_features.loc[ticker].get("entropy", 0.5)) if ticker in quant_features.index else 0.5
        hawkes = float(quant_features.loc[ticker].get("hawkes_downside_intensity", 0.0)) if ticker in quant_features.index else 0.0
        hurst = float(quant_features.loc[ticker].get("hurst_exponent", 0.5)) if ticker in quant_features.index else 0.5
        volatility_confidence = float(np.clip(1.0 / (1.0 + sigma * np.sqrt(TRADING_DAYS_PER_YEAR)), 0.0, 1.0))
        entropy_confidence = float(np.clip(1.0 - entropy, 0.0, 1.0))
        hawkes_confidence = float(np.clip(1.0 - hawkes, 0.0, 1.0))
        hurst_confidence = float(np.clip(1.0 - abs(hurst - 0.5), 0.0, 1.0))
        confidence = 0.35 * volatility_confidence + 0.25 * entropy_confidence + 0.25 * hawkes_confidence + 0.15 * hurst_confidence
        confidence = float(np.clip(confidence * regime_multiplier, 0.0, 1.0))

        trend_weight = float(np.clip(0.50 + (hurst - 0.5), 0.25, 0.75))
        mean_reversion_weight = (1.0 - trend_weight) * (1.25 if regime_type in {"neutral", "risk_off"} else 0.75)
        mean_reversion_weight = float(np.clip(mean_reversion_weight, 0.05, 0.35))
        projected_target = (
            0.50 * gbm_median_target + 0.35 * kalman_target + 0.15 * (trend_weight * kalman_target + mean_reversion_weight * ou_target)
        )

        max_move = float(np.clip(3.0 * sigma * np.sqrt(h), 0.03, 0.35))
        quant_target = _clip_target(
            confidence * projected_target + (1.0 - confidence) * spot,
            spot,
            max_move,
        )

        gbm_median_targets[ticker] = _clip_target(gbm_median_target, spot, max_move)
        gbm_expected_targets[ticker] = _clip_target(gbm_expected_target, spot, max_move)
        kalman_targets[ticker] = _clip_target(kalman_target, spot, max_move)
        ou_targets[ticker] = _clip_target(ou_target, spot, max_move)
        quant_targets[ticker] = quant_target
        confidences[ticker] = confidence
        selected_methods[ticker] = "regime_confidence_blend"

    quant_target_price = pd.Series(quant_targets, dtype=float).reindex(close.columns).fillna(current)
    old_target = old_target.fillna(current)
    target_blend_weight = pd.Series(float(np.clip(blend_weight, 0.0, 1.0)), index=close.columns, dtype=float)
    final_blended_target = (1.0 - target_blend_weight) * old_target + target_blend_weight * quant_target_price

    return {
        "old_target_price": old_target.astype(float),
        "quant_target_price": quant_target_price.astype(float),
        "target_blend_weight": target_blend_weight.astype(float),
        "final_blended_target": final_blended_target.astype(float),
        "gbm_median_target": pd.Series(gbm_median_targets, dtype=float).reindex(close.columns).fillna(current),
        "gbm_expected_target": pd.Series(gbm_expected_targets, dtype=float).reindex(close.columns).fillna(current),
        "gbm_target": pd.Series(gbm_median_targets, dtype=float).reindex(close.columns).fillna(current),
        "kalman_target": pd.Series(kalman_targets, dtype=float).reindex(close.columns).fillna(current),
        "ou_target": pd.Series(ou_targets, dtype=float).reindex(close.columns).fillna(current),
        "target_confidence": pd.Series(confidences, dtype=float).reindex(close.columns).fillna(0.0),
        "target_method_selected": pd.Series(selected_methods, dtype=object).reindex(close.columns).fillna("fallback"),
    }
