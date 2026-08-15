from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_features import (
    egarch11_forecast_variance,
    fft_low_frequency_energy,
    garch11_forecast_variance,
    haar_wavelet_energy_ratio,
    hurst_exponent,
    kalman_local_level,
    shannon_entropy,
)

EPS = 1e-12

DEFAULT_COMPONENT_WEIGHTS = {
    "kalman_trend_score": 0.25,
    "momentum_score": 0.20,
    "hurst_persistence_score": 0.15,
    "entropy_cleanliness_score": 0.10,
    "volatility_stability_score": 0.10,
    "regime_trend_score": 0.10,
    "correlation_diversification_score": 0.05,
    "cycle_stability_score": 0.05,
}


def _clean_series(series: pd.Series) -> pd.Series:
    return pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()


def _sigmoid(value: float) -> float:
    x = float(np.clip(value, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _safe_score(value: float, default: float = 0.5) -> float:
    value = _safe_float(value, default)
    return float(np.clip(value, 0.0, 1.0))


def _pct_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.dropna().empty:
        ranked = pd.Series(0.5, index=series.index, dtype=float)
    else:
        ranked = values.rank(pct=True).fillna(0.5)
    if invert:
        ranked = 1.0 - ranked
    return ranked.clip(0.0, 1.0)


def _risk_adjusted_return(prices: pd.Series, returns: pd.Series, lookback: int) -> float:
    if len(prices) <= lookback or len(returns) < max(5, lookback // 2):
        return 0.0
    total_return = _safe_float(prices.iloc[-1] / prices.iloc[-lookback - 1] - 1.0)
    vol = _safe_float(returns.tail(lookback).std(ddof=1), 0.0)
    if vol <= EPS:
        return 0.0
    return float(total_return / (vol * np.sqrt(lookback)))


def _kalman_trend_score(prices: pd.Series, returns: pd.Series) -> tuple[float, dict[str, float]]:
    filtered = kalman_local_level(prices)
    if filtered.empty:
        return 0.5, {"kalman_slope_5d": 0.0, "kalman_slope_10d": 0.0, "kalman_slope_20d": 0.0}

    slopes: dict[str, float] = {}
    scores = []
    daily_vol = max(_safe_float(returns.tail(30).std(ddof=1), 0.0), EPS)
    for horizon in (5, 10, 20):
        if len(filtered) <= horizon:
            norm_slope = 0.0
        else:
            move = _safe_float(filtered.iloc[-1] / filtered.iloc[-horizon - 1] - 1.0)
            norm_slope = float(move / (daily_vol * np.sqrt(horizon)))
        slopes[f"kalman_slope_{horizon}d"] = norm_slope
        scores.append(_sigmoid(norm_slope * 0.9))
    return float(np.mean(scores)) if scores else 0.5, slopes


def _regime_trend_score(market_regime: dict[str, object] | None) -> tuple[float, str]:
    if not market_regime:
        return 0.5, "regime_unavailable"
    risk_score = _safe_float(market_regime.get("risk_score"), 0.0)
    confidence = _safe_float(market_regime.get("regime_confidence"), abs(risk_score))
    regime = str(market_regime.get("regime", market_regime.get("regime_type", "neutral")))
    if regime == "risk_on":
        base = 0.62 + 0.28 * confidence
    elif regime == "risk_off":
        base = 0.40 + 0.15 * max(risk_score, 0.0)
    else:
        base = 0.45 + 0.20 * abs(risk_score) * confidence
    return _safe_score(base), f"{regime}_risk_score={risk_score:.3f}"


def _correlation_scores(
    returns_df: pd.DataFrame,
    tickers: list[str],
    market_regime: dict[str, object] | None,
) -> pd.Series:
    if returns_df.empty or len(tickers) <= 1:
        return pd.Series(0.5, index=tickers, dtype=float)
    available = [ticker for ticker in tickers if ticker in returns_df.columns]
    corr = returns_df[available].corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
    risk_score = _safe_float((market_regime or {}).get("risk_score"), 0.0)
    weak_regime_penalty = 1.15 if risk_score < 0 else 0.85
    scores = {}
    for ticker in tickers:
        if ticker not in corr.columns or len(corr.columns) <= 1:
            scores[ticker] = 0.5
            continue
        avg_corr = _safe_float(corr.loc[ticker, [c for c in corr.columns if c != ticker]].mean(), 0.5)
        scores[ticker] = _safe_score(1.0 - weak_regime_penalty * avg_corr, 0.5)
    return pd.Series(scores, dtype=float)


def compute_trend_persistence(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame | None = None,
    selected_tickers: list[str] | None = None,
    diagnostics_df: pd.DataFrame | None = None,
    market_regime: dict[str, object] | None = None,
    component_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    tickers = selected_tickers or list(prices_df.columns)
    tickers = [ticker for ticker in tickers if ticker in prices_df.columns]
    if returns_df is None or returns_df.empty:
        returns_df = prices_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if diagnostics_df is None:
        diagnostics_df = pd.DataFrame(index=tickers)

    rows: list[dict[str, object]] = []
    regime_score, regime_reason = _regime_trend_score(market_regime)

    for ticker in tickers:
        prices = _clean_series(prices_df[ticker])
        returns = _clean_series(returns_df[ticker]) if ticker in returns_df.columns else prices.pct_change(fill_method=None).dropna()
        if len(prices) < 30 or returns.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "kalman_trend_score": 0.5,
                    "momentum_score": 0.5,
                    "hurst_persistence_score": 0.5,
                    "entropy_cleanliness_score": 0.5,
                    "raw_volatility": np.nan,
                    "regime_trend_score": regime_score,
                    "cycle_stability_score": 0.5,
                    "trend_persistence_reason": "insufficient_history",
                }
            )
            continue

        kalman_score, kalman_slopes = _kalman_trend_score(prices, returns)
        mom20 = _risk_adjusted_return(prices, returns, 20)
        mom60 = _risk_adjusted_return(prices, returns, 60)
        momentum_raw = 0.60 * mom20 + 0.40 * mom60
        momentum_score = _sigmoid(momentum_raw * 0.85)

        diag_row = diagnostics_df.loc[ticker] if ticker in diagnostics_df.index else pd.Series(dtype=float)
        hurst = _safe_float(diag_row.get("hurst_exponent"), np.nan)
        if not np.isfinite(hurst):
            hurst = hurst_exponent(prices)
        hurst_score = _safe_score(_sigmoid((hurst - 0.50) * 8.0))

        entropy = _safe_float(diag_row.get("shannon_entropy"), np.nan)
        if not np.isfinite(entropy):
            entropy = shannon_entropy(returns)
        entropy_score = _safe_score(1.0 - entropy)

        garch_vol = _safe_float(diag_row.get("garch_volatility"), np.nan)
        egarch_vol = _safe_float(diag_row.get("egarch_volatility"), np.nan)
        if not np.isfinite(garch_vol):
            garch_vol = float(np.sqrt(max(garch11_forecast_variance(returns), 0.0)))
        if not np.isfinite(egarch_vol):
            egarch_vol = float(np.sqrt(max(egarch11_forecast_variance(returns), 0.0)))
        realized_vol = _safe_float(returns.tail(30).std(ddof=1), 0.0)
        raw_volatility = float(np.nanmean([garch_vol, egarch_vol, realized_vol]))

        fft_energy = _safe_float(diag_row.get("fft_low_freq_energy"), np.nan)
        wavelet_energy = _safe_float(diag_row.get("haar_wavelet_energy"), np.nan)
        if not np.isfinite(fft_energy):
            fft_energy = fft_low_frequency_energy(prices)
        if not np.isfinite(wavelet_energy):
            wavelet_energy = haar_wavelet_energy_ratio(prices)
        cycle_score = _safe_score(0.65 * fft_energy + 0.35 * (1.0 - wavelet_energy))

        rows.append(
            {
                "ticker": ticker,
                "kalman_trend_score": _safe_score(kalman_score),
                "momentum_score": _safe_score(momentum_score),
                "momentum_20d_risk_adjusted": mom20,
                "momentum_60d_risk_adjusted": mom60,
                "hurst_exponent": hurst,
                "hurst_persistence_score": hurst_score,
                "shannon_entropy": entropy,
                "entropy_cleanliness_score": entropy_score,
                "raw_volatility": raw_volatility,
                "regime_trend_score": regime_score,
                "cycle_stability_score": cycle_score,
                "fft_low_freq_energy": fft_energy,
                "haar_wavelet_energy": wavelet_energy,
                "trend_persistence_reason": regime_reason,
                **kalman_slopes,
            }
        )

    result = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(index=tickers)
    if result.empty:
        return result

    result["volatility_stability_score"] = _pct_rank(result["raw_volatility"], invert=True)
    strong_momentum_bonus = (result["momentum_score"].astype(float) - 0.5).clip(lower=0.0) * 0.35
    result["volatility_stability_score"] = (result["volatility_stability_score"] + strong_momentum_bonus).clip(0.0, 1.0)
    result["correlation_diversification_score"] = _correlation_scores(returns_df, list(result.index), market_regime)

    weights = DEFAULT_COMPONENT_WEIGHTS.copy()
    if component_weights:
        weights.update(component_weights)
    weight_sum = sum(max(float(v), 0.0) for v in weights.values())
    if weight_sum <= EPS:
        weights = DEFAULT_COMPONENT_WEIGHTS.copy()
        weight_sum = sum(weights.values())

    score = pd.Series(0.0, index=result.index, dtype=float)
    for column, weight in weights.items():
        if column not in result.columns:
            result[column] = 0.5
        score = score + result[column].astype(float).fillna(0.5).clip(0.0, 1.0) * (float(weight) / weight_sum)
    result["trend_persistence_score"] = score.clip(0.0, 1.0)
    result["trend_persistence_confidence"] = (
        0.45 * (result["trend_persistence_score"] - 0.5).abs() * 2.0
        + 0.25 * result["entropy_cleanliness_score"].astype(float)
        + 0.20 * result["hurst_persistence_score"].astype(float)
        + 0.10 * result["cycle_stability_score"].astype(float)
    ).clip(0.0, 1.0)

    actions = []
    reasons = []
    for ticker, row in result.iterrows():
        score_value = _safe_float(row["trend_persistence_score"], 0.5)
        entropy_value = _safe_float(row["entropy_cleanliness_score"], 0.5)
        momentum_value = _safe_float(row["momentum_score"], 0.5)
        hurst_value = _safe_float(row["hurst_persistence_score"], 0.5)
        if score_value >= 0.72 and momentum_value >= 0.58:
            action = "strong_trend"
            reason = "kalman_momentum_persistence_aligned"
        elif score_value >= 0.58:
            action = "valid_trend"
            reason = "trend_persistence_positive"
        elif entropy_value < 0.35 or score_value < 0.42:
            action = "choppy/noise"
            reason = "high_entropy_or_low_persistence"
        else:
            action = "weak_trend"
            reason = "mixed_or_weak_components"
        if hurst_value < 0.42 and momentum_value < 0.55:
            action = "choppy/noise"
            reason = "mean_reversion_bias_without_momentum"
        actions.append(action)
        reasons.append(reason)
    result["trend_persistence_action"] = actions
    result["trend_persistence_reason"] = reasons
    return result.replace([np.inf, -np.inf], np.nan).fillna(
        {
            "kalman_trend_score": 0.5,
            "momentum_score": 0.5,
            "hurst_persistence_score": 0.5,
            "entropy_cleanliness_score": 0.5,
            "volatility_stability_score": 0.5,
            "regime_trend_score": 0.5,
            "correlation_diversification_score": 0.5,
            "cycle_stability_score": 0.5,
            "trend_persistence_score": 0.5,
            "trend_persistence_confidence": 0.5,
        }
    )


def apply_trend_persistence_to_expected_returns(
    adjusted_expected_returns: pd.Series,
    trend_persistence_df: pd.DataFrame,
) -> pd.Series:
    scores = (
        trend_persistence_df.get("trend_persistence_score", pd.Series(dtype=float))
        .reindex(adjusted_expected_returns.index)
        .astype(float)
        .fillna(0.5)
        .clip(0.0, 1.0)
    )
    adjusted = adjusted_expected_returns.astype(float) * (0.55 + 0.45 * scores)
    low_score_mask = scores < 0.25
    adjusted.loc[low_score_mask] = np.minimum(adjusted.loc[low_score_mask], 0.0)
    return adjusted.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_ema_trend_persistence_comparison(
    timing_df: pd.DataFrame,
    trend_persistence_df: pd.DataFrame,
) -> pd.DataFrame:
    tickers = list(trend_persistence_df.index)
    comparison = pd.DataFrame(index=tickers)
    comparison["ema_timing_score"] = (
        timing_df.get("ema_timing_score", pd.Series(dtype=float))
        .reindex(tickers)
        .astype(float)
        .fillna(0.5)
    )
    comparison["trend_persistence_score"] = (
        trend_persistence_df.get("trend_persistence_score", pd.Series(dtype=float))
        .reindex(tickers)
        .astype(float)
        .fillna(0.5)
    )
    comparison["difference"] = comparison["trend_persistence_score"] - comparison["ema_timing_score"]
    comparison["ema_action"] = (
        timing_df.get("timing_action", pd.Series(dtype=object))
        .reindex(tickers)
        .fillna("unknown")
    )
    comparison["trend_persistence_action"] = (
        trend_persistence_df.get("trend_persistence_action", pd.Series(dtype=object))
        .reindex(tickers)
        .fillna("unknown")
    )
    ema_score = comparison["ema_timing_score"].astype(float)
    trend_score = comparison["trend_persistence_score"].astype(float)
    ema_action = comparison["ema_action"].astype(str)
    trend_action = comparison["trend_persistence_action"].astype(str)
    comparison["agreement"] = np.where(
        comparison["difference"].abs() <= 0.15,
        "agreement",
        "disagreement",
    )
    comparison["would_change_status"] = comparison["agreement"] == "disagreement"
    disagreement_reasons = []
    for ticker in tickers:
        if ticker not in timing_df.index:
            reason = "missing EMA"
        elif ticker not in trend_persistence_df.index:
            reason = "missing trend persistence"
        else:
            e_score = _safe_float(ema_score.loc[ticker], 0.5)
            t_score = _safe_float(trend_score.loc[ticker], 0.5)
            e_action = str(ema_action.loc[ticker])
            t_action = str(trend_action.loc[ticker])
            ema_is_bearish = e_score < 0.40 or e_action in {"no_buy", "reduce", "exit_structure_break"}
            ema_is_bullish = e_score >= 0.58 or e_action in {"add_on_short_pullback", "add_on_long_support"}
            trend_is_bullish = t_score >= 0.58 or t_action in {"strong_trend", "valid_trend"}
            trend_is_weak = 0.42 <= t_score < 0.58 or t_action == "weak_trend"
            trend_is_choppy = t_score < 0.42 or t_action == "choppy/noise"

            if ema_is_bearish and trend_is_bullish:
                reason = "EMA bearish / trend bullish"
            elif ema_is_bullish and trend_is_weak:
                reason = "EMA bullish / trend weak"
            elif (not ema_is_bearish and not ema_is_bullish) and trend_is_bullish:
                reason = "EMA neutral / trend bullish"
            elif ema_is_bullish and trend_is_choppy:
                reason = "EMA bullish / trend choppy"
            elif trend_is_choppy:
                reason = "trend choppy/noise"
            else:
                reason = "score distance disagreement" if abs(t_score - e_score) > 0.15 else "aligned"
        disagreement_reasons.append(reason)
    comparison["disagreement_reason"] = disagreement_reasons
    return comparison
