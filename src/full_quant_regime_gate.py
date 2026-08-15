from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _series_mean(values: object, default: float = np.nan) -> float:
    if values is None:
        return default
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return default
    return float(series.mean())


def evaluate_full_quant_regime_gate(
    *,
    regime: str | None = None,
    market_mode: str | None = None,
    spy_macro_regime: str | None = None,
    regime_confidence: float | None = None,
    regime_score: float | None = None,
    volatility_condition: float | None = None,
    vix_z: float | None = None,
    breadth: float | None = None,
    spy_momentum_20d: float | None = None,
    average_entropy: float | None = None,
    average_trend_persistence_score: float | None = None,
) -> dict[str, object]:
    """
    Research gate for full_quant_research.

    Conservative rule based on prior attribution:
    - avoid 2022-like weak/choppy/high-vol regimes;
    - allow full quant only when regime data is clear enough and risk conditions are supportive.
    """
    regime = str(regime or "missing").lower()
    market_mode = str(market_mode or "unknown").lower()
    spy_macro_regime = str(spy_macro_regime or "unknown").lower()
    confidence = _safe_float(regime_confidence, np.nan)
    score = _safe_float(regime_score, np.nan)
    vol_ratio = _safe_float(volatility_condition, np.nan)
    vix_z_value = _safe_float(vix_z, np.nan)
    breadth_value = _safe_float(breadth, np.nan)
    spy_mom_20 = _safe_float(spy_momentum_20d, np.nan)
    entropy = _safe_float(average_entropy, np.nan)
    trend_score = _safe_float(average_trend_persistence_score, np.nan)

    missing_core = regime == "missing" or not np.isfinite(confidence) or not np.isfinite(score)
    high_noise = np.isfinite(entropy) and entropy > 0.78
    weak_breadth = np.isfinite(breadth_value) and breadth_value < 0.45
    bearish_spy = spy_macro_regime == "bearish" or (np.isfinite(spy_mom_20) and spy_mom_20 < -0.03)
    high_vol_stress = (np.isfinite(vix_z_value) and vix_z_value > 1.25) or (np.isfinite(vol_ratio) and vol_ratio > 1.35 and score < 0.15)
    weak_trend = np.isfinite(trend_score) and trend_score < 0.45

    if missing_core:
        decision = "fallback_baseline"
        reason = "missing_or_unclear_regime_data"
    elif regime == "risk_off":
        decision = "fallback_baseline"
        reason = "risk_off_regime"
    elif high_vol_stress:
        decision = "fallback_baseline"
        reason = "2022_like_high_vol_stress"
    elif bearish_spy:
        decision = "fallback_baseline"
        reason = "bearish_spy_or_negative_market_momentum"
    elif high_noise:
        decision = "fallback_baseline"
        reason = "entropy_noise_too_high"
    elif weak_breadth:
        decision = "fallback_baseline"
        reason = "weak_market_breadth"
    elif weak_trend:
        decision = "fallback_baseline"
        reason = "weak_trend_persistence"
    elif regime == "risk_on" and confidence >= 0.30:
        decision = "allow_full_quant"
        reason = "risk_on_with_sufficient_confidence"
    elif regime == "neutral" and confidence >= 0.45 and np.isfinite(spy_mom_20) and spy_mom_20 > 0 and not weak_breadth:
        decision = "allow_full_quant"
        reason = "constructive_neutral_regime"
    elif market_mode == "aggressive" and confidence >= 0.40 and not weak_breadth:
        decision = "allow_full_quant"
        reason = "aggressive_market_mode"
    else:
        decision = "fallback_baseline"
        reason = "insufficient_full_quant_edge"

    return {
        "gate_decision": decision,
        "allow_full_quant": decision == "allow_full_quant",
        "reason": reason,
        "regime": regime,
        "market_mode": market_mode,
        "spy_macro_regime": spy_macro_regime,
        "regime_confidence": confidence,
        "regime_score": score,
        "volatility_condition": vol_ratio,
        "vix_z": vix_z_value,
        "breadth": breadth_value,
        "spy_momentum_20d": spy_mom_20,
        "average_entropy": entropy,
        "average_trend_persistence_score": trend_score,
    }


def average_entropy_from_diagnostics(diagnostics_df: pd.DataFrame | None) -> float:
    if diagnostics_df is None or diagnostics_df.empty or "shannon_entropy" not in diagnostics_df.columns:
        return np.nan
    return _series_mean(diagnostics_df["shannon_entropy"], np.nan)


def average_trend_score(trend_persistence_df: pd.DataFrame | None) -> float:
    if trend_persistence_df is None or trend_persistence_df.empty or "trend_persistence_score" not in trend_persistence_df.columns:
        return np.nan
    return _series_mean(trend_persistence_df["trend_persistence_score"], np.nan)
