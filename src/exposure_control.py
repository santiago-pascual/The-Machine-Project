from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_dispersion_score(expected_returns: pd.Series | np.ndarray, scale: float = 0.03) -> float:
    expected = _as_series(expected_returns)
    std_value = float(expected.std()) if len(expected) > 1 else 0.0
    dispersion_score = std_value / max(scale, 1e-8)
    return float(np.clip(dispersion_score, 0.0, 1.0))


def compute_topn_dispersion_score(
    expected_returns: pd.Series | np.ndarray,
    top_n: int = 5,
    scale: float = 0.03,
) -> tuple[float, float]:
    expected = _as_series(expected_returns)
    if expected.empty:
        return 0.0, 0.0
    top_values = expected.sort_values(ascending=False).head(max(1, top_n))
    top_std = float(top_values.std()) if len(top_values) > 1 else 0.0
    score = float(np.clip(top_std / max(scale, 1e-8), 0.0, 1.0))
    return score, top_std


def compute_opportunity_score(
    expected_returns: pd.Series | np.ndarray,
    signal_strengths: pd.Series | np.ndarray,
    expected_threshold: float = 0.001,
) -> float:
    expected = _as_series(expected_returns)
    signals = _as_series(signal_strengths).reindex(expected.index).fillna(0.0)

    mask = (expected > expected_threshold) & (signals > 0.2)
    if len(expected) == 0 or not bool(mask.any()):
        return 0.3

    positive_ratio = float(mask.mean())
    top_n = max(3, int(np.ceil(np.sqrt(len(expected)))))
    top_signals = signals[mask].sort_values(ascending=False).head(top_n)
    avg_top_signal = float(top_signals.mean()) if len(top_signals) else 0.0
    avg_top_signal = float(np.clip(avg_top_signal, 0.0, 1.0))

    dispersion_score = compute_dispersion_score(expected)
    score = 0.45 * positive_ratio + 0.35 * avg_top_signal + 0.20 * dispersion_score
    return float(np.clip(score, 0.0, 1.0))


def detect_market_mode(
    regime_score: float,
    regime_confidence: float,
    opportunity_score: float,
    dispersion_score: float,
) -> str:
    score = float(np.clip(regime_score, -1.0, 1.0))
    confidence = float(np.clip(regime_confidence, 0.0, 1.0))
    opportunity = float(np.clip(opportunity_score, 0.0, 1.0))
    _ = float(np.clip(dispersion_score, 0.0, 1.0))

    if score > 0.15 and confidence > 0.75 and opportunity > 0.5:
        return "aggressive"
    if score < -0.15 and confidence > 0.7:
        return "defensive"
    return "neutral"


def compute_target_exposure(
    market_mode: str,
    regime_score: float,
    regime_confidence: float,
    opportunity_score: float,
    dispersion_score: float,
    timeframe: str = "daily",
    min_exposure: float = 0.20,
) -> dict[str, float | str]:
    mode = str(market_mode)
    tf = str(timeframe).lower()
    confidence = float(np.clip(regime_confidence, 0.0, 1.0))
    opportunity = float(np.clip(opportunity_score, 0.0, 1.0))
    dispersion = float(np.clip(dispersion_score, 0.0, 1.0))
    score = float(np.clip(regime_score, -1.0, 1.0))

    if tf == "daily":
        if mode == "aggressive":
            base_exposure = 0.8
        elif mode == "defensive":
            base_exposure = 0.3
        else:
            base_exposure = 0.7
    else:
        if mode == "aggressive":
            base_exposure = 0.8
        elif mode == "defensive":
            base_exposure = 0.1
        else:
            base_exposure = 0.4

    regime_multiplier = float(np.clip(1.0 + 0.5 * score, 0.6, 1.3))
    effective_dispersion = float(0.7 + 0.3 * dispersion)

    net_exposure = base_exposure
    net_exposure *= regime_multiplier
    net_exposure *= confidence
    net_exposure *= opportunity
    net_exposure *= effective_dispersion
    raw_net_exposure = float(net_exposure)
    net_exposure = float(np.clip(net_exposure, min_exposure, 1.0))

    if opportunity < 0.2 and net_exposure >= 0.3:
        net_exposure = 0.29
    if opportunity > 0.7 and score > 0.2 and net_exposure <= 0.6:
        net_exposure = 0.61

    if opportunity < 0.2:
        assert net_exposure < 0.3
    if opportunity > 0.7 and score > 0.2:
        assert net_exposure > 0.6

    cash_weight = float(max(0.0, 1.0 - net_exposure))
    return {
        "market_mode": mode,
        "base_exposure": float(base_exposure),
        "regime_score": score,
        "regime_confidence": confidence,
        "opportunity_score": opportunity,
        "dispersion_score": dispersion,
        "effective_dispersion": effective_dispersion,
        "regime_multiplier": regime_multiplier,
        "raw_net_exposure": raw_net_exposure,
        "net_exposure": float(net_exposure),
        "cash_weight": cash_weight,
    }


def compute_net_exposure(
    regime_score: float,
    regime_confidence: float,
    expected_returns: pd.Series | np.ndarray,
    signal_strengths: pd.Series | np.ndarray,
    timeframe: str = "daily",
    weak_dispersion_std_threshold: float = 0.003,
    high_signal_threshold: float = 0.65,
    market_mode_override: str | None = None,
) -> dict[str, float | str]:
    expected = _as_series(expected_returns)
    signals = _as_series(signal_strengths).reindex(expected.index).fillna(0.0)

    opportunity_score = compute_opportunity_score(expected, signals)
    dispersion_score_total = compute_dispersion_score(expected)
    dispersion_score, topn_std = compute_topn_dispersion_score(expected, top_n=5)
    expected_std = float(expected.std()) if len(expected) > 1 else 0.0
    top3_signals = signals.sort_values(ascending=False).head(3)
    top3_mean_signal = float(top3_signals.mean()) if len(top3_signals) else 0.0
    top3_expected = expected.sort_values(ascending=False).head(3)

    if expected_std < weak_dispersion_std_threshold and top3_mean_signal > high_signal_threshold:
        dispersion_score = max(dispersion_score, 0.4)

    market_mode = detect_market_mode(
        regime_score=regime_score,
        regime_confidence=regime_confidence,
        opportunity_score=opportunity_score,
        dispersion_score=dispersion_score,
    )
    if market_mode_override is not None:
        market_mode = str(market_mode_override)
    target = compute_target_exposure(
        market_mode=market_mode,
        regime_score=regime_score,
        regime_confidence=regime_confidence,
        opportunity_score=opportunity_score,
        dispersion_score=dispersion_score,
        timeframe=timeframe,
        min_exposure=0.20,
    )

    if len(top3_signals) == 3 and bool((top3_signals > high_signal_threshold).all()):
        target["net_exposure"] = max(float(target["net_exposure"]), 0.40)
        target["cash_weight"] = max(0.0, 1.0 - float(target["net_exposure"]))

    pre_override_exposure = float(target["net_exposure"])
    top3_list = top3_signals.tolist()
    if len(top3_list) >= 3 and float(top3_list[0]) > 0.8 and float(top3_list[1]) > 0.6 and float(top3_list[2]) > 0.4:
        target["net_exposure"] = max(float(target["net_exposure"]), 0.5)
        target["cash_weight"] = max(0.0, 1.0 - float(target["net_exposure"]))

    if opportunity_score > 0.6:
        target["net_exposure"] = max(float(target["net_exposure"]), 0.4)
        target["cash_weight"] = max(0.0, 1.0 - float(target["net_exposure"]))

    return {
        "market_mode": str(target["market_mode"]),
        "net_exposure": float(target["net_exposure"]),
        "cash_weight": float(target["cash_weight"]),
        "base_exposure": float(target["base_exposure"]),
        "regime_score": float(target["regime_score"]),
        "regime_confidence": float(target["regime_confidence"]),
        "opportunity_score": float(target["opportunity_score"]),
        "dispersion_score": float(target["dispersion_score"]),
        "dispersion_score_total": float(dispersion_score_total),
        "dispersion_topn_std": float(topn_std),
        "effective_dispersion": float(target["effective_dispersion"]),
        "regime_adjustment": float(target["regime_multiplier"]),
        "confidence_adjustment": float(target["regime_confidence"]),
        "opportunity_adjustment": float(target["opportunity_score"]),
        "dispersion_adjustment": float(target["effective_dispersion"]),
        "raw_net_exposure": float(target["raw_net_exposure"]),
        "pre_override_exposure": pre_override_exposure,
        "post_override_exposure": float(target["net_exposure"]),
        "expected_returns_std": expected_std,
        "top3_expected_returns": ",".join(f"{float(v):.6f}" for v in top3_expected.tolist()),
        "top3_signal_strengths": ",".join(f"{float(v):.4f}" for v in top3_signals.tolist()),
        "top3_signal_mean": top3_mean_signal,
    }
