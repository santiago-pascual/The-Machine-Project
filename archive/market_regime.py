from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _safe_last_close(data: pd.DataFrame) -> float | None:
    if data.empty or "Close" not in data.columns:
        return None
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.empty or close.shape[1] == 0:
            return None
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def _fetch_vix() -> float | None:
    try:
        vix_data = yf.download("^VIX", period="2mo", interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None
    return _safe_last_close(vix_data)


def _fetch_spy_trend() -> tuple[float, float, float] | None:
    try:
        spy_data = yf.download("SPY", period="8mo", interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None
    if spy_data.empty or "Close" not in spy_data.columns:
        return None
    close = spy_data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.empty or close.shape[1] == 0:
            return None
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 110:
        return None
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    slope = float((ema20.iloc[-1] - ema20.iloc[-6]) / max(abs(ema20.iloc[-1]), 1e-8))
    return float(ema20.iloc[-1]), float(ema100.iloc[-1]), slope


def _fetch_spy_data(period: str = "12mo") -> pd.Series | None:
    try:
        spy_data = yf.download("SPY", period=period, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None
    if spy_data.empty or "Close" not in spy_data.columns:
        return None
    close = spy_data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.empty or close.shape[1] == 0:
            return None
        close = close.iloc[:, 0]
    close = close.dropna()
    return close if not close.empty else None


def _fetch_fear_greed(default_value: float = 50.0) -> float:
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        payload = response.json()
        fg_now = payload.get("fear_and_greed", {}).get("score")
        if fg_now is None:
            return default_value
        return float(fg_now)
    except Exception:
        return default_value


def _fear_greed_proxy(prices_df: pd.DataFrame) -> float:
    returns = prices_df.pct_change().dropna(how="all")
    if returns.empty:
        return 50.0
    momentum_20 = prices_df.ffill().iloc[-1] / prices_df.ffill().shift(20).iloc[-1] - 1
    momentum_20 = momentum_20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    breadth = float((momentum_20 > 0).mean()) if len(momentum_20) else 0.5
    avg_momentum = float(momentum_20.mean()) if len(momentum_20) else 0.0
    score = 50.0 + 35.0 * (breadth - 0.5) + 15.0 * np.tanh(avg_momentum * 10.0)
    return _clamp(float(score), 0.0, 100.0)


def compute_market_regime(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    """
    Compute a market regime score in [-1, 1] and regime type.
    """
    if prices_df.empty:
        return {
            "regime_score": 0.0,
            "regime_type": "neutral",
            "regime_confidence": 0.7,
            "vix": float("nan"),
            "fear_greed": 50.0,
            "realized_vol": float("nan"),
            "market_volatility": float("nan"),
            "trend_flag": 0.0,
            "trend_slope": 0.0,
            "market_drawdown": 0.0,
        }

    if returns_df is None or returns_df.empty:
        returns_df = prices_df.pct_change().dropna(how="all")

    realized_vol = float(returns_df.std().mean()) if not returns_df.empty else 0.0
    realized_vol = float(np.nan_to_num(realized_vol, nan=0.0, posinf=0.0, neginf=0.0))

    vix_value = _fetch_vix()
    fear_greed_raw = _fetch_fear_greed(default_value=float("nan"))
    fear_greed = float(fear_greed_raw) if np.isfinite(fear_greed_raw) else _fear_greed_proxy(prices_df)
    spy_trend = _fetch_spy_trend()
    spy_close = _fetch_spy_data(period="12mo")
    if spy_close is not None and len(spy_close) > 20:
        spy_returns = np.log(spy_close / spy_close.shift(1)).dropna()
        market_volatility = float(spy_returns.tail(60).std()) if not spy_returns.empty else realized_vol
        rolling_peak = spy_close.cummax()
        drawdown = (spy_close / rolling_peak - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        market_drawdown = float(drawdown.iloc[-1])
    else:
        market_volatility = realized_vol
        market_drawdown = 0.0

    score = 0.0

    if vix_value is None:
        vix_component = 0.0
        regime_type = "neutral"
    elif vix_value >= 30:
        vix_component = 0.25
        regime_type = "high_volatility"
    elif vix_value <= 15:
        vix_component = 0.15
        regime_type = "low_volatility"
    else:
        vix_component = -0.10
        regime_type = "neutral"
    score += vix_component

    if fear_greed <= 25:
        fg_component = 0.25
    elif fear_greed >= 75:
        fg_component = -0.25
    else:
        fg_component = 0.0
    score += fg_component

    vol_component = _clamp((0.02 - market_volatility) * 10.0, -0.20, 0.20)
    score += vol_component
    score += _clamp((-market_drawdown) * 0.5, -0.1, 0.2)

    trend_flag = 0.0
    trend_slope = 0.0
    if spy_trend is not None:
        ema20, ema100, trend_slope = spy_trend
        if ema20 > ema100:
            trend_flag = 1.0
            score += 0.20
        elif ema20 < ema100:
            trend_flag = -1.0
            score -= 0.20
        score += _clamp(np.tanh(trend_slope * 30.0) * 0.15, -0.15, 0.15)

    regime_score = _clamp(score, -1.0, 1.0)
    if regime_type == "high_volatility":
        regime_confidence = 1.2
    elif regime_type == "low_volatility":
        regime_confidence = 1.0
    else:
        regime_confidence = 0.7

    return {
        "regime_score": regime_score,
        "regime_type": regime_type,
        "regime_confidence": float(regime_confidence),
        "vix": float(vix_value) if vix_value is not None else float("nan"),
        "fear_greed": float(fear_greed),
        "realized_vol": float(realized_vol),
        "market_volatility": float(market_volatility),
        "trend_flag": float(trend_flag),
        "trend_slope": float(trend_slope),
        "market_drawdown": float(market_drawdown),
    }
