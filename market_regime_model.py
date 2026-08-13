from __future__ import annotations

import io
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd
import yfinance as yf

from quant_research_features import gaussian_hmm_two_state, hurst_exponent


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


def _safe_tanh(value: float) -> float:
    return float(np.tanh(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)))


def _download_close_series(ticker: str, period: str = "12mo") -> pd.Series:
    try:
        with _temporary_disable_proxies():
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                data = yf.download(
                    ticker,
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
    return close.dropna().astype(float)


def compute_market_regime_model(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    spy_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    strict_market_data: bool = False,
) -> dict[str, float | str]:
    if prices_df.empty or returns_df.empty:
        return {
            "risk_score": 0.0,
            "regime": "neutral",
            "regime_confidence": 0.3,
            "vix": float("nan"),
            "vix_z": 0.0,
            "vix_trend": 1.0,
            "spy_momentum_20d": 0.0,
            "spy_momentum_60d": 0.0,
            "realized_vol": 0.0,
            "vol_regime_ratio": 0.0,
            "breadth": 0.5,
        }

    if vix_close is None:
        vix_close = _download_close_series("^VIX", period="6mo")
    if spy_close is None:
        spy_close = _download_close_series("SPY", period="12mo")

    if strict_market_data and (vix_close.empty or spy_close.empty):
        raise ValueError("SPY o ^VIX no disponibles para market regime (strict mode).")

    if vix_close.empty:
        vix_current = 20.0
        vix_20d_mean = 20.0
        vix_60d_mean = 20.0
        vix_60d_std = 1.0
    else:
        vix_current = float(vix_close.iloc[-1])
        vix_20d_mean = float(vix_close.tail(20).mean())
        vix_60d_mean = float(vix_close.tail(60).mean())
        vix_60d_std = float(vix_close.tail(60).std()) if len(vix_close) > 1 else 1.0

    vix_trend = vix_current / max(vix_20d_mean, 1e-8)
    vix_z = (vix_current - vix_60d_mean) / max(vix_60d_std, 1e-8)

    if spy_close.empty or len(spy_close) < 61:
        spy_momentum_20d = 0.0
        spy_momentum_60d = 0.0
        spy_returns = pd.Series(dtype=float)
        hmm_high_vol_probability = 0.5
        spy_hurst = 0.5
    else:
        spy_momentum_20d = float(spy_close.iloc[-1] / spy_close.iloc[-21] - 1) if len(spy_close) > 21 else 0.0
        spy_momentum_60d = float(spy_close.iloc[-1] / spy_close.iloc[-61] - 1) if len(spy_close) > 61 else 0.0
        spy_returns = np.log(spy_close / spy_close.shift(1)).dropna()
        hmm_state = gaussian_hmm_two_state(spy_returns)
        hmm_high_vol_probability = float(hmm_state["high_vol_probability"])
        spy_hurst = float(hurst_exponent(spy_close))

    realized_vol = float(spy_returns.tail(20).std()) if len(spy_returns) > 1 else float(returns_df.std().mean())
    vix_as_decimal = max(vix_current / 100.0, 1e-6)
    vol_regime_ratio = float(realized_vol / vix_as_decimal)

    recent_returns = returns_df.tail(20)
    breadth = float((recent_returns.mean() > 0).mean()) if not recent_returns.empty else 0.5

    risk_score_raw = 0.0
    risk_score_raw += -0.3 * _safe_tanh(vix_z)
    risk_score_raw += -0.2 * _safe_tanh(vix_trend - 1.0)
    risk_score_raw += 0.3 * _safe_tanh(spy_momentum_20d * 10.0)
    risk_score_raw += 0.2 * _safe_tanh(spy_momentum_60d * 10.0)
    risk_score_raw += 0.2 * _safe_tanh(vol_regime_ratio - 1.0)
    risk_score_raw += 0.2 * _safe_tanh((breadth - 0.5) * 4.0)
    risk_score_raw += 0.15 * _safe_tanh((spy_hurst - 0.5) * 6.0) * np.sign(spy_momentum_20d)
    risk_score = _safe_tanh(risk_score_raw)

    if risk_score > 0.3:
        regime = "risk_on"
    elif risk_score < -0.3:
        regime = "risk_off"
    else:
        regime = "neutral"

    regime_confidence = float(np.clip(abs(risk_score), 0.3, 1.0))

    return {
        "risk_score": float(risk_score),
        "regime": regime,
        "regime_confidence": regime_confidence,
        "vix": float(vix_current),
        "vix_z": float(vix_z),
        "vix_trend": float(vix_trend),
        "spy_momentum_20d": float(spy_momentum_20d),
        "spy_momentum_60d": float(spy_momentum_60d),
        "realized_vol": float(realized_vol),
        "vol_regime_ratio": float(vol_regime_ratio),
        "breadth": float(breadth),
        "hmm_high_vol_probability": float(hmm_high_vol_probability),
        "hurst_exponent": float(spy_hurst),
    }
