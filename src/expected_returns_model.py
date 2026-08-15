from __future__ import annotations

import io
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd
import yfinance as yf

from quant_research_features import compute_asset_quant_features

TRADING_DAYS_PER_YEAR = 252


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def get_current_prices(
    tickers: list[str] | pd.Index | tuple[str, ...],
    fallback_prices: pd.Series | None = None,
    chunk_size: int = 50,
) -> pd.Series:
    """
    Get the most recent available prices for the requested tickers.
    Falls back to the last historical price if live retrieval fails.
    """
    ticker_list = [str(ticker).upper() for ticker in tickers]
    fallback_series = (
        pd.Series(fallback_prices, dtype=float).reindex(ticker_list)
        if fallback_prices is not None
        else pd.Series(index=ticker_list, dtype=float)
    )
    current_prices = fallback_series.copy()

    for start in range(0, len(ticker_list), max(1, int(chunk_size))):
        chunk = ticker_list[start : start + max(1, int(chunk_size))]
        try:
            with _temporary_disable_proxies(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                raw_data = yf.download(
                    chunk,
                    period="1d",
                    interval="1m",
                    progress=False,
                    auto_adjust=False,
                    group_by="ticker",
                    threads=False,
                    timeout=20,
                )
        except Exception:
            continue

        if raw_data.empty:
            continue

        for ticker in chunk:
            try:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    if ticker in raw_data.columns.get_level_values(0):
                        close_series = raw_data[ticker]["Close"].dropna()
                    else:
                        close_series = pd.Series(dtype=float)
                else:
                    close_series = raw_data["Close"].dropna()

                if not close_series.empty:
                    current_prices.loc[ticker] = float(close_series.iloc[-1])
            except Exception:
                continue

    missing_tickers = current_prices[current_prices.isna()].index.tolist()
    if missing_tickers:
        print(f"[WARNING] Missing live prices for {len(missing_tickers)} tickers. Using historical fallback prices.")

    return current_prices.fillna(fallback_series)


def compute_expected_returns(
    prices_df: pd.DataFrame,
    target_prices: dict[str, float] | pd.Series,
    use_live_prices: bool = True,
    risk_free_daily: float = 0.0,
    center_distribution: bool = False,
    dead_zone: float = 0.001,
    regime_score: float = 0.0,
    regime_type: str = "neutral",
    regime_signal_threshold: float = 0.2,
    use_raw_target_return: bool = False,
) -> tuple[pd.Series, dict[str, dict[str, float | str]]]:
    """
    Compute forward-looking expected daily returns from target prices and recent price dynamics.
    """
    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    target_series = pd.Series(target_prices, dtype=float)
    collapse_threshold = 1e-4
    fallback_prices = prices_df.ffill().iloc[-1]
    current_prices = get_current_prices(prices_df.columns, fallback_prices=fallback_prices) if use_live_prices else fallback_prices
    quant_features = compute_asset_quant_features(prices_df).reindex(prices_df.columns)
    expected_returns: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float | str]] = {}

    for ticker in prices_df.columns:
        price_series = prices_df[ticker].dropna()
        if price_series.empty:
            expected_returns[ticker] = 0.0
            diagnostics[ticker] = {
                "target_price": float("nan"),
                "current_price": float("nan"),
                "total_return": 0.0,
                "time_to_target": float(TRADING_DAYS_PER_YEAR),
                "expected_daily_return": 0.0,
                "raw_target_return_exact": 0.0,
                "raw_expected_daily_return_exact": 0.0,
                "raw_target_price_exact": float("nan"),
                "signal_strength_adjustment_value": 0.0,
                "final_expected_return_after_adjustments": 0.0,
                "momentum": 0.0,
                "volatility": 0.0,
                "mu": 0.0,
                "penalization_applied": False,
                "status": "missing_price_data",
            }
            continue

        historical_last_price = float(price_series.iloc[-1])
        current_price = float(current_prices.get(ticker, historical_last_price))
        target_price = float(target_series.get(ticker, current_price))

        if current_price <= 0 or not np.isfinite(current_price):
            expected_returns[ticker] = 0.0
            diagnostics[ticker] = {
                "target_price": target_price,
                "current_price": current_price,
                "total_return": 0.0,
                "time_to_target": float(TRADING_DAYS_PER_YEAR),
                "expected_daily_return": 0.0,
                "raw_target_return_exact": 0.0,
                "raw_expected_daily_return_exact": 0.0,
                "raw_target_price_exact": float("nan"),
                "signal_strength_adjustment_value": 0.0,
                "final_expected_return_after_adjustments": 0.0,
                "momentum": 0.0,
                "volatility": 0.0,
                "mu": 0.0,
                "penalization_applied": False,
                "status": "invalid_current_price",
            }
            continue

        status = "reachable"
        penalization_applied = False
        historical_gap = abs(current_price / historical_last_price - 1) if historical_last_price > 0 else 0.0
        if historical_gap > 0.10:
            print(f"[WARNING] Live price for {ticker} differs materially from last historical close ({historical_gap:.2%}).")
            status = "live_price_gap_warning"

        total_return = _clamp((target_price / current_price) - 1, -0.95, 5.0)
        mean_last_10 = float(price_series.tail(10).mean())
        momentum = (current_price / mean_last_10) - 1 if mean_last_10 > 0 else 0.0
        acceleration = abs(momentum)
        acceleration_norm = _clamp(acceleration / 0.01, 0.0, 1.0)

        log_returns = np.log(price_series / price_series.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        recent_log_returns = log_returns.tail(60)
        volatility = float(recent_log_returns.std()) if not recent_log_returns.empty else 0.0
        quant_row = quant_features.loc[ticker] if ticker in quant_features.index else pd.Series(dtype=float)
        garch_volatility = float(quant_row.get("garch_volatility", volatility))
        egarch_volatility = float(quant_row.get("egarch_volatility", volatility))
        if np.isfinite(garch_volatility) and garch_volatility > 0:
            volatility = max(volatility, garch_volatility)
        if np.isfinite(egarch_volatility) and egarch_volatility > 0:
            volatility = max(volatility, egarch_volatility)
        volatility_norm = _clamp(volatility / 0.04, 0.0, 1.0)

        base_min = 5
        base_max = 60
        n_base = base_max - acceleration_norm * (base_max - base_min)
        n_adjusted = n_base + volatility_norm * 20
        n_window = int(round(_clamp(n_adjusted, base_min, len(price_series))))

        mu_window = log_returns.tail(n_window)
        mu = float(mu_window.mean()) if not mu_window.empty else 0.0
        ema20 = price_series.ewm(span=20, adjust=False).mean()
        ema50 = price_series.ewm(span=50, adjust=False).mean()
        ema100 = price_series.ewm(span=100, adjust=False).mean()
        if len(price_series) >= 100:
            if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) > float(ema100.iloc[-1]):
                trend_alignment = 1.0
            elif float(ema20.iloc[-1]) < float(ema50.iloc[-1]) < float(ema100.iloc[-1]):
                trend_alignment = -1.0
            else:
                trend_alignment = 0.0
        else:
            trend_alignment = 0.0
        trend_slope = 0.0
        if len(ema20) > 5 and float(ema20.iloc[-1]) != 0:
            trend_slope = float((ema20.iloc[-1] - ema20.iloc[-6]) / ema20.iloc[-1])
        trend_strength = _clamp(0.5 + 0.5 * np.tanh(trend_slope * 50) + 0.25 * trend_alignment, 0.0, 1.0)

        dynamic_cap = _clamp(
            45 + volatility_norm * 140 + min(abs(total_return), 2.0) * 70,
            30,
            float(TRADING_DAYS_PER_YEAR),
        )

        if total_return > 0 and mu > 0:
            denominator = max(mu, 1e-8)
            calculated_time = float(np.log1p(total_return) / denominator)
            if calculated_time > dynamic_cap * 1.35:
                time_to_target = dynamic_cap
                status = "unreachable_target"
            else:
                time_to_target = min(calculated_time, dynamic_cap)
        else:
            time_to_target = dynamic_cap
            if status in {"reachable", "live_price_gap_warning"}:
                status = "bearish_or_low_probability"

        time_to_target = _clamp(time_to_target, 1.0, dynamic_cap)
        expected_daily_return = np.power(max(1 + total_return, 1e-8), 1 / time_to_target) - 1
        expected_daily_return = float(expected_daily_return)
        raw_expected_daily_return_exact = float(expected_daily_return)

        if total_return > 1.0:
            penalty = float(np.exp(-total_return))
            expected_daily_return *= penalty
            penalization_applied = True

        expected_daily_return *= 1 / (1 + volatility)
        expected_daily_return *= 1 + min(acceleration, 0.05)
        expected_daily_return = _clamp(float(expected_daily_return), -0.95, 1.0)

        fallback_applied = False
        historical_proxy = float(price_series.pct_change().dropna().tail(20).mean()) if len(price_series) > 1 else 0.0
        momentum_proxy = _clamp(acceleration * 0.5, -0.05, 0.05)
        fallback_expected_return = np.nanmean([historical_proxy, momentum_proxy, mu])
        fallback_expected_return = _clamp(float(fallback_expected_return), -0.05, 0.05)

        if status in {"bearish_or_low_probability", "unreachable_target"}:
            expected_daily_return = min(expected_daily_return, 0.0)
            expected_daily_return = _clamp(expected_daily_return, -0.05, 0.0)

        if not np.isfinite(expected_daily_return):
            expected_daily_return = 0.0
            status = "numerical_fallback"

        if (not np.isfinite(expected_daily_return)) or abs(expected_daily_return) < 1e-8:
            expected_daily_return = fallback_expected_return
            fallback_applied = True
            if status == "reachable" and expected_daily_return <= 0:
                status = "weak_positive_target"
            if status in {"bearish_or_low_probability", "unreachable_target"}:
                expected_daily_return = min(expected_daily_return, 0.0)

        expected_returns[ticker] = expected_daily_return
        diagnostics[ticker] = {
            "target_price": target_price,
            "current_price": current_price,
            "historical_last_price": historical_last_price,
            "total_return": total_return,
            "time_to_target": time_to_target,
            "raw_target_return_exact": float(total_return),
            "raw_expected_daily_return_exact": float(raw_expected_daily_return_exact),
            "raw_target_price_exact": float(target_price),
            "raw_target_expected_daily_return": expected_daily_return,
            "expected_daily_return": expected_daily_return,
            "momentum": momentum,
            "volatility": volatility,
            "mu": mu,
            "trend_alignment": trend_alignment,
            "trend_strength": trend_strength,
            "penalization_applied": penalization_applied,
            "fallback_applied": fallback_applied,
            "target_validity": True,
            "status": status,
        }

    expected_daily_returns = pd.Series(expected_returns, dtype=float).reindex(prices_df.columns).fillna(0.0)
    if float(expected_daily_returns.std()) < collapse_threshold:
        historical_fallback = prices_df.pct_change().tail(20).mean().reindex(prices_df.columns).fillna(0.0)
        momentum_fallback = pd.Series(
            {ticker: _clamp(float(diagnostics[ticker]["momentum"]) * 0.5, -0.05, 0.05) for ticker in prices_df.columns},
            dtype=float,
        )
        expected_daily_returns = (0.6 * historical_fallback + 0.4 * momentum_fallback).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        for ticker in prices_df.columns:
            diagnostics[ticker]["expected_daily_return"] = float(expected_daily_returns.loc[ticker])
            diagnostics[ticker]["fallback_applied"] = True
            if diagnostics[ticker]["status"] == "reachable" and expected_daily_returns.loc[ticker] <= 0:
                diagnostics[ticker]["status"] = "weak_positive_target"

    raw_signal_strength = {}
    for ticker in prices_df.columns:
        asset_volatility = float(diagnostics[ticker]["volatility"])
        asset_expected_return = float(diagnostics[ticker]["expected_daily_return"])
        epsilon = 1e-6
        raw_signal_strength[ticker] = (asset_expected_return + epsilon) / (asset_volatility + epsilon)

    signal_series = pd.Series(raw_signal_strength, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal_min = float(signal_series.min())
    signal_max = float(signal_series.max())
    if signal_max - signal_min > 1e-12:
        normalized_signal = (signal_series - signal_min) / (signal_max - signal_min)
    else:
        normalized_signal = pd.Series(0.0, index=signal_series.index)

    for ticker in prices_df.columns:
        diagnostics[ticker]["signal_strength"] = float(normalized_signal.loc[ticker])

    for ticker in prices_df.columns:
        adjusted_expected_return = float(diagnostics[ticker]["expected_daily_return"])
        signal_strength = float(diagnostics[ticker]["signal_strength"])
        target_validity = diagnostics[ticker].get("target_validity", True)
        trend_alignment = float(diagnostics[ticker].get("trend_alignment", 0.0))
        trend_strength = float(diagnostics[ticker].get("trend_strength", 0.5))
        momentum = float(diagnostics[ticker].get("momentum", 0.0))
        quant_row = quant_features.loc[ticker] if ticker in quant_features.index else pd.Series(dtype=float)

        if target_validity in {False, "unrealistic", "invalid"}:
            adjusted_expected_return *= 0.3
            diagnostics[ticker]["penalization_applied"] = True
        if trend_alignment < 0:
            adjusted_expected_return *= 0.5
            diagnostics[ticker]["penalization_applied"] = True
        if momentum < 0:
            adjusted_expected_return *= 0.7
            diagnostics[ticker]["penalization_applied"] = True
        if trend_strength < 0.4:
            adjusted_expected_return *= 0.75
            diagnostics[ticker]["penalization_applied"] = True

        entropy = float(quant_row.get("entropy", 0.5))
        hawkes_downside_intensity = float(quant_row.get("hawkes_downside_intensity", 0.0))
        hurst_value = float(quant_row.get("hurst_exponent", 0.5))
        ou_half_life_value = float(quant_row.get("ou_half_life", TRADING_DAYS_PER_YEAR))
        kalman_residual_vol = float(quant_row.get("kalman_residual_vol", 0.0))

        confidence_penalty = 0.0
        confidence_penalty += 0.012 * _clamp(entropy, 0.0, 1.0)
        confidence_penalty += 0.010 * _clamp(hawkes_downside_intensity, 0.0, 1.0)
        confidence_penalty += 0.004 * _clamp(abs(hurst_value - 0.5) / 0.5, 0.0, 1.0)
        confidence_penalty += 0.003 * _clamp(ou_half_life_value / TRADING_DAYS_PER_YEAR, 0.0, 1.0)
        confidence_penalty += 0.001 * _clamp(kalman_residual_vol / 0.05, 0.0, 1.0)
        quant_multiplier = float(np.clip(1.0 - confidence_penalty, 0.97, 1.0))

        if adjusted_expected_return > 0:
            adjusted_expected_return *= quant_multiplier

        adjusted_expected_return -= risk_free_daily
        adjusted_expected_return -= 0.001
        diagnostics[ticker]["pre_signal_adjustment_expected_daily_return"] = float(adjusted_expected_return)
        diagnostics[ticker]["raw_target_return_mode"] = bool(use_raw_target_return)
        diagnostics[ticker]["signal_strength_adjustment_disabled"] = bool(use_raw_target_return)
        diagnostics[ticker]["regime_adjustment_applied"] = False
        baseline_adjusted_expected_return = float(adjusted_expected_return)
        baseline_adjusted_expected_return *= signal_strength**1.5
        baseline_adjusted_expected_return *= 0.3 + 0.7 * signal_strength
        if signal_strength < 0.3:
            baseline_adjusted_expected_return = min(baseline_adjusted_expected_return, 0.0)
        if diagnostics[ticker].get("status") == "bearish_or_low_probability":
            baseline_adjusted_expected_return = min(baseline_adjusted_expected_return, 0.0)

        if signal_strength > regime_signal_threshold:
            diagnostics[ticker]["regime_adjustment_applied"] = True
            if regime_type in {"high_volatility", "risk_on"}:
                baseline_adjusted_expected_return *= 1.2
            elif regime_type in {"low_volatility"}:
                baseline_adjusted_expected_return *= 1.1
            elif regime_type in {"risk_off"}:
                baseline_adjusted_expected_return *= 0.6
            elif regime_type in {"neutral"}:
                baseline_adjusted_expected_return *= 0.85
            else:
                baseline_adjusted_expected_return *= 0.7

        diagnostics[ticker]["baseline_adjusted_expected_daily_return"] = float(baseline_adjusted_expected_return)
        diagnostics[ticker]["signal_strength_adjustment_value"] = float(
            baseline_adjusted_expected_return - diagnostics[ticker].get("pre_signal_adjustment_expected_daily_return", 0.0)
        )
        if use_raw_target_return:
            adjusted_expected_return = float(diagnostics[ticker]["raw_target_expected_daily_return"])
            diagnostics[ticker]["regime_adjustment_applied"] = False
        else:
            adjusted_expected_return = baseline_adjusted_expected_return

        if abs(adjusted_expected_return) < dead_zone:
            adjusted_expected_return = float(np.random.uniform(-0.0005, 0.0005))
        diagnostics[ticker]["expected_daily_return"] = adjusted_expected_return
        diagnostics[ticker]["final_expected_return_after_adjustments"] = float(adjusted_expected_return)
        expected_daily_returns.loc[ticker] = adjusted_expected_return

        for col in quant_features.columns:
            diagnostics[ticker][col] = (
                float(quant_features.loc[ticker, col])
                if ticker in quant_features.index and pd.notna(quant_features.loc[ticker, col])
                else 0.0
            )
        diagnostics[ticker]["quant_multiplier"] = float(np.clip(quant_multiplier, 0.97, 1.03))

    mean_ret = float(expected_daily_returns.mean()) if len(expected_daily_returns) else 0.0
    expected_daily_returns = expected_daily_returns - mean_ret
    for ticker in prices_df.columns:
        centered_value = float(expected_daily_returns.loc[ticker])
        if abs(centered_value) < dead_zone:
            centered_value = float(np.random.uniform(-0.0005, 0.0005))
            expected_daily_returns.loc[ticker] = centered_value
        diagnostics[ticker]["expected_daily_return"] = centered_value
        diagnostics[ticker]["final_expected_return_after_adjustments"] = float(centered_value)

    if center_distribution:
        expected_returns_mean = float(expected_daily_returns.mean())
        expected_daily_returns = expected_daily_returns - expected_returns_mean
        for ticker in prices_df.columns:
            centered_return = float(expected_daily_returns.loc[ticker])
            if abs(centered_return) < dead_zone:
                centered_return = float(np.random.uniform(-0.0005, 0.0005))
                expected_daily_returns.loc[ticker] = centered_return
            diagnostics[ticker]["expected_daily_return"] = centered_return
            diagnostics[ticker]["final_expected_return_after_adjustments"] = float(centered_return)

    if float(expected_daily_returns.std()) < collapse_threshold:
        tiny_noise = np.linspace(-1e-5, 1e-5, num=len(expected_daily_returns))
        expected_daily_returns = (expected_daily_returns + tiny_noise).astype(float)
        for i, ticker in enumerate(expected_daily_returns.index):
            val = float(expected_daily_returns.iloc[i])
            if diagnostics[ticker].get("status") == "bearish_or_low_probability":
                val = min(val, 0.0)
            if abs(val) < dead_zone:
                val = float(np.random.uniform(-0.0005, 0.0005))
            expected_daily_returns.loc[ticker] = val
            diagnostics[ticker]["expected_daily_return"] = val
            diagnostics[ticker]["final_expected_return_after_adjustments"] = float(val)
        print("[WARNING] Model collapse: expected returns too similar")

    expected_daily_returns = expected_daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return expected_daily_returns, diagnostics
