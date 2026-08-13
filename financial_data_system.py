from __future__ import annotations

import os
import builtins
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import io
import sys
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from action_signals import generate_action_signals_report
from advanced_target_model import generate_targets_advanced
from black_litterman_model import print_black_litterman_diagnostics
from calibrated_forecast_research import (
    CalibratedForecastConfig,
    apply_walk_forward_calibrated_forecasts,
    print_calibrated_forecast_research_report,
)
from covariance_estimation import calculate_covariance, covariance_method_metadata, print_covariance_diagnostics
from ema_timing_model import apply_timing_to_expected_returns, compute_asset_timing, compute_spy_ema_regime
from expected_returns_model import compute_expected_returns
from exposure_control import compute_net_exposure
from factor_attribution import print_factor_attribution
from forecast_calibration import save_and_evaluate_forecasts
from full_quant_regime_gate import average_entropy_from_diagnostics, average_trend_score, evaluate_full_quant_regime_gate
from heuristic_calibration_diagnostics import print_heuristic_calibration_diagnostics
from heuristic_audit import print_heuristic_audit_report
from information_coefficient import print_information_coefficient_report
from market_regime_model import compute_market_regime_model
from portfolio_optimizer import PortfolioOptimizer
from quant_target_model import generate_quant_targets
from risk_metrics import print_institutional_risk_metrics
from trend_persistence_engine import (
    apply_trend_persistence_to_expected_returns,
    build_ema_trend_persistence_comparison,
    compute_trend_persistence,
)


lookback_days = 252


def _daily_download_end_date() -> datetime:
    """Return yfinance end date for daily research downloads.

    Yahoo treats end as exclusive. The default remains today's date for normal
    production behavior; daily_research_run can set YFINANCE_END_DATE_OFFSET_DAYS=1
    so same-day completed closes are not missed by the paper/research pipeline.
    """
    offset = int(os.getenv("YFINANCE_END_DATE_OFFSET_DAYS", "0") or "0")
    return datetime.today() + timedelta(days=offset)

DEFAULT_RISK_FREE_RATE = 0.037
TRADING_DAYS_PER_YEAR = 252


REPORT_SECTION_HEADERS = {
    "TREND_PERSISTENCE": "TREND PERSISTENCE ENGINE",
    "TREND_PERSISTENCE_BREAKDOWN": "TREND PERSISTENCE COMPONENT BREAKDOWN",
    "EMA_VS_TREND_PERSISTENCE": "EMA VS TREND PERSISTENCE COMPARISON",
    "EMA_VS_TREND_PERSISTENCE_ANALYSIS": "EMA VS TREND PERSISTENCE DISAGREEMENT ANALYSIS",
    "BLACK_LITTERMAN": "BLACK-LITTERMAN",
    "INSTITUTIONAL_RISK": "INSTITUTIONAL RISK METRICS",
    "ACTION_SIGNALS": "ACTION SIGNALS",
    "PAPER_META_FILTER": "PAPER META FILTER",
    "PAPER_TRADING": "PAPER TRADING SIMULATION",
    "FORECAST_CALIBRATION": "FORECAST",
    "FINAL_ALLOCATION": "FINAL ALLOCATION",
    "WALK_FORWARD": "WALK-FORWARD",
    "TRIPLE_BARRIER": "TRIPLE BARRIER",
    "MODEL_MODE": "MODEL MODE REPORT",
    "MODEL_MODE_COMPARISON": "BASELINE VS FULL QUANT RESEARCH COMPARISON",
    "CALIBRATED_FORECAST_RESEARCH": "CALIBRATED FORECAST RESEARCH MODE",
    "RAW_TARGET_RESEARCH": "RAW TARGET RETURN RESEARCH MODE",
}

LATE_COMPACT_REPORT_SECTIONS = {
    "INSTITUTIONAL_RISK",
    "BLACK_LITTERMAN",
    "ACTION_SIGNALS",
    "PAPER_META_FILTER",
    "PAPER_TRADING",
    "FORECAST_CALIBRATION",
    "INFORMATION_COEFFICIENT",
    "HEURISTIC_AUDIT",
    "HEURISTIC_CALIBRATION",
    "FACTOR_ATTRIBUTION",
}


class _CompactReportPrinter:
    def __init__(
        self,
        original_print,
        sections_to_show: Iterable[str],
        max_lines_per_section: int = 14,
        early_exit_after_final_allocation: bool = False,
    ) -> None:
        self.original_print = original_print
        self.allowed_sections = set(sections_to_show)
        self.max_lines_per_section = max_lines_per_section
        self.early_exit_after_final_allocation = early_exit_after_final_allocation
        self.active_section: str | None = None
        self.section_line_count = 0

    def _section_for_header(self, text: str) -> str | None:
        if "=====" not in text:
            return None
        normalized = text.upper()
        for section_key, header_text in REPORT_SECTION_HEADERS.items():
            if header_text in normalized:
                if section_key in {
                    "TREND_PERSISTENCE_BREAKDOWN",
                }:
                    return "TREND_PERSISTENCE"
                if section_key in {
                    "EMA_VS_TREND_PERSISTENCE_ANALYSIS",
                }:
                    return "EMA_VS_TREND_PERSISTENCE"
                return section_key
        return "__UNKNOWN__"

    def __call__(self, *args, **kwargs) -> None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args)
        section_key = self._section_for_header(text)
        if section_key is not None:
            self.active_section = section_key if section_key in self.allowed_sections else None
            self.section_line_count = 0
            if self.active_section is None:
                return

        if self.active_section is None:
            return

        lines = text.splitlines() or [""]
        remaining = self.max_lines_per_section - self.section_line_count
        if remaining <= 0:
            return
        shown = lines[:remaining]
        self.section_line_count += len(shown)
        self.original_print("\n".join(shown), end=end)
        if len(lines) > remaining:
            self.original_print(
                f"[compact mode] section truncated after {self.max_lines_per_section} lines."
            )
        if (
            self.early_exit_after_final_allocation
            and self.active_section == "FINAL_ALLOCATION"
            and "Use final_weight_percent" in text
        ):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def _install_compact_report_mode(
    sections_to_show: Iterable[str],
    max_lines_per_section: int = 14,
    early_exit_after_final_allocation: bool = False,
):
    original_print = builtins.print
    builtins.print = _CompactReportPrinter(
        original_print=original_print,
        sections_to_show=sections_to_show,
        max_lines_per_section=max_lines_per_section,
        early_exit_after_final_allocation=early_exit_after_final_allocation,
    )
    return original_print


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


def _build_download_params(period: str | None = None, days: int | None = None) -> dict:
    """
    Build yfinance download parameters from a period string or a number of days.
    """
    if period and days is not None:
        raise ValueError("Use either 'period' or 'days', not both.")

    if not period and days is None:
        raise ValueError("You must provide 'period' or 'days'.")

    if period:
        return {"period": period, "interval": "1d", "progress": False, "auto_adjust": False}

    if days is None or days <= 0:
        raise ValueError("'days' must be a positive integer.")

    end_date = _daily_download_end_date()
    start_date = end_date - timedelta(days=days)
    return {
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "interval": "1d",
        "progress": False,
        "auto_adjust": False,
    }


def _extract_close_series(raw_data: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Extract a normalized close-price Series from yfinance output.
    """
    if raw_data.empty:
        raise ValueError(f"No data returned for {ticker}.")

    if "Close" not in raw_data.columns:
        raise ValueError(f"No 'Close' column found for {ticker}.")

    close_data = raw_data["Close"]

    if isinstance(close_data, pd.DataFrame):
        if close_data.empty or close_data.shape[1] == 0:
            raise ValueError(f"Empty close-price data for {ticker}.")
        close_series = close_data.iloc[:, 0]
    else:
        close_series = close_data

    close_series = close_series.dropna()
    close_series.name = ticker

    if close_series.empty:
        raise ValueError(f"Close price series is empty after cleaning for {ticker}.")

    return close_series


def download_close_prices(
    tickers: Iterable[str],
    period: str | None = None,
    days: int | None = None,
) -> pd.DataFrame:
    """
    Download daily close prices for the requested tickers.
    Invalid tickers are skipped with a warning.
    """
    ticker_list = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
    if not ticker_list:
        raise ValueError("The ticker list cannot be empty.")

    if period:
        effective_lookback = None
    else:
        effective_lookback = days if days is not None else lookback_days
        if effective_lookback <= 0:
            raise ValueError("'days' must be a positive integer.")
        end_date = _daily_download_end_date()
        start_date = end_date - timedelta(days=effective_lookback * 2)
    price_series: dict[str, pd.Series] = {}

    for ticker in ticker_list:
        try:
            with _temporary_disable_proxies():
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    if period:
                        raw_data = yf.download(
                            ticker,
                            period=period,
                            interval="1d",
                            progress=False,
                            auto_adjust=False,
                            threads=False,
                            timeout=20,
                        )
                    else:
                        raw_data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            interval="1d",
                            progress=False,
                            auto_adjust=False,
                            threads=False,
                            timeout=20,
                        )
            close_prices = _extract_close_series(raw_data, ticker)
            if effective_lookback is not None:
                close_prices = close_prices.tail(effective_lookback)
            close_prices = close_prices.dropna()
        except Exception as exc:
            print(f"[WARNING] Error downloading {ticker}: {exc}")
            continue

        price_series[ticker] = close_prices

    if not price_series:
        raise ValueError("No valid data could be downloaded for the provided tickers.")

    prices_df = pd.concat(price_series.values(), axis=1, sort=False)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.sort_index(inplace=True)
    print("Rows before cleaning:", len(prices_df))
    prices_df = prices_df.ffill()
    prices_df = prices_df.dropna(how="all")
    if effective_lookback is not None:
        prices_df = prices_df.tail(effective_lookback)
    print("Rows after cleaning:", len(prices_df))

    if effective_lookback is not None:
        min_valid_rows = int(effective_lookback * 0.95)
    else:
        min_valid_rows = max(60, int(len(prices_df) * 0.80))

    valid_tickers = []
    for ticker in ticker_list:
        if ticker not in prices_df.columns:
            continue
        series = prices_df[ticker].dropna()
        if len(series) < min_valid_rows:
            continue
        if series.nunique() <= 1:
            continue
        valid_tickers.append(ticker)

    invalid_tickers = [ticker for ticker in ticker_list if ticker not in valid_tickers]
    if invalid_tickers:
        print(f"Removed invalid tickers: {invalid_tickers}")
    if not valid_tickers:
        raise ValueError("No valid tickers available after cleaning downloaded data.")
    prices_df = prices_df[valid_tickers]
    prices_df = prices_df.dropna(how="any")

    if effective_lookback is not None:
        minimum_expected_rows = int(effective_lookback * 0.8)
        if len(prices_df) < minimum_expected_rows:
            print(
                f"[WARNING] Only {len(prices_df)} market days were downloaded; "
                f"expected at least {minimum_expected_rows} for lookback_days={effective_lookback}."
            )

    return prices_df


def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns from a price DataFrame.
    """
    if prices_df.empty:
        raise ValueError("The prices DataFrame is empty.")

    cleaned_prices = prices_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if cleaned_prices.shape[0] < 3:
        raise ValueError("Not enough aligned price rows to calculate daily returns.")

    returns_df = cleaned_prices.pct_change(fill_method=None).dropna(how="all")
    returns_df = returns_df.dropna(axis=1, how="all").dropna(how="any")
    returns_df = returns_df.loc[:, returns_df.std() > 0]
    if returns_df.shape[0] < 2 or returns_df.shape[1] < 2:
        raise ValueError("Not enough valid return data after cleaning.")
    return returns_df


def calculate_covariance_matrix(
    returns_df: pd.DataFrame,
    method: str = "ledoit_wolf",
    shrinkage_intensity: float = 0.10,
) -> pd.DataFrame:
    """
    Calculate the covariance matrix for all assets in the returns DataFrame.
    """
    if returns_df.empty:
        raise ValueError("The returns DataFrame is empty.")

    return calculate_covariance(returns_df, method=method, shrinkage_intensity=shrinkage_intensity)


def calculate_volatility(returns_df: pd.DataFrame) -> pd.Series:
    """
    Calculate the volatility for each asset in the returns DataFrame.
    """
    if returns_df.empty:
        raise ValueError("The returns DataFrame is empty.")

    return returns_df.std()


def calculate_asset_sharpe_ratio(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0,
) -> pd.Series:
    """
    Calculate the Sharpe ratio for each asset in the returns DataFrame.
    """
    if returns_df.empty:
        raise ValueError("The returns DataFrame is empty.")

    volatility = calculate_volatility(returns_df)
    sharpe_ratio = (returns_df.mean() - risk_free_rate) / volatility
    return sharpe_ratio.dropna()


def _validate_weights(weights: Iterable[float], n_assets: int) -> np.ndarray:
    """
    Validate and normalize the weights container shape for portfolio calculations.
    """
    weights_array = np.asarray(list(weights), dtype=float)

    if weights_array.ndim != 1:
        raise ValueError("'weights' must be a one-dimensional list or array.")

    if len(weights_array) != n_assets:
        raise ValueError(
            f"'weights' length ({len(weights_array)}) must match the number of assets ({n_assets})."
        )

    return weights_array


def calculate_portfolio_return(returns_df: pd.DataFrame, weights: Iterable[float]) -> float:
    """
    Calculate the portfolio return using the mean historical returns and asset weights.
    """
    if returns_df.empty:
        raise ValueError("The returns DataFrame is empty.")

    weights_array = _validate_weights(weights, returns_df.shape[1])
    portfolio_return = np.dot(weights_array, returns_df.mean())
    return float(portfolio_return)


def calculate_portfolio_volatility(cov_matrix: pd.DataFrame, weights: Iterable[float]) -> float:
    """
    Calculate the portfolio volatility using the covariance matrix and asset weights.
    """
    if cov_matrix.empty:
        raise ValueError("The covariance matrix is empty.")

    weights_array = _validate_weights(weights, cov_matrix.shape[1])
    portfolio_variance = weights_array.T @ cov_matrix.to_numpy() @ weights_array
    portfolio_volatility = np.sqrt(portfolio_variance)
    return float(portfolio_volatility)


def calculate_sharpe_ratio(
    portfolio_return: float,
    portfolio_volatility: float,
    risk_free_rate: float = 0,
) -> float:
    """
    Calculate the portfolio Sharpe ratio from historical portfolio metrics.
    """
    if portfolio_volatility == 0:
        raise ValueError("Portfolio volatility cannot be zero.")

    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return float(sharpe_ratio)


def compare_covariance_optimizer_outputs(
    *,
    returns_df: pd.DataFrame,
    expected_daily_returns: pd.Series,
    rf_daily: float,
    optimizer_config: dict[str, object],
    shrinkage_intensity: float = 0.10,
) -> pd.DataFrame:
    """
    Diagnostic-only optimizer comparison using historical vs Ledoit-Wolf covariance.
    The production allocation remains controlled by the existing optimizer path.
    """
    rows: list[dict[str, object]] = []
    for method in ("historical", "ledoit_wolf"):
        cov_matrix = calculate_covariance_matrix(
            returns_df,
            method=method,
            shrinkage_intensity=shrinkage_intensity,
        )
        cov_meta = covariance_method_metadata(
            returns_df,
            method=method,
            shrinkage_intensity=shrinkage_intensity,
        )
        cov_diag = covariance_diagnostics_for_comparison(cov_matrix, method, cov_meta)
        optimizer = PortfolioOptimizer(
            returns_df=returns_df,
            rf_daily=rf_daily,
            expected_daily_returns=expected_daily_returns,
            covariance_matrix=cov_matrix,
            **optimizer_config,
        )
        with redirect_stdout(io.StringIO()):
            weights, sharpe, portfolio_return, portfolio_volatility, _ = optimizer.optimize()
        weight_series = pd.Series(weights, index=returns_df.columns, dtype=float)
        concentration = float(np.square(weight_series).sum())
        rows.append(
            {
                "covariance_method": method,
                "weights": weight_series.sort_values(ascending=False).round(4).to_dict(),
                "portfolio_return": float(portfolio_return),
                "volatility": float(portfolio_volatility),
                "sharpe": float(sharpe),
                "concentration_hhi": concentration,
                "cash": np.nan,
                "condition_number": cov_diag["condition_number"],
                "average_correlation": cov_diag["average_correlation"],
                "shrinkage_method_used": cov_meta["shrinkage_method_used"],
                "shrinkage_intensity": cov_meta["shrinkage_intensity"],
            }
        )
    return pd.DataFrame(rows)


def covariance_diagnostics_for_comparison(
    cov_matrix: pd.DataFrame,
    method: str,
    metadata: dict[str, float | str],
) -> dict[str, float | str]:
    from covariance_estimation import covariance_diagnostics

    return covariance_diagnostics(
        cov_matrix,
        method,
        shrinkage_method_used=str(metadata["shrinkage_method_used"]),
        shrinkage_intensity=float(metadata["shrinkage_intensity"]),
    )


def get_risk_free_rate(default_rate: float = DEFAULT_RISK_FREE_RATE) -> tuple[float, float]:
    """
    Download the Federal Funds Rate from the FRED API and return both annual
    and daily risk-free rates. Falls back to a default rate if the API fails.
    """
    api_key = "28556facd59b4c6cd5554266559aec06"
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=FEDFUNDS&api_key={api_key}&file_type=json"
    )

    try:
        with _temporary_disable_proxies():
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()["observations"]
    except Exception as exc:
        print(f"[WARNING] Could not download risk-free rate from FRED: {exc}")
        return default_rate, default_rate / TRADING_DAYS_PER_YEAR

    values = [float(item["value"]) for item in data if item.get("value") not in {None, "."}]
    last_values = values[-5:]

    if not last_values:
        print("[WARNING] No valid risk-free rate data found in FRED. Using default value.")
        return default_rate, default_rate / TRADING_DAYS_PER_YEAR

    rf_annual = float(sum(last_values) / len(last_values) / 100)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    return rf_annual, rf_daily


def generate_target_prices(
    prices_df: pd.DataFrame,
    method: str = "percentage",
    manual_targets: dict[str, float] | None = None,
    base_target_pct: float = 0.10,
    momentum_k: float = 1.0,
    volatility_k: float = 1.5,
) -> pd.Series:
    """
    Generate dynamic target prices for all assets in prices_df.
    """
    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    manual_series = pd.Series(manual_targets or {}, dtype=float)
    latest_prices = prices_df.ffill().iloc[-1]
    volatility_series = prices_df.pct_change().std().reindex(prices_df.columns).fillna(0.0)
    momentum_series = (
        prices_df.ffill().iloc[-1] / prices_df.ffill().tail(10).mean() - 1
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    generated_targets: dict[str, float] = {}

    for ticker in prices_df.columns:
        current_price = float(latest_prices.loc[ticker])
        if ticker in manual_series and np.isfinite(manual_series.loc[ticker]) and manual_series.loc[ticker] > 0:
            generated_targets[ticker] = float(manual_series.loc[ticker])
            continue

        momentum = max(float(momentum_series.loc[ticker]), 0.001)
        volatility = max(float(volatility_series.loc[ticker]), 0.001)

        if method == "manual_dict":
            target_pct = max(base_target_pct, momentum_k * momentum)
        elif method == "volatility_adjusted":
            target_pct = max(base_target_pct * 0.5, volatility_k * volatility + momentum_k * momentum)
        elif method == "percentage":
            target_pct = max(base_target_pct, momentum_k * momentum)
        else:
            raise ValueError(f"Unsupported target generation method: {method}")

        target_pct = float(np.clip(target_pct, 0.02, 0.75))
        target_price = current_price * (1 + target_pct)
        generated_targets[ticker] = max(float(target_price), current_price * 1.01)

    target_prices = pd.Series(generated_targets, dtype=float).reindex(prices_df.columns)
    if target_prices.isna().any():
        raise ValueError("Target price generation failed for one or more tickers.")

    return target_prices


def _status_reason(status: str) -> str:
    reason_map = {
        "unreachable_target": "Target too far relative to momentum/volatility",
        "bearish_or_low_probability": "Negative expected return or weak drift",
        "live_price_gap_warning": "Live price differs materially from recent close",
        "reachable": "Expected return remains actionable",
        "missing_price_data": "Insufficient historical price data",
        "invalid_current_price": "Invalid current price",
        "numerical_fallback": "Numerical stability fallback applied",
    }
    return reason_map.get(status, "No specific reason available")


def _prune_small_weights(weights: np.ndarray, labels: list[str], threshold: float = 0.01) -> pd.Series:
    weight_series = pd.Series(weights, index=labels, dtype=float)
    pruned = weight_series[weight_series >= threshold]
    if pruned.empty:
        pruned = weight_series.sort_values(ascending=False).head(1)
    pruned = pruned / pruned.sum()
    return pruned


def run_backtest(
    prices_df: pd.DataFrame,
    rebalance_frequency: str = "weekly",
    lookback_window: int = 252,
    verbose: bool = False,
    max_rebalances: int | None = None,
) -> pd.DataFrame:
    if prices_df.empty:
        raise ValueError("prices_df está vacío.")

    freq = str(rebalance_frequency).lower()
    if freq == "daily":
        step = 1
    elif freq == "weekly":
        step = 5
    elif freq == "monthly":
        step = 21
    else:
        raise ValueError("rebalance_frequency debe ser 'daily', 'weekly' o 'monthly'.")

    px = prices_df.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    initial_tickers = list(px.columns)
    px = px.ffill()

    valid_cols = [c for c in px.columns if not px[c].isna().all()]
    removed_cols = [c for c in px.columns if c not in valid_cols]
    px = px[valid_cols]
    final_tickers = list(px.columns)

    ema_max_period = 200
    min_history_required = max(int(lookback_window), 252, int(ema_max_period + 50))
    start_t = min_history_required - 1

    def _vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def _download_single_close(ticker: str, period: str) -> pd.Series:
        with _temporary_disable_proxies():
            data = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
        if data.empty or "Close" not in data.columns:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            if close.empty or close.shape[1] == 0:
                return pd.Series(dtype=float)
            close = close.iloc[:, 0]
        return close.dropna().astype(float)

    spy_close_full = _download_single_close("SPY", period="5y")
    vix_close_full = _download_single_close("^VIX", period="5y")
    spy_available = not spy_close_full.empty
    vix_available = not vix_close_full.empty

    print("\n===== DATA INTEGRITY CHECK =====")
    print(f"filas de prices_df: {len(px)}")
    print(f"tickers válidos: {len(final_tickers)}")
    print(f"SPY disponible: {spy_available}")
    print(f"VIX disponible: {vix_available}")
    print(f"NaN en SPY: {int(spy_close_full.isna().sum()) if spy_available else 'N/A'}")
    print(f"NaN en VIX: {int(vix_close_full.isna().sum()) if vix_available else 'N/A'}")
    print(f"fecha inicial SPY: {spy_close_full.index.min().date() if spy_available else 'N/A'}")
    print(f"fecha final SPY: {spy_close_full.index.max().date() if spy_available else 'N/A'}")
    print(f"fecha inicial VIX: {vix_close_full.index.min().date() if vix_available else 'N/A'}")
    print(f"fecha final VIX: {vix_close_full.index.max().date() if vix_available else 'N/A'}")

    if not spy_available or not vix_available:
        raise RuntimeError("No se pudo descargar SPY o ^VIX al inicio. Backtest detenido.")

    _vprint("\n===== UNIVERSE CHECK =====")
    _vprint(f"cantidad de tickers iniciales: {len(initial_tickers)}")
    _vprint(f"cantidad de tickers con datos válidos: {len(valid_cols)}")
    _vprint(f"tickers removidos por NaN o datos insuficientes: {removed_cols}")
    _vprint(f"tickers finales usados en backtest: {final_tickers}")

    _vprint("\n===== WARMUP CHECK =====")
    _vprint(f"filas totales: {len(px)}")
    _vprint(f"lookback_window: {lookback_window}")
    _vprint(f"ema_max_period detectado: {ema_max_period}")
    _vprint(f"min_history_required: {min_history_required}")

    if len(px) <= start_t + step + 1:
        raise ValueError("No hay suficientes datos para cumplir warm-up + forward window.")

    _vprint(f"fecha real de inicio del backtest: {px.index[start_t].date()}")

    rf_annual, rf_daily = get_risk_free_rate()
    records: list[dict] = []
    selected_history: list[set[str]] = []
    turnovers: list[float] = []
    skipped_periods = 0
    available_assets_counts: list[int] = []
    selected_assets_counts: list[int] = []
    cash_weights: list[float] = []
    sum_asset_weights: list[float] = []
    optimized_period_returns: list[float] = []
    equal_weight_period_returns: list[float] = []
    capital = 1.0
    executed_rebalances = 0

    for t in range(start_t, len(px) - step - 2, step):
        if max_rebalances is not None and executed_rebalances >= int(max_rebalances):
            break
        historical_prices = px.iloc[: t + 1]
        forward_prices = px.iloc[t + 1 : t + 2 + step]
        if forward_prices.empty:
            break

        available_tickers = [
            c for c in historical_prices.columns if historical_prices[c].dropna().shape[0] >= min_history_required
        ]
        historical_prices = historical_prices[available_tickers]
        available_assets_counts.append(len(available_tickers))

        if len(available_tickers) < 2:
            skipped_periods += 1
            continue

        returns_hist = calculate_daily_returns(historical_prices)
        if returns_hist.shape[1] < 2 or returns_hist.empty:
            skipped_periods += 1
            continue

        hist_end = historical_prices.index[-1]
        spy_hist = spy_close_full.loc[spy_close_full.index <= hist_end]
        vix_hist = vix_close_full.loc[vix_close_full.index <= hist_end]
        market_regime = compute_market_regime_model(
            prices_df=historical_prices,
            returns_df=returns_hist,
            spy_close=spy_hist,
            vix_close=vix_hist,
            strict_market_data=True,
        )
        regime_score = float(market_regime["risk_score"])
        regime_type = str(market_regime["regime"])
        regime_confidence = float(market_regime.get("regime_confidence", 0.3))

        target_prices = generate_target_prices(
            historical_prices,
            method="volatility_adjusted",
            manual_targets=None,
        )
        expected_daily_returns, diagnostics = compute_expected_returns(
            historical_prices,
            target_prices,
            use_live_prices=False,
            risk_free_daily=rf_daily,
            center_distribution=False,
            dead_zone=0.001,
            regime_score=regime_score,
            regime_type=regime_type,
            regime_signal_threshold=0.2,
            use_raw_target_return=False,
        )
        expected_daily_returns = expected_daily_returns.reindex(available_tickers).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        diagnostics_df = pd.DataFrame(diagnostics).T.reindex(available_tickers)
        if "signal_strength" not in diagnostics_df.columns:
            diagnostics_df["signal_strength"] = 0.0
        signal_strength = pd.to_numeric(diagnostics_df["signal_strength"], errors="coerce").fillna(0.0)

        timing_df = compute_asset_timing(historical_prices, timing_mode="daily")
        timing_adjusted_returns = apply_timing_to_expected_returns(expected_daily_returns, timing_df)

        candidates = timing_adjusted_returns[(timing_adjusted_returns > 0) & (signal_strength > 0.15)]
        if candidates.empty:
            selected = timing_adjusted_returns.sort_values(ascending=False).head(min(5, len(timing_adjusted_returns))).index.tolist()
        else:
            selected = candidates.sort_values(ascending=False).head(min(5, len(candidates))).index.tolist()

        if len(selected) < 2:
            skipped_periods += 1
            continue
        selected_assets_counts.append(len(selected))

        selected_set = set(selected)
        if selected_history:
            prev = selected_history[-1]
            union = len(prev.union(selected_set))
            turnover = (len(prev.symmetric_difference(selected_set)) / union) if union else 0.0
            turnovers.append(float(turnover))
        selected_history.append(selected_set)

        selected_returns_hist = returns_hist[selected]
        expected_selected = timing_adjusted_returns.reindex(selected).fillna(0.0)

        optimizer = PortfolioOptimizer(
            returns_df=selected_returns_hist,
            rf_daily=rf_daily,
            expected_daily_returns=expected_selected,
            use_expected_returns=True,
            alpha=0.5,
            no_opportunity=False,
            defensive_mode=True,
            max_weight=0.50,
            regime_score=regime_score,
            regime_type=regime_type,
            regime_confidence=regime_confidence,
        )
        stdout_ctx = nullcontext() if verbose else redirect_stdout(io.StringIO())
        with stdout_ctx:
            best_weights, _, _, _, _ = optimizer.optimize()
        w = pd.Series(best_weights, index=selected, dtype=float).clip(lower=0.0)
        if float(w.sum()) <= 0:
            w[:] = 1.0 / len(w)
        else:
            w = w / w.sum()

        # Decisión con data hasta t; retorno desde t+1 hasta t+step+1.
        buy_start = forward_prices.index[0]
        buy_end = forward_prices.index[-1]

        # Alignment checks
        missing_in_forward = [a for a in selected if a not in forward_prices.columns]
        if missing_in_forward:
            print(f"[WARNING] alignment error - missing assets in future_prices: {missing_in_forward}")
            skipped_periods += 1
            continue

        start_prices = px.loc[buy_start, selected].replace([np.inf, -np.inf], np.nan)
        end_prices = px.loc[buy_end, selected].replace([np.inf, -np.inf], np.nan)
        forward_returns = (end_prices / start_prices - 1.0).replace([np.inf, -np.inf], np.nan)
        w_aligned = w.reindex(forward_returns.index).astype(float)
        if not w_aligned.index.equals(forward_returns.index):
            print("[WARNING] alignment error - weights index does not match forward returns index.")
            skipped_periods += 1
            continue

        if forward_returns.isna().any():
            nan_tickers = forward_returns[forward_returns.isna()].index.tolist()
            print(
                f"[WARNING] NaN en forward_returns | date={buy_end.date()} | tickers={nan_tickers} | "
                f"entry={buy_start.date()} exit={buy_end.date()}"
            )
            skipped_periods += 1
            continue

        weighted_returns = w_aligned * forward_returns
        portfolio_return_before_costs = float(weighted_returns.sum(min_count=1)) if not weighted_returns.empty else np.nan
        transaction_cost = 0.0
        period_return = float(portfolio_return_before_costs - transaction_cost) if np.isfinite(portfolio_return_before_costs) else 0.0

        cash_weight = float(max(0.0, 1.0 - w_aligned.sum()))

        zero_return_cause = None
        if len(selected) == 0:
            zero_return_cause = "no selected assets"
        elif not np.isfinite(w_aligned.sum()) or float(w_aligned.sum()) <= 0:
            zero_return_cause = "all weights zero"
        elif forward_prices.shape[0] < 2:
            zero_return_cause = "future prices empty"
        elif forward_returns.dropna().empty:
            zero_return_cause = "alignment error"
        elif float(forward_returns.dropna().abs().sum()) == 0.0:
            zero_return_cause = "price movement zero"
        elif cash_weight >= 0.999999:
            zero_return_cause = "all cash"

        equal_w = pd.Series(1.0 / len(selected), index=selected, dtype=float)
        equal_weight_return = float((equal_w * forward_returns).sum(min_count=1))

        if executed_rebalances < 10:
            print("\n===== RETURN DEBUG =====")
            print(f"date: {buy_end.date()}")
            print(f"entry_date: {buy_start.date()}")
            print(f"exit_date: {buy_end.date()}")
            print(f"selected_assets: {selected}")
            print(f"weights: {w_aligned.to_dict()}")
            print(f"sum_weights: {float(w_aligned.sum()):.6f}")
            print(f"cash_weight: {cash_weight:.6f}")
            print(f"future_prices.shape: {forward_prices.shape}")
            print("future_prices.head():")
            print(forward_prices.head())
            print("future_prices.tail():")
            print(forward_prices.tail())
            print(f"future_returns: {forward_returns.to_dict()}")
            print(f"portfolio_return_before_costs: {portfolio_return_before_costs}")
            print(f"transaction_cost: {transaction_cost}")
            print(f"final_period_return: {period_return}")
            if period_return == 0.0:
                print(f"zero_return_cause: {zero_return_cause if zero_return_cause else 'unknown'}")
            print("\n===== MANUAL RETURN CHECK =====")
            print(f"date: {buy_end.date()}")
            print(f"selected_assets: {selected}")
            print(f"weights activos: {w_aligned.to_dict()}")
            print(f"entry_prices: {start_prices.to_dict()}")
            print(f"exit_prices: {end_prices.to_dict()}")
            print(f"asset_returns individuales: {forward_returns.to_dict()}")
            print(f"weighted_asset_returns: {weighted_returns.to_dict()}")
            print(f"portfolio_return_before_costs: {portfolio_return_before_costs}")
            print(f"transaction_cost: {transaction_cost}")
            print(f"final_period_return: {period_return}")

        if len(selected) == 0:
            raise ValueError("selected_assets vacío.")
        if not np.issubdtype(w_aligned.dtype, np.number) or float(w_aligned.sum()) <= 0:
            raise ValueError("weights inválidos: no numéricos o suma <= 0.")
        if forward_prices.shape[0] < 2:
            raise ValueError("future_prices debe tener al menos 2 filas.")

        capital *= (1.0 + period_return)
        cash_weights.append(cash_weight)
        sum_asset_weights.append(float(w_aligned.sum()))
        optimized_period_returns.append(period_return)
        equal_weight_period_returns.append(equal_weight_return)

        _vprint("\n===== LOOKAHEAD CHECK =====")
        _vprint(f"fecha de decisión: {px.index[t].date()}")
        _vprint(f"última fecha usada para señales: {historical_prices.index[-1].date()}")
        _vprint(f"primera fecha usada para retorno forward: {buy_start.date()}")
        _vprint(f"forward > histórica: {buy_start > historical_prices.index[-1]}")

        records.append(
            {
                "date": buy_end,
                "return": period_return,
                "cumulative_return": capital - 1.0,
                "selected_assets": selected,
            }
        )
        executed_rebalances += 1

    if not records:
        raise ValueError("No se generaron períodos válidos para el backtest.")

    backtest_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    returns = backtest_df["return"].astype(float)
    n_periods = len(backtest_df)
    periods_per_year = 252 / step
    total_return = float((1.0 + returns).prod() - 1.0)
    annualized_return = float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0) if n_periods > 0 else 0.0
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0
    sharpe = float((annualized_return - rf_annual) / annualized_volatility) if annualized_volatility > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    win_rate = float((returns > 0).mean()) if n_periods else 0.0
    avg_ret = float(returns.mean()) if n_periods else 0.0

    avg_available = float(np.mean(available_assets_counts)) if available_assets_counts else 0.0
    avg_selected = float(np.mean(selected_assets_counts)) if selected_assets_counts else 0.0
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    print("\n===== BACKTEST RESULTS 100 REBALANCES =====")
    print(f"total return: {total_return:.6f}")
    print(f"annualized return: {annualized_return:.6f}")
    print(f"annualized volatility: {annualized_volatility:.6f}")
    print(f"Sharpe ratio: {sharpe:.6f}")
    print(f"max drawdown: {max_drawdown:.6f}")
    print(f"win rate: {win_rate:.2%}")
    print(f"average return per period: {avg_ret:.6f}")

    non_zero_count = int((returns != 0).sum())
    avg_cash_weight = float(np.mean(cash_weights)) if cash_weights else 0.0
    avg_sum_asset_weights = float(np.mean(sum_asset_weights)) if sum_asset_weights else 0.0
    avg_opt_ret = float(np.mean(optimized_period_returns)) if optimized_period_returns else 0.0
    avg_eq_ret = float(np.mean(equal_weight_period_returns)) if equal_weight_period_returns else 0.0

    print("\n===== BACKTEST DIAGNOSTICS 100 REBALANCES =====")
    print(f"non_zero_count: {non_zero_count}")
    print(f"número de rebalanceos: {n_periods}")
    print(f"fecha inicial real: {backtest_df['date'].iloc[0].date()}")
    print(f"fecha final: {backtest_df['date'].iloc[-1].date()}")
    print(f"promedio de activos disponibles: {avg_available:.2f}")
    print(f"promedio de activos seleccionados: {avg_selected:.2f}")
    print(f"turnover promedio: {avg_turnover:.4f}")
    print(f"promedio cash_weight: {avg_cash_weight:.6f}")
    print(f"promedio sum_weights activos: {avg_sum_asset_weights:.6f}")
    print(f"cantidad de períodos saltados por falta de datos: {skipped_periods}")
    print(f"avg optimized return: {avg_opt_ret:.8f}")
    print(f"avg equal-weight selected return: {avg_eq_ret:.8f}")
    print(f"optimized minus equal-weight: {(avg_opt_ret - avg_eq_ret):.8f}")
    backtest_df.to_csv("backtest_daily_audit_clean.csv", index=False)
    print("\nHEAD:")
    print(backtest_df.head())
    print("\nTAIL:")
    print(backtest_df.tail())

    return backtest_df


def analyze_backtest_performance(
    backtest_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> None:
    if backtest_df.empty:
        print("[WARNING] backtest_df está vacío.")
        return
    if "date" not in backtest_df.columns or "return" not in backtest_df.columns or "selected_assets" not in backtest_df.columns:
        print("[WARNING] backtest_df no contiene columnas requeridas: date, return, selected_assets.")
        return

    bt = backtest_df.copy()
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt.sort_values("date").reset_index(drop=True)
    px = prices_df.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().ffill()

    print("\n===== BACKTEST PERIOD DIAGNOSTIC =====")
    print("Retorno por trade:")
    print(bt[["date", "return"]].to_string(index=False))

    best_idx = int(bt["return"].idxmax())
    worst_idx = int(bt["return"].idxmin())
    print(
        f"Mejor período: {bt.loc[best_idx, 'date'].date()} | return={float(bt.loc[best_idx, 'return']):.6f}"
    )
    print(
        f"Peor período: {bt.loc[worst_idx, 'date'].date()} | return={float(bt.loc[worst_idx, 'return']):.6f}"
    )

    positive = bt.loc[bt["return"] > 0, "return"]
    negative = bt.loc[bt["return"] < 0, "return"]
    avg_pos = float(positive.mean()) if len(positive) else 0.0
    avg_neg = float(negative.mean()) if len(negative) else 0.0
    gain_loss = float(avg_pos / abs(avg_neg)) if abs(avg_neg) > 1e-12 else 0.0
    print(f"Promedio retornos positivos: {avg_pos:.6f}")
    print(f"Promedio retornos negativos: {avg_neg:.6f}")
    print(f"Ratio gain/loss: {gain_loss:.6f}")

    contrib_rows: list[dict[str, float | str | pd.Timestamp]] = []
    for i in range(len(bt)):
        current_date = pd.to_datetime(bt.loc[i, "date"])
        assets = list(bt.loc[i, "selected_assets"]) if isinstance(bt.loc[i, "selected_assets"], (list, tuple, set)) else []
        if not assets:
            continue
        if current_date not in px.index:
            continue
        current_idx = px.index.get_loc(current_date)
        if isinstance(current_idx, slice):
            current_idx = current_idx.start
        if current_idx <= 0:
            continue
        prev_date = px.index[current_idx - 1]
        curr_prices = px.loc[current_date, assets].replace([np.inf, -np.inf], np.nan)
        prev_prices = px.loc[prev_date, assets].replace([np.inf, -np.inf], np.nan)
        asset_ret = (curr_prices / prev_prices - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        if asset_ret.empty:
            continue
        ew = 1.0 / len(asset_ret)
        for asset, r in asset_ret.items():
            contrib_rows.append(
                {
                    "date": current_date,
                    "asset": str(asset),
                    "contribution": float(ew * float(r)),
                }
            )

    contrib_df = pd.DataFrame(contrib_rows)
    print("\n===== ASSET CONTRIBUTION DIAGNOSTIC =====")
    if contrib_df.empty:
        print("No hay contribuciones para analizar.")
        return

    summary = (
        contrib_df.groupby("asset", as_index=True)["contribution"]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "total_contribution", "mean": "avg_contribution", "count": "periods_selected"})
        .sort_values("total_contribution", ascending=False)
    )
    neg_rate = (
        contrib_df.assign(neg=contrib_df["contribution"] < 0)
        .groupby("asset", as_index=True)["neg"]
        .mean()
        .rename("negative_contribution_rate")
    )
    summary = summary.join(neg_rate, how="left").fillna(0.0)
    print("Top activos por contribución:")
    print(summary.head(10))
    print("Activos que destruyen valor:")
    print(summary.sort_values("total_contribution", ascending=True).head(10))

    recurrent_losers = summary[
        (summary["total_contribution"] < 0) & (summary["negative_contribution_rate"] >= 0.5)
    ]
    print("Activos recurrentemente perdedores:")
    print(recurrent_losers if not recurrent_losers.empty else "Ninguno detectado")


def _minmax_scale(series: pd.Series, clip_low: float = 0.0, clip_high: float = 1.0) -> pd.Series:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s_min = float(s.min()) if len(s) else 0.0
    s_max = float(s.max()) if len(s) else 0.0
    if s_max - s_min < 1e-12:
        return pd.Series(0.5, index=s.index, dtype=float).clip(clip_low, clip_high)
    scaled = (s - s_min) / (s_max - s_min)
    return scaled.clip(clip_low, clip_high)


def _count_ranking_swaps(base_order: list[str], final_order: list[str]) -> int:
    if not base_order or not final_order:
        return 0
    common = [asset for asset in base_order if asset in set(final_order)]
    if len(common) <= 1:
        return 0
    base_pos = {asset: i for i, asset in enumerate(common)}
    final_pos = {asset: i for i, asset in enumerate(final_order) if asset in base_pos}
    swaps = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a = common[i]
            b = common[j]
            if (base_pos[a] - base_pos[b]) * (final_pos.get(a, 0) - final_pos.get(b, 0)) < 0:
                swaps += 1
    return swaps


def calculate_downside_metrics(returns_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    metrics: dict[str, dict[str, float]] = {}
    for ticker in returns_df.columns:
        s = pd.Series(returns_df[ticker], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            metrics[ticker] = {
                "downside_volatility": 0.0,
                "downside_ratio": 0.0,
                "recent_drawdown": 0.0,
                "return_skew": 0.0,
                "consistency": 0.0,
                "stability_raw": 0.0,
            }
            continue

        total_vol = float(s.std()) if len(s) > 1 else 0.0
        negative = s[s < 0]
        downside_vol = float(negative.std()) if len(negative) > 1 else 0.0
        downside_ratio = float(downside_vol / (total_vol + 1e-12))
        downside_ratio = float(np.clip(downside_ratio, 0.0, 3.0))

        recent = s.tail(window).fillna(0.0)
        recent_curve = (1.0 + recent).cumprod()
        running_max = recent_curve.cummax()
        drawdown_series = recent_curve / running_max - 1.0
        recent_drawdown = float(abs(min(float(drawdown_series.min()), 0.0)))

        return_skew = float(s.skew()) if len(s) > 2 else 0.0
        consistency = float((s > 0).mean())
        rolling_std = s.rolling(20).std().dropna()
        stability_raw = float(rolling_std.std()) if len(rolling_std) > 1 else 0.0

        metrics[ticker] = {
            "downside_volatility": downside_vol,
            "downside_ratio": downside_ratio,
            "recent_drawdown": recent_drawdown,
            "return_skew": return_skew,
            "consistency": consistency,
            "stability_raw": stability_raw,
        }

    return pd.DataFrame(metrics).T.reindex(returns_df.columns).fillna(0.0)


def calculate_quality_score(
    returns_df: pd.DataFrame,
    risk_free_daily: float,
    downside_df: pd.DataFrame,
) -> pd.Series:
    sharpe = ((returns_df.mean() - risk_free_daily) / (returns_df.std() + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe_norm = _minmax_scale(sharpe)

    downside_inv = (1.0 / (1.0 + downside_df["downside_ratio"].astype(float))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    downside_norm = _minmax_scale(downside_inv)

    stability_score = (1.0 / (1.0 + downside_df["stability_raw"].astype(float) * 50.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stability_norm = _minmax_scale(stability_score)

    consistency_norm = _minmax_scale(downside_df["consistency"].astype(float).fillna(0.0))

    quality = (
        0.35 * sharpe_norm
        + 0.25 * downside_norm
        + 0.20 * stability_norm
        + 0.20 * consistency_norm
    ).clip(0.0, 1.0)
    return quality.reindex(returns_df.columns).fillna(0.5)


CORE_TICKERS = [
    "AAPL", "NVDA", "MSTR",
    "SNAP", "OKLO", "JMIA", "XYZ", "RKLB", "RBLX",
    "AVGO", "INTC",
    "TWLO", "SPOT", "TEAM", "SPCE", "SNOW",
    "TSM", "LRCX", "TSLA", "ASTS", "RGTI", "KEEL",
]


GLOBAL_IMPORTANT_TICKERS = [
    "CCJ", "YPF", "VIST",
    "ASML", "ARM", "TSM", "BABA", "TCEHY", "TM", "SONY",
    "NVO", "SAP", "SHEL", "BP", "RIO", "BHP", "VALE",
    "MELI", "SHOP", "SE", "NU", "PBR", "EC", "GLOB",
    "UBER", "COIN", "PLTR", "AMD", "SMCI", "MU", "NET",
]


NASDAQ_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
    "TSLA", "COST", "NFLX", "ASML", "TMUS", "CSCO", "PEP", "AMD",
    "AZN", "LIN", "INTU", "QCOM", "TXN", "AMGN", "ISRG", "BKNG",
    "AMAT", "ADBE", "PDD", "ARM", "HON", "GILD", "CMCSA", "PANW",
    "ADP", "VRTX", "SBUX", "MU", "MELI", "ADI", "LRCX", "KLAC",
    "CRWD", "MDLZ", "REGN", "CEG", "SNPS", "CDNS", "MAR", "ORLY",
    "CTAS", "DASH", "FTNT", "PYPL", "CSX", "ABNB", "ROP", "WDAY",
    "MNST", "ADSK", "AEP", "NXPI", "PAYX", "MRVL", "CHTR", "KDP",
    "PCAR", "ROST", "FAST", "ODFL", "CPRT", "DDOG", "TEAM", "EA",
    "KHC", "EXC", "BKR", "TTWO", "VRSK", "XEL", "ZS", "FANG",
    "CCEP", "GEHC", "IDXX", "MCHP", "CSGP", "DXCM", "ON", "ANSS",
    "BIIB", "CDW", "GFS", "ILMN", "MDB", "WBD", "MRNA", "SIRI",
]


NASDAQ_DEFAULT_LIMIT = 250
NASDAQ_EXCLUDED_SYMBOL_PATTERNS = (
    "$", "^", ".", "/", "=", "+",
    "W", "WS", "WT", "WTS", "U", "UN", "UNIT", "R", "RT", "RIGHT",
    "P", "PR", "PRA", "PRB", "PRC", "PRD", "PRE", "PRF", "PRG", "PRH",
)


def _dedupe_tickers(tickers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for ticker in tickers:
        symbol = str(ticker).strip().upper().replace(".", "-")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned


def _is_tradeable_common_symbol(symbol: str) -> bool:
    symbol = str(symbol).strip().upper()
    if not symbol or len(symbol) > 5:
        return False
    if any(token in symbol for token in ("$", "^", ".", "/", "=", "+")):
        return False
    if symbol.endswith(NASDAQ_EXCLUDED_SYMBOL_PATTERNS):
        return False
    return symbol.isalnum()


def fetch_nasdaq_listed_tickers(limit: int | None = NASDAQ_DEFAULT_LIMIT) -> list[str]:
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(url, timeout=20)
        response.raise_for_status()
        nasdaq_df = pd.read_csv(io.StringIO(response.text), sep="|")
        nasdaq_df = nasdaq_df[
            (nasdaq_df["Symbol"].astype(str) != "File Creation Time")
            & (nasdaq_df["Test Issue"].astype(str).str.upper() == "N")
            & (nasdaq_df["ETF"].astype(str).str.upper() == "N")
            & (nasdaq_df["Financial Status"].astype(str).str.upper() == "N")
        ]
        tickers = _dedupe_tickers(
            symbol for symbol in nasdaq_df["Symbol"].tolist() if _is_tradeable_common_symbol(symbol)
        )
        if limit is not None:
            tickers = tickers[: max(0, int(limit))]
        return tickers
    except Exception as exc:
        print(f"[WARNING] Could not fetch full Nasdaq universe. Using fallback list. Error: {exc}")
        fallback = NASDAQ_FALLBACK_TICKERS.copy()
        if limit is not None:
            fallback = fallback[: max(0, int(limit))]
        return fallback


def build_trading_universe(
    include_full_nasdaq: bool = True,
    nasdaq_limit: int | None = NASDAQ_DEFAULT_LIMIT,
) -> list[str]:
    nasdaq_tickers = (
        fetch_nasdaq_listed_tickers(limit=nasdaq_limit)
        if include_full_nasdaq
        else NASDAQ_FALLBACK_TICKERS[: max(0, int(nasdaq_limit or len(NASDAQ_FALLBACK_TICKERS)))]
    )
    return _dedupe_tickers([
        *CORE_TICKERS,
        *GLOBAL_IMPORTANT_TICKERS,
        *nasdaq_tickers,
    ])


def main() -> None:
    tickers = build_trading_universe(include_full_nasdaq=True, nasdaq_limit=NASDAQ_DEFAULT_LIMIT)
    model_mode = os.getenv("MODEL_MODE", "baseline").strip().lower() or "baseline"
    allowed_model_modes = {"baseline", "full_quant_research", "regime_gated_full_quant", "calibrated_forecast_research", "raw_target_research"}
    if model_mode not in allowed_model_modes:
        raise ValueError(f"Invalid model_mode={model_mode}. Allowed values: {sorted(allowed_model_modes)}")
    run_model_mode_comparison = os.getenv("RUN_MODEL_MODE_COMPARISON", "0").strip().lower() in {"1", "true", "yes", "on"}
    use_expected_returns = True
    use_live_prices = True
    center_expected_returns = False
    expected_returns_alpha = 0.5
    defensive_mode = True
    paper_trading_enabled = os.getenv("PAPER_TRADING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    paper_model_mode = os.getenv("PAPER_MODEL_MODE", "baseline").strip().lower() or "baseline"
    paper_overwrite_same_day = os.getenv("PAPER_OVERWRITE_SAME_DAY", "0").strip().lower() in {"1", "true", "yes", "on"}
    paper_meta_filter_enabled = os.getenv("PAPER_META_FILTER_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    paper_meta_filter_model = os.getenv("PAPER_META_FILTER_MODEL", "logistic_isotonic").strip().lower() or "logistic_isotonic"
    paper_meta_filter_threshold = float(os.getenv("PAPER_META_FILTER_THRESHOLD", "0.65"))
    use_walk_forward_calibrated_forecasts = (
        os.getenv("USE_WALK_FORWARD_CALIBRATED_FORECASTS", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    use_raw_target_return = os.getenv("USE_RAW_TARGET_RETURN", "0").strip().lower() in {"1", "true", "yes", "on"}
    calibrated_forecast_file = os.getenv(
        "WALK_FORWARD_CALIBRATED_FORECASTS_FILE",
        "walk_forward_calibrated_forecasts.csv",
    )
    calibrated_forecast_horizon_days = int(os.getenv("CALIBRATED_FORECAST_HORIZON_DAYS", "20"))
    calibrated_forecast_max_staleness_days = int(os.getenv("CALIBRATED_FORECAST_MAX_STALENESS_DAYS", "45"))
    optimizer_max_weight = 0.50
    exclude_bearish_assets = False
    signal_strength_threshold = 0.2
    no_opportunity_threshold = 0.001
    timeframe = "daily"
    timing_mode = "daily"
    use_trend_persistence_for_timing = False
    compact_report_mode = False
    report_sections_to_show = [
        "TREND_PERSISTENCE",
        "EMA_VS_TREND_PERSISTENCE",
        "BLACK_LITTERMAN",
        "INSTITUTIONAL_RISK",
        "ACTION_SIGNALS",
        "FINAL_ALLOCATION",
        "WALK_FORWARD",
        "TRIPLE_BARRIER",
    ]
    compact_report_env = os.getenv("COMPACT_REPORT_MODE", "").strip().lower()
    if compact_report_env in {"1", "true", "yes", "on"}:
        compact_report_mode = True
    report_sections_env = os.getenv("REPORT_SECTIONS_TO_SHOW", "").strip()
    if report_sections_env:
        report_sections_to_show = [
            section.strip().upper()
            for section in report_sections_env.split(",")
            if section.strip()
        ]
    if compact_report_mode:
        _install_compact_report_mode(
            report_sections_to_show,
            early_exit_after_final_allocation=not (
                set(report_sections_to_show) & LATE_COMPACT_REPORT_SECTIONS
            ),
        )
    def report_enabled(section_name: str) -> bool:
        return (not compact_report_mode) or (section_name in set(report_sections_to_show))
    post_timing_edge_threshold = 0.0002
    epsilon = 1e-6
    target_method = "basic"
    basic_target_generation_method = "volatility_adjusted"
    use_quant_target_blend = False
    quant_target_blend_weight = 0.15
    quant_target_horizon_days = 20
    covariance_method = "ledoit_wolf"
    shrinkage_intensity = 0.10
    should_run_walk_forward_backtest = False
    run_triple_barrier_labeling = False
    run_triple_barrier_feature_validation_flag = False
    run_regime_performance_attribution_flag = False
    run_barrier_parameter_optimization_flag = False
    run_threshold_optimization_flag = False
    run_robustness_validation_flag = False
    run_research_governance_flag = False
    run_research_dashboard = False
    run_overfitting_reduction_plan_flag = False
    run_larger_walk_forward_expansion_flag = False
    run_clean_research_evaluation_flag = False
    run_paper_trading_monitor_flag = False
    allow_manual_targets_for_expected_returns = False
    manual_long_term_targets = {
        "AAPL": 240.0,
        "NVDA": 1400.0,
        "MSTR": 2200.0,
    }
    manual_short_term_targets: dict[str, float] = {}
    if model_mode == "full_quant_research":
        use_trend_persistence_for_timing = True
        use_quant_target_blend = False
    if model_mode == "calibrated_forecast_research":
        use_walk_forward_calibrated_forecasts = True
    if model_mode == "raw_target_research":
        use_raw_target_return = True

    # Choose only one option:
    period = None
    days = lookback_days

    try:
        prices_df = download_close_prices(tickers=tickers, period=period, days=days)
        if should_run_walk_forward_backtest:
            from walk_forward_backtester import WalkForwardConfig, run_walk_forward_backtest

            run_walk_forward_backtest(
                prices_df,
                config=WalkForwardConfig(),
            )
            return
        if run_triple_barrier_labeling:
            from triple_barrier_labeling import generate_triple_barrier_labels

            generate_triple_barrier_labels(prices_df=prices_df)
            return
        if run_triple_barrier_feature_validation_flag:
            from triple_barrier_feature_validation import run_triple_barrier_feature_validation

            run_triple_barrier_feature_validation()
            return
        if run_regime_performance_attribution_flag:
            from regime_performance_attribution import run_regime_performance_attribution

            run_regime_performance_attribution()
            return
        if run_barrier_parameter_optimization_flag:
            from barrier_parameter_optimization import run_barrier_parameter_optimization

            run_barrier_parameter_optimization(prices_df=prices_df)
            return
        if run_threshold_optimization_flag:
            from threshold_optimization import run_threshold_optimization

            run_threshold_optimization()
            return
        if run_robustness_validation_flag:
            from robustness_validation import run_robustness_validation

            run_robustness_validation()
            return
        if run_research_governance_flag:
            from research_governance import run_research_governance_report

            run_research_governance_report()
            return
        if run_research_dashboard:
            from research_dashboard import run_research_dashboard as _run_research_dashboard

            _run_research_dashboard(compact=compact_report_mode)
            return
        if run_overfitting_reduction_plan_flag:
            from overfitting_reduction_plan import run_overfitting_reduction_plan

            run_overfitting_reduction_plan()
            return
        if run_larger_walk_forward_expansion_flag:
            from larger_walk_forward_expansion import run_larger_walk_forward_expansion

            run_larger_walk_forward_expansion(prices_df=prices_df)
            return
        if run_clean_research_evaluation_flag:
            from clean_research_evaluation import run_clean_research_evaluation

            run_clean_research_evaluation()
            return
        if run_paper_trading_monitor_flag:
            from paper_trading_monitor import run_paper_trading_monitor

            run_paper_trading_monitor()
            return
        if run_model_mode_comparison:
            from model_mode_comparison import run_model_mode_comparison as _run_model_mode_comparison

            _run_model_mode_comparison(prices_df=prices_df)
            return
        if use_live_prices:
            print(
                f"[INFO] Live price mode enabled for {len(prices_df.columns)} tickers. "
                "Missing live prices will use last historical close."
            )
        advanced_target_outputs: dict[str, pd.Series] | None = None
        if target_method == "advanced":
            advanced_target_outputs = generate_targets_advanced(prices_df)
            target_prices = advanced_target_outputs["target_price"]
        else:
            target_prices = generate_target_prices(
                prices_df,
                method=basic_target_generation_method,
                manual_targets=(
                    manual_short_term_targets
                    if allow_manual_targets_for_expected_returns
                    else None
                ),
            )
        returns_df = calculate_daily_returns(prices_df)
        prices_df = prices_df[returns_df.columns]
        target_prices = pd.Series(target_prices, dtype=float).reindex(prices_df.columns)
        manual_long_term_target_series = (
            pd.Series(manual_long_term_targets, dtype=float)
            .reindex(prices_df.columns)
        )
        manual_target_gap_pct = (
            manual_long_term_target_series / prices_df.ffill().iloc[-1].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        market_regime = compute_market_regime_model(prices_df=prices_df, returns_df=returns_df)
        regime_score = float(market_regime["risk_score"])
        regime_type = str(market_regime["regime"])
        regime_confidence = float(market_regime.get("regime_confidence", 0.3))
        quant_target_outputs = generate_quant_targets(
            prices_df=prices_df,
            old_target_prices=target_prices,
            regime_type=regime_type,
            horizon_days=quant_target_horizon_days,
            blend_weight=quant_target_blend_weight,
        )
        full_quant_gate = {
            "gate_decision": "not_applicable",
            "allow_full_quant": False,
            "reason": "mode_not_regime_gated",
            "regime": regime_type,
            "market_mode": "unknown",
            "spy_macro_regime": "unknown",
            "regime_confidence": regime_confidence,
            "regime_score": regime_score,
            "volatility_condition": market_regime.get("vol_regime_ratio", np.nan),
            "vix_z": market_regime.get("vix_z", np.nan),
            "breadth": market_regime.get("breadth", np.nan),
            "spy_momentum_20d": market_regime.get("spy_momentum_20d", np.nan),
            "average_entropy": np.nan,
            "average_trend_persistence_score": np.nan,
        }
        if model_mode == "regime_gated_full_quant":
            full_quant_gate = evaluate_full_quant_regime_gate(
                regime=regime_type,
                market_mode="aggressive" if regime_type == "risk_on" and regime_confidence > 0.5 else "neutral",
                regime_confidence=regime_confidence,
                regime_score=regime_score,
                volatility_condition=market_regime.get("vol_regime_ratio", np.nan),
                vix_z=market_regime.get("vix_z", np.nan),
                breadth=market_regime.get("breadth", np.nan),
                spy_momentum_20d=market_regime.get("spy_momentum_20d", np.nan),
            )
            use_trend_persistence_for_timing = bool(full_quant_gate["allow_full_quant"])
        target_source_used = (
            "quant_target_price"
            if model_mode == "full_quant_research"
            or (model_mode == "regime_gated_full_quant" and bool(full_quant_gate["allow_full_quant"]))
            else ("quant_blended_target" if use_quant_target_blend else "baseline_target")
        )
        quant_target_decision_status = (
            "active_primary"
            if model_mode == "full_quant_research"
            or (model_mode == "regime_gated_full_quant" and bool(full_quant_gate["allow_full_quant"]))
            else ("active_blend" if use_quant_target_blend else "diagnostic")
        )
        timing_source_used = "trend_persistence" if use_trend_persistence_for_timing else "ema_timing"
        expected_return_source_used = f"compute_expected_returns({target_source_used})"
        if use_raw_target_return:
            expected_return_source_used = "compute_expected_returns(raw_target_return_mode)"
        black_litterman_status = "diagnostic"
        if model_mode == "full_quant_research" or (
            model_mode == "regime_gated_full_quant" and bool(full_quant_gate["allow_full_quant"])
        ):
            target_prices = quant_target_outputs["quant_target_price"].reindex(prices_df.columns)
        elif use_quant_target_blend:
            target_prices = quant_target_outputs["final_blended_target"].reindex(prices_df.columns)
        covariance_matrix = calculate_covariance_matrix(
            returns_df,
            method=covariance_method,
            shrinkage_intensity=shrinkage_intensity,
        )
        covariance_metadata = covariance_method_metadata(
            returns_df,
            method=covariance_method,
            shrinkage_intensity=shrinkage_intensity,
        )
        volatility = calculate_volatility(returns_df)
        n_assets = returns_df.shape[1]
        weights = np.ones(n_assets) / n_assets
        rf_annual, rf_daily = get_risk_free_rate()
        expected_daily_returns, expected_diagnostics = compute_expected_returns(
            prices_df,
            target_prices,
            use_live_prices=use_live_prices,
            risk_free_daily=rf_daily,
            center_distribution=center_expected_returns,
            dead_zone=0.001,
            regime_score=regime_score,
            regime_type=regime_type,
            regime_signal_threshold=0.2,
            use_raw_target_return=use_raw_target_return,
        )
        if target_method == "advanced" and advanced_target_outputs is not None:
            target_confidence = advanced_target_outputs["target_confidence"].reindex(prices_df.columns).fillna(0.5)
            current_prices = prices_df.ffill().iloc[-1].reindex(prices_df.columns).replace(0, np.nan).fillna(1.0)
            expected_daily_returns = ((target_prices / current_prices) - 1) * target_confidence
            expected_daily_returns = expected_daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            for ticker in prices_df.columns:
                if ticker not in expected_diagnostics:
                    expected_diagnostics[ticker] = {}
                expected_diagnostics[ticker]["target_confidence"] = float(target_confidence.loc[ticker])
                expected_diagnostics[ticker]["target_validity"] = str(advanced_target_outputs["target_validity"].loc[ticker])
                expected_diagnostics[ticker]["expected_move"] = float(advanced_target_outputs["expected_move"].loc[ticker])
                expected_diagnostics[ticker]["upper_1sigma"] = float(advanced_target_outputs["upper_1sigma"].loc[ticker])
                expected_diagnostics[ticker]["upper_2sigma"] = float(advanced_target_outputs["upper_2sigma"].loc[ticker])
                expected_diagnostics[ticker]["lower_1sigma"] = float(advanced_target_outputs["lower_1sigma"].loc[ticker])
                adjusted_expected = float(expected_daily_returns.loc[ticker])
                target_validity = str(advanced_target_outputs["target_validity"].loc[ticker]).lower()
                trend_alignment = float(expected_diagnostics[ticker].get("trend_alignment", 0.0))
                momentum = float(expected_diagnostics[ticker].get("momentum", 0.0))

                if target_validity != "valid":
                    adjusted_expected *= 0.3
                if trend_alignment < 0:
                    adjusted_expected *= 0.5
                if momentum < 0:
                    adjusted_expected *= 0.7
                if abs(adjusted_expected) < 0.001:
                    adjusted_expected = 0.0
                signal_strength = float(expected_diagnostics[ticker].get("signal_strength", 0.0))
                if signal_strength > 0.2:
                    if regime_type == "high_volatility":
                        adjusted_expected *= 1.2
                    elif regime_type == "low_volatility":
                        adjusted_expected *= 1.1
                    else:
                        adjusted_expected *= 0.7

                expected_daily_returns.loc[ticker] = adjusted_expected
                expected_diagnostics[ticker]["expected_daily_return"] = adjusted_expected

        for ticker in prices_df.columns:
            if ticker not in expected_diagnostics:
                expected_diagnostics[ticker] = {}
            manual_long_term_target = manual_long_term_target_series.get(ticker, np.nan)
            expected_diagnostics[ticker]["manual_long_term_target"] = (
                float(manual_long_term_target) if pd.notna(manual_long_term_target) else np.nan
            )
            expected_diagnostics[ticker]["manual_target_gap_pct"] = (
                float(manual_target_gap_pct.get(ticker, np.nan))
                if pd.notna(manual_target_gap_pct.get(ticker, np.nan))
                else np.nan
            )
            expected_diagnostics[ticker]["manual_targets_allowed_for_expected_returns"] = bool(
                allow_manual_targets_for_expected_returns
            )
            expected_diagnostics[ticker]["old_target_price"] = float(quant_target_outputs["old_target_price"].loc[ticker])
            expected_diagnostics[ticker]["quant_target_price"] = float(quant_target_outputs["quant_target_price"].loc[ticker])
            expected_diagnostics[ticker]["target_blend_weight"] = float(
                quant_target_outputs["target_blend_weight"].loc[ticker] if use_quant_target_blend else 0.0
            )
            expected_diagnostics[ticker]["gbm_median_target"] = float(quant_target_outputs["gbm_median_target"].loc[ticker])
            expected_diagnostics[ticker]["gbm_expected_target"] = float(quant_target_outputs["gbm_expected_target"].loc[ticker])
            expected_diagnostics[ticker]["gbm_target"] = float(quant_target_outputs["gbm_median_target"].loc[ticker])
            expected_diagnostics[ticker]["kalman_target"] = float(quant_target_outputs["kalman_target"].loc[ticker])
            expected_diagnostics[ticker]["ou_target"] = float(quant_target_outputs["ou_target"].loc[ticker])
            expected_diagnostics[ticker]["target_confidence_quant"] = float(
                quant_target_outputs["target_confidence"].loc[ticker]
            )
            expected_diagnostics[ticker]["target_method_selected"] = str(
                quant_target_outputs["target_method_selected"].loc[ticker]
            )
            expected_diagnostics[ticker]["quant_target_blend_enabled"] = bool(use_quant_target_blend)

        raw_expected_returns = expected_daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        norm_mean_before = float(raw_expected_returns.mean())
        norm_std_before = float(raw_expected_returns.std()) if len(raw_expected_returns) > 1 else 0.0
        if norm_std_before > 0:
            normalized_returns = (raw_expected_returns - norm_mean_before) / norm_std_before
        else:
            normalized_returns = raw_expected_returns * 0.0

        normalized_returns = normalized_returns.clip(-2.5, 2.5)
        scaled_returns = normalized_returns * 0.003

        target_confidence_series = pd.Series(
            {
                ticker: float(expected_diagnostics.get(ticker, {}).get("target_confidence", np.nan))
                for ticker in prices_df.columns
            },
            index=prices_df.columns,
            dtype=float,
        )
        signal_confidence_series = pd.Series(
            {
                ticker: float(expected_diagnostics.get(ticker, {}).get("signal_strength", 0.5))
                for ticker in prices_df.columns
            },
            index=prices_df.columns,
            dtype=float,
        )
        confidence_series = target_confidence_series.fillna(signal_confidence_series).clip(0.0, 1.0)
        scaled_returns *= (0.5 + 0.5 * confidence_series)
        scaled_returns.loc[scaled_returns < -0.003] *= 1.2

        if regime_type == "risk_off":
            scaled_returns *= 0.6
        elif regime_type == "neutral":
            scaled_returns *= 0.85
        elif regime_type == "risk_on":
            scaled_returns *= 1.1

        expected_daily_returns = scaled_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for ticker in prices_df.columns:
            if ticker not in expected_diagnostics:
                expected_diagnostics[ticker] = {}
            expected_diagnostics[ticker]["expected_daily_return"] = float(expected_daily_returns.loc[ticker])

        norm_mean_after = float(expected_daily_returns.mean())
        norm_std_after = float(expected_daily_returns.std()) if len(expected_daily_returns) > 1 else 0.0
        norm_min_after = float(expected_daily_returns.min()) if len(expected_daily_returns) else 0.0
        norm_max_after = float(expected_daily_returns.max()) if len(expected_daily_returns) else 0.0
        norm_positive_pct = float((expected_daily_returns > 0).mean()) if len(expected_daily_returns) else 0.0
        norm_negative_pct = float((expected_daily_returns < 0).mean()) if len(expected_daily_returns) else 0.0

        diagnostics_df_full = pd.DataFrame(expected_diagnostics).T
        downside_df = calculate_downside_metrics(returns_df, window=30)
        quality_score_all = calculate_quality_score(returns_df, rf_daily, downside_df).reindex(prices_df.columns).fillna(0.5)

        downside_penalty_input = (
            0.45 * downside_df["downside_ratio"].reindex(prices_df.columns).fillna(0.0)
            + 0.35 * downside_df["recent_drawdown"].reindex(prices_df.columns).fillna(0.0)
            + 0.20 * downside_df["return_skew"].reindex(prices_df.columns).fillna(0.0).clip(upper=0.0).abs()
        )
        downside_risk_norm = _minmax_scale(downside_penalty_input.reindex(prices_df.columns).fillna(0.0))
        downside_penalty_multiplier_all = (1.05 - 0.20 * downside_risk_norm).clip(0.85, 1.05)
        downside_penalty_multiplier_all = downside_penalty_multiplier_all.reindex(prices_df.columns).fillna(1.0)
        expected_before_downside_all = expected_daily_returns.reindex(prices_df.columns).fillna(0.0)
        expected_daily_returns = (expected_before_downside_all * downside_penalty_multiplier_all).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        downside_rank_changed = (
            expected_before_downside_all.sort_values(ascending=False).index.tolist()
            != expected_daily_returns.sort_values(ascending=False).index.tolist()
        )
        downside_std_before = float(expected_before_downside_all.std()) if len(expected_before_downside_all) > 1 else 0.0
        downside_std_after = float(expected_daily_returns.std()) if len(expected_daily_returns) > 1 else 0.0

        for ticker in prices_df.columns:
            if ticker not in expected_diagnostics:
                expected_diagnostics[ticker] = {}
            expected_diagnostics[ticker]["downside_volatility"] = float(downside_df.loc[ticker, "downside_volatility"])
            expected_diagnostics[ticker]["downside_ratio"] = float(downside_df.loc[ticker, "downside_ratio"])
            expected_diagnostics[ticker]["recent_drawdown"] = float(downside_df.loc[ticker, "recent_drawdown"])
            expected_diagnostics[ticker]["return_skew"] = float(downside_df.loc[ticker, "return_skew"])
            expected_diagnostics[ticker]["quality_score"] = float(quality_score_all.loc[ticker])
            expected_diagnostics[ticker]["downside_multiplier"] = float(downside_penalty_multiplier_all.loc[ticker])
            expected_diagnostics[ticker]["expected_daily_return"] = float(expected_daily_returns.loc[ticker])

        diagnostics_df_full = pd.DataFrame(expected_diagnostics).T.reindex(prices_df.columns)
        diagnostics_critical_cols = [
            "quality_score",
            "downside_ratio",
            "recent_drawdown",
            "return_skew",
            "downside_volatility",
            "downside_multiplier",
            "garch_volatility",
            "egarch_volatility",
            "kalman_residual_vol",
            "savgol_slope",
            "fft_low_freq_energy",
            "haar_wavelet_energy",
            "hurst_exponent",
            "ou_half_life",
            "hawkes_downside_intensity",
            "entropy",
            "fractal_dimension",
            "lyapunov_proxy",
            "hill_tail_index",
            "quant_market_quality",
            "quant_multiplier",
        ]
        diagnostics_missing_detected = [col for col in diagnostics_critical_cols if col not in diagnostics_df_full.columns]
        diagnostics_created_defaults: list[str] = []
        for col in diagnostics_critical_cols:
            if col not in diagnostics_df_full.columns:
                diagnostics_df_full[col] = np.nan
                diagnostics_created_defaults.append(col)

        diagnostics_df_full["quality_score"] = (
            diagnostics_df_full["quality_score"]
            .reindex(prices_df.columns)
            .combine_first(quality_score_all.reindex(prices_df.columns))
        )
        diagnostics_df_full["downside_ratio"] = (
            diagnostics_df_full["downside_ratio"]
            .reindex(prices_df.columns)
            .combine_first(downside_df["downside_ratio"].reindex(prices_df.columns))
        )
        diagnostics_df_full["recent_drawdown"] = (
            diagnostics_df_full["recent_drawdown"]
            .reindex(prices_df.columns)
            .combine_first(downside_df["recent_drawdown"].reindex(prices_df.columns))
        )
        diagnostics_df_full["return_skew"] = (
            diagnostics_df_full["return_skew"]
            .reindex(prices_df.columns)
            .combine_first(downside_df["return_skew"].reindex(prices_df.columns))
        )
        diagnostics_df_full["downside_volatility"] = (
            diagnostics_df_full["downside_volatility"]
            .reindex(prices_df.columns)
            .combine_first(downside_df["downside_volatility"].reindex(prices_df.columns))
        )

        diagnostics_nan_counts = {
            col: int(diagnostics_df_full[col].isna().sum()) if col in diagnostics_df_full.columns else -1
            for col in diagnostics_critical_cols
        }
        diagnostics_shape_final = diagnostics_df_full.shape
        raw_target_research_report = pd.DataFrame()
        raw_target_research_metadata = {
            "raw_target_return_used": bool(use_raw_target_return),
            "signal_strength_adjustment_disabled": bool(use_raw_target_return),
            "regime_adjustment_status": "disabled" if use_raw_target_return else "production_default",
            "final_scaling_status": "production_default_downstream_scaling",
            "fallback_count": 0,
            "average_delta_vs_baseline_expected_return": 0.0,
        }
        if use_raw_target_return:
            baseline_expected_proxy = pd.Series(
                {
                    ticker: float(expected_diagnostics.get(ticker, {}).get("baseline_adjusted_expected_daily_return", np.nan))
                    for ticker in prices_df.columns
                },
                dtype=float,
            ).reindex(prices_df.columns)
            raw_expected_proxy = pd.Series(
                {
                    ticker: float(expected_diagnostics.get(ticker, {}).get("raw_target_expected_daily_return", np.nan))
                    for ticker in prices_df.columns
                },
                dtype=float,
            ).reindex(prices_df.columns)
            current_expected = expected_daily_returns.reindex(prices_df.columns).fillna(0.0)
            fallback_mask = raw_expected_proxy.isna()
            raw_target_research_report = pd.DataFrame(
                {
                    "baseline_pipeline_expected_return": baseline_expected_proxy,
                    "raw_target_expected_return": raw_expected_proxy,
                    "current_expected_return": current_expected,
                    "delta_vs_baseline": current_expected - baseline_expected_proxy.fillna(current_expected),
                    "fallback_to_baseline": fallback_mask,
                }
            )
            raw_target_research_metadata["fallback_count"] = int(fallback_mask.sum())
            raw_target_research_metadata["average_delta_vs_baseline_expected_return"] = float(
                raw_target_research_report["delta_vs_baseline"].abs().mean()
            )
        calibrated_forecast_report = pd.DataFrame()
        calibrated_forecast_metadata: dict[str, object] = {
            "calibrated_forecasts_used": 0,
            "fallback_original_forecasts": int(len(expected_daily_returns)),
            "calibrated_confidence_used": 0,
            "average_calibration_adjustment": 0.0,
            "failure_reason": "disabled",
        }
        if use_walk_forward_calibrated_forecasts:
            expected_daily_returns, diagnostics_df_full, calibrated_forecast_report, calibrated_forecast_metadata = (
                apply_walk_forward_calibrated_forecasts(
                    expected_daily_returns=expected_daily_returns.reindex(prices_df.columns).fillna(0.0),
                    diagnostics_df=diagnostics_df_full,
                    current_date=prices_df.index[-1],
                    config=CalibratedForecastConfig(
                        forecast_file=calibrated_forecast_file,
                        horizon_days=calibrated_forecast_horizon_days,
                        max_staleness_days=calibrated_forecast_max_staleness_days,
                    ),
                )
            )
            expected_return_source_used = (
                "walk_forward_calibrated_forecasts"
                if int(calibrated_forecast_metadata.get("calibrated_forecasts_used", 0)) > 0
                else "baseline_expected_returns(fallback_from_calibrated_mode)"
            )
            for ticker in prices_df.columns:
                if ticker not in expected_diagnostics:
                    expected_diagnostics[ticker] = {}
                expected_diagnostics[ticker]["expected_daily_return"] = float(
                    expected_daily_returns.reindex([ticker]).fillna(0.0).iloc[0]
                )
                if "target_confidence" in diagnostics_df_full.columns:
                    conf_value = diagnostics_df_full["target_confidence"].reindex([ticker]).iloc[0]
                    if pd.notna(conf_value):
                        expected_diagnostics[ticker]["target_confidence"] = float(conf_value)

        status_series = pd.Series(
            {ticker: expected_diagnostics[ticker]["status"] for ticker in prices_df.columns}
        )
        optimization_universe = status_series[status_series != "unreachable_target"].index.tolist()
        if exclude_bearish_assets:
            optimization_universe = [
                ticker for ticker in optimization_universe if status_series[ticker] != "bearish_or_low_probability"
            ]
        optimization_universe = [
            ticker
            for ticker in optimization_universe
            if float(diagnostics_df_full.loc[ticker, "signal_strength"]) >= signal_strength_threshold
        ]
        if not optimization_universe:
            optimization_universe = list(prices_df.columns)

        positive_returns = pd.Series(
            [float(expected_daily_returns.loc[ticker]) for ticker in optimization_universe if float(expected_daily_returns.loc[ticker]) > 0],
            dtype=float,
        )
        if len(positive_returns) > 0:
            edge_threshold = max(0.0001, float(positive_returns.quantile(0.4)))
        else:
            edge_threshold = 0.0001
        if regime_type == "risk_off":
            edge_threshold *= 1.3
        elif regime_type == "risk_on":
            edge_threshold *= 0.85
        signal_edge_threshold = 0.15
        fallback_activated = False
        fallback_reason = "none"

        expected_series = expected_daily_returns.reindex(optimization_universe).astype(float).fillna(0.0)
        signal_series = diagnostics_df_full.loc[optimization_universe, "signal_strength"].astype(float).fillna(0.0)
        volatility_series_selection = volatility.reindex(optimization_universe).astype(float).fillna(0.0)
        risk_adjusted_return_series = (expected_series / (volatility_series_selection + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        quality_series_selection = quality_score_all.reindex(optimization_universe).astype(float).fillna(0.5)
        target_confidence_series = pd.Series(
            [float(expected_diagnostics.get(ticker, {}).get("target_confidence", np.nan)) for ticker in optimization_universe],
            index=optimization_universe,
            dtype=float,
        ).fillna(0.5)
        asset_sharpe_for_selection = (
            (returns_df.mean() - rf_daily) / returns_df.std()
        ).replace([np.inf, -np.inf], np.nan).reindex(optimization_universe).fillna(0.0)

        expected_rank = expected_series.rank(pct=True, method="average")
        signal_rank = signal_series.rank(pct=True, method="average")
        confidence_rank = target_confidence_series.rank(pct=True, method="average")
        sharpe_rank = asset_sharpe_for_selection.rank(pct=True, method="average")
        risk_adjusted_rank = risk_adjusted_return_series.rank(pct=True, method="average")
        quality_rank = quality_series_selection.rank(pct=True, method="average")

        base_selection_score = (
            0.40 * expected_rank
            + 0.24 * signal_rank
            + 0.14 * confidence_rank
            + 0.08 * sharpe_rank
        ).fillna(0.0)
        enriched_selection_score = (
            0.36 * expected_rank
            + 0.20 * signal_rank
            + 0.14 * confidence_rank
            + 0.10 * sharpe_rank
            + 0.10 * risk_adjusted_rank
            + 0.10 * quality_rank
        ).fillna(0.0)

        selection_score_series = (0.65 * base_selection_score + 0.35 * enriched_selection_score).fillna(0.0)

        positive_expected = expected_series[expected_series > 0]
        if not positive_expected.empty:
            expected_median_positive = float(positive_expected.median())
            signal_median_positive = float(signal_series.reindex(positive_expected.index).median())
        else:
            expected_median_positive = 0.0
            signal_median_positive = float(signal_series.median()) if len(signal_series) else 0.0

        score_threshold = float(selection_score_series.quantile(0.70)) if len(selection_score_series) else 1.0
        candidate_mask = (
            (selection_score_series >= score_threshold)
            | (
                (expected_series > expected_median_positive)
                & (signal_series > signal_median_positive)
            )
        )
        candidate_tickers = selection_score_series[candidate_mask].sort_values(ascending=False).index.tolist()

        positive_real_count = int((expected_series > 0).sum())
        positive_edge_count = int((expected_series > edge_threshold).sum())

        if len(candidate_tickers) >= 4:
            selected_tickers = candidate_tickers[:4]
        elif len(candidate_tickers) >= 2:
            selected_tickers = candidate_tickers[:]
        else:
            selected_tickers = candidate_tickers[:]
        selected_n_pre_fallback = len(selected_tickers)

        if len(selected_tickers) < 2:
            fallback_ranked = selection_score_series.sort_values(ascending=False).index.tolist()
            for ticker in fallback_ranked:
                if ticker in selected_tickers:
                    continue
                selected_tickers.append(ticker)
                if len(selected_tickers) >= 2:
                    break
            fallback_activated = True
            fallback_reason = "candidate_count < 2"

        selected_n_post_fallback = len(selected_tickers)
        if selected_n_post_fallback == selected_n_pre_fallback:
            fallback_reason = "none"

        if len(selected_tickers) > 4:
            selected_tickers = selected_tickers[:4]
            selected_n_post_fallback = len(selected_tickers)

        discarded_low_edge = [
            ticker
            for ticker in optimization_universe
            if ticker not in selected_tickers
            and (
                float(expected_daily_returns.loc[ticker]) <= edge_threshold
                or float(diagnostics_df_full.loc[ticker, "signal_strength"]) <= signal_edge_threshold
            )
        ]
        discarded_assets = [ticker for ticker in optimization_universe if ticker not in selected_tickers]

        filtered_returns_df = returns_df[selected_tickers]
        filtered_expected_daily_returns = expected_daily_returns[selected_tickers]
        selected_prices_df = prices_df[selected_tickers]
        timing_df = compute_asset_timing(selected_prices_df, timing_mode=timing_mode)
        timing_adjusted_expected_returns = apply_timing_to_expected_returns(
            adjusted_expected_returns=filtered_expected_daily_returns,
            timing_df=timing_df,
        )
        spy_ema_regime = compute_spy_ema_regime(timing_mode=timing_mode)
        spy_macro_regime = str(spy_ema_regime["spy_macro_regime"])
        spy_macro_score = float(spy_ema_regime.get("macro_ema_score", 0.5))
        timing_adjusted_expected_returns = (
            timing_adjusted_expected_returns * (0.7 + 0.6 * spy_macro_score)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        trend_persistence_df = compute_trend_persistence(
            prices_df=selected_prices_df,
            returns_df=filtered_returns_df,
            selected_tickers=selected_tickers,
            diagnostics_df=diagnostics_df_full.reindex(selected_tickers),
            market_regime=market_regime,
        )
        ema_trend_persistence_comparison = build_ema_trend_persistence_comparison(
            timing_df=timing_df,
            trend_persistence_df=trend_persistence_df,
        )
        if use_trend_persistence_for_timing:
            timing_adjusted_expected_returns = (
                apply_trend_persistence_to_expected_returns(
                    adjusted_expected_returns=filtered_expected_daily_returns,
                    trend_persistence_df=trend_persistence_df,
                )
                * (0.7 + 0.6 * spy_macro_score)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_expected_returns_for_selection = timing_adjusted_expected_returns.copy()
        final_signal_strengths_for_selection = (
            diagnostics_df_full.loc[final_expected_returns_for_selection.index, "signal_strength"]
            .astype(float)
            .fillna(0.0)
        )
        filtered_expected_daily_returns_pre_timing = filtered_expected_daily_returns.copy()
        selection_score_selected = selection_score_series.reindex(selected_tickers).fillna(0.0)
        timing_df["selection_score"] = selection_score_selected
        timing_df["timing_adjusted_return"] = timing_adjusted_expected_returns.reindex(timing_df.index).fillna(0.0)
        for trend_col in [
            "trend_persistence_score",
            "trend_persistence_confidence",
            "trend_persistence_action",
            "trend_persistence_reason",
        ]:
            if trend_col in trend_persistence_df.columns:
                timing_df[trend_col] = trend_persistence_df[trend_col].reindex(timing_df.index)
        timing_integration_df = pd.DataFrame(
            {
                "adjusted_expected_before_timing": filtered_expected_daily_returns_pre_timing,
                "timing_adjusted_after": timing_adjusted_expected_returns,
            }
        )
        selected_n_natural = len(selected_tickers)
        if regime_type == "risk_on" and regime_confidence > 0.5:
            market_mode_pre_timing = "aggressive"
        elif regime_type == "risk_off" and regime_confidence > 0.5:
            market_mode_pre_timing = "defensive"
        else:
            market_mode_pre_timing = "neutral"

        positive_timing_returns = final_expected_returns_for_selection[final_expected_returns_for_selection > 0]
        post_timing_edge_threshold_raw = (
            max(post_timing_edge_threshold, float(positive_timing_returns.quantile(0.5)))
            if len(positive_timing_returns) > 0
            else post_timing_edge_threshold
        )
        if market_mode_pre_timing == "aggressive" and regime_type == "risk_on":
            post_timing_threshold_min = 0.00002
            post_timing_threshold_max = 0.00022
        elif market_mode_pre_timing == "neutral":
            post_timing_threshold_min = 0.00005
            post_timing_threshold_max = 0.00045
        else:
            post_timing_threshold_min = 0.00008
            post_timing_threshold_max = 0.00080

        positives_and_signal_pool = [
            ticker
            for ticker in selected_tickers
            if float(final_expected_returns_for_selection.reindex([ticker]).fillna(0.0).iloc[0]) > 0
            and float(diagnostics_df_full.loc[ticker, "signal_strength"]) > 0.20
        ]
        if (
            regime_type == "risk_on"
            and market_mode_pre_timing == "aggressive"
            and selected_n_natural >= 4
            and len(positives_and_signal_pool) >= 4
        ):
            post_timing_edge_threshold_raw *= 0.85
        dynamic_post_timing_threshold = float(
            np.clip(post_timing_edge_threshold_raw, post_timing_threshold_min, post_timing_threshold_max)
        )

        timing_adjusted_ranked = final_expected_returns_for_selection.sort_values(ascending=False).index.tolist()
        post_timing_removed_reasons: dict[str, str] = {}
        post_timing_selected_tickers = []
        for ticker in timing_adjusted_ranked:
            timed_ret = float(final_expected_returns_for_selection.loc[ticker])
            entry_valid_ticker = bool(timing_df.loc[ticker, "entry_valid"]) if ticker in timing_df.index else False
            timing_score_ticker = (
                float(timing_df.loc[ticker, "ema_timing_score"])
                if ticker in timing_df.index and "ema_timing_score" in timing_df.columns
                else 0.0
            )

            deterioration = timed_ret <= 0
            soft_valid = entry_valid_ticker and timed_ret > 0 and timing_score_ticker >= 0.25
            pass_threshold = timed_ret > dynamic_post_timing_threshold
            keep_asset = pass_threshold or soft_valid

            if keep_asset:
                post_timing_selected_tickers.append(ticker)
            else:
                if deterioration:
                    post_timing_removed_reasons[ticker] = "timing_adjusted_return<=0"
                elif not entry_valid_ticker:
                    post_timing_removed_reasons[ticker] = "entry_invalid_and_low_edge"
                else:
                    post_timing_removed_reasons[ticker] = "below_dynamic_post_timing_threshold"

        if not post_timing_selected_tickers:
            post_timing_selected_tickers = sorted(
                selected_tickers,
                key=lambda ticker: float(final_expected_returns_for_selection.loc[ticker]),
                reverse=True,
            )[: min(2, len(selected_tickers))]
            for ticker in selected_tickers:
                if ticker not in post_timing_selected_tickers and ticker not in post_timing_removed_reasons:
                    post_timing_removed_reasons[ticker] = "fallback_replacement"

        selected_n_after_post_timing = len(post_timing_selected_tickers)
        diversification_floor_used = False
        diversification_floor_reason = "none"

        if (
            selected_n_natural >= 4
            and regime_type == "risk_on"
            and market_mode_pre_timing == "aggressive"
        ):
            strong_entry_candidates = [
                ticker
                for ticker in selected_tickers
                if bool(timing_df.loc[ticker, "entry_valid"])
                and float(timing_adjusted_expected_returns.loc[ticker]) > 0
            ]
            if len(post_timing_selected_tickers) < 3 and len(strong_entry_candidates) >= 3:
                fill_ranked = sorted(
                    strong_entry_candidates,
                    key=lambda ticker: float(final_expected_returns_for_selection.loc[ticker]),
                    reverse=True,
                )
                for ticker in fill_ranked:
                    if ticker not in post_timing_selected_tickers:
                        post_timing_selected_tickers.append(ticker)
                    if len(post_timing_selected_tickers) >= 3:
                        break
                diversification_floor_used = True
                diversification_floor_reason = "risk_on_aggressive_natural4_guard"

        if regime_type == "risk_on" and market_mode_pre_timing == "aggressive":
            diversification_pool = [
                ticker
                for ticker in selected_tickers
                if bool(timing_df.loc[ticker, "entry_valid"])
                and float(timing_adjusted_expected_returns.loc[ticker]) > 0
                and float(diagnostics_df_full.loc[ticker, "signal_strength"]) > 0.2
            ]
            if len(diversification_pool) >= 3 and len(post_timing_selected_tickers) < 3:
                fill_ranked = sorted(
                    diversification_pool,
                    key=lambda ticker: float(final_expected_returns_for_selection.loc[ticker]),
                    reverse=True,
                )
                for ticker in fill_ranked:
                    if ticker not in post_timing_selected_tickers:
                        post_timing_selected_tickers.append(ticker)
                    if len(post_timing_selected_tickers) >= 3:
                        break
                diversification_floor_used = True
                diversification_floor_reason = "aggressive_min3_diversification_floor"

        if regime_type == "risk_on" and market_mode_pre_timing == "aggressive":
            preserve4_pool = [
                ticker
                for ticker in selected_tickers
                if float(final_expected_returns_for_selection.reindex([ticker]).fillna(0.0).iloc[0]) > 0
                and float(diagnostics_df_full.loc[ticker, "signal_strength"]) > 0.2
                and (
                    float(timing_df.loc[ticker, "ema_timing_score"])
                    if ticker in timing_df.index and "ema_timing_score" in timing_df.columns
                    else 0.0
                ) > 0.2
            ]
            if selected_n_natural >= 4 and len(preserve4_pool) >= 4 and len(post_timing_selected_tickers) < 4:
                fill_ranked = sorted(
                    preserve4_pool,
                    key=lambda ticker: float(final_expected_returns_for_selection.loc[ticker]),
                    reverse=True,
                )
                for ticker in fill_ranked:
                    if ticker not in post_timing_selected_tickers:
                        post_timing_selected_tickers.append(ticker)
                    if len(post_timing_selected_tickers) >= 4:
                        break
                diversification_floor_used = True
                diversification_floor_reason = "aggressive_preserve4_guard"

        post_timing_selected_tickers = sorted(
            post_timing_selected_tickers,
            key=lambda ticker: float(final_expected_returns_for_selection.loc[ticker]),
            reverse=True,
        )
        if len(post_timing_selected_tickers) > 4:
            for ticker in post_timing_selected_tickers[4:]:
                post_timing_removed_reasons[ticker] = "max_positions_cap_4"
            post_timing_selected_tickers = post_timing_selected_tickers[:4]

        fourth_asset_check = {
            "candidate": "none",
            "expected_return": 0.0,
            "signal_strength": 0.0,
            "diversification_benefit": 0.0,
            "accepted": "n/a",
            "reason": "not_applicable",
        }
        if len(post_timing_selected_tickers) == 4:
            core_assets = post_timing_selected_tickers[:3]
            fourth_candidate = post_timing_selected_tickers[3]
            fourth_return = float(final_expected_returns_for_selection.loc[fourth_candidate])
            fourth_signal = float(diagnostics_df_full.loc[fourth_candidate, "signal_strength"])
            corr_matrix = returns_df[post_timing_selected_tickers].corr().fillna(0.0)
            avg_abs_corr_with_core = float(
                corr_matrix.loc[fourth_candidate, core_assets].abs().mean()
            ) if core_assets else 1.0
            diversification_benefit = float(1.0 - avg_abs_corr_with_core)
            top3_returns = final_expected_returns_for_selection.reindex(core_assets).fillna(0.0).sort_values(ascending=False)
            top3_sum = float(top3_returns.sum())
            top3_concentration = float(top3_returns.iloc[0] / top3_sum) if top3_sum > 0 else 0.0

            cond_a = fourth_return > 0 and fourth_return >= (dynamic_post_timing_threshold * 0.9)
            cond_b = diversification_benefit > 0.35
            cond_c = top3_concentration > 0.55
            cond_d = fourth_signal > 0.20
            accept_fourth = bool((cond_a and cond_d) or cond_b or cond_c)

            fourth_asset_check = {
                "candidate": fourth_candidate,
                "expected_return": fourth_return,
                "signal_strength": fourth_signal,
                "diversification_benefit": diversification_benefit,
                "accepted": "yes" if accept_fourth else "no",
                "reason": "edge_or_diversification_or_concentration" if accept_fourth else "insufficient_edge_and_diversification",
            }

            if not accept_fourth:
                post_timing_selected_tickers = core_assets
                post_timing_removed_reasons[fourth_candidate] = "fourth_asset_rejected_by_quality_guard"

        post_timing_removed_assets = [ticker for ticker in selected_tickers if ticker not in post_timing_selected_tickers]
        post_timing_positive_before = int((final_expected_returns_for_selection > 0).sum())
        post_timing_positive_after = int(
            (final_expected_returns_for_selection.reindex(post_timing_selected_tickers).fillna(0.0) > 0).sum()
        )

        selected_tickers = post_timing_selected_tickers
        selected_n_post_threshold = len(selected_tickers)
        discarded_low_edge = [
            ticker
            for ticker in optimization_universe
                if ticker not in selected_tickers
                and (
                    float(final_expected_returns_for_selection.reindex([ticker]).fillna(0.0).iloc[0]) <= dynamic_post_timing_threshold
                    or float(diagnostics_df_full.loc[ticker, "signal_strength"]) <= signal_edge_threshold
                )
        ]
        discarded_assets = [ticker for ticker in optimization_universe if ticker not in selected_tickers]
        filtered_returns_df = returns_df[selected_tickers]
        returns_before_scaling = final_expected_returns_for_selection.reindex(selected_tickers).fillna(0.0).astype(float)
        filtered_signal_strengths = final_signal_strengths_for_selection.reindex(selected_tickers).fillna(0.0)

        scaling_method = "convex_power_tanh_plus_volatility"
        scaling_exponent = 1.25
        scaling_alpha = 0.45
        volatility_strength = 0.15
        volatility_adjustment_applied = True

        abs_returns = returns_before_scaling.abs()
        ref_scale = float(abs_returns.quantile(0.75)) if len(abs_returns) else 0.0
        ref_scale = max(ref_scale, 1e-6)

        convex_core = np.tanh(np.power(abs_returns / ref_scale, scaling_exponent))
        convex_multiplier = 1.0 + scaling_alpha * convex_core
        convex_multiplier = convex_multiplier.clip(lower=1.0, upper=2.5)
        scaled_returns_raw = returns_before_scaling * convex_multiplier

        selected_volatility = volatility.reindex(selected_tickers).astype(float)
        vol_median = float(selected_volatility.median()) if len(selected_volatility) else 0.0
        if not np.isfinite(vol_median) or vol_median <= 0:
            vol_median = 1e-6
        vol_relative = (selected_volatility / vol_median).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_multiplier = (1.0 + volatility_strength * (vol_relative - 1.0)).clip(lower=0.90, upper=1.25)
        scaled_returns_raw = scaled_returns_raw * vol_multiplier

        if len(scaled_returns_raw) > 2:
            lower_clip = float(scaled_returns_raw.quantile(0.02))
            upper_clip = float(scaled_returns_raw.quantile(0.98))
            scaled_returns_raw = scaled_returns_raw.clip(lower=lower_clip, upper=upper_clip)
        else:
            lower_clip = float(scaled_returns_raw.min()) if len(scaled_returns_raw) else 0.0
            upper_clip = float(scaled_returns_raw.max()) if len(scaled_returns_raw) else 0.0

        std_before_scaling = float(returns_before_scaling.std()) if len(returns_before_scaling) > 1 else 0.0
        std_after_scaling = float(scaled_returns_raw.std()) if len(scaled_returns_raw) > 1 else 0.0
        if std_after_scaling <= std_before_scaling and std_before_scaling > 0:
            center = float(scaled_returns_raw.mean())
            scaled_returns_raw = center + (scaled_returns_raw - center) * 1.20
            std_after_scaling = float(scaled_returns_raw.std()) if len(scaled_returns_raw) > 1 else 0.0

        rank_before = returns_before_scaling.sort_values(ascending=False).index.tolist()
        rank_after_raw = scaled_returns_raw.sort_values(ascending=False).index.tolist()
        ranking_unchanged = rank_before == rank_after_raw
        if not ranking_unchanged:
            sorted_scaled_values = np.sort(scaled_returns_raw.to_numpy())[::-1]
            scaled_returns = pd.Series(sorted_scaled_values, index=rank_before, dtype=float).reindex(selected_tickers)
        else:
            scaled_returns = scaled_returns_raw.reindex(selected_tickers).astype(float)

        marginal_threshold = 0.0005
        if len(scaled_returns) > 0:
            marginal_mask = returns_before_scaling.abs() < marginal_threshold
            scaled_returns.loc[marginal_mask] = np.sign(scaled_returns.loc[marginal_mask]) * np.minimum(
                scaled_returns.loc[marginal_mask].abs(),
                (returns_before_scaling.loc[marginal_mask].abs() * 1.15 + 1e-8),
            )

        final_expected_returns_for_selection = scaled_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pre_quality_expected_selected = final_expected_returns_for_selection.copy()
        selected_quality_score = quality_score_all.reindex(selected_tickers).astype(float).fillna(0.5)
        selected_downside_multiplier = downside_penalty_multiplier_all.reindex(selected_tickers).astype(float).fillna(1.0)
        quality_multiplier = (0.90 + 0.18 * selected_quality_score).clip(0.90, 1.08)
        combined_multiplier = (selected_downside_multiplier * quality_multiplier).clip(0.80, 1.12)

        dominance_guard_acted = False
        dominance_guard_assets: list[str] = []
        dominance_guard_min_multiplier = 0.92
        signal_for_guard = final_signal_strengths_for_selection.reindex(selected_tickers).fillna(0.0)
        if len(pre_quality_expected_selected) >= 2:
            base_rank_guard = pre_quality_expected_selected.sort_values(ascending=False)
            sig_rank_guard = signal_for_guard.reindex(base_rank_guard.index).fillna(0.0)
            top1_ticker = base_rank_guard.index[0]
            top2_ticker = base_rank_guard.index[1]
            top1_ret = float(base_rank_guard.iloc[0])
            top2_ret = float(base_rank_guard.iloc[1])
            top1_sig = float(sig_rank_guard.loc[top1_ticker])
            top2_sig = float(sig_rank_guard.loc[top2_ticker])
            clearly_dominant = (
                top1_ret > 0
                and (top1_ret - top2_ret) > max(0.00012, abs(top1_ret) * 0.20)
                and (top1_sig - top2_sig) > 0.10
            )
            if clearly_dominant:
                old_mult = float(combined_multiplier.loc[top1_ticker])
                new_mult = max(old_mult, dominance_guard_min_multiplier)
                if new_mult > old_mult:
                    combined_multiplier.loc[top1_ticker] = new_mult
                    dominance_guard_acted = True
                    dominance_guard_assets.append(top1_ticker)

        close_gap_threshold = max(0.00006, float(pre_quality_expected_selected.std()) * 0.18 if len(pre_quality_expected_selected) > 1 else 0.00006)
        final_expected_returns_for_selection = (
            final_expected_returns_for_selection * combined_multiplier
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        base_order_for_preservation = pre_quality_expected_selected.sort_values(ascending=False).index.tolist()
        final_preserved = final_expected_returns_for_selection.copy()
        preserve_eps = 1e-8
        for i in range(len(base_order_for_preservation) - 1):
            hi = base_order_for_preservation[i]
            lo = base_order_for_preservation[i + 1]
            base_hi = float(pre_quality_expected_selected.loc[hi])
            base_lo = float(pre_quality_expected_selected.loc[lo])
            if (base_hi - base_lo) > close_gap_threshold and float(final_preserved.loc[hi]) < float(final_preserved.loc[lo]):
                final_preserved.loc[hi] = float(final_preserved.loc[lo]) + preserve_eps
                preserve_eps *= 1.05
        final_expected_returns_for_selection = final_preserved.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        edge_base_order = pre_quality_expected_selected.sort_values(ascending=False).index.tolist()
        edge_final_order = final_expected_returns_for_selection.sort_values(ascending=False).index.tolist()
        edge_swaps_count = _count_ranking_swaps(edge_base_order, edge_final_order)
        edge_ranking_changed = edge_base_order != edge_final_order
        edge_impact_pct = (
            (final_expected_returns_for_selection - pre_quality_expected_selected)
            / (pre_quality_expected_selected.abs() + 1e-8)
            * 100.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        selection_filter_removed: dict[str, str] = {}
        if regime_type == "risk_on" and market_mode_pre_timing == "aggressive" and len(final_expected_returns_for_selection) >= 4:
            sorted_returns = final_expected_returns_for_selection.sort_values(ascending=False)
            top1_return = float(sorted_returns.iloc[0]) if len(sorted_returns) else 0.0
            system_vol_floor = float(final_expected_returns_for_selection.std()) if len(final_expected_returns_for_selection) > 1 else 0.0
            absolute_floor = max(0.00005, system_vol_floor * 0.15)
            relative_floor = max(0.0, top1_return * 0.07)
            combined_floor = max(absolute_floor, relative_floor)

            keep = []
            for idx, ticker in enumerate(sorted_returns.index.tolist()):
                ret_val = float(sorted_returns.loc[ticker])
                sig_val = float(final_signal_strengths_for_selection.reindex([ticker]).fillna(0.0).iloc[0])
                if idx < 3:
                    keep.append(ticker)
                    continue
                if ret_val <= 0:
                    selection_filter_removed[ticker] = "non_positive_expected_return"
                    continue
                if sig_val <= 0.20:
                    selection_filter_removed[ticker] = "low_signal_strength"
                    continue
                if ret_val < combined_floor:
                    selection_filter_removed[ticker] = "below_relative_absolute_floor"
                    continue
                keep.append(ticker)

            if len(keep) >= 3:
                final_expected_returns_for_selection = final_expected_returns_for_selection.reindex(keep).fillna(0.0)
                selected_tickers = keep
                filtered_returns_df = returns_df[selected_tickers]
                final_signal_strengths_for_selection = final_signal_strengths_for_selection.reindex(selected_tickers).fillna(0.0)
                timing_df = timing_df.reindex(selected_tickers)
        final_signal_strengths_for_selection = (
            final_signal_strengths_for_selection.reindex(final_expected_returns_for_selection.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        timing_df = timing_df.reindex(final_expected_returns_for_selection.index)
        timing_df["timing_action"] = timing_df.get("timing_action", pd.Series(index=timing_df.index, dtype=object)).fillna("hold")
        timing_df["timing_reason"] = timing_df.get("timing_reason", pd.Series(index=timing_df.index, dtype=object)).fillna("score_based_hold")

        final_series_frame = pd.DataFrame(
            {
                "expected": final_expected_returns_for_selection,
                "signal": final_signal_strengths_for_selection,
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        final_series_frame = final_series_frame.loc[
            [idx for idx in final_series_frame.index if idx in filtered_returns_df.columns]
        ]
        final_expected_returns_for_selection = final_series_frame["expected"].astype(float)
        final_signal_strengths_for_selection = final_series_frame["signal"].astype(float)
        selected_tickers = final_expected_returns_for_selection.index.tolist()
        filtered_returns_df = filtered_returns_df.reindex(columns=selected_tickers).dropna(how="any")
        final_expected_returns_for_selection = final_expected_returns_for_selection.reindex(filtered_returns_df.columns).fillna(0.0)
        final_signal_strengths_for_selection = final_signal_strengths_for_selection.reindex(filtered_returns_df.columns).fillna(0.0)
        timing_df = timing_df.reindex(filtered_returns_df.columns).fillna(
            {
                "timing_action": "hold",
                "timing_reason": "score_based_hold",
            }
        )
        optimized_expected_returns = final_expected_returns_for_selection.copy()
        selected_quality_score = quality_score_all.reindex(filtered_returns_df.columns).fillna(0.5)
        selected_downside_ratio = downside_df["downside_ratio"].reindex(filtered_returns_df.columns).fillna(0.0)
        selected_downside_multiplier = downside_penalty_multiplier_all.reindex(filtered_returns_df.columns).fillna(1.0)
        quality_multiplier_selected = (0.90 + 0.18 * selected_quality_score).clip(0.90, 1.08)
        combined_multiplier_selected = (selected_downside_multiplier * quality_multiplier_selected).clip(0.80, 1.12)
        quality_corr_selected = float(
            selected_quality_score.corr(final_expected_returns_for_selection)
        ) if len(selected_quality_score) > 1 else 0.0

        std_after_scaling_final = (
            float(final_expected_returns_for_selection.std()) if len(final_expected_returns_for_selection) > 1 else 0.0
        )
        top_assets_boost = False
        if len(final_expected_returns_for_selection) >= 3:
            top_idx = returns_before_scaling.sort_values(ascending=False).head(3).index
            mid_idx = returns_before_scaling.sort_values(ascending=False).iloc[3:].index
            top_before = float(returns_before_scaling.reindex(top_idx).mean())
            top_after = float(final_expected_returns_for_selection.reindex(top_idx).mean())
            mid_before = float(returns_before_scaling.reindex(mid_idx).mean()) if len(mid_idx) else 0.0
            mid_after = float(final_expected_returns_for_selection.reindex(mid_idx).mean()) if len(mid_idx) else 0.0
            top_assets_boost = (top_after - top_before) >= (mid_after - mid_before)
        else:
            top_assets_boost = True

        positive_share = float((final_expected_returns_for_selection > epsilon).mean()) if len(final_expected_returns_for_selection) else 0.0
        negative_share = float((final_expected_returns_for_selection < -epsilon).mean()) if len(final_expected_returns_for_selection) else 0.0
        neutral_share = float((final_expected_returns_for_selection.abs() <= epsilon).mean()) if len(final_expected_returns_for_selection) else 1.0
        unreachable_share = float((status_series == "unreachable_target").mean()) if len(status_series) else 0.0
        if len(final_expected_returns_for_selection) <= 1:
            positive_for_dispersion = expected_daily_returns[expected_daily_returns > 0].sort_values(ascending=False)
            if len(positive_for_dispersion) >= 5:
                dispersion_reference_returns = positive_for_dispersion.head(5)
            elif len(positive_for_dispersion) >= 2:
                dispersion_reference_returns = positive_for_dispersion
            else:
                dispersion_reference_returns = expected_daily_returns[selected_tickers]
            dispersion_reference_name = "top5_positive" if len(positive_for_dispersion) >= 5 else "all_positive"
        else:
            dispersion_reference_returns = final_expected_returns_for_selection
            dispersion_reference_name = "selected"

        expected_returns_std = (
            float(dispersion_reference_returns.std()) if len(dispersion_reference_returns) > 1 else 0.0
        )
        no_opportunity = (
            len(final_expected_returns_for_selection) == 0
            or float(final_expected_returns_for_selection.max()) < no_opportunity_threshold
        )
        max_expected_return = float(final_expected_returns_for_selection.max())
        mean_expected_return = float(final_expected_returns_for_selection.mean())
        asset_sharpe_ratio = calculate_asset_sharpe_ratio(returns_df, risk_free_rate=rf_daily)
        asset_sharpe_annual = asset_sharpe_ratio * np.sqrt(TRADING_DAYS_PER_YEAR)
        portfolio_return = calculate_portfolio_return(returns_df, weights)
        portfolio_volatility = calculate_portfolio_volatility(covariance_matrix, weights)
        portfolio_sharpe_ratio = calculate_sharpe_ratio(
            portfolio_return,
            portfolio_volatility,
            risk_free_rate=rf_daily,
        )
        portfolio_sharpe_annual = portfolio_sharpe_ratio * np.sqrt(TRADING_DAYS_PER_YEAR)

        if candidate_tickers:
            exposure_expected_returns = expected_daily_returns[candidate_tickers]
            exposure_signal_strengths = diagnostics_df_full.loc[candidate_tickers, "signal_strength"].astype(float)
        else:
            exposure_expected_returns = (
                dispersion_reference_returns
                if len(filtered_expected_daily_returns) <= 1
                else filtered_expected_daily_returns
            )
            exposure_signal_strengths = diagnostics_df_full.loc[
                exposure_expected_returns.index,
                "signal_strength",
            ].astype(float)
        if regime_type == "risk_on" and regime_confidence > 0.5:
            market_mode = "aggressive"
        elif regime_type == "risk_off" and regime_confidence > 0.5:
            market_mode = "defensive"
        else:
            market_mode = "neutral"

        exposure_info = compute_net_exposure(
            regime_score=regime_score,
            regime_confidence=regime_confidence,
            expected_returns=exposure_expected_returns,
            signal_strengths=exposure_signal_strengths,
            timeframe=timeframe,
            market_mode_override=market_mode,
        )
        regime_base_exposure = {"risk_on": 0.8, "neutral": 0.6, "risk_off": 0.3}.get(regime_type, 0.6)
        raw_net_exposure = float(exposure_info["net_exposure"])
        net_exposure = 0.85 * raw_net_exposure + 0.15 * regime_base_exposure
        net_exposure *= (0.90 + 0.20 * regime_confidence)
        if raw_net_exposure < 0.25:
            net_exposure = min(net_exposure, raw_net_exposure + 0.20)
            exposure_gap_reason = "limited_uplift_from_low_raw"
        else:
            exposure_gap_reason = "balanced_blend_raw_regime"
        net_exposure_before_spy_timing = float(np.clip(net_exposure, 0.05, 1.0))
        if spy_macro_regime == "bullish":
            net_exposure = net_exposure_before_spy_timing * 1.08
        elif spy_macro_regime == "neutral":
            net_exposure = net_exposure_before_spy_timing * 1.00
        else:
            net_exposure = net_exposure_before_spy_timing * 0.80
        net_exposure = float(np.clip(net_exposure, 0.15, 1.00))
        exposure_absolute_gap = float(abs(net_exposure - raw_net_exposure))
        exposure_relative_gap = float(exposure_absolute_gap / max(raw_net_exposure, 1e-8))
        exposure_info["net_exposure"] = net_exposure
        exposure_info["cash_weight"] = float(max(0.0, 1.0 - net_exposure))
        exposure_info["regime_base_exposure"] = float(regime_base_exposure)
        target_invested_weight = net_exposure
        selected_n = int(filtered_returns_df.shape[1])
        original_max_weight = float(optimizer_max_weight)
        adjusted_max_weight = original_max_weight

        if selected_n > 0:
            required_max_weight = target_invested_weight / selected_n
            if adjusted_max_weight < required_max_weight:
                adjusted_max_weight = required_max_weight + 0.01
            optimizer_required_max_weight = 1.0 / selected_n
            if adjusted_max_weight < optimizer_required_max_weight:
                adjusted_max_weight = optimizer_required_max_weight + 0.01
            adjusted_max_weight = float(min(1.0, adjusted_max_weight))
        else:
            required_max_weight = 1.0
            adjusted_max_weight = 1.0

        optimizer = PortfolioOptimizer(
            returns_df=filtered_returns_df,
            rf_daily=rf_daily,
            expected_daily_returns=optimized_expected_returns,
            use_expected_returns=use_expected_returns,
            alpha=expected_returns_alpha,
            no_opportunity=no_opportunity,
            defensive_mode=defensive_mode,
            max_weight=adjusted_max_weight,
            regime_score=regime_score,
            regime_type=regime_type,
            regime_confidence=regime_confidence,
            covariance_matrix=covariance_matrix.reindex(
                index=filtered_returns_df.columns,
                columns=filtered_returns_df.columns,
            ),
        )
        best_weights, best_sharpe, best_return, best_volatility, history = optimizer.optimize()
        raw_weight_series = pd.Series(best_weights, index=list(filtered_returns_df.columns), dtype=float)
        active_weight_series = raw_weight_series[raw_weight_series >= 0.03]
        if active_weight_series.empty:
            active_weight_series = raw_weight_series.sort_values(ascending=False).head(1)
        retained_weight_sum = float(active_weight_series.sum())
        normalized_active_weights = active_weight_series / max(retained_weight_sum, 1e-12)
        invested_fraction = net_exposure * retained_weight_sum
        exposure_scaled_weights = (normalized_active_weights * invested_fraction).astype(float)
        if exposure_scaled_weights.sum() > 0:
            exposure_scaled_weights = exposure_scaled_weights.sort_values(ascending=False)
        cash_weight = float(max(0.0, 1.0 - invested_fraction))
        if cash_weight > 0:
            exposure_scaled_weights.loc["CASH"] = cash_weight
        final_allocation_table = pd.DataFrame(
            {
                "final_weight_decimal": exposure_scaled_weights,
                "final_weight_percent": exposure_scaled_weights * 100.0,
                "allocation_per_1000": exposure_scaled_weights * 1000.0,
            }
        ).sort_values("final_weight_decimal", ascending=False)
        selected_days_to_target = (
            pd.to_numeric(diagnostics_df_full.get("time_to_target", pd.Series(dtype=float)), errors="coerce")
            .reindex(selected_tickers)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(TRADING_DAYS_PER_YEAR)
            .clip(lower=1.0)
        )
        final_asset_weights = exposure_scaled_weights.drop(labels=["CASH"], errors="ignore")
        if len(final_asset_weights) and float(final_asset_weights.sum()) > 0:
            normalized_final_asset_weights = final_asset_weights / float(final_asset_weights.sum())
            portfolio_horizon_days = float(
                (normalized_final_asset_weights * selected_days_to_target.reindex(final_asset_weights.index)).sum()
            )
        else:
            portfolio_horizon_days = float(TRADING_DAYS_PER_YEAR)
        portfolio_expected_total_return = float((1.0 + best_return) ** portfolio_horizon_days - 1.0)
        portfolio_total_volatility = float(best_volatility * np.sqrt(portfolio_horizon_days))
        final_active_positions = int((exposure_scaled_weights.drop(labels=["CASH"], errors="ignore") > 0).sum())
        non_cash_weights = exposure_scaled_weights.drop(labels=["CASH"], errors="ignore").sort_values(ascending=False)
        top1_weight = float(non_cash_weights.iloc[0]) if len(non_cash_weights) >= 1 else 0.0
        top2_cum_weight = float(non_cash_weights.iloc[:2].sum()) if len(non_cash_weights) >= 2 else top1_weight
        top3_cum_weight = float(non_cash_weights.iloc[:3].sum()) if len(non_cash_weights) >= 3 else top2_cum_weight

        threshold_positive_before = int((final_expected_returns_for_selection > 0).sum())
        threshold_positive_after = int((final_expected_returns_for_selection > dynamic_post_timing_threshold).sum())
        threshold_clamp_used = f"{post_timing_threshold_min:.6f} / {post_timing_threshold_max:.6f}"
        threshold_final_used = float(dynamic_post_timing_threshold)
        no_nan_final_series = bool(final_expected_returns_for_selection.notna().all())
        no_nan_ranking = bool(
            pd.DataFrame(
                {
                    "ret": final_expected_returns_for_selection,
                    "sig": final_signal_strengths_for_selection,
                }
            ).notna().all().all()
        )
        final_pipeline_consistent = bool(
            no_nan_final_series
            and no_nan_ranking
            and len(selected_tickers) == len(filtered_returns_df.columns)
            and set(selected_tickers) == set(filtered_returns_df.columns.tolist())
        )
        portfolio_expected_avg = float(final_expected_returns_for_selection.mean()) if len(final_expected_returns_for_selection) else 0.0
        portfolio_dispersion = float(final_expected_returns_for_selection.std()) if len(final_expected_returns_for_selection) > 1 else 0.0
        positive_sum = float(final_expected_returns_for_selection[final_expected_returns_for_selection > 0].sum())
        top1_edge = float(final_expected_returns_for_selection.sort_values(ascending=False).iloc[0]) if len(final_expected_returns_for_selection) else 0.0
        edge_concentration = float(top1_edge / positive_sum) if positive_sum > 0 else 1.0
        portfolio_quality_avg = float(selected_quality_score.mean()) if len(selected_quality_score) else 0.0
        if len(non_cash_weights) and float(non_cash_weights.sum()) > 0:
            norm_non_cash = non_cash_weights / float(non_cash_weights.sum())
            downside_aligned = selected_downside_ratio.reindex(norm_non_cash.index).fillna(0.0)
            portfolio_downside_agg = float((norm_non_cash * downside_aligned).sum())
        else:
            portfolio_downside_agg = 0.0
        portfolio_is_strong = bool(
            portfolio_expected_avg > 0.0003
            and edge_concentration < 0.75
            and portfolio_quality_avg > 0.45
            and portfolio_downside_agg < 1.20
        )
        best_sharpe_annual = best_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)
        covariance_optimizer_comparison = compare_covariance_optimizer_outputs(
            returns_df=filtered_returns_df,
            expected_daily_returns=optimized_expected_returns,
            rf_daily=rf_daily,
            shrinkage_intensity=shrinkage_intensity,
            optimizer_config={
                "use_expected_returns": use_expected_returns,
                "alpha": expected_returns_alpha,
                "no_opportunity": no_opportunity,
                "defensive_mode": defensive_mode,
                "max_weight": adjusted_max_weight,
                "regime_score": regime_score,
                "regime_type": regime_type,
                "regime_confidence": regime_confidence,
                "n_generations": 200,
                "random_seed": 42,
            },
        )
        covariance_optimizer_comparison["cash"] = cash_weight
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"\nLookback Days Used: {lookback_days}")
    print("\n===== MODEL MODE REPORT =====")
    print(f"active model mode: {model_mode}")
    print(f"timing source used: {timing_source_used}")
    print(f"target source used: {target_source_used}")
    print(f"expected return source used: {expected_return_source_used}")
    print(f"covariance method used: {covariance_method}")
    print(f"Black-Litterman status: {black_litterman_status}")
    print(f"quant target status: {quant_target_decision_status}")
    print(
        "walk-forward calibrated forecasts: "
        f"{'active' if use_walk_forward_calibrated_forecasts else 'diagnostic/disabled'}"
    )
    if use_walk_forward_calibrated_forecasts:
        print_calibrated_forecast_research_report(calibrated_forecast_report, calibrated_forecast_metadata)
    if use_raw_target_return:
        print("\n===== RAW TARGET RETURN RESEARCH MODE =====")
        print(f"raw target return used: {raw_target_research_metadata['raw_target_return_used']}")
        print(f"signal_strength adjustment disabled: {raw_target_research_metadata['signal_strength_adjustment_disabled']}")
        print(f"regime adjustment status: {raw_target_research_metadata['regime_adjustment_status']}")
        print(f"final scaling status: {raw_target_research_metadata['final_scaling_status']}")
        print(f"fallback count: {raw_target_research_metadata['fallback_count']}")
        print(
            "average delta vs baseline expected return: "
            f"{float(raw_target_research_metadata['average_delta_vs_baseline_expected_return']):.8f}"
        )
        if not raw_target_research_report.empty:
            changed = raw_target_research_report.copy()
            changed["abs_delta"] = changed["delta_vs_baseline"].abs()
            print("\ntickers most changed:")
            print(
                changed.sort_values("abs_delta", ascending=False)
                .head(10)
                [[
                    "baseline_pipeline_expected_return",
                    "raw_target_expected_return",
                    "current_expected_return",
                    "delta_vs_baseline",
                    "fallback_to_baseline",
                ]]
                .to_string()
            )
    if model_mode == "regime_gated_full_quant":
        print("\n===== FULL QUANT REGIME GATE =====")
        print(f"current regime: {full_quant_gate.get('regime', regime_type)}")
        print(f"gate decision: {full_quant_gate.get('gate_decision', 'fallback_baseline')}")
        print(f"reason: {full_quant_gate.get('reason', 'unknown')}")
        print(f"regime confidence: {float(full_quant_gate.get('regime_confidence', np.nan)):.4f}")
        print(f"volatility condition: {float(full_quant_gate.get('volatility_condition', np.nan)):.4f}")
        print(f"entropy/noise condition: {float(full_quant_gate.get('average_entropy', np.nan)):.4f}")
    if target_method == "advanced":
        print("Target Generation Method Used: advanced")
    else:
        print(f"Target Generation Method Used: basic ({basic_target_generation_method})")

    print("\nPrices DataFrame:")
    print(prices_df.tail())

    print("\nGenerated Target Prices:")
    print(target_prices)

    print("\nReturns DataFrame:")
    print(returns_df.tail())

    print("\nCovariance Matrix:")
    print(covariance_matrix)
    print_covariance_diagnostics(
        covariance_matrix,
        covariance_method,
        shrinkage_method_used=str(covariance_metadata["shrinkage_method_used"]),
        shrinkage_intensity=float(covariance_metadata["shrinkage_intensity"]),
    )

    print("\nVolatility by Asset:")
    print(volatility)

    print("\nSharpe Ratio by Asset:")
    print(asset_sharpe_ratio)

    print("\nAnnualized Sharpe Ratio by Asset:")
    print(asset_sharpe_annual)

    print("\nPortfolio Weights:")
    print(pd.Series(weights, index=returns_df.columns, name="weight"))

    print("\nPortfolio Metrics:")
    print(f"Risk-Free Annual: {rf_annual}")
    print(f"Risk-Free Daily: {rf_daily}")
    print(f"Market Regime: {regime_type} ({regime_score:.4f})")
    print(f"Regime Confidence: {regime_confidence:.4f}")
    print(f"No Opportunity Detected: {no_opportunity}")
    if no_opportunity:
        print(
            "No opportunity detected: "
            f"max_return = {max_expected_return:.6f}, mean_return = {mean_expected_return:.6f}"
        )
    print(f"Portfolio Return: {portfolio_return}")
    print(f"Portfolio Volatility: {portfolio_volatility}")
    print(f"Sharpe (Daily): {portfolio_sharpe_ratio}")
    print(f"Sharpe (Annualized): {portfolio_sharpe_annual}")

    print("\n===== NORMALIZATION CHECK =====")
    print(f"mean antes: {norm_mean_before:.8f}")
    print(f"std antes: {norm_std_before:.8f}")
    print(f"mean después: {norm_mean_after:.8f}")
    print(f"std después: {norm_std_after:.8f}")
    print(f"min/max después: {norm_min_after:.8f} / {norm_max_after:.8f}")
    print(f"% positivos vs negativos: {norm_positive_pct:.2%} / {norm_negative_pct:.2%}")

    print("\n===== MARKET REGIME MODEL =====")
    print(f"VIX actual: {float(market_regime.get('vix', float('nan'))):.4f}")
    print(f"VIX z-score: {float(market_regime.get('vix_z', 0.0)):.4f}")
    print(f"SPY momentum (20d, 60d): {float(market_regime.get('spy_momentum_20d', 0.0)):.6f}, {float(market_regime.get('spy_momentum_60d', 0.0)):.6f}")
    print(f"realized vol: {float(market_regime.get('realized_vol', 0.0)):.6f}")
    print(f"vol ratio: {float(market_regime.get('vol_regime_ratio', 0.0)):.6f}")
    print(f"breadth: {float(market_regime.get('breadth', 0.0)):.4f}")
    print(f"HMM high-vol probability: {float(market_regime.get('hmm_high_vol_probability', 0.5)):.4f}")
    print(f"Hurst exponent: {float(market_regime.get('hurst_exponent', 0.5)):.4f}")
    print(f"risk_score: {float(market_regime.get('risk_score', 0.0)):.4f}")
    print(f"regime: {regime_type}")
    print(f"regime_confidence: {regime_confidence:.4f}")

    print("\n===== EMA TIMING MODEL =====")
    if "ema_timing_score" in timing_df.columns:
        print(
            timing_df[
                [
                    "entry_valid",
                    "trend_score",
                    "structure_score",
                    "short_pullback_score",
                    "long_pullback_score",
                    "extension_penalty",
                    "ema_timing_score",
                    "timing_action",
                    "timing_reason",
                ]
            ]
        )
    else:
        print(timing_df)

    print("\n===== EMA CALIBRATION CHECK =====")
    calibration_cols = [
        "trend_score",
        "structure_score",
        "short_pullback_score",
        "long_pullback_score",
        "extension_penalty",
        "ema_timing_score",
        "timing_adjusted_return",
        "selection_score",
        "timing_action",
        "timing_reason",
    ]
    available_cols = [col for col in calibration_cols if col in timing_df.columns]
    print(timing_df[available_cols])
    ema_score_series = timing_df["ema_timing_score"].astype(float) if "ema_timing_score" in timing_df.columns else pd.Series(dtype=float)
    if not ema_score_series.empty:
        print("ema_timing_score distribution:")
        print(ema_score_series.describe())
        if "timing_adjusted_return" in timing_df.columns:
            corr_val = float(
                ema_score_series.corr(timing_df["timing_adjusted_return"].astype(float))
            ) if len(ema_score_series) > 1 else 0.0
            print(f"correlation(score, timing_adjusted_return): {corr_val:.6f}")

    print("\n===== TREND PERSISTENCE ENGINE =====")
    if "trend_persistence_df" in locals() and not trend_persistence_df.empty:
        trend_summary = pd.DataFrame(
            {
                "ema_timing_score": timing_df.get("ema_timing_score", pd.Series(dtype=float)).reindex(trend_persistence_df.index),
                "trend_persistence_score": trend_persistence_df["trend_persistence_score"],
                "trend_persistence_action": trend_persistence_df["trend_persistence_action"],
            }
        )
        print(trend_summary)
        print(f"use_trend_persistence_for_timing: {use_trend_persistence_for_timing}")
    else:
        print("Trend persistence diagnostics unavailable.")

    print("\n===== TREND PERSISTENCE COMPONENT BREAKDOWN =====")
    if (
        "trend_persistence_df" in locals()
        and not trend_persistence_df.empty
        and "ema_trend_persistence_comparison" in locals()
        and not ema_trend_persistence_comparison.empty
    ):
        trend_breakdown_cols = [
            "ema_timing_score",
            "trend_persistence_score",
            "kalman_trend_score",
            "momentum_score",
            "hurst_persistence_score",
            "entropy_cleanliness_score",
            "volatility_stability_score",
            "regime_trend_score",
            "correlation_diversification_score",
            "cycle_stability_score",
            "trend_persistence_action",
            "ema_action",
            "agreement",
            "disagreement_reason",
        ]
        trend_breakdown = trend_persistence_df.join(
            ema_trend_persistence_comparison[
                ["ema_timing_score", "ema_action", "agreement", "disagreement_reason"]
            ],
            how="left",
        )
        print(trend_breakdown[[col for col in trend_breakdown_cols if col in trend_breakdown.columns]])
    else:
        print("Trend persistence component breakdown unavailable.")

    print("\n===== EMA VS TREND PERSISTENCE COMPARISON =====")
    if "ema_trend_persistence_comparison" in locals() and not ema_trend_persistence_comparison.empty:
        print(ema_trend_persistence_comparison)
        changed_status = ema_trend_persistence_comparison[
            ema_trend_persistence_comparison["would_change_status"].astype(bool)
        ].index.tolist()
        print(f"tickers that would change status: {changed_status}")
    else:
        print("EMA comparison unavailable.")

    print("\n===== EMA VS TREND PERSISTENCE DISAGREEMENT ANALYSIS =====")
    if "ema_trend_persistence_comparison" in locals() and not ema_trend_persistence_comparison.empty:
        disagreement_analysis_cols = [
            "ema_timing_score",
            "trend_persistence_score",
            "difference",
            "ema_action",
            "trend_persistence_action",
            "agreement",
            "disagreement_reason",
        ]
        disagreement_analysis = ema_trend_persistence_comparison[
            ema_trend_persistence_comparison["agreement"].eq("disagreement")
        ]
        if disagreement_analysis.empty:
            print("No disagreements.")
        else:
            print(disagreement_analysis[disagreement_analysis_cols])
            print("disagreement reason counts:")
            print(disagreement_analysis["disagreement_reason"].value_counts())
    else:
        print("Disagreement analysis unavailable.")

    print("\n===== SPY EMA REGIME =====")
    print(f"SPY price: {float(spy_ema_regime.get('spy_price', 0.0)):.4f}")
    print(
        "EMA21 / EMA30 / EMA150 / EMA200: "
        f"{float(spy_ema_regime.get('spy_ema21', 0.0)):.4f} / "
        f"{float(spy_ema_regime.get('spy_ema30', 0.0)):.4f} / "
        f"{float(spy_ema_regime.get('spy_ema150', 0.0)):.4f} / "
        f"{float(spy_ema_regime.get('spy_ema200', 0.0)):.4f}"
    )
    print(f"macro_ema_score: {float(spy_ema_regime.get('macro_ema_score', 0.5)):.4f}")
    print(f"spy_macro_regime: {spy_macro_regime}")

    print("\n===== TIMING INTEGRATION CHECK =====")
    print(timing_integration_df)
    print(f"post_timing_edge_threshold (dynamic): {dynamic_post_timing_threshold:.6f}")
    print(f"net_exposure antes SPY timing: {net_exposure_before_spy_timing:.4f}")
    print(f"net_exposure después SPY timing: {net_exposure:.4f}")

    print("\n===== POST TIMING FILTER CHECK =====")
    print(f"post_timing_edge_threshold raw: {post_timing_edge_threshold_raw:.6f}")
    print(f"post_timing_edge_threshold final: {dynamic_post_timing_threshold:.6f}")
    print(
        "clamp min/max aplicado: "
        f"{post_timing_threshold_min:.6f} / {post_timing_threshold_max:.6f}"
    )
    print(f"cantidad de activos positivos antes del filtro: {post_timing_positive_before}")
    print(f"cantidad después del filtro: {post_timing_positive_after}")
    print(f"activos removidos por post-timing: {post_timing_removed_assets}")
    print("razón por activo:")
    print(pd.Series(post_timing_removed_reasons, dtype=object))

    print("\n===== FOURTH ASSET CHECK =====")
    print(f"4to candidato: {fourth_asset_check['candidate']}")
    print(f"expected return: {float(fourth_asset_check['expected_return']):.6f}")
    print(f"signal strength: {float(fourth_asset_check['signal_strength']):.4f}")
    print(f"diversification benefit: {float(fourth_asset_check['diversification_benefit']):.4f}")
    print(f"accepted/rejected: {fourth_asset_check['accepted']}")
    print(f"motivo: {fourth_asset_check['reason']}")

    print("\n===== DIVERSIFICATION GUARD =====")
    print(f"selected_n natural: {selected_n_natural}")
    print(f"selected_n after post-timing: {selected_n_after_post_timing}")
    print(f"selected_n final: {selected_n}")
    print(f"diversification floor used: {'yes' if diversification_floor_used else 'no'}")
    print(f"motivo: {diversification_floor_reason}")

    print("\n===== FILTERED ASSETS =====")
    filtered_assets_rows = []
    for ticker in prices_df.columns:
        status = str(expected_diagnostics[ticker]["status"])
        included = ticker in optimization_universe
        filtered_assets_rows.append(
            {
                "Ticker": ticker,
                "Status": "included" if included else status,
                "Reason": "Included in optimization universe" if included else _status_reason(status),
            }
        )
    print(pd.DataFrame(filtered_assets_rows).set_index("Ticker"))

    print("\n===== FINAL SELECTION =====")
    final_selection_rows = []
    discarded_tickers = [ticker for ticker in prices_df.columns if ticker not in selected_tickers]
    for ticker in selected_tickers:
        final_selection_rows.append(
            {
                "Ticker": ticker,
                "Decision": "selected",
                "Reason": "Top-ranked by signal strength / expected return",
            }
        )
    for ticker in discarded_tickers:
        final_selection_rows.append(
            {
                "Ticker": ticker,
                "Decision": "discarded",
                "Reason": "Outside top-N selection or no positive edge",
            }
        )
    print(pd.DataFrame(final_selection_rows).set_index("Ticker"))

    print("\n===== EXPECTED RETURNS MODEL =====")
    diagnostics_df = diagnostics_df_full.copy()
    model_columns = [
        "current_price",
        "target_price",
        "manual_long_term_target",
        "manual_target_gap_pct",
        "manual_targets_allowed_for_expected_returns",
        "old_target_price",
        "quant_target_price",
        "target_blend_weight",
        "gbm_median_target",
        "gbm_expected_target",
        "kalman_target",
        "ou_target",
        "target_confidence_quant",
        "target_method_selected",
        "quant_target_blend_enabled",
        "total_return",
        "momentum",
        "volatility",
        "mu",
        "time_to_target",
        "expected_daily_return",
        "signal_strength",
        "quality_score",
        "downside_ratio",
        "recent_drawdown",
        "return_skew",
        "garch_volatility",
        "egarch_volatility",
        "kalman_residual_vol",
        "savgol_slope",
        "fft_low_freq_energy",
        "haar_wavelet_energy",
        "hurst_exponent",
        "ou_half_life",
        "hawkes_downside_intensity",
        "entropy",
        "fractal_dimension",
        "lyapunov_proxy",
        "hill_tail_index",
        "quant_market_quality",
        "quant_multiplier",
        "penalization_applied",
        "status",
    ]
    model_labels = [
        "Current Price",
        "Target Price",
        "Manual Long-Term Target",
        "Manual Target Gap (%)",
        "Manual Targets Allowed For Expected Returns",
        "Old Target Price",
        "Quant Target Price",
        "Target Blend Weight",
        "GBM Median Target",
        "GBM Expected Target",
        "Kalman Target",
        "OU Target",
        "Quant Target Confidence",
        "Target Method Selected",
        "Quant Target Blend Enabled",
        "Total Return (%)",
        "Momentum",
        "Volatility",
        "Mu",
        "Days to Target",
        "Expected Daily Return",
        "Signal Strength",
        "Quality Score",
        "Downside Ratio",
        "Recent Drawdown",
        "Return Skew",
        "GARCH Volatility",
        "EGARCH Volatility",
        "Kalman Residual Vol",
        "Savitzky-Golay Slope",
        "FFT Low-Freq Energy",
        "Haar Wavelet Energy",
        "Hurst Exponent",
        "OU Half-Life",
        "Hawkes Downside Intensity",
        "Entropy",
        "Fractal Dimension",
        "Lyapunov Proxy",
        "Hill Tail Index",
        "Quant Market Quality",
        "Quant Multiplier",
        "Penalization Applied",
        "Status",
    ]
    if target_method == "advanced":
        model_columns += ["expected_move", "target_confidence", "target_validity"]
        model_labels += ["Expected Move", "Target Confidence", "Target Validity"]
    existing_model_columns = [c for c in model_columns if c in diagnostics_df.columns]
    missing_model_columns = [c for c in model_columns if c not in diagnostics_df.columns]
    selected_model_labels = [label for col, label in zip(model_columns, model_labels) if col in existing_model_columns]
    diagnostics_df = diagnostics_df.loc[prices_df.columns, existing_model_columns]
    diagnostics_df.columns = selected_model_labels
    if "Total Return (%)" in diagnostics_df.columns:
        diagnostics_df["Total Return (%)"] = diagnostics_df["Total Return (%)"] * 100
    if "Manual Target Gap (%)" in diagnostics_df.columns:
        diagnostics_df["Manual Target Gap (%)"] = diagnostics_df["Manual Target Gap (%)"] * 100
    print(diagnostics_df)

    print("\n===== QUANT TARGET DIAGNOSTICS =====")
    quant_target_diag = pd.DataFrame(
        {
            "old_target_price": quant_target_outputs["old_target_price"].reindex(prices_df.columns),
            "manual_long_term_target": manual_long_term_target_series.reindex(prices_df.columns),
            "manual_target_gap_pct": manual_target_gap_pct.reindex(prices_df.columns) * 100.0,
            "manual_targets_allowed_for_expected_returns": pd.Series(
                bool(allow_manual_targets_for_expected_returns),
                index=prices_df.columns,
            ),
            "quant_target_price": quant_target_outputs["quant_target_price"].reindex(prices_df.columns),
            "final_blended_target": quant_target_outputs.get(
                "final_blended_target",
                quant_target_outputs["quant_target_price"],
            ).reindex(prices_df.columns),
            "target_blend_weight": (
                quant_target_outputs["target_blend_weight"].reindex(prices_df.columns)
                if use_quant_target_blend
                else pd.Series(0.0, index=prices_df.columns)
            ),
            "gbm_median_target": quant_target_outputs["gbm_median_target"].reindex(prices_df.columns),
            "gbm_expected_target": quant_target_outputs["gbm_expected_target"].reindex(prices_df.columns),
            "kalman_target": quant_target_outputs["kalman_target"].reindex(prices_df.columns),
            "ou_target": quant_target_outputs["ou_target"].reindex(prices_df.columns),
            "target_confidence": quant_target_outputs["target_confidence"].reindex(prices_df.columns),
            "target_method_selected": quant_target_outputs["target_method_selected"].reindex(prices_df.columns),
            "blend_enabled": pd.Series(bool(use_quant_target_blend), index=prices_df.columns),
        }
    )
    print(quant_target_diag.replace([np.inf, -np.inf], np.nan).fillna(0.0))

    print("\n===== TARGET COMPARISON REPORT =====")
    target_comparison_report = quant_target_diag.copy()
    required_target_report_columns = [
        "old_target_price",
        "quant_target_price",
        "gbm_median_target",
        "gbm_expected_target",
        "kalman_target",
        "ou_target",
        "target_confidence",
        "final_blended_target",
    ]
    missing_target_report_columns = [
        col for col in required_target_report_columns if col not in target_comparison_report.columns
    ]
    if "final_blended_target" not in target_comparison_report.columns:
        target_comparison_report["final_blended_target"] = target_comparison_report.get(
            "quant_target_price",
            pd.Series(np.nan, index=target_comparison_report.index),
        )
    old_target_safe = target_comparison_report["old_target_price"].replace(0, np.nan)
    target_comparison_report["old_vs_quant_diff_pct"] = (
        (target_comparison_report["quant_target_price"] / old_target_safe - 1.0) * 100.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target_comparison_report["gbm_expected_vs_median_pct"] = (
        (target_comparison_report["gbm_expected_target"] / target_comparison_report["gbm_median_target"].replace(0, np.nan) - 1.0)
        * 100.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gbm_reference = target_comparison_report[["gbm_median_target", "gbm_expected_target"]].mean(axis=1)
    kalman_direction = target_comparison_report["kalman_target"] - target_comparison_report["old_target_price"]
    ou_direction = target_comparison_report["ou_target"] - target_comparison_report["old_target_price"]
    target_comparison_report["ou_against_trend"] = (
        np.sign(kalman_direction.replace(0, np.nan)).fillna(0.0)
        != np.sign(ou_direction.replace(0, np.nan)).fillna(0.0)
    ) & (
        (target_comparison_report["ou_target"] - gbm_reference).abs()
        / target_comparison_report["old_target_price"].replace(0, np.nan).abs()
        > 0.08
    )

    suspicious_reasons = {}
    for ticker, row in target_comparison_report.iterrows():
        reasons = []
        if abs(float(row["old_vs_quant_diff_pct"])) > 15.0:
            reasons.append("quant_vs_old_diff_gt_15pct")
        if float(row["target_confidence"]) < 0.35:
            reasons.append("low_target_confidence")
        if bool(row["ou_against_trend"]):
            reasons.append("ou_pulling_against_kalman_gbm")
        if float(row["gbm_expected_vs_median_pct"]) > 5.0:
            reasons.append("gbm_expected_much_above_median")
        suspicious_reasons[ticker] = ", ".join(reasons) if reasons else "ok"

    target_comparison_report["suspicious_flags"] = pd.Series(suspicious_reasons)
    print(f"target_comparison_report.columns: {target_comparison_report.columns.tolist()}")
    print(f"missing columns detected: {missing_target_report_columns}")
    target_report_display_columns = [
        "old_target_price",
        "quant_target_price",
        "gbm_median_target",
        "gbm_expected_target",
        "kalman_target",
        "ou_target",
        "target_confidence",
        "final_blended_target",
        "old_vs_quant_diff_pct",
        "gbm_expected_vs_median_pct",
        "ou_against_trend",
        "suspicious_flags",
    ]
    existing_target_report_display_columns = [
        col for col in target_report_display_columns if col in target_comparison_report.columns
    ]
    print(
        target_comparison_report[existing_target_report_display_columns].sort_values(
            "old_vs_quant_diff_pct",
            key=lambda s: s.abs(),
            ascending=False,
        )
    )

    print("\n===== DIAGNOSTICS CONSISTENCY CHECK =====")
    print(f"columnas faltantes detectadas: {diagnostics_missing_detected}")
    print(f"columnas creadas por default: {diagnostics_created_defaults}")
    print(f"columnas ausentes del model_columns final: {missing_model_columns}")
    print("NaN por columna crítica:")
    print(pd.Series(diagnostics_nan_counts))
    print(f"shape final diagnostics_df: {diagnostics_shape_final}")

    print("\n===== DOWNSIDE METRICS =====")
    downside_print = downside_df.reindex(selected_tickers).fillna(0.0)
    print(downside_print)

    print("\n===== QUALITY SCORE =====")
    quality_print = pd.DataFrame(
        {
            "quality_score": quality_score_all.reindex(selected_tickers).fillna(0.5),
            "signal_strength": final_signal_strengths_for_selection.reindex(selected_tickers).fillna(0.0),
            "expected_daily_return": final_expected_returns_for_selection.reindex(selected_tickers).fillna(0.0),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    print(quality_print.sort_values("quality_score", ascending=False))

    print("\n===== UNIVERSE EXPECTED RETURN DISTRIBUTION =====")
    universe_expected_returns = raw_expected_returns.replace([np.inf, -np.inf], np.nan).dropna()
    universe_positive_count = int((universe_expected_returns > epsilon).sum())
    universe_negative_count = int((universe_expected_returns < -epsilon).sum())
    universe_near_zero_count = int((universe_expected_returns.abs() <= epsilon).sum())
    print(f"count total assets: {len(universe_expected_returns)}")
    print(f"count positive raw expected returns: {universe_positive_count}")
    print(f"count negative raw expected returns: {universe_negative_count}")
    print(f"count near-zero expected returns: {universe_near_zero_count}")
    print(f"mean: {float(universe_expected_returns.mean()) if len(universe_expected_returns) else 0.0:.8f}")
    print(f"std: {float(universe_expected_returns.std()) if len(universe_expected_returns) > 1 else 0.0:.8f}")
    print(f"min: {float(universe_expected_returns.min()) if len(universe_expected_returns) else 0.0:.8f}")
    print(f"max: {float(universe_expected_returns.max()) if len(universe_expected_returns) else 0.0:.8f}")
    print("25/50/75 percentiles:")
    print(universe_expected_returns.quantile([0.25, 0.50, 0.75]) if len(universe_expected_returns) else pd.Series(dtype=float))

    print("\n===== FILTER ATTRITION REPORT =====")
    assets_positive_before_timing = int((expected_daily_returns.reindex(prices_df.columns).fillna(0.0) > 0).sum())
    assets_positive_after_timing = int((timing_adjusted_expected_returns > 0).sum()) if "timing_adjusted_expected_returns" in locals() else 0
    assets_above_edge_threshold = int((expected_series > edge_threshold).sum()) if "expected_series" in locals() else 0
    print(f"total universe assets: {len(prices_df.columns)}")
    print(f"assets positive before timing: {assets_positive_before_timing}")
    print(f"assets positive after timing: {assets_positive_after_timing}")
    print(f"assets above edge threshold: {assets_above_edge_threshold}")
    print(f"final selected assets: {len(selected_tickers)}")
    print(f"final optimized assets: {filtered_returns_df.shape[1]}")

    print("\n===== FINAL SELECTED RETURN DISTRIBUTION =====")
    print(f"std(final_selected_expected_returns): {expected_returns_std:.8f}")
    print(f"% selected assets with return > 0: {positive_share:.2%}")
    print(f"% selected assets with return < 0: {negative_share:.2%}")
    print(f"% selected assets neutral: {neutral_share:.2%}")
    print(f"% universe unreachable: {unreachable_share:.2%}")

    print("\n===== FINAL SELECTED SIGNAL DIAGNOSTIC =====")
    print(f"% negativos: {(final_expected_returns_for_selection < 0).mean() * 100:.2f}%")
    print(f"% positivos: {(final_expected_returns_for_selection > 0).mean() * 100:.2f}%")
    print(f"% cero: {(final_expected_returns_for_selection == 0).mean() * 100:.2f}%")

    print("\n===== FINAL SELECTED RETURN PERCENTILES =====")
    percentiles = final_expected_returns_for_selection.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    print(percentiles)
    repeated_counts = final_expected_returns_for_selection.round(6).value_counts().sort_values(ascending=False)
    print("Repeated values:")
    print(repeated_counts.head(10))

    print("\n===== OPPORTUNITY RANKING =====")
    opportunity_ranking = pd.DataFrame(
        {
            "Expected Daily Return": final_expected_returns_for_selection,
            "Signal Strength": final_signal_strengths_for_selection,
            "Status": status_series.reindex(final_expected_returns_for_selection.index).fillna("unknown"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().sort_values(by=["Signal Strength", "Expected Daily Return"], ascending=False)
    print(opportunity_ranking)

    print("\n===== ADJUSTED EXPECTED RETURNS =====")
    print(final_expected_returns_for_selection.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False))

    print("\n===== THRESHOLD CALIBRATION CHECK =====")
    print(f"raw threshold: {post_timing_edge_threshold_raw:.6f}")
    print(f"final threshold: {threshold_final_used:.6f}")
    print(f"clamp usado: {threshold_clamp_used}")
    print(f"market_mode: {market_mode}")
    print(f"positivos antes: {threshold_positive_before}")
    print(f"positivos después: {threshold_positive_after}")

    print("\n===== RETURN SCALING CHECK =====")
    print("returns before scaling (top 5):")
    print(returns_before_scaling.sort_values(ascending=False).head(5))
    print("returns after scaling (top 5):")
    print(final_expected_returns_for_selection.sort_values(ascending=False).head(5))
    print(f"scaling method used: {scaling_method}")
    print(
        "exponent / parameters usados: "
        f"exponent={scaling_exponent:.2f}, alpha={scaling_alpha:.2f}, vol_strength={volatility_strength:.2f}"
    )
    print(f"volatility adjustment aplicado: {'si' if volatility_adjustment_applied else 'no'}")
    print(
        "max/min antes y después: "
        f"{float(returns_before_scaling.max()):.8f}/{float(returns_before_scaling.min()):.8f} -> "
        f"{float(final_expected_returns_for_selection.max()):.8f}/{float(final_expected_returns_for_selection.min()):.8f}"
    )
    print(
        "validación interna: "
        f"ranking_unchanged={ranking_unchanged}, "
        f"std_before={std_before_scaling:.8f}, std_after={std_after_scaling_final:.8f}, "
        f"top_assets_boost={top_assets_boost}, clip={lower_clip:.8f}/{upper_clip:.8f}"
    )

    print("\n===== DOWNSIDE IMPACT CHECK =====")
    print(f"returns std antes downside: {downside_std_before:.8f}")
    print(f"returns std después downside: {downside_std_after:.8f}")
    print(f"ranking cambiado: {downside_rank_changed}")

    print("\n===== QUALITY IMPACT CHECK =====")
    print(f"correlación quality_score vs expected_return: {quality_corr_selected:.6f}")
    print("distribución quality_score:")
    print(selected_quality_score.describe())

    print("\n===== MULTIPLIER CALIBRATION CHECK =====")
    print(
        "downside_multiplier min/max/mean: "
        f"{float(selected_downside_multiplier.min()):.4f} / "
        f"{float(selected_downside_multiplier.max()):.4f} / "
        f"{float(selected_downside_multiplier.mean()):.4f}"
    )
    print(
        "quality_multiplier min/max/mean: "
        f"{float(quality_multiplier_selected.min()):.4f} / "
        f"{float(quality_multiplier_selected.max()):.4f} / "
        f"{float(quality_multiplier_selected.mean()):.4f}"
    )
    print(
        "combined_multiplier min/max/mean: "
        f"{float(combined_multiplier_selected.min()):.4f} / "
        f"{float(combined_multiplier_selected.max()):.4f} / "
        f"{float(combined_multiplier_selected.mean()):.4f}"
    )

    print("\n===== EDGE PRESERVATION CHECK =====")
    print(f"ranking base: {edge_base_order}")
    print(f"ranking final: {edge_final_order}")
    print(f"swaps: {edge_swaps_count}")
    print(f"ranking_changed: {edge_ranking_changed}")
    print("impacto porcentual por activo:")
    print(edge_impact_pct.reindex(final_expected_returns_for_selection.index).sort_values(ascending=False))
    print(f"dominance_guard_actuó: {dominance_guard_acted}")
    print(f"dominance_guard_assets: {dominance_guard_assets}")

    print("\n===== SELECTION FILTER CHECK =====")
    print(f"activos removidos por filtro relativo: {list(selection_filter_removed.keys())}")
    print("razón:")
    print(pd.Series(selection_filter_removed, dtype=object))

    if expected_returns_std < 1e-4:
        print("\nModel collapse debug:")
        print(f"std(expected_returns) = {expected_returns_std:.8f}")
        print(final_expected_returns_for_selection.sort_values(ascending=False))

    print("\nOptimized Portfolio Weights:")
    print(exposure_scaled_weights.rename("optimized_weight"))

    print("\n===== FINAL ALLOCATION INCLUDING CASH =====")
    print(final_allocation_table)
    print(f"Total allocation: {float(final_allocation_table['final_weight_decimal'].sum()):.6f}")
    print("Use final_weight_percent as the real portfolio allocation.")
    late_report_sections = {
        "INSTITUTIONAL_RISK",
        "BLACK_LITTERMAN",
        "ACTION_SIGNALS",
        "FORECAST_CALIBRATION",
        "INFORMATION_COEFFICIENT",
        "HEURISTIC_AUDIT",
        "HEURISTIC_CALIBRATION",
        "FACTOR_ATTRIBUTION",
    }
    compact_filter_active = isinstance(builtins.print, _CompactReportPrinter)
    if compact_filter_active and not (set(report_sections_to_show) & late_report_sections):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    print("\n===== CONCENTRATION CHECK =====")
    print(f"número de activos seleccionados: {selected_n}")
    print(f"peso top1: {top1_weight:.4f}")
    print(f"peso top2 acumulado: {top2_cum_weight:.4f}")
    print(f"peso top3 acumulado: {top3_cum_weight:.4f}")
    print(f"cash: {cash_weight:.4f}")

    if report_enabled("INSTITUTIONAL_RISK"):
        print_institutional_risk_metrics(
            returns_df=returns_df,
            final_weights=exposure_scaled_weights,
            selected_tickers=selected_tickers,
        )

    print("\n===== SELECTED ASSETS EXPECTED RETURN % =====")
    selected_assets_expected_df = pd.DataFrame(
        {
            "expected_daily_return": final_expected_returns_for_selection.reindex(selected_tickers).fillna(0.0),
            "expected_percent": final_expected_returns_for_selection.reindex(selected_tickers).fillna(0.0) * 100,
            "days_to_target": selected_days_to_target.reindex(selected_tickers).fillna(TRADING_DAYS_PER_YEAR),
            "expected_total_return": (
                (1.0 + final_expected_returns_for_selection.reindex(selected_tickers).fillna(0.0))
                ** selected_days_to_target.reindex(selected_tickers).fillna(TRADING_DAYS_PER_YEAR)
                - 1.0
            ),
            "expected_total_percent": (
                (
                    (1.0 + final_expected_returns_for_selection.reindex(selected_tickers).fillna(0.0))
                    ** selected_days_to_target.reindex(selected_tickers).fillna(TRADING_DAYS_PER_YEAR)
                    - 1.0
                )
                * 100.0
            ),
            "risk_adjusted_return": (
                final_expected_returns_for_selection.reindex(selected_tickers).fillna(0.0)
                / (volatility.reindex(selected_tickers).fillna(0.0) + 1e-8)
            ),
            "signal_strength": final_signal_strengths_for_selection.reindex(selected_tickers).fillna(0.0),
            "timing_action": timing_df.reindex(selected_tickers)["timing_action"].fillna("hold"),
            "timing_reason": timing_df.reindex(selected_tickers)["timing_reason"].fillna("score_based_hold"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    print(selected_assets_expected_df.sort_values("expected_daily_return", ascending=False))

    print("\n===== EXPOSURE CONTROL =====")
    print(f"net_exposure: {exposure_info['net_exposure']:.4f}")
    print(f"regime_adjustment: {float(exposure_info['regime_adjustment']):.4f}")
    print(f"confidence_adjustment: {float(exposure_info['confidence_adjustment']):.4f}")
    print(f"opportunity_adjustment: {float(exposure_info['opportunity_adjustment']):.4f}")
    print(f"dispersion_adjustment: {float(exposure_info['dispersion_adjustment']):.4f}")
    print(f"cash_weight: {exposure_info['cash_weight']:.4f}")

    print("\n===== MARKET MODE =====")
    print(f"mode: {exposure_info['market_mode']}")
    print(f"opportunity_score: {float(exposure_info['opportunity_score']):.4f}")
    print(f"dispersion_score: {float(exposure_info['dispersion_score']):.4f}")
    print(f"net_exposure: {float(exposure_info['net_exposure']):.4f}")

    print("\n===== EXPOSURE GAP CHECK =====")
    print(f"raw_net_exposure: {raw_net_exposure:.4f}")
    print(f"final_net_exposure: {net_exposure:.4f}")
    print(f"absolute_gap: {exposure_absolute_gap:.4f}")
    print(f"relative_gap: {exposure_relative_gap:.4f}")
    print(f"motivo del ajuste: {exposure_gap_reason}")

    print("\n===== AGGRESSIVENESS CHECK =====")
    print(f"top 3 expected returns: {exposure_info['top3_expected_returns']}")
    print(f"top 3 signal strength: {exposure_info['top3_signal_strengths']}")
    print(
        "dispersion (top N vs total): "
        f"{float(exposure_info['dispersion_score']):.4f} vs {float(exposure_info['dispersion_score_total']):.4f}"
    )
    print(
        "exposure before/after override: "
        f"{float(exposure_info['pre_override_exposure']):.4f} -> {float(exposure_info['post_override_exposure']):.4f}"
    )

    print("\n===== OPTIMIZER FEASIBILITY =====")
    print(f"selected_n: {selected_n}")
    print(f"net_exposure: {net_exposure:.4f}")
    print(f"target_invested_weight: {target_invested_weight:.4f}")
    print(f"original max_weight: {original_max_weight:.4f}")
    print(f"adjusted max_weight: {adjusted_max_weight:.4f}")

    print("\n===== SELECTION QUALITY =====")
    print(f"count positivos reales: {positive_real_count}")
    print(f"count positivos > edge_threshold: {positive_edge_count}")
    print(f"selected assets: {selected_tickers}")
    print(f"discarded for low edge: {discarded_low_edge}")
    print(f"final active positions: {final_active_positions}")

    print("\n===== SELECTION ENGINE CHECK =====")
    print("selection_score top 10:")
    print(selection_score_series.sort_values(ascending=False).head(10))
    print(f"candidate count: {len(candidate_tickers)}")
    print(f"selected_n natural: {selected_n_pre_fallback}")
    print(f"fallback used: {fallback_activated}")
    print(f"fallback reason: {fallback_reason}")
    print(f"final selected assets: {selected_tickers}")

    print("\n===== ANTI-COLLAPSE CHECK =====")
    print(f"edge_threshold usado: {edge_threshold:.6f}")
    print(f"positivos totales: {positive_real_count}")
    print(f"activos post filtro: {len(candidate_tickers)}")
    print(f"selected_n final: {selected_n}")
    print(f"fallback activado: {fallback_activated}")
    print(f"dispersion source: {dispersion_reference_name}")

    print("\n===== FINAL CONSISTENCY CHECK =====")
    print(f"regime: {regime_type}")
    print(f"market_mode: {market_mode}")
    print(f"regime_confidence: {regime_confidence:.4f}")
    print(f"selected_n pre-fallback: {selected_n_pre_fallback}")
    print(f"selected_n post-fallback: {selected_n_post_fallback}")
    print(f"fallback reason: {fallback_reason}")
    print(f"final selected assets: {selected_tickers}")
    print(f"discarded assets: {discarded_assets}")

    print("\n===== FINAL PIPELINE CONSISTENCY =====")
    print(f"cantidad de candidatos: {len(candidate_tickers)}")
    print(f"cantidad post timing: {selected_n_after_post_timing}")
    print(f"cantidad post threshold: {selected_n_post_threshold}")
    print(f"cantidad final seleccionada: {len(selected_tickers)}")
    print(f"cantidad final optimizada: {filtered_returns_df.shape[1]}")
    print(f"no NaN final_expected_returns: {no_nan_final_series}")
    print(f"no NaN ranking inputs: {no_nan_ranking}")
    print(f"pipeline consistente: {final_pipeline_consistent}")

    print("\n===== PORTFOLIO QUALITY CHECK =====")
    print(f"promedio expected_return: {portfolio_expected_avg:.8f}")
    print(f"dispersión entre activos: {portfolio_dispersion:.8f}")
    print(f"concentración de edge (top1): {edge_concentration:.4f}")
    print(f"quality promedio: {portfolio_quality_avg:.4f}")
    print(f"downside agregado estimado: {portfolio_downside_agg:.4f}")
    print(f"portfolio_is_strong: {portfolio_is_strong}")

    target_confidence_for_bl = pd.Series(
        {
            ticker: float(diagnostics_df_full.get("target_confidence_quant", pd.Series(dtype=float)).reindex(prices_df.columns).get(ticker, np.nan))
            for ticker in prices_df.columns
        },
        index=prices_df.columns,
        dtype=float,
    ).fillna(
        pd.to_numeric(diagnostics_df_full.get("target_confidence", pd.Series(dtype=float)), errors="coerce")
        .reindex(prices_df.columns)
        .fillna(0.5)
    )
    if report_enabled("BLACK_LITTERMAN"):
        print_black_litterman_diagnostics(
            covariance_matrix=covariance_matrix.reindex(index=prices_df.columns, columns=prices_df.columns),
            expected_returns=expected_daily_returns.reindex(prices_df.columns).fillna(0.0),
            benchmark_weights=exposure_scaled_weights.drop(labels=["CASH"], errors="ignore"),
            target_confidence=target_confidence_for_bl,
            signal_strength=diagnostics_df_full["signal_strength"].reindex(prices_df.columns).fillna(0.5),
            quality_score=quality_score_all.reindex(prices_df.columns).fillna(0.5),
        )

    if report_enabled("FACTOR_ATTRIBUTION"):
        print_factor_attribution(
            selected_tickers=selected_tickers,
            diagnostics_df_full=diagnostics_df_full.reindex(prices_df.columns),
            final_expected_returns=final_expected_returns_for_selection,
            timing_df=timing_df,
            regime_score=regime_score,
            regime_type=regime_type,
        )

    if report_enabled("HEURISTIC_CALIBRATION"):
        print_heuristic_calibration_diagnostics(
            expected_returns=expected_daily_returns.reindex(prices_df.columns).fillna(0.0),
            signal_strength=diagnostics_df_full["signal_strength"].reindex(prices_df.columns).fillna(0.0),
            selection_score=selection_score_series.reindex(prices_df.columns).fillna(0.0),
            current_selected_assets=selected_tickers,
            returns_df=returns_df.reindex(columns=prices_df.columns),
            covariance_matrix=covariance_matrix.reindex(index=prices_df.columns, columns=prices_df.columns),
            rf_daily=rf_daily,
            cash_weight=cash_weight,
            regime_type=regime_type,
        )

    print("\n===== COVARIANCE OPTIMIZER COMPARISON =====")
    print(covariance_optimizer_comparison)

    print("\nOptimized Portfolio Metrics:")
    print(f"Best Optimized Return: {best_return}")
    print(f"Best Optimized Volatility: {best_volatility}")
    print(f"Portfolio Horizon Days: {portfolio_horizon_days:.2f}")
    print(f"Expected Total Return over Horizon: {portfolio_expected_total_return:.6f}")
    print(f"Expected Total Volatility over Horizon: {portfolio_total_volatility:.6f}")
    print(f"Sharpe (Daily): {best_sharpe}")
    print(f"Sharpe (Annualized): {best_sharpe_annual}")

    if report_enabled("HEURISTIC_AUDIT"):
        print_heuristic_audit_report(".")

    snapshot_expected_returns = expected_daily_returns.reindex(prices_df.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    snapshot_expected_returns.update(final_expected_returns_for_selection)
    snapshot_weight_percent = final_allocation_table["final_weight_percent"].drop(labels=["CASH"], errors="ignore")
    current_prices_for_actions = diagnostics_df_full.get(
        "current_price",
        prices_df.ffill().iloc[-1],
    )
    current_prices_for_actions = pd.to_numeric(
        pd.Series(current_prices_for_actions).reindex(prices_df.columns),
        errors="coerce",
    ).fillna(prices_df.ffill().iloc[-1].reindex(prices_df.columns))
    action_signals_df = pd.DataFrame()
    if report_enabled("ACTION_SIGNALS") or paper_trading_enabled:
        action_signals_df = generate_action_signals_report(
            final_allocation_table=final_allocation_table,
            selected_tickers=selected_tickers,
            current_prices=current_prices_for_actions,
            target_prices=target_prices.reindex(prices_df.columns),
            expected_daily_returns=snapshot_expected_returns,
            diagnostics_df_full=diagnostics_df_full.reindex(prices_df.columns),
        )
    if paper_trading_enabled:
        from paper_trading_simulator import update_paper_trading_simulation

        paper_allocation_table = final_allocation_table
        if paper_meta_filter_enabled:
            from paper_meta_filter import PaperMetaFilterConfig, apply_paper_meta_filter

            paper_allocation_table, _ = apply_paper_meta_filter(
                final_allocation_table=final_allocation_table,
                diagnostics_df_full=diagnostics_df_full.reindex(prices_df.columns),
                expected_daily_returns=snapshot_expected_returns,
                current_date=prices_df.index[-1] if len(prices_df.index) else None,
                config=PaperMetaFilterConfig(
                    model=paper_meta_filter_model,
                    threshold=paper_meta_filter_threshold,
                ),
                print_report=report_enabled("PAPER_META_FILTER") or report_enabled("PAPER_TRADING"),
            )
        update_paper_trading_simulation(
            final_allocation_table=paper_allocation_table,
            action_signals=action_signals_df,
            prices_df=prices_df,
            model_mode=paper_model_mode,
            current_prices=current_prices_for_actions,
            overwrite_same_day=paper_overwrite_same_day,
        )
    if report_enabled("FORECAST_CALIBRATION"):
        save_and_evaluate_forecasts(
            prices_df=prices_df,
            target_prices=target_prices.reindex(prices_df.columns),
            expected_daily_returns=snapshot_expected_returns,
            diagnostics_df=diagnostics_df_full.reindex(prices_df.columns),
            selected_assets=selected_tickers,
            final_weight_percent=snapshot_weight_percent,
            regime=regime_type,
            overwrite_same_day=paper_overwrite_same_day,
        )
    if report_enabled("INFORMATION_COEFFICIENT"):
        print_information_coefficient_report(
            prices_df=prices_df,
            diagnostics_df=diagnostics_df_full.reindex(prices_df.columns),
            timing_df=timing_df,
            regime_score=regime_score,
            regime_confidence=regime_confidence,
        )


if __name__ == "__main__":
    main()
