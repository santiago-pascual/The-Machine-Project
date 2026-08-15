from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _clean_returns(returns: pd.Series) -> pd.Series:
    return pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _max_drawdown(returns: pd.Series) -> float:
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _downside_deviation(returns: pd.Series, annualize: bool = True) -> float:
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    downside = clean[clean < 0.0]
    if downside.empty:
        return 0.0
    value = float(np.sqrt(np.mean(np.square(downside))))
    return value * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else value


def _historical_var(returns: pd.Series, confidence: float) -> float:
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    loss = -clean
    return float(np.quantile(loss, confidence))


def _historical_cvar(returns: pd.Series, confidence: float) -> float:
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    loss = -clean
    var = float(np.quantile(loss, confidence))
    tail = loss[loss >= var]
    return float(tail.mean()) if not tail.empty else var


def _gaussian_var_cvar(returns: pd.Series, confidence: float) -> tuple[float, float]:
    clean = _clean_returns(returns)
    if len(clean) < 2:
        return 0.0, 0.0
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1))
    if sigma <= 0 or not np.isfinite(sigma):
        return 0.0, 0.0

    tail_probability = 1.0 - confidence
    normal = NormalDist()
    z = normal.inv_cdf(tail_probability)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    var_return = mu + sigma * z
    cvar_return = mu - sigma * phi / max(tail_probability, 1e-12)
    return float(max(0.0, -var_return)), float(max(0.0, -cvar_return))


def compute_return_risk_metrics(returns: pd.Series) -> dict[str, float]:
    clean = _clean_returns(returns)
    if clean.empty:
        return {
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "cvar_99": 0.0,
            "gaussian_var_95": 0.0,
            "gaussian_var_99": 0.0,
            "gaussian_cvar_95": 0.0,
            "gaussian_cvar_99": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "downside_deviation": 0.0,
            "annualized_volatility": 0.0,
            "annualized_return_estimate": 0.0,
            "observations": 0.0,
        }

    annualized_return = float(clean.mean() * TRADING_DAYS_PER_YEAR)
    annualized_volatility = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(clean) > 1 else 0.0
    downside_deviation = _downside_deviation(clean, annualize=True)
    max_drawdown = _max_drawdown(clean)
    sortino = annualized_return / downside_deviation if downside_deviation > 0 else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    gaussian_var_95, gaussian_cvar_95 = _gaussian_var_cvar(clean, 0.95)
    gaussian_var_99, gaussian_cvar_99 = _gaussian_var_cvar(clean, 0.99)

    return {
        "var_95": _historical_var(clean, 0.95),
        "var_99": _historical_var(clean, 0.99),
        "cvar_95": _historical_cvar(clean, 0.95),
        "cvar_99": _historical_cvar(clean, 0.99),
        "gaussian_var_95": gaussian_var_95,
        "gaussian_var_99": gaussian_var_99,
        "gaussian_cvar_95": gaussian_cvar_95,
        "gaussian_cvar_99": gaussian_cvar_99,
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "max_drawdown": float(max_drawdown),
        "downside_deviation": float(downside_deviation),
        "annualized_volatility": float(annualized_volatility),
        "annualized_return_estimate": float(annualized_return),
        "observations": float(len(clean)),
    }


def compute_portfolio_returns_with_cash(
    returns_df: pd.DataFrame,
    final_weights: pd.Series,
) -> pd.Series:
    if returns_df.empty:
        return pd.Series(dtype=float)
    weights = pd.to_numeric(pd.Series(final_weights), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    asset_weights = weights.drop(labels=["CASH"], errors="ignore")
    aligned_assets = [ticker for ticker in asset_weights.index if ticker in returns_df.columns]
    if not aligned_assets:
        return pd.Series(0.0, index=returns_df.index, dtype=float)
    aligned_returns = returns_df[aligned_assets].dropna(how="any")
    aligned_weights = asset_weights.reindex(aligned_assets).astype(float)
    return aligned_returns.dot(aligned_weights)


def compute_selected_asset_risk_metrics(
    returns_df: pd.DataFrame,
    selected_tickers: list[str],
) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for ticker in selected_tickers:
        if ticker not in returns_df.columns:
            continue
        metrics = compute_return_risk_metrics(returns_df[ticker])
        rows[ticker] = {
            "var_95": metrics["var_95"],
            "cvar_95": metrics["cvar_95"],
            "sortino_ratio": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "downside_deviation": metrics["downside_deviation"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def print_institutional_risk_metrics(
    *,
    returns_df: pd.DataFrame,
    final_weights: pd.Series,
    selected_tickers: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    print("\n===== INSTITUTIONAL RISK METRICS =====")
    portfolio_returns = compute_portfolio_returns_with_cash(returns_df, final_weights)
    portfolio_metrics = pd.Series(compute_return_risk_metrics(portfolio_returns), name="portfolio")
    selected_asset_metrics = compute_selected_asset_risk_metrics(returns_df, selected_tickers)

    print("\nPortfolio level:")
    print("Cash treatment: included as zero-return asset via final portfolio weights.")
    portfolio_display = portfolio_metrics[
        [
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
            "gaussian_var_95",
            "gaussian_var_99",
            "gaussian_cvar_95",
            "gaussian_cvar_99",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "downside_deviation",
            "annualized_volatility",
            "annualized_return_estimate",
            "observations",
        ]
    ]
    print(portfolio_display)

    print("\nSelected asset level:")
    print(selected_asset_metrics if not selected_asset_metrics.empty else "No selected asset risk metrics available.")
    return portfolio_metrics, selected_asset_metrics
