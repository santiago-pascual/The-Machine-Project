from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_series(values: pd.Series | dict | None, index: pd.Index, default: float = 0.0) -> pd.Series:
    if values is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(pd.Series(values).reindex(index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_inverse(matrix: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    regularized = matrix + ridge * np.eye(matrix.shape[0])
    try:
        return np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized)


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    data = pd.concat([left, right], axis=1).dropna()
    if len(data) < 2 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(data.iloc[:, 0].rank().corr(data.iloc[:, 1].rank()))


def compute_black_litterman_diagnostics(
    *,
    covariance_matrix: pd.DataFrame,
    expected_returns: pd.Series,
    benchmark_weights: pd.Series | None = None,
    target_confidence: pd.Series | None = None,
    signal_strength: pd.Series | None = None,
    quality_score: pd.Series | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    omega_floor: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    tickers = pd.Index(covariance_matrix.columns.astype(str))
    sigma_df = covariance_matrix.reindex(index=tickers, columns=tickers).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sigma = sigma_df.to_numpy(dtype=float)
    n_assets = len(tickers)

    if n_assets == 0:
        return pd.DataFrame(), {"fallback_used": "empty_universe"}

    model_returns = _safe_series(expected_returns, tickers, default=0.0)

    if benchmark_weights is not None:
        weights = _safe_series(benchmark_weights, tickers, default=0.0).clip(lower=0.0)
        if float(weights.sum()) > 0:
            weights = weights / float(weights.sum())
            fallback_used = "portfolio_weights"
        else:
            weights = pd.Series(1.0 / n_assets, index=tickers)
            fallback_used = "equal_weights_empty_benchmark"
    else:
        weights = pd.Series(1.0 / n_assets, index=tickers)
        fallback_used = "equal_weights_no_market_cap"

    delta = float(risk_aversion)
    tau_value = float(tau)
    pi = pd.Series(delta * sigma @ weights.to_numpy(dtype=float), index=tickers, name="equilibrium_return")

    target_conf = _safe_series(target_confidence, tickers, default=0.5).clip(0.0, 1.0)
    signal = _safe_series(signal_strength, tickers, default=0.5).clip(0.0, 1.0)
    quality = _safe_series(quality_score, tickers, default=0.5).clip(0.0, 1.0)
    confidence = (0.40 * target_conf + 0.35 * signal + 0.25 * quality).clip(0.05, 0.95)

    p_matrix = np.eye(n_assets)
    q_vector = model_returns.to_numpy(dtype=float)
    view_variance = np.diag(sigma).copy()
    median_variance = float(np.nanmedian(view_variance[view_variance > 0])) if np.any(view_variance > 0) else 1e-4
    view_variance = np.where(view_variance > 0, view_variance, median_variance)
    omega_diag = np.maximum(omega_floor, tau_value * view_variance * (1.0 - confidence.to_numpy()) / confidence.to_numpy())
    omega = np.diag(omega_diag)

    tau_sigma = tau_value * sigma
    inv_tau_sigma = _safe_inverse(tau_sigma)
    inv_omega = _safe_inverse(omega)
    posterior_cov_inv = inv_tau_sigma + p_matrix.T @ inv_omega @ p_matrix
    posterior_cov = _safe_inverse(posterior_cov_inv)
    posterior_mean = posterior_cov @ (
        inv_tau_sigma @ pi.to_numpy(dtype=float)
        + p_matrix.T @ inv_omega @ q_vector
    )

    bl_returns = pd.Series(posterior_mean, index=tickers, name="black_litterman_return")
    diagnostics = pd.DataFrame(
        {
            "original_expected_return": model_returns,
            "equilibrium_return": pi,
            "black_litterman_return": bl_returns,
            "BL_minus_original": bl_returns - model_returns,
            "confidence_used": confidence,
            "BL_rank": bl_returns.rank(ascending=False, method="min").astype(int),
            "original_rank": model_returns.rank(ascending=False, method="min").astype(int),
        }
    ).sort_values("BL_rank")

    agreement = {
        "rank_correlation": _rank_correlation(model_returns, bl_returns),
        "top10_original": model_returns.sort_values(ascending=False).head(10).index.tolist(),
        "top10_bl": bl_returns.sort_values(ascending=False).head(10).index.tolist(),
        "overlap_count": len(
            set(model_returns.sort_values(ascending=False).head(10).index)
            & set(bl_returns.sort_values(ascending=False).head(10).index)
        ),
        "largest_positive_adjustments": (bl_returns - model_returns).sort_values(ascending=False).head(10),
        "largest_negative_adjustments": (bl_returns - model_returns).sort_values(ascending=True).head(10),
        "fallback_used": fallback_used,
        "risk_aversion": delta,
        "tau": tau_value,
        "number_of_tickers": int(n_assets),
        "covariance_shape": sigma_df.shape,
        "expected_returns_shape": model_returns.shape,
        "average_confidence": float(confidence.mean()) if len(confidence) else np.nan,
    }
    return diagnostics, agreement


def print_black_litterman_diagnostics(
    *,
    covariance_matrix: pd.DataFrame,
    expected_returns: pd.Series,
    benchmark_weights: pd.Series | None = None,
    target_confidence: pd.Series | None = None,
    signal_strength: pd.Series | None = None,
    quality_score: pd.Series | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    diagnostics, agreement = compute_black_litterman_diagnostics(
        covariance_matrix=covariance_matrix,
        expected_returns=expected_returns,
        benchmark_weights=benchmark_weights,
        target_confidence=target_confidence,
        signal_strength=signal_strength,
        quality_score=quality_score,
        risk_aversion=risk_aversion,
        tau=tau,
    )

    print("\n===== BLACK-LITTERMAN DIAGNOSTICS =====")
    if diagnostics.empty:
        print("No Black-Litterman diagnostics available.")
        return diagnostics, agreement
    print(f"number of tickers used: {agreement.get('number_of_tickers')}")
    print(f"covariance shape: {agreement.get('covariance_shape')}")
    print(f"expected returns shape: {agreement.get('expected_returns_shape')}")
    print(f"fallback weights used: {agreement.get('fallback_used')}")
    print(f"fallback used: {agreement.get('fallback_used')}")
    print(f"tau: {agreement.get('tau')}")
    print(f"delta: {agreement.get('risk_aversion')}")
    print(f"average confidence: {agreement.get('average_confidence')}")
    print(diagnostics)

    print("\n===== BLACK-LITTERMAN AGREEMENT CHECK =====")
    print(f"rank correlation: {agreement.get('rank_correlation')}")
    print(f"top 10 original: {agreement.get('top10_original')}")
    print(f"top 10 BL: {agreement.get('top10_bl')}")
    print(f"overlap count: {agreement.get('overlap_count')}")
    print("largest positive BL adjustments:")
    print(agreement.get("largest_positive_adjustments"))
    print("largest negative BL adjustments:")
    print(agreement.get("largest_negative_adjustments"))
    return diagnostics, agreement
