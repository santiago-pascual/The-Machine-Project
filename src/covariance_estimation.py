from __future__ import annotations

import numpy as np
import pandas as pd


def historical_covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
    if returns_df.empty:
        raise ValueError("returns_df cannot be empty.")
    return returns_df.cov()


def manual_diagonal_shrinkage_covariance(
    returns_df: pd.DataFrame,
    shrinkage_intensity: float = 0.10,
) -> pd.DataFrame:
    if returns_df.empty:
        raise ValueError("returns_df cannot be empty.")
    clean_returns = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean_returns.shape[0] < 3:
        return historical_covariance(returns_df)

    shrinkage = float(np.clip(shrinkage_intensity, 0.0, 1.0))
    sample_cov = clean_returns.cov()
    target = pd.DataFrame(
        np.diag(np.diag(sample_cov.to_numpy(dtype=float))),
        index=sample_cov.index,
        columns=sample_cov.columns,
    )
    return (1.0 - shrinkage) * sample_cov + shrinkage * target


def ledoit_wolf_covariance(
    returns_df: pd.DataFrame,
    shrinkage_intensity: float = 0.10,
) -> tuple[pd.DataFrame, str, float]:
    if returns_df.empty:
        raise ValueError("returns_df cannot be empty.")
    clean_returns = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean_returns.shape[0] < 3:
        return historical_covariance(returns_df), "historical", 0.0

    try:
        from sklearn.covariance import LedoitWolf
    except Exception:
        return (
            manual_diagonal_shrinkage_covariance(clean_returns, shrinkage_intensity=shrinkage_intensity),
            "manual_diagonal_shrinkage",
            float(np.clip(shrinkage_intensity, 0.0, 1.0)),
        )

    model = LedoitWolf().fit(clean_returns.to_numpy(dtype=float))
    return (
        pd.DataFrame(model.covariance_, index=clean_returns.columns, columns=clean_returns.columns),
        "sklearn_ledoit_wolf",
        float(getattr(model, "shrinkage_", np.nan)),
    )


def calculate_covariance(
    returns_df: pd.DataFrame,
    method: str = "ledoit_wolf",
    shrinkage_intensity: float = 0.10,
) -> pd.DataFrame:
    method_normalized = str(method).lower()
    if method_normalized == "historical":
        return historical_covariance(returns_df)
    if method_normalized == "manual_diagonal_shrinkage":
        return manual_diagonal_shrinkage_covariance(
            returns_df,
            shrinkage_intensity=shrinkage_intensity,
        )
    if method_normalized == "ledoit_wolf":
        covariance, _, _ = ledoit_wolf_covariance(
            returns_df,
            shrinkage_intensity=shrinkage_intensity,
        )
        return covariance
    raise ValueError("covariance_method must be 'historical', 'ledoit_wolf', or 'manual_diagonal_shrinkage'.")


def covariance_method_metadata(
    returns_df: pd.DataFrame,
    method: str = "ledoit_wolf",
    shrinkage_intensity: float = 0.10,
) -> dict[str, float | str]:
    method_normalized = str(method).lower()
    if method_normalized == "historical":
        return {
            "shrinkage_method_used": "historical",
            "shrinkage_intensity": 0.0,
        }
    if method_normalized == "manual_diagonal_shrinkage":
        return {
            "shrinkage_method_used": "manual_diagonal_shrinkage",
            "shrinkage_intensity": float(np.clip(shrinkage_intensity, 0.0, 1.0)),
        }
    if method_normalized == "ledoit_wolf":
        _, shrinkage_method_used, used_intensity = ledoit_wolf_covariance(
            returns_df,
            shrinkage_intensity=shrinkage_intensity,
        )
        return {
            "shrinkage_method_used": shrinkage_method_used,
            "shrinkage_intensity": used_intensity,
        }
    return {
        "shrinkage_method_used": "unknown",
        "shrinkage_intensity": np.nan,
    }


def covariance_diagnostics(
    cov_matrix: pd.DataFrame,
    method: str,
    shrinkage_method_used: str | None = None,
    shrinkage_intensity: float | None = None,
) -> dict[str, float | str]:
    matrix = cov_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    if matrix.size == 0:
        return {
            "condition_number": np.nan,
            "determinant": np.nan,
            "average_correlation": np.nan,
            "covariance_method_used": method,
            "shrinkage_method_used": shrinkage_method_used or method,
            "shrinkage_intensity": 0.0 if shrinkage_intensity is None else float(shrinkage_intensity),
        }

    condition_number = float(np.linalg.cond(matrix))
    determinant = float(np.linalg.det(matrix))
    diagonal = np.sqrt(np.clip(np.diag(matrix), 1e-16, None))
    corr = matrix / np.outer(diagonal, diagonal)
    np.fill_diagonal(corr, np.nan)
    average_correlation = float(np.nanmean(corr)) if corr.size else np.nan
    return {
        "condition_number": condition_number,
        "determinant": determinant,
        "average_correlation": average_correlation,
        "covariance_method_used": method,
        "shrinkage_method_used": shrinkage_method_used or method,
        "shrinkage_intensity": 0.0 if shrinkage_intensity is None else float(shrinkage_intensity),
    }


def print_covariance_diagnostics(
    cov_matrix: pd.DataFrame,
    method: str,
    shrinkage_method_used: str | None = None,
    shrinkage_intensity: float | None = None,
) -> dict[str, float | str]:
    diagnostics = covariance_diagnostics(
        cov_matrix,
        method,
        shrinkage_method_used=shrinkage_method_used,
        shrinkage_intensity=shrinkage_intensity,
    )
    print("\n===== COVARIANCE DIAGNOSTICS =====")
    print(f"condition number: {float(diagnostics['condition_number']):.6e}")
    print(f"determinant: {float(diagnostics['determinant']):.6e}")
    print(f"average correlation: {float(diagnostics['average_correlation']):.6f}")
    print(f"covariance method used: {diagnostics['covariance_method_used']}")
    print(f"shrinkage method used: {diagnostics['shrinkage_method_used']}")
    print(f"shrinkage intensity: {float(diagnostics['shrinkage_intensity']):.6f}")
    return diagnostics
