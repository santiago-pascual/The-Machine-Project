from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_series(values: pd.Series | None, index: pd.Index, default: float = 0.0) -> pd.Series:
    if values is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(pd.Series(values).reindex(index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _select_shadow_assets(
    expected_returns: pd.Series,
    signal_strength: pd.Series,
    selection_score: pd.Series,
    max_assets: int,
    signal_threshold: float = 0.15,
    edge_threshold: float | None = None,
    score_threshold: float | None = None,
) -> list[str]:
    expected = expected_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal = signal_strength.reindex(expected.index).fillna(0.0)
    score = selection_score.reindex(expected.index).fillna(0.0)

    mask = (expected > 0) & (signal >= signal_threshold)
    if edge_threshold is not None:
        mask &= expected > edge_threshold
    if score_threshold is not None:
        mask &= score >= score_threshold

    candidates = pd.DataFrame(
        {
            "expected": expected,
            "signal": signal,
            "score": score,
        }
    ).loc[mask]
    if candidates.empty:
        candidates = pd.DataFrame(
            {
                "expected": expected,
                "signal": signal,
                "score": score,
            }
        ).loc[expected > 0]

    if candidates.empty:
        return []

    candidates["rank_score"] = (
        0.50 * candidates["expected"].rank(pct=True)
        + 0.30 * candidates["signal"].rank(pct=True)
        + 0.20 * candidates["score"].rank(pct=True)
    )
    n_assets = int(max(1, min(max_assets, len(candidates))))
    return candidates.sort_values(["rank_score", "expected"], ascending=False).head(n_assets).index.tolist()


def _evaluate_shadow_portfolio(
    selected_assets: list[str],
    expected_returns: pd.Series,
    returns_df: pd.DataFrame,
    covariance_matrix: pd.DataFrame | None,
    rf_daily: float,
    cash_weight: float,
) -> dict[str, object]:
    if not selected_assets:
        return {
            "selected_assets": [],
            "selected_n": 0,
            "expected_portfolio_return": 0.0,
            "volatility": 0.0,
            "sharpe": np.nan,
            "cash": 1.0,
            "concentration_hhi": 0.0,
        }

    selected_expected = expected_returns.reindex(selected_assets).fillna(0.0)
    positive = selected_expected.clip(lower=0.0)
    if float(positive.sum()) > 0:
        raw_weights = positive / float(positive.sum())
    else:
        raw_weights = pd.Series(1.0 / len(selected_assets), index=selected_assets)

    raw_weights = raw_weights.clip(upper=0.50)
    raw_weights = raw_weights / max(float(raw_weights.sum()), 1e-12)
    invested_weight = max(0.0, 1.0 - float(cash_weight))
    asset_weights = raw_weights * invested_weight
    portfolio_return = float(np.dot(asset_weights.reindex(selected_assets), selected_expected))

    if covariance_matrix is not None and not covariance_matrix.empty:
        cov = covariance_matrix.reindex(index=selected_assets, columns=selected_assets).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        variance = float(
            asset_weights.reindex(selected_assets).to_numpy().T @ cov.to_numpy() @ asset_weights.reindex(selected_assets).to_numpy()
        )
        volatility = float(np.sqrt(max(variance, 0.0)))
    else:
        aligned_returns = returns_df.reindex(columns=selected_assets).dropna(how="any")
        volatility = float(aligned_returns.dot(asset_weights.reindex(selected_assets)).std()) if not aligned_returns.empty else 0.0

    sharpe = (portfolio_return - rf_daily * invested_weight) / volatility if volatility > 0 else np.nan
    concentration = float(np.square(raw_weights).sum())
    return {
        "selected_assets": selected_assets,
        "selected_n": len(selected_assets),
        "expected_portfolio_return": portfolio_return,
        "volatility": volatility,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "cash": float(cash_weight),
        "concentration_hhi": concentration,
    }


def _turnover(current_assets: list[str], shadow_assets: list[str]) -> float:
    current = set(current_assets)
    shadow = set(shadow_assets)
    union = current | shadow
    if not union:
        return 0.0
    return float(len(current ^ shadow) / len(union))


def _sensitivity_flag(row: pd.Series, baseline: dict[str, object]) -> str:
    turnover = float(row.get("turnover_vs_current", 0.0))
    selected_delta = abs(int(row.get("selected_n", 0)) - int(baseline.get("selected_n", 0)))
    sharpe = float(row.get("sharpe", np.nan))
    baseline_sharpe = float(baseline.get("sharpe", np.nan))
    sharpe_delta = abs(sharpe - baseline_sharpe) if np.isfinite(sharpe) and np.isfinite(baseline_sharpe) else 0.0

    if turnover >= 0.50 or selected_delta >= 2 or sharpe_delta >= 0.10:
        return "dangerously sensitive"
    if turnover >= 0.25 or selected_delta >= 1 or sharpe_delta >= 0.04:
        return "sensitive"
    return "stable"


def _test_grid() -> list[dict[str, object]]:
    return [
        {"group": "expected_return", "heuristic": "dead_zone", "current": 0.001, "lower": 0.0005, "higher": 0.0015},
        {"group": "expected_return", "heuristic": "constant_penalty", "current": 0.001, "lower": 0.0005, "higher": 0.0015},
        {"group": "expected_return", "heuristic": "signal_power", "current": 1.5, "lower": 1.2, "higher": 1.8},
        {"group": "expected_return", "heuristic": "weak_signal_floor", "current": 0.3, "lower": 0.2, "higher": 0.4},
        {"group": "selection", "heuristic": "signal_strength_threshold", "current": 0.2, "lower": 0.15, "higher": 0.25},
        {"group": "selection", "heuristic": "edge_quantile", "current": 0.4, "lower": 0.3, "higher": 0.5},
        {"group": "selection", "heuristic": "score_quantile", "current": 0.70, "lower": 0.60, "higher": 0.80},
        {"group": "selection", "heuristic": "post_timing_edge_threshold", "current": 0.0002, "lower": 0.0001, "higher": 0.0004},
        {"group": "regime", "heuristic": "risk_on_multiplier", "current": 1.2, "lower": 1.0, "higher": 1.35},
        {"group": "regime", "heuristic": "neutral_multiplier", "current": 0.85, "lower": 0.70, "higher": 1.0},
        {"group": "regime", "heuristic": "risk_off_multiplier", "current": 0.6, "lower": 0.45, "higher": 0.75},
        {"group": "regime", "heuristic": "high_volatility_multiplier", "current": 1.2, "lower": 1.0, "higher": 1.35},
    ]


def _apply_shadow_adjustment(
    expected_returns: pd.Series,
    signal_strength: pd.Series,
    heuristic: str,
    current_value: float,
    test_value: float,
    regime_type: str,
) -> pd.Series:
    adjusted = expected_returns.copy()
    signal = signal_strength.reindex(adjusted.index).fillna(0.0).clip(lower=0.0, upper=1.0)

    if heuristic == "dead_zone":
        adjusted = adjusted.mask(adjusted.abs() < test_value, 0.0)
    elif heuristic == "constant_penalty":
        adjusted = adjusted + current_value - test_value
    elif heuristic == "signal_power":
        ratio = (signal.pow(test_value) / signal.pow(current_value).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        adjusted = adjusted * ratio
    elif heuristic == "weak_signal_floor":
        adjusted = adjusted.mask(signal < test_value, adjusted.clip(upper=0.0))
    elif heuristic.endswith("_multiplier"):
        active = (
            (heuristic == "risk_on_multiplier" and regime_type == "risk_on")
            or (heuristic == "neutral_multiplier" and regime_type == "neutral")
            or (heuristic == "risk_off_multiplier" and regime_type == "risk_off")
            or (heuristic == "high_volatility_multiplier" and regime_type == "high_volatility")
        )
        if active and current_value != 0:
            adjusted = adjusted * (test_value / current_value)

    return adjusted.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_heuristic_calibration_diagnostics(
    *,
    expected_returns: pd.Series,
    signal_strength: pd.Series,
    selection_score: pd.Series,
    current_selected_assets: list[str],
    returns_df: pd.DataFrame,
    covariance_matrix: pd.DataFrame | None,
    rf_daily: float,
    cash_weight: float,
    regime_type: str,
) -> pd.DataFrame:
    index = pd.Index(expected_returns.index.astype(str))
    expected = _clean_series(expected_returns, index, default=0.0)
    signal = _clean_series(signal_strength, index, default=0.0)
    score = _clean_series(selection_score, index, default=0.0)
    max_assets = max(1, len(current_selected_assets))

    baseline = _evaluate_shadow_portfolio(
        selected_assets=current_selected_assets,
        expected_returns=expected,
        returns_df=returns_df,
        covariance_matrix=covariance_matrix,
        rf_daily=rf_daily,
        cash_weight=cash_weight,
    )

    rows: list[dict[str, object]] = []
    positive_returns = expected[expected > 0]
    current_edge_threshold = float(positive_returns.quantile(0.4)) if len(positive_returns) else 0.0001
    current_score_threshold = float(score.quantile(0.70)) if len(score) else 1.0

    for config in _test_grid():
        for scenario in ("lower", "current", "higher"):
            test_value = float(config[scenario])
            heuristic = str(config["heuristic"])
            adjusted = _apply_shadow_adjustment(
                expected_returns=expected,
                signal_strength=signal,
                heuristic=heuristic,
                current_value=float(config["current"]),
                test_value=test_value,
                regime_type=regime_type,
            )

            signal_threshold = 0.15
            edge_threshold: float | None = None
            score_threshold: float | None = None
            if heuristic == "signal_strength_threshold":
                signal_threshold = test_value
            elif heuristic == "edge_quantile":
                positives = adjusted[adjusted > 0]
                edge_threshold = float(positives.quantile(test_value)) if len(positives) else current_edge_threshold
            elif heuristic == "score_quantile":
                score_threshold = float(score.quantile(test_value)) if len(score) else current_score_threshold
            elif heuristic == "post_timing_edge_threshold":
                edge_threshold = test_value

            selected_assets = _select_shadow_assets(
                adjusted,
                signal,
                score,
                max_assets=max_assets,
                signal_threshold=signal_threshold,
                edge_threshold=edge_threshold,
                score_threshold=score_threshold,
            )
            evaluated = _evaluate_shadow_portfolio(
                selected_assets=selected_assets,
                expected_returns=adjusted,
                returns_df=returns_df,
                covariance_matrix=covariance_matrix,
                rf_daily=rf_daily,
                cash_weight=cash_weight,
            )
            rows.append(
                {
                    "group": config["group"],
                    "heuristic": heuristic,
                    "scenario": scenario,
                    "test_value": test_value,
                    "selected_tickers": ", ".join(selected_assets),
                    "turnover_vs_current": _turnover(current_selected_assets, selected_assets),
                    **{k: v for k, v in evaluated.items() if k != "selected_assets"},
                }
            )

    diagnostics = pd.DataFrame(rows)
    diagnostics["sensitivity_flag"] = diagnostics.apply(lambda row: _sensitivity_flag(row, baseline), axis=1)
    diagnostics["calibration_note"] = diagnostics["sensitivity_flag"].map(
        {
            "stable": "Reasonable in this shadow test; keep monitoring.",
            "sensitive": "Material impact; calibrate with forecast history / walk-forward tests.",
            "dangerously sensitive": "Can materially change portfolio; needs calibration before tuning.",
        }
    )
    return diagnostics


def print_heuristic_calibration_diagnostics(
    *,
    expected_returns: pd.Series,
    signal_strength: pd.Series,
    selection_score: pd.Series,
    current_selected_assets: list[str],
    returns_df: pd.DataFrame,
    covariance_matrix: pd.DataFrame | None,
    rf_daily: float,
    cash_weight: float,
    regime_type: str,
) -> pd.DataFrame:
    diagnostics = build_heuristic_calibration_diagnostics(
        expected_returns=expected_returns,
        signal_strength=signal_strength,
        selection_score=selection_score,
        current_selected_assets=current_selected_assets,
        returns_df=returns_df,
        covariance_matrix=covariance_matrix,
        rf_daily=rf_daily,
        cash_weight=cash_weight,
        regime_type=regime_type,
    )
    print("\n===== HEURISTIC CALIBRATION DIAGNOSTICS =====")
    display_cols = [
        "group",
        "heuristic",
        "scenario",
        "test_value",
        "selected_n",
        "selected_tickers",
        "expected_portfolio_return",
        "volatility",
        "sharpe",
        "cash",
        "turnover_vs_current",
        "concentration_hhi",
        "sensitivity_flag",
        "calibration_note",
    ]
    print(diagnostics[display_cols].to_string(index=False))

    dangerous = diagnostics[diagnostics["sensitivity_flag"] == "dangerously sensitive"]
    if not dangerous.empty:
        print("\nDangerously sensitive heuristics:")
        for heuristic in dangerous["heuristic"].drop_duplicates().tolist():
            print(f"- {heuristic}: needs calibration using forecast history or walk-forward backtest.")
    return diagnostics
