from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from ema_timing_model import apply_timing_to_expected_returns, compute_asset_timing
from expected_returns_model import compute_expected_returns
from exposure_control import compute_net_exposure
from financial_data_system import (
    calculate_daily_returns,
    download_close_prices,
    generate_target_prices,
    get_risk_free_rate,
)
from market_regime_model import compute_market_regime_model
from portfolio_optimizer import PortfolioOptimizer
from quant_target_model import generate_quant_targets

TEST_TICKERS = ["AAPL", "NVDA", "MSTR", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ"]
LOOKBACK_DAYS = 252
OPTIMIZER_TEST_GENERATIONS = 75


def _target_flags(row: pd.Series) -> str:
    reasons: list[str] = []
    old = float(row["baseline_target"])
    quant = float(row["quant_target"])
    if old != 0 and abs(quant / old - 1.0) > 0.15:
        reasons.append("quant_vs_old_diff_gt_15pct")
    if float(row["target_confidence"]) < 0.35:
        reasons.append("low_target_confidence")
    if old != 0:
        gbm_ref = 0.5 * (float(row["gbm_median_target"]) + float(row["gbm_expected_target"]))
        kalman_direction = float(row["kalman_target"]) - old
        ou_direction = float(row["ou_target"]) - old
        ou_against = np.sign(kalman_direction) != np.sign(ou_direction)
        ou_far = abs(float(row["ou_target"]) - gbm_ref) / abs(old) > 0.08
        if ou_against and ou_far:
            reasons.append("ou_pulling_against_kalman_gbm")
    if float(row["gbm_median_target"]) != 0:
        gbm_gap = float(row["gbm_expected_target"]) / float(row["gbm_median_target"]) - 1.0
        if gbm_gap > 0.05:
            reasons.append("gbm_expected_much_above_median")
    return ", ".join(reasons) if reasons else "ok"


def _run_variant(
    name: str,
    prices_df: pd.DataFrame,
    rf_daily: float,
    regime: dict[str, float | str],
    use_quant_blend: bool,
    blend_weight: float,
) -> dict[str, object]:
    returns_df = calculate_daily_returns(prices_df)
    base_targets = generate_target_prices(prices_df, method="volatility_adjusted", manual_targets=None)
    quant_outputs = generate_quant_targets(
        prices_df=prices_df,
        old_target_prices=base_targets,
        regime_type=str(regime["regime"]),
        horizon_days=20,
        blend_weight=blend_weight,
    )
    target_prices = (
        quant_outputs["final_blended_target"].reindex(prices_df.columns)
        if use_quant_blend
        else base_targets.reindex(prices_df.columns)
    )

    expected_returns, diagnostics = compute_expected_returns(
        prices_df=prices_df,
        target_prices=target_prices,
        use_live_prices=False,
        risk_free_daily=rf_daily,
        center_distribution=False,
        dead_zone=0.001,
        regime_score=float(regime["risk_score"]),
        regime_type=str(regime["regime"]),
        regime_signal_threshold=0.2,
    )
    diagnostics_df = pd.DataFrame(diagnostics).T.reindex(prices_df.columns)
    signal_strength = pd.to_numeric(diagnostics_df.get("signal_strength", 0.0), errors="coerce").fillna(0.0)

    timing_df = compute_asset_timing(prices_df, timing_mode="daily")
    timing_adjusted_returns = apply_timing_to_expected_returns(expected_returns, timing_df)
    candidates = timing_adjusted_returns[(timing_adjusted_returns > 0) & (signal_strength > 0.15)]
    if candidates.empty:
        selected_assets = timing_adjusted_returns.sort_values(ascending=False).head(4).index.tolist()
    else:
        selected_assets = candidates.sort_values(ascending=False).head(min(4, len(candidates))).index.tolist()
    if len(selected_assets) < 2:
        selected_assets = timing_adjusted_returns.sort_values(ascending=False).head(2).index.tolist()

    selected_returns = returns_df[selected_assets]
    selected_expected = timing_adjusted_returns.reindex(selected_assets).fillna(0.0)

    optimizer = PortfolioOptimizer(
        returns_df=selected_returns,
        rf_daily=rf_daily,
        expected_daily_returns=selected_expected,
        use_expected_returns=True,
        alpha=0.5,
        no_opportunity=False,
        defensive_mode=True,
        max_weight=0.50,
        regime_score=float(regime["risk_score"]),
        regime_type=str(regime["regime"]),
        regime_confidence=float(regime.get("regime_confidence", 0.3)),
        n_generations=OPTIMIZER_TEST_GENERATIONS,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        weights, sharpe, best_return, best_volatility, _ = optimizer.optimize()

    raw_weights = pd.Series(weights, index=selected_assets, dtype=float)
    raw_weights = raw_weights.clip(lower=0.0)
    raw_weights = raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else pd.Series(1 / len(selected_assets), index=selected_assets)

    exposure_info = compute_net_exposure(
        regime_score=float(regime["risk_score"]),
        regime_confidence=float(regime.get("regime_confidence", 0.3)),
        expected_returns=timing_adjusted_returns,
        signal_strengths=signal_strength,
        timeframe="daily",
        market_mode_override=None,
    )
    net_exposure = float(exposure_info["net_exposure"])
    final_weights = (raw_weights * net_exposure).sort_values(ascending=False)
    cash_weight = max(0.0, 1.0 - float(final_weights.sum()))
    final_weights.loc["CASH"] = cash_weight

    target_report = pd.DataFrame(
        {
            "baseline_target": base_targets.reindex(prices_df.columns),
            "quant_target": quant_outputs["quant_target_price"].reindex(prices_df.columns),
            "blended_target_5pct": quant_outputs["final_blended_target"].reindex(prices_df.columns),
            "gbm_median_target": quant_outputs["gbm_median_target"].reindex(prices_df.columns),
            "gbm_expected_target": quant_outputs["gbm_expected_target"].reindex(prices_df.columns),
            "kalman_target": quant_outputs["kalman_target"].reindex(prices_df.columns),
            "ou_target": quant_outputs["ou_target"].reindex(prices_df.columns),
            "target_confidence": quant_outputs["target_confidence"].reindex(prices_df.columns),
        }
    )
    target_report["old_vs_quant_diff_pct"] = (
        target_report["quant_target"] / target_report["baseline_target"].replace(0, np.nan) - 1.0
    ) * 100.0
    target_report["suspicious_flags"] = target_report.apply(_target_flags, axis=1)

    return {
        "name": name,
        "target_report": target_report,
        "expected_returns": expected_returns,
        "timing_adjusted_returns": timing_adjusted_returns,
        "selected_assets": selected_assets,
        "weights": final_weights,
        "cash": cash_weight,
        "best_return": float(best_return),
        "best_volatility": float(best_volatility),
        "sharpe": float(sharpe),
    }


def run_quant_blend_comparison() -> None:
    prices_df = download_close_prices(TEST_TICKERS, days=LOOKBACK_DAYS)
    returns_df = calculate_daily_returns(prices_df)
    regime = compute_market_regime_model(prices_df=prices_df, returns_df=returns_df)
    _, rf_daily = get_risk_free_rate()

    baseline = _run_variant(
        name="BASELINE",
        prices_df=prices_df,
        rf_daily=rf_daily,
        regime=regime,
        use_quant_blend=False,
        blend_weight=0.05,
    )
    blend = _run_variant(
        name="QUANT_BLEND_5PCT",
        prices_df=prices_df,
        rf_daily=rf_daily,
        regime=regime,
        use_quant_blend=True,
        blend_weight=0.05,
    )

    selected_baseline = set(baseline["selected_assets"])
    selected_blend = set(blend["selected_assets"])
    union = selected_baseline | selected_blend
    turnover = len(selected_baseline ^ selected_blend) / len(union) if union else 0.0

    print("\n===== BASELINE vs QUANT_BLEND_5PCT TARGET REPORT =====")
    target_comparison = baseline["target_report"].copy()
    target_comparison["baseline_expected_daily_return"] = baseline["expected_returns"].reindex(target_comparison.index)
    target_comparison["blend_expected_daily_return"] = blend["expected_returns"].reindex(target_comparison.index)
    print(target_comparison.sort_values("old_vs_quant_diff_pct", key=lambda s: s.abs(), ascending=False))

    print("\n===== BASELINE vs QUANT_BLEND_5PCT SUMMARY =====")
    summary = pd.DataFrame(
        {
            "BASELINE": {
                "selected_assets": baseline["selected_assets"],
                "cash": baseline["cash"],
                "portfolio_expected_return": baseline["best_return"],
                "portfolio_volatility": baseline["best_volatility"],
                "sharpe": baseline["sharpe"],
            },
            "QUANT_BLEND_5PCT": {
                "selected_assets": blend["selected_assets"],
                "cash": blend["cash"],
                "portfolio_expected_return": blend["best_return"],
                "portfolio_volatility": blend["best_volatility"],
                "sharpe": blend["sharpe"],
            },
        }
    )
    print(summary)

    print("\n===== WEIGHTS =====")
    weights_report = pd.concat(
        [
            baseline["weights"].rename("BASELINE"),
            blend["weights"].rename("QUANT_BLEND_5PCT"),
        ],
        axis=1,
    ).fillna(0.0)
    print(weights_report)

    print("\n===== CHANGE FLAGS =====")
    print(f"selection_turnover_vs_baseline: {turnover:.4f}")
    print(f"selection_changed_materially: {turnover > 0.50}")
    print(f"sharpe_delta: {blend['sharpe'] - baseline['sharpe']:.6f}")
    suspicious = target_comparison[target_comparison["suspicious_flags"] != "ok"]["suspicious_flags"]
    print(f"suspicious_target_flags_count: {len(suspicious)}")
    print(suspicious)
    nearly_identical = (
        turnover == 0
        and abs(blend["sharpe"] - baseline["sharpe"]) < 1e-4
        and abs(blend["cash"] - baseline["cash"]) < 1e-4
    )
    print(f"results_nearly_identical: {nearly_identical}")


if __name__ == "__main__":
    run_quant_blend_comparison()
