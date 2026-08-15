from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from model_mode_comparison import _decision_changes, _metrics
from trend_vs_ema_backtest import _download_reduced_prices
from triple_barrier_labeling import generate_triple_barrier_labels
from walk_forward_backtester import (
    DEFAULT_REDUCED_UNIVERSE,
    WalkForwardConfig,
    run_walk_forward_backtest,
)

OUTPUT_FILE = "full_quant_robustness_walk_forward.csv"
WINDOWS = [
    ("2022-06-01_to_2022-12-31", "2022-06-01", "2022-12-31"),
    ("2023-01-01_to_2023-12-31", "2023-01-01", "2023-12-31"),
    ("2024-01-01_to_2024-12-31", "2024-01-01", "2024-12-31"),
    ("2025-01-01_to_latest", "2025-01-01", None),
]


def _run_window_mode(
    prices_df: pd.DataFrame,
    *,
    window_name: str,
    start_date: str,
    end_date: str | None,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timing_model = "trend_persistence" if mode == "full_quant_research" else "ema"
    target_model = "quant" if mode == "full_quant_research" else "basic"
    safe_name = window_name.replace("-", "").replace("_to_", "__")
    cfg = WalkForwardConfig(
        start_date=start_date,
        end_date=end_date,
        step_size_days=5,
        max_test_dates=30,
        reduced_universe=DEFAULT_REDUCED_UNIVERSE,
        optimizer_generations_backtest=50,
        disable_live_prices=True,
        lookback_window=252,
        min_history_required=252,
        output_predictions=f"full_quant_robustness__{safe_name}__{mode}_predictions.csv",
        output_portfolio_returns=f"full_quant_robustness__{safe_name}__{mode}_portfolio_returns.csv",
        output_summary=f"full_quant_robustness__{safe_name}__{mode}_summary.csv",
        timing_model=timing_model,
        target_model=target_model,
        model_mode=mode,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        predictions, portfolio, summary = run_walk_forward_backtest(prices_df, config=cfg)
        labels = generate_triple_barrier_labels(
            prices_df=prices_df,
            predictions_df=predictions,
            horizons=(5, 10, 20),
            tp_multiple=1.0,
            sl_multiple=1.0,
            output_path=f"full_quant_robustness__{safe_name}__{mode}_triple_barrier_labels.csv",
        )
    return predictions, portfolio, summary, labels


def _verdict(results: pd.DataFrame) -> str:
    valid = results.dropna(subset=["sharpe_difference"])
    if valid.empty:
        return "not robust"
    wins = int((valid["sharpe_difference"] > 0).sum())
    total = len(valid)
    avg_diff = float(valid["sharpe_difference"].mean())
    dd_improved = int((valid["full_quant_max_drawdown"] > valid["baseline_max_drawdown"]).sum())
    if wins == total and avg_diff > 0.25 and dd_improved >= max(1, total - 1):
        return "robust improvement"
    if wins >= max(1, int(np.ceil(total * 0.60))) and avg_diff > 0:
        return "mixed"
    if wins <= 1 or avg_diff <= 0:
        return "not robust"
    return "likely overfit"


def run_full_quant_robustness_walk_forward(
    prices_df: pd.DataFrame | None = None,
    output_path: str | Path = OUTPUT_FILE,
) -> pd.DataFrame:
    if prices_df is None:
        prices_df = _download_reduced_prices(period="5y")
    if prices_df is None or prices_df.empty:
        raise ValueError("No price data available.")

    rows = []
    for window_name, start_date, end_date in WINDOWS:
        baseline_predictions, _, baseline_summary, baseline_labels = _run_window_mode(
            prices_df,
            window_name=window_name,
            start_date=start_date,
            end_date=end_date,
            mode="baseline",
        )
        full_predictions, _, full_summary, full_labels = _run_window_mode(
            prices_df,
            window_name=window_name,
            start_date=start_date,
            end_date=end_date,
            mode="full_quant_research",
        )
        gated_predictions, _, gated_summary, gated_labels = _run_window_mode(
            prices_df,
            window_name=window_name,
            start_date=start_date,
            end_date=end_date,
            mode="regime_gated_full_quant",
        )

        baseline = _metrics(baseline_summary, baseline_labels)
        full = _metrics(full_summary, full_labels)
        gated = _metrics(gated_summary, gated_labels)
        changes = _decision_changes(baseline_predictions, full_predictions)
        gated_changes = _decision_changes(baseline_predictions, gated_predictions)
        overlap = float(changes["jaccard_overlap"].mean()) if not changes.empty else np.nan
        gated_overlap = float(gated_changes["jaccard_overlap"].mean()) if not gated_changes.empty else np.nan
        rows.append(
            {
                "window": window_name,
                "baseline_sharpe": baseline.get("realized_sharpe", np.nan),
                "full_quant_sharpe": full.get("realized_sharpe", np.nan),
                "regime_gated_sharpe": gated.get("realized_sharpe", np.nan),
                "sharpe_difference": full.get("realized_sharpe", np.nan) - baseline.get("realized_sharpe", np.nan),
                "regime_gated_sharpe_difference": gated.get("realized_sharpe", np.nan) - baseline.get("realized_sharpe", np.nan),
                "baseline_return": baseline.get("realized_return", np.nan),
                "full_quant_return": full.get("realized_return", np.nan),
                "regime_gated_return": gated.get("realized_return", np.nan),
                "return_difference": full.get("realized_return", np.nan) - baseline.get("realized_return", np.nan),
                "regime_gated_return_difference": gated.get("realized_return", np.nan) - baseline.get("realized_return", np.nan),
                "baseline_volatility": baseline.get("realized_volatility", np.nan),
                "full_quant_volatility": full.get("realized_volatility", np.nan),
                "regime_gated_volatility": gated.get("realized_volatility", np.nan),
                "baseline_sortino": baseline.get("Sortino", np.nan),
                "full_quant_sortino": full.get("Sortino", np.nan),
                "regime_gated_sortino": gated.get("Sortino", np.nan),
                "baseline_calmar": baseline.get("Calmar", np.nan),
                "full_quant_calmar": full.get("Calmar", np.nan),
                "regime_gated_calmar": gated.get("Calmar", np.nan),
                "baseline_max_drawdown": baseline.get("max_drawdown", np.nan),
                "full_quant_max_drawdown": full.get("max_drawdown", np.nan),
                "regime_gated_max_drawdown": gated.get("max_drawdown", np.nan),
                "baseline_cash": baseline.get("average_cash", np.nan),
                "full_quant_cash": full.get("average_cash", np.nan),
                "regime_gated_cash": gated.get("average_cash", np.nan),
                "baseline_selected_assets": baseline.get("average_selected_count", np.nan),
                "full_quant_selected_assets": full.get("average_selected_count", np.nan),
                "regime_gated_selected_assets": gated.get("average_selected_count", np.nan),
                "baseline_turnover": baseline.get("average_turnover", np.nan),
                "full_quant_turnover": full.get("average_turnover", np.nan),
                "regime_gated_turnover": gated.get("average_turnover", np.nan),
                "baseline_TP_rate": baseline.get("TP_rate", np.nan),
                "full_quant_TP_rate": full.get("TP_rate", np.nan),
                "regime_gated_TP_rate": gated.get("TP_rate", np.nan),
                "baseline_SL_rate": baseline.get("SL_rate", np.nan),
                "full_quant_SL_rate": full.get("SL_rate", np.nan),
                "regime_gated_SL_rate": gated.get("SL_rate", np.nan),
                "baseline_TP_minus_SL": baseline.get("TP_minus_SL", np.nan),
                "full_quant_TP_minus_SL": full.get("TP_minus_SL", np.nan),
                "regime_gated_TP_minus_SL": gated.get("TP_minus_SL", np.nan),
                "average_overlap": overlap,
                "regime_gated_overlap": gated_overlap,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    result.to_csv("regime_gated_full_quant_comparison.csv", index=False)

    display_cols = [
        "window",
        "baseline_sharpe",
        "full_quant_sharpe",
        "regime_gated_sharpe",
        "sharpe_difference",
        "regime_gated_sharpe_difference",
        "baseline_return",
        "full_quant_return",
        "regime_gated_return",
        "baseline_max_drawdown",
        "full_quant_max_drawdown",
        "regime_gated_max_drawdown",
        "baseline_cash",
        "full_quant_cash",
        "regime_gated_cash",
        "average_overlap",
        "regime_gated_overlap",
    ]
    print("\n===== FULL QUANT ROBUSTNESS WALK-FORWARD =====")
    print(result[display_cols].to_string(index=False))
    verdict = _verdict(result)
    print("\n===== FULL QUANT ROBUSTNESS VERDICT =====")
    print(verdict)
    print(f"\nSaved: {Path(output_path).resolve()}")
    print(f"Saved: {Path('regime_gated_full_quant_comparison.csv').resolve()}")
    return result


if __name__ == "__main__":
    run_full_quant_robustness_walk_forward()
