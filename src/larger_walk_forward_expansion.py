from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trend_vs_ema_backtest import _download_reduced_prices
from triple_barrier_labeling import generate_triple_barrier_labels
from walk_forward_backtester import (
    DEFAULT_REDUCED_UNIVERSE,
    WalkForwardConfig,
    run_walk_forward_backtest,
)

BASELINE_OUTPUT = "larger_walk_forward_baseline.csv"
GATED_OUTPUT = "larger_walk_forward_regime_gated.csv"
SUMMARY_OUTPUT = "larger_walk_forward_summary.csv"


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _metrics_from_outputs(summary: pd.DataFrame, labels: pd.DataFrame, mode: str, elapsed_seconds: float) -> dict[str, object]:
    row = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
    selected = labels[labels["selected"].astype(bool)] if not labels.empty and "selected" in labels.columns else labels.iloc[0:0]
    if selected.empty:
        tp_rate = np.nan
        sl_rate = np.nan
        timeout_rate = np.nan
        selected_sample_size = 0
    else:
        selected_sample_size = len(selected)
        tp_rate = float(selected["first_touch_type"].eq("take_profit").mean())
        sl_rate = float(selected["first_touch_type"].eq("stop_loss").mean())
        timeout_rate = float(selected["first_touch_type"].eq("vertical_timeout").mean())

    promotion_blocked = _promotion_still_blocked()
    return {
        "model_mode": mode,
        "number_of_test_dates": int(_safe_float(row.get("number_of_test_dates", 0), 0)),
        "selected_only_sample_size": selected_sample_size,
        "passes_minimum_150": bool(selected_sample_size >= 150),
        "promotion_still_blocked": promotion_blocked,
        "realized_return": _safe_float(row.get("realized_return", np.nan)),
        "realized_volatility": _safe_float(row.get("realized_volatility", np.nan)),
        "Sharpe": _safe_float(row.get("realized_sharpe", np.nan)),
        "Sortino": _safe_float(row.get("Sortino", np.nan)),
        "Calmar": _safe_float(row.get("Calmar", np.nan)),
        "max_drawdown": _safe_float(row.get("max_drawdown", np.nan)),
        "average_cash": _safe_float(row.get("average_cash", np.nan)),
        "average_turnover": _safe_float(row.get("average_turnover", np.nan)),
        "TP_rate": tp_rate,
        "SL_rate": sl_rate,
        "TP_minus_SL": tp_rate - sl_rate if pd.notna(tp_rate) and pd.notna(sl_rate) else np.nan,
        "timeout_rate": timeout_rate,
        "hit_rate_5d": _safe_float(row.get("hit_rate_5d", np.nan)),
        "hit_rate_10d": _safe_float(row.get("hit_rate_10d", np.nan)),
        "hit_rate_20d": _safe_float(row.get("hit_rate_20d", np.nan)),
        "direction_accuracy_5d": _safe_float(row.get("direction_accuracy_5d", np.nan)),
        "direction_accuracy_10d": _safe_float(row.get("direction_accuracy_10d", np.nan)),
        "direction_accuracy_20d": _safe_float(row.get("direction_accuracy_20d", np.nan)),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _promotion_still_blocked() -> bool:
    dashboard = _read_csv("research_dashboard_summary.csv")
    if not dashboard.empty and {"metric", "value"}.issubset(dashboard.columns):
        status = dashboard.loc[dashboard["metric"].astype(str).eq("promotion_status"), "value"]
        if not status.empty and str(status.iloc[-1]).lower() == "blocked":
            return True
        pbo = dashboard.loc[dashboard["metric"].astype(str).eq("PBO_proxy"), "value"]
        if not pbo.empty and _safe_float(pbo.iloc[-1], 1.0) > 0.30:
            return True
        warning = dashboard.loc[dashboard["metric"].astype(str).eq("overfitting_warning_level"), "value"]
        if not warning.empty and str(warning.iloc[-1]).lower() in {"high", "extreme"}:
            return True
    return False


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _run_mode(
    prices_df: pd.DataFrame,
    *,
    mode: str,
    max_test_dates: int,
    start_date: str,
    end_date: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if mode == "regime_gated_full_quant":
        timing_model = "ema"
        target_model = "basic"
        output_prefix = "larger_walk_forward_regime_gated"
    else:
        timing_model = "ema"
        target_model = "basic"
        output_prefix = "larger_walk_forward_baseline"

    config = WalkForwardConfig(
        start_date=start_date,
        end_date=end_date,
        step_size_days=5,
        max_test_dates=max_test_dates,
        reduced_universe=DEFAULT_REDUCED_UNIVERSE,
        optimizer_generations_backtest=50,
        disable_live_prices=True,
        lookback_window=252,
        min_history_required=252,
        output_predictions=f"{output_prefix}_predictions.csv",
        output_portfolio_returns=f"{output_prefix}_portfolio_returns.csv",
        output_summary=f"{output_prefix}_raw_summary.csv",
        timing_model=timing_model,
        target_model=target_model,
        model_mode=mode,
    )
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        predictions, portfolio, summary = run_walk_forward_backtest(prices_df, config=config)
        labels = generate_triple_barrier_labels(
            prices_df=prices_df,
            predictions_df=predictions,
            horizons=(5, 10, 20),
            tp_multiple=1.0,
            sl_multiple=1.0,
            output_path=f"{output_prefix}_triple_barrier_labels.csv",
        )
    elapsed = time.time() - start
    output_file = BASELINE_OUTPUT if mode == "baseline" else GATED_OUTPUT
    portfolio.to_csv(output_file, index=False)
    metrics = _metrics_from_outputs(summary, labels, mode, elapsed)
    return predictions, portfolio, summary, metrics


def print_expansion_report(summary: pd.DataFrame) -> None:
    display_cols = [
        "model_mode",
        "number_of_test_dates",
        "selected_only_sample_size",
        "passes_minimum_150",
        "promotion_still_blocked",
        "realized_return",
        "realized_volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_cash",
        "average_turnover",
        "TP_rate",
        "SL_rate",
        "TP_minus_SL",
        "hit_rate_5d",
        "direction_accuracy_5d",
    ]
    print("\n===== LARGER WALK-FORWARD DATA EXPANSION =====")
    print(summary[display_cols].to_string(index=False))
    print("\n===== SAMPLE SIZE CHECK =====")
    for _, row in summary.iterrows():
        blocked = not bool(row["passes_minimum_150"])
        print(
            f"{row['model_mode']}: selected-only sample size = {int(row['selected_only_sample_size'])}; "
            f"passes minimum 150 = {bool(row['passes_minimum_150'])}; "
            f"promotion still blocked = {bool(row.get('promotion_still_blocked', blocked))}"
        )


def run_larger_walk_forward_expansion(
    *,
    prices_df: pd.DataFrame | None = None,
    max_test_dates: int = 100,
    run_regime_gated: bool = True,
    start_date: str = "2022-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    if prices_df is None:
        prices_df = _download_reduced_prices(period="5y")
    if prices_df is None or prices_df.empty:
        raise ValueError("No price data available for larger walk-forward expansion.")

    rows: list[dict[str, object]] = []
    _, _, _, baseline_metrics = _run_mode(
        prices_df,
        mode="baseline",
        max_test_dates=max_test_dates,
        start_date=start_date,
        end_date=end_date,
    )
    rows.append(baseline_metrics)

    if run_regime_gated:
        try:
            _, _, _, gated_metrics = _run_mode(
                prices_df,
                mode="regime_gated_full_quant",
                max_test_dates=max_test_dates,
                start_date=start_date,
                end_date=end_date,
            )
            rows.append(gated_metrics)
        except Exception as exc:
            rows.append(
                {
                    "model_mode": "regime_gated_full_quant",
                    "number_of_test_dates": 0,
                    "selected_only_sample_size": 0,
                    "passes_minimum_150": False,
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    print_expansion_report(summary)
    print(f"\nSaved: {Path(BASELINE_OUTPUT).resolve()}")
    if run_regime_gated:
        print(f"Saved: {Path(GATED_OUTPUT).resolve()}")
    print(f"Saved: {Path(SUMMARY_OUTPUT).resolve()}")
    return summary


if __name__ == "__main__":
    run_larger_walk_forward_expansion()
