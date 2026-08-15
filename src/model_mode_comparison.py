from __future__ import annotations

import contextlib
import io
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

DEFAULT_OUTPUT_FILE = "model_mode_comparison.csv"
REGIME_GATED_OUTPUT_FILE = "regime_gated_full_quant_comparison.csv"
CALIBRATED_FORECAST_RESEARCH_OUTPUT_FILE = "calibrated_forecast_research_comparison.csv"
RAW_TARGET_RESEARCH_OUTPUT_FILE = "raw_target_research_comparison.csv"


def _read_csv_safe(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _summary_metrics_for_mode(summary: pd.DataFrame, mode: str) -> dict[str, float]:
    if summary.empty or "model_mode" not in summary.columns:
        return {}
    rows = summary[summary["model_mode"].astype(str).eq(mode)]
    if rows.empty:
        return {}
    row = rows.iloc[-1]
    return {
        "realized_return": float(row.get("realized_return", np.nan)),
        "volatility": float(row.get("realized_volatility", np.nan)),
        "Sharpe": float(row.get("Sharpe", row.get("realized_sharpe", np.nan))),
        "Sortino": float(row.get("Sortino", np.nan)),
        "Calmar": float(row.get("Calmar", np.nan)),
        "max_drawdown": float(row.get("max_drawdown", np.nan)),
        "cash": float(row.get("average_cash", np.nan)),
        "turnover": float(row.get("average_turnover", np.nan)),
        "TP_rate": float(row.get("TP_rate", np.nan)),
        "SL_rate": float(row.get("SL_rate", np.nan)),
        "TP_minus_SL": float(row.get("TP_minus_SL", np.nan)),
        "hit_rate": float(row.get("hit_rate_20d", row.get("hit_rate", np.nan))),
        "sample_size": float(row.get("selected_only_sample_size", row.get("number_of_test_dates", np.nan))),
    }


def _calibrated_shadow_metrics(shadow_results: pd.DataFrame) -> dict[str, float]:
    if shadow_results.empty or "candidate" not in shadow_results.columns:
        return {}
    rows = shadow_results[shadow_results["candidate"].astype(str).eq("wf_calibrated_forecast_shadow")]
    if rows.empty:
        rows = shadow_results[shadow_results["candidate"].astype(str).str.contains("calibrated", case=False, na=False)]
    if rows.empty:
        return {}
    row = rows.iloc[-1]
    return {
        "realized_return": float(row.get("realized_return", np.nan)),
        "volatility": float(row.get("volatility", np.nan)),
        "Sharpe": float(row.get("Sharpe", np.nan)),
        "Sortino": float(row.get("Sortino", np.nan)),
        "Calmar": float(row.get("Calmar", np.nan)),
        "max_drawdown": float(row.get("max_drawdown", np.nan)),
        "cash": float(row.get("average_cash", np.nan)),
        "turnover": float(row.get("turnover", np.nan)),
        "TP_rate": np.nan,
        "SL_rate": np.nan,
        "TP_minus_SL": np.nan,
        "hit_rate": float(row.get("hit_rate", np.nan)),
        "sample_size": np.nan,
    }


def run_calibrated_forecast_research_comparison(
    output_path: str | Path = CALIBRATED_FORECAST_RESEARCH_OUTPUT_FILE,
    summary_path: str | Path = "larger_walk_forward_summary.csv",
    calibrated_shadow_path: str | Path = "walk_forward_calibrated_forecast_shadow_results.csv",
) -> pd.DataFrame:
    summary = _read_csv_safe(summary_path)
    shadow = _read_csv_safe(calibrated_shadow_path)
    baseline = _summary_metrics_for_mode(summary, "baseline")
    gated = _summary_metrics_for_mode(summary, "regime_gated_full_quant")
    calibrated = _calibrated_shadow_metrics(shadow)
    metrics = sorted(set(baseline) | set(gated) | set(calibrated))
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "baseline": baseline.get(metric, np.nan),
                "regime_gated_full_quant": gated.get(metric, np.nan),
                "calibrated_forecast_research": calibrated.get(metric, np.nan),
                "calibrated_minus_baseline": (
                    float(calibrated.get(metric, np.nan)) - float(baseline.get(metric, np.nan))
                    if pd.notna(calibrated.get(metric, np.nan)) and pd.notna(baseline.get(metric, np.nan))
                    else np.nan
                ),
                "calibrated_minus_regime_gated": (
                    float(calibrated.get(metric, np.nan)) - float(gated.get(metric, np.nan))
                    if pd.notna(calibrated.get(metric, np.nan)) and pd.notna(gated.get(metric, np.nan))
                    else np.nan
                ),
                "source_note": "baseline/regime_gated from larger_walk_forward_summary; calibrated from walk_forward_calibrated_forecast_shadow",
            }
            for metric in metrics
        ]
    )
    comparison.to_csv(output_path, index=False)
    print("\n===== BASELINE VS REGIME GATED VS CALIBRATED FORECAST RESEARCH =====")
    print(comparison.to_string(index=False))
    print("comparison note: calibrated metrics come from shadow research output; use as research comparison only.")
    missing_sources = []
    if summary.empty:
        missing_sources.append(str(summary_path))
    if shadow.empty:
        missing_sources.append(str(calibrated_shadow_path))
    print(f"missing sources: {missing_sources if missing_sources else 'none'}")
    print(f"Saved: {Path(output_path).resolve()}")
    return comparison


def _raw_target_metrics(raw_results: pd.DataFrame) -> dict[str, float]:
    if raw_results.empty or "variant" not in raw_results.columns:
        return {}
    rows = raw_results[raw_results["variant"].astype(str).eq("raw_target_return_only")]
    if rows.empty:
        rows = raw_results[raw_results["variant"].astype(str).eq("no_signal_strength_adjustment")]
    if rows.empty:
        return {}
    row = rows.iloc[-1]
    return {
        "realized_return": float(row.get("realized_return", np.nan)),
        "volatility": float(row.get("volatility", np.nan)),
        "Sharpe": float(row.get("Sharpe", np.nan)),
        "Sortino": float(row.get("Sortino", np.nan)),
        "Calmar": float(row.get("Calmar", np.nan)),
        "max_drawdown": float(row.get("max_drawdown", np.nan)),
        "cash": float(row.get("average_cash", np.nan)),
        "turnover": float(row.get("turnover", np.nan)),
        "TP_rate": float(row.get("TP_rate", np.nan)),
        "SL_rate": float(row.get("SL_rate", np.nan)),
        "TP_minus_SL": float(row.get("TP_minus_SL", np.nan)),
        "hit_rate": float(row.get("hit_rate", np.nan)),
        "sample_size": float(row.get("sample_size", np.nan)),
    }


def run_raw_target_research_comparison(
    output_path: str | Path = RAW_TARGET_RESEARCH_OUTPUT_FILE,
    summary_path: str | Path = "larger_walk_forward_summary.csv",
    raw_results_path: str | Path = "expected_return_ablation_results.csv",
) -> pd.DataFrame:
    summary = _read_csv_safe(summary_path)
    raw_results = _read_csv_safe(raw_results_path)
    baseline = _summary_metrics_for_mode(summary, "baseline")
    gated = _summary_metrics_for_mode(summary, "regime_gated_full_quant")
    raw_target = _raw_target_metrics(raw_results)
    metrics = sorted(set(baseline) | set(gated) | set(raw_target))
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "baseline": baseline.get(metric, np.nan),
                "regime_gated_full_quant": gated.get(metric, np.nan),
                "raw_target_research": raw_target.get(metric, np.nan),
                "raw_target_minus_baseline": (
                    float(raw_target.get(metric, np.nan)) - float(baseline.get(metric, np.nan))
                    if pd.notna(raw_target.get(metric, np.nan)) and pd.notna(baseline.get(metric, np.nan))
                    else np.nan
                ),
                "raw_target_minus_regime_gated": (
                    float(raw_target.get(metric, np.nan)) - float(gated.get(metric, np.nan))
                    if pd.notna(raw_target.get(metric, np.nan)) and pd.notna(gated.get(metric, np.nan))
                    else np.nan
                ),
                "source_note": "baseline/regime_gated from larger_walk_forward_summary; raw_target from expected_return_ablation_results",
            }
            for metric in metrics
        ]
    )
    comparison.to_csv(output_path, index=False)
    print("\n===== BASELINE VS REGIME GATED VS RAW TARGET RESEARCH =====")
    print(comparison.to_string(index=False))
    missing_sources = []
    if summary.empty:
        missing_sources.append(str(summary_path))
    if raw_results.empty:
        missing_sources.append(str(raw_results_path))
    print(f"missing sources: {missing_sources if missing_sources else 'none'}")
    print(f"Saved: {Path(output_path).resolve()}")
    return comparison


def _run_mode(prices_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mode == "full_quant_research":
        timing_model = "trend_persistence"
        target_model = "quant"
    elif mode == "regime_gated_full_quant":
        timing_model = "ema"
        target_model = "basic"
    else:
        timing_model = "ema"
        target_model = "basic"

    cfg = WalkForwardConfig(
        step_size_days=5,
        max_test_dates=20,
        reduced_universe=DEFAULT_REDUCED_UNIVERSE,
        optimizer_generations_backtest=50,
        disable_live_prices=True,
        lookback_window=252,
        min_history_required=252,
        output_predictions=f"model_mode__{mode}_predictions.csv",
        output_portfolio_returns=f"model_mode__{mode}_portfolio_returns.csv",
        output_summary=f"model_mode__{mode}_summary.csv",
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
            output_path=f"model_mode__{mode}_triple_barrier_labels.csv",
        )
    return predictions, portfolio, summary, labels


def _metrics(summary: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    if summary.empty:
        base = {
            "realized_return": np.nan,
            "realized_volatility": np.nan,
            "realized_sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
            "average_cash": np.nan,
            "average_selected_count": np.nan,
            "average_turnover": np.nan,
        }
    else:
        row = summary.iloc[0]
        base = {
            "realized_return": float(row.get("realized_return", np.nan)),
            "realized_volatility": float(row.get("realized_volatility", np.nan)),
            "realized_sharpe": float(row.get("realized_sharpe", np.nan)),
            "Sortino": float(row.get("Sortino", np.nan)),
            "Calmar": float(row.get("Calmar", np.nan)),
            "max_drawdown": float(row.get("max_drawdown", np.nan)),
            "average_cash": float(row.get("average_cash", np.nan)),
            "average_selected_count": float(row.get("average_selected_count", np.nan)),
            "average_turnover": float(row.get("average_turnover", np.nan)),
            "hit_rate_5d": float(row.get("hit_rate_5d", np.nan)),
            "hit_rate_10d": float(row.get("hit_rate_10d", np.nan)),
            "hit_rate_20d": float(row.get("hit_rate_20d", np.nan)),
            "direction_accuracy_5d": float(row.get("direction_accuracy_5d", np.nan)),
            "direction_accuracy_10d": float(row.get("direction_accuracy_10d", np.nan)),
            "direction_accuracy_20d": float(row.get("direction_accuracy_20d", np.nan)),
        }
    selected = labels[labels["selected"].astype(bool)] if not labels.empty and "selected" in labels.columns else labels.iloc[0:0]
    if selected.empty:
        base.update({"TP_rate": np.nan, "SL_rate": np.nan, "timeout_rate": np.nan, "TP_minus_SL": np.nan})
    else:
        tp = float((selected["first_touch_type"] == "take_profit").mean())
        sl = float((selected["first_touch_type"] == "stop_loss").mean())
        timeout = float((selected["first_touch_type"] == "vertical_timeout").mean())
        base.update({"TP_rate": tp, "SL_rate": sl, "timeout_rate": timeout, "TP_minus_SL": tp - sl})
    return base


def _decision_changes(baseline_predictions: pd.DataFrame, quant_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = sorted(set(baseline_predictions.get("date", [])) | set(quant_predictions.get("date", [])))
    for date in dates:
        b = set(
            baseline_predictions[(baseline_predictions["date"] == date) & baseline_predictions["selected"].astype(bool)]["ticker"].astype(
                str
            )
        )
        q = set(quant_predictions[(quant_predictions["date"] == date) & quant_predictions["selected"].astype(bool)]["ticker"].astype(str))
        overlap = b & q
        rows.append(
            {
                "date": date,
                "baseline_only": sorted(b - q),
                "full_quant_only": sorted(q - b),
                "overlap": sorted(overlap),
                "overlap_count": len(overlap),
                "jaccard_overlap": len(overlap) / len(b | q) if (b | q) else 1.0,
            }
        )
    return pd.DataFrame(rows)


def run_model_mode_comparison(
    prices_df: pd.DataFrame | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    if prices_df is None:
        prices_df = _download_reduced_prices(period="5y")
    if prices_df is None or prices_df.empty:
        raise ValueError("No price data available for model mode comparison.")

    baseline_predictions, baseline_portfolio, baseline_summary, baseline_labels = _run_mode(prices_df, "baseline")
    quant_predictions, quant_portfolio, quant_summary, quant_labels = _run_mode(prices_df, "full_quant_research")
    gated_predictions, gated_portfolio, gated_summary, gated_labels = _run_mode(prices_df, "regime_gated_full_quant")

    baseline = _metrics(baseline_summary, baseline_labels)
    quant = _metrics(quant_summary, quant_labels)
    gated = _metrics(gated_summary, gated_labels)
    metrics = sorted(set(baseline) | set(quant) | set(gated))
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "baseline": baseline.get(metric, np.nan),
                "full_quant_research": quant.get(metric, np.nan),
                "regime_gated_full_quant": gated.get(metric, np.nan),
                "full_quant_minus_baseline": (
                    float(quant.get(metric, np.nan)) - float(baseline.get(metric, np.nan))
                    if pd.notna(quant.get(metric, np.nan)) and pd.notna(baseline.get(metric, np.nan))
                    else np.nan
                ),
                "regime_gated_minus_baseline": (
                    float(gated.get(metric, np.nan)) - float(baseline.get(metric, np.nan))
                    if pd.notna(gated.get(metric, np.nan)) and pd.notna(baseline.get(metric, np.nan))
                    else np.nan
                ),
            }
            for metric in metrics
        ]
    )
    decision_changes = _decision_changes(baseline_predictions, quant_predictions)
    gated_decision_changes = _decision_changes(baseline_predictions, gated_predictions)
    comparison.to_csv(output_path, index=False)
    decision_changes.to_csv(Path(output_path).with_name("model_mode_decision_changes.csv"), index=False)
    gated_path = Path(REGIME_GATED_OUTPUT_FILE)
    comparison.to_csv(gated_path, index=False)
    gated_decision_changes.to_csv(gated_path.with_name("regime_gated_full_quant_decision_changes.csv"), index=False)

    print("\n===== BASELINE VS FULL QUANT RESEARCH COMPARISON =====")
    print(comparison.to_string(index=False))
    print("\nFull Quant decision changes:")
    print(f"average overlap: {float(decision_changes['jaccard_overlap'].mean()) if not decision_changes.empty else np.nan:.4f}")
    print(decision_changes.head(10).to_string(index=False))
    print("\n===== BASELINE VS FULL QUANT VS REGIME GATED COMPARISON =====")
    key_metrics = comparison[
        comparison["metric"].isin(
            [
                "realized_sharpe",
                "realized_return",
                "realized_volatility",
                "max_drawdown",
                "Sortino",
                "Calmar",
                "average_cash",
                "average_turnover",
                "TP_rate",
                "SL_rate",
                "TP_minus_SL",
            ]
        )
    ]
    print(key_metrics.to_string(index=False))
    print("\nRegime-gated decision changes:")
    print(f"average overlap: {float(gated_decision_changes['jaccard_overlap'].mean()) if not gated_decision_changes.empty else np.nan:.4f}")
    print(gated_decision_changes.head(10).to_string(index=False))
    print(f"\nSaved: {Path(output_path).resolve()}")
    print(f"Saved: {Path(output_path).with_name('model_mode_decision_changes.csv').resolve()}")
    print(f"Saved: {gated_path.resolve()}")
    print(f"Saved: {gated_path.with_name('regime_gated_full_quant_decision_changes.csv').resolve()}")
    run_calibrated_forecast_research_comparison()
    run_raw_target_research_comparison()
    return comparison


if __name__ == "__main__":
    run_model_mode_comparison()
