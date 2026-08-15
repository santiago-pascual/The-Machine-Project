from __future__ import annotations

from pathlib import Path
import contextlib
import io
import os

import numpy as np
import pandas as pd
import yfinance as yf

from triple_barrier_labeling import generate_triple_barrier_labels
from walk_forward_backtester import DEFAULT_REDUCED_UNIVERSE, WalkForwardConfig, run_walk_forward_backtest


DEFAULT_OUTPUT_FILE = "trend_vs_ema_backtest.csv"


@contextlib.contextmanager
def _temporary_disable_proxies():
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    previous = {key: os.environ.get(key) for key in proxy_keys}
    try:
        for key in proxy_keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _download_reduced_prices(period: str = "5y") -> pd.DataFrame:
    with _temporary_disable_proxies():
        data = yf.download(
            DEFAULT_REDUCED_UNIVERSE,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
            group_by="column",
            timeout=30,
        )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            close = data["Adj Close"]
        else:
            return pd.DataFrame()
    else:
        close = data[["Close"]].rename(columns={"Close": DEFAULT_REDUCED_UNIVERSE[0]}) if "Close" in data.columns else pd.DataFrame()
    close = close.replace([np.inf, -np.inf], np.nan).ffill().dropna(axis=1, thresh=260)
    return close.dropna(how="all")


def _summary_from_outputs(summary_df: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float | str]:
    if summary_df.empty:
        base: dict[str, float | str] = {
            "realized_return": np.nan,
            "realized_volatility": np.nan,
            "realized_sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
            "average_cash": np.nan,
            "average_selected_count": np.nan,
            "average_turnover": np.nan,
            "hit_rate_5d": np.nan,
            "hit_rate_10d": np.nan,
            "hit_rate_20d": np.nan,
            "direction_accuracy_5d": np.nan,
            "direction_accuracy_10d": np.nan,
            "direction_accuracy_20d": np.nan,
        }
    else:
        row = summary_df.iloc[0]
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


def _run_mode(prices_df: pd.DataFrame, timing_model: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = WalkForwardConfig(
        step_size_days=5,
        max_test_dates=20,
        reduced_universe=DEFAULT_REDUCED_UNIVERSE,
        optimizer_generations_backtest=50,
        disable_live_prices=True,
        lookback_window=252,
        min_history_required=252,
        output_predictions=f"{prefix}_predictions.csv",
        output_portfolio_returns=f"{prefix}_portfolio_returns.csv",
        output_summary=f"{prefix}_summary.csv",
        timing_model=timing_model,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        predictions, portfolio, summary = run_walk_forward_backtest(prices_df, config=config)
        labels = generate_triple_barrier_labels(
            prices_df=prices_df,
            predictions_df=predictions,
            horizons=(5, 10, 20),
            tp_multiple=1.0,
            sl_multiple=1.0,
            output_path=f"{prefix}_triple_barrier_labels.csv",
        )
    return predictions, portfolio, summary, labels


def _selection_changes(ema_predictions: pd.DataFrame, trend_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = sorted(set(ema_predictions.get("date", [])) | set(trend_predictions.get("date", [])))
    for date in dates:
        ema_set = set(
            ema_predictions[
                (ema_predictions["date"] == date) & (ema_predictions["selected"].astype(bool))
            ]["ticker"].astype(str)
        )
        trend_set = set(
            trend_predictions[
                (trend_predictions["date"] == date) & (trend_predictions["selected"].astype(bool))
            ]["ticker"].astype(str)
        )
        overlap = ema_set & trend_set
        rows.append(
            {
                "date": date,
                "ema_only": sorted(ema_set - trend_set),
                "trend_only": sorted(trend_set - ema_set),
                "overlap": sorted(overlap),
                "ema_selected_count": len(ema_set),
                "trend_selected_count": len(trend_set),
                "overlap_count": len(overlap),
                "jaccard_overlap": len(overlap) / len(ema_set | trend_set) if (ema_set | trend_set) else 1.0,
            }
        )
    return pd.DataFrame(rows)


def run_trend_vs_ema_backtest(
    prices_df: pd.DataFrame | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    if prices_df is None:
        prices_df = _download_reduced_prices(period="5y")
    if prices_df is None or prices_df.empty:
        raise ValueError("No price data available for trend vs EMA backtest.")

    ema_predictions, ema_portfolio, ema_summary, ema_labels = _run_mode(
        prices_df=prices_df,
        timing_model="ema",
        prefix="trend_vs_ema__ema",
    )
    trend_predictions, trend_portfolio, trend_summary, trend_labels = _run_mode(
        prices_df=prices_df,
        timing_model="trend_persistence",
        prefix="trend_vs_ema__trend_persistence",
    )

    ema_metrics = _summary_from_outputs(ema_summary, ema_labels)
    trend_metrics = _summary_from_outputs(trend_summary, trend_labels)
    metric_order = [
        "realized_return",
        "realized_volatility",
        "realized_sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_cash",
        "average_selected_count",
        "average_turnover",
        "TP_rate",
        "SL_rate",
        "timeout_rate",
        "TP_minus_SL",
        "hit_rate_5d",
        "hit_rate_10d",
        "hit_rate_20d",
        "direction_accuracy_5d",
        "direction_accuracy_10d",
        "direction_accuracy_20d",
    ]
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "EMA mode": ema_metrics.get(metric, np.nan),
                "Trend Persistence mode": trend_metrics.get(metric, np.nan),
                "Difference": (
                    float(trend_metrics.get(metric, np.nan)) - float(ema_metrics.get(metric, np.nan))
                    if pd.notna(trend_metrics.get(metric, np.nan)) and pd.notna(ema_metrics.get(metric, np.nan))
                    else np.nan
                ),
            }
            for metric in metric_order
        ]
    )
    selection_changes = _selection_changes(ema_predictions, trend_predictions)
    avg_jaccard = float(selection_changes["jaccard_overlap"].mean()) if not selection_changes.empty else np.nan
    turnover_diff = float(trend_metrics.get("average_turnover", np.nan)) - float(ema_metrics.get("average_turnover", np.nan))

    comparison.to_csv(output_path, index=False)
    selection_changes.to_csv(Path(output_path).with_name("trend_vs_ema_decision_changes.csv"), index=False)

    print("\n===== EMA VS TREND PERSISTENCE WALK-FORWARD TEST =====")
    print(comparison.to_string(index=False))

    print("\n===== TIMING MODEL DECISION CHANGES =====")
    print(f"average selection overlap: {avg_jaccard:.4f}")
    print(f"turnover difference: {turnover_diff:.6f}")
    print("sample decision changes:")
    display_cols = ["date", "ema_only", "trend_only", "overlap", "jaccard_overlap"]
    print(selection_changes[display_cols].head(10).to_string(index=False))

    print(f"\nSaved: {Path(output_path).resolve()}")
    print(f"Saved: {Path(output_path).with_name('trend_vs_ema_decision_changes.csv').resolve()}")
    return comparison


if __name__ == "__main__":
    run_trend_vs_ema_backtest()
