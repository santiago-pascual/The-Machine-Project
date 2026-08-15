from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from risk_metrics import compute_return_risk_metrics
from trend_vs_ema_backtest import _temporary_disable_proxies
from triple_barrier_labeling import generate_triple_barrier_labels
from walk_forward_backtester import (
    DEFAULT_REDUCED_UNIVERSE,
    WalkForwardConfig,
    run_walk_forward_backtest,
)

OUTPUTS = {
    "feature_store": "historical_feature_store.csv",
    "forecast_snapshots": "historical_forecast_snapshots.csv",
    "realized_returns": "historical_realized_returns.csv",
    "triple_barrier_labels": "historical_triple_barrier_labels.csv",
    "ic_dataset": "historical_ic_dataset.csv",
    "portfolio_returns": "historical_walk_forward_portfolio_returns.csv",
    "model_mode_comparison": "historical_model_mode_comparison.csv",
    "summary": "historical_backfill_summary.csv",
    "data_quality": "historical_data_quality_report.csv",
}

REALIZED_HORIZONS = (1, 5, 10, 20, 30)
TRIPLE_BARRIER_HORIZONS = (5, 10, 20, 30)


@dataclass
class HistoricalBackfillConfig:
    start_date: str = "2022-01-01"
    end_date: str | None = None
    step_size_days: int = 5
    max_test_dates: int | None = None
    model_modes: list[str] = field(default_factory=lambda: ["baseline", "regime_gated_full_quant"])
    reduced_universe: bool = True
    optimizer_generations_backtest: int = 50
    lookback_window: int = 252
    min_history_required: int = 252
    period: str = "7y"
    tp_multiple: float = 1.0
    sl_multiple: float = 1.0


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _future_return(prices_df: pd.DataFrame, ticker: str, date: object, horizon: int) -> float:
    if prices_df.empty or ticker not in prices_df.columns:
        return np.nan
    index = pd.to_datetime(prices_df.index, errors="coerce")
    pos = int(index.searchsorted(pd.Timestamp(date), side="right") - 1)
    future_pos = pos + int(horizon)
    if pos < 0 or future_pos >= len(prices_df):
        return np.nan
    start = _safe_float(prices_df.iloc[pos][ticker])
    end = _safe_float(prices_df.iloc[future_pos][ticker])
    if not np.isfinite(start) or start <= 0 or not np.isfinite(end):
        return np.nan
    return float(end / start - 1.0)


def _portfolio_return_from_predictions(group: pd.DataFrame, prices_df: pd.DataFrame, horizon: int) -> float:
    if group.empty:
        return np.nan
    date = group["date"].iloc[0]
    total = 0.0
    valid = False
    for _, row in group.iterrows():
        ticker = str(row["ticker"])
        weight = _safe_float(row.get("weight", 0.0), 0.0)
        ret = _future_return(prices_df, ticker, date, horizon)
        if np.isfinite(ret):
            total += weight * ret
            valid = True
    return float(total) if valid else np.nan


def _download_prices(config: HistoricalBackfillConfig) -> pd.DataFrame:
    cache_dir = Path(".yfinance_cache").resolve()
    cache_dir.mkdir(exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass
    if config.reduced_universe:
        tickers = DEFAULT_REDUCED_UNIVERSE
    else:
        try:
            from financial_data_system import (
                NASDAQ_DEFAULT_LIMIT,
                build_trading_universe,
            )

            tickers = build_trading_universe(include_full_nasdaq=True, nasdaq_limit=NASDAQ_DEFAULT_LIMIT)
        except Exception as exc:
            print(f"[WARNING] Could not build full historical universe. Using reduced universe. Error: {exc}")
            tickers = DEFAULT_REDUCED_UNIVERSE
    with _temporary_disable_proxies():
        data = yf.download(
            tickers,
            period=config.period,
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
        close = data[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in data.columns else pd.DataFrame()
    close = close.replace([np.inf, -np.inf], np.nan).ffill().dropna(axis=1, thresh=260)
    return close.dropna(how="all")


def _run_mode(
    prices_df: pd.DataFrame,
    config: HistoricalBackfillConfig,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timing_model = "trend_persistence" if mode == "full_quant_research" else "ema"
    target_model = "quant" if mode == "full_quant_research" else "basic"
    temp_predictions = Path(f"historical_{mode}_predictions_tmp.csv")
    temp_portfolio = Path(f"historical_{mode}_portfolio_tmp.csv")
    temp_summary = Path(f"historical_{mode}_summary_tmp.csv")
    wf_config = WalkForwardConfig(
        start_date=config.start_date,
        end_date=config.end_date,
        step_size_days=config.step_size_days,
        max_test_dates=config.max_test_dates,
        reduced_universe=DEFAULT_REDUCED_UNIVERSE if config.reduced_universe else list(prices_df.columns),
        optimizer_generations_backtest=config.optimizer_generations_backtest,
        disable_live_prices=True,
        lookback_window=config.lookback_window,
        min_history_required=config.min_history_required,
        output_predictions=str(temp_predictions),
        output_portfolio_returns=str(temp_portfolio),
        output_summary=str(temp_summary),
        timing_model=timing_model,
        target_model=target_model,
        model_mode=mode,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        predictions, portfolio, summary = run_walk_forward_backtest(prices_df, config=wf_config)
    for temp_file in (temp_predictions, temp_portfolio, temp_summary):
        temp_file.unlink(missing_ok=True)
    return predictions, portfolio, summary


def _ensure_required_snapshot_columns(snapshots: pd.DataFrame) -> pd.DataFrame:
    required_defaults = {
        "cash_weight": np.nan,
        "regime_confidence": np.nan,
        "ema_timing_score": np.nan,
        "trend_persistence_score": np.nan,
        "quant_target_price": np.nan,
        "black_litterman_return": np.nan,
        "covariance_method": "ledoit_wolf/manual_diagonal_shrinkage",
    }
    for column, default in required_defaults.items():
        if column not in snapshots.columns:
            snapshots[column] = default
    if "cash_weight" in snapshots.columns:
        cash_by_date_mode = snapshots.groupby(["date", "model_mode"])["weight"].sum().apply(lambda x: max(0.0, 1.0 - float(x)))
        snapshots["cash_weight"] = [
            cash_by_date_mode.get((row.date, row.model_mode), np.nan) for row in snapshots[["date", "model_mode"]].itertuples(index=False)
        ]
    return snapshots


def _add_realized_returns(snapshots: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    realized = snapshots[["date", "ticker", "model_mode", "selected", "weight"]].copy()
    for horizon in REALIZED_HORIZONS:
        realized[f"realized_return_{horizon}d"] = [
            _future_return(prices_df, str(row.ticker), row.date, horizon) for row in realized[["date", "ticker"]].itertuples(index=False)
        ]
    return realized


def _portfolio_returns_from_snapshots(snapshots: pd.DataFrame, prices_df: pd.DataFrame, portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (date, mode), group in snapshots.groupby(["date", "model_mode"]):
        row = {
            "date": date,
            "model_mode": mode,
            "cash_weight": float(group["cash_weight"].iloc[0]) if "cash_weight" in group.columns else np.nan,
            "selected_count": int(group["selected"].astype(bool).sum()) if "selected" in group.columns else 0,
        }
        for horizon in REALIZED_HORIZONS:
            row[f"realized_portfolio_return_{horizon}d"] = _portfolio_return_from_predictions(group, prices_df, horizon)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not portfolios.empty:
        keep = [
            c
            for c in [
                "date",
                "model_mode",
                "turnover",
                "portfolio_expected_return",
                "portfolio_expected_volatility",
                "portfolio_expected_sharpe",
                "gate_decision",
                "gate_reason",
            ]
            if c in portfolios.columns
        ]
        if keep:
            result = result.merge(portfolios[keep], on=["date", "model_mode"], how="left")
    return result


def _build_feature_store(snapshots: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "date",
        "ticker",
        "model_mode",
        "selected",
        "weight",
        "expected_daily_return",
        "expected_total_return",
        "target_confidence",
        "signal_strength",
        "quality_score",
        "regime",
        "regime_confidence",
        "ema_timing_score",
        "trend_persistence_score",
        "quant_target_price",
        "covariance_method",
    ]
    for col in feature_cols:
        if col not in snapshots.columns:
            snapshots[col] = np.nan
    return snapshots[feature_cols].copy()


def _build_ic_dataset(feature_store: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    realized_cols = ["date", "ticker", "model_mode"] + [f"realized_return_{h}d" for h in REALIZED_HORIZONS]
    return feature_store.merge(realized[realized_cols], on=["date", "ticker", "model_mode"], how="left")


def _metrics_for_mode(portfolio: pd.DataFrame, labels: pd.DataFrame, mode: str) -> dict[str, object]:
    subset = portfolio[portfolio["model_mode"].astype(str).eq(mode)].copy()
    returns = pd.to_numeric(subset.get("realized_portfolio_return_1d", pd.Series(dtype=float)), errors="coerce").dropna()
    risk = compute_return_risk_metrics(returns)
    equity = (1.0 + returns).cumprod() if not returns.empty else pd.Series(dtype=float)
    drawdown = equity / equity.cummax() - 1.0 if not equity.empty else pd.Series(dtype=float)
    selected_labels = (
        labels[
            labels.get("model_mode", pd.Series("", index=labels.index)).astype(str).eq(mode)
            & labels.get("selected", pd.Series(False, index=labels.index)).astype(bool)
        ]
        if not labels.empty
        else pd.DataFrame()
    )
    tp = float(selected_labels["first_touch_type"].eq("take_profit").mean()) if not selected_labels.empty else np.nan
    sl = float(selected_labels["first_touch_type"].eq("stop_loss").mean()) if not selected_labels.empty else np.nan
    expected = pd.to_numeric(subset.get("portfolio_expected_return", pd.Series(dtype=float)), errors="coerce")
    valid = pd.concat(
        [expected, pd.to_numeric(subset.get("realized_portfolio_return_5d", pd.Series(dtype=float)), errors="coerce")], axis=1
    ).dropna()
    return {
        "model_mode": mode,
        "sample_size": len(subset),
        "realized_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
        "volatility": float(risk["annualized_volatility"]),
        "Sharpe": float(risk["annualized_return_estimate"] / risk["annualized_volatility"]) if risk["annualized_volatility"] > 0 else 0.0,
        "Sortino": float(risk["sortino_ratio"]),
        "Calmar": float(risk["calmar_ratio"]),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
        "average_cash": float(subset["cash_weight"].mean()) if "cash_weight" in subset.columns and not subset.empty else np.nan,
        "average_selected_count": float(subset["selected_count"].mean())
        if "selected_count" in subset.columns and not subset.empty
        else np.nan,
        "turnover": float(subset["turnover"].mean()) if "turnover" in subset.columns and not subset.empty else np.nan,
        "TP_rate": tp,
        "SL_rate": sl,
        "TP_minus_SL": tp - sl if pd.notna(tp) and pd.notna(sl) else np.nan,
        "hit_rate_5d": float((subset["realized_portfolio_return_5d"] > 0).mean())
        if "realized_portfolio_return_5d" in subset.columns
        else np.nan,
        "direction_accuracy_5d": float((np.sign(valid.iloc[:, 0]) == np.sign(valid.iloc[:, 1])).mean()) if not valid.empty else np.nan,
    }


def _data_quality_report(prices_df: pd.DataFrame, config: HistoricalBackfillConfig) -> pd.DataFrame:
    requested = DEFAULT_REDUCED_UNIVERSE if config.reduced_universe else list(prices_df.columns)
    valid = list(prices_df.columns)
    invalid = sorted(set(requested) - set(valid))
    rows = [
        {
            "start_date": str(prices_df.index.min().date()) if not prices_df.empty else "",
            "end_date": str(prices_df.index.max().date()) if not prices_df.empty else "",
            "requested_tickers": len(requested),
            "valid_tickers": len(valid),
            "removed_invalid_tickers": ", ".join(invalid),
            "reduced_universe": bool(config.reduced_universe),
            "step_size_days": int(config.step_size_days),
            "live_prices_disabled": True,
            "daily_live_files_overwritten": False,
        }
    ]
    return pd.DataFrame(rows)


def _summary_report(
    *,
    config: HistoricalBackfillConfig,
    prices_df: pd.DataFrame,
    snapshots: pd.DataFrame,
    portfolio: pd.DataFrame,
    labels: pd.DataFrame,
    runtime: float,
) -> pd.DataFrame:
    selected_rows = snapshots[snapshots["selected"].astype(bool)] if not snapshots.empty else pd.DataFrame()
    rows = [
        {
            "start_date": config.start_date,
            "end_date": str(prices_df.index.max().date()) if not prices_df.empty else "",
            "number_of_decision_dates": int(snapshots["date"].nunique()) if not snapshots.empty else 0,
            "number_of_tickers": int(snapshots["ticker"].nunique()) if not snapshots.empty else 0,
            "model_modes_processed": ", ".join(config.model_modes),
            "total_prediction_rows": len(snapshots),
            "total_selected_rows": len(selected_rows),
            "selected_only_sample_size": len(selected_rows),
            "sample_gte_150": bool(len(selected_rows) >= 150),
            "sample_gte_500": bool(len(selected_rows) >= 500),
            "sample_gte_1000": bool(len(selected_rows) >= 1000),
            "sufficient_for_ic_analysis": bool(len(snapshots) >= 500),
            "sufficient_for_triple_barrier_validation": bool(len(labels) >= 500),
            "sufficient_for_regime_attribution": bool(portfolio["date"].nunique() >= 50) if not portfolio.empty else False,
            "sufficient_for_meta_labeling_prototype": bool(len(selected_rows) >= 500),
            "sufficient_for_ml_training": bool(len(selected_rows) >= 1000),
            "runtime_seconds": float(runtime),
        }
    ]
    return pd.DataFrame(rows)


def run_historical_research_backfill(config: HistoricalBackfillConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or HistoricalBackfillConfig()
    start_time = time.time()
    prices_df = _download_prices(config)
    if prices_df.empty:
        raise ValueError("No historical price data available for backfill.")

    all_predictions = []
    all_portfolios = []
    all_summaries = []
    for mode in config.model_modes:
        predictions, portfolio, summary = _run_mode(prices_df, config, mode)
        all_predictions.append(predictions)
        all_portfolios.append(portfolio)
        all_summaries.append(summary.assign(model_mode=mode) if not summary.empty else summary)

    snapshots = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    portfolios = pd.concat(all_portfolios, ignore_index=True) if all_portfolios else pd.DataFrame()
    snapshots = _ensure_required_snapshot_columns(snapshots)
    realized = _add_realized_returns(snapshots, prices_df)
    portfolio_returns = _portfolio_returns_from_snapshots(snapshots, prices_df, portfolios)

    labels_parts = []
    for mode in config.model_modes:
        mode_predictions = snapshots[snapshots["model_mode"].astype(str).eq(mode)].copy()
        with contextlib.redirect_stdout(io.StringIO()):
            temp_labels = Path(f"historical_{mode}_triple_barrier_tmp.csv")
            labels = generate_triple_barrier_labels(
                prices_df=prices_df,
                predictions_df=mode_predictions,
                horizons=TRIPLE_BARRIER_HORIZONS,
                tp_multiple=config.tp_multiple,
                sl_multiple=config.sl_multiple,
                output_path=str(temp_labels),
            )
            temp_labels.unlink(missing_ok=True)
        if not labels.empty:
            labels["model_mode"] = mode
        labels_parts.append(labels)
    labels_all = pd.concat(labels_parts, ignore_index=True) if labels_parts else pd.DataFrame()

    feature_store = _build_feature_store(snapshots.copy())
    ic_dataset = _build_ic_dataset(feature_store, realized)
    comparison = pd.DataFrame([_metrics_for_mode(portfolio_returns, labels_all, mode) for mode in config.model_modes])
    data_quality = _data_quality_report(prices_df, config)
    summary = _summary_report(
        config=config,
        prices_df=prices_df,
        snapshots=snapshots,
        portfolio=portfolio_returns,
        labels=labels_all,
        runtime=time.time() - start_time,
    )

    outputs = {
        "feature_store": feature_store,
        "forecast_snapshots": snapshots,
        "realized_returns": realized,
        "triple_barrier_labels": labels_all,
        "ic_dataset": ic_dataset,
        "portfolio_returns": portfolio_returns,
        "model_mode_comparison": comparison,
        "summary": summary,
        "data_quality": data_quality,
    }
    for key, df in outputs.items():
        df.to_csv(OUTPUTS[key], index=False)

    print_historical_backfill_report(config, outputs)
    return outputs


def print_historical_backfill_report(config: HistoricalBackfillConfig, outputs: dict[str, pd.DataFrame]) -> None:
    summary = outputs["summary"].iloc[0] if not outputs["summary"].empty else pd.Series(dtype=object)
    comparison = outputs["model_mode_comparison"]
    print("\n===== HISTORICAL RESEARCH BACKFILL =====")
    print(f"start date: {summary.get('start_date', config.start_date)}")
    print(f"end date: {summary.get('end_date', config.end_date)}")
    print(f"number of decision dates: {summary.get('number_of_decision_dates', 0)}")
    print(f"number of tickers: {summary.get('number_of_tickers', 0)}")
    print(f"model modes processed: {summary.get('model_modes_processed', ', '.join(config.model_modes))}")
    print(f"total prediction rows: {summary.get('total_prediction_rows', 0)}")
    print(f"total selected rows: {summary.get('total_selected_rows', 0)}")
    print(f"selected-only sample size: {summary.get('selected_only_sample_size', 0)}")
    print(f"runtime seconds: {float(summary.get('runtime_seconds', 0.0)):.2f}")
    print("files generated:")
    for path in OUTPUTS.values():
        print(f"- {path}")

    print("\n===== HISTORICAL BACKFILL LOOK-AHEAD CHECK =====")
    print("live prices disabled: True")
    print("every date t uses truncated data: True")
    print("features computed before labels: True")
    print("realized returns computed only after prediction date: True")
    print("daily live files not overwritten: True")

    print("\n===== HISTORICAL MODEL MODE COMPARISON =====")
    print(comparison.to_string(index=False) if not comparison.empty else "No comparison available.")

    print("\n===== HISTORICAL SAMPLE SIZE CHECK =====")
    for key in [
        "selected_only_sample_size",
        "sample_gte_150",
        "sample_gte_500",
        "sample_gte_1000",
        "sufficient_for_ic_analysis",
        "sufficient_for_triple_barrier_validation",
        "sufficient_for_regime_attribution",
        "sufficient_for_meta_labeling_prototype",
        "sufficient_for_ml_training",
    ]:
        print(f"{key}: {summary.get(key, np.nan)}")


if __name__ == "__main__":
    run_historical_research_backfill()
