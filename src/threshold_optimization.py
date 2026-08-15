from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_FILE = "threshold_optimization.csv"


@dataclass(frozen=True)
class ThresholdConfig:
    signal_strength_threshold: float
    target_confidence_threshold: float
    quality_score_threshold: float
    expected_return_threshold: float
    max_selected_assets: int
    min_selected_assets: int


DEFAULT_CURRENT_CONFIG = ThresholdConfig(
    signal_strength_threshold=0.20,
    target_confidence_threshold=0.40,
    quality_score_threshold=0.40,
    expected_return_threshold=0.0000,
    max_selected_assets=4,
    min_selected_assets=2,
)


def _load_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    data = pd.read_csv(file_path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "ticker" in data.columns:
        data["ticker"] = data["ticker"].astype(str)
    return data.dropna(subset=[c for c in ["date", "ticker"] if c in data.columns])


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    defaults = {
        "signal_strength": 0.0,
        "target_confidence": 0.5,
        "quality_score": 0.5,
        "expected_daily_return": 0.0,
        "weight": 0.0,
        "selected": False,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default
    for column in [
        "signal_strength",
        "target_confidence",
        "quality_score",
        "expected_daily_return",
        "weight",
        "realized_return_5d",
        "realized_return_10d",
        "realized_return_20d",
    ]:
        if column in data.columns:
            data[column] = _safe_numeric(data[column], default=0.0)
    data["selected"] = data["selected"].astype(bool)
    return data


def _merge_labels(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return predictions
    keep_cols = [
        "date",
        "ticker",
        "horizon",
        "label",
        "first_touch_type",
        "realized_return_at_barrier",
        "time_to_first_touch",
    ]
    label_cols = [column for column in keep_cols if column in labels.columns]
    if not {"date", "ticker", "horizon"}.issubset(label_cols):
        return predictions
    merged_rows = []
    for horizon in sorted(labels["horizon"].dropna().unique()):
        horizon = int(horizon)
        horizon_predictions = predictions.copy()
        horizon_predictions["horizon"] = horizon
        horizon_labels = labels[labels["horizon"] == horizon][label_cols].copy()
        merged_rows.append(
            horizon_predictions.merge(
                horizon_labels,
                on=["date", "ticker", "horizon"],
                how="left",
                suffixes=("", "_label"),
            )
        )
    return pd.concat(merged_rows, ignore_index=True) if merged_rows else predictions


def _select_by_thresholds(data: pd.DataFrame, config: ThresholdConfig) -> pd.DataFrame:
    mask = (
        (data["signal_strength"] >= config.signal_strength_threshold)
        & (data["target_confidence"] >= config.target_confidence_threshold)
        & (data["quality_score"] >= config.quality_score_threshold)
        & (data["expected_daily_return"] >= config.expected_return_threshold)
    )
    candidates = data.loc[mask].copy()
    if candidates.empty:
        return candidates

    selected_groups = []
    for _, group in candidates.groupby("date", sort=True):
        ranked = group.sort_values(
            ["expected_daily_return", "signal_strength", "target_confidence", "quality_score"],
            ascending=False,
        )
        selected = ranked.head(config.max_selected_assets).copy()
        if len(selected) < config.min_selected_assets:
            continue
        selected_groups.append(selected)
    return pd.concat(selected_groups, ignore_index=True) if selected_groups else candidates.iloc[0:0].copy()


def _weighted_period_returns(selected: pd.DataFrame, horizon: int) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    realized_col = f"realized_return_{horizon}d"
    if realized_col not in selected.columns:
        realized_col = "realized_return_at_barrier"
    if realized_col not in selected.columns:
        return pd.Series(dtype=float)

    rows = []
    for date, group in selected.groupby("date", sort=True):
        returns = _safe_numeric(group[realized_col], default=np.nan).dropna()
        if returns.empty:
            continue
        weights = _safe_numeric(group.loc[returns.index, "weight"], default=0.0).clip(lower=0.0)
        if float(weights.sum()) <= 0:
            weights = pd.Series(np.ones(len(returns)) / len(returns), index=returns.index)
        else:
            weights = weights / weights.sum()
        rows.append((date, float(np.dot(weights.to_numpy(), returns.to_numpy()))))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows), dtype=float).sort_index()


def _turnover_proxy(selected: pd.DataFrame) -> float:
    previous: set[str] | None = None
    turnovers: list[float] = []
    for _, group in selected.groupby("date", sort=True):
        current = set(group["ticker"].astype(str))
        if previous is not None:
            union = current | previous
            turnovers.append(1.0 - (len(current & previous) / len(union) if union else 1.0))
        previous = current
    return float(np.mean(turnovers)) if turnovers else 0.0


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _sortino(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    if downside_std <= 0:
        return 0.0
    return float(returns.mean() / downside_std)


def _cash_proxy(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 1.0
    cash_values = []
    for _, group in selected.groupby("date", sort=True):
        invested = float(_safe_numeric(group["weight"], default=0.0).clip(lower=0.0).sum())
        cash_values.append(float(np.clip(1.0 - invested, 0.0, 1.0)))
    return float(np.mean(cash_values)) if cash_values else 1.0


def _evaluate_config(data: pd.DataFrame, config: ThresholdConfig, horizon: int) -> dict[str, object]:
    selected = _select_by_thresholds(data, config)
    period_returns = _weighted_period_returns(selected, horizon)
    realized_col = f"realized_return_{horizon}d"
    asset_returns = _safe_numeric(selected[realized_col], default=np.nan).dropna() if realized_col in selected.columns else pd.Series(dtype=float)

    if "first_touch_type" in selected.columns:
        first_touch = selected["first_touch_type"].fillna("")
        tp_rate = float((first_touch == "take_profit").mean()) if len(first_touch) else np.nan
        sl_rate = float((first_touch == "stop_loss").mean()) if len(first_touch) else np.nan
        timeout_rate = float((first_touch == "vertical_timeout").mean()) if len(first_touch) else np.nan
    elif "label" in selected.columns:
        labels = _safe_numeric(selected["label"], default=np.nan).dropna()
        tp_rate = float((labels == 1).mean()) if len(labels) else np.nan
        sl_rate = float((labels == -1).mean()) if len(labels) else np.nan
        timeout_rate = float((labels == 0).mean()) if len(labels) else np.nan
    else:
        tp_rate = sl_rate = timeout_rate = np.nan

    volatility = float(period_returns.std(ddof=1)) if len(period_returns) > 1 else 0.0
    sharpe = float(period_returns.mean() / volatility) if volatility > 0 else 0.0
    selected_count = selected.groupby("date")["ticker"].nunique() if not selected.empty else pd.Series(dtype=float)

    return {
        **config.__dict__,
        "horizon": int(horizon),
        "sample_size": int(len(selected)),
        "test_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "average_realized_return": float(asset_returns.mean()) if not asset_returns.empty else np.nan,
        "average_portfolio_return": float(period_returns.mean()) if not period_returns.empty else np.nan,
        "average_realized_return_5d": _horizon_mean(data, selected, 5),
        "average_realized_return_10d": _horizon_mean(data, selected, 10),
        "average_realized_return_20d": _horizon_mean(data, selected, 20),
        "Sharpe": sharpe,
        "Sortino": _sortino(period_returns),
        "max_drawdown_proxy": _max_drawdown(period_returns),
        "TP_rate": tp_rate,
        "SL_rate": sl_rate,
        "timeout_rate": timeout_rate,
        "TP_minus_SL": tp_rate - sl_rate if pd.notna(tp_rate) and pd.notna(sl_rate) else np.nan,
        "average_selected_count": float(selected_count.mean()) if not selected_count.empty else 0.0,
        "average_cash_proxy": _cash_proxy(selected),
        "turnover_proxy": _turnover_proxy(selected),
    }


def _horizon_mean(all_data: pd.DataFrame, selected: pd.DataFrame, horizon: int) -> float:
    del all_data
    column = f"realized_return_{horizon}d"
    if selected.empty or column not in selected.columns:
        return np.nan
    values = _safe_numeric(selected[column], default=np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _grid(
    signal_strength_thresholds: Iterable[float],
    target_confidence_thresholds: Iterable[float],
    quality_score_thresholds: Iterable[float],
    expected_return_thresholds: Iterable[float],
    max_selected_assets_values: Iterable[int],
    min_selected_assets_values: Iterable[int],
) -> list[ThresholdConfig]:
    return [
        ThresholdConfig(float(signal), float(confidence), float(quality), float(expected), int(max_assets), int(min_assets))
        for signal, confidence, quality, expected, max_assets, min_assets in product(
            signal_strength_thresholds,
            target_confidence_thresholds,
            quality_score_thresholds,
            expected_return_thresholds,
            max_selected_assets_values,
            min_selected_assets_values,
        )
        if int(min_assets) <= int(max_assets)
    ]


def _classification_warning(row: pd.Series, defaults: ThresholdConfig, min_sample_size: int) -> list[str]:
    warnings: list[str] = []
    if int(row.get("sample_size", 0)) < min_sample_size:
        warnings.append("sample_size_too_small")
    if float(row.get("average_selected_count", 0.0)) < 2.0:
        warnings.append("improves_by_selecting_too_few_assets")
    if float(row.get("turnover_proxy", 0.0)) > 0.80:
        warnings.append("extreme_turnover")
    if pd.notna(row.get("TP_rate")) and pd.notna(row.get("SL_rate")) and float(row["TP_rate"]) < float(row["SL_rate"]):
        warnings.append("TP_rate_below_SL_rate")
    distance = (
        abs(float(row["signal_strength_threshold"]) - defaults.signal_strength_threshold)
        + abs(float(row["target_confidence_threshold"]) - defaults.target_confidence_threshold)
        + abs(float(row["quality_score_threshold"]) - defaults.quality_score_threshold)
        + abs(float(row["expected_return_threshold"]) - defaults.expected_return_threshold) * 100
        + abs(int(row["max_selected_assets"]) - defaults.max_selected_assets) / 10
    )
    if distance > 0.75:
        warnings.append("far_from_current_defaults")
    return warnings


def run_threshold_optimization(
    *,
    predictions_path: str | Path = "walk_forward_predictions.csv",
    portfolio_returns_path: str | Path = "walk_forward_portfolio_returns.csv",
    triple_barrier_labels_path: str | Path = "triple_barrier_labels.csv",
    barrier_optimization_path: str | Path = "barrier_parameter_optimization.csv",
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
    horizons: Iterable[int] = (5, 10, 20),
    signal_strength_thresholds: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6),
    target_confidence_thresholds: Iterable[float] = (0.4, 0.5, 0.6, 0.7),
    quality_score_thresholds: Iterable[float] = (0.4, 0.5, 0.6, 0.7),
    expected_return_thresholds: Iterable[float] = (0.0000, 0.0005, 0.0010, 0.0015),
    max_selected_assets_values: Iterable[int] = (3, 4, 5, 6, 8),
    min_selected_assets_values: Iterable[int] = (1, 2, 3),
    current_defaults: ThresholdConfig = DEFAULT_CURRENT_CONFIG,
    min_sample_size: int = 50,
) -> pd.DataFrame:
    predictions = _prepare_predictions(_load_csv(predictions_path))
    portfolio_returns = _load_csv(portfolio_returns_path)
    labels = _load_csv(triple_barrier_labels_path)
    barrier_results = _load_csv(barrier_optimization_path)

    print("\n===== THRESHOLD OPTIMIZATION REPORT =====")
    print(f"predictions rows: {len(predictions)}")
    print(f"portfolio return rows: {len(portfolio_returns)}")
    print(f"triple barrier rows: {len(labels)}")
    print(f"barrier optimization rows: {len(barrier_results)}")

    if predictions.empty:
        result = pd.DataFrame()
        result.to_csv(output_path, index=False)
        print("No walk-forward predictions available.")
        return result

    data = _merge_labels(predictions, labels)
    configs = _grid(
        signal_strength_thresholds,
        target_confidence_thresholds,
        quality_score_thresholds,
        expected_return_thresholds,
        max_selected_assets_values,
        min_selected_assets_values,
    )

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        horizon_data = data[data["horizon"] == int(horizon)].copy() if "horizon" in data.columns else data.copy()
        for config in configs:
            rows.append(_evaluate_config(horizon_data, config, int(horizon)))

    result = pd.DataFrame(rows)
    if result.empty:
        result.to_csv(output_path, index=False)
        print("No threshold configurations evaluated.")
        return result

    result["warning_flags"] = result.apply(
        lambda row: ", ".join(_classification_warning(row, current_defaults, min_sample_size)),
        axis=1,
    )
    result = result.sort_values(
        ["Sharpe", "average_portfolio_return", "TP_minus_SL", "SL_rate", "sample_size"],
        ascending=[False, False, False, True, False],
    )
    result.to_csv(output_path, index=False)

    display_cols = [
        "horizon",
        "signal_strength_threshold",
        "target_confidence_threshold",
        "quality_score_threshold",
        "expected_return_threshold",
        "max_selected_assets",
        "min_selected_assets",
        "sample_size",
        "test_dates",
        "average_portfolio_return",
        "average_realized_return",
        "Sharpe",
        "Sortino",
        "max_drawdown_proxy",
        "TP_rate",
        "SL_rate",
        "timeout_rate",
        "TP_minus_SL",
        "average_selected_count",
        "average_cash_proxy",
        "turnover_proxy",
        "warning_flags",
    ]
    print(result[display_cols].head(20).to_string(index=False))

    baseline = _evaluate_config(data[data["horizon"] == 20].copy() if "horizon" in data.columns else data.copy(), current_defaults, 20)
    best = result.iloc[0]
    print("\n===== THRESHOLD ROBUSTNESS WARNING =====")
    print(f"current defaults: {current_defaults}")
    print(f"baseline Sharpe: {baseline['Sharpe']:.6f}")
    print(f"best Sharpe: {float(best['Sharpe']):.6f}")
    print(f"baseline avg portfolio return: {baseline['average_portfolio_return']:.6f}")
    print(f"best avg portfolio return: {float(best['average_portfolio_return']):.6f}")
    print(f"baseline TP/SL/timeout: {baseline['TP_rate']:.4f} / {baseline['SL_rate']:.4f} / {baseline['timeout_rate']:.4f}")
    print(f"best TP/SL/timeout: {float(best['TP_rate']):.4f} / {float(best['SL_rate']):.4f} / {float(best['timeout_rate']):.4f}")
    print(f"best warning flags: {best['warning_flags'] or 'none'}")
    if int(best["sample_size"]) < min_sample_size:
        print("[WARNING] Best configuration has small sample size. Do not tune production thresholds from this alone.")
    if float(best["average_selected_count"]) < 2.0:
        print("[WARNING] Best configuration improves by selecting very few assets.")
    if float(best["turnover_proxy"]) > 0.80:
        print("[WARNING] Best configuration has extreme turnover proxy.")
    if pd.notna(best["TP_rate"]) and pd.notna(best["SL_rate"]) and float(best["TP_rate"]) < float(best["SL_rate"]):
        print("[WARNING] Best configuration has TP rate below SL rate.")

    print(f"\nSaved: {Path(output_path).resolve()}")
    return result


if __name__ == "__main__":
    run_threshold_optimization()
