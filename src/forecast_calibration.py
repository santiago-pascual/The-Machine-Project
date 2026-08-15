from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FORECAST_HISTORY_FILE = "forecast_history.csv"
DEFAULT_HORIZONS = (5, 10, 20)


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _get_optional_series(
    diagnostics_df: pd.DataFrame,
    column: str,
    index: pd.Index,
    default: float = np.nan,
) -> pd.Series:
    if column not in diagnostics_df.columns:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(diagnostics_df[column], errors="coerce").reindex(index)


def build_prediction_snapshot(
    *,
    prices_df: pd.DataFrame,
    target_prices: pd.Series,
    expected_daily_returns: pd.Series,
    diagnostics_df: pd.DataFrame,
    selected_assets: Iterable[str],
    final_weight_percent: pd.Series,
    regime: str,
) -> pd.DataFrame:
    tickers = pd.Index(prices_df.columns.astype(str), name="ticker")
    decision_date = pd.Timestamp(prices_df.index[-1]).normalize()
    current_prices = prices_df.ffill().iloc[-1].reindex(tickers)
    expected_daily = pd.to_numeric(expected_daily_returns, errors="coerce").reindex(tickers).fillna(0.0)
    days_to_target = _get_optional_series(diagnostics_df, "time_to_target", tickers, default=20.0).fillna(20.0).clip(lower=1.0)
    expected_total = (1.0 + expected_daily).pow(days_to_target) - 1.0
    selected_set = {str(ticker) for ticker in selected_assets}
    weights = pd.to_numeric(final_weight_percent, errors="coerce").reindex(tickers).fillna(0.0)

    target_confidence = _get_optional_series(diagnostics_df, "target_confidence", tickers)
    if target_confidence.isna().all():
        target_confidence = _get_optional_series(diagnostics_df, "target_confidence_quant", tickers)

    snapshot = pd.DataFrame(
        {
            "date": decision_date.strftime("%Y-%m-%d"),
            "ticker": tickers,
            "current_price": pd.to_numeric(current_prices, errors="coerce").values,
            "target_price": pd.to_numeric(target_prices, errors="coerce").reindex(tickers).values,
            "expected_daily_return": expected_daily.values,
            "expected_total_return": expected_total.values,
            "target_confidence": target_confidence.values,
            "signal_strength": _get_optional_series(diagnostics_df, "signal_strength", tickers, default=0.0).fillna(0.0).values,
            "quality_score": _get_optional_series(diagnostics_df, "quality_score", tickers, default=np.nan).values,
            "regime": str(regime),
            "selected": [ticker in selected_set for ticker in tickers],
            "final_weight_percent": weights.values,
            "raw_target_return_exact": _get_optional_series(diagnostics_df, "raw_target_return_exact", tickers).values,
            "raw_expected_daily_return_exact": _get_optional_series(diagnostics_df, "raw_expected_daily_return_exact", tickers).values,
            "raw_target_price_exact": _get_optional_series(diagnostics_df, "raw_target_price_exact", tickers).values,
            "time_to_target": days_to_target.values,
            "signal_strength_adjustment_value": _get_optional_series(diagnostics_df, "signal_strength_adjustment_value", tickers).values,
            "final_expected_return_after_adjustments": _get_optional_series(diagnostics_df, "final_expected_return_after_adjustments", tickers).values,
        }
    )
    return snapshot.replace([np.inf, -np.inf], np.nan)


def append_prediction_snapshot(
    snapshot_df: pd.DataFrame,
    history_path: str | Path = FORECAST_HISTORY_FILE,
    overwrite_same_day: bool = False,
) -> Path:
    path = Path(history_path)
    key_cols = ["date", "ticker"]
    snapshot = snapshot_df.copy()
    snapshot["date"] = pd.to_datetime(snapshot["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    snapshot["ticker"] = snapshot["ticker"].astype(str)

    if path.exists():
        history = pd.read_csv(path)
        if not history.empty and all(col in history.columns for col in key_cols):
            history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            history["ticker"] = history["ticker"].astype(str)
        else:
            history = pd.DataFrame(columns=snapshot.columns)
    else:
        history = pd.DataFrame(columns=snapshot.columns)

    existing_keys = set(zip(history.get("date", pd.Series(dtype=str)), history.get("ticker", pd.Series(dtype=str))))
    snapshot_keys = list(zip(snapshot["date"], snapshot["ticker"]))
    duplicate_mask = pd.Series([key in existing_keys for key in snapshot_keys], index=snapshot.index)
    duplicate_rows = int(duplicate_mask.sum())

    if overwrite_same_day and duplicate_rows > 0 and not history.empty:
        update_keys = set(snapshot.loc[duplicate_mask, key_cols].itertuples(index=False, name=None))
        keep_history = [
            (date, ticker) not in update_keys
            for date, ticker in zip(history["date"], history["ticker"])
        ]
        history = history.loc[keep_history].copy()
        rows_to_add = snapshot.copy()
        rows_overwritten = duplicate_rows
    else:
        rows_to_add = snapshot.loc[~duplicate_mask].copy()
        rows_overwritten = 0

    if not rows_to_add.empty:
        combined_columns = list(dict.fromkeys(list(history.columns) + list(rows_to_add.columns)))
        rows_to_add = rows_to_add.reindex(columns=combined_columns)
        history = history.reindex(columns=combined_columns)
        write_header = not path.exists() or path.stat().st_size == 0
        schema_changed = path.exists() and set(pd.read_csv(path, nrows=0).columns) != set(combined_columns)
        if overwrite_same_day and duplicate_rows > 0 or schema_changed:
            combined = pd.concat([history, rows_to_add], axis=0, ignore_index=True)
            combined.to_csv(path, index=False)
        else:
            rows_to_add.to_csv(path, mode="a", header=write_header, index=False)

    total_rows = len(load_forecast_history(path)) if path.exists() else 0
    print("\n===== FORECAST HISTORY UPDATE =====")
    print(f"new rows added: {len(rows_to_add)}")
    print(f"duplicate rows skipped: {0 if overwrite_same_day else duplicate_rows}")
    print(f"rows overwritten: {rows_overwritten}")
    print(f"total history rows: {total_rows}")
    return path


def load_forecast_history(history_path: str | Path = FORECAST_HISTORY_FILE) -> pd.DataFrame:
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame()
    history = pd.read_csv(path)
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    return history


def add_forward_evaluation_labels(
    history_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    if history_df.empty or prices_df.empty:
        return history_df.copy()

    evaluated = history_df.copy()
    price_data = prices_df.copy()
    price_data.index = pd.to_datetime(price_data.index, errors="coerce").normalize()
    price_data = price_data[~price_data.index.isna()].sort_index()
    price_index = price_data.index

    for horizon in horizons:
        evaluated[f"realized_return_{horizon}d"] = np.nan

    for row_idx, row in evaluated.iterrows():
        ticker = str(row.get("ticker", ""))
        decision_date = row.get("date")
        if ticker not in price_data.columns or pd.isna(decision_date):
            continue

        decision_pos = int(price_index.searchsorted(pd.Timestamp(decision_date), side="right") - 1)
        if decision_pos < 0:
            continue

        start_price = _safe_float(price_data.iloc[decision_pos][ticker])
        if not np.isfinite(start_price) or start_price <= 0:
            start_price = _safe_float(row.get("current_price"))
        if not np.isfinite(start_price) or start_price <= 0:
            continue

        for horizon in horizons:
            future_pos = decision_pos + int(horizon)
            if future_pos >= len(price_data):
                continue
            future_price = _safe_float(price_data.iloc[future_pos][ticker])
            if np.isfinite(future_price) and future_price > 0:
                evaluated.at[row_idx, f"realized_return_{horizon}d"] = future_price / start_price - 1.0

    return evaluated


def _metric_block(
    df: pd.DataFrame,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {"observations": float(len(df))}
    expected_daily = pd.to_numeric(
        df.get("expected_daily_return", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    )

    for horizon in horizons:
        realized_col = f"realized_return_{horizon}d"
        realized = pd.to_numeric(
            df.get(realized_col, pd.Series(index=df.index, dtype=float)),
            errors="coerce",
        )
        expected_horizon = (1.0 + expected_daily).pow(horizon) - 1.0
        valid = expected_horizon.notna() & realized.notna()

        if not valid.any():
            metrics[f"direction_accuracy_{horizon}d"] = np.nan
            metrics[f"mae_{horizon}d"] = np.nan
            metrics[f"rmse_{horizon}d"] = np.nan
            metrics[f"correlation_{horizon}d"] = np.nan
            metrics[f"hit_rate_{horizon}d"] = np.nan
            metrics[f"false_positive_rate_{horizon}d"] = np.nan
            continue

        exp_valid = expected_horizon[valid]
        real_valid = realized[valid]
        errors = exp_valid - real_valid
        positive_predictions = exp_valid > 0
        metrics[f"direction_accuracy_{horizon}d"] = float((np.sign(exp_valid) == np.sign(real_valid)).mean())
        metrics[f"mae_{horizon}d"] = float(errors.abs().mean())
        metrics[f"rmse_{horizon}d"] = float(np.sqrt((errors.pow(2)).mean()))
        metrics[f"correlation_{horizon}d"] = float(exp_valid.corr(real_valid)) if len(exp_valid) > 1 else np.nan
        metrics[f"hit_rate_{horizon}d"] = float(((exp_valid > 0) & (real_valid > 0)).sum() / max(1, positive_predictions.sum()))
        metrics[f"false_positive_rate_{horizon}d"] = float(((exp_valid > 0) & (real_valid < 0)).sum() / max(1, positive_predictions.sum()))

    return metrics


def compute_calibration_metrics(
    evaluated_history: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, dict[str, float]]:
    if evaluated_history.empty:
        return {"universe": _metric_block(evaluated_history, horizons), "selected": _metric_block(evaluated_history, horizons)}

    horizon_cols = [f"realized_return_{horizon}d" for horizon in horizons]
    evaluable = evaluated_history.dropna(subset=horizon_cols, how="all")
    selected = (
        evaluable[evaluable["selected"].astype(bool)]
        if "selected" in evaluable.columns
        else evaluable.iloc[0:0]
    )
    return {
        "universe": _metric_block(evaluable, horizons),
        "selected": _metric_block(selected, horizons),
    }


def compute_confidence_quintiles(
    evaluated_history: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    realized_col = f"realized_return_{horizon}d"
    required = ["target_confidence", realized_col]
    if evaluated_history.empty or any(col not in evaluated_history.columns for col in required):
        return pd.DataFrame()

    data = evaluated_history[required].copy()
    data["target_confidence"] = pd.to_numeric(data["target_confidence"], errors="coerce")
    data[realized_col] = pd.to_numeric(data[realized_col], errors="coerce")
    data = data.dropna()
    if len(data) < 5 or data["target_confidence"].nunique() < 2:
        return pd.DataFrame()

    data["confidence_quintile"] = pd.qcut(
        data["target_confidence"],
        q=5,
        labels=False,
        duplicates="drop",
    )
    return data.groupby("confidence_quintile", dropna=True)[realized_col].agg(["count", "mean", "median"])


def print_calibration_report(
    evaluated_history: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> None:
    metrics = compute_calibration_metrics(evaluated_history, horizons=horizons)
    universe = metrics["universe"]
    selected = metrics["selected"]

    print("\n===== FORECAST CALIBRATION =====")
    print(f"observations evaluated: {int(universe.get('observations', 0))}")

    for horizon in horizons:
        print(f"direction accuracy {horizon}d: {universe.get(f'direction_accuracy_{horizon}d', np.nan):.4f}")
    for horizon in horizons:
        print(f"MAE {horizon}d: {universe.get(f'mae_{horizon}d', np.nan):.6f}")
    for horizon in horizons:
        print(f"RMSE {horizon}d: {universe.get(f'rmse_{horizon}d', np.nan):.6f}")
    for horizon in horizons:
        print(f"correlation {horizon}d: {universe.get(f'correlation_{horizon}d', np.nan):.4f}")
    for horizon in horizons:
        print(f"hit rate {horizon}d: {universe.get(f'hit_rate_{horizon}d', np.nan):.4f}")
    for horizon in horizons:
        print(f"false positive rate {horizon}d: {universe.get(f'false_positive_rate_{horizon}d', np.nan):.4f}")

    print("\nSelected assets only:")
    print(f"observations evaluated: {int(selected.get('observations', 0))}")
    for horizon in horizons:
        print(
            f"{horizon}d accuracy / MAE / corr: "
            f"{selected.get(f'direction_accuracy_{horizon}d', np.nan):.4f} / "
            f"{selected.get(f'mae_{horizon}d', np.nan):.6f} / "
            f"{selected.get(f'correlation_{horizon}d', np.nan):.4f}"
        )

    print("\nConfidence quintiles, average realized return:")
    quintiles = compute_confidence_quintiles(evaluated_history, horizon=max(horizons))
    print(quintiles if not quintiles.empty else "Not enough evaluated observations yet.")


def save_and_evaluate_forecasts(
    *,
    prices_df: pd.DataFrame,
    target_prices: pd.Series,
    expected_daily_returns: pd.Series,
    diagnostics_df: pd.DataFrame,
    selected_assets: Iterable[str],
    final_weight_percent: pd.Series,
    regime: str,
    history_path: str | Path = FORECAST_HISTORY_FILE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    overwrite_same_day: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot = build_prediction_snapshot(
        prices_df=prices_df,
        target_prices=target_prices,
        expected_daily_returns=expected_daily_returns,
        diagnostics_df=diagnostics_df,
        selected_assets=selected_assets,
        final_weight_percent=final_weight_percent,
        regime=regime,
    )
    path = append_prediction_snapshot(
        snapshot,
        history_path=history_path,
        overwrite_same_day=overwrite_same_day,
    )
    history = load_forecast_history(path)
    evaluated = add_forward_evaluation_labels(history, prices_df, horizons=horizons)
    evaluated.to_csv(Path(history_path).with_name("forecast_history_evaluated.csv"), index=False)
    print_calibration_report(evaluated, horizons=horizons)
    return snapshot, evaluated
