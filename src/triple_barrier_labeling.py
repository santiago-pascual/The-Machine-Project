from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_OUTPUT_FILE = "triple_barrier_labels.csv"


def _load_prediction_source(
    walk_forward_path: str | Path = "walk_forward_predictions.csv",
    forecast_evaluated_path: str | Path = "forecast_history_evaluated.csv",
    forecast_history_path: str | Path = "forecast_history.csv",
) -> tuple[pd.DataFrame, str]:
    for path, source in [
        (Path(walk_forward_path), "walk_forward_predictions"),
        (Path(forecast_evaluated_path), "forecast_history_evaluated"),
        (Path(forecast_history_path), "forecast_history"),
    ]:
        if path.exists():
            data = pd.read_csv(path)
            if not data.empty and {"date", "ticker", "current_price"}.issubset(data.columns):
                return data, source
    return pd.DataFrame(), "none"


def _daily_volatility(prices_df: pd.DataFrame, ticker: str, decision_pos: int, lookback: int = 60) -> float:
    if ticker not in prices_df.columns:
        return 0.0
    start = max(0, decision_pos - lookback + 1)
    returns = prices_df[ticker].iloc[start : decision_pos + 1].pct_change(fill_method=None).dropna()
    vol = float(returns.std()) if len(returns) > 1 else 0.0
    return vol if np.isfinite(vol) and vol > 0 else 0.0


def _resolve_decision_position(price_index: pd.DatetimeIndex, date_value: object) -> int | None:
    date = pd.Timestamp(date_value).normalize()
    pos = int(price_index.searchsorted(date, side="right") - 1)
    if pos < 0 or pos >= len(price_index):
        return None
    return pos


def _label_one_path(
    *,
    prices_df: pd.DataFrame,
    ticker: str,
    decision_pos: int,
    horizon: int,
    current_price: float,
    tp_multiple: float,
    sl_multiple: float,
    daily_volatility: float,
) -> dict[str, object] | None:
    vertical_pos = decision_pos + horizon
    if ticker not in prices_df.columns or vertical_pos >= len(prices_df):
        return None
    if not np.isfinite(current_price) or current_price <= 0:
        return None

    volatility_horizon = float(daily_volatility * np.sqrt(horizon))
    take_profit_price = current_price * (1.0 + tp_multiple * volatility_horizon)
    stop_loss_price = current_price * (1.0 - sl_multiple * volatility_horizon)
    future_path = prices_df[ticker].iloc[decision_pos + 1 : vertical_pos + 1].dropna()
    if future_path.empty:
        return None

    first_touch_date = pd.NaT
    first_touch_type = "vertical_timeout"
    label = 0

    for path_date, price in future_path.items():
        price_value = float(price)
        hit_tp = price_value >= take_profit_price
        hit_sl = price_value <= stop_loss_price
        if hit_tp and hit_sl:
            first_touch_date = path_date
            first_touch_type = "ambiguous_touch"
            label = 0
            break
        if hit_tp:
            first_touch_date = path_date
            first_touch_type = "take_profit"
            label = 1
            break
        if hit_sl:
            first_touch_date = path_date
            first_touch_type = "stop_loss"
            label = -1
            break

    if pd.isna(first_touch_date):
        first_touch_date = future_path.index[-1]

    terminal_price = float(prices_df.loc[first_touch_date, ticker])
    realized_return = terminal_price / current_price - 1.0
    path_returns = future_path / current_price - 1.0
    max_favorable = float(path_returns.max()) if not path_returns.empty else 0.0
    max_adverse = float(path_returns.min()) if not path_returns.empty else 0.0
    time_to_touch = int(prices_df.index.get_loc(first_touch_date) - decision_pos)

    return {
        "horizon": int(horizon),
        "current_price": float(current_price),
        "take_profit_price": float(take_profit_price),
        "stop_loss_price": float(stop_loss_price),
        "vertical_barrier_date": prices_df.index[vertical_pos].strftime("%Y-%m-%d"),
        "first_touch_date": pd.Timestamp(first_touch_date).strftime("%Y-%m-%d"),
        "first_touch_type": first_touch_type,
        "label": int(label),
        "realized_return_at_barrier": float(realized_return),
        "max_favorable_excursion": max_favorable,
        "max_adverse_excursion": max_adverse,
        "time_to_first_touch": time_to_touch,
        "daily_volatility": float(daily_volatility),
        "volatility_horizon": volatility_horizon,
    }


def generate_triple_barrier_labels(
    *,
    prices_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    tp_multiple: float = 1.0,
    sl_multiple: float = 1.0,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    price_data = prices_df.copy()
    price_data.index = pd.to_datetime(price_data.index, errors="coerce").normalize()
    price_data = price_data[~price_data.index.isna()].sort_index().ffill()

    if predictions_df is None:
        predictions_df, source = _load_prediction_source()
    else:
        predictions_df = predictions_df.copy()
        source = "provided_predictions"

    if predictions_df.empty:
        labels = pd.DataFrame()
        labels.to_csv(output_path, index=False)
        print("\n===== TRIPLE BARRIER LABELING REPORT =====")
        print("No prediction source available.")
        return labels

    predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors="coerce").dt.normalize()
    predictions_df["ticker"] = predictions_df["ticker"].astype(str)
    rows: list[dict[str, object]] = []

    for _, prediction in predictions_df.dropna(subset=["date", "ticker"]).iterrows():
        ticker = str(prediction["ticker"])
        decision_pos = _resolve_decision_position(price_data.index, prediction["date"])
        if decision_pos is None:
            continue
        current_price = float(prediction.get("current_price", np.nan))
        if not np.isfinite(current_price) or current_price <= 0:
            if ticker in price_data.columns:
                current_price = float(price_data.iloc[decision_pos][ticker])
        daily_vol = _daily_volatility(price_data, ticker, decision_pos)

        for horizon in horizons:
            label_data = _label_one_path(
                prices_df=price_data,
                ticker=ticker,
                decision_pos=decision_pos,
                horizon=int(horizon),
                current_price=current_price,
                tp_multiple=float(tp_multiple),
                sl_multiple=float(sl_multiple),
                daily_volatility=daily_vol,
            )
            if label_data is None:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(prediction["date"]).strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "selected": bool(prediction.get("selected", False)),
                    **label_data,
                }
            )

    labels = pd.DataFrame(rows)
    labels.to_csv(output_path, index=False)
    print_triple_barrier_report(labels, source=source)
    return labels


def _distribution(labels: pd.DataFrame) -> pd.Series:
    if labels.empty:
        return pd.Series(dtype=float)
    return labels["first_touch_type"].value_counts(normalize=True).sort_index()


def print_triple_barrier_report(labels: pd.DataFrame, source: str = "unknown") -> None:
    print("\n===== TRIPLE BARRIER LABELING REPORT =====")
    print(f"source: {source}")
    print(f"number of labeled observations: {len(labels)}")
    if labels.empty:
        return

    distribution = labels["first_touch_type"].value_counts(normalize=True)
    print(f"% take profit: {float(distribution.get('take_profit', 0.0)):.2%}")
    print(f"% stop loss: {float(distribution.get('stop_loss', 0.0)):.2%}")
    print(f"% vertical timeout: {float(distribution.get('vertical_timeout', 0.0)):.2%}")
    print("average return by label:")
    print(labels.groupby("label")["realized_return_at_barrier"].mean())
    print(f"average time to first touch: {float(labels['time_to_first_touch'].mean()):.2f}")

    selected = labels[labels["selected"].astype(bool)]
    universe = labels
    print("selected-assets-only label distribution:")
    print(_distribution(selected) if not selected.empty else "No selected asset labels.")
    print("full-universe label distribution:")
    print(_distribution(universe))
