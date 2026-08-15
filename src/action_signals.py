from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PORTFOLIO_STATE_FILE = "portfolio_state.csv"


def _safe_numeric_series(data: pd.Series | dict | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if data is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(pd.Series(data).reindex(index), errors="coerce")


def load_previous_portfolio_state(state_path: str | Path = PORTFOLIO_STATE_FILE) -> pd.DataFrame:
    path = Path(state_path)
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "final_weight_percent"])
    state = pd.read_csv(path)
    if "ticker" not in state.columns:
        return pd.DataFrame(columns=["ticker", "final_weight_percent"])
    if "final_weight_percent" not in state.columns:
        state["final_weight_percent"] = 0.0
    state["ticker"] = state["ticker"].astype(str)
    state["final_weight_percent"] = pd.to_numeric(state["final_weight_percent"], errors="coerce").fillna(0.0)
    return state


def save_portfolio_state(
    final_allocation_table: pd.DataFrame,
    state_path: str | Path = PORTFOLIO_STATE_FILE,
) -> Path:
    path = Path(state_path)
    state = final_allocation_table.reset_index().rename(columns={"index": "ticker"}).copy()
    state = state[["ticker", "final_weight_percent"]]
    state["ticker"] = state["ticker"].astype(str)
    state["date_saved"] = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    state.to_csv(path, index=False)
    return path


def _target_confidence(diagnostics_df: pd.DataFrame, index: pd.Index) -> pd.Series:
    for column in ("target_confidence", "target_confidence_quant"):
        if column in diagnostics_df.columns:
            return pd.to_numeric(diagnostics_df[column], errors="coerce").reindex(index)
    return pd.Series(np.nan, index=index, dtype=float)


def build_action_signals(
    *,
    final_allocation_table: pd.DataFrame,
    selected_tickers: list[str],
    current_prices: pd.Series,
    target_prices: pd.Series,
    expected_daily_returns: pd.Series,
    diagnostics_df_full: pd.DataFrame,
    state_path: str | Path = PORTFOLIO_STATE_FILE,
) -> pd.DataFrame:
    previous_state = load_previous_portfolio_state(state_path)
    previous_weights = (
        previous_state.set_index("ticker")["final_weight_percent"]
        if not previous_state.empty
        else pd.Series(dtype=float)
    )
    selected_set = {str(ticker) for ticker in selected_tickers}
    previous_selected = set(previous_weights[previous_weights > 0].index.astype(str)) - {"CASH"}

    current_weights = final_allocation_table["final_weight_percent"].copy()
    current_weights.index = current_weights.index.astype(str)
    current_assets = set(current_weights.index.astype(str)) - {"CASH"}
    selected_assets = current_assets | selected_set
    sell_assets = previous_selected - selected_assets

    universe = pd.Index(
        sorted(
            set(diagnostics_df_full.index.astype(str))
            | current_assets
            | selected_set
            | sell_assets
            | {"CASH"}
        )
    )
    signal_strength = _safe_numeric_series(diagnostics_df_full.get("signal_strength"), universe, default=0.0).fillna(0.0)
    target_confidence = _target_confidence(diagnostics_df_full, universe)
    expected_returns = _safe_numeric_series(expected_daily_returns, universe, default=0.0).fillna(0.0)
    current_price_series = _safe_numeric_series(current_prices, universe)
    target_price_series = _safe_numeric_series(target_prices, universe)
    final_weights = _safe_numeric_series(current_weights, universe, default=0.0).fillna(0.0)
    prev_weights = _safe_numeric_series(previous_weights, universe, default=0.0).fillna(0.0)

    positive_expected = expected_returns[expected_returns > 0]
    strong_threshold = float(positive_expected.quantile(0.70)) if len(positive_expected) else 0.0
    strong_threshold = max(strong_threshold, 0.0002)

    rows: list[dict[str, object]] = []
    for ticker in universe:
        current_price = float(current_price_series.get(ticker, np.nan))
        target_price = float(target_price_series.get(ticker, np.nan))
        distance = (
            (target_price / current_price - 1.0) * 100.0
            if np.isfinite(current_price) and current_price > 0 and np.isfinite(target_price)
            else np.nan
        )
        final_weight = float(final_weights.get(ticker, 0.0))
        previous_weight = float(prev_weights.get(ticker, 0.0))
        expected_return = float(expected_returns.get(ticker, 0.0))
        confidence = float(target_confidence.get(ticker, np.nan))
        selected_now = ticker in selected_set or (ticker in current_assets and final_weight > 0)
        selected_before = ticker in previous_selected

        action = "HOLD"
        reason = "Existing selected asset remains in portfolio"

        if ticker == "CASH":
            action = "HOLD"
            reason = "Cash allocation"
        elif selected_before and not selected_now:
            action = "SELL"
            reason = "Previously selected asset is no longer selected"
        elif selected_now and not selected_before:
            action = "BUY"
            reason = "New selected asset"
        elif selected_now and selected_before:
            action = "HOLD"
            reason = "Still selected"
        elif expected_return >= strong_threshold and float(signal_strength.get(ticker, 0.0)) >= 0.2:
            action = "WATCHLIST"
            reason = "Strong candidate but not selected"
        else:
            continue

        if ticker != "CASH" and selected_now:
            if np.isfinite(current_price) and np.isfinite(target_price) and current_price >= target_price:
                action = "TAKE_PROFIT"
                reason = "Current price is at or above target price"
            elif np.isfinite(distance) and 0.0 <= distance <= 2.0:
                action = "REDUCE / NEAR_TARGET"
                reason = "Current price is within 2% of target price"
            elif np.isfinite(confidence) and confidence < 0.35:
                action = "REDUCE_CONFIDENCE"
                reason = "Target confidence deteriorated below 0.35"

        rows.append(
            {
                "ticker": ticker,
                "action": action,
                "current_price": current_price,
                "target_price": target_price,
                "distance_to_target_pct": distance,
                "expected_daily_return": expected_return,
                "final_weight_percent": final_weight,
                "previous_weight_percent": previous_weight,
                "weight_change_percent": final_weight - previous_weight,
                "target_confidence": confidence,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def print_action_signals(action_signals: pd.DataFrame) -> None:
    print("\n===== ACTION SIGNALS =====")
    if action_signals.empty:
        print("No action signals generated.")
        return

    groups = [
        ("BUY", ["BUY"]),
        ("HOLD", ["HOLD"]),
        ("SELL", ["SELL"]),
        ("REDUCE / TAKE_PROFIT", ["REDUCE / NEAR_TARGET", "REDUCE_CONFIDENCE", "TAKE_PROFIT"]),
        ("WATCHLIST", ["WATCHLIST"]),
    ]
    for title, actions in groups:
        subset = action_signals[action_signals["action"].isin(actions)]
        print(f"\n--- {title} ---")
        print(subset if not subset.empty else "None")


def generate_action_signals_report(
    *,
    final_allocation_table: pd.DataFrame,
    selected_tickers: list[str],
    current_prices: pd.Series,
    target_prices: pd.Series,
    expected_daily_returns: pd.Series,
    diagnostics_df_full: pd.DataFrame,
    state_path: str | Path = PORTFOLIO_STATE_FILE,
) -> pd.DataFrame:
    action_signals = build_action_signals(
        final_allocation_table=final_allocation_table,
        selected_tickers=selected_tickers,
        current_prices=current_prices,
        target_prices=target_prices,
        expected_daily_returns=expected_daily_returns,
        diagnostics_df_full=diagnostics_df_full,
        state_path=state_path,
    )
    print_action_signals(action_signals)
    save_portfolio_state(final_allocation_table, state_path=state_path)
    return action_signals
