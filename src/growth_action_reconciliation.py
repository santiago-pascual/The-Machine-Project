from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATE_FILE = "growth_candidate_paper_state.csv"
ACTION_SIGNALS_FILE = "growth_candidate_action_signals.csv"
REBALANCE_REPORT_FILE = "growth_candidate_rebalance_report.csv"


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _num(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _latest_previous_state(current_date: str) -> pd.DataFrame:
    state = _read_csv(STATE_FILE)
    if state.empty or "date" not in state.columns:
        return pd.DataFrame()
    state = state.copy()
    state["date"] = pd.to_datetime(state["date"], errors="coerce")
    current_ts = pd.Timestamp(current_date).normalize()
    previous = state[state["date"].dt.normalize().lt(current_ts)]
    if previous.empty:
        return pd.DataFrame()
    latest_date = previous["date"].max()
    return previous[previous["date"].eq(latest_date)].copy()


def _append_or_update(path: str | Path, rows: pd.DataFrame, date: str, overwrite_same_day: bool) -> None:
    if rows.empty:
        return
    existing = _read_csv(path)
    if not existing.empty and "date" in existing.columns:
        if overwrite_same_day:
            existing = existing[existing["date"].astype(str).ne(str(date))]
        elif existing["date"].astype(str).eq(str(date)).any():
            return
    out = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows.copy()
    out.to_csv(path, index=False)


def _current_metadata(current_allocation: pd.DataFrame) -> dict[str, dict[str, object]]:
    if current_allocation.empty or "ticker" not in current_allocation.columns:
        return {}
    cols = [
        "raw_target_rank",
        "raw_target_return_exact",
        "quality_pass",
        "passed_tradability_filter",
        "holding_quality_classification",
    ]
    meta: dict[str, dict[str, object]] = {}
    for _, row in current_allocation.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker == "CASH":
            continue
        meta[ticker] = {col: row.get(col, np.nan) for col in cols if col in current_allocation.columns}
    return meta


def _action_for_weights(old_weight: float, new_weight: float, tolerance: float) -> tuple[str, str]:
    if old_weight > tolerance and new_weight <= tolerance:
        return "SELL", "removed_from_growth_allocation"
    if old_weight <= tolerance and new_weight > tolerance:
        return "BUY", "new_growth_selection"
    if new_weight - old_weight > tolerance:
        return "INCREASE", "target_weight_increased"
    if old_weight - new_weight > tolerance:
        return "REDUCE", "target_weight_reduced"
    return "HOLD", "still_selected_same_weight"


def reconcile_growth_actions(
    *,
    current_allocation: pd.DataFrame,
    current_date: str,
    portfolio_value: float,
    previous_prices: pd.Series | dict[str, float] | None = None,
    overwrite_same_day: bool = False,
    tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    previous = _latest_previous_state(current_date)
    previous_prices = pd.Series(previous_prices if previous_prices is not None else {}, dtype=float)
    current = current_allocation.copy() if current_allocation is not None else pd.DataFrame()

    prev_positions = previous[previous.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().ne("CASH")].copy() if not previous.empty else pd.DataFrame()
    prev_weights = {
        str(row["ticker"]).strip().upper(): _num(row.get("paper_position_weight", 0.0))
        for _, row in prev_positions.iterrows()
    }
    prev_values = {
        str(row["ticker"]).strip().upper(): _num(row.get("paper_position_value", np.nan), np.nan)
        for _, row in prev_positions.iterrows()
    }
    old_cash_rows = previous[previous.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")] if not previous.empty else pd.DataFrame()
    old_cash = _num(old_cash_rows.iloc[-1].get("paper_position_weight", np.nan), 1.0 - sum(prev_weights.values())) if not old_cash_rows.empty else 1.0 - sum(prev_weights.values())

    if "ticker" in current.columns:
        current = current[current["ticker"].astype(str).str.upper().ne("CASH")].copy()
    weight_col = "final_growth_weight" if "final_growth_weight" in current.columns else "paper_position_weight"
    current_weights = {
        str(row["ticker"]).strip().upper(): _num(row.get(weight_col, 0.0))
        for _, row in current.iterrows()
        if str(row.get("ticker", "")).strip().upper() != "CASH"
    }
    new_cash = _num(current["cash_weight"].dropna().iloc[0], 1.0 - sum(current_weights.values())) if "cash_weight" in current.columns and current["cash_weight"].notna().any() else 1.0 - sum(current_weights.values())
    metadata = _current_metadata(current)

    rows: list[dict[str, object]] = []
    all_tickers = sorted(set(prev_weights) | set(current_weights))
    for ticker in all_tickers:
        old_weight = _num(prev_weights.get(ticker, 0.0))
        new_weight = _num(current_weights.get(ticker, 0.0))
        weight_change = new_weight - old_weight
        action, reason = _action_for_weights(old_weight, new_weight, tolerance)
        meta = metadata.get(ticker, {})
        old_value = prev_values.get(ticker, old_weight * portfolio_value)
        if not np.isfinite(old_value):
            old_value = old_weight * portfolio_value
        rows.append(
            {
                "date": current_date,
                "ticker": ticker,
                "action": action,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "weight_change": weight_change,
                "old_position_value": old_value,
                "new_position_value": new_weight * portfolio_value,
                "estimated_trade_value": abs(weight_change) * portfolio_value,
                "execution_price": _num(previous_prices.get(ticker, np.nan), np.nan),
                "reason": reason,
                "raw_target_rank": meta.get("raw_target_rank", np.nan),
                "raw_target_return_exact": meta.get("raw_target_return_exact", np.nan),
                "quality_filter_pass": meta.get("quality_pass", np.nan),
                "tradability_filter_pass": meta.get("passed_tradability_filter", np.nan),
                "holding_quality_classification": meta.get("holding_quality_classification", np.nan),
            }
        )

    cash_change = new_cash - old_cash
    rows.append(
        {
            "date": current_date,
            "ticker": "CASH",
            "action": "CASH_CHANGE" if abs(cash_change) > tolerance else "HOLD",
            "old_weight": old_cash,
            "new_weight": new_cash,
            "weight_change": cash_change,
            "old_position_value": old_cash * portfolio_value,
            "new_position_value": new_cash * portfolio_value,
            "estimated_trade_value": abs(cash_change) * portfolio_value,
            "execution_price": 1.0,
            "reason": "cash_rebalance" if abs(cash_change) > tolerance else "cash_unchanged",
            "raw_target_rank": np.nan,
            "raw_target_return_exact": np.nan,
            "quality_filter_pass": np.nan,
            "tradability_filter_pass": np.nan,
            "holding_quality_classification": np.nan,
        }
    )

    signals = pd.DataFrame(rows)
    non_cash = signals[signals["ticker"].ne("CASH")].copy()
    turnover = float(non_cash["weight_change"].abs().sum()) if not non_cash.empty else 0.0
    reconstructed = all(
        abs(_num(row.old_weight) + _num(row.weight_change) - _num(row.new_weight)) <= tolerance
        for row in signals.itertuples(index=False)
    )
    turnover_matches = abs(turnover - float(non_cash["weight_change"].abs().sum())) <= tolerance
    reconciliation_passed = bool(reconstructed and turnover_matches)

    report = pd.DataFrame(
        [
            {
                "date": current_date,
                "previous_non_cash_holdings": ",".join(sorted(prev_weights)),
                "current_non_cash_holdings": ",".join(sorted(current_weights)),
                "old_cash": old_cash,
                "new_cash": new_cash,
                "cash_change": cash_change,
                "turnover": turnover,
                "buy_count": int(signals["action"].eq("BUY").sum()),
                "sell_count": int(signals["action"].eq("SELL").sum()),
                "increase_count": int(signals["action"].eq("INCREASE").sum()),
                "reduce_count": int(signals["action"].eq("REDUCE").sum()),
                "hold_count": int(signals["action"].eq("HOLD").sum()),
                "cash_change_count": int(signals["action"].eq("CASH_CHANGE").sum()),
                "reconstructed_portfolio_matches": reconstructed,
                "turnover_matches_abs_weight_change": turnover_matches,
                "reconciliation_passed": reconciliation_passed,
                "warning": "" if reconciliation_passed else "Rebalance reconciliation failed",
            }
        ]
    )

    _append_or_update(ACTION_SIGNALS_FILE, signals, current_date, overwrite_same_day)
    _append_or_update(REBALANCE_REPORT_FILE, report, current_date, overwrite_same_day)
    return signals, report, {"turnover": turnover, "reconciliation_passed": reconciliation_passed}


def signals_to_trade_rows(signals: pd.DataFrame, *, model_mode: str, variant: str) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    trade_actions = {"BUY", "SELL", "INCREASE", "REDUCE"}
    trades = signals[signals["action"].isin(trade_actions)].copy()
    if trades.empty:
        return pd.DataFrame()
    trades["previous_weight"] = trades["old_weight"]
    trades["trade_weight_change"] = trades["weight_change"]
    trades["model_mode"] = model_mode
    trades["growth_paper_variant"] = variant
    return trades[
        [
            "date",
            "ticker",
            "action",
            "previous_weight",
            "new_weight",
            "trade_weight_change",
            "execution_price",
            "reason",
            "model_mode",
            "growth_paper_variant",
            "raw_target_rank",
            "raw_target_return_exact",
            "quality_filter_pass",
            "tradability_filter_pass",
            "holding_quality_classification",
        ]
    ].copy()
