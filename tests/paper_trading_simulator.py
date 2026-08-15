from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PAPER_STATE_FILE = "paper_portfolio_state.csv"
PAPER_TRADES_FILE = "paper_trades_log.csv"
PAPER_PERFORMANCE_FILE = "paper_performance.csv"
DEFAULT_INITIAL_CAPITAL = 100000.0
SUPPORTED_PAPER_MODEL_MODES = {
    "baseline",
    "full_quant_research",
    "regime_gated_full_quant",
    "calibrated_forecast_research",
    "raw_target_research",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _today_string(prices_df: pd.DataFrame | None = None) -> str:
    if prices_df is not None and not prices_df.empty:
        return pd.Timestamp(prices_df.index[-1]).strftime("%Y-%m-%d")
    return pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def _latest_prices(prices_df: pd.DataFrame, current_prices: pd.Series | None = None) -> pd.Series:
    if current_prices is not None:
        return pd.to_numeric(pd.Series(current_prices), errors="coerce")
    if prices_df.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(prices_df.ffill().iloc[-1], errors="coerce")


def _prepare_allocation(final_allocation_table: pd.DataFrame) -> pd.Series:
    if final_allocation_table.empty or "final_weight_percent" not in final_allocation_table.columns:
        return pd.Series(dtype=float)
    weights = pd.to_numeric(final_allocation_table["final_weight_percent"], errors="coerce").fillna(0.0) / 100.0
    weights.index = weights.index.astype(str)
    return weights.clip(lower=0.0)


def _previous_portfolio_value(performance_path: str | Path = PAPER_PERFORMANCE_FILE) -> float:
    perf = _read_csv(performance_path)
    if perf.empty or "portfolio_value" not in perf.columns:
        return DEFAULT_INITIAL_CAPITAL
    values = pd.to_numeric(perf["portfolio_value"], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else DEFAULT_INITIAL_CAPITAL


def _previous_state(state_path: str | Path = PAPER_STATE_FILE) -> pd.DataFrame:
    state = _read_csv(state_path)
    if state.empty:
        return pd.DataFrame()
    state["ticker"] = state["ticker"].astype(str)
    for col in ["paper_position_weight", "paper_position_value", "entry_price", "current_price"]:
        if col in state.columns:
            state[col] = pd.to_numeric(state[col], errors="coerce")
    return state


def _skip_same_day(path: str | Path, date: str, overwrite_same_day: bool) -> bool:
    if overwrite_same_day:
        return False
    df = _read_csv(path)
    return not df.empty and "date" in df.columns and df["date"].astype(str).eq(str(date)).any()


def _append_rows(path: str | Path, rows: pd.DataFrame, date: str, overwrite_same_day: bool = False) -> None:
    file_path = Path(path)
    existing = _read_csv(file_path)
    if overwrite_same_day and not existing.empty and "date" in existing.columns:
        existing = existing[~existing["date"].astype(str).eq(str(date))]
    output = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows
    output.to_csv(file_path, index=False)


def _compute_realized_daily_return(previous: pd.DataFrame, prices: pd.Series) -> float:
    if previous.empty or "ticker" not in previous.columns:
        return 0.0
    total = 0.0
    for _, row in previous.iterrows():
        ticker = str(row.get("ticker", ""))
        if ticker == "CASH":
            continue
        prev_weight = _safe_float(row.get("paper_position_weight", 0.0))
        prev_price = _safe_float(row.get("current_price", np.nan), np.nan)
        current_price = _safe_float(prices.get(ticker, np.nan), np.nan)
        if np.isfinite(prev_price) and prev_price > 0 and np.isfinite(current_price):
            total += prev_weight * (current_price / prev_price - 1.0)
    return float(total)


def _performance_metrics(perf: pd.DataFrame) -> dict[str, float]:
    if perf.empty or "daily_return" not in perf.columns:
        return {"volatility": 0.0, "Sharpe": 0.0, "max_drawdown": 0.0, "cumulative_return": 0.0}
    returns = pd.to_numeric(perf["daily_return"], errors="coerce").dropna()
    if returns.empty:
        return {"volatility": 0.0, "Sharpe": 0.0, "max_drawdown": 0.0, "cumulative_return": 0.0}
    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
    annual_return = float((1.0 + returns).prod() ** (252 / max(1, len(returns))) - 1.0)
    sharpe = annual_return / volatility if volatility > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "volatility": volatility,
        "Sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "cumulative_return": float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0,
    }


def _governance_message() -> str:
    dashboard = _read_csv("research_dashboard_summary.csv")
    if dashboard.empty or not {"metric", "value"}.issubset(dashboard.columns):
        return "Paper trading allowed. Governance status unavailable."
    status = dashboard.loc[dashboard["metric"].astype(str).eq("promotion_status"), "value"]
    if not status.empty and str(status.iloc[-1]).lower() == "blocked":
        return "Paper trading allowed, production promotion blocked."
    return "Paper trading allowed. No production promotion performed."


def update_paper_trading_simulation(
    *,
    final_allocation_table: pd.DataFrame,
    action_signals: pd.DataFrame,
    prices_df: pd.DataFrame,
    model_mode: str = "baseline",
    current_prices: pd.Series | None = None,
    overwrite_same_day: bool = False,
    state_path: str | Path = PAPER_STATE_FILE,
    trades_path: str | Path = PAPER_TRADES_FILE,
    performance_path: str | Path = PAPER_PERFORMANCE_FILE,
) -> dict[str, pd.DataFrame]:
    date = _today_string(prices_df)
    if str(model_mode) not in SUPPORTED_PAPER_MODEL_MODES:
        print(f"[WARNING] Unknown paper model mode: {model_mode}. State will still be recorded.")
    if _skip_same_day(performance_path, date, overwrite_same_day):
        print("\n===== PAPER TRADING SIMULATION =====")
        print(f"date {date} already exists. Skipped. Use overwrite_same_day=True to update.")
        return {
            "state": _read_csv(state_path),
            "trades": pd.DataFrame(),
            "performance": _read_csv(performance_path),
        }

    prices = _latest_prices(prices_df, current_prices)
    previous = _previous_state(state_path)
    previous_weights = (
        previous.set_index("ticker")["paper_position_weight"]
        if not previous.empty and "paper_position_weight" in previous.columns
        else pd.Series(dtype=float)
    )
    target_weights = _prepare_allocation(final_allocation_table)
    if "CASH" not in target_weights.index:
        target_weights.loc["CASH"] = max(0.0, 1.0 - float(target_weights.sum()))
    target_weights = target_weights / max(1.0, float(target_weights.sum()))

    previous_value = _previous_portfolio_value(performance_path)
    daily_return = _compute_realized_daily_return(previous, prices)
    portfolio_value = previous_value * (1.0 + daily_return)

    all_tickers = sorted(set(previous_weights.index.astype(str)) | set(target_weights.index.astype(str)))
    action_map = (
        action_signals.set_index("ticker")["action"].astype(str)
        if not action_signals.empty and {"ticker", "action"}.issubset(action_signals.columns)
        else pd.Series(dtype=str)
    )
    reason_map = (
        action_signals.set_index("ticker")["reason"].astype(str)
        if not action_signals.empty and {"ticker", "reason"}.issubset(action_signals.columns)
        else pd.Series(dtype=str)
    )
    rows = []
    trade_rows = []
    turnover = 0.0
    for ticker in all_tickers:
        prev_weight = _safe_float(previous_weights.get(ticker, 0.0))
        new_weight = _safe_float(target_weights.get(ticker, 0.0))
        change = new_weight - prev_weight
        turnover += abs(change)
        current_price = 1.0 if ticker == "CASH" else _safe_float(prices.get(ticker, np.nan), np.nan)
        prev_entry = np.nan
        prev_price = np.nan
        if not previous.empty and ticker in set(previous["ticker"].astype(str)):
            prev_row = previous[previous["ticker"].astype(str).eq(ticker)].iloc[-1]
            prev_entry = _safe_float(prev_row.get("entry_price", np.nan), np.nan)
            prev_price = _safe_float(prev_row.get("current_price", np.nan), np.nan)
        entry_price = current_price if prev_weight <= 0 and new_weight > 0 else prev_entry
        unrealized = (
            current_price / entry_price - 1.0
            if ticker != "CASH" and np.isfinite(entry_price) and entry_price > 0 and np.isfinite(current_price)
            else 0.0
        )
        realized = (
            current_price / prev_price - 1.0
            if ticker != "CASH" and prev_weight > 0 and new_weight <= 0 and np.isfinite(prev_price) and prev_price > 0
            else 0.0
        )
        action = str(action_map.get(ticker, "HOLD" if abs(change) < 1e-8 else ("BUY" if change > 0 else "SELL")))
        reason = str(reason_map.get(ticker, "paper rebalance"))
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "paper_position_weight": new_weight,
                "paper_position_value": portfolio_value * new_weight,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_return": unrealized,
                "realized_return": realized,
                "action": action,
                "model_mode": model_mode,
                "cash_weight": float(target_weights.get("CASH", 0.0)),
            }
        )
        if abs(change) > 1e-8:
            trade_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": action,
                    "previous_weight": prev_weight,
                    "new_weight": new_weight,
                    "trade_weight_change": change,
                    "execution_price": current_price,
                    "reason": reason,
                    "model_mode": model_mode,
                }
            )

    state = pd.DataFrame(rows)
    trades = pd.DataFrame(trade_rows)
    _append_rows(state_path, state, date, overwrite_same_day=overwrite_same_day)
    if not trades.empty:
        _append_rows(trades_path, trades, date, overwrite_same_day=overwrite_same_day)

    existing_perf = _read_csv(performance_path)
    temp_perf = pd.concat(
        [
            existing_perf,
            pd.DataFrame(
                [
                    {
                        "date": date,
                        "model_mode": model_mode,
                        "portfolio_value": portfolio_value,
                        "daily_return": daily_return,
                        "cash_weight": float(target_weights.get("CASH", 0.0)),
                        "turnover": turnover / 2.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metrics = _performance_metrics(temp_perf)
    performance_row = pd.DataFrame(
        [
            {
                "date": date,
                "model_mode": model_mode,
                "portfolio_value": portfolio_value,
                "daily_return": daily_return,
                "cumulative_return": metrics["cumulative_return"],
                "volatility": metrics["volatility"],
                "Sharpe": metrics["Sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "cash_weight": float(target_weights.get("CASH", 0.0)),
                "turnover": turnover / 2.0,
            }
        ]
    )
    _append_rows(performance_path, performance_row, date, overwrite_same_day=overwrite_same_day)

    print_paper_trading_report(state, trades, performance_row, model_mode)
    return {"state": state, "trades": trades, "performance": performance_row}


def print_paper_trading_report(
    state: pd.DataFrame,
    trades: pd.DataFrame,
    performance: pd.DataFrame,
    model_mode: str,
) -> None:
    print("\n===== PAPER TRADING SIMULATION =====")
    print(f"model mode: {model_mode}")
    if not performance.empty:
        row = performance.iloc[-1]
        print(f"portfolio value: {float(row['portfolio_value']):.2f}")
        print(f"daily return: {float(row['daily_return']):.6f}")
        print(f"cumulative return: {float(row['cumulative_return']):.6f}")
        print(f"cash: {float(row['cash_weight']):.4f}")
        print(f"turnover: {float(row['turnover']):.4f}")
        print(f"max drawdown: {float(row['max_drawdown']):.6f}")
    holdings = state[state["paper_position_weight"] > 0].copy() if not state.empty else pd.DataFrame()
    print("current holdings:")
    display_cols = ["ticker", "paper_position_weight", "paper_position_value", "current_price", "unrealized_return", "action"]
    print(holdings[display_cols].to_string(index=False) if not holdings.empty else "None")
    print("actions taken:")
    print(trades.to_string(index=False) if not trades.empty else "None")
    print(_governance_message())
