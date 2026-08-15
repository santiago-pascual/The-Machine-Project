from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MONITOR_CSV = "paper_trading_monitor_report.csv"
MONITOR_TXT = "paper_trading_monitor_report.txt"


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _dashboard_value(dashboard: pd.DataFrame, metric: str, default: Any = np.nan) -> Any:
    if dashboard.empty or not {"metric", "value"}.issubset(dashboard.columns):
        return default
    rows = dashboard[dashboard["metric"].astype(str).eq(metric)]
    if rows.empty:
        return default
    return rows.iloc[-1]["value"]


def _paper_metrics(performance: pd.DataFrame, state: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    if performance.empty:
        return {
            "paper_history_status": "missing",
            "paper_model_mode": "missing",
            "paper_cumulative_return": np.nan,
            "paper_daily_return": np.nan,
            "paper_volatility": np.nan,
            "paper_sharpe": np.nan,
            "paper_sortino": np.nan,
            "paper_calmar": np.nan,
            "paper_max_drawdown": np.nan,
            "paper_cash": np.nan,
            "paper_turnover": np.nan,
            "number_of_trades": int(len(trades)),
            "trade_win_rate": np.nan,
            "current_holdings": "",
        }
    perf = performance.copy()
    returns = pd.to_numeric(perf.get("daily_return", pd.Series(dtype=float)), errors="coerce").dropna()
    last = perf.iloc[-1]
    cumulative = _safe_float(last.get("cumulative_return", np.nan))
    daily_return = _safe_float(last.get("daily_return", np.nan))
    volatility = _safe_float(last.get("volatility", np.nan))
    sharpe = _safe_float(last.get("Sharpe", np.nan))
    max_dd = _safe_float(last.get("max_drawdown", np.nan))
    if not np.isfinite(volatility) and len(returns) > 1:
        volatility = float(returns.std() * np.sqrt(252))
    if not np.isfinite(sharpe) and len(returns) > 1 and volatility > 0:
        ann_return = float((1.0 + returns).prod() ** (252 / len(returns)) - 1.0)
        sharpe = ann_return / volatility
    downside = returns[returns < 0]
    downside_dev = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
    ann_return = float((1.0 + returns).prod() ** (252 / max(1, len(returns))) - 1.0) if len(returns) else np.nan
    sortino = ann_return / downside_dev if downside_dev > 0 and np.isfinite(ann_return) else 0.0
    calmar = ann_return / abs(max_dd) if np.isfinite(ann_return) and np.isfinite(max_dd) and max_dd < 0 else 0.0
    cash = _safe_float(last.get("cash_weight", np.nan))
    turnover = _safe_float(last.get("turnover", np.nan))
    holdings = ""
    if not state.empty and {"ticker", "paper_position_weight"}.issubset(state.columns):
        latest_date = state["date"].astype(str).max() if "date" in state.columns else None
        latest_state = state[state["date"].astype(str).eq(latest_date)] if latest_date else state
        weights = pd.to_numeric(latest_state["paper_position_weight"], errors="coerce").fillna(0.0)
        holdings = ", ".join(latest_state.loc[weights > 0, "ticker"].astype(str).tolist())
    win_rate = np.nan
    if not trades.empty and "realized_return" in trades.columns:
        realized = pd.to_numeric(trades["realized_return"], errors="coerce").dropna()
        win_rate = float((realized > 0).mean()) if len(realized) else np.nan
    return {
        "paper_history_status": "too_short" if len(perf) < 20 else "ok",
        "paper_model_mode": str(last.get("model_mode", "missing")),
        "paper_cumulative_return": cumulative,
        "paper_daily_return": daily_return,
        "paper_volatility": volatility,
        "paper_sharpe": sharpe,
        "paper_sortino": sortino,
        "paper_calmar": calmar,
        "paper_max_drawdown": max_dd,
        "paper_cash": cash,
        "paper_turnover": turnover,
        "number_of_trades": int(len(trades)),
        "trade_win_rate": win_rate,
        "current_holdings": holdings,
    }


def _expected_metrics(larger: pd.DataFrame, clean_eval: pd.DataFrame, dashboard: pd.DataFrame, model_mode: str) -> dict[str, Any]:
    expected = {
        "expected_model_mode": model_mode,
        "expected_sharpe": np.nan,
        "expected_return": np.nan,
        "expected_volatility": np.nan,
        "expected_max_drawdown": np.nan,
        "expected_cash": np.nan,
        "expected_turnover": np.nan,
        "governed_classification": _dashboard_value(dashboard, "governed_promotion_classification", "missing"),
    }
    if not larger.empty and "model_mode" in larger.columns:
        rows = larger[larger["model_mode"].astype(str).eq(model_mode)]
        if rows.empty and model_mode != "baseline":
            rows = larger[larger["model_mode"].astype(str).eq("regime_gated_full_quant")]
        if rows.empty:
            rows = larger[larger["model_mode"].astype(str).eq("baseline")]
        if not rows.empty:
            row = rows.iloc[-1]
            expected.update(
                {
                    "expected_model_mode": str(row.get("model_mode", model_mode)),
                    "expected_sharpe": _safe_float(row.get("Sharpe", np.nan)),
                    "expected_return": _safe_float(row.get("realized_return", np.nan)),
                    "expected_volatility": _safe_float(row.get("realized_volatility", np.nan)),
                    "expected_max_drawdown": _safe_float(row.get("max_drawdown", np.nan)),
                    "expected_cash": _safe_float(row.get("average_cash", np.nan)),
                    "expected_turnover": _safe_float(row.get("average_turnover", np.nan)),
                }
            )
    if not clean_eval.empty and "trial_group" in clean_eval.columns:
        governed = clean_eval[clean_eval["trial_group"].astype(str).eq("governed_trials")]
        if not governed.empty:
            expected["governed_classification"] = str(governed.iloc[-1].get("promotion_classification", expected["governed_classification"]))
    return expected


def _drift_flags(paper: dict[str, Any], expected: dict[str, Any], state: pd.DataFrame, forecast: pd.DataFrame) -> list[str]:
    flags: list[str] = []
    if paper["paper_history_status"] == "missing":
        return ["paper_history_missing"]
    if paper["paper_history_status"] == "too_short":
        flags.append("paper_history_too_short")
    if np.isfinite(paper["paper_max_drawdown"]) and np.isfinite(expected["expected_max_drawdown"]):
        if paper["paper_max_drawdown"] < expected["expected_max_drawdown"] * 1.25:
            flags.append("drawdown_worse_than_expected")
    if np.isfinite(paper["paper_turnover"]) and np.isfinite(expected["expected_turnover"]):
        if paper["paper_turnover"] > expected["expected_turnover"] * 1.50:
            flags.append("turnover_higher_than_expected")
    if np.isfinite(paper["paper_cash"]) and np.isfinite(expected["expected_cash"]):
        if abs(paper["paper_cash"] - expected["expected_cash"]) > 0.20:
            flags.append("cash_materially_different")
    if np.isfinite(paper["paper_cumulative_return"]) and np.isfinite(expected["expected_return"]):
        if paper["paper_cumulative_return"] < expected["expected_return"] * 0.50:
            flags.append("underperforming_expected_return")
    if np.isfinite(paper["paper_volatility"]) and np.isfinite(expected["expected_volatility"]):
        if paper["paper_volatility"] > expected["expected_volatility"] * 1.50:
            flags.append("volatility_higher_than_expected")
    if not state.empty and not forecast.empty and {"ticker", "paper_position_weight"}.issubset(state.columns):
        latest_date = state["date"].astype(str).max() if "date" in state.columns else None
        current_holdings = set(state.loc[state["date"].astype(str).eq(latest_date) & (pd.to_numeric(state["paper_position_weight"], errors="coerce") > 0), "ticker"].astype(str)) if latest_date else set()
        expected_selected = set()
        if {"ticker", "selected"}.issubset(forecast.columns):
            expected_selected = set(forecast[forecast["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]["ticker"].astype(str))
        if expected_selected and len(current_holdings.symmetric_difference(expected_selected)) > max(2, len(expected_selected) * 0.5):
            flags.append("selected_tickers_diverge_from_model")
    return flags or ["no_material_drift_detected"]


def build_paper_trading_monitor_report() -> pd.DataFrame:
    state = _read_csv("paper_portfolio_state.csv")
    trades = _read_csv("paper_trades_log.csv")
    performance = _read_csv("paper_performance.csv")
    dashboard = _read_csv("research_dashboard_summary.csv")
    clean_eval = _read_csv("clean_research_evaluation.csv")
    larger = _read_csv("larger_walk_forward_summary.csv")
    forecast = _read_csv("forecast_history.csv")
    paper = _paper_metrics(performance, state, trades)
    expected = _expected_metrics(larger, clean_eval, dashboard, paper.get("paper_model_mode", "baseline"))
    flags = _drift_flags(paper, expected, state, forecast)
    promotion_status = str(_dashboard_value(dashboard, "promotion_status", "blocked"))
    governance_status = str(_dashboard_value(dashboard, "governance_status", "missing"))
    row = {
        **paper,
        **expected,
        "paper_sharpe_vs_expected": paper["paper_sharpe"] - expected["expected_sharpe"] if np.isfinite(paper["paper_sharpe"]) and np.isfinite(expected["expected_sharpe"]) else np.nan,
        "paper_drawdown_vs_expected": paper["paper_max_drawdown"] - expected["expected_max_drawdown"] if np.isfinite(paper["paper_max_drawdown"]) and np.isfinite(expected["expected_max_drawdown"]) else np.nan,
        "paper_return_vs_expected": paper["paper_cumulative_return"] - expected["expected_return"] if np.isfinite(paper["paper_cumulative_return"]) and np.isfinite(expected["expected_return"]) else np.nan,
        "paper_turnover_vs_expected": paper["paper_turnover"] - expected["expected_turnover"] if np.isfinite(paper["paper_turnover"]) and np.isfinite(expected["expected_turnover"]) else np.nan,
        "drift_flags": ", ".join(flags),
        "governance_status": governance_status,
        "promotion_status": promotion_status,
    }
    return pd.DataFrame([row])


def _format_report(report: pd.DataFrame) -> str:
    row = report.iloc[0] if not report.empty else pd.Series(dtype=object)
    lines = [
        "===== PAPER TRADING MONITOR =====",
        f"paper model mode: {row.get('paper_model_mode', 'missing')}",
        f"paper cumulative return: {_safe_float(row.get('paper_cumulative_return', np.nan)):.6f}",
        f"paper Sharpe: {_safe_float(row.get('paper_sharpe', np.nan)):.6f}",
        f"paper max drawdown: {_safe_float(row.get('paper_max_drawdown', np.nan)):.6f}",
        f"expected Sharpe: {_safe_float(row.get('expected_sharpe', np.nan)):.6f}",
        f"expected max drawdown: {_safe_float(row.get('expected_max_drawdown', np.nan)):.6f}",
        f"paper turnover vs expected: {_safe_float(row.get('paper_turnover_vs_expected', np.nan)):.6f}",
        f"paper cash: {_safe_float(row.get('paper_cash', np.nan)):.6f}",
        f"expected cash: {_safe_float(row.get('expected_cash', np.nan)):.6f}",
        f"drift flags: {row.get('drift_flags', 'missing')}",
        f"governance status: {row.get('governance_status', 'missing')}",
        f"promotion status: {row.get('promotion_status', 'blocked')}",
        f"current holdings: {row.get('current_holdings', '')}",
    ]
    if str(row.get("promotion_status", "blocked")).lower() == "blocked":
        lines.append("Paper trading continues. Production promotion remains blocked.")
    if row.get("paper_history_status") in {"missing", "too_short"}:
        lines.append(f"history note: {row.get('paper_history_status')}")
    return "\n".join(lines)


def run_paper_trading_monitor() -> pd.DataFrame:
    report = build_paper_trading_monitor_report()
    report.to_csv("paper_trading_monitor_report.csv", index=False)
    text = _format_report(report)
    Path("paper_trading_monitor_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {Path('paper_trading_monitor_report.csv').resolve()}")
    print(f"Saved: {Path('paper_trading_monitor_report.txt').resolve()}")
    return report


if __name__ == "__main__":
    run_paper_trading_monitor()
