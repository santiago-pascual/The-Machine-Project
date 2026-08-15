from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ABLATION_RESULTS = "expected_return_ablation_results.csv"
ABLATION_DAILY = "expected_return_ablation_daily_returns.csv"
ABLATION_TRADES = "expected_return_ablation_trades.csv"
HISTORICAL_PORTFOLIO = "historical_walk_forward_portfolio_returns.csv"
HISTORICAL_SNAPSHOTS = "historical_forecast_snapshots.csv"
HISTORICAL_LABELS = "historical_triple_barrier_labels.csv"

RESULTS_FILE = "raw_target_research_backtest_results.csv"
DAILY_FILE = "raw_target_research_backtest_daily_returns.csv"
TRADES_FILE = "raw_target_research_backtest_trades.csv"
GOVERNANCE_FILE = "raw_target_research_governance.csv"
TRADING_DAYS = 252


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _max_drawdown(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if len(returns) < 2:
        return np.nan
    vol = float(returns.std(ddof=0))
    if vol <= 0:
        return np.nan
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    downside = returns[returns < 0]
    if returns.empty or len(downside) < 2:
        return np.nan
    down_std = float(downside.std(ddof=0))
    if down_std <= 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (down_std * np.sqrt(TRADING_DAYS)))


def _calmar(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    dd = abs(_max_drawdown(returns))
    if returns.empty or not np.isfinite(dd) or dd <= 0:
        return np.nan
    annualized = (1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0
    return float(annualized / dd)


def _portfolio_mode_metrics(portfolio: pd.DataFrame, snapshots: pd.DataFrame, labels: pd.DataFrame, mode: str) -> dict[str, object]:
    daily = portfolio[portfolio["model_mode"].astype(str).eq(mode)].copy()
    trades = snapshots[
        snapshots["model_mode"].astype(str).eq(mode)
        & _bool(snapshots.get("selected", pd.Series(False, index=snapshots.index)))
    ].copy()
    returns = _num(daily.get("realized_portfolio_return_1d", pd.Series(dtype=float))).dropna()
    label_metrics = _label_metrics(labels, trades)
    return {
        "model_mode": mode,
        "realized_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
        "volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan,
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "average_cash": float(_num(daily.get("cash_weight", pd.Series(dtype=float))).mean()),
        "average_selected_count": float(_num(daily.get("selected_count", pd.Series(dtype=float))).mean()),
        "turnover": float(_num(daily.get("turnover", pd.Series(dtype=float))).mean()),
        "hit_rate": float((_num(trades.get("realized_return_20d", pd.Series(dtype=float))) > 0).mean()) if not trades.empty else np.nan,
        "direction_accuracy": float((returns > 0).mean()) if not returns.empty else np.nan,
        "sample_size": len(trades),
        **label_metrics,
    }


def _label_metrics(labels: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    if labels.empty or trades.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.normalize()
    labels20 = labels[labels["horizon"].astype(str).eq("20")] if "horizon" in labels else labels
    merged = trades[["date", "ticker"]].drop_duplicates().merge(labels20, on=["date", "ticker"], how="left")
    if merged.empty or "first_touch_type" not in merged:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean())
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _raw_target_metrics(ablation_results: pd.DataFrame) -> dict[str, object]:
    rows = ablation_results[ablation_results["variant"].astype(str).eq("raw_target_return_only")]
    if rows.empty:
        rows = ablation_results[ablation_results["variant"].astype(str).eq("no_signal_strength_adjustment")]
    if rows.empty:
        return {"model_mode": "raw_target_research"}
    row = rows.iloc[0]
    return {
        "model_mode": "raw_target_research",
        "realized_return": float(row.get("realized_return", np.nan)),
        "volatility": float(row.get("volatility", np.nan)),
        "Sharpe": float(row.get("Sharpe", np.nan)),
        "Sortino": float(row.get("Sortino", np.nan)),
        "Calmar": float(row.get("Calmar", np.nan)),
        "max_drawdown": float(row.get("max_drawdown", np.nan)),
        "average_cash": float(row.get("average_cash", np.nan)),
        "average_selected_count": float(row.get("selected_count", np.nan)),
        "turnover": float(row.get("turnover", np.nan)),
        "TP_rate": float(row.get("TP_rate", np.nan)),
        "SL_rate": float(row.get("SL_rate", np.nan)),
        "TP_minus_SL": float(row.get("TP_minus_SL", np.nan)),
        "hit_rate": float(row.get("hit_rate", np.nan)),
        "direction_accuracy": np.nan,
        "sample_size": int(row.get("sample_size", 0)),
    }


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    rows = results.set_index("model_mode")
    raw = rows.loc["raw_target_research"]
    baseline = rows.loc["baseline"]
    gated = rows.loc["regime_gated_full_quant"]
    best_reference_sharpe = max(float(baseline["Sharpe"]), float(gated["Sharpe"]))
    raw_sharpe = float(raw["Sharpe"])
    raw_dd = float(raw["max_drawdown"])
    reference_dd = min(float(baseline["max_drawdown"]), float(gated["max_drawdown"]))
    if raw_sharpe > best_reference_sharpe and raw_dd >= reference_dd * 1.75:
        classification = "eligible for paper testing"
        reason = "beats_references_with_acceptable_drawdown"
    elif raw_sharpe > best_reference_sharpe:
        classification = "research only"
        reason = "beats_references_but_drawdown_risk"
    elif float(raw["realized_return"]) > max(float(baseline["realized_return"]), float(gated["realized_return"])):
        classification = "research only"
        reason = "higher_return_but_sharpe_or_drawdown_not_superior"
    else:
        classification = "reject"
        reason = "does_not_beat_references"
    return pd.DataFrame(
        [
            {
                "candidate": "raw_target_research",
                "classification": classification,
                "reason": reason,
                "raw_target_sharpe": raw_sharpe,
                "baseline_sharpe": float(baseline["Sharpe"]),
                "regime_gated_sharpe": float(gated["Sharpe"]),
                "raw_target_max_drawdown": raw_dd,
                "production_change": "none",
            }
        ]
    )


def run_raw_target_research_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ablation_results = _read_csv(ABLATION_RESULTS)
    ablation_daily = _read_csv(ABLATION_DAILY)
    ablation_trades = _read_csv(ABLATION_TRADES)
    portfolio = _read_csv(HISTORICAL_PORTFOLIO)
    snapshots = _read_csv(HISTORICAL_SNAPSHOTS)
    labels = _read_csv(HISTORICAL_LABELS)
    if ablation_results.empty:
        raise ValueError("expected_return_ablation_results.csv is required.")
    for frame in [portfolio, snapshots, labels, ablation_daily, ablation_trades]:
        if not frame.empty and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()

    results = pd.DataFrame(
        [
            _portfolio_mode_metrics(portfolio, snapshots, labels, "baseline"),
            _portfolio_mode_metrics(portfolio, snapshots, labels, "regime_gated_full_quant"),
            _raw_target_metrics(ablation_results),
        ]
    )
    raw_daily = ablation_daily[ablation_daily["variant"].astype(str).eq("raw_target_return_only")].copy() if "variant" in ablation_daily else pd.DataFrame()
    raw_trades = ablation_trades[ablation_trades["variant"].astype(str).eq("raw_target_return_only")].copy() if "variant" in ablation_trades else pd.DataFrame()
    daily_out = pd.concat(
        [
            portfolio[portfolio["model_mode"].astype(str).isin(["baseline", "regime_gated_full_quant"])],
            raw_daily.assign(model_mode="raw_target_research"),
        ],
        ignore_index=True,
        sort=False,
    )
    trades_out = pd.concat(
        [
            snapshots[snapshots["model_mode"].astype(str).isin(["baseline", "regime_gated_full_quant"])],
            raw_trades.assign(model_mode="raw_target_research"),
        ],
        ignore_index=True,
        sort=False,
    )
    governance = _governance(results)

    results.to_csv(RESULTS_FILE, index=False)
    daily_out.to_csv(DAILY_FILE, index=False)
    trades_out.to_csv(TRADES_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== RAW TARGET RESEARCH BACKTEST =====")
    print("source: expected_return_ablation_results.csv + historical walk-forward references")
    print("\n===== BASELINE VS REGIME GATED VS RAW TARGET =====")
    cols = ["model_mode", "realized_return", "volatility", "Sharpe", "Sortino", "Calmar", "max_drawdown", "average_cash", "average_selected_count", "turnover", "TP_rate", "SL_rate", "TP_minus_SL", "hit_rate", "direction_accuracy", "sample_size"]
    print(results[cols].to_string(index=False))
    print("\n===== RAW TARGET GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_FILE).resolve()}")
    print(f"Saved: {Path(TRADES_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, daily_out, trades_out, governance


if __name__ == "__main__":
    run_raw_target_research_backtest()
