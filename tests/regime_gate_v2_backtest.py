from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REGIME_V2_DAILY_FILE = "regime_v2_daily_state.csv"
REGIME_V2_PERFORMANCE_FILE = "regime_v2_performance_attribution.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
TRIPLE_BARRIER_FILE = "historical_triple_barrier_labels.csv"
PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"

RESULTS_FILE = "regime_gate_v2_backtest_results.csv"
DAILY_RETURNS_FILE = "regime_gate_v2_daily_returns.csv"
TRADES_FILE = "regime_gate_v2_trades.csv"
GOVERNANCE_FILE = "regime_gate_v2_governance.csv"
TRADING_DAYS = 252

ALLOW_FULL_QUANT_REGIMES = {"trend_up_low_vol", "trend_up_high_vol", "sideways_high_vol"}
BLOCK_FULL_QUANT_REGIMES = {"sideways_low_vol", "stress", "choppy"}
MIN_REGIME_SAMPLE_SIZE = 30
MIN_REGIME_CONFIDENCE = 0.25


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
    downside_std = float(downside.std(ddof=0))
    if downside_std <= 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (downside_std * np.sqrt(TRADING_DAYS)))


def _calmar(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    dd = abs(_max_drawdown(returns))
    if returns.empty or not np.isfinite(dd) or dd <= 0:
        return np.nan
    ann = (1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0
    return float(ann / dd)


def _prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regime_v2 = _read_csv(REGIME_V2_DAILY_FILE)
    regime_perf = _read_csv(REGIME_V2_PERFORMANCE_FILE)
    snapshots = _read_csv(SNAPSHOTS_FILE)
    realized = _read_csv(REALIZED_FILE)
    labels = _read_csv(TRIPLE_BARRIER_FILE)
    portfolio = _read_csv(PORTFOLIO_FILE)
    for frame in [regime_v2, snapshots, realized, labels, portfolio]:
        if not frame.empty and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if not snapshots.empty and not realized.empty:
        realized_cols = [
            "realized_return_1d",
            "realized_return_5d",
            "realized_return_10d",
            "realized_return_20d",
            "realized_return_30d",
        ]
        available = [col for col in realized_cols if col in realized.columns]
        snapshots = snapshots.drop(columns=[col for col in available if col in snapshots.columns], errors="ignore")
        snapshots = snapshots.merge(
            realized[["date", "ticker", "model_mode"] + available],
            on=["date", "ticker", "model_mode"],
            how="left",
        )
    return regime_v2, regime_perf, snapshots, labels, portfolio


def _regime_sample_map(regime_perf: pd.DataFrame) -> dict[str, int]:
    if regime_perf.empty or not {"regime", "sample_size"}.issubset(regime_perf.columns):
        return {}
    v2 = regime_perf[regime_perf.get("regime_system", "").astype(str).eq("v2")] if "regime_system" in regime_perf else regime_perf
    return {str(row["regime"]): int(row["sample_size"]) for _, row in v2.iterrows()}


def _build_v2_decisions(regime_v2: pd.DataFrame, regime_perf: pd.DataFrame) -> pd.DataFrame:
    sample_map = _regime_sample_map(regime_perf)
    decisions: list[dict[str, object]] = []
    for _, row in regime_v2.iterrows():
        label = str(row.get("regime_v2_label", "unknown"))
        confidence = float(row.get("regime_v2_confidence", np.nan)) if pd.notna(row.get("regime_v2_confidence", np.nan)) else 0.0
        sample_size = int(sample_map.get(label, 0))
        if confidence < MIN_REGIME_CONFIDENCE:
            source_mode = "baseline"
            decision = "fallback_baseline"
            reason = "low_regime_v2_confidence"
        elif sample_size < MIN_REGIME_SAMPLE_SIZE:
            source_mode = "baseline"
            decision = "fallback_baseline"
            reason = "low_regime_sample_size"
        elif label in ALLOW_FULL_QUANT_REGIMES:
            source_mode = "regime_gated_full_quant"
            decision = "allow_full_quant"
            reason = f"strong_v2_regime:{label}"
        elif label in BLOCK_FULL_QUANT_REGIMES:
            source_mode = "baseline"
            decision = "fallback_baseline"
            reason = f"blocked_v2_regime:{label}"
        else:
            source_mode = "baseline"
            decision = "fallback_baseline"
            reason = f"unapproved_v2_regime:{label}"
        decisions.append(
            {
                "date": row["date"],
                "regime_v2_label": label,
                "regime_v2_confidence": confidence,
                "regime_v2_sample_size": sample_size,
                "v2_gate_decision": decision,
                "v2_gate_reason": reason,
                "source_model_mode": source_mode,
            }
        )
    return pd.DataFrame(decisions)


def _build_candidate_daily(portfolio: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty or decisions.empty:
        return pd.DataFrame()
    merged = decisions.merge(
        portfolio,
        left_on=["date", "source_model_mode"],
        right_on=["date", "model_mode"],
        how="left",
    )
    merged["model_mode"] = "regime_gate_v2"
    return merged


def _build_candidate_trades(snapshots: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or decisions.empty:
        return pd.DataFrame()
    selected = snapshots[_bool(snapshots.get("selected", pd.Series(False, index=snapshots.index)))].copy()
    trades = decisions.merge(
        selected,
        left_on=["date", "source_model_mode"],
        right_on=["date", "model_mode"],
        how="left",
        suffixes=("", "_snapshot"),
    )
    trades["candidate_model_mode"] = "regime_gate_v2"
    return trades


def _daily_for_mode(portfolio: pd.DataFrame, mode: str) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame()
    out = portfolio[portfolio["model_mode"].astype(str).eq(mode)].copy()
    out["source_model_mode"] = mode
    return out


def _trades_for_mode(snapshots: pd.DataFrame, mode: str) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    data = snapshots[
        snapshots["model_mode"].astype(str).eq(mode) & _bool(snapshots.get("selected", pd.Series(False, index=snapshots.index)))
    ].copy()
    data["candidate_model_mode"] = mode
    return data


def _label_metrics(labels: pd.DataFrame, trades: pd.DataFrame, mode: str) -> dict[str, float]:
    if labels.empty or trades.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    keys = ["date", "ticker"]
    label20 = labels[labels["horizon"].astype(str).eq("20")].copy() if "horizon" in labels else labels.copy()
    merged = trades[keys].dropna().drop_duplicates().merge(label20, on=keys, how="left")
    if merged.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean()) if "first_touch_type" in merged else np.nan
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean()) if "first_touch_type" in merged else np.nan
    if "realized_return_at_barrier" in merged:
        hit = float((_num(merged["realized_return_at_barrier"]) > 0).mean())
    elif "realized_return_20d" in trades:
        hit = float((_num(trades["realized_return_20d"]) > 0).mean())
    else:
        hit = np.nan
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan, "hit_rate": hit}


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame, labels: pd.DataFrame, mode: str) -> dict[str, object]:
    return_col = "realized_portfolio_return_1d"
    returns = _num(daily.get(return_col, pd.Series(dtype=float))).dropna()
    if returns.empty:
        base = {
            "model_mode": mode,
            "realized_return": np.nan,
            "volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
            "average_cash": np.nan,
            "average_selected_count": np.nan,
            "turnover": np.nan,
            "direction_accuracy": np.nan,
            "sample_size": 0,
        }
    else:
        total_return = float((1.0 + returns).prod() - 1.0)
        vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan
        base = {
            "model_mode": mode,
            "realized_return": total_return,
            "volatility": vol,
            "Sharpe": _sharpe(returns),
            "Sortino": _sortino(returns),
            "Calmar": _calmar(returns),
            "max_drawdown": _max_drawdown(returns),
            "average_cash": float(_num(daily.get("cash_weight", pd.Series(dtype=float))).mean()),
            "average_selected_count": float(_num(daily.get("selected_count", pd.Series(dtype=float))).mean()),
            "turnover": float(_num(daily.get("turnover", pd.Series(dtype=float))).mean()),
            "direction_accuracy": float((returns > 0).mean()),
            "sample_size": len(trades),
        }
    base.update(_label_metrics(labels, trades, mode))
    return base


def _performance_by_regime(candidate_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if candidate_daily.empty or "regime_v2_label" not in candidate_daily.columns:
        return pd.DataFrame()
    for label, group in candidate_daily.groupby("regime_v2_label"):
        returns = _num(group["realized_portfolio_return_1d"])
        rows.append(
            {
                "regime_v2_label": label,
                "dates": len(group),
                "source_full_quant_share": float(group["source_model_mode"].astype(str).eq("regime_gated_full_quant").mean()),
                "realized_return": float((1.0 + returns.dropna()).prod() - 1.0) if returns.notna().any() else np.nan,
                "Sharpe": _sharpe(returns),
                "max_drawdown": _max_drawdown(returns),
                "average_cash": float(_num(group["cash_weight"]).mean()) if "cash_weight" in group else np.nan,
                "turnover": float(_num(group["turnover"]).mean()) if "turnover" in group else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("dates", ascending=False)


def _governance(results: pd.DataFrame, perf_by_regime: pd.DataFrame) -> pd.DataFrame:
    row = results.set_index("model_mode")
    candidate = row.loc["regime_gate_v2"] if "regime_gate_v2" in row.index else pd.Series(dtype=float)
    baseline = row.loc["baseline"] if "baseline" in row.index else pd.Series(dtype=float)
    old = row.loc["regime_gated_full_quant"] if "regime_gated_full_quant" in row.index else pd.Series(dtype=float)
    candidate_sharpe = float(candidate.get("Sharpe", np.nan))
    best_old_sharpe = max(float(baseline.get("Sharpe", -999.0)), float(old.get("Sharpe", -999.0)))
    candidate_dd = float(candidate.get("max_drawdown", np.nan))
    old_dd_floor = min(float(baseline.get("max_drawdown", 0.0)), float(old.get("max_drawdown", 0.0)))
    tiny_driver = False
    if not perf_by_regime.empty:
        total_dates = max(1, int(perf_by_regime["dates"].sum()))
        top_return = perf_by_regime.sort_values("realized_return", ascending=False).iloc[0]
        tiny_driver = int(top_return["dates"]) / total_dates < 0.10
    if not np.isfinite(candidate_sharpe) or candidate_sharpe <= best_old_sharpe:
        classification = "rejected"
        reason = "does_not_beat_baseline_or_old_gate_sharpe"
    elif candidate_dd < old_dd_floor * 1.25:
        classification = "research only"
        reason = "sharpe_improves_but_drawdown_expands"
    elif tiny_driver:
        classification = "research only"
        reason = "improvement_driven_by_tiny_regime"
    else:
        classification = "eligible for paper testing"
        reason = "improves_sharpe_without_material_drawdown_penalty"
    return pd.DataFrame(
        [
            {
                "candidate": "regime_gate_v2",
                "classification": classification,
                "reason": reason,
                "candidate_sharpe": candidate_sharpe,
                "baseline_sharpe": float(baseline.get("Sharpe", np.nan)),
                "old_gate_sharpe": float(old.get("Sharpe", np.nan)),
                "candidate_max_drawdown": candidate_dd,
                "production_change": "none",
            }
        ]
    )


def run_regime_gate_v2_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regime_v2, regime_perf, snapshots, labels, portfolio = _prepare_inputs()
    if regime_v2.empty or portfolio.empty:
        raise ValueError("regime_v2_daily_state.csv and historical_walk_forward_portfolio_returns.csv are required.")
    decisions = _build_v2_decisions(regime_v2, regime_perf)
    candidate_daily = _build_candidate_daily(portfolio, decisions)
    candidate_trades = _build_candidate_trades(snapshots, decisions)

    baseline_daily = _daily_for_mode(portfolio, "baseline")
    old_daily = _daily_for_mode(portfolio, "regime_gated_full_quant")
    baseline_trades = _trades_for_mode(snapshots, "baseline")
    old_trades = _trades_for_mode(snapshots, "regime_gated_full_quant")

    results = pd.DataFrame(
        [
            _metrics(baseline_daily, baseline_trades, labels, "baseline"),
            _metrics(old_daily, old_trades, labels, "regime_gated_full_quant"),
            _metrics(candidate_daily, candidate_trades, labels, "regime_gate_v2"),
        ]
    )
    perf_by_regime = _performance_by_regime(candidate_daily)
    governance = _governance(results, perf_by_regime)

    daily_out = pd.concat(
        [
            baseline_daily.assign(backtest_model="baseline"),
            old_daily.assign(backtest_model="regime_gated_full_quant"),
            candidate_daily.assign(backtest_model="regime_gate_v2"),
        ],
        ignore_index=True,
        sort=False,
    )
    trades_out = pd.concat(
        [
            baseline_trades.assign(backtest_model="baseline"),
            old_trades.assign(backtest_model="regime_gated_full_quant"),
            candidate_trades.assign(backtest_model="regime_gate_v2"),
        ],
        ignore_index=True,
        sort=False,
    )

    results.to_csv(RESULTS_FILE, index=False)
    daily_out.to_csv(DAILY_RETURNS_FILE, index=False)
    trades_out.to_csv(TRADES_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== REGIME GATE V2 BACKTEST =====")
    print(f"dates tested: {candidate_daily['date'].nunique() if not candidate_daily.empty else 0}")
    print(
        f"full quant allowed dates: {int(candidate_daily['source_model_mode'].astype(str).eq('regime_gated_full_quant').sum()) if not candidate_daily.empty else 0}"
    )
    print(
        f"baseline fallback dates: {int(candidate_daily['source_model_mode'].astype(str).eq('baseline').sum()) if not candidate_daily.empty else 0}"
    )

    print("\n===== OLD GATE VS V2 GATE =====")
    cols = [
        "model_mode",
        "realized_return",
        "volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_cash",
        "average_selected_count",
        "turnover",
        "TP_rate",
        "SL_rate",
        "TP_minus_SL",
        "hit_rate",
        "direction_accuracy",
        "sample_size",
    ]
    print(results[cols].to_string(index=False))

    print("\n===== REGIME GATE V2 PERFORMANCE BY REGIME =====")
    print(perf_by_regime.to_string(index=False) if not perf_by_regime.empty else "insufficient data")

    print("\n===== REGIME GATE V2 GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_RETURNS_FILE).resolve()}")
    print(f"Saved: {Path(TRADES_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, daily_out, trades_out, governance


if __name__ == "__main__":
    run_regime_gate_v2_backtest()
