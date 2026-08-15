from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATE_FILE = "growth_candidate_paper_state.csv"
PERFORMANCE_FILE = "growth_candidate_paper_performance.csv"
TRADES_FILE = "growth_candidate_paper_trades.csv"
MONITOR_FILE = "growth_candidate_paper_monitor.csv"
BACKTEST_FILE = "growth_final_selection_results.csv"
COST_BACKTEST_FILE = "growth_final_cost_slippage_results.csv"

OUT_TRACKING = "growth_live_tracking.csv"
OUT_HEALTH = "growth_live_health.csv"
OUT_DRIFT = "growth_live_drift.csv"
OUT_GOVERNANCE = "growth_live_tracking_governance.csv"

MODEL_NAME = "growth_champion_final"
EXPECTED_WINDOW = "2020-01-01"
EXPECTED_COST_SCENARIO = "realistic_us_liquid"
MIN_OBSERVATIONS = 20
PROMOTION_MONTHS = 6


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out.dropna(subset=[col]).sort_values(col)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _periods_per_year(dates: pd.Series) -> float:
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return 252.0
    step = dates.diff().dt.days.dropna().median()
    return float(365.25 / step) if pd.notna(step) and step > 0 else 252.0


def _drawdown(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def _live_metrics(perf: pd.DataFrame) -> dict[str, object]:
    perf = _dates(perf)
    if perf.empty:
        return {
            "days_tracked": 0,
            "months_tracked": 0,
            "live_CAGR": np.nan,
            "live_Sharpe": np.nan,
            "live_Sortino": np.nan,
            "live_drawdown": np.nan,
            "live_max_drawdown": np.nan,
            "live_exposure": np.nan,
            "live_turnover": np.nan,
        }
    returns = _num(perf.get("daily_return", pd.Series(index=perf.index, dtype=float))).fillna(0.0)
    ppy = _periods_per_year(perf["date"])
    equity = (1.0 + returns).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = max((perf["date"].max() - perf["date"].min()).days / 365.25, len(returns) / ppy, 1e-9)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    vol = float(returns.std(ddof=0) * np.sqrt(ppy))
    sharpe = float((returns.mean() * ppy) / vol) if vol > 0 else np.nan
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(ppy) if (returns < 0).any() else np.nan
    sortino = float((returns.mean() * ppy) / downside) if pd.notna(downside) and downside > 0 else np.nan
    dd = _drawdown(returns)
    months = int(perf["date"].dt.to_period("M").nunique())
    return {
        "days_tracked": len(perf),
        "months_tracked": months,
        "live_total_return": total,
        "live_CAGR": cagr,
        "live_Sharpe": sharpe,
        "live_Sortino": sortino,
        "live_drawdown": float(dd.iloc[-1]) if not dd.empty else np.nan,
        "live_max_drawdown": float(dd.min()) if not dd.empty else np.nan,
        "live_exposure": float(_num(perf.get("exposure", pd.Series(index=perf.index, dtype=float))).dropna().iloc[-1]) if "exposure" in perf.columns and _num(perf["exposure"]).dropna().any() else np.nan,
        "live_average_exposure": float(_num(perf.get("exposure", pd.Series(index=perf.index, dtype=float))).mean()) if "exposure" in perf.columns else np.nan,
        "live_turnover": float(_num(perf.get("turnover", pd.Series(index=perf.index, dtype=float))).mean()) if "turnover" in perf.columns else np.nan,
    }


def _expected_metrics() -> dict[str, object]:
    cost = _read_csv(COST_BACKTEST_FILE)
    row = pd.DataFrame()
    if not cost.empty:
        row = cost[
            cost.get("candidate", "").astype(str).eq("growth_champion_v3")
            & cost.get("window_start", "").astype(str).eq(EXPECTED_WINDOW)
            & cost.get("cost_scenario", "").astype(str).eq(EXPECTED_COST_SCENARIO)
        ]
    if row.empty:
        bt = _read_csv(BACKTEST_FILE)
        if not bt.empty:
            row = bt[
                bt.get("candidate", "").astype(str).eq("growth_champion_v3")
                & bt.get("window_start", "").astype(str).eq(EXPECTED_WINDOW)
            ]
    if row.empty:
        return {
            "expected_source": "missing",
            "expected_CAGR": np.nan,
            "expected_Sharpe": np.nan,
            "expected_Sortino": np.nan,
            "expected_max_drawdown": np.nan,
            "expected_exposure": np.nan,
            "expected_turnover": np.nan,
        }
    r = row.iloc[0]
    return {
        "expected_source": f"{EXPECTED_WINDOW}_{EXPECTED_COST_SCENARIO}",
        "expected_CAGR": r.get("CAGR", np.nan),
        "expected_Sharpe": r.get("Sharpe", np.nan),
        "expected_Sortino": r.get("Sortino", np.nan),
        "expected_max_drawdown": r.get("max_drawdown", np.nan),
        "expected_exposure": r.get("average_exposure", np.nan),
        "expected_turnover": r.get("average_turnover", np.nan),
    }


def _latest_state() -> pd.DataFrame:
    state = _dates(_read_csv(STATE_FILE))
    if state.empty:
        return state
    return state[state["date"].eq(state["date"].max())].copy()


def _signal_quality(trades: pd.DataFrame, perf: pd.DataFrame, state: pd.DataFrame) -> dict[str, object]:
    trades = _dates(trades)
    if trades.empty:
        return {
            "buy_signals": 0,
            "hold_signals": 0,
            "sell_signals": 0,
            "soft_exit_retained": 0,
            "hit_rate": np.nan,
            "average_winner": np.nan,
            "average_loser": np.nan,
            "win_loss_ratio": np.nan,
        }
    actions = trades.get("action", pd.Series(index=trades.index, dtype=str)).astype(str).str.upper()
    perf = _dates(perf)
    returns = _num(perf.get("daily_return", pd.Series(dtype=float))).dropna() if not perf.empty else pd.Series(dtype=float)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    state_text = ",".join(state.get("soft_exit_status", pd.Series(dtype=str)).dropna().astype(str).tolist()) if not state.empty and "soft_exit_status" in state.columns else ""
    retained_from_reason = trades.get("reason", pd.Series(index=trades.index, dtype=str)).astype(str).str.contains("soft_exit|dual_trend|growth", case=False, na=False).sum()
    avg_winner = float(winners.mean()) if not winners.empty else np.nan
    avg_loser = float(losers.mean()) if not losers.empty else np.nan
    return {
        "buy_signals": int(actions.eq("BUY").sum() + actions.eq("ADD").sum()),
        "hold_signals": int(actions.eq("HOLD").sum()),
        "sell_signals": int(actions.eq("SELL").sum() + actions.eq("REDUCE").sum()),
        "soft_exit_retained": int(retained_from_reason),
        "hit_rate": float((returns > 0).mean()) if not returns.empty else np.nan,
        "average_winner": avg_winner,
        "average_loser": avg_loser,
        "win_loss_ratio": abs(avg_winner / avg_loser) if pd.notna(avg_winner) and pd.notna(avg_loser) and avg_loser != 0 else np.nan,
    }


def _drift_level(value: float, mild: float, significant: float, critical: float) -> str:
    if not np.isfinite(value):
        return "not_enough_data"
    abs_value = abs(value)
    if abs_value >= critical:
        return "critical drift"
    if abs_value >= significant:
        return "significant drift"
    if abs_value >= mild:
        return "mild drift"
    return "normal"


def _drift_metrics(live: dict[str, object], expected: dict[str, object], state: pd.DataFrame) -> pd.DataFrame:
    weights = pd.Series(dtype=float)
    if not state.empty and "paper_position_weight" in state.columns:
        non_cash = state[state.get("ticker", "").astype(str).ne("CASH")]
        weights = _num(non_cash["paper_position_weight"]).dropna()
    concentration = float(weights.max()) if not weights.empty else np.nan
    expected_exposure = float(expected.get("expected_exposure", np.nan))
    expected_turnover = float(expected.get("expected_turnover", np.nan))
    rows = [
        {
            "drift_type": "return_drift",
            "live_value": live.get("live_CAGR", np.nan),
            "expected_value": expected.get("expected_CAGR", np.nan),
            "difference": float(live.get("live_CAGR", np.nan)) - float(expected.get("expected_CAGR", np.nan)) if np.isfinite(float(live.get("live_CAGR", np.nan))) and np.isfinite(float(expected.get("expected_CAGR", np.nan))) else np.nan,
        },
        {
            "drift_type": "Sharpe_drift",
            "live_value": live.get("live_Sharpe", np.nan),
            "expected_value": expected.get("expected_Sharpe", np.nan),
            "difference": float(live.get("live_Sharpe", np.nan)) - float(expected.get("expected_Sharpe", np.nan)) if np.isfinite(float(live.get("live_Sharpe", np.nan))) and np.isfinite(float(expected.get("expected_Sharpe", np.nan))) else np.nan,
        },
        {
            "drift_type": "exposure_drift",
            "live_value": live.get("live_average_exposure", np.nan),
            "expected_value": expected_exposure,
            "difference": float(live.get("live_average_exposure", np.nan)) - expected_exposure if np.isfinite(float(live.get("live_average_exposure", np.nan))) and np.isfinite(expected_exposure) else np.nan,
        },
        {
            "drift_type": "turnover_drift",
            "live_value": live.get("live_turnover", np.nan),
            "expected_value": expected_turnover,
            "difference": float(live.get("live_turnover", np.nan)) - expected_turnover if np.isfinite(float(live.get("live_turnover", np.nan))) and np.isfinite(expected_turnover) else np.nan,
        },
        {
            "drift_type": "concentration_drift",
            "live_value": concentration,
            "expected_value": 0.25,
            "difference": concentration - 0.25 if np.isfinite(concentration) else np.nan,
        },
    ]
    out = pd.DataFrame(rows)
    out["drift_classification"] = [
        _drift_level(out.iloc[0]["difference"], 0.10, 0.20, 0.35),
        _drift_level(out.iloc[1]["difference"], 0.50, 1.00, 2.00),
        _drift_level(out.iloc[2]["difference"], 0.10, 0.20, 0.35),
        _drift_level(out.iloc[3]["difference"], 0.05, 0.10, 0.20),
        _drift_level(out.iloc[4]["difference"], 0.10, 0.20, 0.35),
    ]
    return out


def _health(live: dict[str, object], expected: dict[str, object], drift: pd.DataFrame, signal: dict[str, object]) -> pd.DataFrame:
    days = int(live.get("days_tracked", 0))
    months = int(live.get("months_tracked", 0))
    if days < MIN_OBSERVATIONS:
        status = "WARMUP"
        governance = "tracking_warmup"
        reason = f"Only {days} observations; require at least {MIN_OBSERVATIONS} for drift/health evaluation."
    else:
        classes = drift["drift_classification"].astype(str).tolist()
        if "critical drift" in classes:
            status = "FAILED"
            governance = "investigation_required"
            reason = "Critical drift detected."
        elif "significant drift" in classes:
            status = "DEGRADED"
            governance = "drift_detected"
            reason = "Significant drift detected."
        elif "mild drift" in classes:
            status = "WATCHLIST"
            governance = "drift_detected"
            reason = "Mild drift detected."
        else:
            status = "HEALTHY"
            governance = "healthy_tracking"
            reason = "Live tracking is within expected bands."
    drift_score = 0
    weights = {"normal": 0, "mild drift": 25, "significant drift": 60, "critical drift": 100, "not_enough_data": 0}
    if days >= MIN_OBSERVATIONS and not drift.empty:
        drift_score = int(max(weights.get(x, 0) for x in drift["drift_classification"].astype(str)))
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
                "model": MODEL_NAME,
                "status": status,
                "governance_classification": governance,
                "days_tracked": days,
                "months_tracked": months,
                "drift_score": drift_score,
                "expected_CAGR": expected.get("expected_CAGR", np.nan),
                "realized_CAGR": live.get("live_CAGR", np.nan),
                "expected_Sharpe": expected.get("expected_Sharpe", np.nan),
                "realized_Sharpe": live.get("live_Sharpe", np.nan),
                "expected_max_drawdown": expected.get("expected_max_drawdown", np.nan),
                "realized_max_drawdown": live.get("live_max_drawdown", np.nan),
                "warnings": reason,
                "promotion_status": "blocked_warmup" if months < PROMOTION_MONTHS else "eligible_for_review_if_healthy",
                "reason": reason,
            }
        ]
    )


def run_live_tracking_monitor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf = _dates(_read_csv(PERFORMANCE_FILE))
    state = _latest_state()
    trades = _dates(_read_csv(TRADES_FILE))
    monitor = _dates(_read_csv(MONITOR_FILE))
    expected = _expected_metrics()
    live = _live_metrics(perf)
    signal = _signal_quality(trades, perf, state)
    drift = _drift_metrics(live, expected, state)
    health = _health(live, expected, drift, signal)

    latest_monitor = monitor.iloc[-1].to_dict() if not monitor.empty else {}
    tracking = pd.DataFrame([{**live, **expected, **signal, **latest_monitor}])
    governance = health.copy()
    governance["production_changed"] = False
    governance["paper_changed"] = False
    governance["parameter_tuning"] = False

    tracking.to_csv(OUT_TRACKING, index=False)
    health.to_csv(OUT_HEALTH, index=False)
    drift.to_csv(OUT_DRIFT, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)
    _print_report(tracking, health, drift, governance)
    return tracking, health, drift


def _pct(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def _fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _print_report(tracking: pd.DataFrame, health: pd.DataFrame, drift: pd.DataFrame, governance: pd.DataFrame) -> None:
    print("\n===== GROWTH LIVE TRACKING MONITOR =====")
    if tracking.empty:
        print("No tracking data.")
        return
    row = tracking.iloc[0]
    h = health.iloc[0]
    print(f"model: {MODEL_NAME}")
    print(f"days tracked: {int(row.get('days_tracked', 0))}")
    print(f"status: {h.get('status')}")
    print(f"drift score: {h.get('drift_score')}")
    print(f"expected CAGR: {_pct(row.get('expected_CAGR'))}")
    print(f"realized CAGR: {_pct(row.get('live_CAGR'))}")
    print(f"expected Sharpe: {_fmt(row.get('expected_Sharpe'))}")
    print(f"realized Sharpe: {_fmt(row.get('live_Sharpe'))}")
    print(f"warnings: {h.get('warnings')}")

    print("\n===== LIVE VS BACKTEST =====")
    cols = [
        "live_CAGR",
        "expected_CAGR",
        "live_Sharpe",
        "expected_Sharpe",
        "live_Sortino",
        "expected_Sortino",
        "live_max_drawdown",
        "expected_max_drawdown",
        "live_average_exposure",
        "expected_exposure",
        "live_turnover",
        "expected_turnover",
    ]
    print(tracking[[c for c in cols if c in tracking.columns]].to_string(index=False))

    print("\n===== DRIFT DETECTION =====")
    print(drift.to_string(index=False))

    print("\n===== SIGNAL QUALITY =====")
    sig_cols = ["buy_signals", "hold_signals", "sell_signals", "soft_exit_retained", "hit_rate", "average_winner", "average_loser", "win_loss_ratio"]
    print(tracking[[c for c in sig_cols if c in tracking.columns]].to_string(index=False))

    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))


if __name__ == "__main__":
    run_live_tracking_monitor()
