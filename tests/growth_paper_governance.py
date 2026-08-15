from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PERFORMANCE_FILE = "growth_candidate_paper_performance.csv"
STATE_FILE = "growth_candidate_paper_state.csv"
TRADES_FILE = "growth_candidate_paper_trades.csv"
CONFIG_FILE = "growth_candidate_paper_config.json"

REPORT_FILE = "growth_paper_governance_report.csv"
HISTORY_FILE = "growth_paper_governance_history.csv"
MONTHLY_FILE = "growth_paper_monthly_report.csv"

EXPECTED_BENCHMARK_FILE = "raw_target_2020_vs_benchmark.csv"
MIN_GOVERNANCE_ROLLING_WINDOW = 20


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _num(value) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(pd.Series(value), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _date_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date")


def _metric_returns(perf: pd.DataFrame) -> pd.Series:
    if perf.empty:
        return pd.Series(dtype=float)
    if "daily_return" in perf.columns:
        returns = _num(perf["daily_return"]).fillna(0.0)
    elif "return" in perf.columns:
        returns = _num(perf["return"]).fillna(0.0)
    else:
        returns = pd.Series(np.zeros(len(perf)), index=perf.index)
    returns.index = pd.to_datetime(perf["date"], errors="coerce")
    return returns.dropna()


def _drawdown_from_returns(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    if returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return equity, drawdown


def _rolling_sharpe(returns: pd.Series, window: int = 20) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    rolling_mean = returns.rolling(window, min_periods=2).mean()
    rolling_std = returns.rolling(window, min_periods=2).std(ddof=0)
    return (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(252)


def _rolling_sortino(returns: pd.Series, window: int = 20) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    rolling_mean = returns.rolling(window, min_periods=2).mean()

    def downside_std(x: pd.Series) -> float:
        downside = x[x < 0.0]
        if downside.empty:
            return 0.0
        return float(downside.std(ddof=0))

    downside = returns.rolling(window, min_periods=2).apply(downside_std, raw=False)
    return (rolling_mean / downside.replace(0.0, np.nan)) * np.sqrt(252)


def _var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    clean = returns.dropna()
    if clean.empty:
        return np.nan, np.nan
    q = float(clean.quantile(1.0 - alpha))
    tail = clean[clean <= q]
    cvar = float(tail.mean()) if not tail.empty else q
    return q, cvar


def _underwater_duration(drawdown: pd.Series) -> int:
    if drawdown.empty:
        return 0
    current = 0
    max_duration = 0
    for value in drawdown.fillna(0.0):
        if value < 0.0:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return int(max_duration)


def _benchmark_expectations() -> dict[str, float]:
    expected = _read_csv(EXPECTED_BENCHMARK_FILE)
    if expected.empty or "model" not in expected.columns:
        return {}
    out: dict[str, float] = {}
    for benchmark in ["SPY_buy_hold", "QQQ_buy_hold"]:
        row = expected[expected["model"].astype(str).eq(benchmark)]
        if row.empty:
            continue
        prefix = "SPY" if benchmark.startswith("SPY") else "QQQ"
        for col in ["total_return", "CAGR", "Sharpe", "max_drawdown"]:
            if col in row.columns:
                out[f"expected_{prefix}_{col}"] = float(_num(row.iloc[0][col]).iloc[0])
    return out


def _paper_month_span(perf: pd.DataFrame) -> int:
    if perf.empty:
        return 0
    dates = pd.to_datetime(perf["date"], errors="coerce").dropna()
    if dates.empty:
        return 0
    months = dates.dt.to_period("M").nunique()
    return int(months)


def _status_from_rules(
    rolling_sharpe: float,
    current_drawdown: float,
    paper_days: int,
    paper_months: int,
    excess_vs_spy: float | None,
    benchmark_available: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if paper_days < MIN_GOVERNANCE_ROLLING_WINDOW:
        reasons.append("insufficient paper history for rolling Sharpe / governance evaluation")
        reasons.append(f"paper observations {paper_days} < required rolling window {MIN_GOVERNANCE_ROLLING_WINDOW}")
        return "WARMUP", reasons

    if not np.isfinite(rolling_sharpe):
        reasons.append("insufficient paper history for rolling Sharpe / governance evaluation")
        return "WARMUP", reasons

    if current_drawdown < -0.30:
        reasons.append("drawdown > 30%")
        return "FAILED", reasons
    if rolling_sharpe < 0.0:
        reasons.append("rolling Sharpe < 0")
        return "FAILED", reasons

    if current_drawdown < -0.20:
        reasons.append("drawdown > 20%")
        return "DEGRADED", reasons
    if rolling_sharpe < 0.5:
        reasons.append("rolling Sharpe < 0.5")
        return "DEGRADED", reasons

    if rolling_sharpe < 1.0:
        reasons.append("rolling Sharpe between 0.5 and 1")
        return "WATCHLIST", reasons

    if benchmark_available and excess_vs_spy is not None and np.isfinite(excess_vs_spy) and excess_vs_spy < 0 and paper_months >= 1:
        reasons.append("underperforming SPY over available paper window")
        return "WATCHLIST", reasons

    if current_drawdown < -0.15:
        reasons.append("drawdown between 15% and 20%")
        return "WATCHLIST", reasons

    reasons.append("rolling Sharpe > 1 and drawdown < 15%")
    if not benchmark_available:
        reasons.append("benchmark return series unavailable locally; SPY/QQQ relative rule not fully evaluated")
    return "HEALTHY", reasons


def _recommendation(
    status: str,
    paper_months: int,
    excess_vs_spy: float | None,
    excess_vs_qqq: float | None,
    rolling_sharpe: float,
) -> tuple[str, str]:
    promotion_ready = (
        paper_months >= 6
        and np.isfinite(rolling_sharpe)
        and rolling_sharpe > 0.0
        and excess_vs_spy is not None
        and excess_vs_qqq is not None
        and np.isfinite(excess_vs_spy)
        and np.isfinite(excess_vs_qqq)
        and excess_vs_spy > 0.0
        and excess_vs_qqq > 0.0
        and status not in {"FAILED", "WARMUP"}
    )
    if promotion_ready:
        return "candidate ready for live consideration", "promotion_gate_passed"
    if status == "WARMUP":
        return "continue paper", "promotion_blocked_insufficient_history"
    if status == "FAILED":
        return "suspend candidate", "promotion_blocked_failed_state"
    if status in {"DEGRADED", "WATCHLIST"}:
        return "review model", "promotion_blocked_watch_or_degraded"
    return "continue paper", "promotion_blocked_insufficient_paper_history"


def _monthly_report(perf: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    if perf.empty or returns.empty:
        return pd.DataFrame(
            [
                {
                    "month": "missing",
                    "monthly_return": np.nan,
                    "average_exposure": np.nan,
                    "average_cash": np.nan,
                    "average_turnover": np.nan,
                    "observations": 0,
                }
            ]
        )
    working = perf.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["return"] = returns.reindex(working["date"]).to_numpy()
    working["month"] = working["date"].dt.to_period("M").astype(str)
    for col in ["exposure", "cash_weight", "turnover"]:
        if col in working.columns:
            working[col] = _num(working[col])
        else:
            working[col] = np.nan
    rows = []
    for month, group in working.groupby("month", dropna=True):
        month_return = float((1.0 + group["return"].fillna(0.0)).prod() - 1.0)
        rows.append(
            {
                "month": month,
                "monthly_return": month_return,
                "average_exposure": float(group["exposure"].mean()) if group["exposure"].notna().any() else np.nan,
                "average_cash": float(group["cash_weight"].mean()) if group["cash_weight"].notna().any() else np.nan,
                "average_turnover": float(group["turnover"].mean()) if group["turnover"].notna().any() else np.nan,
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows)


def _append_history(row: dict[str, object]) -> None:
    existing = _read_csv(HISTORY_FILE)
    output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) if not existing.empty else pd.DataFrame([row])
    output.to_csv(HISTORY_FILE, index=False)


def run_growth_paper_governance() -> pd.DataFrame:
    perf = _date_frame(_read_csv(PERFORMANCE_FILE))
    state = _date_frame(_read_csv(STATE_FILE))
    trades = _date_frame(_read_csv(TRADES_FILE))

    now = pd.Timestamp.now()
    if perf.empty:
        report = pd.DataFrame(
            [
                {
                    "timestamp": now.isoformat(),
                    "current_status": "DEGRADED",
                    "recommendation": "review model",
                    "promotion_status": "promotion_blocked_missing_paper_performance",
                    "reason": "growth_candidate_paper_performance.csv missing or empty",
                }
            ]
        )
        report.to_csv(REPORT_FILE, index=False)
        _append_history(report.iloc[0].to_dict())
        pd.DataFrame().to_csv(MONTHLY_FILE, index=False)
        print("\n===== GROWTH PAPER GOVERNANCE =====")
        print("current status: DEGRADED")
        print("recommendation: review model")
        print("reason: missing growth paper performance file")
        return report

    returns = _metric_returns(perf)
    equity, drawdown = _drawdown_from_returns(returns)
    rolling_20d_return = returns.rolling(20, min_periods=1).apply(lambda x: (1.0 + x).prod() - 1.0, raw=False)
    rolling_60d_return = returns.rolling(60, min_periods=1).apply(lambda x: (1.0 + x).prod() - 1.0, raw=False)
    rolling_sharpe = _rolling_sharpe(returns, 20)
    rolling_sortino = _rolling_sortino(returns, 20)
    var_95, cvar_95 = _var_cvar(returns, 0.95)

    latest_date = pd.to_datetime(perf["date"], errors="coerce").max()
    latest_perf = perf[perf["date"].eq(latest_date)].iloc[-1]
    latest_state = state[state["date"].eq(latest_date)].copy() if not state.empty else pd.DataFrame()
    latest_trades = trades[trades["date"].eq(latest_date)].copy() if not trades.empty else pd.DataFrame()

    cumulative_return = float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan
    latest_rolling_20 = float(rolling_20d_return.dropna().iloc[-1]) if not rolling_20d_return.dropna().empty else np.nan
    latest_rolling_60 = float(rolling_60d_return.dropna().iloc[-1]) if not rolling_60d_return.dropna().empty else np.nan
    latest_sharpe = float(rolling_sharpe.dropna().iloc[-1]) if not rolling_sharpe.dropna().empty else np.nan
    latest_sortino = float(rolling_sortino.dropna().iloc[-1]) if not rolling_sortino.dropna().empty else np.nan
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan
    current_drawdown = float(drawdown.iloc[-1]) if not drawdown.empty else np.nan

    exposure = float(_num(latest_perf.get("exposure", np.nan)).iloc[0])
    cash = float(_num(latest_perf.get("cash_weight", np.nan)).iloc[0])
    turnover = float(_num(latest_perf.get("turnover", np.nan)).iloc[0])
    model_mode = str(latest_perf.get("model_mode", "growth_champion_v2"))
    variant = str(latest_perf.get("growth_paper_variant", "growth_v1_exposure_cap_60"))

    paper_months = _paper_month_span(perf)
    paper_days = len(perf)
    active_holdings = (
        latest_state[latest_state["ticker"].astype(str).ne("CASH")]["ticker"].astype(str).tolist()
        if not latest_state.empty and "ticker" in latest_state.columns
        else []
    )

    # Live paper benchmark return series is intentionally not fabricated. If a
    # daily SPY/QQQ paper-aligned series is added later, this section can use it.
    benchmark_available = False
    excess_vs_spy = np.nan
    excess_vs_qqq = np.nan
    information_ratio_spy = np.nan
    information_ratio_qqq = np.nan
    benchmark_reason = "paper-aligned SPY/QQQ daily returns unavailable locally"
    expected_benchmarks = _benchmark_expectations()

    status, reasons = _status_from_rules(
        latest_sharpe,
        current_drawdown,
        paper_days,
        paper_months,
        None if not np.isfinite(excess_vs_spy) else excess_vs_spy,
        benchmark_available,
    )
    if paper_months < 6:
        reasons.append("promotion gate requires at least 6 months of paper history")
    recommendation, promotion_status = _recommendation(
        status,
        paper_months,
        None if not np.isfinite(excess_vs_spy) else excess_vs_spy,
        None if not np.isfinite(excess_vs_qqq) else excess_vs_qqq,
        latest_sharpe,
    )

    if status == "HEALTHY" and paper_months < 6:
        recommendation = "continue paper"
        promotion_status = "promotion_blocked_insufficient_paper_history"

    report_row = {
        "timestamp": now.isoformat(),
        "date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "",
        "model_mode": model_mode,
        "growth_paper_variant": variant,
        "current_status": status,
        "recommendation": recommendation,
        "promotion_status": promotion_status,
        "reason": " | ".join(reasons),
        "paper_days": paper_days,
        "paper_months": paper_months,
        "cumulative_return": cumulative_return,
        "rolling_20d_return": latest_rolling_20,
        "rolling_60d_return": latest_rolling_60,
        "rolling_sharpe": latest_sharpe,
        "rolling_sortino": latest_sortino,
        "drawdown": current_drawdown,
        "max_drawdown": max_drawdown,
        "underwater_duration": _underwater_duration(drawdown),
        "exposure": exposure,
        "cash": cash,
        "turnover": turnover,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "SPY_excess_return": excess_vs_spy,
        "QQQ_excess_return": excess_vs_qqq,
        "SPY_information_ratio": information_ratio_spy,
        "QQQ_information_ratio": information_ratio_qqq,
        "benchmark_status": "available" if benchmark_available else "missing",
        "benchmark_reason": benchmark_reason,
        "holdings": ",".join(active_holdings),
        "trades_today": len(latest_trades) if not latest_trades.empty else 0,
        **expected_benchmarks,
    }

    report = pd.DataFrame([report_row])
    report.to_csv(REPORT_FILE, index=False)
    _append_history(report_row)
    _monthly_report(perf, returns).to_csv(MONTHLY_FILE, index=False)

    print("\n===== GROWTH PAPER GOVERNANCE =====")
    print(f"current status: {status}")
    print(f"benchmark comparison: {benchmark_reason}")
    print(f"rolling Sharpe: {latest_sharpe:.4f}" if np.isfinite(latest_sharpe) else "rolling Sharpe: unavailable")
    print(f"rolling 20d return: {latest_rolling_20:.4%}" if np.isfinite(latest_rolling_20) else "rolling 20d return: unavailable")
    print(f"rolling 60d return: {latest_rolling_60:.4%}" if np.isfinite(latest_rolling_60) else "rolling 60d return: unavailable")
    print(f"drawdown: {current_drawdown:.4%}" if np.isfinite(current_drawdown) else "drawdown: unavailable")
    print(f"exposure: {exposure:.2%}" if np.isfinite(exposure) else "exposure: unavailable")
    print(f"cash: {cash:.2%}" if np.isfinite(cash) else "cash: unavailable")
    print(f"recommendation: {recommendation}")
    print(f"promotion status: {promotion_status}")
    print(f"reason: {' | '.join(reasons)}")
    print(f"Saved: {Path(REPORT_FILE).resolve()}")
    print(f"Saved: {Path(HISTORY_FILE).resolve()}")
    print(f"Saved: {Path(MONTHLY_FILE).resolve()}")
    return report


if __name__ == "__main__":
    run_growth_paper_governance()
