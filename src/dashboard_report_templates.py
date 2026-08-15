from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard_alert_engine import build_alert_engine
from dashboard_data_layer import (
    CSV_FILES,
    latest,
    latest_market_date,
    next_rebalance_date,
    numeric,
)
from dashboard_report_assets import fmt_money, fmt_pct, fmt_pct_points


@dataclass
class ReportBundle:
    report_type: str
    title: str
    start_date: str
    end_date: str
    generated_at: str
    sections: dict[str, Any]
    source_audit: pd.DataFrame
    status: str
    warnings: list[str]


def _series_stats(returns: pd.Series) -> dict[str, Any]:
    r = numeric(returns).dropna()
    if r.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "Calmar": np.nan, "Max Drawdown": np.nan, "Volatility": np.nan}
    equity = (1 + r).cumprod()
    periods = max(len(r), 1)
    cagr = equity.iloc[-1] ** (252 / periods) - 1 if periods > 1 else r.iloc[-1]
    vol = r.std(ddof=0) * np.sqrt(252) if len(r) > 1 else np.nan
    sharpe = r.mean() / r.std(ddof=0) * np.sqrt(252) if len(r) > 1 and r.std(ddof=0) else np.nan
    downside = r[r < 0].std(ddof=0)
    sortino = r.mean() / downside * np.sqrt(252) if len(r) > 1 and downside else np.nan
    dd = equity / equity.cummax() - 1
    maxdd = dd.min() if not dd.empty else np.nan
    calmar = cagr / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan
    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar, "Max Drawdown": maxdd, "Volatility": vol}


def _period_window(perf: pd.DataFrame, report_type: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if perf.empty or "date" not in perf.columns:
        return perf.copy()
    work = perf.sort_values("date").copy()
    end = pd.to_datetime(end_date, errors="coerce") if end_date else work["date"].max()
    if pd.isna(end):
        end = work["date"].max()
    if start_date:
        start = pd.to_datetime(start_date, errors="coerce")
    elif report_type.lower() == "daily":
        start = end
    elif report_type.lower() == "weekly":
        start = end - pd.Timedelta(days=7)
    elif report_type.lower() == "monthly":
        start = end - pd.DateOffset(months=1)
    elif report_type.lower() == "quarterly":
        start = end - pd.DateOffset(months=3)
    else:
        start = work["date"].min()
    if pd.isna(start):
        start = work["date"].min()
    return work[(work["date"] >= start.normalize()) & (work["date"] <= end.normalize())].copy()


def _source_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    official_keys = [
        "official_performance",
        "official_state",
        "official_actions",
        "official_rebalance_report",
        "official_monitor",
        "official_benchmark_daily",
        "official_benchmark_equity",
        "official_market_data_integrity",
        "official_accounting_reconciliation",
        "active_alerts",
        "anti_overfitting_governance",
        "out_of_sample_governance",
        "parameter_governance",
    ]
    rows = []
    for key in official_keys:
        path = CSV_FILES.get(key, f"{key}.csv")
        df = data.get(key, pd.DataFrame())
        latest_date = ""
        if not df.empty and "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce").dropna()
            latest_date = d.max().date().isoformat() if not d.empty else ""
        rows.append({"source_key": key, "source_file": path, "exists": Path(path).exists(), "rows": len(df), "latest_date": latest_date})
    return pd.DataFrame(rows)


def build_report_bundle(
    data: dict[str, pd.DataFrame], report_type: str = "daily", start_date: str | None = None, end_date: str | None = None
) -> ReportBundle:
    warnings: list[str] = []
    perf_all = data.get("official_performance", pd.DataFrame()).copy()
    perf = _period_window(perf_all, report_type, start_date, end_date)
    latest_perf = latest(perf_all).iloc[-1] if not latest(perf_all).empty else pd.Series(dtype=object)
    state = latest(data.get("official_state", pd.DataFrame())).copy()
    actions = data.get("official_actions", pd.DataFrame()).copy()
    actions_period = actions.copy()
    if not actions_period.empty and "date" in actions_period.columns and not perf.empty:
        actions_period = actions_period[(actions_period["date"] >= perf["date"].min()) & (actions_period["date"] <= perf["date"].max())]
    monitor = latest(data.get("official_monitor", pd.DataFrame()))
    monitor_row = monitor.iloc[-1] if not monitor.empty else pd.Series(dtype=object)
    bench = latest(data.get("official_benchmark_equity", pd.DataFrame()))
    bench_row = bench.iloc[-1] if not bench.empty else pd.Series(dtype=object)
    alerts_result = build_alert_engine(data, write_outputs=False)
    active_alerts = alerts_result["active"]
    history = alerts_result["history"]
    non_cash = (
        state[~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")].copy()
        if not state.empty and "ticker" in state.columns
        else pd.DataFrame()
    )
    weights = numeric(non_cash.get("paper_position_weight", pd.Series(dtype=float))) if not non_cash.empty else pd.Series(dtype=float)
    hhi = float((weights.dropna() ** 2).sum()) if not weights.empty else np.nan
    largest_position = "n/a"
    if not non_cash.empty and "paper_position_weight" in non_cash.columns:
        idx = numeric(non_cash["paper_position_weight"]).idxmax()
        largest_position = f"{non_cash.loc[idx, 'ticker']} ({fmt_pct(non_cash.loc[idx, 'paper_position_weight'])})"
    returns = (
        numeric(perf.get("gross_daily_return", perf.get("daily_return", pd.Series(dtype=float))))
        if not perf.empty
        else pd.Series(dtype=float)
    )
    stats = _series_stats(returns)
    drawdown_series = []
    if not perf.empty and "gross_portfolio_value" in perf.columns:
        eq = numeric(perf["gross_portfolio_value"])
        drawdown_series = (eq / eq.cummax() - 1).fillna(0).tolist()
    monthly = pd.DataFrame()
    if not perf_all.empty and "date" in perf_all.columns and "gross_daily_return" in perf_all.columns:
        m = perf_all.copy()
        m["year"] = m["date"].dt.year
        m["month"] = m["date"].dt.month
        monthly = (
            m.groupby(["year", "month"])["gross_daily_return"]
            .apply(lambda s: (1 + numeric(s).fillna(0)).prod() - 1)
            .reset_index(name="return")
        )
    research_rows = []
    for key, name in [
        ("anti_overfitting_governance", "Anti-overfitting"),
        ("out_of_sample_governance", "Out-of-sample"),
        ("parameter_governance", "Parameter stability"),
    ]:
        df = latest(data.get(key, pd.DataFrame()))
        if not df.empty:
            row = df.iloc[-1].to_dict()
            row["check"] = name
            research_rows.append(row)
    source_audit = _source_audit(data)
    if perf.empty:
        warnings.append("Official performance period has no rows.")
    if active_alerts.empty:
        active_alert_count = 0
    else:
        active_alert_count = len(active_alerts)
    sections = {
        "executive": {
            "portfolio_value": fmt_money(latest_perf.get("gross_portfolio_value", latest_perf.get("portfolio_value", np.nan))),
            "net_portfolio_value": fmt_money(latest_perf.get("estimated_net_portfolio_value", np.nan)),
            "gross_return": fmt_pct(latest_perf.get("gross_cumulative_return", np.nan)),
            "net_return": fmt_pct(latest_perf.get("estimated_net_cumulative_return", np.nan)),
            "spy_return": fmt_pct_points(bench_row.get("SPY_cumulative_pct", np.nan)),
            "qqq_return": fmt_pct_points(bench_row.get("QQQ_cumulative_pct", np.nan)),
            "exposure": fmt_pct(latest_perf.get("exposure", np.nan)),
            "cash": fmt_pct(latest_perf.get("cash_weight", np.nan)),
            "drawdown": fmt_pct(latest_perf.get("current_drawdown", latest_perf.get("max_drawdown", np.nan))),
            "holdings": ", ".join(non_cash.get("ticker", pd.Series(dtype=str)).astype(str).tolist()) if not non_cash.empty else "none",
            "largest_position": largest_position,
            "largest_risk_contributor": largest_position,
            "health_score": f"{alerts_result['health_score']:.1f}% ({alerts_result['health_label']})",
            "active_alerts": active_alert_count,
            "governance_status": str(monitor_row.get("governance_status", "unavailable")),
            "next_rebalance": next_rebalance_date(data),
        },
        "performance": {"rows": perf, "stats": stats, "drawdown": drawdown_series, "monthly": monthly},
        "portfolio": {"holdings": state, "non_cash": non_cash, "hhi": hhi},
        "risk": {
            "volatility": stats.get("Volatility"),
            "target_volatility": 0.22,
            "VaR95": returns.quantile(0.05) if not returns.empty else np.nan,
            "CVaR95": returns[returns <= returns.quantile(0.05)].mean() if len(returns) > 2 else np.nan,
            "HHI": hhi,
        },
        "execution": {
            "actions": actions_period,
            "turnover": fmt_pct(latest_perf.get("turnover", np.nan)),
            "cost": fmt_money(latest_perf.get("estimated_execution_cost", np.nan)),
        },
        "research": pd.DataFrame(research_rows),
        "governance": {
            "monitor": monitor,
            "market_data": latest(data.get("official_market_data_governance", pd.DataFrame())),
            "accounting": latest(data.get("official_accounting_reconciliation", pd.DataFrame())),
        },
        "alerts": {
            "active": active_alerts,
            "history": history,
            "health_score": alerts_result["health_score"],
            "health_label": alerts_result["health_label"],
        },
    }
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = perf["date"].min().date().isoformat() if not perf.empty and "date" in perf.columns else "unavailable"
    end = perf["date"].max().date().isoformat() if not perf.empty and "date" in perf.columns else latest_market_date(data)
    status = "report_generator_warning" if warnings else "report_generator_pass"
    return ReportBundle(
        report_type=report_type,
        title=f"{report_type.title()} Institutional Report",
        start_date=start,
        end_date=end,
        generated_at=generated_at,
        sections=sections,
        source_audit=source_audit,
        status=status,
        warnings=warnings,
    )
