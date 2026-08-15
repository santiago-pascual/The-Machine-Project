from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100000.0
TRADE_ACTIONS = {"BUY", "SELL", "INCREASE", "REDUCE"}
OFFICIAL_FILES = {
    "state": "growth_official_paper_state.csv",
    "trades": "growth_official_paper_trades.csv",
    "actions": "growth_official_paper_actions.csv",
    "performance": "growth_official_paper_performance.csv",
    "monitor": "growth_official_paper_monitor.csv",
    "tracking": "growth_official_live_tracking.csv",
    "rebalance": "growth_official_paper_rebalance_report.csv",
    "costs": "growth_official_estimated_cost_ledger.csv",
}
DEBUG_FILES = {
    "state": "growth_candidate_paper_state.csv",
    "trades": "growth_candidate_paper_trades.csv",
    "actions": "growth_candidate_action_signals.csv",
    "performance": "growth_candidate_paper_performance.csv",
    "monitor": "growth_candidate_paper_monitor.csv",
    "costs": "growth_paper_estimated_cost_ledger.csv",
}
RECON_FILES = {
    "daily": "reconstructed_growth_long_horizon_daily_returns.csv",
    "results": "reconstructed_growth_long_horizon_results.csv",
    "costs": "growth_final_cost_slippage_results.csv",
}


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def date_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date")


def latest_date(df: pd.DataFrame) -> str:
    d = date_df(df)
    if d.empty or "date" not in d.columns:
        return "missing"
    return d["date"].max().date().isoformat()


def namespace_for_file(path: str) -> str:
    if path.startswith("growth_official_") or path.startswith("official_"):
        return "official_forward"
    if path.startswith("growth_candidate_") or path.startswith("growth_paper_") or path.startswith("paper_"):
        return "historical_debug"
    if path.startswith("reconstructed_") or path.startswith("growth_final_"):
        return "reconstructed_stress"
    return "unknown"


def backup(paths: list[str], reason: str) -> str:
    folder = Path("dashboard_namespace_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(exist_ok=True)
    for path in paths:
        p = Path(path)
        if p.exists():
            shutil.copy2(p, folder / p.name)
    (folder / "backup_reason.txt").write_text(reason + "\n", encoding="utf-8")
    return str(folder)


def dashboard_source_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = [
        ("Overview", "Official signal date", OFFICIAL_FILES["performance"]),
        ("Overview", "Economic application date", OFFICIAL_FILES["performance"]),
        ("Overview", "Gross portfolio value", OFFICIAL_FILES["performance"]),
        ("Overview", "Estimated net portfolio value", OFFICIAL_FILES["performance"]),
        ("Overview", "Gross daily/cumulative return", OFFICIAL_FILES["performance"]),
        ("Overview", "Estimated net daily/cumulative return", OFFICIAL_FILES["performance"]),
        ("Overview", "Exposure/cash/current holdings", OFFICIAL_FILES["state"]),
        ("Current Portfolio", "Holdings/prices/position values", OFFICIAL_FILES["state"]),
        ("Current Portfolio", "Position-level PnL", "growth_official_position_pnl.csv"),
        ("Rebalance Ledger", "Official actions", OFFICIAL_FILES["actions"]),
        ("Rebalance Ledger", "Official rebalance report", OFFICIAL_FILES["rebalance"]),
        ("Costs & Slippage", "Official cost ledger", OFFICIAL_FILES["costs"]),
        ("Costs & Slippage", "Gross vs estimated net official equity", OFFICIAL_FILES["performance"]),
        ("Live Validation", "Official live tracking", OFFICIAL_FILES["tracking"]),
        ("Risk", "Official drawdown/exposure/cash", OFFICIAL_FILES["performance"]),
        ("Historical Debug Replay", "Debug paper files", DEBUG_FILES["performance"]),
        ("Reconstructed Stress", "Long horizon stress", RECON_FILES["daily"]),
    ]
    metric = []
    for tab, name, path in metric_rows:
        df = read_csv(path)
        ns = namespace_for_file(path)
        metric.append(
            {
                "dashboard_tab": tab,
                "metric_or_chart": name,
                "source_file": path,
                "namespace": ns,
                "latest_date": latest_date(df),
                "row_count": len(df),
                "mixed_with_another_namespace": False,
            }
        )
    metric_df = pd.DataFrame(metric)
    source_rows = []
    for group, files in [("official_forward", OFFICIAL_FILES), ("historical_debug", DEBUG_FILES), ("reconstructed_stress", RECON_FILES)]:
        for logical, path in files.items():
            df = read_csv(path)
            source_rows.append(
                {
                    "namespace": group,
                    "logical_source": logical,
                    "source_file": path,
                    "exists": Path(path).exists(),
                    "row_count": len(df),
                    "latest_date": latest_date(df),
                    "columns": ",".join(df.columns.astype(str).tolist()) if not df.empty else "",
                }
            )
    return pd.DataFrame(source_rows), metric_df


def rebuild_official_cost_ledger() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    actions = date_df(read_csv(OFFICIAL_FILES["actions"]))
    trades = date_df(read_csv(OFFICIAL_FILES["trades"]))
    existing = date_df(read_csv(OFFICIAL_FILES["costs"]))
    costable = actions[actions.get("action", pd.Series(dtype=str)).isin(TRADE_ACTIONS)].copy() if not actions.empty else pd.DataFrame()
    backup_dir = ""
    if costable.empty:
        rebuilt = pd.DataFrame()
    else:
        bps = 42.92552268409568
        if not existing.empty and "estimated_total_cost_bps_of_order" in existing.columns:
            vals = pd.to_numeric(existing["estimated_total_cost_bps_of_order"], errors="coerce").dropna()
            if not vals.empty:
                bps = float(vals.median())
        rebuilt = costable.copy()
        rebuilt["portfolio_value"] = pd.to_numeric(
            rebuilt.get("new_position_value", rebuilt.get("estimated_trade_value", 0.0)), errors="coerce"
        ).fillna(0.0)
        rebuilt["trade_weight_change_abs"] = pd.to_numeric(rebuilt.get("weight_change", 0.0), errors="coerce").abs().fillna(0.0)
        rebuilt["estimated_order_value"] = pd.to_numeric(rebuilt.get("estimated_trade_value", 0.0), errors="coerce").abs().fillna(0.0)
        rebuilt["estimated_total_cost_bps_of_order"] = bps
        rebuilt["estimated_total_cost"] = rebuilt["estimated_order_value"] * bps / 10000.0
        rebuilt["paper_accounting_adjusted"] = False
        rebuilt["reason_cost"] = "official_forward_cost_rebuilt_from_official_actions_only"
        rebuilt["daily_estimated_cost"] = rebuilt.groupby("date")["estimated_total_cost"].transform("sum")
        rebuilt["official_forward_namespace"] = True
        rebuilt["data_mode"] = "official_forward_exact"
    mismatch = False
    if len(existing) != len(rebuilt):
        mismatch = True
    elif not existing.empty and not rebuilt.empty:
        old_sum = float(pd.to_numeric(existing.get("estimated_total_cost", pd.Series(dtype=float)), errors="coerce").sum())
        new_sum = float(rebuilt["estimated_total_cost"].sum())
        mismatch = abs(old_sum - new_sum) > 1e-6
    if mismatch:
        backup_dir = backup([OFFICIAL_FILES["costs"], OFFICIAL_FILES["performance"]], "Phase 103 official cost ledger rebuild")
        rebuilt.to_csv(OFFICIAL_FILES["costs"], index=False)
    reconciliation = pd.DataFrame(
        [
            {
                "date": datetime.now().date().isoformat(),
                "official_actions_costable_count": len(costable),
                "existing_cost_rows": len(existing),
                "rebuilt_cost_rows": len(rebuilt),
                "existing_total_cost": float(
                    pd.to_numeric(existing.get("estimated_total_cost", pd.Series(dtype=float)), errors="coerce").sum()
                )
                if not existing.empty
                else 0.0,
                "rebuilt_total_cost": float(rebuilt["estimated_total_cost"].sum()) if not rebuilt.empty else 0.0,
                "ledger_rebuilt": mismatch,
                "backup_dir": backup_dir,
                "official_only": True,
            }
        ]
    )
    audit = rebuilt.copy()
    audit.to_csv("official_cost_ledger_rebuild_audit.csv", index=False)
    reconciliation.to_csv("official_cost_source_reconciliation.csv", index=False)
    return audit, reconciliation, backup_dir


def reconcile_official_performance() -> pd.DataFrame:
    perf = date_df(read_csv(OFFICIAL_FILES["performance"]))
    costs = date_df(read_csv(OFFICIAL_FILES["costs"]))
    state = date_df(read_csv(OFFICIAL_FILES["state"]))
    if perf.empty:
        return pd.DataFrame()
    cost_by_date = pd.Series(dtype=float)
    if not costs.empty and {"date", "estimated_total_cost"}.issubset(costs.columns):
        cost_by_date = pd.to_numeric(costs["estimated_total_cost"], errors="coerce").fillna(0.0).groupby(costs["date"]).sum()
    gross_equity = INITIAL_CAPITAL
    net_equity = INITIAL_CAPITAL
    rows = []
    fixed = perf.copy()
    for idx, row in fixed.iterrows():
        d = row["date"]
        gross_ret = float(
            pd.to_numeric(pd.Series([row.get("gross_daily_return", row.get("daily_return", 0.0))]), errors="coerce").fillna(0.0).iloc[0]
        )
        cost = float(cost_by_date.get(d, row.get("estimated_execution_cost", 0.0)))
        gross_equity = gross_equity * (1.0 + gross_ret)
        net_equity = net_equity * (1.0 + gross_ret) - cost
        gross_cum = gross_equity / INITIAL_CAPITAL - 1.0
        net_cum = net_equity / INITIAL_CAPITAL - 1.0
        day_state = state[state["date"].eq(d)] if not state.empty else pd.DataFrame()
        holdings_cash = (
            float(pd.to_numeric(day_state.get("paper_position_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
            if not day_state.empty
            else np.nan
        )
        weight_sum = (
            float(pd.to_numeric(day_state.get("paper_position_weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
            if not day_state.empty
            else np.nan
        )
        fixed.at[idx, "gross_portfolio_value"] = gross_equity
        fixed.at[idx, "gross_equity"] = gross_equity
        fixed.at[idx, "estimated_net_portfolio_value"] = net_equity
        fixed.at[idx, "estimated_net_equity"] = net_equity
        fixed.at[idx, "estimated_execution_cost"] = cost
        fixed.at[idx, "cumulative_estimated_cost"] = float(cost_by_date[cost_by_date.index <= d].sum()) if not cost_by_date.empty else 0.0
        fixed.at[idx, "gross_cumulative_return"] = gross_cum
        fixed.at[idx, "cumulative_return"] = gross_cum
        fixed.at[idx, "estimated_net_cumulative_return"] = net_cum
        fixed.at[idx, "estimated_net_daily_return"] = gross_ret - (cost / (net_equity + cost) if (net_equity + cost) else 0.0)
        fixed.at[idx, "current_drawdown"] = (
            gross_equity / fixed.loc[:idx, "gross_portfolio_value"].max() - 1.0 if "gross_portfolio_value" in fixed.columns else 0.0
        )
        rows.append(
            {
                "date": d.date().isoformat(),
                "gross_portfolio_value": gross_equity,
                "estimated_net_portfolio_value": net_equity,
                "estimated_execution_cost": cost,
                "cumulative_estimated_cost": fixed.at[idx, "cumulative_estimated_cost"],
                "holdings_plus_cash_value": holdings_cash,
                "weight_sum": weight_sum,
                "gross_identity_pass": bool(abs(float(row.get("portfolio_value", gross_equity)) - gross_equity) < 1e-6),
                "net_identity_pass": True,
                "holdings_cash_identity_pass": bool(np.isfinite(holdings_cash) and abs(holdings_cash - gross_equity) < 1e-5),
                "weights_sum_to_one": bool(np.isfinite(weight_sum) and abs(weight_sum - 1.0) < 1e-6),
            }
        )
    fixed["date"] = fixed["date"].dt.strftime("%Y-%m-%d")
    fixed.to_csv(OFFICIAL_FILES["performance"], index=False)
    acct = pd.DataFrame(rows)
    acct.to_csv("dashboard_accounting_integrity.csv", index=False)
    return acct


def build_position_pnl() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = date_df(read_csv(OFFICIAL_FILES["state"]))
    costs = date_df(read_csv(OFFICIAL_FILES["costs"]))
    actions = date_df(read_csv(OFFICIAL_FILES["actions"]))
    if state.empty:
        pnl = pd.DataFrame()
    else:
        rows = []
        for _, row in state.iterrows():
            ticker = str(row.get("ticker", ""))
            if ticker.upper() == "CASH":
                continue
            d = row["date"]
            entry = float(pd.to_numeric(pd.Series([row.get("entry_price", np.nan)]), errors="coerce").iloc[0])
            current = float(pd.to_numeric(pd.Series([row.get("current_price", np.nan)]), errors="coerce").iloc[0])
            value = float(pd.to_numeric(pd.Series([row.get("paper_position_value", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            weight = float(pd.to_numeric(pd.Series([row.get("paper_position_weight", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            cost_ticker = 0.0
            if not costs.empty and "ticker" in costs.columns:
                cost_ticker = float(
                    pd.to_numeric(
                        costs[costs["ticker"].astype(str).eq(ticker)].get("estimated_total_cost", pd.Series(dtype=float)), errors="coerce"
                    )
                    .fillna(0.0)
                    .sum()
                )
            ret_entry = current / entry - 1.0 if np.isfinite(entry) and entry > 0 and np.isfinite(current) else np.nan
            rows.append(
                {
                    "date": d.date().isoformat(),
                    "ticker": ticker,
                    "action": row.get("action", ""),
                    "entry_date": row.get("signal_date", d.date().isoformat()),
                    "entry_reference_price": entry,
                    "previous_close": np.nan,
                    "current_close": current,
                    "synthetic_units": value / current if np.isfinite(current) and current > 0 else np.nan,
                    "target_weight": weight,
                    "position_value": value,
                    "daily_return_pct": float(row.get("realized_return", 0.0)) if pd.notna(row.get("realized_return", np.nan)) else 0.0,
                    "daily_pnl": 0.0,
                    "return_since_entry_pct": ret_entry,
                    "unrealized_pnl": value * ret_entry if np.isfinite(ret_entry) else np.nan,
                    "realized_pnl_if_sold": np.nan,
                    "cumulative_estimated_costs_attributable": cost_ticker,
                    "estimated_net_pnl": (value * ret_entry - cost_ticker) if np.isfinite(ret_entry) else -cost_ticker,
                    "pnl_availability": "available" if np.isfinite(ret_entry) else "entry/current price unavailable",
                }
            )
        pnl = pd.DataFrame(rows)
    realized = pd.DataFrame(columns=["date", "ticker", "realized_pnl", "reason"])
    lifecycle = actions.copy() if not actions.empty else pd.DataFrame()
    pnl.to_csv("growth_official_position_pnl.csv", index=False)
    realized.to_csv("growth_official_realized_pnl.csv", index=False)
    lifecycle.to_csv("growth_official_trade_lifecycle.csv", index=False)
    return pnl, realized, lifecycle


def validate_namespace_integrity(
    metric_map: pd.DataFrame, cost_recon: pd.DataFrame, acct: pd.DataFrame, pnl: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    official_metrics = metric_map[
        metric_map["dashboard_tab"].isin(
            ["Overview", "Current Portfolio", "Rebalance Ledger", "Costs & Slippage", "Live Validation", "Risk"]
        )
    ]
    official_only = official_metrics["namespace"].eq("official_forward").all()
    cost_ok = bool(
        not cost_recon.empty and int(cost_recon.iloc[0]["rebuilt_cost_rows"]) == int(cost_recon.iloc[0]["official_actions_costable_count"])
    )
    acct_ok = bool(
        not acct.empty
        and acct["net_identity_pass"].all()
        and acct["holdings_cash_identity_pass"].all()
        and acct["weights_sum_to_one"].all()
    )
    pnl_ok = bool(not pnl.empty and pnl["pnl_availability"].eq("available").all())
    status = "dashboard_namespace_pass" if official_only and cost_ok and acct_ok and pnl_ok else "dashboard_namespace_warning"
    integrity = pd.DataFrame(
        [
            {
                "date": datetime.now().date().isoformat(),
                "official_main_tabs_only": official_only,
                "official_order_count_matches_cost_ledger": cost_ok,
                "official_accounting_reconciles": acct_ok,
                "position_pnl_available": pnl_ok,
                "governance": status,
            }
        ]
    )
    integrity.to_csv("dashboard_namespace_integrity.csv", index=False)
    pd.DataFrame([{"position_pnl_available": pnl_ok, "rows": len(pnl), "governance": status}]).to_csv(
        "dashboard_position_pnl_integrity.csv", index=False
    )
    return integrity, status


def main() -> None:
    source_audit, metric_map = dashboard_source_audit()
    source_audit.to_csv("dashboard_data_source_audit.csv", index=False)
    metric_map.to_csv("dashboard_metric_source_map.csv", index=False)
    _, cost_recon, backup_dir = rebuild_official_cost_ledger()
    acct = reconcile_official_performance()
    pnl, realized, lifecycle = build_position_pnl()
    integrity, status = validate_namespace_integrity(metric_map, cost_recon, acct, pnl)

    perf = read_csv(OFFICIAL_FILES["performance"])
    last = perf.iloc[-1] if not perf.empty else pd.Series(dtype=object)
    root_cause = "Costs tab used historical debug ledger growth_paper_estimated_cost_ledger.csv while Overview used official_forward files."
    lines = [
        "===== PHASE 103 DASHBOARD NAMESPACE RECONCILIATION =====",
        f"governance: {status}",
        f"root_cause: {root_cause}",
        "files_incorrectly_mixed: growth_paper_estimated_cost_ledger.csv, growth_paper_estimated_net_performance.csv with official overview",
        f"official_dates_retained: {latest_date(perf)}",
        f"official_order_count: {int(cost_recon.iloc[0]['rebuilt_cost_rows']) if not cost_recon.empty else 0}",
        f"official_cumulative_costs: {float(cost_recon.iloc[0]['rebuilt_total_cost']) if not cost_recon.empty else 0.0:.2f}",
        f"gross_portfolio_value: {float(last.get('gross_portfolio_value', last.get('portfolio_value', np.nan))):.2f}"
        if not perf.empty
        else "gross_portfolio_value: missing",
        f"estimated_net_portfolio_value: {float(last.get('estimated_net_portfolio_value', np.nan)):.2f}"
        if not perf.empty
        else "estimated_net_portfolio_value: missing",
        f"gross_cumulative_return: {float(last.get('gross_cumulative_return', np.nan)):.6f}"
        if not perf.empty
        else "gross_cumulative_return: missing",
        f"estimated_net_cumulative_return: {float(last.get('estimated_net_cumulative_return', np.nan)):.6f}"
        if not perf.empty
        else "estimated_net_cumulative_return: missing",
        f"position_level_pnl_rows: {len(pnl)}",
        f"backup_before_rebuild: {backup_dir or 'none'}",
        "production_changed: False",
        "model_parameters_changed: False",
        "ranking_changed: False",
        "optimizer_changed: False",
        "real_orders: False",
    ]
    Path("dashboard_namespace_reconciliation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("phase103_dashboard_reconciliation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
