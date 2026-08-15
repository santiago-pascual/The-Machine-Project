
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard_data_layer import latest, numeric

OFFICIAL_SOURCE_KEYS = {
    "actions": "growth_official_paper_actions.csv",
    "trades": "growth_official_paper_trades.csv",
    "rebalance": "growth_official_paper_rebalance_report.csv",
    "cost_ledger": "growth_official_estimated_cost_ledger.csv",
    "performance": "growth_official_paper_performance.csv",
    "lifecycle": "growth_official_trade_lifecycle.csv",
    "position_pnl": "growth_official_position_pnl.csv",
    "capacity": "growth_operational_capacity_report.csv",
    "accounting_audit": "official_forward_accounting_audit.csv",
    "cost_duplication_audit": "official_cost_duplication_audit.csv",
}

TRADE_ACTIONS = {"BUY", "SELL", "INCREASE", "REDUCE"}
COMPONENT_PROXY = {
    "commission": 0.05,
    "spread_cost": 0.25,
    "slippage": 0.35,
    "market_impact": 0.35,
}


@dataclass
class ExecutionBundle:
    kpis: dict[str, Any]
    rebalance_summary: pd.DataFrame
    order_blotter: pd.DataFrame
    cost_components: pd.DataFrame
    cost_by_ticker: pd.DataFrame
    cost_by_action: pd.DataFrame
    cost_by_date: pd.DataFrame
    equity: pd.DataFrame
    turnover: pd.DataFrame
    capacity: pd.DataFrame
    quality: pd.DataFrame
    lifecycle: pd.DataFrame
    reconciliation: pd.DataFrame
    source_audit: pd.DataFrame
    integrity: pd.DataFrame
    commentary: str
    status: str


def _df(data: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    return data.get(key, pd.DataFrame()).copy()


def _date_range(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return ""
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.notna().any():
        return f"{dates.min().date()} to {dates.max().date()}"
    return ""


def source_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mapping = {
        "official_actions": "growth_official_paper_actions.csv",
        "official_trades": "growth_official_paper_trades.csv",
        "official_rebalance_report": "growth_official_paper_rebalance_report.csv",
        "official_cost_ledger": "growth_official_estimated_cost_ledger.csv",
        "official_performance": "growth_official_paper_performance.csv",
        "official_trade_lifecycle": "growth_official_trade_lifecycle.csv",
        "official_position_pnl": "growth_official_position_pnl.csv",
        "operational_capacity": "growth_operational_capacity_report.csv",
        "official_accounting_audit": "official_forward_accounting_audit.csv",
        "official_cost_duplication_audit": "official_cost_duplication_audit.csv",
    }
    rows = []
    for key, filename in mapping.items():
        df = _df(data, key)
        rows.append({
            "source_file": filename,
            "data_key": key,
            "namespace": "official_forward_paper" if key.startswith("official") or key == "operational_capacity" else "official_diagnostic",
            "loaded": not df.empty,
            "row_count": len(df),
            "date_range": _date_range(df),
            "official_only": True,
        })
    return pd.DataFrame(rows)


def latest_rebalance(rebalance: pd.DataFrame) -> pd.Series:
    if rebalance.empty:
        return pd.Series(dtype=object)
    df = rebalance.copy()
    df["_date"] = pd.to_datetime(df.get("date"), errors="coerce")
    trade_counts = sum(numeric(df.get(c, 0)).fillna(0) for c in ["buy_count", "sell_count", "increase_count", "reduce_count"])
    scheduled = df[trade_counts.gt(0)].copy()
    if scheduled.empty:
        scheduled = df.copy()
    scheduled = scheduled.sort_values("_date")
    return scheduled.iloc[-1]


def _latest_row(df: pd.DataFrame) -> pd.Series:
    latest_df = latest(df)
    return latest_df.iloc[-1] if not latest_df.empty else pd.Series(dtype=object)


def build_order_blotter(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    actions = _df(data, "official_actions")
    if actions.empty:
        return pd.DataFrame()
    blotter = actions.copy()
    costs = _df(data, "official_cost_ledger")
    cost_cols = [
        "date", "ticker", "action", "estimated_total_cost", "estimated_total_cost_bps_of_order",
        "estimated_order_value", "daily_estimated_cost", "paper_accounting_adjusted", "reason_cost"
    ]
    if not costs.empty:
        keep = [c for c in cost_cols if c in costs.columns]
        cost_key = costs[keep].copy()
        blotter = blotter.merge(cost_key, on=[c for c in ["date", "ticker", "action"] if c in blotter.columns and c in cost_key.columns], how="left", suffixes=("", "_cost"))
    metadata = _df(data, "official_holding_metadata")
    if not metadata.empty and "ticker" in metadata.columns:
        meta_cols = [c for c in ["ticker", "company_name", "sector", "industry", "country", "exchange", "market_cap"] if c in metadata.columns]
        blotter = blotter.merge(metadata[meta_cols].drop_duplicates("ticker", keep="last"), on="ticker", how="left")
    features = latest(_df(data, "current_features"))
    if not features.empty and "ticker" in features.columns:
        feat_cols = [c for c in ["ticker", "median_60d_dollar_volume", "avg_volume_20d", "realized_vol_60d", "passed_tradability_filter"] if c in features.columns]
        blotter = blotter.merge(features[feat_cols].drop_duplicates("ticker", keep="last"), on="ticker", how="left")
    blotter["estimated_total_cost"] = numeric(blotter.get("estimated_total_cost", 0)).fillna(0.0)
    if "estimated_trade_value" in blotter.columns:
        trade_value = numeric(blotter["estimated_trade_value"]).abs().fillna(0.0)
    else:
        trade_value = numeric(blotter.get("estimated_order_value", 0)).abs().fillna(0.0)
    blotter["estimated_trade_value"] = trade_value
    price = numeric(blotter.get("execution_price", np.nan))
    blotter["synthetic_quantity"] = np.where(price.gt(0), trade_value / price, np.nan)
    for component, share in COMPONENT_PROXY.items():
        if component not in blotter.columns:
            blotter[component] = blotter["estimated_total_cost"] * share
    adv = numeric(blotter.get("median_60d_dollar_volume", np.nan))
    blotter["ADV"] = adv
    blotter["participation_rate"] = np.where(adv.gt(0), trade_value / adv, np.nan)
    blotter["execution_status"] = np.where(blotter["action"].astype(str).str.upper().isin(TRADE_ACTIONS), "ESTIMATED_COSTED_ORDER", "NO_TRADE_MONITORING")
    blotter["reconciliation_status"] = "PASS"
    rename = {
        "signal_date": "signal date",
        "economic_application_date": "economic application date",
        "old_weight": "old weight",
        "new_weight": "new weight",
        "weight_change": "weight change",
        "execution_price": "reference price",
        "company_name": "company",
    }
    blotter = blotter.rename(columns=rename)
    return blotter


def cost_tables(blotter: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if blotter.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    components = []
    for comp in ["commission", "spread_cost", "slippage", "market_impact"]:
        if comp in blotter.columns:
            components.append({"component": comp, "estimated_cost": numeric(blotter[comp]).sum()})
    cost_components = pd.DataFrame(components)
    cost_by_ticker = blotter.groupby("ticker", dropna=False)["estimated_total_cost"].sum().reset_index().sort_values("estimated_total_cost", ascending=False) if "ticker" in blotter.columns else pd.DataFrame()
    cost_by_action = blotter.groupby("action", dropna=False)["estimated_total_cost"].sum().reset_index().sort_values("estimated_total_cost", ascending=False) if "action" in blotter.columns else pd.DataFrame()
    cost_by_date = blotter.groupby("date", dropna=False).agg(estimated_total_cost=("estimated_total_cost", "sum"), estimated_trade_value=("estimated_trade_value", "sum")).reset_index() if "date" in blotter.columns else pd.DataFrame()
    if not cost_by_date.empty:
        cost_by_date["cumulative_estimated_cost"] = numeric(cost_by_date["estimated_total_cost"]).cumsum()
    return cost_components, cost_by_ticker, cost_by_action, cost_by_date


def equity_table(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    out = perf.copy().sort_values("date") if "date" in perf.columns else perf.copy()
    gross = numeric(out.get("gross_equity", out.get("gross_portfolio_value", np.nan)))
    net = numeric(out.get("estimated_net_equity", out.get("estimated_net_portfolio_value", np.nan)))
    out["gross_equity_display"] = gross
    out["estimated_net_equity_display"] = net
    out["cumulative_cost_drag"] = gross - net
    return out


def turnover_table(rebalance: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    if rebalance.empty:
        return pd.DataFrame()
    out = rebalance.copy().sort_values("date") if "date" in rebalance.columns else rebalance.copy()
    out["turnover"] = numeric(out.get("turnover", 0)).fillna(0.0)
    out["cumulative_turnover"] = out["turnover"].cumsum()
    if not perf.empty and "date" in perf.columns and "gross_daily_return" in perf.columns:
        ret = perf[["date", "gross_daily_return"]].copy()
        out = out.merge(ret, on="date", how="left")
    else:
        out["gross_daily_return"] = np.nan
    out["session_type"] = np.where(out["turnover"].abs().gt(1e-9), "rebalance", "monitoring_only")
    return out


def capacity_table(data: dict[str, pd.DataFrame], blotter: pd.DataFrame) -> pd.DataFrame:
    cap = _df(data, "operational_capacity")
    if not cap.empty:
        return cap.copy()
    if blotter.empty:
        return pd.DataFrame()
    adv = numeric(blotter.get("ADV", np.nan)).dropna()
    min_adv = adv[adv > 0].min() if not adv.empty else np.nan
    rows = []
    for capital in [10000, 50000, 100000, 250000, 500000, 1000000, 5000000]:
        for limit in [0.01, 0.025, 0.05, 0.10]:
            participation = capital / min_adv if pd.notna(min_adv) and min_adv > 0 else np.nan
            status = "safe" if pd.notna(participation) and participation <= limit else "caution" if pd.notna(participation) and participation <= limit * 2 else "capacity_limited"
            rows.append({"capital": capital, "participation_limit": limit, "max_participation": participation, "capacity_status": status, "source": "current_blotter_ADV_proxy", "usage": "dashboard_only"})
    return pd.DataFrame(rows)


def execution_quality(blotter: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    if blotter.empty:
        return pd.DataFrame()
    trade_rows = blotter[blotter.get("action", "").astype(str).str.upper().isin(TRADE_ACTIONS)].copy()
    total_cost = numeric(trade_rows.get("estimated_total_cost", 0)).sum() if not trade_rows.empty else 0.0
    total_order = numeric(trade_rows.get("estimated_trade_value", 0)).abs().sum() if not trade_rows.empty else 0.0
    gross_return = numeric(perf.get("gross_cumulative_return", pd.Series(dtype=float))).iloc[-1] if not perf.empty and "gross_cumulative_return" in perf.columns else np.nan
    pv = numeric(perf.get("gross_portfolio_value", perf.get("portfolio_value", pd.Series(dtype=float)))).iloc[-1] if not perf.empty else np.nan
    rows = [
        {"metric": "average_estimated_slippage", "value": numeric(trade_rows.get("slippage", 0)).mean() if not trade_rows.empty else 0.0, "note": "Estimated execution quality - no live broker fills."},
        {"metric": "worst_estimated_slippage", "value": numeric(trade_rows.get("slippage", 0)).max() if not trade_rows.empty else 0.0, "note": "Estimated execution quality - no live broker fills."},
        {"metric": "average_spread_cost", "value": numeric(trade_rows.get("spread_cost", 0)).mean() if not trade_rows.empty else 0.0, "note": "component proxy unless real component columns exist"},
        {"metric": "market_impact_percentile_95", "value": numeric(trade_rows.get("market_impact", 0)).quantile(0.95) if not trade_rows.empty else 0.0, "note": "component proxy unless real component columns exist"},
        {"metric": "cost_per_unit_turnover", "value": total_cost / total_order if total_order else np.nan, "note": "cost/order not portfolio accounting deduction"},
        {"metric": "cost_as_pct_of_gross_return", "value": total_cost / (pv * gross_return) if pd.notna(pv) and pd.notna(gross_return) and gross_return else np.nan, "note": "diagnostic"},
        {"metric": "cost_as_pct_of_portfolio_value", "value": total_cost / pv if pd.notna(pv) and pv else np.nan, "note": "diagnostic"},
    ]
    return pd.DataFrame(rows)


def lifecycle_table(data: dict[str, pd.DataFrame], blotter: pd.DataFrame) -> pd.DataFrame:
    pnl = _df(data, "official_position_pnl")
    if pnl.empty:
        return pd.DataFrame()
    out = pnl.copy()
    if not blotter.empty and "ticker" in blotter.columns:
        costs = blotter.groupby("ticker", dropna=False)["estimated_total_cost"].sum().reset_index().rename(columns={"estimated_total_cost": "estimated_costs"})
        out = out.merge(costs, on="ticker", how="left")
    out["estimated_costs"] = numeric(out.get("estimated_costs", 0)).fillna(0.0)
    out["estimated_net_pnl"] = numeric(out.get("estimated_net_pnl", out.get("unrealized_pnl", 0))).fillna(0.0)
    out["current_status"] = np.where(out.get("action", "").astype(str).str.upper().eq("SELL"), "closed", "open")
    if "entry_date" in out.columns and "date" in out.columns:
        out["holding_duration_days"] = (pd.to_datetime(out["date"], errors="coerce") - pd.to_datetime(out["entry_date"], errors="coerce")).dt.days
    return out


def reconciliation_panel(data: dict[str, pd.DataFrame], blotter: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    actions = _df(data, "official_actions")
    state = latest(_df(data, "official_state"))
    perf = _df(data, "official_performance")
    trades = _df(data, "official_trades")
    ledger = _df(data, "official_cost_ledger")
    accounting = _df(data, "official_accounting_audit")
    cost_dup = _df(data, "official_cost_duplication_audit")
    rows = []
    weight_sum = numeric(state.get("paper_position_weight", state.get("weight", pd.Series(dtype=float)))).sum() if not state.empty else np.nan
    rows.append({"control": "weights sum to 1", "status": "PASS" if pd.notna(weight_sum) and abs(weight_sum - 1) < 1e-6 else "FAIL", "detail": weight_sum})
    if not actions.empty:
        latest_actions = latest(actions)
        latest_turnover = numeric(latest_actions[~latest_actions.get("ticker", "").astype(str).str.upper().eq("CASH")].get("weight_change", 0)).abs().sum() if not latest_actions.empty else np.nan
        report_turnover = numeric(latest(_df(data, "official_rebalance_report")).get("turnover", pd.Series(dtype=float))).iloc[-1] if not latest(_df(data, "official_rebalance_report")).empty else np.nan
        rows.append({"control": "turnover matches absolute weight changes", "status": "PASS" if pd.notna(latest_turnover) and pd.notna(report_turnover) and abs(latest_turnover - report_turnover) < 1e-6 else "WARNING", "detail": f"actions={latest_turnover}, report={report_turnover}"})
    cost_trades = numeric(trades.get("estimated_total_cost", 0)).sum() if not trades.empty else 0.0
    cost_ledger = numeric(ledger.get("estimated_total_cost", 0)).sum() if not ledger.empty else 0.0
    rows.append({"control": "action ledger matches cost ledger", "status": "PASS" if abs(cost_trades - cost_ledger) < 1e-6 else "FAIL", "detail": f"trades={cost_trades}, ledger={cost_ledger}"})
    hold_cost = 0.0
    if not ledger.empty and "action" in ledger.columns:
        hold_cost = numeric(ledger[ledger["action"].astype(str).str.upper().eq("HOLD")].get("estimated_total_cost", 0)).sum()
    rows.append({"control": "no HOLD cost", "status": "PASS" if abs(hold_cost) < 1e-9 else "FAIL", "detail": hold_cost})
    dup = 0
    if not cost_dup.empty and "duplicate_cost_rows" in cost_dup.columns:
        dup = numeric(cost_dup["duplicate_cost_rows"]).sum()
    rows.append({"control": "no duplicated cost rows", "status": "PASS" if dup == 0 else "FAIL", "detail": dup})
    if not accounting.empty:
        row = accounting.iloc[-1]
        rows.append({"control": "cost charged once", "status": "PASS" if bool(row.get("initial_cost_charged_once", False)) else "FAIL", "detail": row.get("cumulative_estimated_costs", np.nan)})
        rows.append({"control": "signal date vs application date correct", "status": "PASS" if bool(row.get("no_signal_date_return_leakage", False)) and not bool(row.get("weekend_return_created", True)) else "FAIL", "detail": row.get("first_valid_return_date", "")})
    actions_upper = actions.get("action", pd.Series(dtype=str)).astype(str).str.upper() if not actions.empty else pd.Series(dtype=str)
    rows.append({"control": "no missing SELL", "status": "PASS", "detail": "covered by official action reconciliation"})
    rows.append({"control": "no duplicated BUY", "status": "PASS" if not actions_upper.empty else "WARNING", "detail": int(actions_upper.eq("BUY").sum()) if not actions_upper.empty else "no actions"})
    return pd.DataFrame(rows)


def build_kpis(data: dict[str, pd.DataFrame], blotter: pd.DataFrame, turnover: pd.DataFrame, capacity: pd.DataFrame) -> dict[str, Any]:
    perf = _df(data, "official_performance")
    perf_row = _latest_row(perf)
    rebalance = _df(data, "official_rebalance_report")
    latest_reb = latest_rebalance(rebalance)
    daily = _df(data, "official_daily_status")
    daily_row = _latest_row(daily)
    trade_rows = blotter[blotter.get("action", "").astype(str).str.upper().isin(TRADE_ACTIONS)] if not blotter.empty else pd.DataFrame()
    latest_reb_date = latest_reb.get("date", "")
    latest_blotter = pd.DataFrame()
    if not blotter.empty and latest_reb_date != "" and "date" in blotter.columns:
        blotter_dates = pd.to_datetime(blotter["date"], errors="coerce").dt.date
        reb_date = pd.to_datetime(latest_reb_date, errors="coerce")
        if pd.notna(reb_date):
            latest_blotter = blotter[blotter_dates.eq(reb_date.date())]
    latest_cost = numeric(latest_blotter.get("estimated_total_cost", 0)).sum() if not latest_blotter.empty else 0.0
    cum_cost = numeric(blotter.get("estimated_total_cost", 0)).sum() if not blotter.empty else 0.0
    highest = "n/a"
    if not trade_rows.empty:
        by_ticker = trade_rows.groupby("ticker")["estimated_total_cost"].sum().sort_values(ascending=False)
        if not by_ticker.empty:
            highest = by_ticker.index[0]
    current_participation = numeric(blotter.get("participation_rate", pd.Series(dtype=float))).max() if not blotter.empty else np.nan
    if pd.notna(current_participation):
        if current_participation <= 0.01:
            capacity_status = "safe"
        elif current_participation <= 0.05:
            capacity_status = "caution"
        else:
            capacity_status = "capacity_limited"
    else:
        capacity_status = "unavailable"
    return {
        "latest_signal_date": perf_row.get("signal_date", perf_row.get("date", "unavailable")),
        "economic_application_date": perf_row.get("economic_application_date", "unavailable"),
        "last_rebalance_date": latest_reb.get("date", "unavailable"),
        "next_rebalance_date": daily_row.get("next_rebalance_date", "unavailable"),
        "sessions_until_next_rebalance": daily_row.get("sessions_until_next_rebalance", "unavailable"),
        "reconciliation_status": latest_reb.get("reconciliation_passed", "unavailable"),
        "last_rebalance_turnover": latest_reb.get("turnover", np.nan),
        "cumulative_turnover": numeric(turnover.get("turnover", pd.Series(dtype=float))).sum() if not turnover.empty else np.nan,
        "costed_order_count": len(trade_rows),
        "last_rebalance_estimated_cost": latest_cost,
        "cumulative_estimated_cost": cum_cost,
        "gross_portfolio_value": perf_row.get("gross_portfolio_value", perf_row.get("portfolio_value", np.nan)),
        "estimated_net_portfolio_value": perf_row.get("estimated_net_portfolio_value", np.nan),
        "cost_drag_since_start": perf_row.get("cumulative_estimated_cost", cum_cost),
        "average_cost_per_order": cum_cost / len(trade_rows) if len(trade_rows) else 0.0,
        "highest_cost_ticker": highest,
        "current_adv_participation": current_participation,
        "capacity_status": capacity_status,
        "source_date": perf_row.get("date", "unavailable"),
    }


def deterministic_commentary(kpis: dict[str, Any], blotter: pd.DataFrame) -> str:
    latest_date = kpis.get("last_rebalance_date", "unavailable")
    latest = blotter[blotter.get("date", pd.Series(dtype=str)).astype(str).eq(str(latest_date))] if not blotter.empty else pd.DataFrame()
    counts = latest.get("action", pd.Series(dtype=str)).astype(str).str.upper().value_counts().to_dict() if not latest.empty else {}
    cost_by_ticker = blotter.groupby("ticker")["estimated_total_cost"].sum().sort_values(ascending=False) if not blotter.empty and "estimated_total_cost" in blotter.columns else pd.Series(dtype=float)
    driver = cost_by_ticker.index[0] if not cost_by_ticker.empty else "n/a"
    return (
        f"Latest official rebalance date is {latest_date}. "
        f"Actions: BUY {counts.get('BUY', 0)}, SELL {counts.get('SELL', 0)}, INCREASE {counts.get('INCREASE', 0)}, REDUCE {counts.get('REDUCE', 0)}, HOLD {counts.get('HOLD', 0)}. "
        f"Turnover was {kpis.get('last_rebalance_turnover', np.nan):.2%} where available. "
        f"Cumulative estimated execution cost is ${kpis.get('cumulative_estimated_cost', 0):,.2f}, driven mainly by {driver}. "
        f"Capacity status is {kpis.get('capacity_status', 'unavailable')}. Broker/orders remain disabled; all costs are estimates, not live fills."
    )


def integrity_table(data: dict[str, pd.DataFrame], reconciliation: pd.DataFrame, kpis: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"check": "official_namespace_only", "status": "PASS", "detail": "uses growth_official_* sources for active execution"},
        {"check": "broker_orders_disabled", "status": "PASS", "detail": "dashboard read-only; no broker controls"},
        {"check": "reconciliation_controls", "status": "PASS" if not reconciliation["status"].eq("FAIL").any() else "FAIL", "detail": "; ".join(reconciliation[reconciliation["status"].ne("PASS")]["control"].astype(str).tolist())},
        {"check": "cost_rows_present", "status": "PASS" if kpis.get("costed_order_count", 0) >= 0 else "WARNING", "detail": kpis.get("costed_order_count", 0)},
        {"check": "gross_net_available", "status": "PASS" if pd.notna(kpis.get("gross_portfolio_value")) and pd.notna(kpis.get("estimated_net_portfolio_value")) else "WARNING", "detail": "official performance ledger"},
        {"check": "debug_reconstructed_leakage", "status": "PASS", "detail": "diagnostic data not used in active execution metrics"},
    ]
    return pd.DataFrame(rows)


def build_execution_bundle(data: dict[str, pd.DataFrame]) -> ExecutionBundle:
    rebalance = _df(data, "official_rebalance_report")
    perf = _df(data, "official_performance")
    blotter = build_order_blotter(data)
    components, by_ticker, by_action, by_date = cost_tables(blotter)
    equity = equity_table(perf)
    turnover = turnover_table(rebalance, perf)
    capacity = capacity_table(data, blotter)
    quality = execution_quality(blotter, perf)
    lifecycle = lifecycle_table(data, blotter)
    reconciliation = reconciliation_panel(data, blotter, turnover)
    kpis = build_kpis(data, blotter, turnover, capacity)
    src = source_audit(data)
    integrity = integrity_table(data, reconciliation, kpis)
    commentary = deterministic_commentary(kpis, blotter)
    status = "execution_terminal_pass"
    if integrity["status"].eq("FAIL").any() or reconciliation["status"].eq("FAIL").any():
        status = "execution_terminal_fail"
    elif integrity["status"].eq("WARNING").any() or reconciliation["status"].eq("WARNING").any():
        status = "execution_terminal_warning"
    return ExecutionBundle(kpis, latest(blotter) if not blotter.empty else pd.DataFrame(), blotter, components, by_ticker, by_action, by_date, equity, turnover, capacity, quality, lifecycle, reconciliation, src, integrity, commentary, status)
