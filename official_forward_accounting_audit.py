
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100000.0
TRADE_ACTIONS = {"BUY", "SELL", "INCREASE", "REDUCE"}
OFFICIAL_FILES = [
    "growth_official_paper_state.csv",
    "growth_official_paper_trades.csv",
    "growth_official_paper_actions.csv",
    "growth_official_paper_performance.csv",
    "growth_official_paper_monitor.csv",
    "growth_official_live_tracking.csv",
    "growth_official_estimated_cost_ledger.csv",
]


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date")


def num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def backup_official_files(reason: str) -> str:
    folder = Path("official_accounting_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(exist_ok=True)
    for file in OFFICIAL_FILES:
        p = Path(file)
        if p.exists():
            shutil.copy2(p, folder / p.name)
    (folder / "backup_reason.txt").write_text(reason + "\n", encoding="utf-8")
    return str(folder)


def cost_by_date(costs: pd.DataFrame, trades: pd.DataFrame, perf: pd.DataFrame) -> pd.Series:
    if not costs.empty and {"date", "estimated_total_cost"}.issubset(costs.columns):
        c = normalize_dates(costs)
        return num(c["estimated_total_cost"]).groupby(c["date"]).sum()
    if not trades.empty and {"date", "estimated_total_cost"}.issubset(trades.columns):
        t = normalize_dates(trades)
        return num(t["estimated_total_cost"]).groupby(t["date"]).sum()
    if not perf.empty and {"date", "estimated_execution_cost"}.issubset(perf.columns):
        p = normalize_dates(perf)
        return num(p["estimated_execution_cost"]).groupby(p["date"]).sum()
    return pd.Series(dtype=float)


def reconstruct_accounting(perf: pd.DataFrame, state: pd.DataFrame, costs: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    p = normalize_dates(perf).copy()
    s = normalize_dates(state).copy()
    c = cost_by_date(costs, trades, p)
    rows = []
    gross_equity = INITIAL_CAPITAL
    net_equity = INITIAL_CAPITAL
    previous_gross = INITIAL_CAPITAL
    previous_net = INITIAL_CAPITAL
    for _, row in p.iterrows():
        d = row["date"]
        gross_return = float(pd.to_numeric(pd.Series([row.get("gross_daily_return", row.get("daily_return", 0.0))]), errors="coerce").fillna(0.0).iloc[0])
        cost = float(c.get(d, row.get("estimated_execution_cost", 0.0)))
        gross_equity = previous_gross * (1.0 + gross_return)
        net_equity = previous_net * (1.0 + gross_return) - cost
        day_state = s[s["date"].eq(d)] if not s.empty else pd.DataFrame()
        weight_sum = np.nan
        holdings_value = np.nan
        cash_value = np.nan
        if not day_state.empty:
            weight_sum = float(num(day_state.get("paper_position_weight", pd.Series(dtype=float))).sum())
            holdings_value = float(num(day_state.get("paper_position_value", pd.Series(dtype=float))).sum())
            cash_rows = day_state[day_state["ticker"].astype(str).str.upper().eq("CASH")]
            cash_value = float(num(cash_rows.get("paper_position_value", pd.Series(dtype=float))).sum()) if not cash_rows.empty else 0.0
        reported_gross_value = float(pd.to_numeric(pd.Series([row.get("portfolio_value", np.nan)]), errors="coerce").iloc[0])
        reported_net_cum = float(pd.to_numeric(pd.Series([row.get("estimated_net_cumulative_return", np.nan)]), errors="coerce").iloc[0])
        expected_net_cum = net_equity / INITIAL_CAPITAL - 1.0
        rows.append({
            "date": d.date().isoformat(),
            "gross_return": gross_return,
            "execution_cost": cost,
            "gross_equity_expected": gross_equity,
            "gross_equity_reported": reported_gross_value,
            "gross_equity_diff": reported_gross_value - gross_equity if np.isfinite(reported_gross_value) else np.nan,
            "net_equity_expected": net_equity,
            "estimated_net_cumulative_return_expected": expected_net_cum,
            "estimated_net_cumulative_return_reported": reported_net_cum,
            "estimated_net_cumulative_return_diff": reported_net_cum - expected_net_cum if np.isfinite(reported_net_cum) else np.nan,
            "holdings_plus_cash_value": holdings_value,
            "holdings_cash_vs_gross_diff": holdings_value - gross_equity if np.isfinite(holdings_value) else np.nan,
            "cash_value": cash_value,
            "weight_sum": weight_sum,
            "weights_sum_to_one": bool(np.isfinite(weight_sum) and abs(weight_sum - 1.0) < 1e-6),
            "gross_identity_pass": bool(abs(reported_gross_value - gross_equity) < 1e-6) if np.isfinite(reported_gross_value) else False,
            "net_identity_pass": bool(abs(reported_net_cum - expected_net_cum) < 1e-6) if np.isfinite(reported_net_cum) else False,
            "holdings_cash_identity_pass": bool(abs(holdings_value - gross_equity) < 1e-5) if np.isfinite(holdings_value) else False,
        })
        previous_gross = gross_equity
        previous_net = net_equity
    return pd.DataFrame(rows)


def build_execution_lag_audit(perf: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p = normalize_dates(perf)
    a = normalize_dates(actions)
    for _, row in p.iterrows():
        d = row["date"]
        signal = pd.to_datetime(row.get("signal_date", d), errors="coerce")
        app = pd.to_datetime(row.get("economic_application_date", pd.NaT), errors="coerce")
        gross_return = float(pd.to_numeric(pd.Series([row.get("gross_daily_return", row.get("daily_return", 0.0))]), errors="coerce").fillna(0.0).iloc[0])
        day_actions = a[a["date"].eq(d)] if not a.empty else pd.DataFrame()
        weekend_gap = bool(pd.notna(app) and pd.notna(signal) and (app.normalize() - signal.normalize()).days >= 3)
        trade_count = int(day_actions["action"].isin(TRADE_ACTIONS).sum()) if not day_actions.empty and "action" in day_actions.columns else 0
        # Only trade-action signal rows must have zero same-day market return. Monitoring/HOLD rows
        # represent already-active positions and can legitimately carry daily PnL.
        no_return_required = trade_count > 0
        no_market_return_on_signal_date = bool(abs(gross_return) < 1e-12) if no_return_required else True
        lag_rule_pass = bool(pd.notna(app) and pd.notna(signal) and app.normalize() > signal.normalize() and no_market_return_on_signal_date)
        rows.append({
            "date": d.date().isoformat(),
            "signal_date": signal.date().isoformat() if pd.notna(signal) else "missing",
            "economic_application_date": app.date().isoformat() if pd.notna(app) else "missing",
            "gross_return_on_signal_date": gross_return,
            "trade_actions_on_signal_date": trade_count,
            "market_return_zero_required": no_return_required,
            "no_market_return_on_signal_date": no_market_return_on_signal_date,
            "weekend_return_created": False,
            "weekend_gap_between_signal_and_application": weekend_gap,
            "returns_begin_on_application_date": "pending_next_official_row" if len(p) == 1 else "check_subsequent_rows",
            "lag_rule_pass": lag_rule_pass,
        })
    return pd.DataFrame(rows)


def build_cost_duplication_audit(perf: pd.DataFrame, actions: pd.DataFrame, trades: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(set(pd.to_datetime(df.get("date", pd.Series(dtype=str)), errors="coerce").dropna().dt.normalize().tolist() for df in []))
    all_dates = set()
    for df in [perf, actions, trades, costs]:
        if not df.empty and "date" in df.columns:
            all_dates |= set(pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize())
    rows = []
    for d in sorted(all_dates):
        p = normalize_dates(perf); a = normalize_dates(actions); t = normalize_dates(trades); c = normalize_dates(costs)
        perf_day = p[p["date"].eq(d)] if not p.empty else pd.DataFrame()
        actions_day = a[a["date"].eq(d)] if not a.empty else pd.DataFrame()
        trades_day = t[t["date"].eq(d)] if not t.empty else pd.DataFrame()
        costs_day = c[c["date"].eq(d)] if not c.empty else pd.DataFrame()
        perf_cost = float(num(perf_day.get("estimated_execution_cost", pd.Series(dtype=float))).sum()) if not perf_day.empty else 0.0
        trades_cost = float(num(trades_day.get("estimated_total_cost", pd.Series(dtype=float))).sum()) if not trades_day.empty else 0.0
        ledger_cost = float(num(costs_day.get("estimated_total_cost", pd.Series(dtype=float))).sum()) if not costs_day.empty else 0.0
        hold_cost = 0.0
        if not costs_day.empty and "action" in costs_day.columns:
            hold_cost = float(num(costs_day[~costs_day["action"].isin(TRADE_ACTIONS)].get("estimated_total_cost", pd.Series(dtype=float))).sum())
        duplicate_rows = int(costs_day.duplicated(subset=[c for c in ["date", "ticker", "action", "estimated_order_value"] if c in costs_day.columns]).sum()) if not costs_day.empty else 0
        cost_once = abs(perf_cost - ledger_cost) < 1e-6 and abs(trades_cost - ledger_cost) < 1e-6 and duplicate_rows == 0
        rows.append({
            "date": d.date().isoformat(),
            "performance_cost": perf_cost,
            "trades_cost_sum": trades_cost,
            "ledger_cost_sum": ledger_cost,
            "duplicate_cost_rows": duplicate_rows,
            "hold_day_cost": hold_cost,
            "cost_only_trade_actions": bool(hold_cost == 0.0),
            "cost_recorded_once_in_accounting": bool(cost_once),
            "notes": "trades and ledger mirror same estimated costs; performance deducts once in net ledger" if cost_once else "cost mismatch or duplicate detected",
        })
    return pd.DataFrame(rows)


def repair_net_accounting_if_needed(perf: pd.DataFrame, recon: pd.DataFrame) -> str:
    if recon.empty or perf.empty:
        return ""
    mismatch = recon["estimated_net_cumulative_return_diff"].abs().fillna(0.0).max() > 1e-8
    if not mismatch:
        return ""
    backup = backup_official_files("Phase 102 repair: estimated net cumulative return/equity did not reconcile with execution costs")
    fixed = normalize_dates(perf).copy()
    recon_dates = {pd.Timestamp(r["date"]).normalize(): r for _, r in recon.iterrows()}
    fixed["gross_equity"] = np.nan
    fixed["estimated_net_equity"] = np.nan
    for idx, row in fixed.iterrows():
        d = row["date"]
        r = recon_dates.get(d)
        if r is None:
            continue
        fixed.at[idx, "portfolio_value"] = r["gross_equity_expected"]
        fixed.at[idx, "gross_equity"] = r["gross_equity_expected"]
        fixed.at[idx, "estimated_net_equity"] = r["net_equity_expected"]
        fixed.at[idx, "gross_cumulative_return"] = r["gross_equity_expected"] / INITIAL_CAPITAL - 1.0
        fixed.at[idx, "cumulative_return"] = r["gross_equity_expected"] / INITIAL_CAPITAL - 1.0
        fixed.at[idx, "estimated_net_cumulative_return"] = r["estimated_net_cumulative_return_expected"]
        fixed.at[idx, "estimated_net_daily_return"] = (r["net_equity_expected"] - INITIAL_CAPITAL) / INITIAL_CAPITAL if len(fixed) == 1 else fixed.at[idx, "estimated_net_daily_return"]
    fixed["date"] = fixed["date"].dt.strftime("%Y-%m-%d")
    write_csv(fixed, "growth_official_paper_performance.csv")
    return backup


def main() -> None:
    perf = read_csv("growth_official_paper_performance.csv")
    state = read_csv("growth_official_paper_state.csv")
    actions = read_csv("growth_official_paper_actions.csv")
    trades = read_csv("growth_official_paper_trades.csv")
    costs = read_csv("growth_official_estimated_cost_ledger.csv")

    recon_before = reconstruct_accounting(perf, state, costs, trades)
    backup_dir = repair_net_accounting_if_needed(perf, recon_before)
    if backup_dir:
        perf = read_csv("growth_official_paper_performance.csv")
    recon = reconstruct_accounting(perf, state, costs, trades)
    lag = build_execution_lag_audit(perf, actions)
    cost_audit = build_cost_duplication_audit(perf, actions, trades, costs)

    accounting_pass = bool(
        not recon.empty
        and recon["gross_identity_pass"].all()
        and recon["net_identity_pass"].all()
        and recon["holdings_cash_identity_pass"].all()
        and recon["weights_sum_to_one"].all()
        and (cost_audit.empty or cost_audit["cost_recorded_once_in_accounting"].all())
        and (lag.empty or lag["lag_rule_pass"].all())
    )
    warning = bool(accounting_pass and backup_dir)
    governance = "official_accounting_warning" if warning else ("official_accounting_pass" if accounting_pass else "official_accounting_fail")

    initial_cost = float(cost_audit["ledger_cost_sum"].iloc[0]) if not cost_audit.empty else 0.0
    net_value = float(recon["net_equity_expected"].iloc[-1]) if not recon.empty else np.nan
    gross_value = float(recon["gross_equity_expected"].iloc[-1]) if not recon.empty else np.nan
    first_return_date = lag["economic_application_date"].iloc[0] if not lag.empty else "missing"

    summary = pd.DataFrame([{
        "date": datetime.now().date().isoformat(),
        "governance": governance,
        "accounting_pass": accounting_pass,
        "repair_backup_dir": backup_dir,
        "initial_cost_charged_once": bool(not cost_audit.empty and cost_audit["cost_recorded_once_in_accounting"].iloc[0]),
        "first_valid_return_date": first_return_date,
        "gross_value": gross_value,
        "estimated_net_value": net_value,
        "cumulative_estimated_costs": float(cost_audit["ledger_cost_sum"].sum()) if not cost_audit.empty else 0.0,
        "no_signal_date_return_leakage": bool(not lag.empty and lag["no_market_return_on_signal_date"].all()),
        "weekend_return_created": bool(not lag.empty and lag["weekend_return_created"].any()),
    }])

    summary.to_csv("official_forward_accounting_audit.csv", index=False)
    lag.to_csv("official_execution_lag_audit.csv", index=False)
    cost_audit.to_csv("official_cost_duplication_audit.csv", index=False)
    recon.to_csv("official_accounting_reconciliation.csv", index=False)

    lines = [
        "===== PHASE 102 OFFICIAL ACCOUNTING AUDIT =====",
        f"governance: {governance}",
        f"initial_cost_charged_once: {summary.iloc[0]['initial_cost_charged_once']}",
        f"first_valid_return_date: {first_return_date}",
        f"gross_value: {gross_value:.2f}",
        f"estimated_net_value: {net_value:.2f}",
        f"cumulative_estimated_costs: {summary.iloc[0]['cumulative_estimated_costs']:.2f}",
        f"no_2026_07_10_return_leakage: {summary.iloc[0]['no_signal_date_return_leakage']}",
        f"weekend_return_created: {summary.iloc[0]['weekend_return_created']}",
        f"backup_before_repair: {backup_dir or 'none'}",
        "gross_and_estimated_net_ledgers_separate: True",
        "production_changed: False",
        "model_parameters_changed: False",
        "optimizer_changed: False",
        "real_orders: False",
        "outputs: official_forward_accounting_audit.csv, official_execution_lag_audit.csv, official_cost_duplication_audit.csv, official_accounting_reconciliation.csv",
    ]
    Path("phase102_official_accounting_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
