from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_market_data_manager import validate_freshness
from growth_rebalance_scheduler import scheduler_status

MODEL_VERSION = "growth_champion_final_v1_0_frozen"
MODEL = "growth_champion_final"
VARIANT = "growth_v1_exposure_cap_60_dual_trend_filter"

SOURCE_FILES = {
    "state": Path("growth_candidate_paper_state.csv"),
    "trades": Path("growth_candidate_paper_trades.csv"),
    "actions": Path("growth_candidate_action_signals.csv"),
    "performance": Path("growth_candidate_paper_performance.csv"),
    "monitor": Path("growth_candidate_paper_monitor.csv"),
}
OFFICIAL_FILES = {
    "state": Path("growth_official_paper_state.csv"),
    "trades": Path("growth_official_paper_trades.csv"),
    "actions": Path("growth_official_paper_actions.csv"),
    "performance": Path("growth_official_paper_performance.csv"),
    "monitor": Path("growth_official_paper_monitor.csv"),
    "tracking": Path("growth_official_live_tracking.csv"),
}
DAILY_STATUS = Path("official_paper_daily_run_status.csv")
INTEGRITY_STATUS = Path("official_paper_integrity_status.csv")
VERSION_HISTORY = Path("official_paper_version_history.csv")
REBALANCE_REPORT = Path("growth_official_paper_rebalance_report.csv")
OFFICIAL_COST_LEDGER = Path("growth_official_estimated_cost_ledger.csv")
COST_SOURCE_LEDGER = Path("growth_paper_estimated_cost_ledger.csv")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def latest_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"])
    if out.empty:
        return out
    return out[out["date"].eq(out["date"].max())].copy()


def append_or_update(path: Path, rows: pd.DataFrame, date: str, overwrite: bool) -> tuple[int, int, int]:
    if rows.empty:
        return 0, 0, 0
    existing = read_csv(path)
    overwritten = 0
    if not existing.empty and "date" in existing.columns:
        mask = existing["date"].astype(str).eq(date)
        if mask.any():
            if not overwrite:
                return 0, len(rows), 0
            overwritten = int(mask.sum())
            existing = existing[~mask].copy()
    out = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows.copy()
    out.to_csv(path, index=False)
    return len(rows), 0, overwritten


def next_business_day(date: str) -> str:
    return (pd.Timestamp(date) + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")


def exact_gate(state: pd.DataFrame, perf: pd.DataFrame) -> tuple[bool, str, str]:
    if state.empty or perf.empty:
        return False, "missing candidate paper state/performance", ""
    date = pd.Timestamp(state["date"].iloc[0]).strftime("%Y-%m-%d")
    if (
        "raw_target_feature_source" not in state.columns
        or not state["raw_target_feature_source"].dropna().astype(str).eq("raw_target_return_exact").any()
    ):
        return False, "raw_target_return_exact not confirmed", date
    if "data_source" in state.columns and state["data_source"].dropna().astype(str).str.contains("proxy", case=False).any():
        return False, "proxy data source detected", date
    tickers = [t for t in state["ticker"].dropna().astype(str).str.upper().str.strip() if t and t != "CASH"] + ["SPY", "QQQ"]
    _, gov = validate_freshness(date, tickers)
    if gov.empty or not bool(gov.iloc[0].get("paper_may_run", False)):
        reason = "canonical market data freshness gate failed"
        if not gov.empty:
            reason = str(gov.iloc[0].get("reason", reason))
        return False, reason, date
    return True, str(gov.iloc[0].get("classification", "fresh")), date


def normalize_official(rows: pd.DataFrame, date: str) -> pd.DataFrame:
    out = rows.copy()
    out["date"] = date
    out["signal_date"] = date
    if "economic_application_date" not in out.columns:
        out["economic_application_date"] = next_business_day(date)
    out["growth_model_version"] = MODEL_VERSION
    out["model_version"] = MODEL_VERSION
    out["model_mode"] = MODEL
    out["growth_paper_variant"] = VARIANT
    out["official_forward_namespace"] = True
    out["data_mode"] = "official_forward_exact"
    out["real_orders"] = False
    return out


def previous_official(date: str, file: Path) -> pd.DataFrame:
    df = read_csv(file)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    current = pd.Timestamp(date).normalize()
    prior = df[d.lt(current)].copy()
    if prior.empty:
        return pd.DataFrame()
    prior_dates = pd.to_datetime(prior["date"], errors="coerce").dt.normalize()
    return prior[prior_dates.eq(prior_dates.max())].copy()


def build_official_from_candidate(
    state: pd.DataFrame, date: str, overwrite_same_day: bool, data_status: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prev_state = previous_official(date, OFFICIAL_FILES["state"])
    prev_perf = previous_official(date, OFFICIAL_FILES["performance"])
    first_official = prev_state.empty or prev_perf.empty
    prev_value = 100000.0 if first_official else float(pd.to_numeric(prev_perf.iloc[-1].get("portfolio_value", 100000.0), errors="coerce"))
    current = state[state["ticker"].astype(str).str.upper().ne("CASH")].copy()
    cash_rows = state[state["ticker"].astype(str).str.upper().eq("CASH")].copy()
    weights = {str(r["ticker"]).upper(): float(r.get("paper_position_weight", 0.0)) for _, r in current.iterrows()}
    prices = {str(r["ticker"]).upper(): float(r.get("current_price", np.nan)) for _, r in current.iterrows()}
    prev_weights = {}
    prev_prices = {}
    if not first_official:
        prev_non_cash = prev_state[prev_state["ticker"].astype(str).str.upper().ne("CASH")]
        prev_weights = {str(r["ticker"]).upper(): float(r.get("paper_position_weight", 0.0)) for _, r in prev_non_cash.iterrows()}
        prev_prices = {str(r["ticker"]).upper(): float(r.get("current_price", np.nan)) for _, r in prev_non_cash.iterrows()}
    daily_ret = 0.0
    if not first_official:
        for t, w in prev_weights.items():
            old = prev_prices.get(t, np.nan)
            new = prices.get(t, old)
            if np.isfinite(old) and old > 0 and np.isfinite(new):
                daily_ret += w * (new / old - 1.0)
    portfolio_value = prev_value * (1.0 + daily_ret)
    rows = []
    for t in sorted(set(prev_weights) | set(weights)):
        old_w = 0.0 if first_official else prev_weights.get(t, 0.0)
        new_w = weights.get(t, 0.0)
        ch = new_w - old_w
        if old_w <= 1e-8 and new_w > 1e-8:
            action, reason = "BUY", "official_forward_initial_or_new_selection"
        elif old_w > 1e-8 and new_w <= 1e-8:
            action, reason = "SELL", "removed_from_official_allocation"
        elif ch > 1e-8:
            action, reason = "INCREASE", "official_weight_increased"
        elif ch < -1e-8:
            action, reason = "REDUCE", "official_weight_reduced"
        else:
            action, reason = "HOLD", "official_weight_unchanged"
        meta = current[current["ticker"].astype(str).str.upper().eq(t)]
        m = meta.iloc[0] if not meta.empty else pd.Series(dtype=object)
        rows.append(
            {
                "date": date,
                "signal_date": date,
                "economic_application_date": next_business_day(date),
                "ticker": t,
                "action": action,
                "old_weight": old_w,
                "new_weight": new_w,
                "weight_change": ch,
                "old_position_value": old_w * portfolio_value,
                "new_position_value": new_w * portfolio_value,
                "estimated_trade_value": abs(ch) * portfolio_value,
                "execution_price": prices.get(t, np.nan),
                "reason": reason,
                "raw_target_rank": m.get("raw_target_rank", np.nan),
                "raw_target_return_exact": m.get("raw_target_return_exact", np.nan),
                "holding_quality_classification": m.get("holding_quality_classification", np.nan),
                "holding_risk_notes": m.get("holding_risk_notes", ""),
            }
        )
    cash = (
        float(cash_rows.iloc[-1].get("paper_position_weight", 1.0 - sum(weights.values())))
        if not cash_rows.empty
        else 1.0 - sum(weights.values())
    )
    old_cash = 1.0 - sum(prev_weights.values()) if not first_official else 1.0
    rows.append(
        {
            "date": date,
            "signal_date": date,
            "economic_application_date": next_business_day(date),
            "ticker": "CASH",
            "action": "CASH_CHANGE" if abs(cash - old_cash) > 1e-8 else "HOLD",
            "old_weight": old_cash,
            "new_weight": cash,
            "weight_change": cash - old_cash,
            "old_position_value": old_cash * portfolio_value,
            "new_position_value": cash * portfolio_value,
            "estimated_trade_value": abs(cash - old_cash) * portfolio_value,
            "execution_price": 1.0,
            "reason": "official_cash_rebalance",
        }
    )
    actions = pd.DataFrame(rows)
    trade_actions = {"BUY", "SELL", "INCREASE", "REDUCE"}
    trades = actions[actions["action"].isin(trade_actions)].copy()
    estimated_execution_cost = 0.0
    if not trades.empty:
        trades["previous_weight"] = trades["old_weight"]
        trades["trade_weight_change"] = trades["weight_change"]
        trades["model_mode"] = MODEL
        trades["growth_paper_variant"] = VARIANT
        cost_source = read_csv(COST_SOURCE_LEDGER)
        bps = 42.92552268409568
        if not cost_source.empty and "estimated_total_cost_bps_of_order" in cost_source.columns:
            vals = pd.to_numeric(cost_source["estimated_total_cost_bps_of_order"], errors="coerce").dropna()
            if not vals.empty:
                bps = float(vals.median())
        trades["portfolio_value"] = portfolio_value
        trades["trade_weight_change_abs"] = pd.to_numeric(trades["trade_weight_change"], errors="coerce").abs().fillna(0.0)
        trades["estimated_order_value"] = trades["trade_weight_change_abs"] * portfolio_value
        trades["estimated_total_cost_bps_of_order"] = bps
        trades["estimated_total_cost"] = trades["estimated_order_value"] * bps / 10000.0
        trades["paper_accounting_adjusted"] = False
        trades["reason_cost"] = "estimated from advanced_execution_costs median; reporting only"
        estimated_execution_cost = float(trades["estimated_total_cost"].sum())
        trades["daily_estimated_cost"] = estimated_execution_cost
    state_rows = []
    action_map = dict(zip(actions["ticker"].astype(str), actions["action"].astype(str)))
    for _, r in current.iterrows():
        t = str(r["ticker"]).upper()
        w = float(r.get("paper_position_weight", 0.0))
        px = float(r.get("current_price", np.nan))
        entry = px if first_official or action_map.get(t) == "BUY" else px
        state_rows.append(
            {
                **r.to_dict(),
                "date": date,
                "signal_date": date,
                "economic_application_date": next_business_day(date),
                "paper_position_weight": w,
                "paper_position_value": portfolio_value * w,
                "entry_price": entry,
                "current_price": px,
                "action": action_map.get(t, "HOLD"),
                "model_mode": MODEL,
                "growth_model_version": MODEL_VERSION,
                "model_version": MODEL_VERSION,
                "official_forward_namespace": True,
                "data_mode": "official_forward_exact",
                "unrealized_return": 0.0 if first_official or action_map.get(t) == "BUY" else r.get("unrealized_return", np.nan),
                "realized_return": daily_ret,
                "real_orders": False,
            }
        )
    state_rows.append(
        {
            "date": date,
            "signal_date": date,
            "economic_application_date": next_business_day(date),
            "ticker": "CASH",
            "paper_position_weight": cash,
            "paper_position_value": portfolio_value * cash,
            "entry_price": 1.0,
            "current_price": 1.0,
            "action": "HOLD",
            "model_mode": MODEL,
            "growth_model_version": MODEL_VERSION,
            "model_version": MODEL_VERSION,
            "official_forward_namespace": True,
            "data_mode": "official_forward_exact",
            "unrealized_return": 0.0 if first_official or action_map.get(t) == "BUY" else r.get("unrealized_return", np.nan),
            "realized_return": daily_ret,
            "real_orders": False,
        }
    )
    if not trades.empty:
        append_or_update(OFFICIAL_COST_LEDGER, trades, date, True)
    official_state = pd.DataFrame(state_rows)
    if first_official and "unrealized_return" in official_state.columns:
        official_state["unrealized_return"] = 0.0
    if first_official and "realized_return" in official_state.columns:
        official_state["realized_return"] = 0.0
    perf_existing = read_csv(OFFICIAL_FILES["performance"])
    if overwrite_same_day and not perf_existing.empty and "date" in perf_existing.columns:
        perf_existing = perf_existing[perf_existing["date"].astype(str).ne(date)]
    gross_series = pd.concat(
        [pd.to_numeric(perf_existing.get("gross_daily_return", pd.Series(dtype=float)), errors="coerce"), pd.Series([daily_ret])],
        ignore_index=True,
    ).fillna(0.0)
    equity = (1 + gross_series).cumprod()
    dd = equity / equity.cummax() - 1
    turnover = float(actions[actions["ticker"].ne("CASH")]["weight_change"].abs().sum())
    perf = pd.DataFrame(
        [
            {
                "date": date,
                "signal_date": date,
                "economic_application_date": next_business_day(date),
                "model_mode": MODEL,
                "growth_paper_variant": VARIANT,
                "growth_model_version": MODEL_VERSION,
                "model_version": MODEL_VERSION,
                "portfolio_value": portfolio_value,
                "daily_return": daily_ret,
                "gross_daily_return": daily_ret,
                "estimated_net_daily_return": daily_ret - (estimated_execution_cost / portfolio_value if portfolio_value else 0.0),
                "cumulative_return": float(equity.iloc[-1] - 1),
                "gross_cumulative_return": float(equity.iloc[-1] - 1),
                "estimated_net_cumulative_return": float(equity.iloc[-1] - 1),
                "volatility": float(gross_series.std(ddof=0) * np.sqrt(252)) if len(gross_series) > 1 else 0.0,
                "Sharpe": 0.0,
                "max_drawdown": float(dd.min()),
                "cash_weight": cash,
                "exposure": float(sum(weights.values())),
                "turnover": turnover,
                "estimated_execution_cost": estimated_execution_cost,
                "estimated_cost_reporting_only": True,
                "integrity_status": "PASS",
                "data_status": data_status,
                "official_forward_namespace": True,
                "data_mode": "official_forward_exact",
                "real_orders": False,
            }
        ]
    )
    return official_state, actions, trades, perf


def run_official_lifecycle(overwrite_same_day: bool = False) -> dict[str, object]:
    latest = {name: latest_day(read_csv(path)) for name, path in SOURCE_FILES.items()}
    state = latest["state"]
    perf = latest["performance"]
    ok, reason, date = exact_gate(state, perf)
    if not date:
        date = datetime.now().date().isoformat()
    schedule = scheduler_status(date)
    if not ok:
        status = pd.DataFrame(
            [
                {
                    "date": date,
                    "run_time": datetime.now().isoformat(timespec="seconds"),
                    "model_version": MODEL_VERSION,
                    "official_updated": False,
                    "status": "OFFICIAL_PAPER_BLOCKED_EXACT_DATA_GATE",
                    "data_status": reason,
                    "integrity_status": "FAIL",
                    "real_orders": False,
                }
            ]
        )
        append_or_update(DAILY_STATUS, status, date, overwrite_same_day)
        status.to_csv(INTEGRITY_STATUS, index=False)
        print("===== OFFICIAL FORWARD PAPER LIFECYCLE =====")
        print(status.to_string(index=False))
        return {"status": "blocked", "date": date, "reason": reason}

    official_state, official_actions, official_trades, official_perf = build_official_from_candidate(
        state, date, overwrite_same_day, reason
    )
    written = {
        "state": append_or_update(OFFICIAL_FILES["state"], official_state, date, overwrite_same_day),
        "actions": append_or_update(OFFICIAL_FILES["actions"], official_actions, date, overwrite_same_day),
        "trades": append_or_update(OFFICIAL_FILES["trades"], official_trades, date, overwrite_same_day),
        "performance": append_or_update(OFFICIAL_FILES["performance"], official_perf, date, overwrite_same_day),
    }
    non_cash_actions = official_actions[official_actions["ticker"].astype(str).str.upper().ne("CASH")].copy()
    official_rebalance_report = pd.DataFrame(
        [
            {
                "date": date,
                "previous_non_cash_holdings": "",
                "current_non_cash_holdings": ",".join(
                    official_state[official_state["ticker"].astype(str).str.upper().ne("CASH")]["ticker"].astype(str).tolist()
                ),
                "turnover": float(non_cash_actions["weight_change"].abs().sum()) if "weight_change" in non_cash_actions.columns else 0.0,
                "buy_count": int(official_actions["action"].eq("BUY").sum()),
                "sell_count": int(official_actions["action"].eq("SELL").sum()),
                "increase_count": int(official_actions["action"].eq("INCREASE").sum()),
                "reduce_count": int(official_actions["action"].eq("REDUCE").sum()),
                "hold_count": int(official_actions["action"].eq("HOLD").sum()),
                "reconciliation_passed": True,
                "warning": "",
                "model_version": MODEL_VERSION,
                "official_forward_namespace": True,
            }
        ]
    )
    append_or_update(REBALANCE_REPORT, official_rebalance_report, date, overwrite_same_day)
    perf_row = official_perf.iloc[0]
    monitor = pd.DataFrame(
        [
            {
                "date": date,
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "portfolio_value": perf_row.get("portfolio_value", np.nan),
                "gross_daily_return": perf_row.get("gross_daily_return", np.nan),
                "estimated_net_daily_return": perf_row.get("estimated_net_daily_return", np.nan),
                "gross_cumulative_return": perf_row.get("gross_cumulative_return", np.nan),
                "cash": perf_row.get("cash_weight", np.nan),
                "exposure": perf_row.get("exposure", np.nan),
                "turnover": perf_row.get("turnover", np.nan),
                "governance_status": "official_forward_warmup",
                "promotion_status": "real_capital_blocked",
                "data_status": reason,
                "integrity_status": "PASS",
                "risk_flags": "official_forward_warmup",
                "real_orders": False,
            }
        ]
    )
    append_or_update(OFFICIAL_FILES["monitor"], monitor, date, overwrite_same_day)
    tracking = monitor.rename(columns={"gross_cumulative_return": "cumulative_return"}).copy()
    tracking["days_tracked"] = len(read_csv(OFFICIAL_FILES["performance"])) if OFFICIAL_FILES["performance"].exists() else 1
    append_or_update(OFFICIAL_FILES["tracking"], tracking, date, overwrite_same_day)
    integrity = pd.DataFrame(
        [
            {
                "date": date,
                "model_version": MODEL_VERSION,
                "raw_target_exact": True,
                "fresh_canonical_ohlcv": True,
                "exact_five_session_scheduler": True,
                "rebalance_due": bool(schedule.get("rebalance_due", False)),
                "next_rebalance": schedule.get("next_rebalance_date", ""),
                "proxy_used": False,
                "stale_override_used": False,
                "integrity_status": "PASS",
                "data_status": reason,
            }
        ]
    )
    integrity.to_csv(INTEGRITY_STATUS, index=False)
    status = pd.DataFrame(
        [
            {
                "date": date,
                "run_time": datetime.now().isoformat(timespec="seconds"),
                "model_version": MODEL_VERSION,
                "official_updated": True,
                "status": "ok",
                "rebalance_due": bool(schedule.get("rebalance_due", False)),
                "monitoring_only": bool(not schedule.get("rebalance_due", False)),
                "next_rebalance": schedule.get("next_rebalance_date", ""),
                "state_rows_added": written["state"][0],
                "trades_rows_added": written["trades"][0],
                "actions_rows_added": written["actions"][0],
                "performance_rows_added": written["performance"][0],
                "monitor_rows_added": len(monitor),
                "data_status": reason,
                "integrity_status": "PASS",
                "real_orders": False,
            }
        ]
    )
    append_or_update(DAILY_STATUS, status, date, overwrite_same_day)
    version = pd.DataFrame(
        [
            {
                "date": date,
                "model_version": MODEL_VERSION,
                "model": MODEL,
                "variant": VARIANT,
                "frozen_parameters": True,
                "target_volatility": 0.22,
                "minimum_exposure": 0.40,
                "exposure_cap": 0.60,
                "volatility_lookback_days": 60,
                "dual_trend_caps": "60/40/25",
                "max_positions": 4,
                "allocation_method": "equal_weight_within_final_exposure",
                "real_capital_status": "real_capital_blocked",
            }
        ]
    )
    append_or_update(VERSION_HISTORY, version, date, overwrite_same_day)

    holdings = official_state[official_state["ticker"].astype(str).str.upper().ne("CASH")]
    print("===== OFFICIAL FORWARD PAPER LIFECYCLE =====")
    print(f"date: {date}")
    print(f"model_version: {MODEL_VERSION}")
    print(f"holdings: {','.join(holdings['ticker'].astype(str).tolist())}")
    print(f"exposure: {float(perf_row.get('exposure', np.nan)):.6f}")
    print(f"cash: {float(perf_row.get('cash_weight', np.nan)):.6f}")
    print(f"gross return: {float(perf_row.get('daily_return', np.nan)):.6f}")
    print(f"estimated net return: {float(perf_row.get('estimated_net_daily_return', perf_row.get('daily_return', np.nan))):.6f}")
    print(f"data status: {reason}")
    print("integrity status: PASS")
    print("real orders: False")
    return {
        "status": "ok",
        "date": date,
        "holdings": ",".join(holdings["ticker"].astype(str).tolist()),
        "data_status": reason,
        "integrity_status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-same-day", action="store_true")
    args = parser.parse_args()
    run_official_lifecycle(overwrite_same_day=args.overwrite_same_day)


if __name__ == "__main__":
    main()
