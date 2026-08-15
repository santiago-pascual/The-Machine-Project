from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100000.0
QUALITY_FILE = "growth_exact_input_availability_audit.csv"
DEBUG_PREFIX = "growth_historical_debug_reconstruction"
OFFICIAL_STATE = "growth_official_paper_state.csv"
OFFICIAL_TRADES = "growth_official_paper_trades.csv"
OFFICIAL_ACTIONS = "growth_official_paper_actions.csv"
OFFICIAL_PERF = "growth_official_paper_performance.csv"
OFFICIAL_MONITOR = "growth_official_paper_monitor.csv"
OFFICIAL_TRACKING = "growth_official_live_tracking.csv"
REPORT = "official_paper_baseline_report.txt"
DATA_QUALITY = "official_paper_data_quality.csv"
COMPARE = "official_vs_debug_history.csv"

SOURCE_FILES = {
    "state": "growth_candidate_paper_state.csv",
    "trades": "growth_candidate_paper_trades.csv",
    "actions": "growth_candidate_action_signals.csv",
    "performance": "growth_candidate_paper_performance.csv",
    "monitor": "growth_candidate_paper_monitor.csv",
}


def read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def dateify(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date")


def determine_exact_live_start() -> tuple[pd.Timestamp, pd.DataFrame]:
    q = dateify(read(QUALITY_FILE))
    perf = dateify(read(SOURCE_FILES["performance"]))
    monitor = dateify(read(SOURCE_FILES["monitor"]))
    schedule = read("growth_rebalance_schedule.csv")
    if not schedule.empty and "market_date" in schedule.columns:
        schedule = schedule.rename(columns={"market_date": "date"})
    schedule = dateify(schedule)
    if q.empty or perf.empty:
        raise ValueError("Missing exact input audit or paper performance history")

    base_cols = ["date"]
    for col in ["raw_target_feature_source", "data_source"]:
        if col in perf.columns:
            base_cols.append(col)
    merged = perf[base_cols].copy()

    if "rebalance_due" in perf.columns:
        merged["rebalance_due"] = perf["rebalance_due"].values
    elif not monitor.empty and "rebalance_due" in monitor.columns:
        merged = merged.merge(monitor[["date", "rebalance_due"]].drop_duplicates("date"), on="date", how="left")
    elif not schedule.empty and "rebalance_due" in schedule.columns:
        merged = merged.merge(schedule[["date", "rebalance_due"]].drop_duplicates("date"), on="date", how="left")
    else:
        merged["rebalance_due"] = False

    merged = merged.merge(q, on="date", how="left")
    for col in ["raw_target_return_exact", "fresh_ohlcv", "current_filters_present", "fresh_volatility_calculation", "exact_rebalance_scheduler", "all_exact_inputs"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            merged[col] = False
    merged["rebalance_due_bool"] = merged["rebalance_due"].astype(str).str.lower().isin(["true", "1", "yes"])
    merged["no_carry_forward_bridge"] = merged["rebalance_due_bool"] & merged["all_exact_inputs"]
    exact_rebalances = merged[merged["no_carry_forward_bridge"]]
    if exact_rebalances.empty:
        raise ValueError("No exact rebalance date available for official forward baseline")
    start = exact_rebalances["date"].min().normalize()
    merged["official_eligible"] = merged["date"].ge(start) & merged["all_exact_inputs"]
    return start, merged


def archive_debug(start: pd.Timestamp) -> dict[str, int]:
    counts = {}
    for name, path in SOURCE_FILES.items():
        df = dateify(read(path))
        if df.empty:
            counts[name] = 0
            continue
        debug = df[df["date"].lt(start)].copy()
        debug["data_mode"] = "historical_debug_reconstruction"
        debug["archive_reason"] = "before exact official forward start; useful for debugging only"
        out = f"{DEBUG_PREFIX}_{name}.csv"
        debug.to_csv(out, index=False)
        counts[name] = len(debug)
    return counts


def official_performance(start: pd.Timestamp) -> pd.DataFrame:
    perf = dateify(read(SOURCE_FILES["performance"]))
    if perf.empty:
        return pd.DataFrame()
    official = perf[perf["date"].ge(start)].copy().reset_index(drop=True)
    if official.empty:
        return official
    official["data_mode"] = "official_forward_paper_exact"
    official["official_start_date"] = start.strftime("%Y-%m-%d")
    official["official_initial_capital"] = INITIAL_CAPITAL
    official["debug_source_daily_return"] = pd.to_numeric(official.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    official["daily_return"] = official["debug_source_daily_return"]
    official.loc[0, "daily_return"] = 0.0
    equity = INITIAL_CAPITAL * (1.0 + official["daily_return"]).cumprod()
    official["portfolio_value"] = equity
    official["cumulative_return"] = equity / INITIAL_CAPITAL - 1.0
    returns = official["daily_return"]
    vol = returns.expanding().std(ddof=0).fillna(0.0) * np.sqrt(252)
    official["volatility"] = vol
    official["Sharpe"] = np.where(vol > 0, returns.expanding().mean() * 252 / vol, 0.0)
    dd = equity / equity.cummax() - 1.0
    official["max_drawdown"] = dd.expanding().min().fillna(0.0)
    official["growth_model_version"] = "growth_champion_final_v1_0_frozen"
    official["official_forward_status"] = "official_forward_warmup"
    official.to_csv(OFFICIAL_PERF, index=False)
    return official


def official_state(start: pd.Timestamp, official_perf: pd.DataFrame) -> pd.DataFrame:
    state = dateify(read(SOURCE_FILES["state"]))
    if state.empty or official_perf.empty:
        return pd.DataFrame()
    official = state[state["date"].ge(start)].copy()
    values = official_perf.set_index("date")["portfolio_value"].to_dict()
    official["paper_position_weight"] = pd.to_numeric(official["paper_position_weight"], errors="coerce").fillna(0.0)
    official["paper_position_value"] = official.apply(lambda r: values.get(r["date"], INITIAL_CAPITAL) * r["paper_position_weight"], axis=1)
    official["data_mode"] = "official_forward_paper_exact"
    official["official_start_date"] = start.strftime("%Y-%m-%d")
    official["growth_model_version"] = "growth_champion_final_v1_0_frozen"
    official.to_csv(OFFICIAL_STATE, index=False)
    return official


def start_reset_actions(start: pd.Timestamp, state_start: pd.DataFrame) -> pd.DataFrame:
    rows = []
    non_cash = state_start[state_start["ticker"].astype(str).ne("CASH")].copy()
    cash = state_start[state_start["ticker"].astype(str).eq("CASH")]
    for _, row in non_cash.iterrows():
        w = float(pd.to_numeric(pd.Series([row.get("paper_position_weight", 0.0)]), errors="coerce").fillna(0).iloc[0])
        rows.append({
            "date": start,
            "ticker": row.get("ticker", ""),
            "action": "BUY",
            "old_weight": 0.0,
            "new_weight": w,
            "weight_change": w,
            "old_position_value": 0.0,
            "new_position_value": INITIAL_CAPITAL * w,
            "estimated_trade_value": INITIAL_CAPITAL * w,
            "reason": "official_forward_baseline_reset_initial_position",
            "data_mode": "official_forward_paper_exact",
            "growth_model_version": "growth_champion_final_v1_0_frozen",
        })
    cash_w = float(pd.to_numeric(cash.get("paper_position_weight", pd.Series([1.0])), errors="coerce").fillna(1.0).iloc[0]) if not cash.empty else 1.0 - sum(r["new_weight"] for r in rows)
    rows.append({
        "date": start,
        "ticker": "CASH",
        "action": "CASH_CHANGE",
        "old_weight": 1.0,
        "new_weight": cash_w,
        "weight_change": cash_w - 1.0,
        "old_position_value": INITIAL_CAPITAL,
        "new_position_value": INITIAL_CAPITAL * cash_w,
        "estimated_trade_value": abs(cash_w - 1.0) * INITIAL_CAPITAL,
        "reason": "official_forward_baseline_reset_cash",
        "data_mode": "official_forward_paper_exact",
        "growth_model_version": "growth_champion_final_v1_0_frozen",
    })
    return pd.DataFrame(rows)


def official_actions_and_trades(start: pd.Timestamp, official_state_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = dateify(read(SOURCE_FILES["actions"]))
    trades = dateify(read(SOURCE_FILES["trades"]))
    state_start = official_state_df[official_state_df["date"].eq(start)].copy()
    reset = start_reset_actions(start, state_start)
    future_actions = actions[actions["date"].gt(start)].copy() if not actions.empty else pd.DataFrame()
    future_actions["data_mode"] = "official_forward_paper_exact"
    official_actions = pd.concat([reset, future_actions], ignore_index=True, sort=False)
    official_actions.to_csv(OFFICIAL_ACTIONS, index=False)
    trade_actions = official_actions[official_actions["action"].isin(["BUY", "SELL", "INCREASE", "REDUCE"])].copy()
    if not trade_actions.empty:
        trade_actions["previous_weight"] = trade_actions.get("old_weight", 0.0)
        trade_actions["new_weight"] = trade_actions.get("new_weight", 0.0)
        trade_actions["trade_weight_change"] = trade_actions.get("weight_change", 0.0)
        trade_actions["model_mode"] = "growth_champion_final"
        trade_actions["growth_paper_variant"] = "growth_v1_exposure_cap_60_dual_trend_filter"
    trade_actions.to_csv(OFFICIAL_TRADES, index=False)
    return official_actions, trade_actions


def official_monitor(start: pd.Timestamp, official_perf: pd.DataFrame, official_state_df: pd.DataFrame) -> pd.DataFrame:
    monitor = dateify(read(SOURCE_FILES["monitor"]))
    base_cols = ["date", "portfolio_value", "daily_return", "cumulative_return", "Sharpe", "max_drawdown", "cash_weight", "exposure", "turnover"]
    official = official_perf[[c for c in base_cols if c in official_perf.columns]].copy()
    if not monitor.empty:
        blocked = {"paper_cumulative_return", "paper_daily_return", "paper_sharpe", "paper_max_drawdown", "cash", "candidate"}
        keep = [c for c in monitor.columns if c not in official.columns and c != "date" and c not in blocked]
        official = official.merge(monitor[["date"] + keep], on="date", how="left")
    official["candidate"] = "growth_champion_final"
    official["paper_cumulative_return"] = official["cumulative_return"] if "cumulative_return" in official.columns else np.nan
    official["paper_daily_return"] = official["daily_return"] if "daily_return" in official.columns else np.nan
    official["paper_sharpe"] = official["Sharpe"] if "Sharpe" in official.columns else np.nan
    official["paper_max_drawdown"] = official["max_drawdown"] if "max_drawdown" in official.columns else np.nan
    official["cash"] = official["cash_weight"] if "cash_weight" in official.columns else np.nan
    official["data_mode"] = "official_forward_paper_exact"
    official["official_start_date"] = start.strftime("%Y-%m-%d")
    official["growth_model_version"] = "growth_champion_final_v1_0_frozen"
    official["official_forward_status"] = "official_forward_warmup" if len(official_perf) < 20 else "official_forward_valid"
    official.to_csv(OFFICIAL_MONITOR, index=False)
    return official


def official_tracking(official_perf: pd.DataFrame) -> pd.DataFrame:
    if official_perf.empty:
        out = pd.DataFrame()
    else:
        latest = official_perf.iloc[-1]
        out = pd.DataFrame([{
            "date": latest["date"],
            "model": "growth_champion_final",
            "growth_model_version": "growth_champion_final_v1_0_frozen",
            "data_mode": "official_forward_paper_exact",
            "days_tracked": len(official_perf),
            "portfolio_value": latest.get("portfolio_value", np.nan),
            "cumulative_return": latest.get("cumulative_return", np.nan),
            "current_drawdown": latest.get("max_drawdown", np.nan),
            "exposure": latest.get("exposure", np.nan),
            "cash": latest.get("cash_weight", np.nan),
            "governance_status": "official_forward_warmup" if len(official_perf) < 20 else "official_forward_valid",
            "promotion_status": "real_capital_blocked",
            "reason": "official forward exact history only; no reconstruction/backfill",
        }])
    out.to_csv(OFFICIAL_TRACKING, index=False)
    return out


def compare_history(start: pd.Timestamp, debug_counts: dict[str, int], official_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, path in SOURCE_FILES.items():
        src = dateify(read(path))
        rows.append({
            "source": name,
            "source_rows": len(src),
            "debug_archived_rows_before_start": debug_counts.get(name, 0),
            "official_rows_from_start": len(src[src["date"].ge(start)]) if not src.empty else 0,
            "official_start_date": start.strftime("%Y-%m-%d"),
            "separation_rule": "debug before start; official exact forward from start only",
        })
    out = pd.DataFrame(rows)
    out.to_csv(COMPARE, index=False)
    return out


def data_quality(start: pd.Timestamp, audit: pd.DataFrame) -> pd.DataFrame:
    out = audit.copy()
    out["exact_live_start_date"] = start.strftime("%Y-%m-%d")
    out["history_classification"] = np.where(out["date"].lt(start), "historical_debug_reconstruction", "official_forward_candidate")
    out["governance"] = np.where(out["date"].eq(start), "official_forward_warmup_start", np.where(out["date"].gt(start), "official_forward_warmup", "debug_only"))
    out.to_csv(DATA_QUALITY, index=False)
    return out


def report(start: pd.Timestamp, official_perf: pd.DataFrame, debug_counts: dict[str, int]) -> str:
    latest = official_perf.iloc[-1].to_dict() if not official_perf.empty else {}
    status = "official_forward_valid" if len(official_perf) >= 20 else "official_forward_warmup"
    lines = [
        "===== OFFICIAL FORWARD PAPER BASELINE RESET =====",
        f"exact_live_start_date: {start.strftime('%Y-%m-%d')}",
        "start rule: first exact rebalance date after raw_target_return_exact availability; no carry-forward from proxy/debug holdings",
        f"official_status: {status}",
        "real_capital_status: real_capital_blocked",
        f"official_rows: {len(official_perf)}",
        f"debug_rows_archived: {sum(debug_counts.values())}",
        f"official_portfolio_value: {latest.get('portfolio_value', '')}",
        f"official_cumulative_return: {latest.get('cumulative_return', '')}",
        "namespaces_created: growth_official_paper_state/trades/actions/performance/monitor/live_tracking",
        "debug_history_preserved: yes",
        "production_changed: False",
        "model_logic_changed: False",
        "optimizer_changed: False",
    ]
    text = "\n".join(lines) + "\n"
    Path(REPORT).write_text(text, encoding="utf-8")
    return text


def main() -> None:
    start, audit = determine_exact_live_start()
    debug_counts = archive_debug(start)
    perf = official_performance(start)
    state = official_state(start, perf)
    actions, trades = official_actions_and_trades(start, state)
    monitor = official_monitor(start, perf, state)
    tracking = official_tracking(perf)
    data_quality(start, audit)
    compare_history(start, debug_counts, perf)
    print(report(start, perf, debug_counts))


if __name__ == "__main__":
    main()
