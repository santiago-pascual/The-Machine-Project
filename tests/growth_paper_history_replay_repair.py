from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from current_growth_feature_generation import (
    MAX_POSITIONS,
    _configured_exposure_cap,
    _dual_trend_filter,
    _vol_target_exposure,
)
from final_selected_holdings_sanity_check import audit_and_filter_selected_holdings
from growth_action_reconciliation import reconcile_growth_actions, signals_to_trade_rows
from growth_universe_quality_filter import apply_growth_universe_quality_filter


INITIAL_CAPITAL = 100000.0
MODEL_NAME = "growth_champion_final"
VARIANT = "growth_v1_exposure_cap_60_dual_trend_filter"

BACKUP_FILES = [
    "growth_candidate_paper_state.csv",
    "growth_candidate_paper_trades.csv",
    "growth_candidate_paper_performance.csv",
    "growth_candidate_paper_monitor.csv",
    "growth_paper_governance_report.csv",
    "growth_paper_governance_history.csv",
    "growth_paper_monthly_report.csv",
    "growth_live_tracking.csv",
    "growth_live_health.csv",
    "growth_live_drift.csv",
    "growth_live_tracking_governance.csv",
    "benchmark_daily_returns.csv",
    "benchmark_equity_curves.csv",
    "growth_candidate_action_signals.csv",
    "growth_candidate_rebalance_report.csv",
]

REBUILD_FILES = [
    "growth_candidate_paper_state.csv",
    "growth_candidate_paper_trades.csv",
    "growth_candidate_paper_performance.csv",
    "growth_candidate_paper_monitor.csv",
    "growth_candidate_action_signals.csv",
    "growth_candidate_rebalance_report.csv",
]


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def num(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def dates_from_existing(old_perf: pd.DataFrame, old_state: pd.DataFrame) -> list[str]:
    dates: set[str] = set()
    for df in [old_perf, old_state]:
        if not df.empty and "date" in df.columns:
            parsed = pd.to_datetime(df["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
            dates.update(parsed.tolist())
    return sorted(dates)


def backup_files() -> tuple[Path, dict[str, int]]:
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(f"paper_history_backup_{stamp}")
    folder.mkdir(exist_ok=False)
    counts: dict[str, int] = {}
    for file in BACKUP_FILES:
        src = Path(file)
        if src.exists():
            shutil.copy2(src, folder / src.name)
            counts[file] = len(read_csv(src))
        else:
            counts[file] = 0
    return folder, counts


def clear_rebuild_files() -> None:
    for file in REBUILD_FILES:
        path = Path(file)
        if path.exists():
            path.unlink()


def normalize_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return forecast
    df = forecast.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for col in [
        "current_price",
        "target_price",
        "expected_daily_return",
        "expected_total_return",
        "target_confidence",
        "signal_strength",
        "quality_score",
        "raw_target_return_exact",
        "raw_expected_daily_return_exact",
        "raw_target_price_exact",
        "time_to_target",
        "signal_strength_adjustment_value",
        "final_expected_return_after_adjustments",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "raw_target_return_exact" in df.columns:
        df["raw_target_return"] = df["raw_target_return_exact"]
    else:
        df["raw_target_return"] = df["target_price"] / df["current_price"] - 1.0
        df["raw_target_return_exact"] = np.nan
    df["raw_target_feature_source"] = np.where(df["raw_target_return_exact"].notna(), "raw_target_return_exact", "target_implied_proxy")
    df["exact_raw_target_available"] = df["raw_target_return_exact"].notna()
    df["raw_target_rank"] = df["raw_target_return"].rank(ascending=False, method="min")
    df["data_source"] = "growth_paper_history_replay"
    df["growth_paper_model"] = MODEL_NAME
    df["growth_paper_variant"] = VARIANT
    df["exact_growth_features_available"] = df["exact_raw_target_available"]
    df["fallback_reason"] = "history replay using forecast_history snapshot for date"
    return df


def latest_previous_state() -> pd.DataFrame:
    state = read_csv("growth_candidate_paper_state.csv")
    if state.empty:
        return state
    latest = state["date"].astype(str).max()
    return state[state["date"].astype(str).eq(latest)].copy()


def previous_prices_for_return(previous: pd.DataFrame, current_prices: dict[str, float]) -> float:
    if previous.empty:
        return 0.0
    ret = 0.0
    for _, row in previous.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if ticker == "CASH":
            continue
        old_price = num(row.get("current_price", np.nan))
        new_price = num(current_prices.get(ticker, np.nan))
        weight = num(row.get("paper_position_weight", 0.0), 0.0)
        if np.isfinite(old_price) and old_price > 0 and np.isfinite(new_price):
            ret += weight * (new_price / old_price - 1.0)
    return float(ret)


def performance_metrics(perf: pd.DataFrame) -> dict[str, float]:
    if perf.empty:
        return {"cumulative_return": 0.0, "volatility": 0.0, "Sharpe": 0.0, "max_drawdown": 0.0}
    returns = pd.to_numeric(perf["daily_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    vol = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float((returns.mean() * 252) / vol) if vol > 0 else 0.0
    dd = equity / equity.cummax() - 1.0
    return {
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "volatility": vol,
        "Sharpe": sharpe,
        "max_drawdown": float(dd.min()),
    }


def append_rows(path: str, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    existing = read_csv(path)
    out = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows.copy()
    out.to_csv(path, index=False)


def build_features_for_date(forecast: pd.DataFrame, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = forecast[forecast["date"].astype(str).eq(date)].copy()
    if day.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = normalize_forecast(day)
    pre_quality_positive = df[pd.to_numeric(df["raw_target_return"], errors="coerce") > 0].sort_values("raw_target_return", ascending=False)
    selected_before_quality = pre_quality_positive.head(MAX_POSITIONS)["ticker"].astype(str).tolist()
    prior = latest_previous_state()
    prior_tickers = prior[prior.get("ticker", pd.Series(dtype=str)).astype(str).ne("CASH")]["ticker"].astype(str).str.upper().tolist() if not prior.empty else []
    yahoo_fetch_candidates = list(dict.fromkeys(selected_before_quality + pre_quality_positive.head(20)["ticker"].astype(str).tolist() + prior_tickers))
    df, _, _ = apply_growth_universe_quality_filter(df, pd.Timestamp(date), yahoo_fetch_tickers=yahoo_fetch_candidates)
    if "quality_pass" not in df.columns:
        df["quality_pass"] = True
    eligible = df["quality_pass"].astype(bool) & (pd.to_numeric(df["raw_target_return"], errors="coerce") > 0)
    df["raw_target_selected"] = False
    retained = df[df["ticker"].isin(prior_tickers) & eligible].sort_values("raw_target_return", ascending=False)["ticker"].tolist()
    fresh = df[eligible & ~df["ticker"].isin(retained)].sort_values("raw_target_return", ascending=False)["ticker"].tolist()
    selected = list(dict.fromkeys(retained + fresh))[:MAX_POSITIONS]
    selected, holdings_audit, _ = audit_and_filter_selected_holdings(df, selected, pd.Timestamp(date), MAX_POSITIONS)
    df["raw_target_selected"] = df["ticker"].isin(selected)
    df["soft_exit_status"] = np.where(
        df["ticker"].isin(prior_tickers) & (pd.to_numeric(df["raw_target_return"], errors="coerce") > 0),
        "retained_positive_raw_target",
        np.where(df["ticker"].isin(prior_tickers), "exit_nonpositive_raw_target", "not_prior_position"),
    )
    df["prior_position_status"] = np.where(df["ticker"].isin(prior_tickers), "prior_position", "new_or_unheld")
    exposure, raw_exposure, rolling_vol, vol_meta = _vol_target_exposure(selected, pd.Timestamp(date), allow_stale=True)
    cap60 = _configured_exposure_cap()
    dual = _dual_trend_filter(pd.Timestamp(date))
    dual_cap = float(dual.get("dual_trend_cap", cap60))
    final_exposure = float(np.clip(min(exposure, cap60, dual_cap), 0.0, 1.0))
    final_weight = final_exposure / len(selected) if selected else 0.0
    df["vol_target_exposure"] = exposure
    df["volatility_target_exposure"] = final_exposure
    df["uncapped_volatility_target_exposure"] = exposure
    df["raw_volatility_target_exposure"] = raw_exposure
    df["rolling_volatility_used"] = rolling_vol
    df["volatility_source"] = vol_meta.get("volatility_source", "")
    df["volatility_source_date"] = vol_meta.get("source_date", "")
    df["volatility_is_fresh"] = vol_meta.get("is_fresh", False)
    df["exposure_cap"] = cap60
    df["exposure_cap_60"] = cap60
    df["dual_trend_cap"] = dual_cap
    df["final_exposure"] = final_exposure
    for key, value in dual.items():
        df[key] = value
    df["final_growth_weight"] = np.where(df["raw_target_selected"], final_weight, 0.0)
    df["cash_weight"] = 1.0 - final_exposure
    if not holdings_audit.empty:
        merge_cols = [c for c in ["ticker", "holding_quality_classification", "holding_risk_notes"] if c in holdings_audit.columns]
        df = df.merge(holdings_audit[merge_cols].drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_audit"))
        for col in ["holding_quality_classification", "holding_risk_notes"]:
            audit_col = f"{col}_audit"
            if audit_col in df.columns:
                df[col] = df[audit_col].combine_first(df.get(col, pd.Series(index=df.index, dtype=object)))
                df = df.drop(columns=[audit_col])
    allocation = df[df["raw_target_selected"]].copy()
    return df, allocation


def replay_date(date: str, forecast: pd.DataFrame, old_perf_for_date: pd.DataFrame, old_state_for_date: pd.DataFrame) -> dict[str, object]:
    features, allocation = build_features_for_date(forecast, date)
    if allocation.empty:
        return {"date": date, "status": "skipped_no_allocation"}
    features.to_csv("current_raw_target_features.csv", index=False)
    features.to_csv("current_growth_features.csv", index=False)
    allocation.to_csv("current_growth_candidate_allocation.csv", index=False)

    previous = latest_previous_state()
    current_prices = dict(zip(features["ticker"].astype(str), pd.to_numeric(features["current_price"], errors="coerce")))
    daily_return = previous_prices_for_return(previous, current_prices)
    previous_perf = read_csv("growth_candidate_paper_performance.csv")
    previous_value = float(pd.to_numeric(previous_perf["portfolio_value"], errors="coerce").dropna().iloc[-1]) if not previous_perf.empty and "portfolio_value" in previous_perf.columns and not pd.to_numeric(previous_perf["portfolio_value"], errors="coerce").dropna().empty else INITIAL_CAPITAL
    portfolio_value = previous_value * (1.0 + daily_return)

    action_signals, rebalance_report, rec = reconcile_growth_actions(
        current_allocation=allocation,
        current_date=date,
        portfolio_value=portfolio_value,
        previous_prices=pd.Series(current_prices),
        overwrite_same_day=True,
    )
    trades = signals_to_trade_rows(action_signals, model_mode=MODEL_NAME, variant=VARIANT)
    selected = allocation["ticker"].astype(str).tolist()
    final_exposure = float(pd.to_numeric(allocation["final_exposure"], errors="coerce").dropna().iloc[0])
    cash = float(pd.to_numeric(allocation["cash_weight"], errors="coerce").dropna().iloc[0])
    final_weight = final_exposure / len(selected) if selected else 0.0

    state_rows = []
    for _, row in allocation.iterrows():
        ticker = str(row["ticker"])
        price = num(row.get("current_price", np.nan))
        old_entry = price
        if not previous.empty:
            prev_row = previous[previous["ticker"].astype(str).eq(ticker)]
            if not prev_row.empty:
                old_entry = num(prev_row.iloc[-1].get("entry_price", price), price)
        state_rows.append(
            {
                "date": date,
                "ticker": ticker,
                "paper_position_weight": final_weight,
                "paper_position_value": portfolio_value * final_weight,
                "entry_price": old_entry if ticker in set(previous.get("ticker", pd.Series(dtype=str)).astype(str)) else price,
                "current_price": price,
                "unrealized_return": price / old_entry - 1.0 if old_entry and np.isfinite(price) and np.isfinite(old_entry) else np.nan,
                "realized_return": daily_return,
                "action": action_signals.loc[action_signals["ticker"].eq(ticker), "action"].iloc[0],
                "model_mode": MODEL_NAME,
                "cash_weight": cash,
                "data_source": "growth_paper_history_replay",
                "raw_target_current_features_available": True,
                "raw_target_feature_source": str(row.get("raw_target_feature_source", "")),
                "growth_paper_variant": VARIANT,
                "exposure_cap": row.get("exposure_cap", np.nan),
                "exposure_cap_60": row.get("exposure_cap_60", np.nan),
                "dual_trend_cap": row.get("dual_trend_cap", np.nan),
                "vol_target_exposure": row.get("vol_target_exposure", np.nan),
                "final_exposure": final_exposure,
                "spy_close": row.get("spy_close", np.nan),
                "spy_ma_200": row.get("spy_ma_200", np.nan),
                "qqq_close": row.get("qqq_close", np.nan),
                "qqq_ma_200": row.get("qqq_ma_200", np.nan),
                "spy_below_200d": row.get("spy_below_200d", np.nan),
                "qqq_below_200d": row.get("qqq_below_200d", np.nan),
                "dual_trend_reason": row.get("dual_trend_reason", ""),
            }
        )
    state_rows.append(
        {
            "date": date,
            "ticker": "CASH",
            "paper_position_weight": cash,
            "paper_position_value": portfolio_value * cash,
            "entry_price": 1.0,
            "current_price": 1.0,
            "unrealized_return": 0.0,
            "realized_return": 0.0,
            "action": action_signals.loc[action_signals["ticker"].eq("CASH"), "action"].iloc[0],
            "model_mode": MODEL_NAME,
            "cash_weight": cash,
            "data_source": "growth_paper_history_replay",
            "raw_target_current_features_available": True,
            "raw_target_feature_source": "",
            "growth_paper_variant": VARIANT,
        }
    )
    state = pd.DataFrame(state_rows)
    append_rows("growth_candidate_paper_state.csv", state)
    append_rows("growth_candidate_paper_trades.csv", trades)

    perf_existing = read_csv("growth_candidate_paper_performance.csv")
    temp = pd.concat([perf_existing, pd.DataFrame([{"date": date, "daily_return": daily_return, "portfolio_value": portfolio_value}])], ignore_index=True)
    metrics = performance_metrics(temp)
    first_alloc = allocation.iloc[0]
    perf_row = pd.DataFrame(
        [
            {
                "date": date,
                "model_mode": MODEL_NAME,
                "portfolio_value": portfolio_value,
                "daily_return": daily_return,
                "cumulative_return": metrics["cumulative_return"],
                "volatility": metrics["volatility"],
                "Sharpe": metrics["Sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "cash_weight": cash,
                "exposure": final_exposure,
                "turnover": rec["turnover"],
                "data_source": "growth_paper_history_replay",
                "raw_target_current_features_available": True,
                "fallback_reason": "history replay repaired",
                "raw_target_feature_source": str(first_alloc.get("raw_target_feature_source", "")),
                "growth_paper_variant": VARIANT,
                "uncapped_exposure": first_alloc.get("uncapped_volatility_target_exposure", np.nan),
                "exposure_cap": first_alloc.get("exposure_cap", np.nan),
                "vol_target_exposure": first_alloc.get("vol_target_exposure", np.nan),
                "exposure_cap_60": first_alloc.get("exposure_cap_60", np.nan),
                "dual_trend_cap": first_alloc.get("dual_trend_cap", np.nan),
                "final_exposure": final_exposure,
                "spy_close": first_alloc.get("spy_close", np.nan),
                "spy_ma_200": first_alloc.get("spy_ma_200", np.nan),
                "qqq_close": first_alloc.get("qqq_close", np.nan),
                "qqq_ma_200": first_alloc.get("qqq_ma_200", np.nan),
                "spy_below_200d": first_alloc.get("spy_below_200d", np.nan),
                "qqq_below_200d": first_alloc.get("qqq_below_200d", np.nan),
                "dual_trend_reason": first_alloc.get("dual_trend_reason", ""),
            }
        ]
    )
    append_rows("growth_candidate_paper_performance.csv", perf_row)
    monitor = pd.DataFrame(
        [
            {
                "date": date,
                "candidate": MODEL_NAME,
                "paper_cumulative_return": metrics["cumulative_return"],
                "paper_daily_return": daily_return,
                "paper_sharpe": metrics["Sharpe"],
                "rolling_sharpe_60_period_proxy": metrics["Sharpe"],
                "paper_max_drawdown": metrics["max_drawdown"],
                "cash": cash,
                "exposure": final_exposure,
                "turnover": rec["turnover"],
                "top_ticker_concentration": final_weight,
                "risk_flags": "paper_history_replayed",
                "promotion_status": "paper trading allowed; production promotion blocked",
            }
        ]
    )
    append_rows("growth_candidate_paper_monitor.csv", monitor)

    old_value = num(old_perf_for_date["portfolio_value"].iloc[-1], np.nan) if not old_perf_for_date.empty and "portfolio_value" in old_perf_for_date.columns else np.nan
    old_ret = num(old_perf_for_date["daily_return"].iloc[-1], np.nan) if not old_perf_for_date.empty and "daily_return" in old_perf_for_date.columns else np.nan
    old_exp = num(old_perf_for_date["exposure"].iloc[-1], np.nan) if not old_perf_for_date.empty and "exposure" in old_perf_for_date.columns else np.nan
    old_hold = ",".join(old_state_for_date.loc[old_state_for_date["ticker"].astype(str).ne("CASH"), "ticker"].astype(str).tolist()) if not old_state_for_date.empty and "ticker" in old_state_for_date.columns else ""
    new_hold = ",".join(selected)
    reasons = []
    if old_hold != new_hold:
        reasons.append("holdings_changed")
    if np.isfinite(old_value) and abs(old_value - portfolio_value) > 1e-6:
        reasons.append("portfolio_value_changed")
    if np.isfinite(old_ret) and abs(old_ret - daily_return) > 1e-10:
        reasons.append("daily_return_changed")
    if np.isfinite(old_exp) and abs(old_exp - final_exposure) > 1e-10:
        reasons.append("exposure_changed")
    return {
        "date": date,
        "status": "replayed",
        "old_portfolio_value": old_value,
        "new_portfolio_value": portfolio_value,
        "old_daily_return": old_ret,
        "new_daily_return": daily_return,
        "old_holdings": old_hold,
        "new_holdings": new_hold,
        "old_exposure": old_exp,
        "new_exposure": final_exposure,
        "action_reconciliation_passed": bool(rec["reconciliation_passed"]),
        "reason_for_differences": ",".join(reasons) if reasons else "unchanged",
    }


def regenerate_dependent_reports() -> None:
    commands = [
        [sys.executable, "growth_paper_governance.py"],
        [sys.executable, "growth_live_tracking_monitor.py"],
        [sys.executable, "benchmark_daily_series_export.py"],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=False, timeout=120)
        except Exception as exc:
            print(f"WARNING: dependent report failed: {' '.join(cmd)} -> {exc}")


def main() -> None:
    old_perf = read_csv("growth_candidate_paper_performance.csv")
    old_state = read_csv("growth_candidate_paper_state.csv")
    forecast = read_csv("forecast_history.csv")
    if forecast.empty or "date" not in forecast.columns:
        raise ValueError("forecast_history.csv is required for replay.")
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    replay_dates = dates_from_existing(old_perf, old_state)
    if not replay_dates:
        raise ValueError("No existing growth paper dates found to replay.")

    backup_dir, backup_counts = backup_files()
    clear_rebuild_files()

    audit_rows = []
    for date in replay_dates:
        old_perf_day = old_perf[old_perf["date"].astype(str).eq(date)] if not old_perf.empty and "date" in old_perf.columns else pd.DataFrame()
        old_state_day = old_state[old_state["date"].astype(str).eq(date)] if not old_state.empty and "date" in old_state.columns else pd.DataFrame()
        audit_rows.append(replay_date(date, forecast, old_perf_day, old_state_day))

    audit = pd.DataFrame(audit_rows)
    audit.to_csv("growth_paper_replay_audit.csv", index=False)
    changes = audit[audit["reason_for_differences"].astype(str).ne("unchanged")].copy() if "reason_for_differences" in audit.columns else pd.DataFrame()
    changes.to_csv("growth_paper_replay_changes.csv", index=False)

    regenerate_dependent_reports()

    perf = read_csv("growth_candidate_paper_performance.csv")
    state = read_csv("growth_candidate_paper_state.csv")
    latest_date = perf["date"].astype(str).max() if not perf.empty else ""
    latest_state = state[state["date"].astype(str).eq(latest_date)] if not state.empty and latest_date else pd.DataFrame()
    final_holdings = ",".join(latest_state.loc[latest_state["ticker"].astype(str).ne("CASH"), "ticker"].astype(str).tolist()) if not latest_state.empty else ""
    latest_perf = perf[perf["date"].astype(str).eq(latest_date)].iloc[-1] if not perf.empty and latest_date else pd.Series(dtype=object)
    biggest_daily = float((pd.to_numeric(audit.get("new_daily_return", pd.Series(dtype=float)), errors="coerce") - pd.to_numeric(audit.get("old_daily_return", pd.Series(dtype=float)), errors="coerce")).abs().max()) if not audit.empty else np.nan
    biggest_value = float((pd.to_numeric(audit.get("new_portfolio_value", pd.Series(dtype=float)), errors="coerce") - pd.to_numeric(audit.get("old_portfolio_value", pd.Series(dtype=float)), errors="coerce")).abs().max()) if not audit.empty else np.nan
    summary = [
        "===== GROWTH PAPER HISTORY REPLAY AND REPAIR =====",
        f"backup_folder: {backup_dir}",
        f"replay_dates: {', '.join(replay_dates)}",
        f"number_of_replay_dates: {len(replay_dates)}",
        f"rows_backed_up: {sum(backup_counts.values())}",
        f"rows_regenerated_state: {len(state)}",
        f"rows_regenerated_performance: {len(perf)}",
        f"dates_changed: {len(changes)}",
        f"biggest_daily_return_change: {biggest_daily}",
        f"biggest_portfolio_value_change: {biggest_value}",
        f"final_corrected_holdings: {final_holdings}",
        f"final_corrected_cumulative_return: {latest_perf.get('cumulative_return', np.nan)}",
        f"final_corrected_exposure: {latest_perf.get('exposure', np.nan)}",
        f"final_corrected_cash: {latest_perf.get('cash_weight', np.nan)}",
        "dashboard_files_updated: yes",
        "production_change: none",
        "real_trading: disabled",
    ]
    Path("growth_paper_replay_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"Saved: {Path('growth_paper_replay_audit.csv').resolve()}")
    print(f"Saved: {Path('growth_paper_replay_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
