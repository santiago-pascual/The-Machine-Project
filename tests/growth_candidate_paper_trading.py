from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from growth_action_reconciliation import reconcile_growth_actions, signals_to_trade_rows
from growth_rebalance_scheduler import scheduler_status

CANDIDATE_NAME = "growth_champion_final"
CANDIDATE_VARIANT = "growth_v1_exposure_cap_60_dual_trend_filter"
CANDIDATE_MODEL_VERSION = "growth_champion_final_v1_0_frozen"
LEGACY_UNCAPPED_MODEL = "growth_v1_uncapped"
BACKTEST_VARIANT = "soft_exit_rule_vol_target_22pct"
BASE_VARIANT = "soft_exit_rule"
CONFIG_FILE = "growth_candidate_paper_config.json"

GROWTH_DAILY_FILE = "growth_volatility_targeting_daily_returns.csv"
GROWTH_RESULTS_FILE = "growth_candidate_deep_validation_results.csv"
GROWTH_GOVERNANCE_FILE = "growth_candidate_deep_governance.csv"
GROWTH_EXPOSURE_FILE = "growth_volatility_targeting_exposure.csv"
EXIT_TRADES_FILE = "exit_rule_walk_forward_trades.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
FORECAST_HISTORY_FILE = "forecast_history.csv"
CURRENT_GROWTH_ALLOCATION_FILE = "current_growth_candidate_allocation.csv"
CURRENT_GROWTH_QUALITY_FILE = "current_growth_universe_quality.csv"
LIFECYCLE_FILE = "model_lifecycle_status.csv"
SUMMARY_FILE = "final_research_summary.txt"
DASHBOARD_FILE = "research_dashboard_summary.csv"

STATE_FILE = "growth_candidate_paper_state.csv"
TRADES_FILE = "growth_candidate_paper_trades.csv"
PERFORMANCE_FILE = "growth_candidate_paper_performance.csv"
MONITOR_FILE = "growth_candidate_paper_monitor.csv"
DEFAULT_INITIAL_CAPITAL = 100000.0



def _growth_paper_config() -> dict[str, object]:
    default = {
        "active_growth_paper_model": CANDIDATE_NAME,
        "active_variant": CANDIDATE_VARIANT,
        "legacy_uncapped_model": LEGACY_UNCAPPED_MODEL,
        "volatility_target": 0.22,
        "exposure_cap": 0.60,
        "max_leverage": 1.0,
        "paper_only": True,
        "real_trading": False,
    }
    path = Path(CONFIG_FILE)
    if not path.exists():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    default.update(loaded if isinstance(loaded, dict) else {})
    return default


def _exposure_cap() -> float:
    cfg = _growth_paper_config()
    return float(np.clip(float(cfg.get("exposure_cap", 0.60)), 0.0, float(cfg.get("max_leverage", 1.0))))

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


def _dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"])


def _today_from_data() -> str:
    daily = _candidate_daily()
    if not daily.empty:
        return pd.Timestamp(daily["date"].max()).strftime("%Y-%m-%d")
    return pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def _append_or_update(path: str | Path, rows: pd.DataFrame, date: str, overwrite_same_day: bool) -> tuple[int, int, int]:
    existing = _read_csv(path)
    if existing.empty:
        rows.to_csv(path, index=False)
        return len(rows), 0, 0
    skipped = 0
    overwritten = 0
    output = existing.copy()
    if "date" in output.columns:
        same_day = output["date"].astype(str).eq(str(date))
        if same_day.any():
            if overwrite_same_day:
                overwritten = int(same_day.sum())
                output = output[~same_day]
            else:
                skipped = len(rows)
                output.to_csv(path, index=False)
                return 0, skipped, overwritten
    output = pd.concat([output, rows], ignore_index=True)
    output.to_csv(path, index=False)
    return len(rows), skipped, overwritten


def _candidate_daily() -> pd.DataFrame:
    daily = _dates(_read_csv(GROWTH_DAILY_FILE))
    if daily.empty:
        return daily
    selector = daily.get("vol_target_variant", daily.get("variant", pd.Series(index=daily.index, dtype=str))).astype(str).eq(BACKTEST_VARIANT)
    daily = daily[selector].copy()
    if daily.empty:
        return daily
    daily["paper_return_proxy"] = _num(daily.get("vol_target_return", daily.get("return", pd.Series(index=daily.index, dtype=float))))
    daily["target_exposure"] = _num(daily.get("target_exposure", pd.Series(index=daily.index, dtype=float))).fillna(0.0).clip(0.0, 1.0)
    daily["cash_weight"] = _num(daily.get("cash_weight", 1.0 - daily["target_exposure"])).fillna(1.0 - daily["target_exposure"]).clip(0.0, 1.0)
    daily["turnover"] = _num(daily.get("turnover", pd.Series(index=daily.index, dtype=float))).fillna(0.0)
    return daily.sort_values("date")


def _latest_candidate_row() -> pd.Series:
    daily = _candidate_daily()
    if daily.empty:
        raise ValueError("Missing growth volatility targeting candidate daily data.")
    return daily.iloc[-1]


def _current_forecast_snapshot() -> pd.DataFrame:
    forecast = _dates(_read_csv(FORECAST_HISTORY_FILE))
    if forecast.empty:
        return forecast
    latest_date = forecast["date"].max()
    latest = forecast[forecast["date"].eq(latest_date)].copy()
    if latest.empty:
        return latest
    for col in ["current_price", "expected_daily_return", "final_weight_percent", "signal_strength", "quality_score"]:
        if col in latest.columns:
            latest[col] = _num(latest[col])
    return latest



def _quality_reason_map() -> dict[str, str]:
    quality = _dates(_read_csv(CURRENT_GROWTH_QUALITY_FILE))
    if quality.empty or "ticker" not in quality.columns:
        return {}
    latest_date = quality["date"].max() if "date" in quality.columns else None
    if latest_date is not None:
        quality = quality[quality["date"].eq(latest_date)].copy()
    if "quality_pass" not in quality.columns:
        return {}
    quality["quality_pass"] = quality["quality_pass"].astype(str).str.lower().isin(["true", "1", "yes"])
    failed = quality[~quality["quality_pass"]].copy()
    if failed.empty:
        return {}
    return dict(zip(failed["ticker"].astype(str), failed.get("exclusion_reason", pd.Series("failed universe quality filter", index=failed.index)).astype(str)))


def _current_growth_allocation() -> pd.DataFrame:
    allocation = _dates(_read_csv(CURRENT_GROWTH_ALLOCATION_FILE))
    if allocation.empty:
        return allocation
    latest_date = allocation["date"].max()
    allocation = allocation[allocation["date"].eq(latest_date)].copy()
    for col in ["current_price", "raw_target_return", "raw_target_return_exact", "raw_expected_daily_return_exact", "raw_target_rank", "final_growth_weight", "cash_weight", "vol_target_exposure", "volatility_target_exposure", "uncapped_volatility_target_exposure", "exposure_cap", "exposure_cap_60", "dual_trend_cap", "final_exposure", "spy_close", "spy_ma_200", "qqq_close", "qqq_ma_200"]:
        if col in allocation.columns:
            allocation[col] = _num(allocation[col])
    if "quality_pass" in allocation.columns:
        allocation["quality_pass"] = allocation["quality_pass"].astype(str).str.lower().isin(["true", "1", "yes"])
        allocation = allocation[allocation["quality_pass"]].copy()
    return allocation


def _current_growth_candidate_row_or_none(historical_latest: pd.Series) -> pd.Series | None:
    allocation = _current_growth_allocation()
    if allocation.empty:
        return None
    current_date = pd.Timestamp(allocation["date"].max())
    historical_date = pd.Timestamp(historical_latest["date"])
    if current_date <= historical_date:
        return None
    tickers = allocation.sort_values("final_growth_weight", ascending=False)["ticker"].astype(str).tolist()
    exposure = float(_num(allocation["final_growth_weight"]).fillna(0.0).sum())
    cap = _exposure_cap()
    dual_trend_cap = float(_num(allocation.get("dual_trend_cap", pd.Series([cap]))).dropna().iloc[0]) if "dual_trend_cap" in allocation.columns and not _num(allocation["dual_trend_cap"]).dropna().empty else cap
    vol_target_exposure = float(_num(allocation.get("vol_target_exposure", allocation.get("uncapped_volatility_target_exposure", pd.Series([exposure])))).dropna().iloc[0]) if len(allocation) else exposure
    final_exposure = float(_num(allocation.get("final_exposure", pd.Series([exposure]))).dropna().iloc[0]) if "final_exposure" in allocation.columns and not _num(allocation["final_exposure"]).dropna().empty else min(exposure, cap, dual_trend_cap)
    exposure = float(np.clip(final_exposure, 0.0, 1.0))
    cash = float(1.0 - exposure)
    source = str(allocation.get("raw_target_feature_source", pd.Series(["unknown"])).dropna().iloc[0]) if "raw_target_feature_source" in allocation.columns and not allocation["raw_target_feature_source"].dropna().empty else "unknown"
    exact_available = bool(source == "raw_target_return_exact")
    fallback_reason = (
        "exact raw target current growth allocation generated from current_growth_feature_generation"
        if exact_available
        else "Growth paper using proxy raw target, not production-parity."
    )
    return pd.Series(
        {
            "date": current_date,
            "selected_tickers": ",".join(tickers),
            "target_exposure": float(np.clip(exposure, 0.0, 1.0)),
            "cash_weight": float(np.clip(cash, 0.0, 1.0)),
            "turnover": np.nan,
            "data_source": "current_growth_candidate_allocation",
            "growth_paper_model": str(allocation.get("growth_paper_model", pd.Series([CANDIDATE_NAME])).dropna().iloc[0]) if "growth_paper_model" in allocation.columns and not allocation["growth_paper_model"].dropna().empty else CANDIDATE_NAME,
            "growth_paper_variant": str(allocation.get("growth_paper_variant", pd.Series([CANDIDATE_VARIANT])).dropna().iloc[0]) if "growth_paper_variant" in allocation.columns and not allocation["growth_paper_variant"].dropna().empty else CANDIDATE_VARIANT,
            "exposure_cap": cap,
            "exposure_cap_60": cap,
            "dual_trend_cap": dual_trend_cap,
            "vol_target_exposure": vol_target_exposure,
            "final_exposure": exposure,
            "spy_close": float(_num(allocation.get("spy_close", pd.Series([np.nan]))).dropna().iloc[0]) if "spy_close" in allocation.columns and not _num(allocation["spy_close"]).dropna().empty else np.nan,
            "spy_ma_200": float(_num(allocation.get("spy_ma_200", pd.Series([np.nan]))).dropna().iloc[0]) if "spy_ma_200" in allocation.columns and not _num(allocation["spy_ma_200"]).dropna().empty else np.nan,
            "qqq_close": float(_num(allocation.get("qqq_close", pd.Series([np.nan]))).dropna().iloc[0]) if "qqq_close" in allocation.columns and not _num(allocation["qqq_close"]).dropna().empty else np.nan,
            "qqq_ma_200": float(_num(allocation.get("qqq_ma_200", pd.Series([np.nan]))).dropna().iloc[0]) if "qqq_ma_200" in allocation.columns and not _num(allocation["qqq_ma_200"]).dropna().empty else np.nan,
            "spy_below_200d": bool(allocation.get("spy_below_200d", pd.Series([False])).iloc[0]) if "spy_below_200d" in allocation.columns else False,
            "qqq_below_200d": bool(allocation.get("qqq_below_200d", pd.Series([False])).iloc[0]) if "qqq_below_200d" in allocation.columns else False,
            "dual_trend_reason": str(allocation.get("dual_trend_reason", pd.Series([""])).iloc[0]) if "dual_trend_reason" in allocation.columns else "",
            "raw_target_current_features_available": exact_available,
            "raw_target_feature_source": source,
            "fallback_reason": fallback_reason,
        }
    )


def _current_candidate_row_or_none(historical_latest: pd.Series) -> pd.Series | None:
    current = _current_forecast_snapshot()
    if current.empty:
        return None
    current_date = pd.Timestamp(current["date"].max())
    historical_date = pd.Timestamp(historical_latest["date"])
    if current_date <= historical_date:
        return None
    if "selected" in current.columns:
        selected = current[current["selected"].astype(bool)].copy()
    else:
        selected = pd.DataFrame()
    if selected.empty and "final_weight_percent" in current.columns:
        selected = current[_num(current["final_weight_percent"]).fillna(0.0) > 0].copy()
    if selected.empty:
        return None
    if "final_weight_percent" in selected.columns:
        selected = selected.sort_values("final_weight_percent", ascending=False)
    tickers = selected["ticker"].astype(str).tolist()
    exposure = float((_num(selected.get("final_weight_percent", pd.Series(index=selected.index, dtype=float))).fillna(0.0) / 100.0).sum())
    if not np.isfinite(exposure) or exposure <= 0:
        exposure = float(historical_latest.get("target_exposure", 0.0))
    exposure = float(np.clip(exposure, 0.0, 1.0))
    return pd.Series(
        {
            "date": current_date,
            "selected_tickers": ",".join(tickers),
            "target_exposure": exposure,
            "cash_weight": 1.0 - exposure,
            "turnover": np.nan,
            "data_source": "current_forecast_history_proxy",
            "raw_target_current_features_available": False,
            "fallback_reason": "current raw_target growth features missing; using latest current forecast/allocation snapshot",
        }
    )


def _latest_candidate_context(allow_proxy_fallback: bool = False) -> tuple[pd.Series, str, bool, str]:
    historical = _latest_candidate_row()
    current_growth = _current_growth_candidate_row_or_none(historical)
    if current_growth is not None:
        raw_available = bool(current_growth.get("raw_target_current_features_available", False))
        return current_growth, "current_growth_candidate_allocation", raw_available, str(current_growth.get("fallback_reason", ""))
    if not allow_proxy_fallback:
        raise ValueError("Current growth features are missing. Run current_growth_feature_generation.py or pass --allow-proxy-fallback.")
    current = _current_candidate_row_or_none(historical)
    if current is not None:
        return current, "current_forecast_history_proxy", False, str(current.get("fallback_reason", ""))
    historical = historical.copy()
    historical["data_source"] = "historical_growth_backtest"
    historical["raw_target_current_features_available"] = True
    historical["fallback_reason"] = "current snapshot unavailable or not newer than historical growth file"
    return historical, "historical_growth_backtest", True, str(historical["fallback_reason"])


def _latest_prices(date: pd.Timestamp, tickers: list[str], prefer_current: bool = True) -> pd.Series:
    allocation = _current_growth_allocation()
    if not allocation.empty and pd.Timestamp(allocation["date"].max()) >= date and "current_price" in allocation.columns:
        prices = _num(allocation.set_index("ticker")["current_price"])
        current_prices = prices.reindex(tickers)
        if current_prices.notna().any():
            return current_prices
    if prefer_current:
        current = _current_forecast_snapshot()
        if not current.empty:
            latest_current_date = pd.Timestamp(current["date"].max())
            if latest_current_date >= date and "current_price" in current.columns:
                prices = _num(current.set_index("ticker")["current_price"])
                current_prices = prices.reindex(tickers)
                if current_prices.notna().any():
                    return current_prices
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty:
        return pd.Series(dtype=float)
    if "model_mode" in snaps.columns:
        baseline = snaps[snaps["model_mode"].eq("baseline")]
        if not baseline.empty:
            snaps = baseline.copy()
    latest = snaps[snaps["date"].le(date)].sort_values("date").drop_duplicates("ticker", keep="last")
    prices = latest.set_index("ticker")["current_price"] if "current_price" in latest.columns else pd.Series(dtype=float)
    prices = _num(prices)
    return prices.reindex(tickers)


def _selected_tickers_for_latest(latest: pd.Series) -> list[str]:
    tickers = [t.strip() for t in str(latest.get("selected_tickers", "")).split(",") if t.strip() and t.strip().lower() != "nan"]
    if tickers:
        return tickers
    trades = _dates(_read_csv(EXIT_TRADES_FILE))
    if trades.empty:
        return []
    trades = trades[(trades["variant"].eq(BASE_VARIANT)) & (trades["date"].eq(latest["date"]))]
    return sorted(trades["ticker"].astype(str).unique().tolist())


def _previous_state() -> pd.DataFrame:
    state = _read_csv(STATE_FILE)
    if state.empty:
        return state
    if "date" in state.columns:
        latest_date = state["date"].astype(str).max()
        state = state[state["date"].astype(str).eq(latest_date)].copy()
    for col in ["paper_position_weight", "paper_position_value", "entry_price", "current_price"]:
        if col in state.columns:
            state[col] = _num(state[col])
    return state


def _previous_value() -> float:
    perf = _read_csv(PERFORMANCE_FILE)
    if perf.empty or "portfolio_value" not in perf.columns:
        return DEFAULT_INITIAL_CAPITAL
    values = _num(perf["portfolio_value"]).dropna()
    return float(values.iloc[-1]) if not values.empty else DEFAULT_INITIAL_CAPITAL


def _realized_daily_return(previous: pd.DataFrame, prices: pd.Series) -> float:
    if previous.empty:
        return 0.0
    total = 0.0
    for _, row in previous.iterrows():
        ticker = str(row.get("ticker", ""))
        if ticker == "CASH":
            continue
        prev_price = float(row.get("current_price", np.nan))
        current_price = float(prices.get(ticker, np.nan))
        weight = float(row.get("paper_position_weight", 0.0))
        if np.isfinite(prev_price) and prev_price > 0 and np.isfinite(current_price):
            total += weight * (current_price / prev_price - 1.0)
    return float(total)


def _performance_metrics(perf: pd.DataFrame) -> dict:
    if perf.empty or "daily_return" not in perf.columns:
        return {"volatility": 0.0, "Sharpe": 0.0, "max_drawdown": 0.0, "cumulative_return": 0.0}
    returns = _num(perf["daily_return"]).dropna()
    if returns.empty:
        return {"volatility": 0.0, "Sharpe": 0.0, "max_drawdown": 0.0, "cumulative_return": 0.0}
    equity = (1.0 + returns).cumprod()
    vol = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float((returns.mean() * 252) / vol) if vol > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min())
    return {"volatility": vol, "Sharpe": sharpe, "max_drawdown": dd, "cumulative_return": float(equity.iloc[-1] - 1.0)}


def _register_candidate() -> None:
    lifecycle = _read_csv(LIFECYCLE_FILE)
    row = {
        "module": CANDIDATE_NAME,
        "category": "growth_paper_candidate",
        "Sharpe impact": "positive",
        "return impact": "positive",
        "risk impact": "controlled_growth_risk",
        "evidence strength": "moderate",
        "sample size": 220,
        "final decision": "default growth paper model",
        "reason": "growth_v1_exposure_cap_60; raw_target exact + soft_exit + vol_target_22pct + exposure_cap_60; CAGR 30.30%, Sharpe 1.341, Sortino 2.266, max DD -19.76%; growth_champion_v2",
    }
    legacy_row = {
        "module": LEGACY_UNCAPPED_MODEL,
        "category": "research_only",
        "Sharpe impact": "positive",
        "return impact": "higher_return_higher_drawdown",
        "risk impact": "uncapped_growth_risk",
        "evidence strength": "moderate",
        "sample size": 220,
        "final decision": "research only",
        "reason": "Legacy uncapped growth_v1 kept available for research; not default paper mode.",
    }
    if lifecycle.empty:
        lifecycle = pd.DataFrame([row, legacy_row])
    else:
        lifecycle = lifecycle[~lifecycle["module"].astype(str).isin([CANDIDATE_NAME, LEGACY_UNCAPPED_MODEL])]
        lifecycle = pd.concat([lifecycle, pd.DataFrame([row, legacy_row])], ignore_index=True)
    lifecycle.to_csv(LIFECYCLE_FILE, index=False)

    marker = "===== GROWTH PAPER CANDIDATE ====="
    text = Path(SUMMARY_FILE).read_text(encoding="utf-8", errors="ignore") if Path(SUMMARY_FILE).exists() else ""
    section = (
        f"\n{marker}\n"
        f"{CANDIDATE_NAME}: {CANDIDATE_VARIANT}\n"
        f"Model version: {CANDIDATE_MODEL_VERSION}\n"
        "Status: default growth paper model, paper only, no production promotion.\n"
        "Definition: raw_target_return_exact + soft_exit_rule + volatility_target_22pct + exposure_cap_60 + dual_trend_filter, no leverage.\n"
        "Evidence: governed operational paper pipeline; real capital blocked.\n"
        "Legacy: growth_v1_uncapped remains research-only.\n"
    )
    if marker in text:
        text = text.split(marker)[0].rstrip() + section
    else:
        text = text.rstrip() + section
    Path(SUMMARY_FILE).write_text(text + "\n", encoding="utf-8")

    dashboard = _read_csv(DASHBOARD_FILE)
    new_rows = pd.DataFrame(
        [
            {"section": "Growth Candidate", "metric": "growth_candidate_status", "value": "default growth paper model"},
            {"section": "Growth Candidate", "metric": "growth_candidate_name", "value": CANDIDATE_NAME},
            {"section": "Growth Candidate", "metric": "growth_candidate_variant", "value": CANDIDATE_VARIANT},
            {"section": "Growth Candidate", "metric": "growth_candidate_sharpe", "value": "1.341078"},
            {"section": "Growth Candidate", "metric": "growth_candidate_max_drawdown", "value": "-0.197609"},
            {"section": "Growth Candidate", "metric": "growth_candidate_exposure_cap", "value": "0.60"},
        ]
    )
    if dashboard.empty or not {"section", "metric", "value"}.issubset(dashboard.columns):
        dashboard = new_rows
    else:
        keys = set(zip(new_rows["section"], new_rows["metric"]))
        keep = ~dashboard.apply(lambda r: (r["section"], r["metric"]) in keys, axis=1)
        dashboard = pd.concat([dashboard[keep], new_rows], ignore_index=True)
    dashboard.to_csv(DASHBOARD_FILE, index=False)


def run_growth_candidate_paper_trading(overwrite_same_day: bool = False, allow_proxy_fallback: bool = False) -> dict:
    _register_candidate()
    latest, data_source, raw_features_available, fallback_reason = _latest_candidate_context(allow_proxy_fallback=allow_proxy_fallback)
    raw_target_feature_source = str(latest.get("raw_target_feature_source", "unknown"))
    if not raw_features_available:
        print("WARNING: Growth paper using proxy raw target, not production-parity.")
    date = pd.Timestamp(latest["date"])
    date_str = date.strftime("%Y-%m-%d")
    observed_tickers = _selected_tickers_for_latest(latest)
    if not observed_tickers:
        raise ValueError("Growth candidate selected tickers are missing.")
    tickers = list(observed_tickers)
    schedule_status = scheduler_status(date_str)
    rebalance_due = bool(schedule_status.get("rebalance_due", False))
    configured_cap = _exposure_cap()
    dual_trend_cap = float(latest.get("dual_trend_cap", configured_cap))
    vol_target_exposure = float(latest.get("vol_target_exposure", latest.get("target_exposure", 0.0)))
    uncapped_exposure = float(latest.get("target_exposure", vol_target_exposure))
    exposure = float(np.clip(min(vol_target_exposure, uncapped_exposure, configured_cap, dual_trend_cap), 0.0, 1.0))
    cash = float(1.0 - exposure)
    spy_close = latest.get("spy_close", np.nan)
    spy_ma_200 = latest.get("spy_ma_200", np.nan)
    qqq_close = latest.get("qqq_close", np.nan)
    qqq_ma_200 = latest.get("qqq_ma_200", np.nan)
    spy_below_200d = bool(latest.get("spy_below_200d", False))
    qqq_below_200d = bool(latest.get("qqq_below_200d", False))
    dual_trend_reason = str(latest.get("dual_trend_reason", ""))

    previous = _previous_state()
    previous_weights = previous.set_index("ticker")["paper_position_weight"].to_dict() if not previous.empty and "ticker" in previous.columns else {}
    previous_non_cash_weights = {str(k): float(v) for k, v in previous_weights.items() if str(k).upper() != "CASH"}
    monitoring_only = bool(not rebalance_due and previous_non_cash_weights)
    if monitoring_only:
        tickers = sorted(previous_non_cash_weights)
        exposure = float(sum(previous_non_cash_weights.values()))
        cash = float(previous_weights.get("CASH", max(0.0, 1.0 - exposure)))
        if not previous.empty:
            first_prev = previous.iloc[0]
            vol_target_exposure = float(first_prev.get("vol_target_exposure", vol_target_exposure))
            uncapped_exposure = float(first_prev.get("vol_target_exposure", uncapped_exposure))
            dual_trend_cap = float(first_prev.get("dual_trend_cap", dual_trend_cap))

    prices = _latest_prices(date, tickers, prefer_current=True)
    if prices.dropna().empty:
        raise ValueError("Current prices for growth paper tickers are missing.")

    previous_prices = prices.copy()
    if not previous.empty:
        previous_tickers = previous[previous["ticker"].astype(str).ne("CASH")]["ticker"].astype(str).tolist()
        previous_prices = _latest_prices(date, sorted(set(tickers + previous_tickers + observed_tickers)), prefer_current=True)
    daily_return = _realized_daily_return(previous, previous_prices)
    previous_value = _previous_value()
    portfolio_value = previous_value * (1.0 + daily_return)
    target_weights = {ticker: (previous_non_cash_weights.get(ticker, 0.0) if monitoring_only else exposure / len(tickers)) for ticker in tickers}
    target_weight = exposure / len(tickers)

    current_allocation = _current_growth_allocation()
    if monitoring_only:
        current_allocation = pd.DataFrame(
            {
                "date": [date_str] * len(tickers),
                "ticker": tickers,
                "final_growth_weight": [target_weights[t] for t in tickers],
                "cash_weight": [cash] * len(tickers),
                "rebalance_due": [False] * len(tickers),
                "monitoring_only": [True] * len(tickers),
                "observed_candidate_tickers": [",".join(observed_tickers)] * len(tickers),
                "signals_observed_but_not_executed": [",".join(sorted(set(observed_tickers) ^ set(tickers)))] * len(tickers),
            }
        )
    elif current_allocation.empty or data_source != "current_growth_candidate_allocation":
        current_allocation = pd.DataFrame(
            {
                "date": [date_str] * len(tickers),
                "ticker": tickers,
                "final_growth_weight": [target_weight] * len(tickers),
                "cash_weight": [cash] * len(tickers),
            }
        )
    action_signals, rebalance_report, reconciliation = reconcile_growth_actions(
        current_allocation=current_allocation,
        current_date=date_str,
        portfolio_value=portfolio_value,
        previous_prices=previous_prices,
        overwrite_same_day=overwrite_same_day,
    )
    action_by_ticker = dict(zip(action_signals["ticker"].astype(str), action_signals["action"].astype(str))) if not action_signals.empty else {}
    state_rows = []
    trade_rows = []
    for ticker in tickers:
        current_price = float(prices.get(ticker, np.nan))
        previous_weight = float(previous_weights.get(ticker, 0.0))
        entry_price = current_price
        if not previous.empty and ticker in previous_weights:
            prev_row = previous[previous["ticker"].astype(str).eq(ticker)]
            if not prev_row.empty and np.isfinite(float(prev_row.iloc[0].get("entry_price", np.nan))):
                entry_price = float(prev_row.iloc[0]["entry_price"])
        unrealized = current_price / entry_price - 1.0 if np.isfinite(current_price) and entry_price > 0 else np.nan
        ticker_target_weight = target_weights.get(ticker, target_weight)
        action = action_by_ticker.get(ticker, "BUY" if previous_weight <= 0 else ("REDUCE" if ticker_target_weight < previous_weight else ("INCREASE" if ticker_target_weight > previous_weight else "HOLD")))
        state_rows.append(
            {
                "date": date_str,
                "ticker": ticker,
                "paper_position_weight": ticker_target_weight,
                "paper_position_value": portfolio_value * ticker_target_weight,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_return": unrealized,
                "realized_return": daily_return,
                "action": action,
                "model_mode": CANDIDATE_NAME,
                "growth_paper_variant": CANDIDATE_VARIANT,
                "growth_model_version": CANDIDATE_MODEL_VERSION,
                "exposure_cap": configured_cap,
                "exposure_cap_60": configured_cap,
                "dual_trend_cap": dual_trend_cap,
                "vol_target_exposure": vol_target_exposure,
                "final_exposure": exposure,
                "spy_close": spy_close,
                "spy_ma_200": spy_ma_200,
                "qqq_close": qqq_close,
                "qqq_ma_200": qqq_ma_200,
                "spy_below_200d": spy_below_200d,
                "qqq_below_200d": qqq_below_200d,
                "dual_trend_reason": dual_trend_reason,
                "cash_weight": cash,
                "data_source": data_source,
                "raw_target_current_features_available": raw_features_available,
                "raw_target_feature_source": raw_target_feature_source,
                "rebalance_due": rebalance_due,
                "monitoring_only": monitoring_only,
                "previous_rebalance_date": schedule_status.get("previous_rebalance_date", ""),
                "next_rebalance_date": schedule_status.get("next_rebalance_date", ""),
                "sessions_since_last_rebalance": schedule_status.get("sessions_since_last_rebalance", np.nan),
                "observed_candidate_tickers": ",".join(observed_tickers),
            }
        )
        trade_rows.append(
            {
                "date": date_str,
                "ticker": ticker,
                "action": action,
                "previous_weight": previous_weight,
                "new_weight": ticker_target_weight,
                "trade_weight_change": ticker_target_weight - previous_weight,
                "execution_price": current_price,
                "reason": CANDIDATE_VARIANT,
                "model_mode": CANDIDATE_NAME,
                "growth_paper_variant": CANDIDATE_VARIANT,
                "growth_model_version": CANDIDATE_MODEL_VERSION,
                "exposure_cap": configured_cap,
                "exposure_cap_60": configured_cap,
                "dual_trend_cap": dual_trend_cap,
                "vol_target_exposure": vol_target_exposure,
                "final_exposure": exposure,
                "spy_close": spy_close,
                "spy_ma_200": spy_ma_200,
                "qqq_close": qqq_close,
                "qqq_ma_200": qqq_ma_200,
                "spy_below_200d": spy_below_200d,
                "qqq_below_200d": qqq_below_200d,
                "dual_trend_reason": dual_trend_reason,
                "data_source": data_source,
                "raw_target_current_features_available": raw_features_available,
                "raw_target_feature_source": raw_target_feature_source,
            }
        )
    state_rows.append(
        {
            "date": date_str,
            "ticker": "CASH",
            "paper_position_weight": cash,
            "paper_position_value": portfolio_value * cash,
            "entry_price": 1.0,
            "current_price": 1.0,
            "unrealized_return": 0.0,
            "realized_return": 0.0,
            "action": "HOLD",
            "model_mode": CANDIDATE_NAME,
            "growth_paper_variant": CANDIDATE_VARIANT,
            "growth_model_version": CANDIDATE_MODEL_VERSION,
            "exposure_cap": configured_cap,
            "cash_weight": cash,
            "data_source": data_source,
            "raw_target_current_features_available": raw_features_available,
            "raw_target_feature_source": raw_target_feature_source,
            "rebalance_due": rebalance_due,
            "monitoring_only": monitoring_only,
            "previous_rebalance_date": schedule_status.get("previous_rebalance_date", ""),
            "next_rebalance_date": schedule_status.get("next_rebalance_date", ""),
            "sessions_since_last_rebalance": schedule_status.get("sessions_since_last_rebalance", np.nan),
            "observed_candidate_tickers": ",".join(observed_tickers),
        }
    )
    quality_fail_reasons = _quality_reason_map()
    removed = sorted(set(previous_weights) - set(tickers) - {"CASH"})
    for ticker in removed:
        removal_reason = "removed_by_universe_quality_filter: " + quality_fail_reasons[ticker] if ticker in quality_fail_reasons else "removed_by_growth_candidate"
        trade_rows.append(
            {
                "date": date_str,
                "ticker": ticker,
                "action": "SELL",
                "previous_weight": previous_weights.get(ticker, 0.0),
                "new_weight": 0.0,
                "trade_weight_change": -float(previous_weights.get(ticker, 0.0)),
                "execution_price": float(previous_prices.get(ticker, np.nan)),
                "reason": removal_reason,
                "model_mode": CANDIDATE_NAME,
                "growth_paper_variant": CANDIDATE_VARIANT,
                "growth_model_version": CANDIDATE_MODEL_VERSION,
                "exposure_cap": configured_cap,
            }
        )

    state_df = pd.DataFrame(state_rows)
    trades_df = signals_to_trade_rows(action_signals, model_mode=CANDIDATE_NAME, variant=CANDIDATE_VARIANT)
    if not trades_df.empty:
        trades_df["exposure_cap"] = configured_cap
        trades_df["exposure_cap_60"] = configured_cap
        trades_df["dual_trend_cap"] = dual_trend_cap
        trades_df["vol_target_exposure"] = vol_target_exposure
        trades_df["final_exposure"] = exposure
        trades_df["spy_close"] = spy_close
        trades_df["spy_ma_200"] = spy_ma_200
        trades_df["qqq_close"] = qqq_close
        trades_df["qqq_ma_200"] = qqq_ma_200
        trades_df["spy_below_200d"] = spy_below_200d
        trades_df["qqq_below_200d"] = qqq_below_200d
        trades_df["dual_trend_reason"] = dual_trend_reason
        trades_df["data_source"] = data_source
        trades_df["raw_target_current_features_available"] = raw_features_available
        trades_df["raw_target_feature_source"] = raw_target_feature_source
    turnover = float(reconciliation.get("turnover", 0.0))
    reconciliation_passed = bool(reconciliation.get("reconciliation_passed", False))
    if not reconciliation_passed:
        print('WARNING: Rebalance reconciliation failed')
    perf_existing = _read_csv(PERFORMANCE_FILE)
    temp_perf = pd.concat(
        [
            perf_existing,
            pd.DataFrame(
                [
                    {
                        "date": date_str,
                        "model_mode": CANDIDATE_NAME,
                        "growth_paper_variant": CANDIDATE_VARIANT,
                "growth_model_version": CANDIDATE_MODEL_VERSION,
                        "portfolio_value": portfolio_value,
                        "daily_return": daily_return,
                        "cash_weight": cash,
                        "exposure": exposure,
                        "vol_target_exposure": vol_target_exposure,
                        "exposure_cap_60": configured_cap,
                        "dual_trend_cap": dual_trend_cap,
                        "final_exposure": exposure,
                        "turnover": turnover,
                        "data_source": data_source,
                        "raw_target_current_features_available": raw_features_available,
                        "raw_target_feature_source": raw_target_feature_source,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metrics = _performance_metrics(temp_perf)
    perf_row = pd.DataFrame(
        [
            {
                "date": date_str,
                "model_mode": CANDIDATE_NAME,
                "growth_paper_variant": CANDIDATE_VARIANT,
                "growth_model_version": CANDIDATE_MODEL_VERSION,
                "portfolio_value": portfolio_value,
                "daily_return": daily_return,
                "cumulative_return": metrics["cumulative_return"],
                "volatility": metrics["volatility"],
                "Sharpe": metrics["Sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "cash_weight": cash,
                "exposure": exposure,
                "uncapped_exposure": uncapped_exposure,
                "vol_target_exposure": vol_target_exposure,
                "exposure_cap": configured_cap,
                "exposure_cap_60": configured_cap,
                "dual_trend_cap": dual_trend_cap,
                "final_exposure": exposure,
                "spy_close": spy_close,
                "spy_ma_200": spy_ma_200,
                "qqq_close": qqq_close,
                "qqq_ma_200": qqq_ma_200,
                "spy_below_200d": spy_below_200d,
                "qqq_below_200d": qqq_below_200d,
                "dual_trend_reason": dual_trend_reason,
                "turnover": turnover,
                "data_source": data_source,
                "raw_target_current_features_available": raw_features_available,
                "raw_target_feature_source": raw_target_feature_source,
                "fallback_reason": fallback_reason,
            }
        ]
    )

    state_added, state_skipped, state_overwritten = _append_or_update(STATE_FILE, state_df, date_str, overwrite_same_day)
    trades_added, trades_skipped, trades_overwritten = _append_or_update(TRADES_FILE, trades_df, date_str, overwrite_same_day)
    perf_added, perf_skipped, perf_overwritten = _append_or_update(PERFORMANCE_FILE, perf_row, date_str, overwrite_same_day)
    monitor = _build_monitor(date_str, state_df, perf_row, temp_perf)
    monitor["data_source"] = data_source
    monitor["raw_target_current_features_available"] = raw_features_available
    monitor["raw_target_feature_source"] = raw_target_feature_source
    monitor["fallback_reason"] = fallback_reason
    monitor["growth_paper_variant"] = CANDIDATE_VARIANT
    monitor["growth_model_version"] = CANDIDATE_MODEL_VERSION
    monitor["exposure_cap"] = configured_cap
    monitor["exposure_cap_60"] = configured_cap
    monitor["dual_trend_cap"] = dual_trend_cap
    monitor["vol_target_exposure"] = vol_target_exposure
    monitor["final_exposure"] = exposure
    monitor["spy_close"] = spy_close
    monitor["spy_ma_200"] = spy_ma_200
    monitor["qqq_close"] = qqq_close
    monitor["qqq_ma_200"] = qqq_ma_200
    monitor["spy_below_200d"] = spy_below_200d
    monitor["qqq_below_200d"] = qqq_below_200d
    monitor["dual_trend_reason"] = dual_trend_reason
    monitor["uncapped_exposure"] = uncapped_exposure
    monitor["rebalance_due"] = rebalance_due
    monitor["monitoring_only"] = monitoring_only
    monitor["previous_rebalance_date"] = schedule_status.get("previous_rebalance_date", "")
    monitor["next_rebalance_date"] = schedule_status.get("next_rebalance_date", "")
    monitor["sessions_since_last_rebalance"] = schedule_status.get("sessions_since_last_rebalance", np.nan)
    monitor["observed_candidate_tickers"] = ",".join(observed_tickers)
    monitor_added, monitor_skipped, monitor_overwritten = _append_or_update(MONITOR_FILE, monitor, date_str, overwrite_same_day)

    print("\n===== GROWTH CHAMPION FINAL PAPER MODE =====")
    print(f"model name: {CANDIDATE_NAME}")
    print(f"variant: {CANDIDATE_VARIANT}")
    print(f"model version: {CANDIDATE_MODEL_VERSION}")
    print(f"date: {date_str}")
    print(f"data source: {data_source}")
    print(f"raw target current features available: {raw_features_available}")
    print(f"raw target feature source: {raw_target_feature_source}")
    print(f"fallback reason: {fallback_reason}")
    print(f"rebalance due: {rebalance_due}")
    print(f"monitoring only: {monitoring_only}")
    print("previous rebalance date: " + str(schedule_status.get("previous_rebalance_date", "")))
    print("next rebalance date: " + str(schedule_status.get("next_rebalance_date", "")))
    print("sessions since last rebalance: " + str(schedule_status.get("sessions_since_last_rebalance", np.nan)))
    print(f"exposure cap 60: {configured_cap:.4f}")
    print(f"SPY close / 200D MA / below: {float(spy_close):.4f} / {float(spy_ma_200):.4f} / {spy_below_200d}")
    print(f"QQQ close / 200D MA / below: {float(qqq_close):.4f} / {float(qqq_ma_200):.4f} / {qqq_below_200d}")
    print(f"dual trend cap: {dual_trend_cap:.4f} ({dual_trend_reason})")
    print(f"vol target exposure uncapped: {uncapped_exposure:.4f}")
    print(f"final exposure: {exposure:.4f}")
    print(f"cash: {cash:.4f}")
    print(f"portfolio value: {portfolio_value:.2f}")
    print(f"daily return: {daily_return:.6f}")
    print(f"turnover: {turnover:.4f}")
    print(f"rebalance reconciliation passed: {reconciliation_passed}")
    print("\ncurrent holdings:")
    print(state_df[["ticker", "paper_position_weight", "paper_position_value", "current_price", "action"]].to_string(index=False))
    print("\n===== GROWTH PAPER ACTION RECONCILIATION =====")
    action_cols = ["ticker", "action", "old_weight", "new_weight", "weight_change", "reason"]
    print(action_signals[action_cols].to_string(index=False))
    print("\n===== GROWTH PAPER REBALANCE REPORT =====")
    print(rebalance_report.to_string(index=False))

    print("\n===== GROWTH PAPER MONITOR =====")
    print(monitor.to_string(index=False))
    print(f"\nstate rows added/skipped/overwritten: {state_added}/{state_skipped}/{state_overwritten}")
    print(f"trade rows added/skipped/overwritten: {trades_added}/{trades_skipped}/{trades_overwritten}")
    print(f"performance rows added/skipped/overwritten: {perf_added}/{perf_skipped}/{perf_overwritten}")
    print(f"monitor rows added/skipped/overwritten: {monitor_added}/{monitor_skipped}/{monitor_overwritten}")
    print("real trading: disabled")
    print("production change: none")
    print(f"Saved: {Path(STATE_FILE).resolve()}")
    print(f"Saved: {Path(TRADES_FILE).resolve()}")
    print(f"Saved: {Path(PERFORMANCE_FILE).resolve()}")
    print(f"Saved: {Path(MONITOR_FILE).resolve()}")
    return {
        "date": date_str,
        "state_rows_added": state_added,
        "trades_rows_added": trades_added,
        "performance_rows_added": perf_added,
        "monitor_rows_added": monitor_added,
        "status": "ok" if perf_added or overwrite_same_day else "skipped_duplicate",
        "turnover": turnover,
        "reconciliation_passed": reconciliation_passed,
    }


def _build_monitor(date: str, state_df: pd.DataFrame, perf_row: pd.DataFrame, perf_all: pd.DataFrame) -> pd.DataFrame:
    perf = perf_all.copy()
    perf["daily_return"] = _num(perf.get("daily_return", pd.Series(index=perf.index, dtype=float))).fillna(0.0)
    returns = perf["daily_return"].tail(60)
    rolling_sharpe = 0.0
    if len(returns) >= 10 and returns.std(ddof=0) > 0:
        rolling_sharpe = float((returns.mean() * 252) / (returns.std(ddof=0) * np.sqrt(252)))
    latest = perf_row.iloc[0]
    weights = state_df[state_df["ticker"].ne("CASH")]["paper_position_weight"]
    top_concentration = float(weights.max()) if not weights.empty else 0.0
    flags = []
    if float(latest["max_drawdown"]) < -0.20:
        flags.append("drawdown_worse_than_minus_20")
    if rolling_sharpe < 0 and len(returns) >= 10:
        flags.append("rolling_sharpe_below_zero")
    if float(latest["exposure"]) < 0.50:
        flags.append("exposure_stuck_too_low")
    if float(latest["exposure"]) > 0.95:
        flags.append("exposure_stuck_too_high")
    if float(latest["turnover"]) > 0.75:
        flags.append("turnover_too_high")
    if top_concentration > 0.35:
        flags.append("top_ticker_concentration_too_high")
    if len(perf) < 20:
        flags.append("paper_history_too_short")
    return pd.DataFrame(
        [
            {
                "date": date,
                "candidate": CANDIDATE_NAME,
                "paper_cumulative_return": latest["cumulative_return"],
                "paper_daily_return": latest["daily_return"],
                "paper_sharpe": latest["Sharpe"],
                "rolling_sharpe_60_period_proxy": rolling_sharpe,
                "paper_max_drawdown": latest["max_drawdown"],
                "cash": latest["cash_weight"],
                "exposure": latest["exposure"],
                "turnover": latest["turnover"],
                "top_ticker_concentration": top_concentration,
                "risk_flags": ",".join(flags) if flags else "none",
                "promotion_status": "paper trading allowed; production promotion blocked",
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run growth candidate v1 paper trading simulation.")
    parser.add_argument("--overwrite-same-day", action="store_true")
    parser.add_argument("--allow-proxy-fallback", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_growth_candidate_paper_trading(overwrite_same_day=args.overwrite_same_day, allow_proxy_fallback=args.allow_proxy_fallback)











