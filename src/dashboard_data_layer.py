from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_NAME = "growth_champion_final"
MODEL_VERSION = "growth_champion_final_v1_0_frozen"
VARIANT = "growth_v1_exposure_cap_60_dual_trend_filter"

CSV_FILES: dict[str, str] = {
    "official_performance": "growth_official_paper_performance.csv",
    "official_state": "growth_official_paper_state.csv",
    "official_actions": "growth_official_paper_actions.csv",
    "official_rebalance_report": "growth_official_paper_rebalance_report.csv",
    "official_trades": "growth_official_paper_trades.csv",
    "official_trade_lifecycle": "growth_official_trade_lifecycle.csv",
    "official_monitor": "growth_official_paper_monitor.csv",
    "official_tracking": "growth_official_live_tracking.csv",
    "official_cost_ledger": "growth_official_estimated_cost_ledger.csv",
    "official_cost_duplication_audit": "official_cost_duplication_audit.csv",
    "official_accounting_audit": "official_forward_accounting_audit.csv",
    "official_accounting_reconciliation": "official_accounting_reconciliation.csv",
    "official_position_pnl": "growth_official_position_pnl.csv",
    "official_realized_pnl": "growth_official_realized_pnl.csv",
    "official_benchmark_daily": "growth_official_benchmark_daily.csv",
    "official_benchmark_equity": "growth_official_benchmark_equity.csv",
    "official_integrity": "official_paper_integrity_status.csv",
    "official_daily_status": "official_paper_daily_run_status.csv",
    "official_version_history": "official_paper_version_history.csv",
    "official_market_data_governance": "official_market_data_governance.csv",
    "official_market_data_integrity": "official_market_data_integrity.csv",
    "official_holding_metadata": "official_holding_metadata.csv",
    "secondary_provider_status": "secondary_provider_status.csv",
    "multi_source_price_audit": "multi_source_price_audit.csv",
    "market_data_governance": "market_data_governance.csv",
    "debug_performance": "growth_candidate_paper_performance.csv",
    "debug_state": "growth_candidate_paper_state.csv",
    "debug_actions": "growth_candidate_action_signals.csv",
    "debug_rebalance_report": "growth_candidate_rebalance_report.csv",
    "debug_monitor": "growth_candidate_paper_monitor.csv",
    "debug_benchmark_daily": "benchmark_daily_returns.csv",
    "debug_benchmark_equity": "benchmark_equity_curves.csv",
    "reconstructed_daily": "reconstructed_growth_long_horizon_daily_returns.csv",
    "historical_2008_backtest_daily": "growth_champion_reconstructed_stress_daily.csv",
    "reconstructed_results": "reconstructed_growth_long_horizon_results.csv",
    "final_results": "growth_final_selection_results.csv",
    "final_daily": "growth_final_selection_daily_returns.csv",
    "final_drawdowns": "growth_final_selection_drawdowns.csv",
    "final_stress": "growth_final_selection_stress_periods.csv",
    "cost_results": "growth_final_cost_slippage_results.csv",
    "cost_benchmarks": "growth_final_after_costs_vs_benchmarks.csv",
    "advanced_costs": "advanced_execution_costs.csv",
    "capacity": "capacity_analysis.csv",
    "growth_capacity": "growth_capacity_analysis.csv",
    "operational_capacity": "growth_operational_capacity_report.csv",
    "vol_fresh": "growth_volatility_targeting_fresh.csv",
    "vol_pipeline_audit": "growth_volatility_pipeline_audit.csv",
    "parameter_stability": "parameter_stability_map.csv",
    "parameter_sensitivity": "parameter_sensitivity_results.csv",
    "hmm_oos": "hmm_out_of_sample_results.csv",
    "hmm_comparison": "hmm_model_comparison.csv",
    "hmm_governance": "hmm_governance.csv",
    "current_features": "current_growth_features.csv",
    "current_allocation": "current_growth_candidate_allocation.csv",
    "universe_quality": "current_growth_universe_quality.csv",
    "tradability_report": "growth_tradability_filter_report.csv",
    "tradability_exclusions": "growth_tradability_exclusions.csv",
    "holdings_sanity": "final_selected_holdings_audit.csv",
    "current_raw_target_features": "current_raw_target_features.csv",
    "universe_quality_report": "growth_universe_quality_report.csv",
    "universe_exclusions": "growth_universe_exclusions.csv",
    "holdings_replacements": "final_selected_holdings_replacements.csv",
    "portfolio_explainability": "growth_portfolio_explainability.csv",
    "top20_candidates": "growth_top20_candidates.csv",
    "rejection_report": "growth_candidate_rejection_report.csv",
    "pending_signals": "growth_pending_decision_signals.csv",
    "rebalance_schedule": "growth_rebalance_schedule.csv",
    "risk_contribution": "risk_contribution_reconciliation.csv",
    "execution_cost_reconciliation": "execution_cost_reconciliation.csv",
    "decision_funnel": "growth_decision_funnel.csv",
    "benchmark_chart_audit": "benchmark_chart_source_audit.csv",
    "benchmark_chart_reconciliation": "benchmark_chart_reconciliation.csv",
    "full_cscv_results": "full_cscv_results.csv",
    "pbo_distribution": "pbo_distribution.csv",
    "deflated_sharpe_exact": "deflated_sharpe_exact.csv",
    "effective_trial_count": "effective_trial_count.csv",
    "reality_check_results": "reality_check_results.csv",
    "purged_walk_forward_results": "purged_walk_forward_results.csv",
    "purged_walk_forward_folds": "purged_walk_forward_folds.csv",
    "locked_holdout_results": "locked_holdout_results.csv",
    "ic_decay_results": "ic_decay_results.csv",
    "robustness_plateau_analysis": "robustness_plateau_analysis.csv",
    "rolling_feature_ic": "rolling_feature_ic.csv",
    "alpha_decay_curve": "alpha_decay_curve.csv",
    "structural_break_results": "structural_break_results.csv",
    "feature_distribution_drift": "feature_distribution_drift.csv",
    "garch_model_comparison": "garch_model_comparison.csv",
    "hmm_incremental_portfolio_results": "hmm_incremental_portfolio_results.csv",
    "governed_experiment_registry": "governed_experiment_registry.csv",
    "frozen_champion_registry": "frozen_champion_registry.csv",
    "anti_overfitting_governance": "anti_overfitting_governance.csv",
    "out_of_sample_governance": "out_of_sample_governance.csv",
    "parameter_governance": "parameter_governance.csv",
    "alpha_decay_governance": "alpha_decay_governance.csv",
    "parameter_stability_map": "parameter_stability_map.csv",
    "parameter_sensitivity_results": "parameter_sensitivity_results.csv",
    "hmm_model_comparison": "hmm_model_comparison.csv",
    "growth_system_integrity_report": "growth_system_integrity_report.txt",
    "growth_pipeline_integrity_report": "growth_pipeline_integrity_report.csv",
    "growth_pipeline_stage_validation": "growth_pipeline_stage_validation.csv",
    "growth_rebalance_parity_governance": "growth_rebalance_parity_governance.csv",
    "growth_canonical_governance": "growth_canonical_governance.csv",
    "governance_incident_registry": "governance_incident_registry.csv",
    "phase102_report": "phase102_official_accounting_report.txt",
    "phase109_report": "phase109_risk_terminal_report.txt",
    "phase110_report": "phase110_execution_terminal_report.txt",
    "phase111_report": "phase111_research_terminal_report.txt",
    "growth_paper_governance_history": "growth_paper_governance_history.csv",
    "canonical_price_history": "canonical_price_history.csv",
    "garch_model_comparison": "garch_model_comparison.csv",
    "quant_lab_source_audit": "quant_lab_source_audit.csv",
    "quant_lab_surface_integrity": "quant_lab_surface_integrity.csv",
    "quant_lab_performance_audit": "quant_lab_performance_audit.csv",
    "phase114_report": "phase114_quant_lab_report.txt",
    "historical_replay_source_audit": "historical_replay_source_audit.csv",
    "historical_replay_integrity": "historical_replay_integrity.csv",
    "historical_replay_validation": "historical_replay_validation.csv",
    "phase115_report": "phase115_historical_replay_report.txt",
    "decision_engine_source_audit": "decision_engine_source_audit.csv",
    "decision_engine_integrity": "decision_engine_integrity.csv",
    "decision_funnel_reconciliation": "decision_funnel_reconciliation.csv",
    "phase113_report": "phase113_decision_engine_report.txt",
    "mission_control_source_audit": "mission_control_source_audit.csv",
    "mission_control_integrity": "mission_control_integrity.csv",
    "mission_control_runtime": "mission_control_runtime.csv",
    "phase116_report": "phase116_mission_control_report.txt",
    "alert_engine_source_audit": "alert_engine_source_audit.csv",
    "alert_engine_integrity": "alert_engine_integrity.csv",
    "alert_history": "alert_history.csv",
    "active_alerts": "active_alerts.csv",
    "phase117_report": "phase117_alert_engine_report.txt",
    "alpha_attribution": "alpha_attribution.csv",
    "factor_contribution_phase120": "factor_contribution.csv",
    "forecast_analysis_phase120": "forecast_analysis.csv",
    "ranking_analysis_phase120": "ranking_analysis.csv",
    "position_sizing_analysis_phase120": "position_sizing_analysis.csv",
    "cash_drag_analysis_phase120": "cash_drag_analysis.csv",
    "turnover_analysis_phase120": "turnover_analysis.csv",
    "cost_attribution_phase120": "cost_attribution.csv",
    "holding_period_analysis_phase120": "holding_period_analysis.csv",
    "alpha_reconciliation": "alpha_reconciliation.csv",
    "alpha_attribution_integrity": "alpha_attribution_integrity.csv",
    "phase120_report": "phase120_alpha_attribution_report.txt",
}

SCOPE_CONFIG = {
    "Official Forward Paper": {
        "performance": "official_performance",
        "state": "official_state",
        "actions": "official_actions",
        "rebalance": "official_rebalance_report",
        "monitor": "official_monitor",
        "benchmark_equity": "official_benchmark_equity",
        "benchmark_daily": "official_benchmark_daily",
        "namespace": "official_forward_paper",
    },
    "Historical 2008 Backtest": {
        "performance": "historical_2008_backtest_daily",
        "state": "historical_2008_backtest_daily",
        "actions": "historical_2008_backtest_daily",
        "rebalance": "historical_2008_backtest_daily",
        "monitor": "reconstructed_results",
        "benchmark_equity": "historical_2008_backtest_daily",
        "benchmark_daily": "historical_2008_backtest_daily",
        "namespace": "historical_2008_reconstructed_backtest",
    },
    "Historical Debug Replay": {
        "performance": "historical_2008_backtest_daily",
        "state": "historical_2008_backtest_daily",
        "actions": "historical_2008_backtest_daily",
        "rebalance": "historical_2008_backtest_daily",
        "monitor": "reconstructed_results",
        "benchmark_equity": "historical_2008_backtest_daily",
        "benchmark_daily": "historical_2008_backtest_daily",
        "namespace": "historical_2008_reconstructed_backtest",
    },
    "Reconstructed Stress": {
        "performance": "historical_2008_backtest_daily",
        "state": "historical_2008_backtest_daily",
        "actions": "historical_2008_backtest_daily",
        "rebalance": "historical_2008_backtest_daily",
        "monitor": "reconstructed_results",
        "benchmark_equity": "historical_2008_backtest_daily",
        "benchmark_daily": "historical_2008_backtest_daily",
        "namespace": "historical_2008_reconstructed_backtest",
    },
}

DATE_COLUMNS = [
    "date",
    "Date",
    "signal_date",
    "economic_application_date",
    "entry_date",
    "exit_date",
    "start_date",
    "end_date",
    "timestamp",
]


def numeric(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
            if col != "timestamp":
                out[col] = out[col].dt.normalize()
    return out


def load_csv(path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = Path(path)
    meta: dict[str, Any] = {"path": path, "exists": p.exists(), "rows": 0, "columns": [], "error": ""}
    if not p.exists():
        return pd.DataFrame(), meta
    try:
        df = normalize_dates(pd.read_csv(p))
        meta.update({"rows": len(df), "columns": list(df.columns)})
        if "date" in df.columns and not df.empty:
            dates = pd.to_datetime(df["date"], errors="coerce")
            meta["start_date"] = str(dates.min().date()) if dates.notna().any() else ""
            meta["end_date"] = str(dates.max().date()) if dates.notna().any() else ""
        return df, meta
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
        return pd.DataFrame(), meta


def load_all() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    metas = []
    for name, path in CSV_FILES.items():
        df, meta = load_csv(path)
        data[name] = df
        meta["name"] = name
        metas.append(meta)
    cols = ["name", "path", "exists", "rows", "start_date", "end_date", "columns", "error"]
    diag = pd.DataFrame(metas)
    for col in cols:
        if col not in diag.columns:
            diag[col] = ""
    return data, diag[cols]


def latest(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df.copy()
    clean = df.dropna(subset=[date_col]).copy()
    if clean.empty:
        return clean
    return clean[clean[date_col].eq(clean[date_col].max())].copy()


def _historical_backtest_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy().sort_values("date") if "date" in df.columns else df.copy()
    if "gross_daily_return" in out.columns:
        out["daily_return"] = numeric(out["gross_daily_return"])
        out["return"] = numeric(out["gross_daily_return"])
    if "net_daily_return" in out.columns:
        out["estimated_net_daily_return"] = numeric(out["net_daily_return"])
    if "gross_equity" in out.columns:
        out["portfolio_value"] = numeric(out["gross_equity"]) * 100000
        out["gross_portfolio_value"] = numeric(out["gross_equity"]) * 100000
        out["cumulative_return"] = numeric(out["gross_equity"]) - 1
        out["gross_cumulative_return"] = numeric(out["gross_equity"]) - 1
    if "net_equity" in out.columns:
        out["estimated_net_portfolio_value"] = numeric(out["net_equity"]) * 100000
        out["estimated_net_equity"] = numeric(out["net_equity"]) * 100000
        out["estimated_net_cumulative_return"] = numeric(out["net_equity"]) - 1
    if "cash" in out.columns and "cash_weight" not in out.columns:
        out["cash_weight"] = numeric(out["cash"])
    if "drawdown" in out.columns and "current_drawdown" not in out.columns:
        out["current_drawdown"] = numeric(out["drawdown"])
    if "holdings" in out.columns and "ticker" not in out.columns:
        latest_row = out.tail(1).iloc[0]
        tickers = [t.strip() for t in str(latest_row.get("holdings", "")).split(",") if t.strip()]
        weights_raw = [w.strip() for w in str(latest_row.get("weights", "")).split(",") if w.strip()]
        equal_weight = float(pd.to_numeric(pd.Series([latest_row.get("exposure", 0)]), errors="coerce").fillna(0).iloc[0]) / max(
            len(tickers), 1
        )
        weights = []
        for w in weights_raw:
            val = pd.to_numeric(pd.Series([w]), errors="coerce").iloc[0]
            weights.append(float(val) if pd.notna(val) else equal_weight)
        if tickers:
            rows = []
            for i, ticker in enumerate(tickers):
                weight = weights[i] if i < len(weights) and pd.notna(weights[i]) else equal_weight
                rows.append(
                    {
                        "date": latest_row.get("date"),
                        "ticker": ticker,
                        "paper_position_weight": weight,
                        "action": "HISTORICAL_BACKTEST_HOLDING",
                        "data_mode": latest_row.get("data_mode", "reconstructed"),
                    }
                )
            rows.append(
                {
                    "date": latest_row.get("date"),
                    "ticker": "CASH",
                    "paper_position_weight": latest_row.get("cash", 0),
                    "action": "HISTORICAL_BACKTEST_CASH",
                    "data_mode": latest_row.get("data_mode", "reconstructed"),
                }
            )
            out.attrs["latest_holdings_frame"] = pd.DataFrame(rows)
    return out


def scope_data(data: dict[str, pd.DataFrame], scope: str) -> dict[str, pd.DataFrame]:
    cfg = SCOPE_CONFIG.get(scope, SCOPE_CONFIG["Official Forward Paper"])
    scoped = {k: data.get(v, pd.DataFrame()) for k, v in cfg.items() if k != "namespace"}
    if scope in {"Historical 2008 Backtest", "Historical Debug Replay", "Reconstructed Stress"}:
        hist = _historical_backtest_frame(scoped.get("performance", pd.DataFrame()))
        scoped["performance"] = hist
        scoped["benchmark_equity"] = hist
        scoped["benchmark_daily"] = hist
        scoped["rebalance"] = (
            hist[hist.get("rebalance_due", pd.Series(False, index=hist.index)).astype(str).str.lower().isin(["true", "1"])]
            if not hist.empty and "rebalance_due" in hist.columns
            else hist
        )
        scoped["actions"] = scoped["rebalance"]
        scoped["state"] = hist.attrs.get("latest_holdings_frame", pd.DataFrame())
    return scoped


def get_scope_namespace(scope: str) -> str:
    return SCOPE_CONFIG.get(scope, SCOPE_CONFIG["Official Forward Paper"])["namespace"]


def official_start_date(data: dict[str, pd.DataFrame]) -> str:
    perf = data.get("official_performance", pd.DataFrame())
    if perf.empty or "date" not in perf.columns:
        return "unavailable"
    return str(perf["date"].min().date())


def latest_market_date(data: dict[str, pd.DataFrame]) -> str:
    candidates = []
    for key in ["official_performance", "official_state", "official_benchmark_daily", "current_features"]:
        df = data.get(key, pd.DataFrame())
        if not df.empty and "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce").max()
            if pd.notna(d):
                candidates.append(d)
    return str(max(candidates).date()) if candidates else "unavailable"


def _clean_date_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "unavailable"}:
        return ""
    dt = pd.to_datetime(text, errors="coerce")
    return str(dt.date()) if pd.notna(dt) else text[:10]


def next_rebalance_date(data: dict[str, pd.DataFrame]) -> str:
    """Return the next official rebalance date, deriving it read-only when the run status is blank."""
    daily = latest(data.get("official_daily_status", pd.DataFrame()))
    if not daily.empty:
        row = daily.iloc[-1]
        for col in ["next_rebalance_date", "next_rebalance"]:
            val = _clean_date_value(row.get(col, ""))
            if val:
                return val

    schedule = data.get("rebalance_schedule", pd.DataFrame())
    if schedule.empty or "market_date" not in schedule.columns:
        return "unavailable"
    work = schedule.copy()
    work["market_date"] = pd.to_datetime(work["market_date"], errors="coerce")
    work = work.dropna(subset=["market_date"]).sort_values("market_date")
    if work.empty:
        return "unavailable"

    latest_date = pd.to_datetime(latest_market_date(data), errors="coerce")
    if pd.isna(latest_date):
        latest_date = work["market_date"].max()

    # Prefer an explicit future rebalance from the schedule when available.
    future = work[
        (work["market_date"] > latest_date)
        & work.get("rebalance_due", pd.Series(False, index=work.index)).astype(str).str.lower().isin(["true", "1"])
    ]
    if not future.empty:
        return str(future.iloc[0]["market_date"].date())

    due_mask = work.get("rebalance_due", pd.Series(False, index=work.index)).astype(str).str.lower().isin(["true", "1"])
    last_due = work[due_mask & (work["market_date"] <= latest_date)]
    if last_due.empty:
        return "unavailable"
    last_due_date = last_due.iloc[-1]["market_date"]
    target = last_due_date
    for _ in range(5):
        target = target + pd.offsets.BDay(1)
    return str(pd.Timestamp(target).date())


def _benchmark_returns_from_cache(dates: pd.Series) -> pd.DataFrame:
    clean_dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    if clean_dates.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": clean_dates})
    root = Path("yahoo_ohlcv_price_cache")
    for ticker, col in [("SPY", "SPY"), ("QQQ", "QQQ")]:
        path = root / f"{ticker}.csv"
        if not path.exists():
            continue
        try:
            px = pd.read_csv(path)
        except Exception:
            continue
        date_col = "Date" if "Date" in px.columns else "date" if "date" in px.columns else None
        price_col = "Adj Close" if "Adj Close" in px.columns else "Close" if "Close" in px.columns else None
        if date_col is None or price_col is None:
            continue
        px = px[[date_col, price_col]].copy()
        px["date"] = pd.to_datetime(px[date_col], errors="coerce").dt.normalize()
        px[col + "_price"] = numeric(px[price_col])
        px = px.dropna(subset=["date", col + "_price"]).sort_values("date")
        # Use point-in-time benchmark prices at the same strategy observation dates.
        # Then pct_change across those observation dates captures the whole interval return,
        # instead of incorrectly using only one daily return per rebalance period.
        aligned = pd.merge_asof(
            out[["date"]].sort_values("date"), px[["date", col + "_price"]].sort_values("date"), on="date", direction="backward"
        )
        aligned[col] = aligned[col + "_price"].pct_change().fillna(0.0)
        out = out.merge(aligned[["date", col]], on="date", how="left")
    return out


def _fill_historical_benchmarks(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty or "date" not in out.columns:
        return out
    needs_spy = "SPY" not in out.columns or numeric(out.get("SPY", pd.Series(dtype=float))).notna().sum() == 0
    needs_qqq = "QQQ" not in out.columns or numeric(out.get("QQQ", pd.Series(dtype=float))).notna().sum() == 0
    if not needs_spy and not needs_qqq:
        return out
    bench = _benchmark_returns_from_cache(out["date"])
    if bench.empty:
        return out
    merged = out.merge(bench, on="date", how="left", suffixes=("", "_cache"))
    for col in ["SPY", "QQQ"]:
        cache_col = col + "_cache"
        if cache_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[cache_col]
            else:
                merged[col] = numeric(merged[col]).where(numeric(merged[col]).notna(), numeric(merged[cache_col]))
            merged = merged.drop(columns=[cache_col])
    return merged


def benchmark_curve_for_scope(data: dict[str, pd.DataFrame], scope: str) -> pd.DataFrame:
    scoped = scope_data(data, scope)
    eq = scoped.get("benchmark_equity", pd.DataFrame()).copy()
    if scope == "Official Forward Paper" and not eq.empty:
        cols = {
            "growth_gross_cumulative_pct": "Growth Gross",
            "growth_net_cumulative_pct": "Growth Estimated Net",
            "SPY_cumulative_pct": "SPY",
            "QQQ_cumulative_pct": "QQQ",
        }
        out = eq[["date"]].copy() if "date" in eq.columns else pd.DataFrame()
        for raw, label in cols.items():
            if raw in eq.columns:
                out[label] = numeric(eq[raw])
        return out.dropna(subset=[c for c in out.columns if c != "date"], how="all") if not out.empty else out
    if not eq.empty and "date" in eq.columns:
        out = eq[["date"]].copy()
        mapping = {
            "growth_cumulative_return_pct": "Growth Champion Final",
            "spy_cumulative_return_pct": "SPY",
            "qqq_cumulative_return_pct": "QQQ",
        }
        for raw, label in mapping.items():
            if raw in eq.columns:
                out[label] = numeric(eq[raw])
        # Historical reconstructed files sometimes store daily benchmark columns as blank.
        # If cumulative benchmark series are unavailable, fall through to daily/cache reconstruction.
        non_date_cols = [c for c in out.columns if c != "date"]
        if non_date_cols and any(numeric(out[c]).notna().sum() > 0 for c in non_date_cols):
            return _fill_historical_benchmarks(out)
    daily = scoped.get("benchmark_daily", pd.DataFrame()).copy()
    if daily.empty or "date" not in daily.columns:
        return pd.DataFrame()
    out = daily[["date"]].copy()
    daily_map = {
        "growth_daily_return": "Growth Champion Final",
        "gross_daily_return": "Growth Champion Final",
        "net_daily_return": "Growth Estimated Net",
        "candidate_return": "Growth Champion Final",
        "growth_gross_return": "Growth Gross",
        "growth_estimated_net_return": "Growth Estimated Net",
        "spy_daily_return": "SPY",
        "SPY_return": "SPY",
        "spy_daily_return": "SPY",
        "qqq_daily_return": "QQQ",
        "QQQ_return": "QQQ",
        "qqq_daily_return": "QQQ",
    }
    for raw, label in daily_map.items():
        if raw in daily.columns and label not in out.columns:
            vals = numeric(daily[raw])
            if vals.notna().sum() > 0:
                out[label] = ((1 + vals.fillna(0)).cumprod() - 1) * 100
    out = _fill_historical_benchmarks(out)
    for label in ["SPY", "QQQ"]:
        if label in out.columns and out[label].abs().max(skipna=True) < 5:
            out[label] = ((1 + numeric(out[label]).fillna(0)).cumprod() - 1) * 100
    return out if len(out.columns) > 1 else pd.DataFrame()


def equity_from_performance(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty or "date" not in perf.columns:
        return pd.DataFrame()
    out = perf.sort_values("date").copy()
    if "gross_equity" in out.columns:
        out["Gross Equity"] = numeric(out["gross_equity"])
    elif "gross_portfolio_value" in out.columns:
        out["Gross Equity"] = numeric(out["gross_portfolio_value"])
    elif "portfolio_value" in out.columns:
        out["Gross Equity"] = numeric(out["portfolio_value"])
    elif "gross_daily_return" in out.columns:
        out["Gross Equity"] = (1 + numeric(out["gross_daily_return"]).fillna(0)).cumprod() * 100000
    if "estimated_net_equity" in out.columns:
        out["Estimated Net Equity"] = numeric(out["estimated_net_equity"])
    elif "estimated_net_portfolio_value" in out.columns:
        out["Estimated Net Equity"] = numeric(out["estimated_net_portfolio_value"])
    elif "net_equity" in out.columns:
        out["Estimated Net Equity"] = numeric(out["net_equity"]) * 100000
    if "Gross Equity" in out.columns:
        out["Drawdown"] = out["Gross Equity"] / out["Gross Equity"].cummax() - 1
    return out


def current_holdings(data: dict[str, pd.DataFrame], scope: str) -> pd.DataFrame:
    state = scope_data(data, scope).get("state", pd.DataFrame())
    if state.empty:
        return pd.DataFrame()
    cur = latest(state).copy()
    if cur.empty:
        return cur
    pnl = data.get("official_position_pnl", pd.DataFrame()) if scope == "Official Forward Paper" else pd.DataFrame()
    if not pnl.empty and "ticker" in pnl.columns:
        cur = cur.merge(latest(pnl).drop(columns=["date"], errors="ignore"), on="ticker", how="left", suffixes=("", "_pnl"))
    return cur


def read_price_cache(tickers: list[str], lookback: int = 260) -> pd.DataFrame:
    frames = []
    root = Path("yahoo_ohlcv_price_cache")
    for ticker in tickers:
        path = root / f"{ticker}.csv"
        if not path.exists():
            continue
        df, _ = load_csv(str(path))
        if df.empty or "Date" not in df.columns:
            continue
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close" if "Close" in df.columns else None
        if price_col is None:
            continue
        tmp = df[["Date", price_col]].dropna().tail(lookback).copy()
        tmp = tmp.rename(columns={"Date": "date", price_col: ticker})
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date")
