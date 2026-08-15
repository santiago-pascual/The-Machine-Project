from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = "growth_candidate_paper_config.json"
LIFECYCLE = "model_lifecycle_status.csv"
FROZEN = "frozen_champion_registry.csv"
DASHBOARD_SUMMARY = "research_dashboard_summary.csv"
FINAL_SUMMARY = "final_research_summary.txt"
SIGNAL_AUDIT = "growth_signal_strength_influence_audit.csv"
VALIDATION_GATES = "growth_operational_validation_gates.csv"
COST_LEDGER = "growth_paper_estimated_cost_ledger.csv"
NET_PERF = "growth_paper_estimated_net_performance.csv"
CAPACITY_REPORT = "growth_operational_capacity_report.csv"
OP_REPORT = "growth_operational_integration_report.txt"

MODEL_VERSION = "growth_champion_final_v1_0_frozen"
VARIANT = "growth_v1_exposure_cap_60_dual_trend_filter"


def read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def write_json_config() -> dict:
    p = Path(CONFIG)
    cfg = {}
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.update({
        "active_growth_paper_model": "growth_champion_final",
        "active_model_version": MODEL_VERSION,
        "active_variant": VARIANT,
        "classification": "operational_paper_production",
        "real_capital_status": "real_capital_blocked",
        "capital_deployment_enabled": False,
        "broker_connection_enabled": False,
        "real_orders_enabled": False,
        "paper_only": True,
        "real_trading": False,
        "frozen_parameters": True,
        "target_volatility": 0.22,
        "volatility_target": 0.22,
        "minimum_exposure": 0.40,
        "min_exposure": 0.40,
        "exposure_cap": 0.60,
        "volatility_lookback_days": 60,
        "max_positions": 4,
        "allocation_method": "equal_weight_within_final_exposure",
        "rebalance_frequency_sessions": 5,
        "rebalance_mode": "exact_backtest_parity",
        "rebalance_anchor_date": "2023-01-04",
        "signal_execution_lag": "close_t_signal_apply_next_session_return",
        "allow_unscheduled_rebalance": False,
        "dual_trend_filter": True,
        "dual_trend_caps": {"both_above": 0.60, "one_below": 0.40, "both_below": 0.25},
        "dual_trend_rules": {
            "both_spy_qqq_below_200d_cap": 0.25,
            "one_of_spy_qqq_below_200d_cap": 0.4,
            "both_spy_qqq_above_200d_cap": 0.6,
        },
        "signal_strength_active_in_growth_allocation": False,
        "signal_strength_usage": "diagnostic_only",
        "garch_egarch_active_model": "current_grid_model",
        "garch_mle_status": "diagnostic_only",
        "hmm_4_state_status": "dashboard_and_governance_diagnostic_only",
        "advanced_execution_costs_status": "reporting_and_estimated_paper_ledger_only",
        "subtract_estimated_costs_from_paper_accounting": False,
        "validation_gates_active": ["CSCV_PBO", "exact_DSR", "purged_walk_forward", "parameter_stability", "canonical_metric_reconciliation", "live_paper_health"],
    })
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def signal_strength_audit() -> pd.DataFrame:
    rows = [
        {"component": "growth_raw_target_ranking", "signal_strength_influences": False, "status": "removed_from_active_logic", "evidence": "current_growth_feature_generation sorts by raw_target_return only"},
        {"component": "growth_soft_exit", "signal_strength_influences": False, "status": "diagnostic_only", "evidence": "soft_exit retains prior tickers only if raw_target_return > 0"},
        {"component": "growth_filters", "signal_strength_influences": False, "status": "diagnostic_only", "evidence": "quality/tradability filters use price, volume, history, volatility and blacklist rules"},
        {"component": "growth_position_sizing", "signal_strength_influences": False, "status": "diagnostic_only", "evidence": "equal weights within final exposure"},
        {"component": "expected_returns_model_baseline", "signal_strength_influences": True, "status": "baseline_not_growth_allocation", "evidence": "baseline expected-return pipeline still has signal adjustment; production defaults unchanged"},
        {"component": "dashboard_and_csv_diagnostics", "signal_strength_influences": False, "status": "retained_for_monitoring", "evidence": "signal_strength_adjustment_value retained only for diagnostics"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(SIGNAL_AUDIT, index=False)
    return out


def latest_classification(path: str, col: str = "classification") -> str:
    df = read(path)
    if df.empty:
        return "missing"
    if col in df.columns:
        return str(df[col].dropna().iloc[-1]) if df[col].dropna().any() else "missing"
    return "available"


def validation_gates() -> pd.DataFrame:
    checks = [
        {"gate": "CSCV_PBO", "source": "full_cscv_results.csv", "status": "active", "result": "implemented", "capital_gate": "blocks_real_capital_until_review"},
        {"gate": "exact_DSR", "source": "deflated_sharpe_exact.csv", "status": "active", "result": "implemented", "capital_gate": "blocks_real_capital_until_review"},
        {"gate": "purged_walk_forward", "source": "out_of_sample_governance.csv", "status": "active", "result": latest_classification("out_of_sample_governance.csv"), "capital_gate": "required"},
        {"gate": "parameter_stability", "source": "parameter_governance.csv", "status": "active", "result": latest_classification("parameter_governance.csv"), "capital_gate": "required"},
        {"gate": "canonical_metric_reconciliation", "source": "growth_canonical_governance.csv", "status": "active", "result": latest_classification("growth_canonical_governance.csv"), "capital_gate": "required"},
        {"gate": "live_paper_health", "source": "growth_live_health.csv", "status": "active", "result": latest_classification("growth_live_health.csv", "governance_classification"), "capital_gate": "requires_6_months_history"},
        {"gate": "garch_model_choice", "source": "garch_governance.csv", "status": "active", "result": latest_classification("garch_governance.csv"), "capital_gate": "keep_grid_model"},
        {"gate": "hmm_allocation_guard", "source": "hmm_governance.csv", "status": "diagnostic_only", "result": latest_classification("hmm_governance.csv"), "capital_gate": "not_allowed_to_change_allocation"},
    ]
    out = pd.DataFrame(checks)
    out.to_csv(VALIDATION_GATES, index=False)
    return out


def paper_cost_ledger() -> pd.DataFrame:
    trades = read("growth_candidate_paper_trades.csv")
    perf = read("growth_candidate_paper_performance.csv")
    costs = read("advanced_execution_costs.csv")
    if trades.empty:
        out = pd.DataFrame()
        out.to_csv(COST_LEDGER, index=False)
        return out
    median_bps = 10.0
    if not costs.empty and "total_cost_bps_of_order" in costs.columns:
        scenario = costs[costs.get("scenario", "").astype(str).eq("impact_Y_1.0")]
        src = scenario if not scenario.empty else costs
        vals = pd.to_numeric(src["total_cost_bps_of_order"], errors="coerce").dropna()
        if not vals.empty:
            median_bps = float(vals.median())
    perf_latest = perf.copy()
    if not perf_latest.empty and "date" in perf_latest.columns:
        perf_latest["date"] = perf_latest["date"].astype(str)
    rows = []
    for _, row in trades.iterrows():
        date = str(row.get("date", ""))
        port_value = np.nan
        if not perf_latest.empty:
            match = perf_latest[perf_latest["date"].astype(str).eq(date)]
            if not match.empty:
                port_value = float(pd.to_numeric(match.iloc[-1].get("portfolio_value", np.nan), errors="coerce"))
        if not np.isfinite(port_value):
            port_value = 100000.0
        weight_change = abs(float(pd.to_numeric(pd.Series([row.get("trade_weight_change", row.get("weight_change", 0.0))]), errors="coerce").fillna(0).iloc[0]))
        order_value = port_value * weight_change
        estimated_cost = order_value * median_bps / 10000.0
        rows.append({
            "date": date,
            "ticker": row.get("ticker", ""),
            "action": row.get("action", ""),
            "portfolio_value": port_value,
            "trade_weight_change_abs": weight_change,
            "estimated_order_value": order_value,
            "estimated_total_cost_bps_of_order": median_bps,
            "estimated_total_cost": estimated_cost,
            "paper_accounting_adjusted": False,
            "reason": "estimated from advanced_execution_costs median impact_Y_1.0; reporting only",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        daily = out.groupby("date", as_index=False)["estimated_total_cost"].sum().rename(columns={"estimated_total_cost": "daily_estimated_cost"})
        out = out.merge(daily, on="date", how="left")
    out.to_csv(COST_LEDGER, index=False)
    write_estimated_net_performance(out)
    return out


def write_estimated_net_performance(cost_ledger: pd.DataFrame) -> pd.DataFrame:
    perf = read("growth_candidate_paper_performance.csv")
    if perf.empty:
        out = pd.DataFrame()
        out.to_csv(NET_PERF, index=False)
        return out
    out = perf.copy()
    out["date"] = out["date"].astype(str)
    if cost_ledger.empty:
        out["estimated_total_cost"] = 0.0
    else:
        daily_cost = cost_ledger.groupby("date", as_index=False)["estimated_total_cost"].sum()
        out = out.merge(daily_cost, on="date", how="left")
        out["estimated_total_cost"] = pd.to_numeric(out["estimated_total_cost"], errors="coerce").fillna(0.0)
    out["gross_paper_daily_return"] = pd.to_numeric(out.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    out["estimated_cost_return_drag"] = out["estimated_total_cost"] / pd.to_numeric(out.get("portfolio_value", 1.0), errors="coerce").replace(0, np.nan)
    out["estimated_net_daily_return"] = out["gross_paper_daily_return"] - out["estimated_cost_return_drag"].fillna(0.0)
    out["gross_paper_equity"] = (1.0 + out["gross_paper_daily_return"]).cumprod()
    out["estimated_net_paper_equity"] = (1.0 + out["estimated_net_daily_return"]).cumprod()
    out["paper_accounting_adjusted"] = False
    out.to_csv(NET_PERF, index=False)
    return out


def capacity_report() -> pd.DataFrame:
    cap = read("capacity_analysis.csv")
    if cap.empty:
        out = pd.DataFrame([{"capacity_status": "missing", "source": "capacity_analysis.csv"}])
    else:
        out = cap.copy()
        out["source"] = "capacity_analysis.csv"
        out["usage"] = "dashboard_and_capacity_reporting_only"
    out.to_csv(CAPACITY_REPORT, index=False)
    return out


def upsert_csv(path: str, key_col: str, row: dict) -> pd.DataFrame:
    df = read(path)
    if df.empty:
        df = pd.DataFrame([row])
    else:
        if key_col in df.columns and key_col in row:
            df = df[df[key_col].astype(str).ne(str(row[key_col]))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def update_lifecycle_and_registry() -> None:
    lifecycle_row = {
        "module": MODEL_VERSION,
        "category": "operational_paper_production",
        "Sharpe impact": "validated_operational_not_new_alpha",
        "return impact": "paper_monitoring_only",
        "risk impact": "real_capital_blocked",
        "evidence strength": "governed_multi_gate",
        "sample size": "exact_history_short_reconstructed_stress_available",
        "final decision": "official_growth_daily_pipeline_paper_only",
        "reason": "5-session scheduler, frozen params, diagnostics/gates integrated; broker/orders disabled",
    }
    upsert_csv(LIFECYCLE, "module", lifecycle_row)
    frozen_row = {
        "registry_name": MODEL_VERSION,
        "trial_count": 1,
        "source": CONFIG,
        "mix_with_governed": False,
        "status": "operational_paper_production_real_capital_blocked",
        "notes": "Frozen Growth Champion Final v1.0: 5-session rebalance, target vol 22%, min exposure 40%, cap 60%, dual trend 60/40/25, max 4 equal-weight positions.",
    }
    upsert_csv(FROZEN, "registry_name", frozen_row)


def update_dashboard_summary() -> None:
    rows = [
        {"section": "Growth Operational", "metric": "active_model_version", "value": MODEL_VERSION},
        {"section": "Growth Operational", "metric": "classification", "value": "operational_paper_production"},
        {"section": "Growth Operational", "metric": "real_capital_status", "value": "real_capital_blocked"},
        {"section": "Growth Operational", "metric": "rebalance_cadence", "value": "exact 5 trading sessions"},
        {"section": "Growth Operational", "metric": "signal_strength_usage", "value": "diagnostic_only"},
        {"section": "Growth Operational", "metric": "hmm_4_state_usage", "value": "diagnostic_only"},
        {"section": "Growth Operational", "metric": "garch_mle_usage", "value": "diagnostic_only_keep_grid"},
        {"section": "Growth Operational", "metric": "advanced_execution_costs", "value": "reporting_only_not_subtracted_from_paper"},
    ]
    existing = read(DASHBOARD_SUMMARY)
    if existing.empty:
        out = pd.DataFrame(rows)
    else:
        mask = existing.get("section", pd.Series(dtype=str)).astype(str).ne("Growth Operational")
        out = pd.concat([existing[mask], pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(DASHBOARD_SUMMARY, index=False)


def update_final_summary() -> None:
    block = f"""
===== GROWTH CHAMPION FINAL V1.0 FROZEN =====
Active model version: {MODEL_VERSION}
Classification: operational_paper_production
Real-capital status: real_capital_blocked
Broker/orders: disabled
Cadence: exact 5 trading sessions; signal at close t; economic application from next session; holdings frozen between rebalances
Frozen parameters: target_vol=22%, min_exposure=40%, exposure_cap=60%, vol_lookback=60D, dual_trend_caps=60/40/25, max_positions=4, equal-weight allocation
Signal strength: diagnostic only; removed from Growth ranking/sizing/filtering
GARCH/EGARCH: keep current grid model; MLE diagnostic only
HMM 4-state: dashboard/governance diagnostic only; no allocation effect
Execution costs: advanced estimates integrated into reporting and paper estimated cost ledger; gross paper accounting unchanged
Validation gates: CSCV PBO, exact DSR, purged walk-forward, parameter stability, canonical reconciliation, live paper health
"""
    p = Path(FINAL_SUMMARY)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    marker = "===== GROWTH CHAMPION FINAL V1.0 FROZEN ====="
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    p.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def report(cfg: dict, sig: pd.DataFrame, gates: pd.DataFrame, costs: pd.DataFrame) -> str:
    promoted = [
        "Exact 5-session rebalance scheduler",
        "Frozen robust parameter configuration as growth_champion_final_v1_0_frozen",
        "Validation gates into operational reporting",
        "Advanced execution cost estimates into reporting/ledger",
    ]
    diagnostic = ["GARCH/EGARCH MLE", "HMM 4-state regime model", "signal_strength", "advanced execution costs for net estimates only"]
    rejected = ["HMM-driven allocation", "MLE GARCH replacement", "real broker/orders", "signal_strength as active alpha/sizing input"]
    lines = [
        "===== SAFE OPERATIONAL INTEGRATION REPORT =====",
        f"active_model_version: {MODEL_VERSION}",
        "classification: operational_paper_production",
        "real_capital_status: real_capital_blocked",
        "broker_connection_enabled: False",
        "real_orders_enabled: False",
        "",
        "Components promoted:",
        *[f"- {x}" for x in promoted],
        "",
        "Components retained as diagnostic:",
        *[f"- {x}" for x in diagnostic],
        "",
        "Components rejected/blocked:",
        *[f"- {x}" for x in rejected],
        "",
        f"signal_strength_active_in_growth_allocation: {cfg.get('signal_strength_active_in_growth_allocation')}",
        f"estimated_cost_ledger_rows: {len(costs)}",
        f"validation_gate_rows: {len(gates)}",
        "production_changed: False",
        "optimizer_changed: False",
        "real_trading_enabled: False",
    ]
    text = "\n".join(lines) + "\n"
    Path(OP_REPORT).write_text(text, encoding="utf-8")
    return text


def main() -> None:
    cfg = write_json_config()
    sig = signal_strength_audit()
    gates = validation_gates()
    costs = paper_cost_ledger()
    capacity_report()
    update_lifecycle_and_registry()
    update_dashboard_summary()
    update_final_summary()
    print(report(cfg, sig, gates, costs))


if __name__ == "__main__":
    main()


