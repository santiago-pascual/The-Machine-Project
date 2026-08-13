from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PLAN_OUTPUT = "overfitting_reduction_plan.csv"
CONFIG_OUTPUT = "constrained_research_config.json"


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _dashboard_value(summary: pd.DataFrame, metric: str, default: Any = np.nan) -> Any:
    if summary.empty or not {"metric", "value"}.issubset(summary.columns):
        return default
    rows = summary[summary["metric"].astype(str).eq(metric)]
    if rows.empty:
        return default
    return rows.iloc[-1]["value"]


def _stable_barrier_region(barrier: pd.DataFrame) -> dict[str, list[float | int]]:
    if barrier.empty:
        return {"horizons": [10, 20], "tp_multiples": [1.0, 1.25], "sl_multiples": [1.0]}
    selected = barrier.copy()
    if "subset" in selected.columns:
        selected = selected[selected["subset"].astype(str).eq("selected_only")]
    for col in ["sample_size", "TP_rate", "SL_rate", "Sharpe", "avg_return"]:
        if col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")
    mask = pd.Series(True, index=selected.index)
    if "sample_size" in selected.columns:
        mask &= selected["sample_size"] >= 50
    if {"TP_rate", "SL_rate"}.issubset(selected.columns):
        mask &= (selected["TP_rate"] >= selected["SL_rate"]) | (selected.get("avg_return", 0) > 0.02)
    if "Sharpe" in selected.columns:
        mask &= selected["Sharpe"] > 0
    stable = selected[mask]
    if stable.empty:
        stable = selected.head(10)
    return {
        "horizons": sorted([int(x) for x in stable.get("horizon", pd.Series([10, 20])).dropna().unique().tolist()])[:3],
        "tp_multiples": sorted([float(x) for x in stable.get("tp_multiple", pd.Series([1.0, 1.25])).dropna().unique().tolist()])[:3],
        "sl_multiples": sorted([float(x) for x in stable.get("sl_multiple", pd.Series([1.0])).dropna().unique().tolist()])[:2],
    }


def _threshold_safe_region(thresholds: pd.DataFrame) -> dict[str, list[float | int]]:
    if thresholds.empty:
        return {
            "signal_strength_threshold": [0.3, 0.4],
            "target_confidence_threshold": [0.5, 0.6],
            "quality_score_threshold": [0.5, 0.6],
            "expected_return_threshold": [0.0005, 0.001],
            "max_selected_assets": [4, 5],
            "min_selected_assets": [2],
        }
    df = thresholds.copy()
    for col in [
        "sample_size",
        "Sharpe",
        "TP_rate",
        "SL_rate",
        "average_selected_count",
        "turnover_proxy",
        "average_cash_proxy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if "sample_size" in df.columns:
        mask &= df["sample_size"] >= max(30, float(df["sample_size"].quantile(0.50)))
    if {"TP_rate", "SL_rate"}.issubset(df.columns):
        mask &= df["TP_rate"] >= df["SL_rate"]
    if "average_selected_count" in df.columns:
        mask &= df["average_selected_count"].between(2, 6)
    if "turnover_proxy" in df.columns:
        mask &= df["turnover_proxy"] <= 1.0
    if "average_cash_proxy" in df.columns:
        mask &= df["average_cash_proxy"].between(0.1, 0.8)
    stable = df[mask]
    if stable.empty:
        stable = df.sort_values("sample_size", ascending=False).head(25) if "sample_size" in df.columns else df.head(25)

    def values(col: str, fallback: list[float | int], limit: int = 3) -> list[float | int]:
        if col not in stable.columns:
            return fallback
        vals = sorted(stable[col].dropna().unique().tolist())
        return vals[:limit] if vals else fallback

    return {
        "signal_strength_threshold": values("signal_strength_threshold", [0.3, 0.4], 2),
        "target_confidence_threshold": values("target_confidence_threshold", [0.5, 0.6], 2),
        "quality_score_threshold": values("quality_score_threshold", [0.5, 0.6], 2),
        "expected_return_threshold": values("expected_return_threshold", [0.0005, 0.001], 2),
        "max_selected_assets": values("max_selected_assets", [4, 5], 2),
        "min_selected_assets": values("min_selected_assets", [2], 1),
    }


def identify_overfitting_sources() -> tuple[pd.DataFrame, dict[str, Any]]:
    dashboard = _read_csv("research_dashboard_summary.csv")
    trial_log = _read_csv("strategy_trial_log.csv")
    thresholds = _read_csv("threshold_optimization.csv")
    barriers = _read_csv("barrier_parameter_optimization.csv")
    robustness = _read_csv("robustness_validation.csv")
    full_quant = _read_csv("full_quant_robustness_walk_forward.csv")

    total_trials = int(pd.to_numeric(trial_log.get("number_of_trials", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    threshold_trials = int(len(thresholds))
    barrier_trials = int(len(barriers))
    pbo = _safe_float(_dashboard_value(dashboard, "PBO_proxy", np.nan), np.nan)
    robustness_score = _safe_float(_dashboard_value(dashboard, "robustness_score", np.nan), np.nan)
    if not np.isfinite(robustness_score) and not robustness.empty and "robustness_score" in robustness.columns:
        robustness_score = float(pd.to_numeric(robustness["robustness_score"], errors="coerce").dropna().mean())
    full_quant_wins = np.nan
    if not full_quant.empty and "sharpe_difference" in full_quant.columns:
        diffs = pd.to_numeric(full_quant["sharpe_difference"], errors="coerce").dropna()
        full_quant_wins = int((diffs > 0).sum())
    rows = []

    def add(source: str, severity: str, evidence: Any, recommendation: str, action: str) -> None:
        rows.append(
            {
                "category": "overfitting_source",
                "item": source,
                "severity": severity,
                "evidence": evidence,
                "recommendation": recommendation,
                "status": action,
            }
        )

    if total_trials > 1000:
        add("excessive_total_trials", "critical", total_trials, "cap total independent trials and require experiment registry", "blocked_until_governed")
    if threshold_trials > 100:
        add("threshold_grid_too_wide", "critical", threshold_trials, "reduce threshold grid to economically justified values only", "blocked")
    if barrier_trials > 50:
        add("tp_sl_grid_too_wide", "high", barrier_trials, "restrict TP/SL grid to stable region and avoid repeated mining", "restricted")
    sample_sizes = []
    for df in [thresholds, barriers, robustness]:
        if not df.empty and "sample_size" in df.columns:
            positive_sizes = pd.to_numeric(df["sample_size"], errors="coerce").dropna()
            sample_sizes.extend(positive_sizes[positive_sizes > 0].tolist())
    if sample_sizes and np.nanmedian(sample_sizes) < 100:
        add("small_sample_sizes", "critical", float(np.nanmedian(sample_sizes)), "increase walk-forward observations before promotion", "blocked_for_promotion")
    if np.isfinite(robustness_score) and robustness_score < 60:
        add("weak_robustness_score", "critical", robustness_score, "do not promote configs below robustness threshold", "blocked_for_promotion")
    if np.isfinite(pbo) and pbo > 0.3:
        add("high_pbo", "critical", pbo, "treat all optimized configs as research-only", "blocked_for_promotion")
    if np.isfinite(full_quant_wins) and full_quant_wins < 3:
        add("inconsistent_full_quant_performance", "high", f"{full_quant_wins}/4 windows improved", "keep full quant diagnostic or gated only", "restricted")
    if not robustness.empty and "rejected" in robustness.columns:
        rejected_rate = robustness["rejected"].astype(str).str.lower().isin(["true", "1", "yes"]).mean()
        if rejected_rate > 0.5:
            add("unstable_or_rejected_best_configs", "high", f"{rejected_rate:.2%} rejected", "ignore rejected top configs", "blocked")

    context = {
        "dashboard": dashboard,
        "trial_log": trial_log,
        "thresholds": thresholds,
        "barriers": barriers,
        "robustness": robustness,
        "full_quant": full_quant,
        "total_trials": total_trials,
        "threshold_trials": threshold_trials,
        "barrier_trials": barrier_trials,
        "pbo": pbo,
        "robustness_score": robustness_score,
    }
    return pd.DataFrame(rows), context


def build_constrained_config(context: dict[str, Any]) -> dict[str, Any]:
    barrier_region = _stable_barrier_region(context["barriers"])
    threshold_region = _threshold_safe_region(context["thresholds"])
    return {
        "trial_budget": {
            "max_new_strategy_variants_per_cycle": 5,
            "max_threshold_configs_per_cycle": 24,
            "max_tp_sl_configs_per_cycle": 12,
            "max_total_independent_trials_before_review": 1000,
        },
        "minimum_evidence_rules": {
            "minimum_selected_only_sample_size": 150,
            "minimum_independent_walk_forward_windows": 4,
            "minimum_robustness_score": 60.0,
            "maximum_PBO_allowed": 0.30,
            "minimum_deflated_sharpe_required": 0.50,
            "minimum_live_paper_trading_days_before_promotion": 60,
        },
        "parameter_constraints": {
            "tp_sl_grid": barrier_region,
            "threshold_grid": threshold_region,
            "reject_if_TP_rate_below_SL_rate": True,
            "allow_TP_below_SL_exception_only_if": {
                "average_return_positive": True,
                "drawdown_controlled": True,
                "robust_across_windows": True,
            },
            "max_turnover_proxy": 1.0,
            "cash_bounds": [0.10, 0.80],
            "selected_asset_bounds": [2, 6],
        },
        "allowed_research_directions": [
            "increase walk-forward sample size",
            "paper-trading forecast calibration",
            "out-of-sample validation of existing signals",
            "regime diagnostics without new parameters",
            "small constrained TP/SL validation inside stable region",
        ],
        "blocked_research_directions": [
            "large threshold grid search",
            "large TP/SL grid search",
            "promoting full_quant_research",
            "promoting regime_gated_full_quant",
            "optimizing on one small sample",
            "adding new alpha layers before validation sample grows",
        ],
    }


def build_reduction_plan() -> tuple[pd.DataFrame, dict[str, Any]]:
    sources, context = identify_overfitting_sources()
    config = build_constrained_config(context)
    rows = sources.to_dict("records")

    def add_rule(item: str, recommendation: str, status: str, severity: str = "policy") -> None:
        rows.append(
            {
                "category": "constrained_research_rule",
                "item": item,
                "severity": severity,
                "evidence": "",
                "recommendation": recommendation,
                "status": status,
            }
        )

    add_rule("trial_budget", "max 5 new strategy variants, 24 threshold configs, 12 TP/SL configs per cycle", "allowed_with_limits")
    add_rule("minimum_evidence", "require sample_size >= 150, 4 independent windows, robustness >= 60, PBO <= 0.30", "required_for_promotion")
    add_rule("threshold_search", "use constrained grid only; no 14k-grid reruns", "restricted")
    add_rule("tp_sl_search", "use stable TP/SL region only; reject TP<SL unless justified", "restricted")
    add_rule("full_quant_research", "keep diagnostic/gated only; no promotion", "blocked_for_promotion")
    add_rule("next_allowed_work", "collect more out-of-sample data and validate existing signals", "allowed")

    return pd.DataFrame(rows), config


def print_overfitting_reduction_plan(plan: pd.DataFrame, config: dict[str, Any]) -> None:
    print("\n===== OVERFITTING REDUCTION PLAN =====")
    if plan.empty:
        print("No overfitting sources detected. Continue research cautiously.")
    else:
        print(plan.to_string(index=False))
    print("\n===== CONSTRAINED RESEARCH CONFIG =====")
    print(json.dumps(config, indent=2, sort_keys=True))


def run_overfitting_reduction_plan() -> tuple[pd.DataFrame, dict[str, Any]]:
    plan, config = build_reduction_plan()
    plan.to_csv(PLAN_OUTPUT, index=False)
    Path(CONFIG_OUTPUT).write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print_overfitting_reduction_plan(plan, config)
    print(f"\nSaved: {Path(PLAN_OUTPUT).resolve()}")
    print(f"Saved: {Path(CONFIG_OUTPUT).resolve()}")
    return plan, config


if __name__ == "__main__":
    run_overfitting_reduction_plan()
