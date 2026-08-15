from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anti_overfitting_framework import (
    _best_observed_sharpe,
    _return_distribution_stats,
    _robustness_score,
    _top_config_isolated,
    build_strategy_trial_log,
    deflated_sharpe_diagnostic,
    pbo_approximation,
)

EXPERIMENT_REGISTRY_FILE = "experiment_registry.csv"
TRIAL_LOG_FILE = "strategy_trial_log.csv"


@dataclass(frozen=True)
class GovernanceConfig:
    max_new_trials_per_day: int = 25
    max_total_trials_before_warning: int = 1000
    min_sample_size_for_promotion: int = 100
    min_out_of_sample_windows_for_promotion: int = 4
    min_deflated_sharpe_for_promotion: float = 0.50
    max_pbo_for_promotion: float = 0.30
    min_robustness_score_for_promotion: float = 60.0
    max_turnover_for_promotion: float = 1.00
    block_promotion_on_high_overfitting_warning: bool = True


def canonical_parameter_hash(parameters: dict[str, Any]) -> str:
    clean = json.dumps(parameters, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _write_registry(registry: pd.DataFrame, path: str | Path = EXPERIMENT_REGISTRY_FILE) -> None:
    registry.to_csv(path, index=False)


def _infer_data_range() -> tuple[str, str]:
    candidates = [
        "walk_forward_predictions.csv",
        "full_quant_robustness__20250101__latest__baseline_predictions.csv",
        "model_mode__baseline_predictions.csv",
    ]
    dates = []
    for path in candidates:
        df = _read_csv(path)
        if not df.empty and "date" in df.columns:
            parsed = pd.to_datetime(df["date"], errors="coerce").dropna()
            dates.extend(parsed.tolist())
    if not dates:
        return "", ""
    series = pd.Series(dates)
    return str(series.min().date()), str(series.max().date())


def _infer_universe_size() -> int:
    df = _read_csv("walk_forward_predictions.csv")
    if not df.empty and "ticker" in df.columns:
        return int(df["ticker"].nunique())
    return 0


def _infer_test_dates() -> int:
    df = _read_csv("walk_forward_predictions.csv")
    if not df.empty and "date" in df.columns:
        return int(df["date"].nunique())
    summary = _read_csv("walk_forward_summary.csv")
    if not summary.empty and "number_of_test_dates" in summary.columns:
        return int(pd.to_numeric(summary["number_of_test_dates"], errors="coerce").fillna(0).max())
    return 0


def register_experiment(
    *,
    experiment_name: str,
    model_mode: str,
    parameters: dict[str, Any],
    objective_metric: str = "",
    result_metric: float | str | None = None,
    created_by: str = "codex",
    notes: str = "",
    registry_path: str | Path = EXPERIMENT_REGISTRY_FILE,
) -> pd.DataFrame:
    registry = _read_csv(registry_path)
    parameter_hash = canonical_parameter_hash(parameters)
    data_start, data_end = _infer_data_range()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    duplicate_mask = pd.Series(False, index=registry.index)
    if not registry.empty and {"experiment_name", "parameter_set_hash"}.issubset(registry.columns):
        duplicate_mask = registry["experiment_name"].astype(str).eq(str(experiment_name)) & registry["parameter_set_hash"].astype(str).eq(
            parameter_hash
        )
    duplicate_existing = bool(duplicate_mask.any()) if not registry.empty else False
    experiment_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{parameter_hash}"
    row = {
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model_mode": model_mode,
        "parameter_set_hash": parameter_hash,
        "data_start": data_start,
        "data_end": data_end,
        "universe_size": _infer_universe_size(),
        "number_of_test_dates": _infer_test_dates(),
        "objective_metric": objective_metric,
        "result_metric": result_metric if result_metric is not None else "",
        "created_by": created_by,
        "notes": notes,
        "duplicate_existing": duplicate_existing,
        "counted_as_new_independent_trial": not duplicate_existing,
    }
    updated = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    _write_registry(updated, registry_path)
    return updated


def trial_budget_report(
    config: GovernanceConfig = GovernanceConfig(),
    registry_path: str | Path = EXPERIMENT_REGISTRY_FILE,
) -> pd.DataFrame:
    registry = _read_csv(registry_path)
    if registry.empty:
        rows = [
            {
                "metric": "registry_rows",
                "value": 0,
                "limit": np.nan,
                "status": "ok",
            }
        ]
        return pd.DataFrame(rows)
    timestamps = pd.to_datetime(registry.get("timestamp"), errors="coerce", utc=True)
    today = datetime.now(timezone.utc).date()
    counted = registry.get("counted_as_new_independent_trial", pd.Series(True, index=registry.index)).astype(bool)
    new_today = int(((timestamps.dt.date == today) & counted).sum())
    total_independent = int(counted.sum())
    rows = [
        {
            "metric": "new_independent_trials_today",
            "value": new_today,
            "limit": config.max_new_trials_per_day,
            "status": "warning" if new_today > config.max_new_trials_per_day else "ok",
        },
        {
            "metric": "total_independent_trials_registry",
            "value": total_independent,
            "limit": config.max_total_trials_before_warning,
            "status": "warning" if total_independent > config.max_total_trials_before_warning else "ok",
        },
    ]
    trial_log = _read_csv(TRIAL_LOG_FILE)
    if not trial_log.empty and "number_of_trials" in trial_log.columns:
        total_detected = int(pd.to_numeric(trial_log["number_of_trials"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "metric": "total_detected_strategy_trials",
                "value": total_detected,
                "limit": config.max_total_trials_before_warning,
                "status": "warning" if total_detected > config.max_total_trials_before_warning else "ok",
            }
        )
    return pd.DataFrame(rows)


def _average_turnover() -> float:
    candidates = [
        "full_quant_robustness_walk_forward.csv",
        "walk_forward_summary.csv",
        "model_mode_comparison.csv",
    ]
    values = []
    for path in candidates:
        df = _read_csv(path)
        if df.empty:
            continue
        turnover_cols = [c for c in df.columns if "turnover" in c.lower()]
        for col in turnover_cols:
            values.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
    return float(np.nanmean(values)) if values else np.nan


def _tp_sl_issue() -> bool:
    candidates = ["barrier_parameter_optimization.csv", "robustness_validation.csv", "full_quant_robustness_walk_forward.csv"]
    for path in candidates:
        df = _read_csv(path)
        if df.empty:
            continue
        tp_cols = [c for c in df.columns if c.endswith("TP_rate") or c == "TP_rate"]
        for tp_col in tp_cols:
            sl_col = tp_col.replace("TP_rate", "SL_rate")
            if sl_col in df.columns:
                tp = pd.to_numeric(df[tp_col], errors="coerce")
                sl = pd.to_numeric(df[sl_col], errors="coerce")
                if bool((tp < sl).fillna(False).any()):
                    return True
    return False


def _single_window_dependency() -> bool:
    df = _read_csv("full_quant_robustness_walk_forward.csv")
    if df.empty or "sharpe_difference" not in df.columns:
        return True
    diffs = pd.to_numeric(df["sharpe_difference"], errors="coerce").dropna()
    if len(diffs) < 2:
        return True
    return int((diffs > 0).sum()) <= 1


def promotion_checklist(config: GovernanceConfig = GovernanceConfig()) -> pd.DataFrame:
    trial_log = build_strategy_trial_log()
    total_trials = int(trial_log["number_of_trials"].sum()) if not trial_log.empty else 1
    best = _best_observed_sharpe()
    dist = _return_distribution_stats()
    sample_size = int(best.get("sample_length", 0) or dist.get("sample_size", 0) or 0)
    dsr = deflated_sharpe_diagnostic(
        observed_sharpe=float(best.get("observed_sharpe", 0.0)),
        number_of_trials=total_trials,
        sample_length=max(2, sample_size),
        skewness=float(dist.get("skewness", 0.0)),
        kurtosis=float(dist.get("kurtosis", 3.0)),
    )
    robustness_score = _robustness_score()
    pbo = pbo_approximation(
        total_trials=total_trials,
        robustness_score=robustness_score,
        sample_size=max(2, sample_size),
        top_config_isolated=_top_config_isolated(),
    )
    windows_df = _read_csv("full_quant_robustness_walk_forward.csv")
    oos_windows = int(windows_df["window"].nunique()) if not windows_df.empty and "window" in windows_df.columns else 0
    turnover = _average_turnover()
    tp_sl_issue = _tp_sl_issue()
    single_window = _single_window_dependency()
    checklist = [
        (
            "sample_size_above_threshold",
            sample_size,
            config.min_sample_size_for_promotion,
            sample_size >= config.min_sample_size_for_promotion,
        ),
        (
            "multiple_oos_windows",
            oos_windows,
            config.min_out_of_sample_windows_for_promotion,
            oos_windows >= config.min_out_of_sample_windows_for_promotion,
        ),
        (
            "deflated_sharpe_above_threshold",
            dsr["deflated_sharpe_estimate"],
            config.min_deflated_sharpe_for_promotion,
            float(dsr["deflated_sharpe_estimate"]) >= config.min_deflated_sharpe_for_promotion,
        ),
        ("pbo_below_threshold", pbo["pbo_proxy"], config.max_pbo_for_promotion, float(pbo["pbo_proxy"]) <= config.max_pbo_for_promotion),
        (
            "robustness_score_above_threshold",
            robustness_score,
            config.min_robustness_score_for_promotion,
            robustness_score >= config.min_robustness_score_for_promotion,
        ),
        (
            "turnover_not_extreme",
            turnover,
            config.max_turnover_for_promotion,
            np.isfinite(turnover) and turnover <= config.max_turnover_for_promotion,
        ),
        ("no_tp_below_sl_issue", int(tp_sl_issue), 0, not tp_sl_issue),
        ("no_single_window_dependency", int(single_window), 0, not single_window),
    ]
    rows = []
    for criterion, value, threshold, passed in checklist:
        rows.append(
            {
                "criterion": criterion,
                "value": value,
                "threshold": threshold,
                "passed": bool(passed),
                "status": "pass" if passed else "fail",
            }
        )
    if config.block_promotion_on_high_overfitting_warning:
        rows.append(
            {
                "criterion": "research_lock_overfitting_warning",
                "value": dsr["warning_level"],
                "threshold": "not high/extreme",
                "passed": dsr["warning_level"] not in {"high", "extreme"},
                "status": "pass" if dsr["warning_level"] not in {"high", "extreme"} else "fail",
            }
        )
    return pd.DataFrame(rows)


def run_research_governance_report(
    *,
    experiment_name: str = "research_governance_audit",
    model_mode: str = "research_only",
    parameters: dict[str, Any] | None = None,
    objective_metric: str = "overfitting_risk",
    result_metric: float | str | None = None,
    notes: str = "governance audit run",
    config: GovernanceConfig = GovernanceConfig(),
) -> dict[str, pd.DataFrame]:
    params = parameters or {
        "module": "research_governance",
        "config": asdict(config),
        "input_files": [
            "strategy_trial_log.csv",
            "robustness_validation.csv",
            "full_quant_robustness_walk_forward.csv",
            "threshold_optimization.csv",
            "barrier_parameter_optimization.csv",
        ],
    }
    registry = register_experiment(
        experiment_name=experiment_name,
        model_mode=model_mode,
        parameters=params,
        objective_metric=objective_metric,
        result_metric=result_metric,
        notes=notes,
    )
    trial_log = build_strategy_trial_log()
    budget = trial_budget_report(config=config)
    checklist = promotion_checklist(config=config)
    promotion_allowed = bool(checklist["passed"].all()) if not checklist.empty else False

    print("\n===== RESEARCH GOVERNANCE REPORT =====")
    print(f"registry rows: {len(registry)}")
    print(f"latest experiment_id: {registry.iloc[-1]['experiment_id'] if not registry.empty else ''}")
    print(f"latest parameter_set_hash: {registry.iloc[-1]['parameter_set_hash'] if not registry.empty else ''}")
    print(f"duplicate existing: {registry.iloc[-1]['duplicate_existing'] if not registry.empty else False}")
    print(f"counted as new independent trial: {registry.iloc[-1]['counted_as_new_independent_trial'] if not registry.empty else False}")
    print("\n===== TRIAL BUDGET REPORT =====")
    print(budget.to_string(index=False))
    print("\n===== STRATEGY PROMOTION CHECKLIST =====")
    print(checklist.to_string(index=False))
    if not promotion_allowed:
        print("\nResearch allowed, production promotion blocked.")
    else:
        print("\nPromotion checklist passed for research review only. No auto-promotion performed.")
    print(f"\nSaved: {Path(EXPERIMENT_REGISTRY_FILE).resolve()}")
    print(f"Saved: {Path(TRIAL_LOG_FILE).resolve()}")
    return {
        "registry": registry,
        "trial_log": trial_log,
        "budget": budget,
        "checklist": checklist,
    }


if __name__ == "__main__":
    run_research_governance_report()
