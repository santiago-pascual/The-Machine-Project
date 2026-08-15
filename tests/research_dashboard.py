from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anti_overfitting_framework import (
    _best_observed_sharpe,
    _return_distribution_stats,
    _robustness_score,
    deflated_sharpe_diagnostic,
    pbo_approximation,
)

DASHBOARD_SUMMARY_FILE = "research_dashboard_summary.csv"
DASHBOARD_REPORT_FILE = "research_dashboard_report.txt"


CSV_SOURCES = {
    "walk_forward_summary": "walk_forward_summary.csv",
    "larger_walk_forward_summary": "larger_walk_forward_summary.csv",
    "full_quant_robustness": "full_quant_robustness_walk_forward.csv",
    "regime_gated_comparison": "regime_gated_full_quant_comparison.csv",
    "strategy_trial_log": "strategy_trial_log.csv",
    "experiment_registry": "experiment_registry.csv",
    "robustness_validation": "robustness_validation.csv",
    "barrier_optimization": "barrier_parameter_optimization.csv",
    "threshold_optimization": "threshold_optimization.csv",
    "triple_barrier_labels": "triple_barrier_labels.csv",
    "triple_barrier_feature_validation": "triple_barrier_feature_validation.csv",
    "regime_performance": "regime_performance_attribution.csv",
    "ic_history": "ic_history.csv",
    "clean_research_evaluation": "clean_research_evaluation.csv",
}


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


def _fmt(value: Any, precision: int = 6) -> str:
    if isinstance(value, str):
        return value
    number = _safe_float(value, np.nan)
    if not np.isfinite(number):
        return "missing"
    return f"{number:.{precision}f}"


def _load_sources() -> dict[str, pd.DataFrame]:
    return {name: _read_csv(path) for name, path in CSV_SOURCES.items()}


def _source_status(sources: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    loaded = [name for name, df in sources.items() if not df.empty]
    missing = [name for name, df in sources.items() if df.empty]
    return loaded, missing


def _trial_total(trial_log: pd.DataFrame) -> int:
    if trial_log.empty or "number_of_trials" not in trial_log.columns:
        return 0
    return int(pd.to_numeric(trial_log["number_of_trials"], errors="coerce").fillna(0).sum())


def _governance_status(registry: pd.DataFrame, total_trials: int) -> str:
    if registry.empty:
        return "registry_missing"
    if total_trials > 1000:
        return "trial_budget_warning"
    return "ok"


def _performance_summary(full_quant: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "baseline_sharpe": np.nan,
        "full_quant_sharpe": np.nan,
        "regime_gated_sharpe": np.nan,
        "best_window": "missing",
        "worst_window": "missing",
        "average_cash": np.nan,
        "max_drawdown": np.nan,
    }
    if full_quant.empty:
        return result
    if "baseline_sharpe" in full_quant.columns:
        result["baseline_sharpe"] = pd.to_numeric(full_quant["baseline_sharpe"], errors="coerce").mean()
    if "full_quant_sharpe" in full_quant.columns:
        result["full_quant_sharpe"] = pd.to_numeric(full_quant["full_quant_sharpe"], errors="coerce").mean()
    if "regime_gated_sharpe" in full_quant.columns:
        result["regime_gated_sharpe"] = pd.to_numeric(full_quant["regime_gated_sharpe"], errors="coerce").mean()
    sharpe_col = "regime_gated_sharpe" if "regime_gated_sharpe" in full_quant.columns else "full_quant_sharpe"
    if sharpe_col in full_quant.columns and "window" in full_quant.columns:
        sharpe = pd.to_numeric(full_quant[sharpe_col], errors="coerce")
        if sharpe.notna().any():
            result["best_window"] = str(full_quant.loc[sharpe.idxmax(), "window"])
            result["worst_window"] = str(full_quant.loc[sharpe.idxmin(), "window"])
    cash_cols = [c for c in ["baseline_cash", "full_quant_cash", "regime_gated_cash"] if c in full_quant.columns]
    if cash_cols:
        result["average_cash"] = full_quant[cash_cols].apply(pd.to_numeric, errors="coerce").mean().mean()
    dd_cols = [c for c in ["baseline_max_drawdown", "full_quant_max_drawdown", "regime_gated_max_drawdown"] if c in full_quant.columns]
    if dd_cols:
        result["max_drawdown"] = full_quant[dd_cols].apply(pd.to_numeric, errors="coerce").min().min()
    return result


def _apply_larger_walk_forward(perf: dict[str, Any], larger: pd.DataFrame) -> dict[str, Any]:
    if larger.empty or "model_mode" not in larger.columns:
        return perf
    data = larger.copy()
    for col in ["Sharpe", "average_cash", "max_drawdown"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    baseline = data[data["model_mode"].astype(str).eq("baseline")]
    gated = data[data["model_mode"].astype(str).eq("regime_gated_full_quant")]
    if not baseline.empty:
        perf["baseline_sharpe"] = _safe_float(baseline.iloc[-1].get("Sharpe", perf.get("baseline_sharpe")))
    if not gated.empty:
        perf["regime_gated_sharpe"] = _safe_float(gated.iloc[-1].get("Sharpe", perf.get("regime_gated_sharpe")))
    if "average_cash" in data.columns:
        perf["average_cash"] = float(data["average_cash"].dropna().mean())
    if "max_drawdown" in data.columns:
        perf["max_drawdown"] = float(data["max_drawdown"].dropna().min())
    perf["best_window"] = "larger_walk_forward_2022_to_latest"
    return perf


def _risk_summary(sources: dict[str, pd.DataFrame], perf: dict[str, Any]) -> dict[str, Any]:
    wf = sources["walk_forward_summary"]
    result = {
        "VaR_95": "missing",
        "CVaR_95": "missing",
        "Sortino": np.nan,
        "Calmar": np.nan,
        "max_drawdown": perf.get("max_drawdown", np.nan),
    }
    if not wf.empty:
        row = wf.iloc[-1]
        result["Sortino"] = _safe_float(row.get("Sortino", np.nan))
        result["Calmar"] = _safe_float(row.get("Calmar", np.nan))
        result["max_drawdown"] = _safe_float(row.get("max_drawdown", result["max_drawdown"]))
    return result


def _triple_barrier_summary(labels: pd.DataFrame) -> dict[str, Any]:
    if labels.empty or "first_touch_type" not in labels.columns:
        return {
            "TP_rate": np.nan,
            "SL_rate": np.nan,
            "timeout_rate": np.nan,
            "selected_TP_rate": np.nan,
            "selected_SL_rate": np.nan,
        }
    tp = labels["first_touch_type"].eq("take_profit")
    sl = labels["first_touch_type"].eq("stop_loss")
    timeout = labels["first_touch_type"].eq("vertical_timeout")
    selected = labels["selected"].astype(bool) if "selected" in labels.columns else pd.Series(False, index=labels.index)
    selected_df = labels[selected]
    return {
        "TP_rate": float(tp.mean()),
        "SL_rate": float(sl.mean()),
        "timeout_rate": float(timeout.mean()),
        "selected_TP_rate": float(selected_df["first_touch_type"].eq("take_profit").mean()) if not selected_df.empty else np.nan,
        "selected_SL_rate": float(selected_df["first_touch_type"].eq("stop_loss").mean()) if not selected_df.empty else np.nan,
    }


def _feature_quality(feature_validation: pd.DataFrame, ic_history: pd.DataFrame) -> dict[str, Any]:
    best = "missing"
    worst = "missing"
    if not feature_validation.empty:
        score_col = "spearman_label_corr" if "spearman_label_corr" in feature_validation.columns else None
        if score_col:
            corr = pd.to_numeric(feature_validation[score_col], errors="coerce")
            if corr.notna().any() and "feature" in feature_validation.columns:
                best = str(feature_validation.loc[corr.idxmax(), "feature"])
                worst = str(feature_validation.loc[corr.idxmin(), "feature"])
    if best == "missing" and not ic_history.empty and "Average_IC" in ic_history.columns and "feature" in ic_history.columns:
        ic = pd.to_numeric(ic_history["Average_IC"], errors="coerce")
        if ic.notna().any():
            best = str(ic_history.loc[ic.idxmax(), "feature"])
            worst = str(ic_history.loc[ic.idxmin(), "feature"])
    return {"best_predictive_feature": best, "worst_predictive_feature": worst}


def _regime_summary(regime_df: pd.DataFrame, robustness: pd.DataFrame) -> dict[str, Any]:
    works = "missing"
    fails = "missing"
    if not regime_df.empty:
        regime_col = "regime" if "regime" in regime_df.columns else regime_df.columns[0]
        score_cols = [c for c in ["portfolio_sharpe", "realized_sharpe", "average_realized_return", "portfolio_return"] if c in regime_df.columns]
        if score_cols:
            score = pd.to_numeric(regime_df[score_cols[0]], errors="coerce")
            if score.notna().any():
                works = str(regime_df.loc[score.idxmax(), regime_col])
                fails = str(regime_df.loc[score.idxmin(), regime_col])
    verdict = "missing"
    if not robustness.empty and "rejection_reason" in robustness.columns:
        reasons = " ".join(robustness["rejection_reason"].fillna("").astype(str).tolist())
        verdict = "weak / diagnostic only" if reasons else "inconclusive"
    return {
        "regimes_where_model_works": works,
        "regimes_where_model_fails": fails,
        "regime_gated_verdict": verdict,
    }


def _anti_overfitting_summary(trial_log: pd.DataFrame) -> dict[str, Any]:
    total_trials = _trial_total(trial_log)
    best = _best_observed_sharpe()
    dist = _return_distribution_stats()
    sample_length = int(best.get("sample_length", 0) or dist.get("sample_size", 0) or 20)
    dsr = deflated_sharpe_diagnostic(
        observed_sharpe=float(best.get("observed_sharpe", 0.0)),
        number_of_trials=max(1, total_trials),
        sample_length=max(2, sample_length),
        skewness=float(dist.get("skewness", 0.0)),
        kurtosis=float(dist.get("kurtosis", 3.0)),
    )
    pbo = pbo_approximation(
        total_trials=max(1, total_trials),
        robustness_score=_robustness_score(),
        sample_size=max(2, sample_length),
        top_config_isolated=False,
    )
    blocked = []
    if dsr["warning_level"] in {"high", "extreme"}:
        blocked.append("deflated_sharpe_warning")
    if float(pbo["pbo_proxy"]) > 0.30:
        blocked.append("pbo_too_high")
    if _robustness_score() < 60:
        blocked.append("robustness_score_low")
    return {
        "total_trials": total_trials,
        "observed_best_sharpe": dsr["observed_sharpe"],
        "expected_max_sharpe": dsr["expected_max_sharpe_from_trials"],
        "deflated_sharpe": dsr["deflated_sharpe_estimate"],
        "overfitting_warning_level": dsr["warning_level"],
        "PBO_proxy": pbo["pbo_proxy"],
        "promotion_blocked_reasons": ", ".join(blocked) if blocked else "none",
    }


def _next_action(anti: dict[str, Any], perf: dict[str, Any], tb: dict[str, Any]) -> str:
    if anti["overfitting_warning_level"] in {"high", "extreme"} or anti["PBO_proxy"] > 0.5:
        return "reduce overfitting"
    if anti["total_trials"] < 50:
        return "collect more data"
    if pd.notna(tb.get("selected_SL_rate", np.nan)) and tb["selected_TP_rate"] < tb["selected_SL_rate"]:
        return "calibrate TP/SL"
    if pd.notna(perf.get("full_quant_sharpe", np.nan)) and perf.get("full_quant_sharpe", 0) < perf.get("baseline_sharpe", 0):
        return "calibrate full quant"
    return "insufficient evidence"


def build_research_dashboard() -> tuple[pd.DataFrame, str, list[str], list[str]]:
    sources = _load_sources()
    loaded, missing = _source_status(sources)
    total_trials = _trial_total(sources["strategy_trial_log"])
    anti = _anti_overfitting_summary(sources["strategy_trial_log"])
    perf = _apply_larger_walk_forward(
        _performance_summary(sources["full_quant_robustness"]),
        sources["larger_walk_forward_summary"],
    )
    risk = _risk_summary(sources, perf)
    tb = _triple_barrier_summary(sources["triple_barrier_labels"])
    feature = _feature_quality(sources["triple_barrier_feature_validation"], sources["ic_history"])
    regime = _regime_summary(sources["regime_performance"], sources["robustness_validation"])
    governance = _governance_status(sources["experiment_registry"], total_trials)
    promotion_status = "blocked" if anti["promotion_blocked_reasons"] != "none" else "research_review_only"
    next_action = _next_action(anti, perf, tb)

    rows = [
        ("System Status", "current_production_mode", "baseline"),
        ("System Status", "research_modes_available", "full_quant_research, regime_gated_full_quant"),
        ("System Status", "promotion_status", promotion_status),
        ("System Status", "overfitting_warning_level", anti["overfitting_warning_level"]),
        ("System Status", "total_strategy_trials", total_trials),
        ("System Status", "governance_status", governance),
        ("Performance Summary", "baseline_sharpe", perf["baseline_sharpe"]),
        ("Performance Summary", "full_quant_sharpe", perf["full_quant_sharpe"]),
        ("Performance Summary", "regime_gated_sharpe", perf["regime_gated_sharpe"]),
        ("Performance Summary", "best_window", perf["best_window"]),
        ("Performance Summary", "worst_window", perf["worst_window"]),
        ("Performance Summary", "average_cash", perf["average_cash"]),
        ("Performance Summary", "max_drawdown", perf["max_drawdown"]),
        ("Risk Summary", "VaR_95", risk["VaR_95"]),
        ("Risk Summary", "CVaR_95", risk["CVaR_95"]),
        ("Risk Summary", "Sortino", risk["Sortino"]),
        ("Risk Summary", "Calmar", risk["Calmar"]),
        ("Risk Summary", "max_drawdown", risk["max_drawdown"]),
        ("Signal Quality", "triple_barrier_TP_rate", tb["TP_rate"]),
        ("Signal Quality", "triple_barrier_SL_rate", tb["SL_rate"]),
        ("Signal Quality", "triple_barrier_timeout_rate", tb["timeout_rate"]),
        ("Signal Quality", "selected_only_TP_rate", tb["selected_TP_rate"]),
        ("Signal Quality", "selected_only_SL_rate", tb["selected_SL_rate"]),
        ("Signal Quality", "best_predictive_feature", feature["best_predictive_feature"]),
        ("Signal Quality", "worst_predictive_feature", feature["worst_predictive_feature"]),
        ("Regime Summary", "regimes_where_model_works", regime["regimes_where_model_works"]),
        ("Regime Summary", "regimes_where_model_fails", regime["regimes_where_model_fails"]),
        ("Regime Summary", "regime_gated_verdict", regime["regime_gated_verdict"]),
        ("Anti-Overfitting Summary", "observed_best_sharpe", anti["observed_best_sharpe"]),
        ("Anti-Overfitting Summary", "expected_max_sharpe", anti["expected_max_sharpe"]),
        ("Anti-Overfitting Summary", "deflated_sharpe", anti["deflated_sharpe"]),
        ("Anti-Overfitting Summary", "PBO_proxy", anti["PBO_proxy"]),
        ("Anti-Overfitting Summary", "promotion_blocked_reasons", anti["promotion_blocked_reasons"]),
        ("Next Action Recommendation", "recommendation", next_action),
    ]
    clean_eval = sources["clean_research_evaluation"]
    if not clean_eval.empty and "trial_group" in clean_eval.columns:
        for _, clean_row in clean_eval.iterrows():
            prefix = "all_time" if clean_row["trial_group"] == "exploratory_trials" else "governed"
            rows.extend(
                [
                    ("Clean Research Evaluation", f"{prefix}_total_trials", clean_row.get("total_trials", np.nan)),
                    ("Clean Research Evaluation", f"{prefix}_PBO_proxy", clean_row.get("PBO_proxy", np.nan)),
                    ("Clean Research Evaluation", f"{prefix}_deflated_sharpe", clean_row.get("deflated_sharpe", np.nan)),
                    ("Clean Research Evaluation", f"{prefix}_promotion_classification", clean_row.get("promotion_classification", "")),
                ]
            )
    summary = pd.DataFrame(rows, columns=["section", "metric", "value"])
    report = _format_report(summary, loaded, missing)
    return summary, report, loaded, missing


def _format_report(summary: pd.DataFrame, loaded: list[str], missing: list[str]) -> str:
    lines = ["===== RESEARCH DASHBOARD =====", ""]
    lines.append(f"Loaded sources: {', '.join(loaded) if loaded else 'none'}")
    lines.append(f"Missing sources: {', '.join(missing) if missing else 'none'}")
    for section, group in summary.groupby("section", sort=False):
        lines.append("")
        lines.append(f"--- {section} ---")
        for _, row in group.iterrows():
            lines.append(f"{row['metric']}: {_fmt(row['value'])}")
    return "\n".join(lines)


def run_research_dashboard(compact: bool = False) -> pd.DataFrame:
    summary, report, loaded, missing = build_research_dashboard()
    summary.to_csv(DASHBOARD_SUMMARY_FILE, index=False)
    Path(DASHBOARD_REPORT_FILE).write_text(report, encoding="utf-8")
    if compact:
        print("\n===== RESEARCH DASHBOARD =====")
        display = summary[summary["section"].isin(["System Status", "Anti-Overfitting Summary", "Next Action Recommendation"])]
        print(display.to_string(index=False))
    else:
        print(report)
    print(f"\nSaved: {Path(DASHBOARD_SUMMARY_FILE).resolve()}")
    print(f"Saved: {Path(DASHBOARD_REPORT_FILE).resolve()}")
    return summary


if __name__ == "__main__":
    run_research_dashboard()
