from __future__ import annotations

import json
from math import erf, log, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_FILE = "clean_research_evaluation.csv"


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _expected_max_sharpe(trials: int, sample_length: int) -> float:
    return float(sqrt(2.0 * log(max(1, trials))) / sqrt(max(2, sample_length)))


def _deflated_sharpe(observed: float, trials: int, sample_length: int) -> tuple[float, float, str]:
    expected_max = _expected_max_sharpe(trials, sample_length)
    denominator = 1.0 / sqrt(max(2, sample_length) - 1.0)
    z_score = (observed - expected_max) / max(denominator, 1e-8)
    p_value = float(1.0 - _normal_cdf(z_score))
    warning = "high" if p_value > 0.20 else "medium" if p_value > 0.05 else "low"
    return float(observed - expected_max), p_value, warning


def _all_time_trials(strategy_trial_log: pd.DataFrame) -> int:
    if strategy_trial_log.empty or "number_of_trials" not in strategy_trial_log.columns:
        return 0
    return int(pd.to_numeric(strategy_trial_log["number_of_trials"], errors="coerce").fillna(0).sum())


def _dashboard_value(dashboard: pd.DataFrame, metric: str, default: Any = np.nan) -> Any:
    if dashboard.empty or not {"metric", "value"}.issubset(dashboard.columns):
        return default
    rows = dashboard[dashboard["metric"].astype(str).eq(metric)]
    if rows.empty:
        return default
    return rows.iloc[-1]["value"]


def _exploratory_summary(
    *,
    trial_log: pd.DataFrame,
    dashboard: pd.DataFrame,
    full_quant: pd.DataFrame,
) -> dict[str, Any]:
    total_trials = _all_time_trials(trial_log)
    best_sharpe = _safe_float(_dashboard_value(dashboard, "observed_best_sharpe", np.nan))
    if not np.isfinite(best_sharpe) and not full_quant.empty:
        sharpe_cols = [c for c in full_quant.columns if "sharpe" in c.lower()]
        values = []
        for col in sharpe_cols:
            values.extend(pd.to_numeric(full_quant[col], errors="coerce").dropna().tolist())
        best_sharpe = float(np.nanmax(values)) if values else np.nan
    sample_length = int(max(2, len(full_quant))) if not full_quant.empty else 20
    expected_max = _expected_max_sharpe(max(1, total_trials), sample_length)
    pbo = _safe_float(_dashboard_value(dashboard, "PBO_proxy", 1.0), 1.0)
    warning = str(_dashboard_value(dashboard, "overfitting_warning_level", "high"))
    return {
        "trial_group": "exploratory_trials",
        "total_trials": total_trials,
        "best_sharpe": best_sharpe,
        "expected_max_sharpe": expected_max,
        "deflated_sharpe": best_sharpe - expected_max if np.isfinite(best_sharpe) else np.nan,
        "PBO_proxy": pbo,
        "overfitting_warning": warning,
        "robustness_score": _safe_float(_dashboard_value(dashboard, "robustness_score", np.nan)),
        "sample_size": sample_length,
        "promotion_classification": "still blocked",
    }


def _governed_summary(
    *,
    larger: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if larger.empty:
        return {
            "trial_group": "governed_trials",
            "total_trials": 0,
            "best_sharpe": np.nan,
            "expected_max_sharpe": np.nan,
            "deflated_sharpe": np.nan,
            "PBO_proxy": 1.0,
            "overfitting_warning": "missing_governed_runs",
            "robustness_score": np.nan,
            "sample_size": 0,
            "promotion_classification": "still blocked",
        }
    df = larger.copy()
    for col in ["Sharpe", "selected_only_sample_size", "number_of_test_dates", "TP_rate", "SL_rate", "average_turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    total_trials = int(len(df))
    best_sharpe = float(df["Sharpe"].max()) if "Sharpe" in df.columns else np.nan
    sample_size = int(df["selected_only_sample_size"].max()) if "selected_only_sample_size" in df.columns else 0
    test_dates = int(df["number_of_test_dates"].max()) if "number_of_test_dates" in df.columns else 0
    deflated, p_value, warning = _deflated_sharpe(best_sharpe, max(1, total_trials), max(2, test_dates))
    pbo_proxy = _governed_pbo_proxy(df, total_trials=total_trials, sample_size=sample_size)
    robustness_score = _governed_robustness_score(df)
    classification = _promotion_classification(
        df=df,
        config=config,
        sample_size=sample_size,
        test_dates=test_dates,
        deflated_sharpe=deflated,
        pbo_proxy=pbo_proxy,
        robustness_score=robustness_score,
    )
    return {
        "trial_group": "governed_trials",
        "total_trials": total_trials,
        "best_sharpe": best_sharpe,
        "expected_max_sharpe": _expected_max_sharpe(max(1, total_trials), max(2, test_dates)),
        "deflated_sharpe": deflated,
        "PBO_proxy": pbo_proxy,
        "overfitting_warning": warning,
        "robustness_score": robustness_score,
        "sample_size": sample_size,
        "number_of_test_dates": test_dates,
        "promotion_classification": classification,
    }


def _governed_pbo_proxy(df: pd.DataFrame, *, total_trials: int, sample_size: int) -> float:
    trial_penalty = min(0.15, log(max(1, total_trials)) / 20.0)
    sample_penalty = 0.05 if sample_size >= 150 else 0.25
    consistency_penalty = 0.0
    if {"TP_rate", "SL_rate"}.issubset(df.columns):
        if bool((df["TP_rate"] < df["SL_rate"]).fillna(False).any()):
            consistency_penalty += 0.20
    if "Sharpe" in df.columns and df["Sharpe"].dropna().std() > 0.75:
        consistency_penalty += 0.10
    return float(np.clip(0.10 + trial_penalty + sample_penalty + consistency_penalty, 0.0, 1.0))


def _governed_robustness_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    score = 50.0
    if "selected_only_sample_size" in df.columns and pd.to_numeric(df["selected_only_sample_size"], errors="coerce").max() >= 150:
        score += 20.0
    if {"TP_rate", "SL_rate"}.issubset(df.columns):
        tp = pd.to_numeric(df["TP_rate"], errors="coerce")
        sl = pd.to_numeric(df["SL_rate"], errors="coerce")
        score += 15.0 if bool((tp >= sl).all()) else -15.0
    if "Sharpe" in df.columns:
        sharpe = pd.to_numeric(df["Sharpe"], errors="coerce").dropna()
        if not sharpe.empty:
            score += 10.0 if sharpe.min() > 0 else -10.0
            score += 5.0 if sharpe.std() < 0.25 else 0.0
    return float(np.clip(score, 0.0, 100.0))


def _promotion_classification(
    *,
    df: pd.DataFrame,
    config: dict[str, Any],
    sample_size: int,
    test_dates: int,
    deflated_sharpe: float,
    pbo_proxy: float,
    robustness_score: float,
) -> str:
    rules = config.get("minimum_evidence_rules", {})
    min_sample = int(rules.get("minimum_selected_only_sample_size", 150))
    max_pbo = float(rules.get("maximum_PBO_allowed", 0.30))
    min_dsr = float(rules.get("minimum_deflated_sharpe_required", 0.50))
    min_robust = float(rules.get("minimum_robustness_score", 60.0))
    if sample_size < min_sample or test_dates < 60:
        return "still blocked"
    if pbo_proxy > max_pbo or robustness_score < min_robust:
        return "eligible for deeper validation"
    if deflated_sharpe >= min_dsr:
        return "eligible for paper trading only"
    return "eligible for deeper validation"


def build_clean_research_evaluation() -> pd.DataFrame:
    trial_log = _read_csv("strategy_trial_log.csv")
    registry = _read_csv("experiment_registry.csv")
    config = _read_json("constrained_research_config.json")
    larger = _read_csv("larger_walk_forward_summary.csv")
    dashboard = _read_csv("research_dashboard_summary.csv")
    full_quant = _read_csv("full_quant_robustness_walk_forward.csv")

    exploratory = _exploratory_summary(
        trial_log=trial_log,
        dashboard=dashboard,
        full_quant=full_quant,
    )
    governed = _governed_summary(
        larger=larger,
        config=config,
    )
    result = pd.DataFrame([exploratory, governed])
    result["governed_has_lower_overfitting_risk"] = (
        result["trial_group"].eq("governed_trials")
        & (pd.to_numeric(result["PBO_proxy"], errors="coerce") < pd.to_numeric(result.loc[result["trial_group"].eq("exploratory_trials"), "PBO_proxy"].iloc[0], errors="coerce"))
    )
    result["registry_rows"] = len(registry)
    return result


def _update_dashboard_with_clean_risk(evaluation: pd.DataFrame) -> None:
    dashboard_path = Path("research_dashboard_summary.csv")
    dashboard = _read_csv(dashboard_path)
    if dashboard.empty:
        return
    rows = []
    for _, row in evaluation.iterrows():
        prefix = "all_time" if row["trial_group"] == "exploratory_trials" else "governed"
        rows.extend(
            [
                {"section": "Clean Research Evaluation", "metric": f"{prefix}_total_trials", "value": row.get("total_trials", np.nan)},
                {"section": "Clean Research Evaluation", "metric": f"{prefix}_PBO_proxy", "value": row.get("PBO_proxy", np.nan)},
                {"section": "Clean Research Evaluation", "metric": f"{prefix}_deflated_sharpe", "value": row.get("deflated_sharpe", np.nan)},
                {"section": "Clean Research Evaluation", "metric": f"{prefix}_promotion_classification", "value": row.get("promotion_classification", "")},
            ]
        )
    clean = dashboard[dashboard["section"].astype(str).ne("Clean Research Evaluation")]
    updated = pd.concat([clean, pd.DataFrame(rows)], ignore_index=True)
    updated.to_csv(dashboard_path, index=False)


def print_clean_research_evaluation(evaluation: pd.DataFrame) -> None:
    print("\n===== CLEAN RESEARCH EVALUATION =====")
    print(evaluation.to_string(index=False))
    if len(evaluation) >= 2:
        exploratory = evaluation[evaluation["trial_group"].eq("exploratory_trials")].iloc[0]
        governed = evaluation[evaluation["trial_group"].eq("governed_trials")].iloc[0]
        cleaner = float(governed["PBO_proxy"]) < float(exploratory["PBO_proxy"])
        print(f"\nGoverned research has lower overfitting risk: {cleaner}")
        print(f"Governed classification: {governed['promotion_classification']}")
        print("Production promotion remains blocked.")


def run_clean_research_evaluation() -> pd.DataFrame:
    evaluation = build_clean_research_evaluation()
    evaluation.to_csv(OUTPUT_FILE, index=False)
    _update_dashboard_with_clean_risk(evaluation)
    print_clean_research_evaluation(evaluation)
    print(f"\nSaved: {Path(OUTPUT_FILE).resolve()}")
    print(f"Updated: {Path('research_dashboard_summary.csv').resolve()}")
    return evaluation


if __name__ == "__main__":
    run_clean_research_evaluation()
