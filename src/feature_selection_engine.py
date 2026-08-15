from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILES = {
    "meta_feature_ranking": "meta_feature_ranking.csv",
    "meta_model_results": "meta_model_results.csv",
    "meta_label_dataset": "meta_label_dataset.csv",
    "kalman_feature_validation": "kalman_feature_validation.csv",
    "kalman_meta_label_comparison": "kalman_meta_label_comparison.csv",
    "historical_ic_dataset": "historical_ic_dataset.csv",
    "triple_barrier_feature_validation": "triple_barrier_feature_validation.csv",
    "ic_history": "ic_history.csv",
}

OUTPUT_REPORT = "feature_selection_report.csv"
OUTPUT_JSON = "selected_feature_set.json"

EXCLUDED_ML_FEATURES = {
    "selected",
    "selected_prediction",
    "weight",
    "target_price",
    "current_price",
    "current_price_prediction",
    "take_profit_price",
    "stop_loss_price",
    "vertical_barrier_date",
    "first_touch_date",
    "first_touch_type",
    "label",
    "realized_return_at_barrier",
    "realized_return_for_meta",
    "meta_label",
    "volatility_horizon",
}


@dataclass
class FeatureSelectionConfig:
    redundancy_threshold: float = 0.85
    min_sample_core: int = 300
    min_sample_supporting: int = 100
    core_threshold: float = 0.70
    supporting_threshold: float = 0.50
    diagnostic_threshold: float = 0.30


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _normalize_abs(series: pd.Series) -> pd.Series:
    values = _safe_numeric(series, 0.0).abs()
    max_value = float(values.max()) if len(values) else 0.0
    if max_value <= 0:
        return pd.Series(0.0, index=series.index)
    return (values / max_value).clip(0.0, 1.0)


def _score_from_classification(value: object) -> float:
    mapping = {
        "excellent": 1.0,
        "strong": 0.9,
        "useful": 0.7,
        "weak": 0.35,
        "noise": 0.05,
    }
    return mapping.get(str(value).strip().lower(), 0.0)


def _collect_feature_names(sources: dict[str, pd.DataFrame]) -> list[str]:
    names: set[str] = set()
    for key in ["meta_feature_ranking", "kalman_feature_validation", "triple_barrier_feature_validation", "ic_history"]:
        df = sources.get(key, pd.DataFrame())
        if not df.empty and "feature" in df.columns:
            names.update(df["feature"].dropna().astype(str).tolist())
    dataset = sources.get("meta_label_dataset", pd.DataFrame())
    if not dataset.empty:
        excluded = {
            "date",
            "ticker",
            "model_mode",
            "selected",
            "regime",
            "label",
            "first_touch_type",
            "realized_return_at_barrier",
            "realized_return_20d",
            "realized_return_for_meta",
            "meta_label",
        } | EXCLUDED_ML_FEATURES
        names.update([c for c in dataset.columns if c not in excluded])
    return sorted(feature for feature in names if feature not in EXCLUDED_ML_FEATURES)


def _meta_evidence(feature: str, meta_ranking: pd.DataFrame) -> dict[str, float]:
    if meta_ranking.empty or "feature" not in meta_ranking.columns:
        return {}
    row = meta_ranking[meta_ranking["feature"].astype(str).eq(feature)]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {
        "information_value": float(row.get("information_value", 0.0) or 0.0),
        "mutual_information": float(row.get("mutual_information", 0.0) or 0.0),
        "meta_spearman": abs(float(row.get("spearman_correlation", 0.0) or 0.0)),
        "meta_feature_score": float(row.get("feature_score", 0.0) or 0.0),
        "meta_sample_size": int(float(row.get("sample_size", 0.0) or 0.0)),
    }


def _kalman_evidence(feature: str, kalman_validation: pd.DataFrame) -> dict[str, float]:
    if kalman_validation.empty or "feature" not in kalman_validation.columns:
        return {}
    row = kalman_validation[kalman_validation["feature"].astype(str).eq(feature)]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {
        "kalman_spearman_ic": abs(float(row.get("spearman_ic", 0.0) or 0.0)),
        "kalman_mutual_information": float(row.get("mutual_information", 0.0) or 0.0),
        "kalman_predictive_score": float(row.get("predictive_score", 0.0) or 0.0),
        "kalman_sample_size": int(float(row.get("sample_size", 0.0) or 0.0)),
    }


def _triple_barrier_evidence(feature: str, tb_validation: pd.DataFrame) -> dict[str, float]:
    if tb_validation.empty or "feature" not in tb_validation.columns:
        return {}
    rows = tb_validation[tb_validation["feature"].astype(str).eq(feature)].copy()
    if rows.empty:
        return {}
    rows["abs_corr"] = _safe_numeric(rows.get("spearman_label_corr", pd.Series(0, index=rows.index))).abs()
    rows["classification_score"] = rows.get("classification", pd.Series("", index=rows.index)).map(_score_from_classification)
    rows["tp_minus_sl"] = _safe_numeric(rows.get("TP_minus_SL_top_quintile", pd.Series(0, index=rows.index)), 0.0)
    horizons = rows["horizon"].nunique() if "horizon" in rows.columns else 1
    return {
        "triple_barrier_corr": float(rows["abs_corr"].mean()),
        "triple_barrier_score": float(rows["classification_score"].mean()),
        "triple_barrier_tp_minus_sl": float(rows["tp_minus_sl"].mean()),
        "triple_barrier_sample_size": int(_safe_numeric(rows.get("sample_size", pd.Series(0, index=rows.index))).max()),
        "horizon_count": int(horizons),
    }


def _ic_history_evidence(feature: str, ic_history: pd.DataFrame) -> dict[str, float]:
    if ic_history.empty or "feature" not in ic_history.columns:
        return {}
    rows = ic_history[ic_history["feature"].astype(str).eq(feature)].copy()
    if rows.empty:
        return {}
    ic_cols = [c for c in ["IC_5D", "IC_10D", "IC_20D", "Average_IC"] if c in rows.columns]
    if not ic_cols:
        return {}
    values = rows[ic_cols].apply(pd.to_numeric, errors="coerce").abs()
    return {
        "ic_history_strength": float(values.mean(axis=1).mean(skipna=True) or 0.0),
        "ic_history_sample_rows": len(rows),
    }


def _redundancy_penalties(features: list[str], dataset: pd.DataFrame, base_scores: pd.Series, threshold: float) -> dict[str, float]:
    if dataset.empty:
        return {feature: 0.0 for feature in features}
    available = [feature for feature in features if feature in dataset.columns]
    if len(available) < 2:
        return {feature: 0.0 for feature in features}
    matrix = dataset[available].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True).fillna(0.0))
    corr = matrix.corr(method="pearson").abs().fillna(0.0)
    penalties = {feature: 0.0 for feature in features}
    for feature in available:
        better = [
            other
            for other in available
            if other != feature and corr.loc[feature, other] >= threshold and base_scores.get(other, 0.0) > base_scores.get(feature, 0.0)
        ]
        if better:
            penalties[feature] = min(0.25, 0.08 * len(better))
    return penalties


def _meta_degradation_penalty(kalman_meta: pd.DataFrame) -> dict[str, float]:
    if kalman_meta.empty or "model_variant" not in kalman_meta.columns:
        return {}
    base = kalman_meta[kalman_meta["model_variant"].astype(str).eq("base_meta_features")]
    plus = kalman_meta[kalman_meta["model_variant"].astype(str).eq("base_plus_kalman")]
    if base.empty or plus.empty:
        return {}
    base_auc = float(base.iloc[0].get("auc", np.nan))
    plus_auc = float(plus.iloc[0].get("auc", np.nan))
    if not np.isfinite(base_auc) or not np.isfinite(plus_auc) or plus_auc >= base_auc:
        return {}
    return {"kalman": min(0.20, max(0.05, base_auc - plus_auc))}


def _classify(row: pd.Series, config: FeatureSelectionConfig) -> tuple[str, str]:
    score = float(row["feature_selection_score"])
    sample_size = int(row.get("max_sample_size", 0))
    if sample_size < config.min_sample_supporting:
        return "NEEDS_MORE_DATA", "collect_more_observations_before_ml_use"
    if row.get("structural_penalty", 0.0) >= 1.0:
        return "REMOVE_FROM_ML", "structural_or_label_leakage_risk"
    if row.get("meta_degradation_penalty", 0.0) > 0.0 and str(row.get("feature", "")).startswith("kalman"):
        if score < config.core_threshold:
            return "DIAGNOSTIC_ONLY", "kalman_degraded_meta_label_auc"
    if score >= config.core_threshold and sample_size >= config.min_sample_core:
        return "CORE", "eligible_as_primary_ml_feature_research"
    if score >= config.supporting_threshold:
        return "SUPPORTING", "use_as_secondary_feature_with_monitoring"
    if score >= config.diagnostic_threshold:
        return "DIAGNOSTIC_ONLY", "retain_for_reports_not_model_input_yet"
    return "REMOVE_FROM_ML", "low_evidence_or_redundant"


def run_feature_selection_engine(config: FeatureSelectionConfig | None = None) -> pd.DataFrame:
    config = config or FeatureSelectionConfig()
    sources = {name: _read_csv(path) for name, path in INPUT_FILES.items()}
    features = _collect_feature_names(sources)
    rows = []
    for feature in features:
        evidence = {"feature": feature}
        evidence.update(_meta_evidence(feature, sources["meta_feature_ranking"]))
        evidence.update(_kalman_evidence(feature, sources["kalman_feature_validation"]))
        evidence.update(_triple_barrier_evidence(feature, sources["triple_barrier_feature_validation"]))
        evidence.update(_ic_history_evidence(feature, sources["ic_history"]))
        rows.append(evidence)

    report = pd.DataFrame(rows).fillna(0.0)
    if report.empty:
        report.to_csv(OUTPUT_REPORT, index=False)
        Path(OUTPUT_JSON).write_text(
            json.dumps({"CORE": [], "SUPPORTING": [], "DIAGNOSTIC_ONLY": [], "REMOVE_FROM_ML": [], "NEEDS_MORE_DATA": []}, indent=2)
        )
        return report

    for column in [
        "information_value",
        "mutual_information",
        "meta_spearman",
        "kalman_spearman_ic",
        "kalman_mutual_information",
        "triple_barrier_corr",
        "triple_barrier_tp_minus_sl",
        "ic_history_strength",
    ]:
        if column not in report.columns:
            report[column] = 0.0
        report[f"{column}_score"] = _normalize_abs(report[column])

    if "meta_feature_score" not in report.columns:
        report["meta_feature_score"] = 0.0
    if "kalman_predictive_score" not in report.columns:
        report["kalman_predictive_score"] = 0.0
    if "triple_barrier_score" not in report.columns:
        report["triple_barrier_score"] = 0.0
    if "horizon_count" not in report.columns:
        report["horizon_count"] = 0

    report["ic_strength_score"] = report[
        ["meta_spearman_score", "kalman_spearman_ic_score", "triple_barrier_corr_score", "ic_history_strength_score"]
    ].max(axis=1)
    report["mi_score"] = report[["mutual_information_score", "kalman_mutual_information_score"]].max(axis=1)
    report["information_value_score"] = report["information_value_score"]
    report["meta_label_importance_score"] = _safe_numeric(report["meta_feature_score"], 0.0).clip(0.0, 1.0)
    report["triple_barrier_predictive_score"] = (
        0.55 * _safe_numeric(report["triple_barrier_score"], 0.0).clip(0.0, 1.0)
        + 0.25 * report["triple_barrier_corr_score"]
        + 0.20 * report["triple_barrier_tp_minus_sl_score"]
    ).clip(0.0, 1.0)
    report["stability_score"] = (pd.to_numeric(report["horizon_count"], errors="coerce").fillna(0.0) / 3.0).clip(0.0, 1.0)
    report["kalman_score"] = _safe_numeric(report["kalman_predictive_score"], 0.0).clip(0.0, 1.0)

    base_score = (
        0.25 * report["ic_strength_score"]
        + 0.18 * report["mi_score"]
        + 0.14 * report["information_value_score"]
        + 0.18 * report["meta_label_importance_score"]
        + 0.15 * report["triple_barrier_predictive_score"]
        + 0.06 * report["stability_score"]
        + 0.04 * report["kalman_score"]
    ).clip(0.0, 1.0)

    report["base_feature_score"] = base_score
    redundancy = _redundancy_penalties(
        features, sources["meta_label_dataset"], base_score.reindex(report.index), config.redundancy_threshold
    )
    report["redundancy_penalty"] = report["feature"].map(redundancy).fillna(0.0)
    degradation = _meta_degradation_penalty(sources["kalman_meta_label_comparison"])
    report["meta_degradation_penalty"] = (
        report["feature"].astype(str).map(lambda x: degradation.get("kalman", 0.0) if x.startswith("kalman") else 0.0)
    )
    report["structural_penalty"] = report["feature"].astype(str).isin(EXCLUDED_ML_FEATURES).astype(float)
    report["feature_selection_score"] = (
        report["base_feature_score"] - report["redundancy_penalty"] - report["meta_degradation_penalty"] - report["structural_penalty"]
    ).clip(0.0, 1.0)

    sample_columns = [c for c in report.columns if c.endswith("sample_size") or c.endswith("sample_rows")]
    report["max_sample_size"] = report[sample_columns].max(axis=1) if sample_columns else 0
    classifications = report.apply(lambda row: _classify(row, config), axis=1)
    report["classification"] = [item[0] for item in classifications]
    report["recommended_usage"] = [item[1] for item in classifications]
    report["evidence_summary"] = report.apply(_evidence_summary, axis=1)
    report = report.sort_values(["feature_selection_score", "max_sample_size"], ascending=False)

    report.to_csv(OUTPUT_REPORT, index=False)
    _write_selected_feature_json(report)
    _print_report(report)
    return report


def _evidence_summary(row: pd.Series) -> str:
    parts = []
    if row.get("ic_strength_score", 0.0) > 0:
        parts.append(f"IC={row.get('ic_strength_score', 0.0):.2f}")
    if row.get("mi_score", 0.0) > 0:
        parts.append(f"MI={row.get('mi_score', 0.0):.2f}")
    if row.get("information_value_score", 0.0) > 0:
        parts.append(f"IV={row.get('information_value_score', 0.0):.2f}")
    if row.get("triple_barrier_predictive_score", 0.0) > 0:
        parts.append(f"TB={row.get('triple_barrier_predictive_score', 0.0):.2f}")
    if row.get("redundancy_penalty", 0.0) > 0:
        parts.append(f"redundancy_penalty={row.get('redundancy_penalty', 0.0):.2f}")
    if row.get("meta_degradation_penalty", 0.0) > 0:
        parts.append(f"meta_degradation_penalty={row.get('meta_degradation_penalty', 0.0):.2f}")
    return "; ".join(parts) if parts else "no_strong_evidence"


def _write_selected_feature_json(report: pd.DataFrame) -> None:
    payload = {}
    for classification in ["CORE", "SUPPORTING", "DIAGNOSTIC_ONLY", "REMOVE_FROM_ML", "NEEDS_MORE_DATA"]:
        payload[classification] = report[report["classification"].eq(classification)]["feature"].astype(str).tolist()
    payload["metadata"] = {
        "production_behavior_changed": False,
        "research_only": True,
        "score_column": "feature_selection_score",
    }
    Path(OUTPUT_JSON).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_report(report: pd.DataFrame) -> None:
    print("\n===== FEATURE SELECTION REPORT =====")
    columns = ["feature", "feature_selection_score", "classification", "evidence_summary", "recommended_usage"]
    print(report[columns].head(25).to_string(index=False))

    print("\n===== FEATURES TO KEEP =====")
    keep = report[report["classification"].isin(["CORE", "SUPPORTING"])]
    print(keep[["feature", "classification", "feature_selection_score"]].head(20).to_string(index=False) if not keep.empty else "none")

    print("\n===== FEATURES TO DEMOTE =====")
    demote = report[report["classification"].eq("DIAGNOSTIC_ONLY")]
    print(
        demote[["feature", "feature_selection_score", "recommended_usage"]].head(20).to_string(index=False) if not demote.empty else "none"
    )

    print("\n===== FEATURES TO REMOVE FROM ML =====")
    remove = report[report["classification"].eq("REMOVE_FROM_ML")]
    print(
        remove[["feature", "feature_selection_score", "recommended_usage"]].head(20).to_string(index=False) if not remove.empty else "none"
    )


if __name__ == "__main__":
    run_feature_selection_engine()
