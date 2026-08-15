from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_FILE = "robustness_validation.csv"

THRESHOLD_DEFAULTS = {
    "signal_strength_threshold": 0.20,
    "target_confidence_threshold": 0.40,
    "quality_score_threshold": 0.40,
    "expected_return_threshold": 0.0000,
    "max_selected_assets": 4,
    "min_selected_assets": 2,
}

BARRIER_BASELINE = {
    "horizon": 20,
    "tp_multiple": 1.0,
    "sl_multiple": 1.0,
    "subset": "selected_only",
}


def _load_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path).replace([np.inf, -np.inf], np.nan)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _score_to_100(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0) * 100.0)


def _parameter_distance(row: pd.Series, defaults: dict[str, float]) -> float:
    components = []
    for key, default in defaults.items():
        if key not in row:
            continue
        scale = max(abs(float(default)), 1.0)
        if key == "expected_return_threshold":
            scale = 0.0015
        components.append(abs(_safe_float(row[key]) - float(default)) / scale)
    return float(np.mean(components)) if components else 0.0


def _barrier_cluster_score(top: pd.DataFrame) -> tuple[float, str]:
    if len(top) < 3:
        return 0.0, "insufficient_top_configs"

    parameter_cols = ["horizon", "tp_multiple", "sl_multiple"]
    numeric = top[parameter_cols].apply(pd.to_numeric, errors="coerce")
    ranges = {
        "horizon": 25.0,
        "tp_multiple": 1.25,
        "sl_multiple": 1.25,
    }
    normalized_std = []
    for col in parameter_cols:
        normalized_std.append(_safe_float(numeric[col].std(ddof=0)) / ranges[col])
    spread = float(np.mean(normalized_std))
    score = float(np.clip(100.0 * (1.0 - spread), 0.0, 100.0))
    if score >= 75:
        status = "stable_cluster"
    elif score >= 45:
        status = "moderate_cluster"
    else:
        status = "isolated_or_unstable"
    return score, status


def _performance_dispersion(top: pd.DataFrame, metric: str = "Sharpe") -> float:
    values = pd.to_numeric(top.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    mean_abs = max(abs(float(values.mean())), 1e-9)
    return float(values.std(ddof=0) / mean_abs)


def _barrier_baseline(barrier: pd.DataFrame) -> pd.Series | None:
    if barrier.empty:
        return None
    mask = pd.Series(True, index=barrier.index)
    for key, value in BARRIER_BASELINE.items():
        if key in barrier.columns:
            mask &= barrier[key] == value
    baseline = barrier[mask]
    if baseline.empty:
        return None
    return baseline.iloc[0]


def _barrier_reasons(best: pd.Series, baseline: pd.Series | None, top10: pd.DataFrame) -> list[str]:
    reasons = []
    sample_size = int(_safe_float(best.get("sample_size")))
    if sample_size < 100:
        reasons.append("small_sample_size")
    if _safe_float(best.get("TP_rate")) < _safe_float(best.get("SL_rate")):
        reasons.append("TP_rate_below_SL_rate")
    if baseline is not None:
        improvement = _safe_float(best.get("Sharpe")) - _safe_float(baseline.get("Sharpe"))
        if improvement < 0.05:
            reasons.append("marginal_improvement_over_baseline")
    dispersion = _performance_dispersion(top10, "Sharpe")
    if dispersion > 0.75:
        reasons.append("top_config_performance_dispersion_high")
    _, cluster_status = _barrier_cluster_score(top10)
    if cluster_status == "isolated_or_unstable":
        reasons.append("best_config_not_in_stable_cluster")
    return reasons


def _barrier_score(best: pd.Series, baseline: pd.Series | None, top10: pd.DataFrame) -> float:
    sample_score = _score_to_100(_safe_float(best.get("sample_size")), 30, 150)
    tp_sl_score = 100.0 if _safe_float(best.get("TP_rate")) > _safe_float(best.get("SL_rate")) else 35.0
    cluster_score, _ = _barrier_cluster_score(top10)
    dispersion_penalty = min(_performance_dispersion(top10, "Sharpe") * 35.0, 35.0)
    baseline_score = 50.0
    if baseline is not None:
        improvement = _safe_float(best.get("Sharpe")) - _safe_float(baseline.get("Sharpe"))
        baseline_score = _score_to_100(improvement, 0.0, 0.20)
    score = 0.30 * sample_score + 0.25 * cluster_score + 0.20 * tp_sl_score + 0.15 * baseline_score + 0.10 * (100.0 - dispersion_penalty)
    return float(np.clip(score, 0.0, 100.0))


def _analyze_barrier(barrier: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if barrier.empty:
        return pd.DataFrame(), {
            "best_barrier_config": "missing",
            "barrier_robustness_score": 0.0,
            "barrier_overfit_warning": "missing_input",
        }
    selected = barrier[barrier.get("subset", "") == "selected_only"].copy()
    if selected.empty:
        selected = barrier.copy()
    selected = selected.sort_values(
        ["Sharpe", "avg_return", "TP_minus_SL", "SL_rate", "avg_time_to_exit"],
        ascending=[False, False, False, True, True],
    )
    top10 = selected.head(10).copy()
    best = selected.iloc[0]
    baseline = _barrier_baseline(barrier)
    reasons = _barrier_reasons(best, baseline, top10)
    cluster_score, cluster_status = _barrier_cluster_score(top10)
    score = _barrier_score(best, baseline, top10)
    rows = []
    for rank, (_, row) in enumerate(top10.iterrows(), start=1):
        row_reasons = _barrier_reasons(row, baseline, top10)
        rows.append(
            {
                "analysis_type": "barrier",
                "rank": rank,
                "horizon": row.get("horizon"),
                "tp_multiple": row.get("tp_multiple"),
                "sl_multiple": row.get("sl_multiple"),
                "sample_size": row.get("sample_size"),
                "Sharpe": row.get("Sharpe"),
                "Sortino": row.get("Sortino"),
                "avg_return": row.get("avg_return"),
                "TP_rate": row.get("TP_rate"),
                "SL_rate": row.get("SL_rate"),
                "TP_minus_SL": row.get("TP_minus_SL"),
                "turnover_proxy": np.nan,
                "average_selected_count": np.nan,
                "parameter_distance": np.nan,
                "robustness_score": score if rank == 1 else np.nan,
                "cluster_status": cluster_status,
                "cluster_score": cluster_score,
                "rejected": bool(row_reasons),
                "rejection_reason": ", ".join(row_reasons),
            }
        )
    summary = {
        "best_barrier_config": f"h={best.get('horizon')}, tp={best.get('tp_multiple')}, sl={best.get('sl_multiple')}",
        "barrier_robustness_score": score,
        "barrier_overfit_warning": ", ".join(reasons) if reasons else "none",
        "barrier_cluster_status": cluster_status,
        "barrier_cluster_score": cluster_score,
    }
    return pd.DataFrame(rows), summary


def _threshold_rejection_reason(row: pd.Series, *, min_sample_size: int = 50) -> list[str]:
    reasons = []
    if int(_safe_float(row.get("sample_size"))) < min_sample_size:
        reasons.append("small_sample_size")
    if _safe_float(row.get("average_selected_count")) < 2.0:
        reasons.append("selected_count_too_low")
    if _safe_float(row.get("TP_rate")) < _safe_float(row.get("SL_rate")):
        reasons.append("TP_rate_below_SL_rate")
    if _safe_float(row.get("turnover_proxy")) > 0.80:
        reasons.append("extreme_turnover")
    distance = _parameter_distance(row, THRESHOLD_DEFAULTS)
    if distance > 0.60:
        reasons.append("too_far_from_defaults")
    warning_flags = str(row.get("warning_flags", ""))
    if warning_flags and warning_flags.lower() != "nan":
        for flag in warning_flags.split(","):
            flag = flag.strip()
            if flag and flag not in reasons:
                reasons.append(flag)
    return reasons


def _threshold_score(row: pd.Series, top20: pd.DataFrame) -> float:
    sample_score = _score_to_100(_safe_float(row.get("sample_size")), 25, 150)
    selected_score = _score_to_100(_safe_float(row.get("average_selected_count")), 1.5, 4.0)
    tp_sl_score = 100.0 if _safe_float(row.get("TP_rate")) > _safe_float(row.get("SL_rate")) else 35.0
    turnover_score = float(np.clip(100.0 * (1.0 - _safe_float(row.get("turnover_proxy"))), 0.0, 100.0))
    distance_score = float(np.clip(100.0 * (1.0 - _parameter_distance(row, THRESHOLD_DEFAULTS)), 0.0, 100.0))
    dispersion_penalty = min(_performance_dispersion(top20, "Sharpe") * 30.0, 30.0)
    score = (
        0.25 * sample_score
        + 0.20 * selected_score
        + 0.20 * tp_sl_score
        + 0.15 * turnover_score
        + 0.15 * distance_score
        + 0.05 * (100.0 - dispersion_penalty)
    )
    return float(np.clip(score, 0.0, 100.0))


def _stable_ranges(accepted: pd.DataFrame, parameter_cols: list[str]) -> dict[str, str]:
    if accepted.empty:
        return {col: "none" for col in parameter_cols}
    ranges = {}
    for col in parameter_cols:
        values = pd.to_numeric(accepted[col], errors="coerce").dropna().sort_values().unique()
        if len(values) == 0:
            ranges[col] = "none"
        elif len(values) == 1:
            ranges[col] = str(values[0])
        else:
            ranges[col] = f"{values[0]} to {values[-1]}"
    return ranges


def _analyze_threshold(thresholds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if thresholds.empty:
        return pd.DataFrame(), {
            "best_threshold_config": "missing",
            "threshold_robustness_score": 0.0,
            "threshold_overfit_warning": "missing_input",
        }
    ranked = thresholds.sort_values(
        ["Sharpe", "Sortino", "average_portfolio_return", "TP_minus_SL", "SL_rate"],
        ascending=[False, False, False, False, True],
    ).copy()
    top20 = ranked.head(20).copy()
    best = ranked.iloc[0]
    best_reasons = _threshold_rejection_reason(best)
    best_score = _threshold_score(best, top20)

    rows = []
    for rank, (_, row) in enumerate(top20.iterrows(), start=1):
        reasons = _threshold_rejection_reason(row)
        rows.append(
            {
                "analysis_type": "threshold",
                "rank": rank,
                "horizon": row.get("horizon"),
                "signal_strength_threshold": row.get("signal_strength_threshold"),
                "target_confidence_threshold": row.get("target_confidence_threshold"),
                "quality_score_threshold": row.get("quality_score_threshold"),
                "expected_return_threshold": row.get("expected_return_threshold"),
                "max_selected_assets": row.get("max_selected_assets"),
                "min_selected_assets": row.get("min_selected_assets"),
                "sample_size": row.get("sample_size"),
                "Sharpe": row.get("Sharpe"),
                "Sortino": row.get("Sortino"),
                "avg_return": row.get("average_portfolio_return"),
                "TP_rate": row.get("TP_rate"),
                "SL_rate": row.get("SL_rate"),
                "TP_minus_SL": row.get("TP_minus_SL"),
                "turnover_proxy": row.get("turnover_proxy"),
                "average_selected_count": row.get("average_selected_count"),
                "parameter_distance": _parameter_distance(row, THRESHOLD_DEFAULTS),
                "robustness_score": best_score if rank == 1 else np.nan,
                "cluster_status": np.nan,
                "cluster_score": np.nan,
                "rejected": bool(reasons),
                "rejection_reason": ", ".join(reasons),
            }
        )

    accepted = ranked[ranked.apply(lambda row: not _threshold_rejection_reason(row), axis=1)]
    parameter_cols = [
        "signal_strength_threshold",
        "target_confidence_threshold",
        "quality_score_threshold",
        "expected_return_threshold",
        "max_selected_assets",
        "min_selected_assets",
    ]
    ranges = _stable_ranges(accepted.head(50), parameter_cols)
    summary = {
        "best_threshold_config": (
            f"h={best.get('horizon')}, signal={best.get('signal_strength_threshold')}, "
            f"confidence={best.get('target_confidence_threshold')}, quality={best.get('quality_score_threshold')}, "
            f"expected={best.get('expected_return_threshold')}, max_assets={best.get('max_selected_assets')}"
        ),
        "threshold_robustness_score": best_score,
        "threshold_overfit_warning": ", ".join(best_reasons) if best_reasons else "none",
        "stable_parameter_ranges": ranges,
        "accepted_config_count": len(accepted),
        "rejected_config_count": int(len(ranked) - len(accepted)),
    }
    return pd.DataFrame(rows), summary


def run_robustness_validation(
    *,
    barrier_path: str | Path = "barrier_parameter_optimization.csv",
    threshold_path: str | Path = "threshold_optimization.csv",
    predictions_path: str | Path = "walk_forward_predictions.csv",
    portfolio_returns_path: str | Path = "walk_forward_portfolio_returns.csv",
    labels_path: str | Path = "triple_barrier_labels.csv",
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    barrier = _load_csv(barrier_path)
    thresholds = _load_csv(threshold_path)
    predictions = _load_csv(predictions_path)
    portfolio_returns = _load_csv(portfolio_returns_path)
    labels = _load_csv(labels_path)

    barrier_report, barrier_summary = _analyze_barrier(barrier)
    threshold_report, threshold_summary = _analyze_threshold(thresholds)
    result = pd.concat([barrier_report, threshold_report], ignore_index=True, sort=False)
    result.to_csv(output_path, index=False)

    barrier_score = _safe_float(barrier_summary.get("barrier_robustness_score"))
    threshold_score = _safe_float(threshold_summary.get("threshold_robustness_score"))
    overall_score = float(np.nanmean([barrier_score, threshold_score]))
    overfit_reasons = [
        str(barrier_summary.get("barrier_overfit_warning", "")),
        str(threshold_summary.get("threshold_overfit_warning", "")),
    ]
    overfit_warning = "; ".join(reason for reason in overfit_reasons if reason and reason != "none") or "none"

    print("\n===== ROBUSTNESS VALIDATION REPORT =====")
    print(f"input rows: barrier={len(barrier)}, threshold={len(thresholds)}, predictions={len(predictions)}, portfolio={len(portfolio_returns)}, labels={len(labels)}")
    print(f"best barrier config: {barrier_summary.get('best_barrier_config')}")
    print(f"best threshold config: {threshold_summary.get('best_threshold_config')}")
    print(f"barrier robustness score: {barrier_score:.2f}")
    print(f"threshold robustness score: {threshold_score:.2f}")
    print(f"robustness_score: {overall_score:.2f}")
    print(f"overfit_warning: {overfit_warning}")

    print("\n===== PARAMETER STABILITY REPORT =====")
    print(f"barrier cluster: {barrier_summary.get('barrier_cluster_status')} ({_safe_float(barrier_summary.get('barrier_cluster_score')):.2f})")
    stable_ranges = threshold_summary.get("stable_parameter_ranges", {})
    if isinstance(stable_ranges, dict):
        print("stable threshold ranges:")
        for key, value in stable_ranges.items():
            print(f"  {key}: {value}")
    print(f"accepted threshold configs: {threshold_summary.get('accepted_config_count', 0)}")
    print(f"rejected threshold configs: {threshold_summary.get('rejected_config_count', 0)}")

    if not result.empty:
        display_cols = [
            "analysis_type",
            "rank",
            "horizon",
            "Sharpe",
            "Sortino",
            "avg_return",
            "sample_size",
            "TP_rate",
            "SL_rate",
            "TP_minus_SL",
            "turnover_proxy",
            "average_selected_count",
            "robustness_score",
            "rejected",
            "rejection_reason",
        ]
        print("\nTop robustness rows:")
        print(result[[col for col in display_cols if col in result.columns]].head(20).to_string(index=False))

    print(f"\nSaved: {Path(output_path).resolve()}")
    print("safe_to_promote: no")
    return result


if __name__ == "__main__":
    run_robustness_validation()
