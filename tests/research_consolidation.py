from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REGISTRY_FILE = "model_lifecycle_status.csv"
SUMMARY_FILE = "final_research_summary.txt"


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _metric_from_results(path: str, row_key: str, row_value: str, metric: str) -> float:
    df = _read_csv(path)
    if df.empty or row_key not in df.columns or metric not in df.columns:
        return np.nan
    rows = df[df[row_key].astype(str).eq(row_value)]
    if rows.empty:
        return np.nan
    return _safe_float(rows.iloc[-1][metric])


def _governance_row(path: str) -> pd.Series:
    df = _read_csv(path)
    if df.empty:
        return pd.Series(dtype=object)
    return df.iloc[0]


def _evidence_strength(sample_size: float, walk_forward: bool, overfit_warning: str = "") -> str:
    if not np.isfinite(sample_size):
        sample_size = 0
    warning = str(overfit_warning).lower()
    if sample_size >= 1000 and walk_forward and "high" not in warning and "extreme" not in warning:
        return "strong"
    if sample_size >= 500 and walk_forward:
        return "moderate"
    if sample_size >= 150:
        return "limited"
    return "weak"


def _impact(value: float, reference: float, higher_is_better: bool = True) -> str:
    if not np.isfinite(value) or not np.isfinite(reference):
        return "unknown"
    delta = value - reference
    if not higher_is_better:
        delta = -delta
    if delta > abs(reference) * 0.05:
        return "positive"
    if delta < -abs(reference) * 0.05:
        return "negative"
    return "neutral"


def _build_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    baseline_sharpe = _metric_from_results("raw_target_research_backtest_results.csv", "model_mode", "baseline", "Sharpe")
    gated_sharpe = _metric_from_results("raw_target_research_backtest_results.csv", "model_mode", "regime_gated_full_quant", "Sharpe")
    baseline_return = _metric_from_results("raw_target_research_backtest_results.csv", "model_mode", "baseline", "realized_return")
    gated_return = _metric_from_results(
        "raw_target_research_backtest_results.csv", "model_mode", "regime_gated_full_quant", "realized_return"
    )
    baseline_dd = _metric_from_results("raw_target_research_backtest_results.csv", "model_mode", "baseline", "max_drawdown")
    gated_dd = _metric_from_results("raw_target_research_backtest_results.csv", "model_mode", "regime_gated_full_quant", "max_drawdown")
    baseline_sample = _metric_from_results("raw_target_research_backtest_results.csv", "model_mode", "baseline", "sample_size")

    rows.append(
        {
            "module": "baseline + old regime gate",
            "category": "active",
            "Sharpe impact": "champion" if gated_sharpe >= baseline_sharpe else "neutral",
            "return impact": "champion" if gated_return >= baseline_return else "neutral",
            "risk impact": "acceptable",
            "evidence strength": _evidence_strength(baseline_sample, True),
            "sample size": baseline_sample,
            "final decision": "current champion model",
            "reason": f"best validated production candidate; Sharpe baseline={baseline_sharpe:.4f}, old_gate={gated_sharpe:.4f}",
        }
    )

    raw_gov = _governance_row("raw_target_research_governance.csv")
    raw_sharpe = _safe_float(raw_gov.get("raw_target_sharpe"))
    raw_dd = _safe_float(raw_gov.get("raw_target_max_drawdown"))
    rows.append(
        {
            "module": "raw_target_alpha",
            "category": "shadow",
            "Sharpe impact": _impact(raw_sharpe, gated_sharpe),
            "return impact": "positive",
            "risk impact": "negative" if raw_dd < gated_dd else "neutral",
            "evidence strength": "moderate",
            "sample size": _metric_from_results(
                "raw_target_research_backtest_results.csv", "model_mode", "raw_target_research", "sample_size"
            ),
            "final decision": str(raw_gov.get("classification", "research only")),
            "reason": str(raw_gov.get("reason", "higher return but drawdown/sharpe not production-ready")),
        }
    )

    rb = _read_csv("raw_target_risk_budgeting_governance.csv")
    if not rb.empty:
        best_shadow = rb.iloc[0]
        rows.append(
            {
                "module": "raw_target_risk_budgeting",
                "category": "shadow",
                "Sharpe impact": _impact(_safe_float(best_shadow.get("Sharpe")), gated_sharpe),
                "return impact": "positive",
                "risk impact": "review",
                "evidence strength": "limited",
                "sample size": np.nan,
                "final decision": str(best_shadow.get("classification", "research only")),
                "reason": str(best_shadow.get("reason", "risk-controlled raw target needs more validation")),
            }
        )

    regime_v2_comp = _read_csv("regime_v2_comparison_vs_old.csv")
    v2_row = (
        regime_v2_comp[regime_v2_comp["system"].astype(str).eq("regime_v2")].iloc[0]
        if not regime_v2_comp.empty and "system" in regime_v2_comp.columns and (regime_v2_comp["system"].astype(str).eq("regime_v2")).any()
        else pd.Series(dtype=object)
    )
    rows.append(
        {
            "module": "regime_engine_v2",
            "category": "shadow",
            "Sharpe impact": "positive" if _safe_float(v2_row.get("weighted_sharpe")) > 3.65 else "unknown",
            "return impact": "neutral",
            "risk impact": "diagnostic",
            "evidence strength": "limited",
            "sample size": np.nan,
            "final decision": "candidate for regime research, not gate replacement",
            "reason": "better regime balance, but regime_gate_v2 backtest did not beat old gate",
        }
    )

    gate_gov = _governance_row("regime_gate_v2_governance.csv")
    rows.append(
        {
            "module": "regime_gate_v2",
            "category": "rejected",
            "Sharpe impact": _impact(_safe_float(gate_gov.get("candidate_sharpe")), gated_sharpe),
            "return impact": "neutral",
            "risk impact": "neutral",
            "evidence strength": "moderate",
            "sample size": np.nan,
            "final decision": str(gate_gov.get("classification", "rejected")),
            "reason": str(gate_gov.get("reason", "does not beat old gate")),
        }
    )

    cal_gov = _governance_row("calibrated_forecast_research_governance.csv")
    rows.append(
        {
            "module": "forecast_calibration_portfolio",
            "category": "rejected",
            "Sharpe impact": "negative",
            "return impact": "mixed",
            "risk impact": "negative",
            "evidence strength": "limited",
            "sample size": _metric_from_results(
                "calibrated_forecast_research_backtest_results.csv", "model_mode", "calibrated_forecast_research", "sample_size"
            ),
            "final decision": str(cal_gov.get("classification", "reject")),
            "reason": str(cal_gov.get("reason", "calibrated portfolio worsened Sharpe/drawdown")),
        }
    )

    final_candidate = _governance_row("final_candidate_governance_report.csv")
    rows.append(
        {
            "module": "meta_filter",
            "category": "rejected",
            "Sharpe impact": "negative",
            "return impact": "negative",
            "risk impact": "mixed",
            "evidence strength": "limited",
            "sample size": np.nan,
            "final decision": str(final_candidate.get("classification", "research only/rejected")),
            "reason": str(final_candidate.get("reason", "reduced return and Sharpe in final candidate test")),
        }
    )

    factor = _read_csv("factor_alpha_model_results.csv")
    best_factor = (
        factor.sort_values("Sharpe", ascending=False).iloc[0]
        if not factor.empty and "Sharpe" in factor.columns
        else pd.Series(dtype=object)
    )
    rows.append(
        {
            "module": "factor_alpha_model",
            "category": "diagnostic",
            "Sharpe impact": _impact(_safe_float(best_factor.get("Sharpe")), gated_sharpe),
            "return impact": "diagnostic",
            "risk impact": "diagnostic",
            "evidence strength": "limited",
            "sample size": _safe_float(best_factor.get("sample_size")),
            "final decision": str(best_factor.get("classification", "diagnostic only")),
            "reason": "expected_daily_return is main alpha; signal/volatility combinations did not improve portfolio",
        }
    )

    kalman = _read_csv("kalman_feature_validation.csv")
    rows.append(
        {
            "module": "kalman_features",
            "category": "diagnostic",
            "Sharpe impact": "unknown",
            "return impact": "unknown",
            "risk impact": "diagnostic",
            "evidence strength": "weak" if kalman.empty else "limited",
            "sample size": len(kalman) if not kalman.empty else 0,
            "final decision": "diagnostic only",
            "reason": "keep as feature diagnostics until stronger walk-forward evidence",
        }
    )

    alpha = _read_csv("alpha_attribution_report.csv")
    if not alpha.empty:
        for _, row in alpha.head(10).iterrows():
            classification = str(row.get("classification", "diagnostic only"))
            category = "diagnostic" if classification != "remove candidate" else "rejected"
            rows.append(
                {
                    "module": f"alpha_factor:{row.get('feature')}",
                    "category": category,
                    "Sharpe impact": "diagnostic",
                    "return impact": "diagnostic",
                    "risk impact": "diagnostic",
                    "evidence strength": "limited",
                    "sample size": np.nan,
                    "final decision": classification,
                    "reason": f"avg_abs_ic={_safe_float(row.get('average_abs_ic')):.4f}; incremental={_safe_float(row.get('incremental_ic_vs_signal')):.4f}",
                }
            )

    return pd.DataFrame(rows)


def _write_summary(registry: pd.DataFrame) -> str:
    active = registry[registry["category"].eq("active")]
    shadow = registry[registry["category"].eq("shadow")]
    diagnostic = registry[registry["category"].eq("diagnostic")]
    rejected = registry[registry["category"].eq("rejected")]
    lines: list[str] = []
    lines.append("===== CURRENT CHAMPION MODEL =====")
    lines.append(active.to_string(index=False) if not active.empty else "No active model found.")
    lines.append("")
    lines.append("===== ACTIVE COMPONENTS =====")
    lines.append("- baseline expected return pipeline")
    lines.append("- old regime gate / regime_gated_full_quant remains champion")
    lines.append("- shrinkage covariance risk estimation")
    lines.append("- EMA timing remains production timing")
    lines.append("")
    lines.append("===== SHADOW COMPONENTS =====")
    lines.append(shadow[["module", "final decision", "reason"]].to_string(index=False) if not shadow.empty else "None")
    lines.append("")
    lines.append("===== DIAGNOSTIC COMPONENTS =====")
    lines.append(diagnostic[["module", "final decision", "reason"]].to_string(index=False) if not diagnostic.empty else "None")
    lines.append("")
    lines.append("===== REJECTED COMPONENTS =====")
    lines.append(rejected[["module", "final decision", "reason"]].to_string(index=False) if not rejected.empty else "None")
    lines.append("")
    lines.append("===== REMAINING RESEARCH ROADMAP =====")
    lines.append("1. Test raw target alpha with stricter drawdown-aware allocation before paper.")
    lines.append("2. Keep regime_engine_v2 diagnostic; do not replace old gate yet.")
    lines.append("3. Audit signal_strength adjustment for possible removal only after true walk-forward shadow validation.")
    lines.append("4. Do not promote meta-filter or calibrated forecast portfolio.")
    lines.append("5. Run anti-overfitting/governance before any paper activation.")
    lines.append("")
    lines.append("===== PROMOTION RULES =====")
    lines.append("- Must beat current champion.")
    lines.append("- Must have sufficient sample size.")
    lines.append("- Must be walk-forward validated.")
    lines.append("- Must improve Sharpe or improve return without unacceptable drawdown.")
    lines.append("- Must pass overfitting checks.")
    lines.append("- No promotion from single metric improvement.")
    text = "\n".join(lines)
    Path(SUMMARY_FILE).write_text(text, encoding="utf-8")
    return text


def run_research_consolidation() -> tuple[pd.DataFrame, str]:
    registry = _build_registry()
    registry.to_csv(REGISTRY_FILE, index=False)
    summary = _write_summary(registry)
    print(summary)
    print(f"\nSaved: {Path(REGISTRY_FILE).resolve()}")
    print(f"Saved: {Path(SUMMARY_FILE).resolve()}")
    return registry, summary


if __name__ == "__main__":
    run_research_consolidation()
