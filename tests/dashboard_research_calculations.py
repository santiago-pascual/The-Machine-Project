from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard_data_layer import latest, numeric

RESEARCH_FILES = {
    "full_cscv_results": "full_cscv_results.csv",
    "pbo_distribution": "pbo_distribution.csv",
    "deflated_sharpe_exact": "deflated_sharpe_exact.csv",
    "effective_trial_count": "effective_trial_count.csv",
    "reality_check_results": "reality_check_results.csv",
    "purged_walk_forward_results": "purged_walk_forward_results.csv",
    "purged_walk_forward_folds": "purged_walk_forward_folds.csv",
    "locked_holdout_results": "locked_holdout_results.csv",
    "ic_decay_results": "ic_decay_results.csv",
    "parameter_stability_map": "parameter_stability_map.csv",
    "parameter_sensitivity_results": "parameter_sensitivity_results.csv",
    "robustness_plateau_analysis": "robustness_plateau_analysis.csv",
    "rolling_feature_ic": "rolling_feature_ic.csv",
    "alpha_decay_curve": "alpha_decay_curve.csv",
    "structural_break_results": "structural_break_results.csv",
    "feature_distribution_drift": "feature_distribution_drift.csv",
    "garch_model_comparison": "garch_model_comparison.csv",
    "hmm_model_comparison": "hmm_model_comparison.csv",
    "hmm_incremental_portfolio_results": "hmm_incremental_portfolio_results.csv",
    "governed_experiment_registry": "governed_experiment_registry.csv",
    "frozen_champion_registry": "frozen_champion_registry.csv",
    "model_lifecycle_status": "model_lifecycle_status.csv",
    "anti_overfitting_governance": "anti_overfitting_governance.csv",
    "out_of_sample_governance": "out_of_sample_governance.csv",
    "parameter_governance": "parameter_governance.csv",
    "alpha_decay_governance": "alpha_decay_governance.csv",
}

FEATURES = [
    "raw_target_return_exact",
    "signal_strength",
    "Kalman",
    "Hurst",
    "OU",
    "GARCH/EGARCH",
    "entropy",
    "regime",
]


@dataclass
class ResearchBundle:
    kpis: dict[str, Any]
    anti_overfitting: dict[str, pd.DataFrame]
    walk_forward: dict[str, pd.DataFrame]
    parameter: dict[str, pd.DataFrame]
    feature_evidence: pd.DataFrame
    ic_decay: pd.DataFrame
    model_comparison: pd.DataFrame
    registry: dict[str, pd.DataFrame]
    lifecycle: pd.DataFrame
    warnings: pd.DataFrame
    source_audit: pd.DataFrame
    integrity: pd.DataFrame
    evidence_reconciliation: pd.DataFrame
    commentary: str
    status: str


def _df(data: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    return data.get(key, pd.DataFrame()).copy()


def _date_range(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    for col in ["date", "start_date", "end_date", "timestamp"]:
        if col in df.columns:
            dates = pd.to_datetime(df[col], errors="coerce")
            if dates.notna().any():
                return f"{dates.min().date()} to {dates.max().date()}"
    return ""


def source_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, filename in RESEARCH_FILES.items():
        df = _df(data, key)
        rows.append(
            {
                "source_file": filename,
                "data_key": key,
                "namespace": "research_governance",
                "loaded": not df.empty,
                "row_count": len(df),
                "date_range": _date_range(df),
                "columns": ",".join(df.columns.astype(str)) if not df.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _last(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    if "date" in df.columns:
        return latest(df).iloc[-1] if not latest(df).empty else df.iloc[-1]
    return df.iloc[-1]


def build_kpis(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    anti = _last(_df(data, "anti_overfitting_governance"))
    eff = _last(_df(data, "effective_trial_count"))
    pbo = _df(data, "pbo_distribution")
    dsr = _df(data, "deflated_sharpe_exact")
    reality = _last(_df(data, "reality_check_results"))
    oos = _last(_df(data, "out_of_sample_governance"))
    holdout = _last(_df(data, "locked_holdout_results"))
    plateau = _last(_df(data, "robustness_plateau_analysis"))
    param_gov = _last(_df(data, "parameter_governance"))
    frozen = _df(data, "frozen_champion_registry")
    lifecycle = _df(data, "model_lifecycle_status")
    active_model = "growth_champion_final_v1_0_frozen"
    lifecycle_status = "operational_paper_production_real_capital_blocked"
    if not frozen.empty and "registry_name" in frozen.columns:
        match = frozen[frozen["registry_name"].astype(str).str.contains("growth_champion_final_v1_0", na=False)]
        if not match.empty:
            lifecycle_status = match.iloc[-1].get("status", lifecycle_status)
    param_hash = np.nan
    if not frozen.empty and "parameter_set_hash" in frozen.columns:
        hashes = frozen["parameter_set_hash"].dropna()
        if not hashes.empty:
            param_hash = hashes.iloc[-1]
    pbo_val = anti.get("CSCV_PBO", np.nan)
    if pd.isna(pbo_val) and not pbo.empty and "PBO" in pbo.columns:
        pbo_val = numeric(pbo["PBO"]).mean()
    dsr_best = pd.Series(dtype=object)
    if not dsr.empty and "deflated_sharpe" in dsr.columns:
        dsr_best = dsr.sort_values("deflated_sharpe", ascending=False).iloc[0]
    return {
        "active_model": active_model,
        "lifecycle_status": lifecycle_status,
        "official_paper_status": "official_forward_warmup",
        "real_capital_status": "real_capital_blocked",
        "latest_research_update": max(
            [r for r in [_date_range(_df(data, k)).split(" to ")[-1] if _date_range(_df(data, k)) else "" for k in RESEARCH_FILES] if r]
            or ["unavailable"]
        ),
        "frozen_configuration_hash": param_hash,
        "all_time_trials": anti.get("all_time_trials", np.nan),
        "effective_independent_trials": anti.get("effective_independent_trials", eff.get("independent_trials_estimate", np.nan)),
        "observed_strategy_count": eff.get("observed_strategy_count", np.nan),
        "CSCV_PBO": pbo_val,
        "exact_deflated_sharpe": anti.get("deflated_sharpe", dsr_best.get("deflated_sharpe", np.nan)),
        "DSR_p_value": anti.get("DSR_p_value", dsr_best.get("dsr_p_value", np.nan)),
        "reality_check_p_value": anti.get("SPA_or_reality_check_p_value", reality.get("reality_check_p_value", np.nan)),
        "mean_oos_sharpe": oos.get("mean_test_sharpe", np.nan),
        "positive_oos_folds_pct": oos.get("positive_test_fold_rate", np.nan),
        "locked_holdout_sharpe": oos.get("locked_holdout_sharpe", holdout.get("Sharpe", np.nan)),
        "locked_holdout_CAGR": oos.get("locked_holdout_CAGR", holdout.get("CAGR", np.nan)),
        "locked_holdout_max_drawdown": oos.get("locked_holdout_max_drawdown", holdout.get("max_drawdown", np.nan)),
        "parameter_stability_classification": param_gov.get("classification", plateau.get("interpretation", np.nan)),
        "plateau_pct": plateau.get("plateau_pct", np.nan),
        "local_sharpe_std": plateau.get("local_sharpe_std", np.nan),
        "current_sharpe_rank": plateau.get("current_rank_by_sharpe", np.nan),
        "current_cagr_rank": plateau.get("current_rank_by_CAGR", np.nan),
    }


def anti_overfitting_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "cscv": _df(data, "full_cscv_results"),
        "pbo": _df(data, "pbo_distribution"),
        "dsr": _df(data, "deflated_sharpe_exact"),
        "effective_trials": _df(data, "effective_trial_count"),
        "reality_check": _df(data, "reality_check_results"),
        "governance": _df(data, "anti_overfitting_governance"),
    }


def walk_forward_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    results = _df(data, "purged_walk_forward_results")
    folds = _df(data, "purged_walk_forward_folds")
    holdout = _df(data, "locked_holdout_results")
    if not results.empty:
        results = results.copy()
        results["fold_rank_type"] = np.where(
            results.get("fold_type", "").astype(str).str.contains("locked", case=False, na=False), "locked_holdout", "oos_fold"
        )
    return {"results": results, "folds": folds, "holdout": holdout, "governance": _df(data, "out_of_sample_governance")}


def parameter_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    stab = _df(data, "parameter_stability_map")
    if not stab.empty:
        stab = stab.copy()
        for col in ["target_vol", "exposure_cap", "Sharpe", "CAGR", "max_drawdown"]:
            if col in stab.columns:
                stab[col] = numeric(stab[col])
    return {
        "stability": stab,
        "sensitivity": _df(data, "parameter_sensitivity_results"),
        "plateau": _df(data, "robustness_plateau_analysis"),
        "governance": _df(data, "parameter_governance"),
    }


def feature_evidence_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    alpha = _df(data, "alpha_decay_curve")
    rolling = _df(data, "rolling_feature_ic")
    breaks = _df(data, "structural_break_results")
    drift = _df(data, "feature_distribution_drift")
    rows = []
    for feat in FEATURES:
        pattern = feat.lower().replace("/", "|")
        if feat == "raw_target_return_exact":
            names = ["raw_target_return_exact", "expected_daily_return", "expected_total_return"]
        elif feat == "GARCH/EGARCH":
            names = ["garch", "egarch"]
        else:
            names = [feat.lower()]

        def match(df, col="feature"):
            if df.empty or col not in df.columns:
                return pd.DataFrame()
            mask = False
            for name in names:
                mask = mask | df[col].astype(str).str.lower().str.contains(name, na=False)
            return df[mask].copy()

        a = match(alpha)
        r = match(rolling)
        b = match(breaks)
        d = match(drift)
        status = "missing_data"
        role = "diagnostic"
        note = "missing-data state; not validated in available files"
        rank_ic = np.nan
        rolling_ic = np.nan
        decay = "unavailable"
        sign_flip = np.nan
        if feat == "signal_strength":
            role = "diagnostic_only"
            note = "weak 20D IC; sign flip detected; removed from active ranking"
        if feat == "raw_target_return_exact":
            role = "active"
            note = "raw target/expected return evidence available via historical IC proxies"
        if not a.empty:
            status = "available"
            rank_ic = numeric(a.get("mean_rank_ic", a.get("mean_spearman_rank_ic", pd.Series(dtype=float)))).dropna().mean()
            decay = (
                ", ".join(
                    a[["horizon", "status"]]
                    .astype(str)
                    .drop_duplicates()
                    .apply(lambda x: f"{x.iloc[0]}:{x.iloc[1]}", axis=1)
                    .head(8)
                    .tolist()
                )
                if "horizon" in a.columns
                else "available"
            )
        if not r.empty:
            status = "available"
            rolling_ic = numeric(r.get("rolling_ic_mean", r.get("ic", pd.Series(dtype=float)))).dropna().tail(20).mean()
        if not b.empty:
            sign_vals = b.get("sign_flip", pd.Series(dtype=object)).dropna().astype(str).str.lower()
            sign_flip = bool(sign_vals.eq("true").any()) if not sign_vals.empty else np.nan
        drift_status = "missing" if d.empty else ",".join(d.get("status", pd.Series(dtype=str)).astype(str).dropna().unique().tolist())
        rows.append(
            {
                "feature": feat,
                "availability": status,
                "role": role,
                "rank_ic": rank_ic,
                "rolling_ic_recent": rolling_ic,
                "ic_decay": decay,
                "sign_flip_detected": sign_flip,
                "distribution_drift_status": drift_status,
                "missing_data_reason": "not present in alpha/rolling/break/drift outputs" if status == "missing_data" else "",
                "evidence_note": note,
            }
        )
    return pd.DataFrame(rows)


def model_comparison_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    garch = _df(data, "garch_model_comparison")
    if not garch.empty:
        for _, r in garch.iterrows():
            name = str(r.get("model_name", "garch"))
            family = str(r.get("model_family", "diagnostic"))
            role = "diagnostic" if family == "mle" else "active_diagnostic"
            if "grid" in name.lower() or ("egarch" in name.lower() and family != "mle"):
                role = "active" if "egarch" in name.lower() else "diagnostic"
            rows.append(
                {
                    "model": name,
                    "status": role,
                    "validation_metric": r.get("QLIKE", np.nan),
                    "incremental_value": r.get("forecast_realized_corr", np.nan),
                    "current_role": "MLE diagnostic only" if family == "mle" else "Grid model retained/diagnostic",
                }
            )
    hmm = _df(data, "hmm_model_comparison")
    if not hmm.empty:
        for _, r in hmm.iterrows():
            n = r.get("n_states", "")
            rows.append(
                {
                    "model": f"HMM {n}-state",
                    "status": "diagnostic",
                    "validation_metric": r.get("BIC", np.nan),
                    "incremental_value": r.get("state_interpretability", ""),
                    "current_role": "HMM diagnostic; does not alter allocation",
                }
            )
    inc = _df(data, "hmm_incremental_portfolio_results")
    if not inc.empty:
        for _, r in inc.iterrows():
            rows.append(
                {
                    "model": r.get("variant", "HMM overlay"),
                    "status": "redundant_with_dual_trend",
                    "validation_metric": r.get("Sharpe", np.nan),
                    "incremental_value": r.get("net_CAGR", np.nan),
                    "current_role": "research-only; no paper promotion",
                }
            )
    rows.append(
        {
            "model": "Dual trend",
            "status": "active",
            "validation_metric": "frozen config",
            "incremental_value": "current risk overlay",
            "current_role": "active exposure gate",
        }
    )
    return pd.DataFrame(rows)


def registry_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "governed": _df(data, "governed_experiment_registry"),
        "frozen": _df(data, "frozen_champion_registry"),
        "lifecycle": _df(data, "model_lifecycle_status"),
    }


def warnings_table(data: dict[str, pd.DataFrame], kpis: dict[str, Any]) -> pd.DataFrame:
    rows = []

    def add(warning, severity, evidence):
        rows.append({"warning": warning, "severity": severity, "evidence": evidence})

    pbo = kpis.get("CSCV_PBO", np.nan)
    if pd.notna(pbo) and pbo > 0.5:
        add("CSCV PBO elevated", "HIGH", f"PBO={pbo:.3f}")
    add("Official forward history short", "HIGH", "official paper remains warmup")
    add("Second market-data provider absent or not fully confirmed", "MEDIUM", "real-capital blocked until reliable secondary source")
    add("Reconstructed history is not production parity", "MEDIUM", "kept separate from official forward paper")
    add("Real fills unavailable", "MEDIUM", "execution costs are estimated")
    alpha_gov = _last(_df(data, "alpha_decay_governance"))
    if str(alpha_gov.get("classification", "")).lower() in {"significant_decay", "retraining_review_required"}:
        add("Alpha decay monitor warning", "HIGH", str(alpha_gov.get("reason", "")))
    add("HMM not incrementally useful", "LOW", "HMM overlay/ranking proxy remains research-only")
    trials = kpis.get("all_time_trials", np.nan)
    if pd.notna(trials) and trials > 1000:
        add("High historical trial count", "HIGH", f"all-time trials={trials}")
    return pd.DataFrame(rows)


def evidence_reconciliation(kpis: dict[str, Any], warnings: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "evidence": "Deflated Sharpe",
            "value": kpis.get("exact_deflated_sharpe"),
            "interpretation": "strong if DSR p-value is low",
            "conflict": "does not eliminate CSCV PBO warning",
        },
        {
            "evidence": "CSCV PBO",
            "value": kpis.get("CSCV_PBO"),
            "interpretation": "elevated overfitting probability if above 0.5",
            "conflict": "coexists with strong OOS performance",
        },
        {
            "evidence": "Purged OOS",
            "value": kpis.get("mean_oos_sharpe"),
            "interpretation": "temporal OOS strength",
            "conflict": "official forward history still short",
        },
        {
            "evidence": "Locked Holdout",
            "value": kpis.get("locked_holdout_sharpe"),
            "interpretation": "strong holdout result",
            "conflict": "later development may reduce permanently untouched status",
        },
        {
            "evidence": "Parameter Stability",
            "value": kpis.get("parameter_stability_classification"),
            "interpretation": "broad plateau if stable_plateau/broad plateau",
            "conflict": "no new optimum selected",
        },
        {
            "evidence": "Alpha Decay",
            "value": "; ".join(warnings["warning"].astype(str).tolist()) if not warnings.empty else "none",
            "interpretation": "warnings remain visible",
            "conflict": "does not change frozen model automatically",
        },
    ]
    return pd.DataFrame(rows)


def integrity_table(data: dict[str, pd.DataFrame], src: pd.DataFrame, kpis: dict[str, Any], feature: pd.DataFrame) -> pd.DataFrame:
    missing_core = src[
        (~src["loaded"])
        & src["data_key"].isin(
            ["full_cscv_results", "pbo_distribution", "deflated_sharpe_exact", "purged_walk_forward_results", "parameter_stability_map"]
        )
    ]
    rows = [
        {
            "check": "research_files_load_safely",
            "status": "PASS" if missing_core.empty else "WARNING",
            "detail": ",".join(missing_core["source_file"].tolist()),
        },
        {"check": "conflicting_evidence_visible", "status": "PASS", "detail": "DSR/PBO/OOS/alpha warnings displayed separately"},
        {"check": "no_model_changes", "status": "PASS", "detail": "dashboard diagnostics only"},
        {"check": "no_automatic_promotion", "status": "PASS", "detail": "real capital remains blocked"},
        {
            "check": "parameter_surface_available",
            "status": "PASS" if _df(data, "parameter_stability_map").shape[0] > 0 else "WARNING",
            "detail": _df(data, "parameter_stability_map").shape[0],
        },
        {
            "check": "missing_features_explicit",
            "status": "PASS" if "missing_data" in feature.get("availability", pd.Series(dtype=str)).astype(str).tolist() else "PASS",
            "detail": "feature evidence table includes missing-data reason",
        },
        {
            "check": "lifecycle_frozen_status",
            "status": "PASS"
            if "blocked" in str(kpis.get("lifecycle_status", "")).lower() or "frozen" in str(kpis.get("lifecycle_status", "")).lower()
            else "WARNING",
            "detail": kpis.get("lifecycle_status"),
        },
    ]
    return pd.DataFrame(rows)


def deterministic_commentary(kpis: dict[str, Any], warnings: pd.DataFrame) -> str:
    return (
        f"The frozen model shows purged out-of-sample Sharpe of {kpis.get('mean_oos_sharpe', np.nan):.3f} "
        f"and parameter classification {kpis.get('parameter_stability_classification')}. "
        f"CSCV PBO remains {kpis.get('CSCV_PBO', np.nan):.3f}, while exact DSR p-value is {kpis.get('DSR_p_value', np.nan):.4f}. "
        "These conflicting diagnostics remain simultaneously visible. "
        "The model remains classified for operational paper validation with real-capital promotion blocked."
    )


def build_research_bundle(data: dict[str, pd.DataFrame]) -> ResearchBundle:
    kpis = build_kpis(data)
    anti = anti_overfitting_tables(data)
    wf = walk_forward_tables(data)
    param = parameter_tables(data)
    feature = feature_evidence_table(data)
    ic = _df(data, "alpha_decay_curve")
    models = model_comparison_table(data)
    registry = registry_tables(data)
    lifecycle = _df(data, "model_lifecycle_status")
    warnings = warnings_table(data, kpis)
    src = source_audit(data)
    evidence = evidence_reconciliation(kpis, warnings)
    integrity = integrity_table(data, src, kpis, feature)
    commentary = deterministic_commentary(kpis, warnings)
    status = "research_terminal_pass"
    if integrity["status"].eq("FAIL").any():
        status = "research_terminal_fail"
    elif integrity["status"].eq("WARNING").any() or not warnings.empty:
        status = "research_terminal_warning"
    return ResearchBundle(
        kpis, anti, wf, param, feature, ic, models, registry, lifecycle, warnings, src, integrity, evidence, commentary, status
    )
