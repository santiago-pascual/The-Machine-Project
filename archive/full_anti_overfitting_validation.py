from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252
OUTPUTS = {
    "cscv": "full_cscv_results.csv",
    "pbo": "pbo_distribution.csv",
    "dsr": "deflated_sharpe_exact.csv",
    "effective_trials": "effective_trial_count.csv",
    "reality": "reality_check_results.csv",
    "registry": "governed_experiment_registry.csv",
    "governance": "anti_overfitting_governance.csv",
    "report": "anti_overfitting_report.txt",
}


@dataclass(frozen=True)
class CandidateSource:
    path: str
    date_col: str = "date"
    return_col: str | None = None
    strategy_cols: tuple[str, ...] = ()
    candidate_prefix: str = ""


SOURCES = [
    CandidateSource("growth_final_selection_daily_returns.csv", return_col="candidate_return", strategy_cols=("candidate", "window_start"), candidate_prefix="final_selection"),
    CandidateSource("growth_final_cost_slippage_daily_returns.csv", return_col="net_return", strategy_cols=("candidate", "window_start", "cost_scenario"), candidate_prefix="final_cost"),
    CandidateSource("growth_crisis_overlay_daily_returns.csv", return_col="overlay_return", strategy_cols=("overlay", "window_start"), candidate_prefix="crisis_overlay"),
    CandidateSource("growth_head_to_head_daily_returns.csv", return_col="return", strategy_cols=("candidate",), candidate_prefix="head_to_head"),
    CandidateSource("growth_volatility_targeting_daily_returns.csv", return_col="return", strategy_cols=("candidate",), candidate_prefix="vol_target"),
    CandidateSource("exit_rule_walk_forward_daily_returns.csv", return_col="return", strategy_cols=("variant",), candidate_prefix="exit_rule"),
    CandidateSource("exit_rule_drawdown_guard_daily_returns.csv", return_col="return", strategy_cols=("variant",), candidate_prefix="drawdown_guard"),
    CandidateSource("raw_target_research_backtest_daily_returns.csv", return_col="portfolio_return", strategy_cols=("model_mode", "variant"), candidate_prefix="raw_target"),
    CandidateSource("raw_target_2020_daily_returns.csv", return_col="return", strategy_cols=("model", "variant"), candidate_prefix="raw_2020"),
    CandidateSource("raw_target_risk_controlled_daily_returns.csv", return_col="portfolio_return", strategy_cols=("variant",), candidate_prefix="raw_risk_control"),
    CandidateSource("raw_target_risk_budgeting_daily_returns.csv", return_col="portfolio_return", strategy_cols=("variant",), candidate_prefix="raw_risk_budget"),
    CandidateSource("production_parity_growth_daily_returns.csv", return_col="return", strategy_cols=("model",), candidate_prefix="production_parity"),
    CandidateSource("reconstructed_growth_long_horizon_daily_returns.csv", return_col="return", strategy_cols=("window_start",), candidate_prefix="reconstructed"),
]


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inv_norm_cdf(p: float) -> float:
    # Peter J. Acklam's rational approximation, sufficient for diagnostics.
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def annualized_sharpe(series: pd.Series) -> float:
    r = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 3:
        return np.nan
    std = float(r.std(ddof=1))
    if std <= 0:
        return np.nan
    return float(r.mean() / std * math.sqrt(TRADING_DAYS))


def make_strategy_name(prefix: str, row: pd.Series, cols: Iterable[str], fallback: str) -> str:
    parts = [prefix]
    for col in cols:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    if len(parts) == 1:
        parts.append(fallback)
    return "__".join(p.replace(" ", "_").replace("/", "_") for p in parts)


def discover_strategy_return_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    registry_rows: list[dict[str, object]] = []
    for source in SOURCES:
        df = read_csv(source.path)
        if df.empty or source.date_col not in df.columns:
            registry_rows.append({"source_file": source.path, "loaded": False, "strategies": 0, "rows": 0, "reason": "missing_or_empty"})
            continue
        date = pd.to_datetime(df[source.date_col], errors="coerce")
        df = df.assign(_date=date).dropna(subset=["_date"])
        return_col = source.return_col
        if return_col not in df.columns:
            numeric_return_cols = [c for c in df.columns if c.lower() in {"return", "daily_return", "candidate_return", "net_return", "overlay_return"}]
            return_col = numeric_return_cols[0] if numeric_return_cols else None
        if not return_col or return_col not in df.columns:
            registry_rows.append({"source_file": source.path, "loaded": False, "strategies": 0, "rows": len(df), "reason": "return_column_missing"})
            continue
        tmp = df.copy()
        tmp["_strategy"] = [make_strategy_name(source.candidate_prefix, row, source.strategy_cols, return_col) for _, row in tmp.iterrows()]
        tmp["_return"] = pd.to_numeric(tmp[return_col], errors="coerce")
        pivot = tmp.pivot_table(index="_date", columns="_strategy", values="_return", aggfunc="mean")
        pivot = pivot.sort_index()
        pieces.append(pivot)
        registry_rows.append(
            {
                "source_file": source.path,
                "loaded": True,
                "strategies": int(pivot.shape[1]),
                "rows": int(len(tmp)),
                "date_min": str(pivot.index.min().date()) if not pivot.empty else "",
                "date_max": str(pivot.index.max().date()) if not pivot.empty else "",
                "return_column": return_col,
                "reason": "loaded",
            }
        )
    if not pieces:
        return pd.DataFrame(), pd.DataFrame(registry_rows)
    matrix = pd.concat(pieces, axis=1, sort=True)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    min_obs = max(60, min(250, int(len(matrix) * 0.20)))
    keep = [c for c in matrix.columns if matrix[c].notna().sum() >= min_obs]
    matrix = matrix[keep].sort_index()
    return matrix, pd.DataFrame(registry_rows)


def effective_trial_count(return_matrix: pd.DataFrame) -> pd.DataFrame:
    if return_matrix.empty or return_matrix.shape[1] < 2:
        return pd.DataFrame(
            [{
                "observed_strategy_count": int(return_matrix.shape[1]),
                "effective_trials_participation_ratio": float(return_matrix.shape[1]),
                "effective_trials_entropy_rank": float(return_matrix.shape[1]),
                "independent_trials_estimate": int(max(1, return_matrix.shape[1])),
                "method": "insufficient_correlation_matrix",
            }]
        )
    aligned = return_matrix.copy().fillna(0.0)
    corr = aligned.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eigvals = np.linalg.eigvalsh(corr.values)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    participation = (total * total / np.square(eigvals).sum()) if np.square(eigvals).sum() > 0 else 1.0
    probs = eigvals / total if total > 0 else np.ones_like(eigvals) / len(eigvals)
    entropy_rank = math.exp(float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()))
    independent = int(max(1, min(return_matrix.shape[1], round(entropy_rank))))
    return pd.DataFrame(
        [{
            "observed_strategy_count": int(return_matrix.shape[1]),
            "effective_trials_participation_ratio": float(participation),
            "effective_trials_entropy_rank": float(entropy_rank),
            "independent_trials_estimate": independent,
            "method": "correlation_eigenvalue_effective_rank",
        }]
    )


def cscv_for_s(return_matrix: pd.DataFrame, s_blocks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = return_matrix.dropna(axis=1, how="all").sort_index()
    if matrix.shape[0] < s_blocks * 6 or matrix.shape[1] < 2 or s_blocks % 2 != 0:
        return pd.DataFrame(), pd.DataFrame()
    block_ids = np.array_split(np.arange(len(matrix)), s_blocks)
    combos = list(itertools.combinations(range(s_blocks), s_blocks // 2))
    rows = []
    pbo_rows = []
    for fold_id, test_blocks in enumerate(combos, start=1):
        test_idx = np.concatenate([block_ids[i] for i in test_blocks])
        train_idx = np.concatenate([block_ids[i] for i in range(s_blocks) if i not in test_blocks])
        train = matrix.iloc[train_idx]
        test = matrix.iloc[test_idx]
        train_scores = train.apply(annualized_sharpe)
        test_scores = test.apply(annualized_sharpe)
        valid = train_scores.dropna().index.intersection(test_scores.dropna().index)
        if len(valid) < 2:
            continue
        train_scores = train_scores.loc[valid]
        test_scores = test_scores.loc[valid]
        is_winner = str(train_scores.idxmax())
        oos_rank = int(test_scores.rank(ascending=False, method="min").loc[is_winner])
        n = len(valid)
        relative_rank = (n - oos_rank) / max(1, n - 1)
        relative_rank = float(np.clip(relative_rank, 1e-6, 1.0 - 1e-6))
        lambda_logit = math.log(relative_rank / (1.0 - relative_rank))
        rows.append(
            {
                "S": s_blocks,
                "fold_id": fold_id,
                "test_blocks": ",".join(map(str, test_blocks)),
                "strategy_count": n,
                "is_winner": is_winner,
                "is_sharpe": float(train_scores.loc[is_winner]),
                "oos_sharpe": float(test_scores.loc[is_winner]),
                "oos_rank": oos_rank,
                "relative_oos_rank": relative_rank,
                "lambda_logit": lambda_logit,
                "overfit_fold": bool(lambda_logit < 0.0),
            }
        )
    fold_df = pd.DataFrame(rows)
    if not fold_df.empty:
        pbo_rows.append(
            {
                "S": s_blocks,
                "folds": int(len(fold_df)),
                "PBO": float(fold_df["overfit_fold"].mean()),
                "median_lambda": float(fold_df["lambda_logit"].median()),
                "mean_is_sharpe": float(fold_df["is_sharpe"].mean()),
                "mean_oos_sharpe": float(fold_df["oos_sharpe"].mean()),
                "method": "CSCV_symmetric_train_test_combinations",
            }
        )
    return fold_df, pd.DataFrame(pbo_rows)


def run_cscv(return_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_folds = []
    all_pbo = []
    for s in [8, 10, 12]:
        folds, pbo = cscv_for_s(return_matrix, s)
        if not folds.empty:
            all_folds.append(folds)
        if not pbo.empty:
            all_pbo.append(pbo)
    return (
        pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame(),
        pd.concat(all_pbo, ignore_index=True) if all_pbo else pd.DataFrame(),
    )


def expected_max_sharpe(independent_trials: int, n_obs: int) -> float:
    trials = max(1, int(independent_trials))
    n_obs = max(3, int(n_obs))
    gamma = 0.5772156649015329
    if trials <= 1:
        return 0.0
    z1 = inv_norm_cdf(1.0 - 1.0 / trials)
    z2 = inv_norm_cdf(1.0 - 1.0 / (trials * math.e))
    return float(((1.0 - gamma) * z1 + gamma * z2) / math.sqrt(n_obs - 1.0) * math.sqrt(TRADING_DAYS))


def exact_dsr(return_matrix: pd.DataFrame, effective_trials: int) -> pd.DataFrame:
    rows = []
    for strategy in return_matrix.columns:
        r = pd.to_numeric(return_matrix[strategy], errors="coerce").dropna()
        if len(r) < 10:
            continue
        sr = annualized_sharpe(r)
        daily_sr = sr / math.sqrt(TRADING_DAYS) if np.isfinite(sr) else np.nan
        skew = float(r.skew()) if len(r) > 2 else 0.0
        kurt = float(r.kurtosis() + 3.0) if len(r) > 3 else 3.0
        sr_star = expected_max_sharpe(effective_trials, len(r))
        daily_sr_star = sr_star / math.sqrt(TRADING_DAYS)
        denom = math.sqrt(max(1e-12, 1.0 - skew * daily_sr + ((kurt - 1.0) / 4.0) * daily_sr * daily_sr))
        z = (daily_sr - daily_sr_star) * math.sqrt(len(r) - 1.0) / denom
        p_value = 1.0 - norm_cdf(z)
        rows.append(
            {
                "strategy": strategy,
                "sample_size": int(len(r)),
                "observed_sharpe": float(sr),
                "skewness": skew,
                "kurtosis": kurt,
                "effective_independent_trials_used": int(effective_trials),
                "expected_max_sharpe": float(sr_star),
                "deflated_sharpe": float(sr - sr_star),
                "dsr_z_score": float(z),
                "dsr_p_value": float(np.clip(p_value, 0.0, 1.0)),
                "survives_5pct": bool(p_value < 0.05 and sr > sr_star),
            }
        )
    return pd.DataFrame(rows).sort_values(["survives_5pct", "deflated_sharpe"], ascending=[False, False])


def reality_check(return_matrix: pd.DataFrame, benchmark_col: str | None = None, bootstrap_samples: int = 1000) -> pd.DataFrame:
    matrix = return_matrix.dropna(axis=1, how="all")
    if matrix.shape[1] < 2 or len(matrix) < 40:
        return pd.DataFrame([{"method": "stationary_bootstrap_reality_check", "feasible": False, "reason": "insufficient_strategy_return_matrix"}])
    if benchmark_col and benchmark_col in matrix.columns:
        benchmark = matrix[benchmark_col]
    else:
        benchmark = pd.Series(0.0, index=matrix.index)
    excess = matrix.sub(benchmark, axis=0).fillna(0.0)
    means = excess.mean()
    best_strategy = str(means.idxmax())
    observed = float(means.max())
    centered = excess.sub(means, axis=1).values
    rng = np.random.default_rng(42)
    max_stats = []
    n = len(excess)
    for _ in range(bootstrap_samples):
        sample_idx = rng.integers(0, n, n)
        sample = centered[sample_idx, :]
        max_stats.append(float(sample.mean(axis=0).max()))
    p_value = float(np.mean(np.array(max_stats) >= observed))
    return pd.DataFrame(
        [{
            "method": "White_reality_check_centered_iid_bootstrap",
            "feasible": True,
            "best_strategy_by_mean_excess": best_strategy,
            "observed_mean_excess_return_per_period": observed,
            "bootstrap_samples": bootstrap_samples,
            "reality_check_p_value": p_value,
            "note": "IID bootstrap approximation; conservative governance diagnostic, not a promotion engine.",
        }]
    )


def build_registries(return_registry: pd.DataFrame, effective_df: pd.DataFrame) -> pd.DataFrame:
    exp = read_csv("experiment_registry.csv")
    trial_log = read_csv("strategy_trial_log.csv")
    total_trials = int(pd.to_numeric(trial_log.get("number_of_trials", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not trial_log.empty else 0
    rows = []
    rows.append(
        {
            "registry_name": "all_time_research_registry",
            "trial_count": total_trials,
            "source": "strategy_trial_log.csv",
            "mix_with_governed": False,
            "status": "historical_exploratory_trials_only",
            "notes": "Preserved for all-time overfitting warning; not mixed with frozen champion evaluation.",
        }
    )
    governed_count = int(exp["counted_as_new_independent_trial"].astype(str).str.lower().eq("true").sum()) if not exp.empty and "counted_as_new_independent_trial" in exp.columns else int(len(exp))
    rows.append(
        {
            "registry_name": "governed_research_registry",
            "trial_count": governed_count,
            "source": "experiment_registry.csv",
            "mix_with_governed": True,
            "status": "pre_registered_governed_experiments",
            "notes": "Governed trials are counted separately from old unrestricted research.",
        }
    )
    config = read_csv("growth_candidate_paper_config.json")
    config_hash = ""
    if Path("growth_candidate_paper_config.json").exists():
        raw = Path("growth_candidate_paper_config.json").read_bytes()
        config_hash = hashlib.sha256(raw).hexdigest()[:16]
    rows.append(
        {
            "registry_name": "frozen_champion_registry",
            "trial_count": 1,
            "source": "growth_candidate_paper_config.json",
            "mix_with_governed": False,
            "status": "frozen_paper_candidate",
            "experiment_id": "growth_champion_final_frozen",
            "hypothesis": "Dual trend filtered growth champion should improve drawdown-adjusted growth without changing production.",
            "pre_registered_metric": "Sharpe, Calmar, max_drawdown, CAGR versus SPY/QQQ",
            "start_date": "frozen_after_growth_final_selection",
            "allowed_parameters": "raw_target_return_exact; soft_exit_rule; vol_target_22pct; exposure_cap_60; dual_trend_filter",
            "pass_fail_rule": "No post-hoc metric switching; production promotion blocked pending paper evidence.",
            "parameter_set_hash": config_hash,
            "notes": "Frozen logic only; new variants must register separately.",
        }
    )
    reg = pd.DataFrame(rows)
    reg.to_csv(OUTPUTS["registry"], index=False)
    for registry_name, output_name in [
        ("all_time_research_registry", "all_time_research_registry.csv"),
        ("governed_research_registry", "governed_research_registry.csv"),
        ("frozen_champion_registry", "frozen_champion_registry.csv"),
    ]:
        reg.loc[reg["registry_name"].eq(registry_name)].to_csv(output_name, index=False)
    return reg


def classify_governance(pbo_df: pd.DataFrame, dsr_df: pd.DataFrame, reality_df: pd.DataFrame, trial_log: pd.DataFrame, effective_df: pd.DataFrame) -> pd.DataFrame:
    all_time_trials = int(pd.to_numeric(trial_log.get("number_of_trials", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not trial_log.empty else 0
    effective_trials = int(effective_df["independent_trials_estimate"].iloc[0]) if not effective_df.empty else 1
    cscv_pbo = float(pbo_df["PBO"].mean()) if not pbo_df.empty else np.nan
    champion_mask = dsr_df["strategy"].astype(str).str.contains("growth_champion_v3|growth_champion_final", case=False, na=False) if not dsr_df.empty else pd.Series(dtype=bool)
    champion_dsr = dsr_df.loc[champion_mask].head(1) if not dsr_df.empty and champion_mask.any() else pd.DataFrame()
    best_dsr = dsr_df.head(1)
    chosen = champion_dsr if not champion_dsr.empty else best_dsr
    dsr_p = float(chosen["dsr_p_value"].iloc[0]) if not chosen.empty else np.nan
    dsr_deflated = float(chosen["deflated_sharpe"].iloc[0]) if not chosen.empty else np.nan
    rc_p = float(reality_df["reality_check_p_value"].iloc[0]) if not reality_df.empty and "reality_check_p_value" in reality_df.columns else np.nan
    survives = bool((not np.isnan(cscv_pbo) and cscv_pbo < 0.35) and (not np.isnan(dsr_p) and dsr_p < 0.10) and (np.isnan(rc_p) or rc_p < 0.10))
    if survives:
        classification = "frozen_paper_candidate"
    elif not np.isnan(cscv_pbo) and cscv_pbo < 0.50 and not np.isnan(dsr_p) and dsr_p < 0.20:
        classification = "robust_candidate"
    elif all_time_trials > 1000 or np.isnan(cscv_pbo):
        classification = "research_only"
    else:
        classification = "statistically_unreliable"
    reasons = []
    if all_time_trials > 1000:
        reasons.append(f"all_time_trials_high={all_time_trials}")
    if not np.isnan(cscv_pbo):
        reasons.append(f"CSCV_PBO={cscv_pbo:.3f}")
    else:
        reasons.append("CSCV_unavailable")
    if not np.isnan(dsr_p):
        reasons.append(f"DSR_p={dsr_p:.3f}")
    if not np.isnan(rc_p):
        reasons.append(f"reality_check_p={rc_p:.3f}")
    return pd.DataFrame(
        [{
            "all_time_trials": all_time_trials,
            "effective_independent_trials": effective_trials,
            "CSCV_PBO": cscv_pbo,
            "deflated_sharpe": dsr_deflated,
            "DSR_p_value": dsr_p,
            "SPA_or_reality_check_p_value": rc_p,
            "growth_champion_final_survives_multiple_testing_correction": survives,
            "classification": classification,
            "production_changed": False,
            "paper_changed": False,
            "parameter_tuning": False,
            "reason": "; ".join(reasons),
        }]
    )


def write_report(registry_df: pd.DataFrame, effective_df: pd.DataFrame, pbo_df: pd.DataFrame, dsr_df: pd.DataFrame, reality_df: pd.DataFrame, governance_df: pd.DataFrame, return_registry: pd.DataFrame) -> None:
    gov = governance_df.iloc[0].to_dict() if not governance_df.empty else {}
    lines = [
        "===== FULL ANTI-OVERFITTING VALIDATION FRAMEWORK =====",
        "",
        "Scope: research/governance only. Production, paper trading, optimizer, ranking logic and model parameters were not modified.",
        "",
        "===== REGISTRY SEPARATION =====",
    ]
    for _, row in registry_df.iterrows():
        lines.append(f"{row.get('registry_name')}: trials={row.get('trial_count')} status={row.get('status')} source={row.get('source')}")
    lines.extend(["", "===== EFFECTIVE TRIAL COUNT ====="])
    if not effective_df.empty:
        row = effective_df.iloc[0]
        lines.append(f"Observable strategy return series: {row['observed_strategy_count']}")
        lines.append(f"Effective independent trials: {row['independent_trials_estimate']}")
        lines.append(f"Method: {row['method']}")
    lines.extend(["", "===== CSCV / PBO ====="])
    if pbo_df.empty:
        lines.append("CSCV could not be computed with the available aligned return matrix.")
    else:
        for _, row in pbo_df.iterrows():
            lines.append(f"S={int(row['S'])}: folds={int(row['folds'])}, PBO={row['PBO']:.3f}, median_lambda={row['median_lambda']:.3f}")
    lines.extend(["", "===== DEFLATED SHARPE ====="])
    if dsr_df.empty:
        lines.append("DSR unavailable: no sufficient return series.")
    else:
        top = dsr_df.head(5)
        for _, row in top.iterrows():
            lines.append(f"{row['strategy']}: Sharpe={row['observed_sharpe']:.3f}, DSR={row['deflated_sharpe']:.3f}, p={row['dsr_p_value']:.3f}")
    lines.extend(["", "===== REALITY CHECK / SPA APPROXIMATION ====="])
    if reality_df.empty:
        lines.append("Reality check unavailable.")
    else:
        row = reality_df.iloc[0]
        lines.append(f"Method: {row.get('method')}")
        lines.append(f"Feasible: {row.get('feasible')}")
        if "reality_check_p_value" in reality_df.columns:
            lines.append(f"Reality Check p-value: {row.get('reality_check_p_value'):.3f}")
    lines.extend(["", "===== GOVERNANCE VERDICT ====="])
    lines.append(f"All-time trials: {gov.get('all_time_trials', '')}")
    lines.append(f"Effective independent trials: {gov.get('effective_independent_trials', '')}")
    lines.append(f"CSCV PBO: {gov.get('CSCV_PBO', '')}")
    lines.append(f"Exact DSR estimate: {gov.get('deflated_sharpe', '')}")
    lines.append(f"SPA/Reality Check p-value: {gov.get('SPA_or_reality_check_p_value', '')}")
    lines.append(f"Growth champion final survives correction: {gov.get('growth_champion_final_survives_multiple_testing_correction', '')}")
    lines.append(f"Classification: {gov.get('classification', '')}")
    lines.append(f"Reason: {gov.get('reason', '')}")
    lines.extend(["", "===== RETURN SOURCES USED ====="])
    for _, row in return_registry.iterrows():
        lines.append(f"{row.get('source_file')}: loaded={row.get('loaded')} strategies={row.get('strategies')} rows={row.get('rows')} reason={row.get('reason')}")
    Path(OUTPUTS["report"]).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    return_matrix, return_registry = discover_strategy_return_matrix()
    return_registry.to_csv("anti_overfitting_return_sources.csv", index=False)
    effective_df = effective_trial_count(return_matrix)
    effective_df.to_csv(OUTPUTS["effective_trials"], index=False)
    effective_trials = int(effective_df["independent_trials_estimate"].iloc[0]) if not effective_df.empty else 1
    cscv_df, pbo_df = run_cscv(return_matrix)
    cscv_df.to_csv(OUTPUTS["cscv"], index=False)
    pbo_df.to_csv(OUTPUTS["pbo"], index=False)
    dsr_df = exact_dsr(return_matrix, effective_trials)
    dsr_df.to_csv(OUTPUTS["dsr"], index=False)
    reality_df = reality_check(return_matrix)
    reality_df.to_csv(OUTPUTS["reality"], index=False)
    registry_df = build_registries(return_registry, effective_df)
    trial_log = read_csv("strategy_trial_log.csv")
    governance_df = classify_governance(pbo_df, dsr_df, reality_df, trial_log, effective_df)
    governance_df.to_csv(OUTPUTS["governance"], index=False)
    write_report(registry_df, effective_df, pbo_df, dsr_df, reality_df, governance_df, return_registry)

    gov = governance_df.iloc[0]
    print("===== FULL ANTI-OVERFITTING VALIDATION FRAMEWORK =====")
    print(f"all_time_trials: {gov['all_time_trials']}")
    print(f"effective_independent_trials: {gov['effective_independent_trials']}")
    print(f"CSCV_PBO: {gov['CSCV_PBO']}")
    print(f"exact_DSR: {gov['deflated_sharpe']}")
    print(f"reality_check_p_value: {gov['SPA_or_reality_check_p_value']}")
    print(f"growth_champion_final_survives: {gov['growth_champion_final_survives_multiple_testing_correction']}")
    print(f"classification: {gov['classification']}")
    print("outputs:", ", ".join(OUTPUTS.values()))


if __name__ == "__main__":
    main()

