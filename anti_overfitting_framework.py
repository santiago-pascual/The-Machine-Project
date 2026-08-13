from __future__ import annotations

from dataclasses import dataclass
from math import erf, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_TRIAL_LOG = "strategy_trial_log.csv"


@dataclass(frozen=True)
class TrialSource:
    category: str
    path: str
    count_mode: str


TRIAL_SOURCES = [
    TrialSource("walk_forward_runs", "walk_forward_summary.csv", "rows"),
    TrialSource("larger_walk_forward_runs", "larger_walk_forward_summary.csv", "rows"),
    TrialSource("model_modes_tested", "model_mode_comparison.csv", "mode_columns"),
    TrialSource("model_modes_tested", "regime_gated_full_quant_comparison.csv", "mode_columns"),
    TrialSource("threshold_configurations", "threshold_optimization.csv", "rows"),
    TrialSource("tp_sl_barrier_configurations", "barrier_parameter_optimization.csv", "unique_barrier_configs"),
    TrialSource("robustness_validation_configs", "robustness_validation.csv", "rows"),
    TrialSource("trend_persistence_variants", "trend_vs_ema_backtest.csv", "mode_columns"),
    TrialSource("full_quant_variants", "full_quant_robustness_walk_forward.csv", "mode_columns"),
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _count_mode_columns(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    mode_like = [
        col
        for col in df.columns
        if col
        in {
            "baseline",
            "full_quant_research",
            "regime_gated_full_quant",
            "ema",
            "trend_persistence",
        }
        or col.endswith("_sharpe")
        or col.endswith("_return")
    ]
    return max(1, len(set(mode_like)))


def _count_trials(df: pd.DataFrame, mode: str) -> int:
    if df.empty:
        return 0
    if mode == "rows":
        return int(len(df))
    if mode == "mode_columns":
        return int(_count_mode_columns(df))
    if mode == "unique_barrier_configs":
        cols = [c for c in ["horizon", "tp_multiple", "sl_multiple", "subset"] if c in df.columns]
        return int(df[cols].drop_duplicates().shape[0]) if cols else int(len(df))
    return int(len(df))


def build_strategy_trial_log(
    sources: list[TrialSource] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_TRIAL_LOG,
) -> pd.DataFrame:
    sources = sources or TRIAL_SOURCES
    rows = []
    cumulative = 0
    for source in sources:
        df = _read_csv(source.path)
        count = _count_trials(df, source.count_mode)
        cumulative += count
        rows.append(
            {
                "trial_category": source.category,
                "number_of_trials": count,
                "source_file": source.path,
                "count_mode": source.count_mode,
                "cumulative_total_trials": cumulative,
                "file_found": Path(source.path).exists(),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def _best_observed_sharpe() -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for path, cols in [
        ("full_quant_robustness_walk_forward.csv", ["baseline_sharpe", "full_quant_sharpe", "regime_gated_sharpe"]),
        ("regime_gated_full_quant_comparison.csv", ["baseline_sharpe", "full_quant_sharpe", "regime_gated_sharpe"]),
        ("trend_vs_ema_backtest.csv", ["ema", "trend_persistence"]),
        ("barrier_parameter_optimization.csv", ["Sharpe"]),
        ("threshold_optimization.csv", ["Sharpe"]),
        ("walk_forward_summary.csv", ["realized_sharpe"]),
        ("larger_walk_forward_summary.csv", ["Sharpe"]),
    ]:
        df = _read_csv(path)
        if df.empty:
            continue
        for col in cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                continue
            idx = series.idxmax()
            candidates.append(
                {
                    "source_file": path,
                    "column": col,
                    "observed_sharpe": float(series.loc[idx]),
                    "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                    "sample_length": _infer_sample_length(path, int(idx) if isinstance(idx, (int, np.integer)) else None),
                }
            )
    if not candidates:
        return {"observed_sharpe": 0.0, "source_file": "", "column": "", "sample_length": 0}
    return max(candidates, key=lambda row: row["observed_sharpe"])


def _infer_sample_length(path: str, row_index: int | None = None) -> int:
    df = _read_csv(path)
    if df.empty:
        return 0
    for col in ["number_of_test_dates", "sample_size", "test_dates"]:
        if col in df.columns:
            if row_index is not None and row_index in df.index:
                return int(max(0, _safe_float(df.loc[row_index, col], 0.0)))
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                return int(max(0, values.max()))
    if "date" in df.columns:
        return int(df["date"].nunique())
    return int(len(df))


def _return_distribution_stats() -> dict[str, float]:
    candidates = [
        "walk_forward_portfolio_returns.csv",
        "model_mode__baseline_portfolio_returns.csv",
        "model_mode__full_quant_research_portfolio_returns.csv",
        "model_mode__regime_gated_full_quant_portfolio_returns.csv",
    ]
    returns = []
    for path in candidates:
        df = _read_csv(path)
        if df.empty:
            continue
        return_cols = [c for c in df.columns if c.startswith("realized_portfolio_return_")]
        for col in return_cols:
            returns.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 3:
        return {"skewness": 0.0, "kurtosis": 3.0, "sample_size": int(len(series))}
    return {
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis() + 3.0),
        "sample_size": int(len(series)),
    }


def expected_max_sharpe_normal(number_of_trials: int, sample_length: int) -> float:
    trials = max(1, int(number_of_trials))
    sample_length = max(2, int(sample_length))
    # Conservative extreme-value approximation for the best Sharpe found by chance.
    return float(sqrt(2.0 * log(trials)) / sqrt(sample_length))


def deflated_sharpe_diagnostic(
    *,
    observed_sharpe: float,
    number_of_trials: int,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, float | str]:
    observed = _safe_float(observed_sharpe, 0.0)
    n_trials = max(1, int(number_of_trials))
    n_obs = max(2, int(sample_length))
    skew = _safe_float(skewness, 0.0)
    kurt = max(1.0, _safe_float(kurtosis, 3.0))
    expected_max = expected_max_sharpe_normal(n_trials, n_obs)
    denominator = sqrt(max(1e-8, (1.0 - skew * observed + ((kurt - 1.0) / 4.0) * observed * observed) / (n_obs - 1.0)))
    z_score = (observed - expected_max) / denominator
    p_value = float(1.0 - _normal_cdf(z_score))
    deflated = float(observed - expected_max)
    if p_value > 0.40 or deflated <= 0:
        warning = "extreme"
    elif p_value > 0.20:
        warning = "high"
    elif p_value > 0.05:
        warning = "medium"
    else:
        warning = "low"
    return {
        "observed_sharpe": float(observed),
        "number_of_trials": int(n_trials),
        "sample_length": int(n_obs),
        "skewness_used": float(skew),
        "kurtosis_used": float(kurt),
        "expected_max_sharpe_from_trials": float(expected_max),
        "deflated_sharpe_estimate": float(deflated),
        "p_value_approximation": float(np.clip(p_value, 0.0, 1.0)),
        "warning_level": warning,
        "approximation": "conservative_normal_extreme_value_with_skew_kurtosis_adjustment",
    }


def pbo_approximation(
    *,
    total_trials: int,
    robustness_score: float,
    sample_size: int,
    top_config_isolated: bool,
) -> dict[str, object]:
    trials_penalty = min(0.45, log(max(1, total_trials)) / 20.0)
    robustness_penalty = max(0.0, (60.0 - _safe_float(robustness_score, 0.0)) / 100.0)
    sample_penalty = 0.25 if sample_size < 50 else 0.15 if sample_size < 100 else 0.05
    isolation_penalty = 0.15 if top_config_isolated else 0.0
    risk = float(np.clip(0.15 + trials_penalty + robustness_penalty + sample_penalty + isolation_penalty, 0.0, 1.0))
    if risk >= 0.75:
        label = "high risk of overfitting"
    elif risk >= 0.55:
        label = "not enough evidence"
    else:
        label = "safe to research further"
    return {
        "pbo_proxy": risk,
        "pbo_label": label,
        "method": "proxy_only_insufficient_cscv_folds",
        "trials_penalty": trials_penalty,
        "robustness_penalty": robustness_penalty,
        "sample_penalty": sample_penalty,
        "isolation_penalty": isolation_penalty,
    }


def cscv_readiness() -> dict[str, object]:
    predictions = _read_csv("walk_forward_predictions.csv")
    threshold = _read_csv("threshold_optimization.csv")
    barrier = _read_csv("barrier_parameter_optimization.csv")
    observations = int(predictions["date"].nunique()) if not predictions.empty and "date" in predictions.columns else 0
    strategies = int(len(threshold) + len(barrier))
    can_run = observations >= 16 and strategies >= 10
    missing = []
    if observations < 16:
        missing.append("more_walk_forward_dates")
    if strategies < 10:
        missing.append("more_strategy_configurations")
    return {
        "number_of_observations": observations,
        "number_of_strategies_configs": strategies,
        "cscv_can_be_run": bool(can_run),
        "missing": ", ".join(missing) if missing else "none",
    }


def _robustness_score() -> float:
    df = _read_csv("robustness_validation.csv")
    if df.empty:
        return 0.0
    for col in ["robustness_score"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return 0.0


def _top_config_isolated() -> bool:
    df = _read_csv("robustness_validation.csv")
    if df.empty or "cluster_status" not in df.columns:
        return True
    text = " ".join(df["cluster_status"].fillna("").astype(str).str.lower().tolist())
    return "isolated" in text and "stable_cluster" not in text


def _final_verdict(dsr: dict[str, object], pbo: dict[str, object], cscv: dict[str, object]) -> str:
    if dsr["warning_level"] in {"extreme", "high"} and pbo["pbo_proxy"] >= 0.65:
        return "do not promote"
    if not bool(cscv["cscv_can_be_run"]):
        return "not enough evidence"
    if pbo["pbo_proxy"] >= 0.75:
        return "high risk of overfitting"
    if dsr["warning_level"] == "low" and pbo["pbo_proxy"] < 0.45:
        return "safe to research further"
    return "not enough evidence"


def run_anti_overfitting_framework() -> dict[str, pd.DataFrame | dict[str, object]]:
    trial_log = build_strategy_trial_log()
    total_trials = int(trial_log["number_of_trials"].sum()) if not trial_log.empty else 1
    best = _best_observed_sharpe()
    dist = _return_distribution_stats()
    sample_length = int(best.get("sample_length", 0) or dist.get("sample_size", 0) or 20)
    dsr = deflated_sharpe_diagnostic(
        observed_sharpe=float(best.get("observed_sharpe", 0.0)),
        number_of_trials=total_trials,
        sample_length=sample_length,
        skewness=float(dist.get("skewness", 0.0)),
        kurtosis=float(dist.get("kurtosis", 3.0)),
    )
    cscv = cscv_readiness()
    pbo = pbo_approximation(
        total_trials=total_trials,
        robustness_score=_robustness_score(),
        sample_size=sample_length,
        top_config_isolated=_top_config_isolated(),
    )
    verdict = _final_verdict(dsr, pbo, cscv)

    print("\n===== STRATEGY TRIAL COUNTER =====")
    print(trial_log.to_string(index=False))
    print("\n===== DEFLATED SHARPE RATIO DIAGNOSTIC =====")
    print(f"best Sharpe source: {best.get('source_file')}::{best.get('column')}")
    for key, value in dsr.items():
        print(f"{key}: {value}")
    print("\n===== MULTIPLE TESTING SHARPE WARNING =====")
    print(f"total strategy trials: {total_trials}")
    print(f"estimated expected max Sharpe from chance: {dsr['expected_max_sharpe_from_trials']:.6f}")
    print("warning: more trials inflate the best observed Sharpe even if no real edge exists.")
    print("\n===== PROBABILITY OF BACKTEST OVERFITTING =====")
    for key, value in pbo.items():
        print(f"{key}: {value}")
    print("\n===== CSCV READINESS CHECK =====")
    for key, value in cscv.items():
        print(f"{key}: {value}")
    print("\n===== ANTI-OVERFITTING VERDICT =====")
    print(verdict)
    print(f"\nSaved: {Path(DEFAULT_OUTPUT_TRIAL_LOG).resolve()}")

    return {
        "trial_log": trial_log,
        "best_sharpe": best,
        "deflated_sharpe": dsr,
        "pbo": pbo,
        "cscv": cscv,
        "verdict": {"verdict": verdict},
    }


if __name__ == "__main__":
    run_anti_overfitting_framework()
