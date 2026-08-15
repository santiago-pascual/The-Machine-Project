from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_hmm_regime_validation import (
    best_fit_for_k,
    build_feature_data,
    decode_states,
    label_states,
    standardize,
)

TRADING_PERIODS = 52.178571  # 5-session cadence annualization from existing purged WF files.
N_STATES = 4
SEEDS = 8
MAX_TURNOVER_INCREASE = 0.15
MAX_DD_TOLERANCE = 0.02
MIN_FOLDS_FOR_GOVERNANCE = 5

OUT_RESULTS = "hmm_incremental_portfolio_results.csv"
OUT_FOLDS = "hmm_incremental_portfolio_folds.csv"
OUT_DAILY = "hmm_incremental_portfolio_daily_returns.csv"
OUT_REGIME = "hmm_incremental_performance_by_regime.csv"
OUT_GOV = "hmm_incremental_governance.csv"
OUT_REPORT = "hmm_incremental_portfolio_report.txt"


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def normalize_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out.dropna(subset=[col]).sort_values(col)


def load_canonical_series() -> tuple[pd.DataFrame, str]:
    exact = normalize_dates(read_csv("growth_champion_canonical_daily.csv"))
    recon = normalize_dates(read_csv("growth_champion_reconstructed_stress_daily.csv"))
    if len(exact) >= 250:
        return exact, "canonical_exact"
    if len(recon) >= 250:
        return recon, "reconstructed_stress_for_oos_incremental_test"
    return exact if not exact.empty else recon, "insufficient_history"


def metrics(
    df: pd.DataFrame, ret_col: str = "strategy_return", turnover_col: str = "turnover", exposure_col: str = "exposure"
) -> dict[str, float]:
    r = pd.to_numeric(df.get(ret_col, pd.Series(dtype=float)), errors="coerce").dropna()
    if r.empty:
        return {
            k: np.nan
            for k in [
                "total_return",
                "net_CAGR",
                "Sharpe",
                "Sortino",
                "Calmar",
                "max_drawdown",
                "turnover",
                "average_exposure",
                "alpha_vs_SPY",
                "alpha_vs_QQQ",
                "information_ratio_vs_SPY",
                "information_ratio_vs_QQQ",
            ]
        }
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    years = max(len(r) / TRADING_PERIODS, 1e-9)
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else np.nan
    vol = float(r.std(ddof=0) * math.sqrt(TRADING_PERIODS)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=0) * math.sqrt(TRADING_PERIODS)) if r.std(ddof=0) > 0 else np.nan
    downside = r[r < 0]
    sortino = (
        float(r.mean() / downside.std(ddof=0) * math.sqrt(TRADING_PERIODS)) if len(downside) > 1 and downside.std(ddof=0) > 0 else np.nan
    )
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 and np.isfinite(cagr) else np.nan
    spy = pd.to_numeric(df.get("spy_daily_return", pd.Series(index=df.index, dtype=float)), errors="coerce")
    qqq = pd.to_numeric(df.get("qqq_daily_return", pd.Series(index=df.index, dtype=float)), errors="coerce")
    alpha_spy = float((r - spy.loc[r.index]).mean() * TRADING_PERIODS) if not spy.dropna().empty else np.nan
    alpha_qqq = float((r - qqq.loc[r.index]).mean() * TRADING_PERIODS) if not qqq.dropna().empty else np.nan
    ir_spy = (
        float((r - spy.loc[r.index]).mean() / (r - spy.loc[r.index]).std(ddof=0) * math.sqrt(TRADING_PERIODS))
        if not spy.dropna().empty and (r - spy.loc[r.index]).std(ddof=0) > 0
        else np.nan
    )
    ir_qqq = (
        float((r - qqq.loc[r.index]).mean() / (r - qqq.loc[r.index]).std(ddof=0) * math.sqrt(TRADING_PERIODS))
        if not qqq.dropna().empty and (r - qqq.loc[r.index]).std(ddof=0) > 0
        else np.nan
    )
    return {
        "total_return": total,
        "net_CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "max_drawdown": max_dd,
        "turnover": float(pd.to_numeric(df.get(turnover_col, pd.Series(dtype=float)), errors="coerce").mean()),
        "average_exposure": float(pd.to_numeric(df.get(exposure_col, pd.Series(dtype=float)), errors="coerce").mean()),
        "cost_drag": float(pd.to_numeric(df.get("cost_drag", pd.Series(index=df.index, dtype=float)), errors="coerce").sum(skipna=True)),
        "alpha_vs_SPY": alpha_spy,
        "alpha_vs_QQQ": alpha_qqq,
        "information_ratio_vs_SPY": ir_spy,
        "information_ratio_vs_QQQ": ir_qqq,
    }


def regime_overlay_factor(label: str, base_exposure: float) -> tuple[float, str]:
    if label == "risk_off":
        return min(base_exposure, 0.25) / base_exposure if base_exposure > 0 else 1.0, "hmm_risk_off_exposure_cap_25"
    if label == "neutral":
        return min(base_exposure, 0.45) / base_exposure if base_exposure > 0 else 1.0, "hmm_neutral_exposure_cap_45"
    return 1.0, "hmm_risk_on_no_change"


def ranking_proxy_factor(label: str, train_regime_mean: dict[str, float], base_exposure: float) -> tuple[float, str]:
    if not train_regime_mean:
        return 1.0, "missing_train_regime_means"
    vals = np.array([v for v in train_regime_mean.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 1.0, "missing_train_regime_means"
    median = float(np.median(vals))
    mean = train_regime_mean.get(label, np.nan)
    if not np.isfinite(mean):
        return 1.0, "unseen_regime_no_change"
    # Proxy for a ranking tilt: reduce allocation in historically weak HMM regimes and modestly add in strong regimes, capped by existing exposure cap.
    if mean < median:
        return 0.85, "hmm_weak_regime_ranking_proxy_reduce_15pct"
    return min(0.60, base_exposure * 1.10) / base_exposure if base_exposure > 0 else 1.0, "hmm_strong_regime_ranking_proxy_add_up_to_cap"


def build_fold_rows(series: pd.DataFrame, feature_df: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    series = series.copy()
    series["base_return"] = pd.to_numeric(series.get("net_daily_return", series.get("gross_daily_return")), errors="coerce")
    series["base_exposure"] = pd.to_numeric(series.get("exposure"), errors="coerce").fillna(0.0)
    series["base_turnover"] = pd.to_numeric(series.get("turnover"), errors="coerce").fillna(0.0)
    for _, f in folds.iterrows():
        fold_id = int(f["fold_id"])
        train_end = pd.Timestamp(f["train_end_after_purge"]).normalize()
        test_start = pd.Timestamp(f["test_start"]).normalize()
        test_end = pd.Timestamp(f["test_end"]).normalize()
        train_feat = feature_df[feature_df.index <= train_end]
        test_feat = feature_df[(feature_df.index >= test_start) & (feature_df.index <= test_end)]
        if len(train_feat) < 500 or test_feat.empty:
            continue
        x_train, mean, std = standardize(train_feat)
        fit, _ = best_fit_for_k(x_train, N_STATES, SEEDS, 500, 1e-4)
        labels = label_states(fit, train_feat, mean, std)
        test_scaled = ((test_feat - mean) / std).to_numpy(dtype=float)
        states = decode_states(test_scaled, fit)
        test_regimes = pd.DataFrame(
            {"date": test_feat.index.normalize(), "hmm_state": states, "hmm_regime_label": [labels[s] for s in states]}
        )
        train_series = series[series["date"] <= train_end].merge(
            pd.DataFrame({"date": train_feat.index.normalize(), "train_dummy": 1}), on="date", how="inner"
        )
        # Use full-sample decoded labels on train fit for causal train-only regime means.
        train_states = decode_states(x_train, fit)
        train_reg = pd.DataFrame({"date": train_feat.index.normalize(), "hmm_regime_label": [labels[s] for s in train_states]})
        train_join = train_series.merge(train_reg, on="date", how="left")
        train_means = train_join.groupby("hmm_regime_label")["base_return"].mean().to_dict()
        test_series = series[(series["date"] >= test_start) & (series["date"] <= test_end)].merge(test_regimes, on="date", how="left")
        if test_series.empty:
            continue
        variants = []
        for variant in ["canonical", "hmm_diagnostic_only", "hmm_exposure_overlay", "hmm_ranking_adjustment_proxy"]:
            tmp = test_series.copy()
            if variant in {"canonical", "hmm_diagnostic_only"}:
                tmp["strategy_return"] = tmp["base_return"]
                tmp["strategy_exposure"] = tmp["base_exposure"]
                tmp["strategy_turnover"] = tmp["base_turnover"]
                tmp["hmm_action"] = "no_allocation_change" if variant == "hmm_diagnostic_only" else "canonical_no_hmm"
            elif variant == "hmm_exposure_overlay":
                factors = tmp.apply(
                    lambda r: regime_overlay_factor(str(r.get("hmm_regime_label", "")), float(r.get("base_exposure", 0.0))), axis=1
                )
                tmp["hmm_factor"] = [x[0] for x in factors]
                tmp["hmm_action"] = [x[1] for x in factors]
                tmp["strategy_return"] = tmp["base_return"] * tmp["hmm_factor"]
                tmp["strategy_exposure"] = tmp["base_exposure"] * tmp["hmm_factor"]
                tmp["strategy_turnover"] = tmp["base_turnover"] + tmp["strategy_exposure"].diff().abs().fillna(0.0) * 0.5
            else:
                factors = tmp.apply(
                    lambda r: ranking_proxy_factor(str(r.get("hmm_regime_label", "")), train_means, float(r.get("base_exposure", 0.0))),
                    axis=1,
                )
                tmp["hmm_factor"] = [x[0] for x in factors]
                tmp["hmm_action"] = [x[1] for x in factors]
                tmp["strategy_return"] = tmp["base_return"] * tmp["hmm_factor"]
                tmp["strategy_exposure"] = np.minimum(0.60, tmp["base_exposure"] * tmp["hmm_factor"])
                tmp["strategy_turnover"] = tmp["base_turnover"] + tmp["strategy_exposure"].diff().abs().fillna(0.0) * 0.5
            tmp["variant"] = variant
            tmp["fold_id"] = fold_id
            tmp["train_end_after_purge"] = train_end
            tmp["test_start"] = test_start
            tmp["test_end"] = test_end
            variants.append(tmp)
            m = metrics(tmp, "strategy_return", "strategy_turnover", "strategy_exposure")
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "variant": variant,
                    "train_end_after_purge": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "observations": len(tmp),
                    **m,
                }
            )
        daily_rows.extend(pd.concat(variants, ignore_index=True, sort=False).to_dict("records"))
    return pd.DataFrame(daily_rows), pd.DataFrame(fold_rows)


def summarize(daily: pd.DataFrame, folds: pd.DataFrame, data_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    regime_rows = []
    for variant, group in daily.groupby("variant"):
        results.append(
            {
                "variant": variant,
                "data_mode": data_mode,
                "folds": int(group["fold_id"].nunique()),
                "observations": len(group),
                **metrics(group, "strategy_return", "strategy_turnover", "strategy_exposure"),
            }
        )
        for regime, rg in group.groupby("hmm_regime_label", dropna=False):
            regime_rows.append(
                {
                    "variant": variant,
                    "hmm_regime_label": regime,
                    "observations": len(rg),
                    **metrics(rg, "strategy_return", "strategy_turnover", "strategy_exposure"),
                }
            )
    results_df = pd.DataFrame(results)
    stability = (
        folds.groupby("variant")
        .agg(
            folds=("fold_id", "nunique"),
            median_oos_Sharpe=("Sharpe", "median"),
            mean_oos_Sharpe=("Sharpe", "mean"),
            positive_sharpe_folds=("Sharpe", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            median_max_drawdown=("max_drawdown", "median"),
            median_turnover=("turnover", "median"),
            median_CAGR=("net_CAGR", "median"),
        )
        .reset_index()
    )
    return results_df, stability, pd.DataFrame(regime_rows)


def governance(results: pd.DataFrame, stability: pd.DataFrame, data_mode: str, exact_rows: int, fold_results: pd.DataFrame) -> pd.DataFrame:
    best_variant = ""
    best_delta = np.nan
    best_dd_delta = np.nan
    best_turnover_increase = np.nan
    improved_folds = 0
    comparable_folds = 0
    if results.empty or stability.empty or fold_results.empty:
        cls = "diagnostic_only"
        reason = "insufficient inputs for HMM incremental portfolio test"
    else:
        base = fold_results.loc[fold_results["variant"].eq("canonical")].set_index("fold_id")
        candidates = [v for v in ["hmm_exposure_overlay", "hmm_ranking_adjustment_proxy"] if fold_results["variant"].eq(v).any()]
        for v in candidates:
            x = fold_results.loc[fold_results["variant"].eq(v)].set_index("fold_id")
            joined = x.join(base[["Sharpe", "max_drawdown", "turnover"]], rsuffix="_base", how="inner")
            if joined.empty:
                continue
            delta = float((joined["Sharpe"] - joined["Sharpe_base"]).median())
            if not np.isfinite(best_delta) or delta > best_delta:
                best_variant = v
                best_delta = delta
                best_dd_delta = float((joined["max_drawdown"] - joined["max_drawdown_base"]).median())
                best_turnover_increase = float(
                    ((joined["turnover"] - joined["turnover_base"]) / joined["turnover_base"].abs().clip(lower=1e-9)).median()
                )
                improved_folds = int((joined["Sharpe"] > joined["Sharpe_base"]).sum())
                comparable_folds = len(joined)
        if not best_variant:
            cls = "diagnostic_only"
            reason = "no comparable HMM portfolio variant generated"
        else:
            stable_folds = improved_folds >= max(1, int(0.55 * comparable_folds))
            if best_delta <= 0.05:
                cls = "redundant_with_dual_trend"
                reason = f"best variant {best_variant} paired median OOS Sharpe delta {best_delta:.3f} is not meaningful; improved {improved_folds}/{comparable_folds} folds"
            elif best_dd_delta < -MAX_DD_TOLERANCE:
                cls = "diagnostic_only"
                reason = f"{best_variant} improves Sharpe but worsens paired median max DD by {best_dd_delta:.3f}"
            elif best_turnover_increase > MAX_TURNOVER_INCREASE:
                cls = "diagnostic_only"
                reason = f"{best_variant} turnover increase {best_turnover_increase:.1%} exceeds 15% limit"
            elif not stable_folds:
                cls = "diagnostic_only"
                reason = f"{best_variant} improvement not stable across most folds; improved {improved_folds}/{comparable_folds}"
            else:
                cls = "incremental_candidate"
                reason = (
                    f"{best_variant} passes paired median Sharpe/DD/turnover gates on reconstructed OOS folds; shadow-only review required"
                )
    if data_mode != "canonical_exact" or exact_rows < 250:
        if cls in {"incremental_candidate", "eligible_for_shadow"}:
            cls = "incremental_candidate"
            reason += "; exact canonical live history is too short, so no paper/production promotion"
    return pd.DataFrame(
        [
            {
                "governance": cls,
                "data_mode": data_mode,
                "exact_canonical_rows": exact_rows,
                "best_variant": best_variant,
                "paired_median_sharpe_delta": best_delta,
                "paired_median_max_dd_delta": best_dd_delta,
                "paired_median_turnover_increase": best_turnover_increase,
                "improved_folds": improved_folds,
                "comparable_folds": comparable_folds,
                "production_changed": False,
                "paper_changed": False,
                "automatic_promotion": False,
                "reason": reason,
            }
        ]
    )


def main() -> None:
    exact = normalize_dates(read_csv("growth_champion_canonical_daily.csv"))
    series, data_mode = load_canonical_series()
    folds = read_csv("purged_walk_forward_folds.csv")
    if folds.empty:
        raise SystemExit("Missing purged_walk_forward_folds.csv")
    feature_df = build_feature_data("2008-01-01")
    if feature_df.empty or series.empty:
        empty = pd.DataFrame([{"governance": "diagnostic_only", "reason": "missing HMM features or canonical series"}])
        empty.to_csv(OUT_GOV, index=False)
        print("===== HMM INCREMENTAL PORTFOLIO TEST =====")
        print("status: missing inputs")
        return
    series = series.copy()
    series["date"] = pd.to_datetime(series["date"], errors="coerce").dt.normalize()
    daily, fold_results = build_fold_rows(series, feature_df, folds)
    results, stability, regime = summarize(daily, fold_results, data_mode)
    gov = governance(results, stability, data_mode, len(exact), fold_results)
    results.to_csv(OUT_RESULTS, index=False)
    fold_results.to_csv(OUT_FOLDS, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    regime.to_csv(OUT_REGIME, index=False)
    gov.to_csv(OUT_GOV, index=False)
    report = [
        "===== FOUR-STATE HMM INCREMENTAL PORTFOLIO TEST =====",
        f"data_mode: {data_mode}",
        f"exact_canonical_rows: {len(exact)}",
        f"folds_evaluated: {fold_results['fold_id'].nunique() if not fold_results.empty else 0}",
        f"governance: {gov.iloc[0]['governance']}",
        f"reason: {gov.iloc[0]['reason']}",
        "note: hmm_ranking_adjustment_proxy is not a production-parity cross-sectional ranking rebuild; it is a causal regime-conditioned proxy because daily constituent ranking panels are not available in canonical history.",
        "production_changed: False",
        "paper_changed: False",
    ]
    Path(OUT_REPORT).write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(
        "outputs: hmm_incremental_portfolio_results.csv, hmm_incremental_portfolio_folds.csv, hmm_incremental_portfolio_daily_returns.csv, hmm_incremental_performance_by_regime.csv, hmm_incremental_governance.csv"
    )


if __name__ == "__main__":
    main()
