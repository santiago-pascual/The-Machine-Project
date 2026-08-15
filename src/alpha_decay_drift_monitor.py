from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROLLING_WINDOWS = [5, 20, 60, 126, 252]
HORIZONS = [1, 5, 10, 20, 40, 60]
FEATURE_ALIASES = {
    "raw_target_return_exact": ["raw_target_return_exact", "expected_total_return", "expected_daily_return"],
    "kalman": ["kalman_trend_score", "kalman_trend", "kalman_signal_to_noise"],
    "hurst": ["hurst_persistence_score", "hurst_exponent"],
    "ou": ["ou_zscore", "ou_half_life", "ou_mean_reversion_score"],
    "garch": ["garch_volatility", "egarch_volatility", "daily_volatility"],
    "entropy": ["entropy_cleanliness_score", "entropy", "sample_entropy"],
    "regime": ["regime_confidence", "trend_persistence_score"],
    "signal_strength": ["signal_strength"],
}


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_dataset() -> pd.DataFrame:
    ic = read_csv("historical_ic_dataset.csv")
    meta = read_csv("meta_label_dataset.csv")
    if ic.empty:
        return pd.DataFrame()
    ic["date"] = pd.to_datetime(ic["date"], errors="coerce")
    if not meta.empty:
        meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
        cols = ["date", "ticker"] + [c for names in FEATURE_ALIASES.values() for c in names if c in meta.columns]
        cols = list(dict.fromkeys([c for c in cols if c in meta.columns]))
        ic = ic.merge(meta[cols], on=["date", "ticker"], how="left", suffixes=("", "_meta"))
        for c in list(ic.columns):
            if c.endswith("_meta"):
                base = c[:-5]
                if base not in ic.columns:
                    ic[base] = ic[c]
                else:
                    ic[base] = ic[base].combine_first(ic[c])
                ic = ic.drop(columns=[c])
    return ic.dropna(subset=["date"]).sort_values(["date", "ticker"])


def resolve_feature(df: pd.DataFrame, canonical: str) -> str | None:
    for col in FEATURE_ALIASES[canonical]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 20:
            return col
    return None


def spearman_ic(group: pd.DataFrame, feature: str, target: str) -> float:
    x = pd.to_numeric(group[feature], errors="coerce")
    y = pd.to_numeric(group[target], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 5:
        return np.nan
    return float(valid["x"].corr(valid["y"], method="spearman"))


def daily_ic(df: pd.DataFrame, feature: str, horizon: int) -> pd.DataFrame:
    target = f"realized_return_{horizon}d"
    if target not in df.columns:
        return pd.DataFrame()
    rows = []
    for dt, group in df.groupby("date"):
        rows.append({"date": dt, "ic": spearman_ic(group, feature, target)})
    out = pd.DataFrame(rows).dropna()
    return out.sort_values("date")


def rolling_feature_ic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for canonical in FEATURE_ALIASES:
        feature = resolve_feature(df, canonical)
        if not feature:
            rows.append({"feature": canonical, "source_column": "", "status": "missing"})
            continue
        for horizon in [5, 20]:
            ic = daily_ic(df, feature, horizon)
            if ic.empty:
                continue
            for window in ROLLING_WINDOWS:
                tmp = ic.copy()
                tmp["rolling_window"] = window
                tmp["rolling_ic_mean"] = tmp["ic"].rolling(window, min_periods=max(3, window // 3)).mean()
                tmp["rolling_ic_vol"] = tmp["ic"].rolling(window, min_periods=max(3, window // 3)).std()
                tmp["horizon"] = f"{horizon}d"
                tmp["feature"] = canonical
                tmp["source_column"] = feature
                rows.extend(tmp.tail(60).to_dict("records"))
    return pd.DataFrame(rows)


def alpha_decay_curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for canonical in FEATURE_ALIASES:
        feature = resolve_feature(df, canonical)
        if not feature:
            rows.append({"feature": canonical, "source_column": "", "status": "missing"})
            continue
        for horizon in HORIZONS:
            target = f"realized_return_{horizon}d"
            if target not in df.columns:
                rows.append({"feature": canonical, "source_column": feature, "horizon": f"{horizon}d", "status": "missing_target"})
                continue
            per_date = daily_ic(df, feature, horizon)
            rows.append(
                {
                    "feature": canonical,
                    "source_column": feature,
                    "horizon": f"{horizon}d",
                    "dates": len(per_date),
                    "mean_rank_ic": float(per_date["ic"].mean()) if not per_date.empty else np.nan,
                    "median_rank_ic": float(per_date["ic"].median()) if not per_date.empty else np.nan,
                    "positive_ic_rate": float((per_date["ic"] > 0).mean()) if not per_date.empty else np.nan,
                    "ic_volatility": float(per_date["ic"].std()) if not per_date.empty else np.nan,
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna()
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(e) < buckets or len(a) < buckets:
        return np.nan
    qs = np.unique(np.quantile(e, np.linspace(0, 1, buckets + 1)))
    if len(qs) < 3:
        return np.nan
    e_counts = pd.cut(e, qs, include_lowest=True).value_counts(normalize=True).sort_index()
    a_counts = pd.cut(a, qs, include_lowest=True).value_counts(normalize=True).sort_index()
    e_pct = e_counts.reindex(e_counts.index, fill_value=0).to_numpy() + 1e-6
    a_pct = a_counts.reindex(e_counts.index, fill_value=0).to_numpy() + 1e-6
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def kl_divergence(expected: pd.Series, actual: pd.Series, buckets: int = 20) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna()
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(e) < buckets or len(a) < buckets:
        return np.nan
    lo, hi = np.nanpercentile(pd.concat([e, a]), [1, 99])
    e_hist, bins = np.histogram(e.clip(lo, hi), bins=buckets, density=True)
    a_hist, _ = np.histogram(a.clip(lo, hi), bins=bins, density=True)
    e_p = e_hist / max(e_hist.sum(), 1e-12) + 1e-6
    a_p = a_hist / max(a_hist.sum(), 1e-12) + 1e-6
    return float(np.sum(a_p * np.log(a_p / e_p)))


def structural_breaks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cutoff = df["date"].quantile(0.65)
    for canonical in FEATURE_ALIASES:
        feature = resolve_feature(df, canonical)
        if not feature:
            rows.append({"feature": canonical, "status": "missing"})
            continue
        ic = daily_ic(df, feature, 20)
        if len(ic) < 30:
            rows.append({"feature": canonical, "source_column": feature, "status": "insufficient_ic"})
            continue
        ic["centered"] = ic["ic"] - ic["ic"].mean()
        ic["cusum"] = ic["centered"].cumsum()
        cusum_stat = float(ic["cusum"].abs().max() / (ic["ic"].std() * np.sqrt(len(ic)) + 1e-12))
        bp_breaks, bp_rss_improvement = bai_perron_style_breaks(ic["ic"].to_numpy())
        first = ic.loc[ic["date"] <= cutoff, "ic"]
        second = ic.loc[ic["date"] > cutoff, "ic"]
        mean_diff = float(second.mean() - first.mean()) if len(first) and len(second) else np.nan
        pooled = np.sqrt(first.var(ddof=1) / max(len(first), 1) + second.var(ddof=1) / max(len(second), 1))
        chow_like_t = float(mean_diff / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append(
            {
                "feature": canonical,
                "source_column": feature,
                "dates": len(ic),
                "first_period_mean_ic": float(first.mean()) if len(first) else np.nan,
                "recent_period_mean_ic": float(second.mean()) if len(second) else np.nan,
                "mean_ic_change": mean_diff,
                "sign_flip": bool(np.sign(first.mean()) != np.sign(second.mean())) if len(first) and len(second) else False,
                "cusum_stat": cusum_stat,
                "chow_like_t_stat": chow_like_t,
                "bai_perron_candidate_breaks": "|".join(str(int(x)) for x in bp_breaks),
                "bai_perron_rss_improvement": bp_rss_improvement,
                "volatility_increase": float(second.std() / first.std()) if first.std() and np.isfinite(first.std()) else np.nan,
                "status": "break_warning" if cusum_stat > 1.5 or abs(chow_like_t) > 2.0 else "ok",
            }
        )
    return pd.DataFrame(rows)


def bai_perron_style_breaks(values: np.ndarray, max_breaks: int = 2) -> tuple[list[int], float]:
    """Dependency-free multiple-break diagnostic on IC means.

    This is not a full Bai-Perron implementation with formal critical values. It
    searches chronological split points and reports RSS improvement versus a
    no-break mean model, which is suitable for monitoring without tuning.
    """
    y = pd.Series(values).dropna().to_numpy(dtype=float)
    n = len(y)
    min_seg = max(15, n // 10)
    if n < min_seg * 3:
        return [], np.nan

    def rss_for(cuts: list[int]) -> float:
        pts = [0] + cuts + [n]
        rss = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            seg = y[a:b]
            if len(seg) == 0:
                continue
            rss += float(((seg - seg.mean()) ** 2).sum())
        return rss

    base_rss = rss_for([])
    best_cuts: list[int] = []
    best_rss = base_rss
    candidates = list(range(min_seg, n - min_seg, max(1, n // 40)))
    for i, cut1 in enumerate(candidates):
        candidate_sets = [[cut1]]
        if max_breaks >= 2:
            for cut2 in candidates[i + 1 :]:
                if cut2 - cut1 >= min_seg:
                    candidate_sets.append([cut1, cut2])
        for cuts in candidate_sets:
            if any((b - a) < min_seg for a, b in zip([0] + cuts, cuts + [n])):
                continue
            current_rss = rss_for(cuts)
            if current_rss < best_rss:
                best_rss = current_rss
                best_cuts = cuts
    improvement = float((base_rss - best_rss) / base_rss) if base_rss > 0 else np.nan
    return best_cuts, improvement


def distribution_drift(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cutoff = df["date"].quantile(0.65)
    early = df.loc[df["date"] <= cutoff]
    recent = df.loc[df["date"] > cutoff]
    resolved = {k: resolve_feature(df, k) for k in FEATURE_ALIASES}
    numeric = [c for c in resolved.values() if c]
    corr_early = early[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman") if numeric else pd.DataFrame()
    corr_recent = recent[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman") if numeric else pd.DataFrame()
    redundancy_delta = (
        float(corr_recent.abs().mean().mean() - corr_early.abs().mean().mean())
        if not corr_early.empty and not corr_recent.empty
        else np.nan
    )
    for canonical, feature in resolved.items():
        if not feature:
            rows.append({"feature": canonical, "source_column": "", "status": "missing"})
            continue
        rows.append(
            {
                "feature": canonical,
                "source_column": feature,
                "early_observations": int(early[feature].notna().sum()),
                "recent_observations": int(recent[feature].notna().sum()),
                "PSI": psi(early[feature], recent[feature]),
                "KL_divergence": kl_divergence(early[feature], recent[feature]),
                "early_mean": float(pd.to_numeric(early[feature], errors="coerce").mean()),
                "recent_mean": float(pd.to_numeric(recent[feature], errors="coerce").mean()),
                "feature_redundancy_delta_global": redundancy_delta,
                "status": "drift_warning",
            }
        )
    out = pd.DataFrame(rows)
    if "PSI" in out.columns:
        out.loc[out["PSI"].fillna(0) < 0.10, "status"] = "stable"
        out.loc[out["PSI"].fillna(0).between(0.10, 0.25), "status"] = "mild_drift"
        out.loc[out["PSI"].fillna(0) >= 0.25, "status"] = "significant_drift"
    return out


def regime_alpha(df: pd.DataFrame) -> pd.DataFrame:
    feature = resolve_feature(df, "raw_target_return_exact")
    rows = []
    if not feature or "regime" not in df.columns:
        return pd.DataFrame([{"status": "missing regime or alpha feature"}])
    target = "realized_return_20d"
    if target not in df.columns:
        return pd.DataFrame([{"status": "missing realized_return_20d"}])
    tmp = df.copy()
    tmp["vol_bucket"] = pd.qcut(
        pd.to_numeric(tmp[target], errors="coerce").abs(), 2, labels=["low_volatility", "high_volatility"], duplicates="drop"
    )
    for col in ["regime", "vol_bucket"]:
        for val, group in tmp.groupby(col, dropna=True):
            rows.append(
                {
                    "group_type": col,
                    "group": str(val),
                    "observations": len(group),
                    "rank_ic_20d": spearman_ic(group, feature, target),
                    "avg_realized_return_20d": float(pd.to_numeric(group[target], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def governance(rolling: pd.DataFrame, decay: pd.DataFrame, breaks: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    ok_decay = decay.loc[decay.get("status", "").eq("ok")] if not decay.empty and "status" in decay.columns else pd.DataFrame()
    raw20 = ok_decay.loc[(ok_decay["feature"].eq("raw_target_return_exact")) & (ok_decay["horizon"].eq("20d"))]
    raw_ic = float(raw20["mean_rank_ic"].iloc[0]) if not raw20.empty else np.nan
    break_warnings = int((breaks.get("status", pd.Series(dtype=str)) == "break_warning").sum()) if not breaks.empty else 0
    significant_drift = int((drift.get("status", pd.Series(dtype=str)) == "significant_drift").sum()) if not drift.empty else 0
    sign_flips = int(breaks.get("sign_flip", pd.Series(dtype=bool)).fillna(False).sum()) if not breaks.empty else 0
    if break_warnings >= 3 or significant_drift >= 3 or sign_flips >= 3:
        classification = "retraining_review_required"
    elif break_warnings >= 1 or significant_drift >= 1 or sign_flips >= 1:
        classification = "significant_decay"
    elif np.isfinite(raw_ic) and raw_ic > 0:
        classification = "stable"
    else:
        classification = "mild_decay"
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "raw_target_rank_ic_20d": raw_ic,
                "break_warnings": break_warnings,
                "significant_distribution_drifts": significant_drift,
                "sign_flips": sign_flips,
                "production_changed": False,
                "paper_changed": False,
                "automatic_retraining": False,
                "reason": f"break_warnings={break_warnings}; significant_drift={significant_drift}; sign_flips={sign_flips}; raw20_ic={raw_ic}",
            }
        ]
    )


def main() -> None:
    df = load_dataset()
    if df.empty:
        empty = pd.DataFrame([{"classification": "retraining_review_required", "reason": "missing historical dataset"}])
        for path in [
            "rolling_feature_ic.csv",
            "alpha_decay_curve.csv",
            "structural_break_results.csv",
            "feature_distribution_drift.csv",
            "alpha_decay_governance.csv",
        ]:
            empty.to_csv(path, index=False)
        print("missing historical dataset")
        return
    rolling = rolling_feature_ic(df)
    decay = alpha_decay_curve(df)
    breaks = structural_breaks(df)
    drift = distribution_drift(df)
    regime = regime_alpha(df)
    gov = governance(rolling, decay, breaks, drift)
    rolling.to_csv("rolling_feature_ic.csv", index=False)
    decay.to_csv("alpha_decay_curve.csv", index=False)
    breaks.to_csv("structural_break_results.csv", index=False)
    drift.to_csv("feature_distribution_drift.csv", index=False)
    regime.to_csv("regime_specific_alpha_decay.csv", index=False)
    gov.to_csv("alpha_decay_governance.csv", index=False)
    print("===== ALPHA DECAY AND STRUCTURAL DRIFT MONITOR =====")
    print(f"rows: {len(df)}")
    print(f"classification: {gov.iloc[0]['classification']}")
    print(f"raw_target_rank_ic_20d: {gov.iloc[0]['raw_target_rank_ic_20d']}")
    print(f"break_warnings: {gov.iloc[0]['break_warnings']}")
    print(f"significant_distribution_drifts: {gov.iloc[0]['significant_distribution_drifts']}")
    print(
        "outputs: rolling_feature_ic.csv, alpha_decay_curve.csv, structural_break_results.csv, feature_distribution_drift.csv, alpha_decay_governance.csv"
    )


if __name__ == "__main__":
    main()
