from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
FEATURE_STORE_FILE = "historical_feature_store.csv"
REALIZED_FILE = "historical_realized_returns.csv"
IC_DATASET_FILE = "historical_ic_dataset.csv"
ALPHA_REPORT_FILE = "alpha_attribution_report.csv"
FACTOR_ALPHA_FILE = "factor_alpha_model_results.csv"

DECOMPOSITION_FILE = "expected_return_decomposition.csv"
STAGE_ATTRIBUTION_FILE = "expected_return_stage_attribution.csv"
GAIN_LOSS_FILE = "expected_return_alpha_gain_loss.csv"
HARMFUL_FILE = "expected_return_harmful_transformations.csv"
HORIZONS = [5, 10, 20]


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _pearson(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].corr(frame["y"]))


def _hit_rate(signal: pd.Series, realized: pd.Series) -> float:
    frame = pd.DataFrame({"s": _num(signal), "r": _num(realized)}).dropna()
    if frame.empty:
        return np.nan
    return float((np.sign(frame["s"]) == np.sign(frame["r"])).mean())


def _monotonicity(signal: pd.Series, realized: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"s": _num(signal), "r": _num(realized)}).dropna()
    if len(frame) < 50 or frame["s"].nunique() < 10:
        return np.nan, np.nan
    try:
        frame["decile"] = pd.qcut(frame["s"].rank(method="first"), 10, labels=False) + 1
    except ValueError:
        return np.nan, np.nan
    grouped = frame.groupby("decile")["r"].mean().reset_index()
    corr = _spearman(grouped["decile"], grouped["r"])
    spread = float(grouped.loc[grouped["decile"].eq(10), "r"].mean() - grouped.loc[grouped["decile"].eq(1), "r"].mean())
    return corr, spread


def _load_dataset() -> pd.DataFrame:
    snapshots = _read_csv(SNAPSHOTS_FILE)
    realized = _read_csv(REALIZED_FILE)
    if snapshots.empty:
        snapshots = _read_csv(IC_DATASET_FILE)
    if snapshots.empty:
        return pd.DataFrame()
    data = snapshots.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date", "ticker"])
    if not realized.empty and {"date", "ticker", "model_mode"}.issubset(realized.columns):
        realized = realized.copy()
        realized["date"] = pd.to_datetime(realized["date"], errors="coerce").dt.normalize()
        realized_cols = [f"realized_return_{h}d" for h in [1, 5, 10, 20, 30] if f"realized_return_{h}d" in realized.columns]
        data = data.drop(columns=[col for col in realized_cols if col in data.columns], errors="ignore")
        data = data.merge(realized[["date", "ticker", "model_mode"] + realized_cols], on=["date", "ticker", "model_mode"], how="left")
    for col in data.columns:
        if col not in {
            "date",
            "ticker",
            "model_mode",
            "selected",
            "regime",
            "timing_model",
            "target_model",
            "covariance_method",
            "gate_decision",
            "gate_reason",
        }:
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0:
                data[col] = converted
    return data


def _stage_frame(data: pd.DataFrame) -> pd.DataFrame:
    out = data[["date", "ticker", "model_mode"]].copy()
    current = _num(data.get("current_price", pd.Series(index=data.index)))
    target = _num(data.get("target_price", pd.Series(index=data.index)))
    raw_target_return = target / current.replace(0, np.nan) - 1.0
    expected = _num(data.get("expected_daily_return", pd.Series(index=data.index))).fillna(0.0)
    signal = _num(data.get("signal_strength", pd.Series(index=data.index))).fillna(0.0)
    confidence = _num(data.get("target_confidence", pd.Series(index=data.index))).fillna(0.5)
    quality = _num(data.get("quality_score", pd.Series(index=data.index))).fillna(0.5)
    ema = _num(data.get("ema_timing_score", pd.Series(index=data.index))).fillna(0.5)
    trend = _num(data.get("trend_persistence_score", pd.Series(index=data.index))).fillna(ema)
    regime_conf = _num(data.get("regime_confidence", pd.Series(index=data.index))).fillna(0.5)
    cash = _num(data.get("cash_weight", pd.Series(index=data.index))).fillna(0.5)

    if raw_target_return.notna().sum() == 0:
        raw_target_return = _num(data.get("expected_total_return", expected * 20.0))

    raw_daily_proxy = np.sign(raw_target_return) * (np.power(1.0 + raw_target_return.clip(lower=-0.999), 1.0 / 20.0) - 1.0).abs()
    before_penalties_proxy = raw_daily_proxy.replace([np.inf, -np.inf], np.nan).fillna(expected)
    risk_free_adjusted_proxy = before_penalties_proxy - 0.000147
    constant_penalty_proxy = risk_free_adjusted_proxy - 0.001
    signal_adjusted_proxy = constant_penalty_proxy * (signal.clip(lower=0.0) ** 1.5) * (0.3 + 0.7 * signal.clip(0.0, 1.0))
    regime_multiplier = np.where(
        data.get("regime", pd.Series("", index=data.index)).astype(str).eq("risk_on"),
        1.2,
        np.where(data.get("regime", pd.Series("", index=data.index)).astype(str).eq("risk_off"), 0.6, 0.85),
    )
    regime_adjusted_proxy = signal_adjusted_proxy * regime_multiplier
    confidence_adjusted_proxy = regime_adjusted_proxy * (0.5 + 0.5 * confidence.clip(0.0, 1.0))
    quality_adjusted_proxy = confidence_adjusted_proxy * (0.90 + 0.18 * quality.clip(0.0, 1.0))
    timing_adjusted_proxy = quality_adjusted_proxy * (0.55 + 0.45 * ema.clip(0.0, 1.0))
    scaling_proxy = expected
    final_expected_return = expected

    stages = {
        "raw_target_return_proxy": raw_target_return,
        "raw_daily_return_proxy": raw_daily_proxy,
        "before_penalties_proxy": before_penalties_proxy,
        "risk_free_adjusted_proxy": risk_free_adjusted_proxy,
        "constant_penalty_proxy": constant_penalty_proxy,
        "signal_strength_adjusted_proxy": signal_adjusted_proxy,
        "regime_adjusted_proxy": regime_adjusted_proxy,
        "confidence_adjusted_proxy": confidence_adjusted_proxy,
        "quality_adjusted_proxy": quality_adjusted_proxy,
        "timing_adjusted_proxy": timing_adjusted_proxy,
        "scaling_final_proxy": scaling_proxy,
        "final_expected_daily_return": final_expected_return,
    }
    for name, values in stages.items():
        out[name] = _num(pd.Series(values, index=data.index))
    for horizon in [1, 5, 10, 20, 30]:
        col = f"realized_return_{horizon}d"
        if col in data.columns:
            out[col] = _num(data[col])
    out["signal_strength"] = signal
    out["target_confidence"] = confidence
    out["quality_score"] = quality
    out["ema_timing_score"] = ema
    out["trend_persistence_score"] = trend
    out["regime_confidence"] = regime_conf
    out["cash_weight"] = cash
    return out


def _stage_attribution(stages: pd.DataFrame) -> pd.DataFrame:
    stage_cols = [
        "raw_target_return_proxy",
        "raw_daily_return_proxy",
        "before_penalties_proxy",
        "risk_free_adjusted_proxy",
        "constant_penalty_proxy",
        "signal_strength_adjusted_proxy",
        "regime_adjusted_proxy",
        "confidence_adjusted_proxy",
        "quality_adjusted_proxy",
        "timing_adjusted_proxy",
        "scaling_final_proxy",
        "final_expected_daily_return",
    ]
    final_var = float(_num(stages["final_expected_daily_return"]).var(ddof=0))
    rows: list[dict[str, object]] = []
    for stage in stage_cols:
        values = _num(stages[stage])
        mono_corr, mono_spread = _monotonicity(values, stages["realized_return_20d"])
        row = {
            "stage": stage,
            "capture_type": "exact" if stage == "final_expected_daily_return" else "proxy",
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "hit_rate_20d": _hit_rate(values, stages["realized_return_20d"]),
            "monotonicity_corr_20d": mono_corr,
            "top_bottom_spread_20d": mono_spread,
            "variance_contribution_to_final": float(values.var(ddof=0) / final_var) if final_var > 0 else np.nan,
        }
        for horizon in HORIZONS:
            target = f"realized_return_{horizon}d"
            if target in stages.columns:
                row[f"pearson_ic_{horizon}d"] = _pearson(values, stages[target])
                row[f"rank_ic_{horizon}d"] = _spearman(values, stages[target])
        rows.append(row)
    return pd.DataFrame(rows)


def _gain_loss(attribution: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = attribution["stage"].tolist()
    for prev, curr in zip(ordered[:-1], ordered[1:]):
        a = attribution[attribution["stage"].eq(prev)].iloc[0]
        b = attribution[attribution["stage"].eq(curr)].iloc[0]
        ic_delta = float(b.get("rank_ic_20d", np.nan) - a.get("rank_ic_20d", np.nan))
        hit_delta = float(b.get("hit_rate_20d", np.nan) - a.get("hit_rate_20d", np.nan))
        mono_delta = float(b.get("monotonicity_corr_20d", np.nan) - a.get("monotonicity_corr_20d", np.nan))
        std_delta = float(b.get("std", np.nan) - a.get("std", np.nan))
        rows.append(
            {
                "from_stage": prev,
                "to_stage": curr,
                "rank_ic_20d_before": a.get("rank_ic_20d", np.nan),
                "rank_ic_20d_after": b.get("rank_ic_20d", np.nan),
                "rank_ic_delta": ic_delta,
                "hit_rate_delta": hit_delta,
                "monotonicity_delta": mono_delta,
                "std_delta": std_delta,
                "improved_ic": ic_delta > 0,
                "compressed_signal": std_delta < 0,
                "added_noise_flag": ic_delta < -0.005 and abs(std_delta) > 1e-6,
            }
        )
    return pd.DataFrame(rows)


def _harmful(gain_loss: pd.DataFrame) -> pd.DataFrame:
    if gain_loss.empty:
        return pd.DataFrame()
    rows = gain_loss[
        (gain_loss["rank_ic_delta"] < -0.002)
        | (gain_loss["hit_rate_delta"] < -0.01)
        | (gain_loss["monotonicity_delta"] < -0.05)
        | (gain_loss["added_noise_flag"])
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=list(gain_loss.columns) + ["governance_classification"])
    rows["governance_classification"] = np.where(rows["rank_ic_delta"] < -0.01, "remove candidate", "review")
    rows["reason"] = rows.apply(
        lambda r: ", ".join(
            reason
            for reason, cond in [
                ("reduces_IC", r["rank_ic_delta"] < -0.002),
                ("reduces_hit_rate", r["hit_rate_delta"] < -0.01),
                ("reduces_monotonicity", r["monotonicity_delta"] < -0.05),
                ("adds_noise", bool(r["added_noise_flag"])),
            ]
            if cond
        ),
        axis=1,
    )
    return rows


def _governance(gain_loss: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in gain_loss.iterrows():
        if row["rank_ic_delta"] > 0.002 and row["hit_rate_delta"] >= -0.005:
            classification = "keep"
        elif row["rank_ic_delta"] < -0.01:
            classification = "remove candidate"
        elif row["rank_ic_delta"] < -0.002 or row["hit_rate_delta"] < -0.01:
            classification = "review"
        else:
            classification = "diagnostic only"
        rows.append(
            {
                "transformation": f"{row['from_stage']} -> {row['to_stage']}",
                "classification": classification,
                "rank_ic_delta": row["rank_ic_delta"],
                "hit_rate_delta": row["hit_rate_delta"],
                "monotonicity_delta": row["monotonicity_delta"],
                "production_change": "none",
            }
        )
    return pd.DataFrame(rows)


def run_expected_return_decomposition() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _load_dataset()
    if data.empty:
        raise ValueError("No historical expected return dataset available.")
    stages = _stage_frame(data)
    attribution = _stage_attribution(stages)
    gain_loss = _gain_loss(attribution)
    harmful = _harmful(gain_loss)
    governance = _governance(gain_loss)

    decomposition = stages.copy()
    decomposition.to_csv(DECOMPOSITION_FILE, index=False)
    attribution.to_csv(STAGE_ATTRIBUTION_FILE, index=False)
    gain_loss.to_csv(GAIN_LOSS_FILE, index=False)
    harmful.to_csv(HARMFUL_FILE, index=False)

    print("\n===== EXPECTED RETURN DECOMPOSITION =====")
    print(f"observations: {len(stages)}")
    print(
        "capture note: final expected return is exact from historical outputs; prior stages are diagnostic proxies unless exposed historically."
    )

    print("\n===== EXPECTED RETURN STAGE ATTRIBUTION =====")
    cols = [
        "stage",
        "capture_type",
        "mean",
        "std",
        "rank_ic_5d",
        "rank_ic_10d",
        "rank_ic_20d",
        "hit_rate_20d",
        "monotonicity_corr_20d",
        "variance_contribution_to_final",
    ]
    print(attribution[cols].to_string(index=False))

    print("\n===== ALPHA GAIN/LOSS BY STAGE =====")
    print(gain_loss.to_string(index=False))

    print("\n===== HARMFUL TRANSFORMATIONS =====")
    print(harmful.to_string(index=False) if not harmful.empty else "none flagged")

    print("\n===== EXPECTED RETURN GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(DECOMPOSITION_FILE).resolve()}")
    print(f"Saved: {Path(STAGE_ATTRIBUTION_FILE).resolve()}")
    print(f"Saved: {Path(GAIN_LOSS_FILE).resolve()}")
    print(f"Saved: {Path(HARMFUL_FILE).resolve()}")
    return decomposition, attribution, gain_loss, harmful


if __name__ == "__main__":
    run_expected_return_decomposition()
