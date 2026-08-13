from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_STORE_FILE = "historical_feature_store.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
TRIPLE_BARRIER_FILE = "historical_triple_barrier_labels.csv"

REGIME_HISTORY_FILE = "regime_v2_history.csv"
REGIME_DAILY_FILE = "regime_v2_daily_state.csv"
PERFORMANCE_FILE = "regime_v2_performance_attribution.csv"
TRANSITION_FILE = "regime_v2_transition_matrix.csv"
COMPARISON_FILE = "regime_v2_comparison_vs_old.csv"
TRADING_DAYS = 252


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


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _rank_pct(series: pd.Series) -> pd.Series:
    values = _num(series)
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return values.rank(pct=True).fillna(0.5)


def _safe_sharpe(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if len(returns) < 2:
        return np.nan
    vol = float(returns.std(ddof=0))
    if vol <= 0:
        return np.nan
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS))


def _max_drawdown(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _prepare_snapshots() -> pd.DataFrame:
    snapshots = _read_csv(SNAPSHOTS_FILE)
    realized = _read_csv(REALIZED_FILE)
    if snapshots.empty:
        return snapshots
    snapshots = snapshots.copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce").dt.normalize()
    snapshots = snapshots.dropna(subset=["date", "ticker"])
    if not realized.empty and {"date", "ticker", "model_mode"}.issubset(realized.columns):
        realized = realized.copy()
        realized["date"] = pd.to_datetime(realized["date"], errors="coerce").dt.normalize()
        realized_cols = [
            "realized_return_1d",
            "realized_return_5d",
            "realized_return_10d",
            "realized_return_20d",
            "realized_return_30d",
        ]
        available = [col for col in realized_cols if col in realized.columns]
        snapshots = snapshots.drop(columns=[col for col in available if col in snapshots.columns], errors="ignore")
        snapshots = snapshots.merge(
            realized[["date", "ticker", "model_mode"] + available],
            on=["date", "ticker", "model_mode"],
            how="left",
        )
    for col in [
        "current_price",
        "expected_daily_return",
        "expected_total_return",
        "signal_strength",
        "target_confidence",
        "quality_score",
        "weight",
        "cash_weight",
        "realized_return_1d",
        "realized_return_5d",
        "realized_return_10d",
        "realized_return_20d",
    ]:
        if col in snapshots.columns:
            snapshots[col] = _num(snapshots[col])
    return snapshots


def _market_proxy(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    base = snapshots[snapshots["model_mode"].astype(str).eq("baseline")].copy()
    if base.empty:
        base = snapshots.copy()
    pivot = base.pivot_table(index="date", columns="ticker", values="current_price", aggfunc="last").sort_index()
    market_return_1d = pivot.pct_change().mean(axis=1)
    rolling_20 = market_return_1d.rolling(20, min_periods=5)
    rolling_60 = market_return_1d.rolling(60, min_periods=20)
    features = pd.DataFrame(index=pivot.index)
    features["market_return_1d"] = market_return_1d
    features["market_return_20d"] = market_return_1d.rolling(20, min_periods=5).sum()
    features["market_return_60d"] = market_return_1d.rolling(60, min_periods=20).sum()
    features["realized_vol_20d"] = rolling_20.std()
    features["realized_vol_60d"] = rolling_60.std()
    features["risk_adjusted_momentum_20d"] = features["market_return_20d"] / (features["realized_vol_20d"] * np.sqrt(20) + 1e-8)
    features["risk_adjusted_momentum_60d"] = features["market_return_60d"] / (features["realized_vol_60d"] * np.sqrt(60) + 1e-8)
    features["trend_slope_20d"] = market_return_1d.rolling(20, min_periods=5).mean()
    features["positive_momentum_share"] = (pivot.pct_change(20) > 0).mean(axis=1)
    features["positive_60d_share"] = (pivot.pct_change(60) > 0).mean(axis=1)
    features["cross_sectional_return"] = pivot.pct_change(20).mean(axis=1)
    features["cross_sectional_dispersion"] = pivot.pct_change(20).std(axis=1)
    returns_20 = pivot.pct_change(20)
    features["average_pairwise_correlation"] = np.nan
    features["pc1_variance_share"] = np.nan
    for idx in range(len(pivot)):
        if idx < 60:
            continue
        window = pivot.iloc[max(0, idx - 60): idx + 1].pct_change().dropna(how="all")
        if window.shape[0] < 20 or window.shape[1] < 3:
            continue
        corr = window.corr().replace([np.inf, -np.inf], np.nan)
        mask = ~np.eye(len(corr), dtype=bool)
        features.iloc[idx, features.columns.get_loc("average_pairwise_correlation")] = float(np.nanmean(corr.to_numpy()[mask]))
        cov = window.fillna(0.0).cov()
        eigvals = np.linalg.eigvalsh(cov.to_numpy())
        eig_sum = float(np.sum(eigvals))
        features.iloc[idx, features.columns.get_loc("pc1_variance_share")] = float(np.max(eigvals) / eig_sum) if eig_sum > 0 else np.nan
    features["vol_percentile"] = _rank_pct(features["realized_vol_20d"])
    features["dispersion_percentile"] = _rank_pct(features["cross_sectional_dispersion"])
    return features.replace([np.inf, -np.inf], np.nan)


def _classify_states(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    out = features.copy()
    trend_raw = 0.45 * out["risk_adjusted_momentum_20d"].fillna(0.0) + 0.35 * out["risk_adjusted_momentum_60d"].fillna(0.0) + 20.0 * out["trend_slope_20d"].fillna(0.0)
    out["trend_score"] = np.tanh(trend_raw / 2.0)
    out["trend_state"] = np.where(out["trend_score"] > 0.20, "uptrend", np.where(out["trend_score"] < -0.20, "downtrend", "sideways"))

    vol_pct = out["vol_percentile"].fillna(0.5)
    out["volatility_state"] = np.select(
        [vol_pct < 0.25, vol_pct < 0.65, vol_pct < 0.85],
        ["low_vol", "normal_vol", "high_vol"],
        default="stress_vol",
    )

    breadth_score = (
        0.45 * out["positive_momentum_share"].fillna(0.5)
        + 0.35 * out["positive_60d_share"].fillna(0.5)
        + 0.20 * _rank_pct(out["cross_sectional_return"]).fillna(0.5)
    )
    out["breadth_score"] = breadth_score.clip(0.0, 1.0)
    out["breadth_state"] = np.select(
        [out["breadth_score"] > 0.62, out["breadth_score"] < 0.42],
        ["broad_participation", "weak_participation"],
        default="narrow_participation",
    )

    corr = out["average_pairwise_correlation"].fillna(out["average_pairwise_correlation"].median()).fillna(0.30)
    pc1 = out["pc1_variance_share"].fillna(out["pc1_variance_share"].median()).fillna(0.35)
    dispersion = out["dispersion_percentile"].fillna(0.5)
    crowding_score = (0.45 * corr.clip(-1, 1) + 0.40 * pc1.clip(0, 1) + 0.15 * (1.0 - dispersion)).clip(0.0, 1.0)
    out["crowding_score"] = crowding_score
    out["correlation_state"] = np.select(
        [(crowding_score > 0.68) & (vol_pct > 0.70), crowding_score > 0.55],
        ["crowded_stress", "correlated"],
        default="diversified",
    )

    labels: list[str] = []
    reasons: list[str] = []
    scores: list[float] = []
    confidences: list[float] = []
    for _, row in out.iterrows():
        trend = str(row["trend_state"])
        vol = str(row["volatility_state"])
        breadth = str(row["breadth_state"])
        corr_state = str(row["correlation_state"])
        if vol == "stress_vol" or corr_state == "crowded_stress":
            label = "stress"
        elif trend == "uptrend" and vol in {"low_vol", "normal_vol"} and breadth == "broad_participation":
            label = "trend_up_low_vol"
        elif trend == "uptrend" and vol in {"high_vol", "stress_vol"}:
            label = "trend_up_high_vol"
        elif trend == "downtrend" and vol in {"high_vol", "stress_vol"}:
            label = "trend_down_high_vol"
        elif trend == "downtrend" and breadth != "weak_participation":
            label = "recovery"
        elif trend == "sideways" and vol in {"low_vol", "normal_vol"}:
            label = "sideways_low_vol"
        elif trend == "sideways" and vol in {"high_vol", "stress_vol"}:
            label = "sideways_high_vol"
        else:
            label = "choppy"
        label_value = {
            "trend_up_low_vol": 0.8,
            "trend_up_high_vol": 0.45,
            "recovery": 0.35,
            "sideways_low_vol": 0.1,
            "sideways_high_vol": -0.2,
            "choppy": -0.1,
            "trend_down_high_vol": -0.65,
            "stress": -0.85,
        }.get(label, 0.0)
        confidence = min(
            1.0,
            max(
                0.0,
                abs(float(row["trend_score"])) * 0.35
                + abs(float(row["breadth_score"]) - 0.5) * 0.70
                + abs(float(row["vol_percentile"]) - 0.5) * 0.45
                + abs(float(row["crowding_score"]) - 0.5) * 0.35,
            ),
        )
        labels.append(label)
        scores.append(label_value)
        confidences.append(max(0.25, confidence))
        reasons.append(f"trend={trend}; vol={vol}; breadth={breadth}; corr={corr_state}")
    out["regime_v2_label"] = labels
    out["regime_v2_score"] = scores
    out["regime_v2_confidence"] = confidences
    out["regime_v2_reason"] = reasons
    return out


def _merge_regime_v2(snapshots: pd.DataFrame, daily_state: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or daily_state.empty:
        return snapshots
    cols = ["regime_v2_label", "regime_v2_score", "regime_v2_confidence", "regime_v2_reason"]
    state = daily_state.reset_index().rename(columns={"index": "date"})
    if "date" not in state.columns:
        state = daily_state.reset_index().rename(columns={daily_state.index.name or "index": "date"})
    state["date"] = pd.to_datetime(state["date"], errors="coerce").dt.normalize()
    return snapshots.merge(state[["date"] + cols], on="date", how="left")


def _transition_matrix(labels: pd.Series) -> pd.DataFrame:
    transitions = pd.DataFrame({"from": labels.shift(1), "to": labels}).dropna()
    if transitions.empty:
        return pd.DataFrame()
    return pd.crosstab(transitions["from"], transitions["to"], normalize="index")


def _performance_by_regime(data: pd.DataFrame, label_col: str, labels: pd.DataFrame) -> pd.DataFrame:
    if data.empty or label_col not in data.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    selected = data[_bool(data.get("selected", pd.Series(False, index=data.index)))].copy()
    source = selected if not selected.empty else data
    for label, group in source.groupby(label_col):
        returns = _num(group.get("realized_return_20d", pd.Series(index=group.index)))
        label_group = pd.DataFrame()
        if not labels.empty and label_col in labels.columns:
            label_group = labels[
                labels[label_col].astype(str).eq(str(label))
                & labels["horizon"].astype(str).eq("20")
                & _bool(labels["selected"])
            ]
        tp = float((label_group["first_touch_type"].astype(str) == "take_profit").mean()) if not label_group.empty else np.nan
        sl = float((label_group["first_touch_type"].astype(str) == "stop_loss").mean()) if not label_group.empty else np.nan
        forecast = _num(group.get("expected_total_return", group.get("expected_daily_return", pd.Series(index=group.index))))
        rows.append(
            {
                "regime_label": label,
                "sample_size": int(returns.notna().sum()),
                "average_forward_return": float(returns.mean()) if returns.notna().any() else np.nan,
                "volatility": float(returns.std(ddof=0)) if returns.notna().sum() > 1 else np.nan,
                "Sharpe": _safe_sharpe(returns),
                "TP_rate": tp,
                "SL_rate": sl,
                "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan,
                "hit_rate": float((returns > 0).mean()) if returns.notna().any() else np.nan,
                "forecast_IC": _spearman(forecast, returns),
                "average_selected_count": float(data[data[label_col].astype(str).eq(str(label))].groupby("date")["selected"].apply(lambda s: _bool(s).sum()).mean()),
                "average_cash": float(_num(data[data[label_col].astype(str).eq(str(label))].get("cash_weight", pd.Series(dtype=float))).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_size", ascending=False)


def _attach_labels(labels: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if labels.empty or data.empty:
        return labels
    keys = ["date", "ticker", "model_mode"]
    if not set(keys).issubset(labels.columns) or not set(keys).issubset(data.columns):
        return labels
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    mapping_cols = keys + ["regime", "regime_v2_label"]
    mapping = data[mapping_cols].drop_duplicates(keys, keep="last")
    return labels.merge(mapping, on=keys, how="left")


def _comparison(old_perf: pd.DataFrame, new_perf: pd.DataFrame) -> pd.DataFrame:
    def summarize(frame: pd.DataFrame, system: str) -> dict[str, object]:
        if frame.empty:
            return {"system": system}
        tiny = int((frame["sample_size"] < 30).sum()) if "sample_size" in frame.columns else 0
        weighted_return = np.average(frame["average_forward_return"].fillna(0.0), weights=frame["sample_size"].clip(lower=1))
        weighted_sharpe = np.average(frame["Sharpe"].fillna(0.0), weights=frame["sample_size"].clip(lower=1))
        return {
            "system": system,
            "regime_count": int(len(frame)),
            "tiny_regime_count": tiny,
            "largest_regime_share": float(frame["sample_size"].max() / max(1, frame["sample_size"].sum())),
            "weighted_average_return": float(weighted_return),
            "weighted_sharpe": float(weighted_sharpe),
            "mean_abs_forecast_IC": float(frame["forecast_IC"].abs().mean(skipna=True)),
            "mean_TP_minus_SL": float(frame["TP_minus_SL"].mean(skipna=True)),
        }
    return pd.DataFrame([summarize(old_perf, "current_regime"), summarize(new_perf, "regime_v2")])


def _governance(new_perf: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    if new_perf.empty:
        classification = "rejected"
        reason = "no_v2_performance_data"
    else:
        largest_share = float(new_perf["sample_size"].max() / max(1, new_perf["sample_size"].sum()))
        tiny_count = int((new_perf["sample_size"] < 30).sum())
        useful_ic = float(new_perf["forecast_IC"].abs().mean(skipna=True)) > 0.03
        tp_ok = float(new_perf["TP_minus_SL"].mean(skipna=True)) > 0
        if largest_share < 0.65 and tiny_count <= max(1, len(new_perf) // 3) and useful_ic and tp_ok:
            classification = "candidate for regime gate replacement"
            reason = "more_balanced_with_positive_signal_quality"
        elif largest_share < 0.75 and (useful_ic or tp_ok):
            classification = "useful for research"
            reason = "improves_balance_but_needs_more_validation"
        elif largest_share < 0.85:
            classification = "diagnostic only"
            reason = "better_distribution_but_weak_predictive_evidence"
        else:
            classification = "rejected"
            reason = "still_dominated_by_large_bucket_or_tiny_regimes"
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "reason": reason,
                "production_change": "none",
                "next_step": "walk-forward regime gate comparison before any promotion",
            }
        ]
    )


def run_regime_engine_v2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshots = _prepare_snapshots()
    labels = _read_csv(TRIPLE_BARRIER_FILE)
    if snapshots.empty:
        raise ValueError("historical_forecast_snapshots.csv is required.")
    market_features = _market_proxy(snapshots)
    daily_state = _classify_states(market_features)
    enriched = _merge_regime_v2(snapshots, daily_state)
    labels = _attach_labels(labels, enriched)

    old_perf = _performance_by_regime(enriched, "regime", labels)
    new_perf = _performance_by_regime(enriched, "regime_v2_label", labels)
    old_perf = old_perf.rename(columns={"regime_label": "regime"}).assign(regime_system="current")
    new_perf = new_perf.rename(columns={"regime_label": "regime"}).assign(regime_system="v2")
    comparison = _comparison(old_perf, new_perf)
    governance = _governance(new_perf, comparison)
    transition = _transition_matrix(daily_state["regime_v2_label"]) if "regime_v2_label" in daily_state else pd.DataFrame()

    history_cols = [
        "trend_state",
        "volatility_state",
        "breadth_state",
        "correlation_state",
        "regime_v2_label",
        "regime_v2_score",
        "regime_v2_confidence",
        "regime_v2_reason",
        "trend_score",
        "breadth_score",
        "crowding_score",
        "vol_percentile",
    ]
    daily_state[history_cols].to_csv(REGIME_HISTORY_FILE)
    daily_state[history_cols].to_csv(REGIME_DAILY_FILE)
    pd.concat([old_perf, new_perf], ignore_index=True).to_csv(PERFORMANCE_FILE, index=False)
    transition.to_csv(TRANSITION_FILE)
    comparison.to_csv(COMPARISON_FILE, index=False)

    print("\n===== REGIME ENGINE V2 =====")
    print(f"dates: {len(daily_state)}")
    print(f"tickers in snapshots: {snapshots['ticker'].nunique()}")
    print(f"production change: none")

    print("\n===== REGIME V2 DISTRIBUTION =====")
    distribution = daily_state["regime_v2_label"].value_counts(dropna=False)
    print(distribution.to_string())

    print("\n===== REGIME V2 TRANSITION MATRIX =====")
    print(transition.round(4).to_string() if not transition.empty else "insufficient data")

    print("\n===== REGIME V2 PERFORMANCE ATTRIBUTION =====")
    cols = ["regime", "sample_size", "average_forward_return", "volatility", "Sharpe", "TP_rate", "SL_rate", "TP_minus_SL", "hit_rate", "forecast_IC", "average_cash"]
    print(new_perf[cols].to_string(index=False))

    print("\n===== REGIME V2 GOVERNANCE =====")
    print(governance.to_string(index=False))

    print("\n===== REGIME V2 VS OLD =====")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {Path(REGIME_HISTORY_FILE).resolve()}")
    print(f"Saved: {Path(REGIME_DAILY_FILE).resolve()}")
    print(f"Saved: {Path(PERFORMANCE_FILE).resolve()}")
    print(f"Saved: {Path(TRANSITION_FILE).resolve()}")
    print(f"Saved: {Path(COMPARISON_FILE).resolve()}")
    return daily_state, new_perf, comparison


if __name__ == "__main__":
    run_regime_engine_v2()
