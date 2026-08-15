from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_STORE_FILE = "historical_feature_store.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"
REGIME_ATTRIBUTION_FILE = "regime_performance_attribution.csv"
TRIPLE_BARRIER_FILE = "historical_triple_barrier_labels.csv"

AUDIT_FILE = "regime_v2_audit.csv"
TRANSITION_FILE = "regime_transition_matrix.csv"
FORECAST_QUALITY_FILE = "regime_forecast_quality.csv"
ASSET_BEHAVIOR_FILE = "regime_asset_behavior.csv"
TRADING_DAYS = 252


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _safe_numeric(x), "y": _safe_numeric(y)}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _max_drawdown(returns: pd.Series) -> float:
    returns = _safe_numeric(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _sortino(returns: pd.Series) -> float:
    returns = _safe_numeric(returns).dropna()
    downside = returns[returns < 0]
    if returns.empty or len(downside) < 2:
        return np.nan
    downside_std = float(downside.std(ddof=0))
    if downside_std <= 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (downside_std * np.sqrt(TRADING_DAYS)))


def _sharpe(returns: pd.Series) -> float:
    returns = _safe_numeric(returns).dropna()
    if len(returns) < 2:
        return np.nan
    vol = float(returns.std(ddof=0))
    if vol <= 0:
        return np.nan
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS))


def _effect_size(a: pd.Series, b: pd.Series) -> float:
    a = _safe_numeric(a).dropna()
    b = _safe_numeric(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(1, len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def _t_stat(a: pd.Series, b: pd.Series) -> float:
    a = _safe_numeric(a).dropna()
    b = _safe_numeric(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    denom = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / denom)


def _date_regime_series(frame: pd.DataFrame, model_mode: str = "baseline") -> pd.Series:
    if frame.empty or not {"date", "regime"}.issubset(frame.columns):
        return pd.Series(dtype=object)
    data = frame.copy()
    if "model_mode" in data.columns and model_mode in set(data["model_mode"].astype(str)):
        data = data[data["model_mode"].astype(str).eq(model_mode)]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    regimes = (
        data.groupby("date")["regime"]
        .agg(lambda x: x.astype(str).mode().iloc[0] if not x.astype(str).mode().empty else str(x.iloc[0]))
        .sort_index()
    )
    regimes = regimes[~regimes.astype(str).str.lower().isin({"nan", "none", ""})]
    return regimes


def _regime_stability(regimes: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if regimes.empty:
        return pd.DataFrame(), pd.DataFrame(), {"average_duration": np.nan, "median_duration": np.nan, "stay_probability": np.nan, "switch_probability": np.nan}
    runs: list[dict[str, object]] = []
    current = str(regimes.iloc[0])
    start = regimes.index[0]
    previous_date = regimes.index[0]
    length = 1
    for date, value in regimes.iloc[1:].items():
        value = str(value)
        if value == current:
            length += 1
            previous_date = date
            continue
        runs.append({"regime": current, "start": start, "end": previous_date, "duration": length})
        current = value
        start = date
        length = 1
        previous_date = date
    previous_date = regimes.index[-1]
    if runs:
        last_start = start
    else:
        last_start = regimes.index[0]
    runs.append({"regime": current, "start": last_start, "end": regimes.index[-1], "duration": length})
    runs_df = pd.DataFrame(runs)

    transitions = pd.DataFrame({"from": regimes.shift(1), "to": regimes}).dropna()
    matrix = pd.crosstab(transitions["from"], transitions["to"], normalize="index") if not transitions.empty else pd.DataFrame()
    stay_prob = float((transitions["from"] == transitions["to"]).mean()) if not transitions.empty else np.nan
    stats = {
        "average_duration": float(runs_df["duration"].mean()) if not runs_df.empty else np.nan,
        "median_duration": float(runs_df["duration"].median()) if not runs_df.empty else np.nan,
        "stay_probability": stay_prob,
        "switch_probability": 1.0 - stay_prob if np.isfinite(stay_prob) else np.nan,
    }
    return runs_df, matrix, stats


def _regime_persistence(snapshots: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or "regime" not in snapshots.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    data = snapshots.copy()
    data["realized_return_20d"] = _safe_numeric(data.get("realized_return_20d", pd.Series(index=data.index)))
    selected = data[_bool_series(data.get("selected", pd.Series(False, index=data.index)))]
    for regime, group in data.groupby("regime"):
        selected_group = selected[selected["regime"].astype(str).eq(str(regime))]
        returns = selected_group["realized_return_20d"] if not selected_group.empty else group["realized_return_20d"]
        label_group = pd.DataFrame()
        if not labels.empty and {"regime", "horizon", "selected"}.issubset(labels.columns):
            label_group = labels[
                labels["regime"].astype(str).eq(str(regime))
                & labels["horizon"].astype(str).eq("20")
                & _bool_series(labels["selected"])
            ]
        tp = float((label_group["first_touch_type"].astype(str) == "take_profit").mean()) if not label_group.empty else np.nan
        sl = float((label_group["first_touch_type"].astype(str) == "stop_loss").mean()) if not label_group.empty else np.nan
        rows.append(
            {
                "section": "persistence",
                "regime": regime,
                "observations": len(returns.dropna()),
                "average_forward_return_20d": float(returns.mean()) if returns.notna().any() else np.nan,
                "average_volatility": float(returns.std(ddof=0)) if returns.notna().sum() > 1 else np.nan,
                "Sharpe": _sharpe(returns),
                "Sortino": _sortino(returns),
                "max_drawdown": _max_drawdown(returns),
                "hit_rate": float((_safe_numeric(returns) > 0).mean()) if returns.notna().any() else np.nan,
                "TP_rate": tp,
                "SL_rate": sl,
            }
        )
    return pd.DataFrame(rows)


def _forecast_quality(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or "regime" not in snapshots.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    data = snapshots.copy()
    data["forecast"] = _safe_numeric(data.get("expected_total_return", data.get("expected_daily_return", pd.Series(index=data.index))))
    data["realized"] = _safe_numeric(data.get("realized_return_20d", pd.Series(index=data.index)))
    data["forecast_direction"] = np.sign(data["forecast"])
    data["realized_direction"] = np.sign(data["realized"])
    data["error"] = data["forecast"] - data["realized"]
    for regime, group in data.groupby("regime"):
        valid = group.dropna(subset=["forecast", "realized"])
        if valid.empty:
            continue
        rows.append(
            {
                "regime": regime,
                "observations": len(valid),
                "MAE": float(valid["error"].abs().mean()),
                "RMSE": float(np.sqrt(np.mean(np.square(valid["error"])))),
                "bias": float(valid["error"].mean()),
                "IC": _spearman(valid["forecast"], valid["realized"]),
                "hit_rate": float((valid["realized"] > 0).mean()),
                "direction_accuracy": float((valid["forecast_direction"] == valid["realized_direction"]).mean()),
                "avg_signal_strength": float(_safe_numeric(valid.get("signal_strength", pd.Series(index=valid.index))).mean()),
                "avg_target_confidence": float(_safe_numeric(valid.get("target_confidence", pd.Series(index=valid.index))).mean()),
                "avg_quality_score": float(_safe_numeric(valid.get("quality_score", pd.Series(index=valid.index))).mean()),
            }
        )
    return pd.DataFrame(rows)


def _separability(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or "regime" not in snapshots.columns:
        return pd.DataFrame()
    data = snapshots.copy()
    metrics = {
        "returns": _safe_numeric(data.get("realized_return_20d", pd.Series(index=data.index))),
        "forecast_error": _safe_numeric(data.get("expected_total_return", pd.Series(index=data.index))) - _safe_numeric(data.get("realized_return_20d", pd.Series(index=data.index))),
        "signal_strength": _safe_numeric(data.get("signal_strength", pd.Series(index=data.index))),
        "quality_score": _safe_numeric(data.get("quality_score", pd.Series(index=data.index))),
    }
    data = data.assign(**metrics)
    rows: list[dict[str, object]] = []
    regimes = sorted(data["regime"].dropna().astype(str).unique())
    for left, right in combinations(regimes, 2):
        left_data = data[data["regime"].astype(str).eq(left)]
        right_data = data[data["regime"].astype(str).eq(right)]
        for metric in metrics:
            a = left_data[metric]
            b = right_data[metric]
            rows.append(
                {
                    "section": "separability",
                    "comparison": f"{left}_vs_{right}",
                    "metric": metric,
                    "effect_size": _effect_size(a, b),
                    "t_stat": _t_stat(a, b),
                    "left_mean": float(a.mean()) if a.notna().any() else np.nan,
                    "right_mean": float(b.mean()) if b.notna().any() else np.nan,
                    "overlap_warning": abs(_effect_size(a, b)) < 0.20 if np.isfinite(_effect_size(a, b)) else True,
                }
            )
    return pd.DataFrame(rows)


def _asset_behavior(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty or not {"ticker", "regime"}.issubset(snapshots.columns):
        return pd.DataFrame()
    data = snapshots.copy()
    data["realized_return_20d"] = _safe_numeric(data.get("realized_return_20d", pd.Series(index=data.index)))
    selected = data[_bool_series(data.get("selected", pd.Series(False, index=data.index)))]
    source = selected if not selected.empty else data
    grouped = (
        source.groupby(["regime", "ticker"], dropna=False)
        .agg(
            observations=("realized_return_20d", "count"),
            avg_realized_return_20d=("realized_return_20d", "mean"),
            hit_rate=("realized_return_20d", lambda x: float((_safe_numeric(x) > 0).mean())),
            avg_expected_total_return=("expected_total_return", "mean"),
            avg_signal_strength=("signal_strength", "mean"),
        )
        .reset_index()
    )
    grouped["rank_in_regime"] = grouped.groupby("regime")["avg_realized_return_20d"].rank(ascending=False, method="dense")
    grouped["consistent_failure"] = (grouped["avg_realized_return_20d"] < 0) & (grouped["hit_rate"] < 0.45) & (grouped["observations"] >= 3)
    return grouped.sort_values(["regime", "rank_in_regime", "ticker"])


def _attach_regime_to_labels(labels: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    if labels.empty or snapshots.empty or "regime" in labels.columns:
        return labels
    keys = ["date", "ticker", "model_mode"]
    if not set(keys).issubset(labels.columns) or not set(keys + ["regime"]).issubset(snapshots.columns):
        return labels
    regime_map = (
        snapshots[keys + ["regime"]]
        .dropna(subset=["regime"])
        .drop_duplicates(subset=keys, keep="last")
    )
    return labels.merge(regime_map, on=keys, how="left")


def _confusion_analysis(snapshots: pd.DataFrame, separability: pd.DataFrame, stability: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if snapshots.empty or "regime" not in snapshots.columns:
        return pd.DataFrame()
    regime_counts = snapshots.groupby("regime")["date"].nunique().sort_values(ascending=False)
    total_dates = max(1, int(snapshots["date"].nunique()))
    neutral_share = float(regime_counts.get("neutral", 0) / total_dates)
    risk_on_neutral = separability[
        separability.get("comparison", pd.Series(dtype=str)).astype(str).isin({"neutral_vs_risk_on", "risk_on_vs_neutral"})
    ]
    avg_abs_effect = float(risk_on_neutral["effect_size"].abs().mean()) if not risk_on_neutral.empty else np.nan
    rows.append(
        {
            "section": "confusion",
            "metric": "neutral_share",
            "value": neutral_share,
            "interpretation": "catch_all_risk" if neutral_share > 0.55 else "not_dominant",
        }
    )
    rows.append(
        {
            "section": "confusion",
            "metric": "risk_on_neutral_avg_abs_effect_size",
            "value": avg_abs_effect,
            "interpretation": "not_separable" if not np.isfinite(avg_abs_effect) or avg_abs_effect < 0.20 else "separable",
        }
    )
    rows.append(
        {
            "section": "confusion",
            "metric": "switch_probability",
            "value": stability.get("switch_probability", np.nan),
            "interpretation": "unstable" if _safe_numeric(pd.Series([stability.get("switch_probability", np.nan)])).iloc[0] > 0.35 else "stable",
        }
    )
    return pd.DataFrame(rows)


def _governance(stability: dict[str, float], persistence: pd.DataFrame, forecast_quality: pd.DataFrame, separability: pd.DataFrame, confusion: pd.DataFrame) -> pd.DataFrame:
    weak_reasons: list[str] = []
    stay_prob = stability.get("stay_probability", np.nan)
    if not np.isfinite(stay_prob) or stay_prob < 0.60:
        weak_reasons.append("low_regime_stability")
    if not separability.empty and float(separability["effect_size"].abs().mean(skipna=True)) < 0.20:
        weak_reasons.append("weak_regime_separability")
    if not forecast_quality.empty and forecast_quality["IC"].abs().mean(skipna=True) < 0.03:
        weak_reasons.append("weak_forecast_quality_by_regime")
    if not confusion.empty and "not_separable" in set(confusion["interpretation"].astype(str)):
        weak_reasons.append("risk_on_neutral_confusion")
    if not persistence.empty:
        bad_regimes = persistence[(persistence["Sharpe"] < 0) | (persistence["hit_rate"] < 0.50)]
        if len(bad_regimes) >= max(1, len(persistence) // 2):
            weak_reasons.append("poor_regime_persistence")
    if len(weak_reasons) >= 3:
        classification = "current regime engine requires redesign"
    elif weak_reasons:
        classification = "current regime engine weak"
    else:
        classification = "current regime engine adequate"
    next_architecture = (
        "test continuous regime score + volatility/trend/noise clusters before HMM"
        if classification != "current regime engine adequate"
        else "keep current regime engine, monitor with larger sample"
    )
    return pd.DataFrame(
        [
            {
                "section": "governance",
                "classification": classification,
                "reasons": ", ".join(weak_reasons) if weak_reasons else "none",
                "recommended_next_architecture": next_architecture,
                "production_change": "none",
            }
        ]
    )


def run_regime_engine_v2_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_store = _read_csv(FEATURE_STORE_FILE)
    snapshots = _read_csv(SNAPSHOTS_FILE)
    _ = _read_csv(REALIZED_FILE)
    _ = _read_csv(PORTFOLIO_FILE)
    attribution = _read_csv(REGIME_ATTRIBUTION_FILE)
    labels = _read_csv(TRIPLE_BARRIER_FILE)
    labels = _attach_regime_to_labels(labels, snapshots)

    source = snapshots if not snapshots.empty else feature_store
    regimes = _date_regime_series(source, model_mode="baseline")
    runs, transition_matrix, stability = _regime_stability(regimes)
    persistence = _regime_persistence(snapshots, labels)
    forecast_quality = _forecast_quality(snapshots)
    separability = _separability(snapshots)
    asset_behavior = _asset_behavior(snapshots)
    confusion = _confusion_analysis(snapshots, separability, stability)
    governance = _governance(stability, persistence, forecast_quality, separability, confusion)

    stability_rows = pd.DataFrame(
        [
            {"section": "stability", "metric": key, "value": value}
            for key, value in stability.items()
        ]
    )
    if not runs.empty:
        duration_by_regime = runs.groupby("regime")["duration"].agg(["count", "mean", "median"]).reset_index()
        duration_by_regime["section"] = "duration_by_regime"
    else:
        duration_by_regime = pd.DataFrame()

    audit = pd.concat(
        [
            stability_rows,
            duration_by_regime,
            persistence,
            separability,
            confusion,
            governance,
            attribution.assign(section="existing_regime_attribution") if not attribution.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )

    audit.to_csv(AUDIT_FILE, index=False)
    transition_matrix.to_csv(TRANSITION_FILE)
    forecast_quality.to_csv(FORECAST_QUALITY_FILE, index=False)
    asset_behavior.to_csv(ASSET_BEHAVIOR_FILE, index=False)

    print("\n===== REGIME ENGINE V2 AUDIT =====")
    print(f"feature rows: {len(feature_store)}")
    print(f"snapshot rows: {len(snapshots)}")
    print(f"regime dates: {len(regimes)}")
    print(f"regimes: {sorted(regimes.dropna().astype(str).unique()) if not regimes.empty else []}")

    print("\n===== REGIME STABILITY =====")
    for key, value in stability.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) and np.isfinite(value) else f"{key}: {value}")
    if not transition_matrix.empty:
        print(transition_matrix.round(4).to_string())

    print("\n===== REGIME PERSISTENCE =====")
    if persistence.empty:
        print("insufficient data")
    else:
        cols = ["regime", "observations", "average_forward_return_20d", "average_volatility", "Sharpe", "Sortino", "max_drawdown", "hit_rate", "TP_rate", "SL_rate"]
        print(persistence[cols].to_string(index=False))

    print("\n===== FORECAST QUALITY BY REGIME =====")
    if forecast_quality.empty:
        print("insufficient data")
    else:
        print(forecast_quality.to_string(index=False))

    print("\n===== REGIME CONFUSION ANALYSIS =====")
    if confusion.empty:
        print("insufficient data")
    else:
        print(confusion.to_string(index=False))

    print("\n===== REGIME GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(AUDIT_FILE).resolve()}")
    print(f"Saved: {Path(TRANSITION_FILE).resolve()}")
    print(f"Saved: {Path(FORECAST_QUALITY_FILE).resolve()}")
    print(f"Saved: {Path(ASSET_BEHAVIOR_FILE).resolve()}")
    return audit, transition_matrix, forecast_quality, asset_behavior


if __name__ == "__main__":
    run_regime_engine_v2_audit()
