from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

KALMAN_FEATURES = [
    "kalman_price",
    "kalman_trend",
    "kalman_trend_slope",
    "kalman_momentum",
    "kalman_volatility_state",
    "kalman_noise_state",
    "kalman_residual",
    "kalman_signal_to_noise",
]

OUTPUT_FEATURES = "kalman_state_features.csv"
OUTPUT_VALIDATION = "kalman_feature_validation.csv"
OUTPUT_META_COMPARISON = "kalman_meta_label_comparison.csv"


@dataclass
class KalmanStateSpaceConfig:
    price_source: str = "historical_forecast_snapshots.csv"
    realized_returns_source: str = "historical_realized_returns.csv"
    ic_dataset_source: str = "historical_ic_dataset.csv"
    meta_label_dataset_source: str = "meta_label_dataset.csv"
    meta_feature_ranking_source: str = "meta_feature_ranking.csv"
    horizon: int = 20
    test_size: float = 0.30


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    return df


def _safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _spearman_no_scipy(feature: pd.Series, target: pd.Series) -> float:
    frame = pd.DataFrame({"feature": pd.to_numeric(feature, errors="coerce"), "target": pd.to_numeric(target, errors="coerce")}).dropna()
    if frame.empty or frame["feature"].nunique() < 2 or frame["target"].nunique() < 2:
        return 0.0
    corr = frame["feature"].rank().corr(frame["target"].rank(), method="pearson")
    return float(corr) if np.isfinite(corr) else 0.0


def _manual_mutual_information(feature: pd.Series, target: pd.Series, bins: int = 5) -> float:
    frame = pd.DataFrame({"feature": pd.to_numeric(feature, errors="coerce"), "target": pd.to_numeric(target, errors="coerce")}).dropna()
    if frame.empty or frame["feature"].nunique() < 2 or frame["target"].nunique() < 2:
        return 0.0
    try:
        feature_bins = pd.qcut(frame["feature"], q=min(bins, frame["feature"].nunique()), labels=False, duplicates="drop")
    except ValueError:
        return 0.0
    contingency = pd.crosstab(feature_bins, frame["target"]).astype(float)
    total = float(contingency.values.sum())
    if total <= 0:
        return 0.0
    pxy = contingency / total
    px = pxy.sum(axis=1).values
    py = pxy.sum(axis=0).values
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            value = float(pxy.iloc[i, j])
            if value > 0 and px[i] > 0 and py[j] > 0:
                mi += value * np.log(value / (px[i] * py[j]))
    return float(mi) if np.isfinite(mi) else 0.0


def _manual_auc(y_true: pd.Series, probabilities: pd.Series) -> float:
    frame = pd.DataFrame({"y": pd.to_numeric(y_true, errors="coerce"), "p": pd.to_numeric(probabilities, errors="coerce")}).dropna()
    if frame.empty or frame["y"].nunique() < 2:
        return np.nan
    ranks = frame["p"].rank(method="average")
    positives = frame["y"].eq(1)
    n_pos = float(positives.sum())
    n_neg = float((~positives).sum())
    if n_pos <= 0 or n_neg <= 0:
        return np.nan
    auc = (ranks[positives].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc) if np.isfinite(auc) else np.nan


def _fit_numpy_logistic(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0).replace(0, 1.0)
    train = ((x_train - mean) / std).fillna(0.0).to_numpy(dtype=float)
    test = ((x_test - mean) / std).fillna(0.0).to_numpy(dtype=float)
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    y = y_train.to_numpy(dtype=float)
    weights = np.zeros(train.shape[1], dtype=float)
    for _ in range(600):
        pred = _sigmoid(train @ weights)
        grad = train.T @ (pred - y) / max(len(y), 1)
        grad[1:] += 0.01 * weights[1:]
        weights -= 0.05 * grad
    return _sigmoid(train @ weights), _sigmoid(test @ weights)


def _kalman_local_trend(prices: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(prices, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return pd.DataFrame()

    returns = values.pct_change(fill_method=None).dropna()
    price_scale = max(float(values.iloc[0]), EPS)
    measurement_var = max(float((returns.std() * price_scale) ** 2), 1e-4) if len(returns) > 1 else 1.0
    level_process_var = measurement_var * 0.02
    slope_process_var = measurement_var * 0.002

    state = np.array([float(values.iloc[0]), 0.0], dtype=float)
    covariance = np.eye(2, dtype=float)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    observation = np.array([[1.0, 0.0]], dtype=float)
    process_cov = np.diag([level_process_var, slope_process_var])
    measurement_cov = np.array([[measurement_var]], dtype=float)

    rows = []
    residuals = []
    for date, obs in values.items():
        predicted_state = transition @ state
        predicted_cov = transition @ covariance @ transition.T + process_cov
        innovation = np.array([[float(obs)]]) - observation @ predicted_state
        innovation_cov = observation @ predicted_cov @ observation.T + measurement_cov
        gain = predicted_cov @ observation.T @ np.linalg.pinv(innovation_cov)
        state = predicted_state + (gain @ innovation).ravel()
        covariance = (np.eye(2) - gain @ observation) @ predicted_cov

        kalman_price = float(state[0])
        kalman_slope = float(state[1])
        residual = float(obs) - kalman_price
        residuals.append(residual)
        rows.append(
            {
                "date": date,
                "kalman_price": kalman_price,
                "kalman_trend": kalman_slope / max(abs(kalman_price), EPS),
                "kalman_trend_slope": kalman_slope,
                "kalman_residual": residual / max(abs(float(obs)), EPS),
            }
        )
    return pd.DataFrame(rows)


def _compute_group_states(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    prices = _safe_numeric(group["current_price"]).replace(0, np.nan).dropna()
    if len(prices) < 10:
        return pd.DataFrame()

    prices.index = group.loc[prices.index, "date"]
    states = _kalman_local_trend(prices)
    if states.empty:
        return states

    states["ticker"] = str(group["ticker"].iloc[0])
    states["model_mode"] = str(group["model_mode"].iloc[0]) if "model_mode" in group.columns else "unknown"
    returns = prices.pct_change(fill_method=None).reindex(states["date"]).fillna(0.0)
    abs_returns = returns.abs()
    realized_vol = returns.rolling(20, min_periods=5).std().reindex(states["date"])
    residual_abs = states["kalman_residual"].abs()

    states["kalman_momentum"] = (
        states["kalman_trend"].rolling(5, min_periods=1).mean()
        / (realized_vol.reset_index(drop=True).replace(0, np.nan).fillna(realized_vol.median() if realized_vol.notna().any() else 0.01).to_numpy() + EPS)
    )
    states["kalman_volatility_state"] = abs_returns.rolling(20, min_periods=5).mean().reindex(states["date"]).fillna(abs_returns.mean()).to_numpy()
    states["kalman_noise_state"] = residual_abs.rolling(20, min_periods=5).mean().fillna(residual_abs.expanding().mean()).to_numpy()
    states["kalman_signal_to_noise"] = states["kalman_trend"].abs() / (states["kalman_noise_state"] + EPS)
    return states.replace([np.inf, -np.inf], np.nan)


def build_kalman_state_features(
    prices_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    config: KalmanStateSpaceConfig | None = None,
) -> pd.DataFrame:
    del returns_df
    config = config or KalmanStateSpaceConfig()
    if prices_df is not None and not prices_df.empty:
        panel = prices_df.reset_index().melt(id_vars=prices_df.index.name or "index", var_name="ticker", value_name="current_price")
        date_col = prices_df.index.name or "index"
        panel = panel.rename(columns={date_col: "date"})
        panel["model_mode"] = "external_prices"
    else:
        panel = _read_csv(config.price_source)
        if panel.empty:
            return pd.DataFrame()
        panel = panel[["date", "ticker", "model_mode", "current_price"]].drop_duplicates(["date", "ticker", "model_mode"])

    panel["current_price"] = _safe_numeric(panel["current_price"])
    panel = panel.dropna(subset=["date", "ticker", "model_mode", "current_price"])
    parts = []
    for _, group in panel.groupby(["model_mode", "ticker"], sort=False):
        states = _compute_group_states(group)
        if not states.empty:
            parts.append(states)
    features = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not features.empty:
        ordered = ["date", "ticker", "model_mode"] + KALMAN_FEATURES
        features = features[[c for c in ordered if c in features.columns]]
    features.to_csv(OUTPUT_FEATURES, index=False)
    return features


def validate_kalman_features(config: KalmanStateSpaceConfig | None = None, features: pd.DataFrame | None = None) -> pd.DataFrame:
    config = config or KalmanStateSpaceConfig()
    features = features if features is not None else _read_csv(OUTPUT_FEATURES)
    realized = _read_csv(config.realized_returns_source)
    ic_dataset = _read_csv(config.ic_dataset_source)
    meta_ranking = _read_csv(config.meta_feature_ranking_source)
    if features.empty or realized.empty:
        validation = pd.DataFrame()
        validation.to_csv(OUTPUT_VALIDATION, index=False)
        return validation

    horizon_col = f"realized_return_{config.horizon}d"
    merged = features.merge(
        realized[["date", "ticker", "model_mode", horizon_col]],
        on=["date", "ticker", "model_mode"],
        how="inner",
    )
    rows = []
    for feature in [c for c in KALMAN_FEATURES if c in merged.columns]:
        rows.append(
            {
                "feature": feature,
                "feature_group": "kalman_state_space",
                "sample_size": int(merged[[feature, horizon_col]].dropna().shape[0]),
                "spearman_ic": _spearman_no_scipy(merged[feature], merged[horizon_col]),
                "mutual_information": _manual_mutual_information(merged[feature], (merged[horizon_col] > 0).astype(int)),
            }
        )

    if not ic_dataset.empty and horizon_col in ic_dataset.columns:
        for feature in ["signal_strength", "expected_daily_return", "expected_total_return", "ema_timing_score", "trend_persistence_score", "quality_score"]:
            if feature in ic_dataset.columns:
                rows.append(
                    {
                        "feature": feature,
                        "feature_group": "existing_model",
                        "sample_size": int(ic_dataset[[feature, horizon_col]].dropna().shape[0]),
                        "spearman_ic": _spearman_no_scipy(ic_dataset[feature], ic_dataset[horizon_col]),
                        "mutual_information": _manual_mutual_information(ic_dataset[feature], (ic_dataset[horizon_col] > 0).astype(int)),
                    }
                )

    validation = pd.DataFrame(rows)
    if not validation.empty:
        validation["abs_spearman_ic"] = validation["spearman_ic"].abs()
        validation["predictive_score"] = (
            validation["abs_spearman_ic"].rank(pct=True) + validation["mutual_information"].rank(pct=True)
        ) / 2.0
        validation = validation.sort_values("predictive_score", ascending=False)
        if not meta_ranking.empty and "feature" in meta_ranking.columns:
            current_top = set(meta_ranking.head(10)["feature"].astype(str))
            validation["in_current_meta_top10"] = validation["feature"].astype(str).isin(current_top)
    validation.to_csv(OUTPUT_VALIDATION, index=False)
    return validation


def _prepare_model_frame(dataset: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = dataset[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True).fillna(0.0))
    y = pd.to_numeric(dataset["meta_label"], errors="coerce").fillna(0).astype(int)
    return x, y


def _model_auc(dataset: pd.DataFrame, features: list[str], config: KalmanStateSpaceConfig) -> dict[str, float]:
    available = [c for c in features if c in dataset.columns]
    if len(dataset) < 50 or len(available) == 0 or dataset["meta_label"].nunique() < 2:
        return {"auc": np.nan, "accuracy": np.nan, "sample_size": len(dataset)}
    dataset = dataset.sort_values("date").copy()
    split_idx = max(1, min(len(dataset) - 1, int(len(dataset) * (1.0 - config.test_size))))
    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:]
    x_train, y_train = _prepare_model_frame(train, available)
    x_test, y_test = _prepare_model_frame(test, available)
    _, test_prob = _fit_numpy_logistic(x_train, y_train, x_test)
    pred = pd.Series(test_prob).ge(0.5).astype(int)
    return {
        "auc": _manual_auc(y_test.reset_index(drop=True), pd.Series(test_prob)),
        "accuracy": float((pred.to_numpy() == y_test.to_numpy()).mean()),
        "sample_size": len(test),
    }


def evaluate_meta_label_integration(config: KalmanStateSpaceConfig | None = None, features: pd.DataFrame | None = None) -> pd.DataFrame:
    config = config or KalmanStateSpaceConfig()
    features = features if features is not None else _read_csv(OUTPUT_FEATURES)
    meta = _read_csv(config.meta_label_dataset_source)
    if features.empty or meta.empty:
        comparison = pd.DataFrame()
        comparison.to_csv(OUTPUT_META_COMPARISON, index=False)
        return comparison

    merged = meta.merge(features, on=["date", "ticker", "model_mode"], how="left")
    base_features = [
        "signal_strength",
        "target_confidence",
        "quality_score",
        "expected_daily_return",
        "expected_total_return",
        "regime_confidence",
        "ema_timing_score",
        "trend_persistence_score",
        "weight",
    ]
    kalman_features = [c for c in KALMAN_FEATURES if c in merged.columns]
    rows = []
    for name, cols in {
        "base_meta_features": base_features,
        "kalman_only": kalman_features,
        "base_plus_kalman": base_features + kalman_features,
    }.items():
        result = _model_auc(merged, cols, config)
        rows.append({"model_variant": name, **result, "feature_count": len([c for c in cols if c in merged.columns])})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_META_COMPARISON, index=False)
    return comparison


def run_kalman_state_space_layer(
    prices_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    config: KalmanStateSpaceConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or KalmanStateSpaceConfig()
    features = build_kalman_state_features(prices_df=prices_df, returns_df=returns_df, config=config)
    validation = validate_kalman_features(config=config, features=features)
    meta_comparison = evaluate_meta_label_integration(config=config, features=features)
    _print_report(features, validation, meta_comparison)
    return {
        "features": features,
        "validation": validation,
        "meta_comparison": meta_comparison,
    }


def _print_report(features: pd.DataFrame, validation: pd.DataFrame, meta_comparison: pd.DataFrame) -> None:
    print("\n===== KALMAN STATE SPACE REPORT =====")
    if features.empty:
        print("No Kalman features available.")
    else:
        print(f"sample size: {len(features)}")
        print(f"average signal/noise ratio: {features['kalman_signal_to_noise'].mean(skipna=True):.6f}")
        strongest = features.sort_values("kalman_trend", ascending=False).groupby("ticker").head(1).head(10)
        weakest = features.sort_values("kalman_trend", ascending=True).groupby("ticker").head(1).head(10)
        high_noise = features.sort_values("kalman_noise_state", ascending=False).groupby("ticker").head(1).head(10)
        low_noise = features.sort_values("kalman_noise_state", ascending=True).groupby("ticker").head(1).head(10)
        print("strongest trend assets:")
        print(strongest[["ticker", "model_mode", "date", "kalman_trend", "kalman_signal_to_noise"]].to_string(index=False))
        print("weakest trend assets:")
        print(weakest[["ticker", "model_mode", "date", "kalman_trend", "kalman_signal_to_noise"]].to_string(index=False))
        print("highest noise assets:")
        print(high_noise[["ticker", "model_mode", "date", "kalman_noise_state"]].to_string(index=False))
        print("lowest noise assets:")
        print(low_noise[["ticker", "model_mode", "date", "kalman_noise_state"]].to_string(index=False))

    print("\n===== KALMAN FEATURE VALIDATION =====")
    if validation.empty:
        print("No validation available.")
    else:
        print(validation[["feature", "feature_group", "sample_size", "spearman_ic", "mutual_information", "predictive_score"]].head(15).to_string(index=False))

    print("\n===== KALMAN META-LABEL COMPARISON =====")
    if meta_comparison.empty:
        print("No meta-label comparison available.")
    else:
        print(meta_comparison.to_string(index=False))

    print("\n===== KALMAN GOVERNANCE =====")
    kalman_validation = validation[validation["feature_group"].eq("kalman_state_space")] if not validation.empty else pd.DataFrame()
    best_score = float(kalman_validation["predictive_score"].max()) if not kalman_validation.empty else 0.0
    best_ic = float(kalman_validation["spearman_ic"].abs().max()) if not kalman_validation.empty else 0.0
    sample_size = int(kalman_validation["sample_size"].max()) if not kalman_validation.empty else 0
    recommendation = "diagnostic_only"
    if sample_size >= 500 and best_ic > 0.05:
        recommendation = "useful_feature"
    if sample_size >= 1000 and best_ic > 0.10:
        recommendation = "strong_feature"
    if sample_size >= 1500 and best_ic > 0.12:
        recommendation = "candidate_for_production"
    print(f"sample size: {sample_size}")
    print(f"best kalman abs IC: {best_ic:.6f}")
    print(f"best kalman predictive score: {best_score:.6f}")
    print(f"recommendation: {recommendation}")


if __name__ == "__main__":
    run_kalman_state_space_layer()
