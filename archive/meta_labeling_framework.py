from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


FEATURE_COLUMNS = [
    "signal_strength",
    "target_confidence",
    "quality_score",
    "expected_daily_return",
    "expected_total_return",
    "regime_confidence",
    "ema_timing_score",
    "trend_persistence_score",
    "kalman_trend_score",
    "momentum_score",
    "hurst_persistence_score",
    "entropy_cleanliness_score",
    "volatility_stability_score",
    "correlation_diversification_score",
    "cycle_stability_score",
    "daily_volatility",
    "downside_risk",
    "weight",
]

INPUT_FILES = {
    "feature_store": "historical_feature_store.csv",
    "forecast_snapshots": "historical_forecast_snapshots.csv",
    "realized_returns": "historical_realized_returns.csv",
    "triple_barrier_labels": "historical_triple_barrier_labels.csv",
    "ic_dataset": "historical_ic_dataset.csv",
}

OUTPUT_FILES = {
    "dataset": "meta_label_dataset.csv",
    "feature_ranking": "meta_feature_ranking.csv",
    "model_results": "meta_model_results.csv",
    "comparison": "meta_labeling_comparison.csv",
}


@dataclass
class MetaLabelingConfig:
    horizon: int = 20
    test_size: float = 0.30
    confidence_threshold: float = 0.50
    selected_only: bool = True


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    return df


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "selected"])


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result


def _load_inputs() -> dict[str, pd.DataFrame]:
    return {name: _coerce_dates(_read_csv(path)) for name, path in INPUT_FILES.items()}


def build_meta_label_dataset(config: MetaLabelingConfig | None = None) -> pd.DataFrame:
    config = config or MetaLabelingConfig()
    inputs = _load_inputs()
    snapshots = inputs["forecast_snapshots"]
    labels = inputs["triple_barrier_labels"]
    realized = inputs["realized_returns"]

    if snapshots.empty:
        return pd.DataFrame()

    snapshots = _ensure_feature_columns(snapshots)
    snapshots["selected"] = _safe_bool(snapshots.get("selected", pd.Series(False, index=snapshots.index)))
    if config.selected_only:
        snapshots = snapshots[snapshots["selected"]].copy()

    label_horizon = labels[labels.get("horizon", pd.Series(dtype=float)).eq(config.horizon)].copy() if not labels.empty else pd.DataFrame()
    label_columns = ["date", "ticker", "model_mode", "label", "first_touch_type", "realized_return_at_barrier"]
    if not label_horizon.empty:
        label_horizon = label_horizon[[c for c in label_columns if c in label_horizon.columns]].drop_duplicates(
            ["date", "ticker", "model_mode"]
        )

    realized_column = f"realized_return_{config.horizon}d"
    realized_columns = ["date", "ticker", "model_mode", realized_column]
    realized_horizon = (
        realized[[c for c in realized_columns if c in realized.columns]].drop_duplicates(["date", "ticker", "model_mode"])
        if not realized.empty and realized_column in realized.columns
        else pd.DataFrame()
    )

    merge_keys = ["date", "ticker", "model_mode"]
    dataset = snapshots.merge(label_horizon, on=merge_keys, how="left")
    if not realized_horizon.empty:
        dataset = dataset.drop(columns=[realized_column], errors="ignore")
        dataset = dataset.merge(realized_horizon, on=merge_keys, how="left")
    elif realized_column not in dataset.columns:
        dataset[realized_column] = np.nan

    if "realized_return_at_barrier" not in dataset.columns:
        dataset["realized_return_at_barrier"] = np.nan
    if "label" not in dataset.columns:
        dataset["label"] = np.nan

    realized_return = _safe_numeric(dataset[realized_column], np.nan)
    barrier_label = _safe_numeric(dataset["label"], np.nan)
    dataset["meta_label"] = ((barrier_label.eq(1)) | (realized_return.gt(0))).astype(int)
    dataset["realized_return_for_meta"] = realized_return.fillna(_safe_numeric(dataset["realized_return_at_barrier"], 0.0))

    keep_columns = [
        "date",
        "ticker",
        "model_mode",
        "selected",
        "regime",
        "label",
        "first_touch_type",
        "realized_return_at_barrier",
        realized_column,
        "realized_return_for_meta",
        "meta_label",
    ] + FEATURE_COLUMNS
    dataset = dataset[[c for c in keep_columns if c in dataset.columns]].copy()
    dataset = dataset.dropna(subset=["date", "ticker", "model_mode", "meta_label"])
    dataset.to_csv(OUTPUT_FILES["dataset"], index=False)
    return dataset


def _information_value(feature: pd.Series, target: pd.Series, bins: int = 5) -> float:
    frame = pd.DataFrame({"feature": pd.to_numeric(feature, errors="coerce"), "target": target}).dropna()
    if frame.empty or frame["target"].nunique() < 2 or frame["feature"].nunique() < 2:
        return 0.0
    try:
        frame["bin"] = pd.qcut(frame["feature"], q=min(bins, frame["feature"].nunique()), duplicates="drop")
    except ValueError:
        return 0.0
    grouped = frame.groupby("bin", observed=False)["target"]
    good = grouped.sum().astype(float)
    bad = grouped.count().astype(float) - good
    good_dist = (good + 0.5) / (good.sum() + 0.5 * len(good))
    bad_dist = (bad + 0.5) / (bad.sum() + 0.5 * len(bad))
    woe = np.log(good_dist / bad_dist)
    iv = ((good_dist - bad_dist) * woe).sum()
    return float(iv) if np.isfinite(iv) else 0.0


def _manual_mutual_information(feature: pd.Series, target: pd.Series, bins: int = 5) -> float:
    frame = pd.DataFrame({"feature": pd.to_numeric(feature, errors="coerce"), "target": target}).dropna()
    if frame.empty or frame["target"].nunique() < 2 or frame["feature"].nunique() < 2:
        return 0.0
    try:
        feature_bins = pd.qcut(frame["feature"], q=min(bins, frame["feature"].nunique()), labels=False, duplicates="drop")
    except ValueError:
        return 0.0
    contingency = pd.crosstab(feature_bins, frame["target"]).astype(float)
    total = contingency.values.sum()
    if total <= 0:
        return 0.0
    pxy = contingency / total
    px = pxy.sum(axis=1).values
    py = pxy.sum(axis=0).values
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            value = pxy.iloc[i, j]
            if value > 0 and px[i] > 0 and py[j] > 0:
                mi += value * np.log(value / (px[i] * py[j]))
    return float(mi) if np.isfinite(mi) else 0.0


def _spearman_no_scipy(feature: pd.Series, target: pd.Series) -> float:
    frame = pd.DataFrame({"feature": pd.to_numeric(feature, errors="coerce"), "target": target}).dropna()
    if frame.empty or frame["feature"].nunique() < 2 or frame["target"].nunique() < 2:
        return 0.0
    corr = frame["feature"].rank().corr(frame["target"].rank(), method="pearson")
    return float(corr) if np.isfinite(corr) else 0.0


def compute_meta_feature_ranking(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        ranking = pd.DataFrame()
        ranking.to_csv(OUTPUT_FILES["feature_ranking"], index=False)
        return ranking

    target = dataset["meta_label"].astype(int)
    rows = []
    available_features = [c for c in FEATURE_COLUMNS if c in dataset.columns]
    numeric_matrix = dataset[available_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill_values = numeric_matrix.median(numeric_only=True).fillna(0.0)
    numeric_matrix = numeric_matrix.fillna(fill_values)

    if SKLEARN_AVAILABLE and target.nunique() > 1 and not numeric_matrix.empty:
        try:
            mi_values = mutual_info_classif(numeric_matrix, target, random_state=42)
            mi_map = dict(zip(available_features, mi_values))
        except Exception:
            mi_map = {}
    else:
        mi_map = {}

    for feature in available_features:
        series = numeric_matrix[feature]
        spearman = _spearman_no_scipy(series, target)
        rows.append(
            {
                "feature": feature,
                "information_value": _information_value(series, target),
                "mutual_information": float(mi_map.get(feature, _manual_mutual_information(series, target))),
                "spearman_correlation": float(spearman) if np.isfinite(spearman) else 0.0,
                "abs_spearman": abs(float(spearman)) if np.isfinite(spearman) else 0.0,
                "sample_size": int(series.notna().sum()),
            }
        )
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking["feature_score"] = (
            ranking["information_value"].rank(pct=True)
            + ranking["mutual_information"].rank(pct=True)
            + ranking["abs_spearman"].rank(pct=True)
        ) / 3.0
        ranking = ranking.sort_values("feature_score", ascending=False)
    ranking.to_csv(OUTPUT_FILES["feature_ranking"], index=False)
    return ranking


def _prepare_model_matrix(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = [c for c in FEATURE_COLUMNS if c in dataset.columns]
    frame = dataset[["date", "meta_label"] + features].copy()
    frame = frame.sort_values("date")
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True).fillna(0.0))
    y = frame["meta_label"].astype(int)
    dates = frame["date"]
    return x, y, dates


def _classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], probabilities: Iterable[float] | None = None) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if SKLEARN_AVAILABLE else float((y_true == y_pred).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if SKLEARN_AVAILABLE else precision,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if SKLEARN_AVAILABLE else recall,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if SKLEARN_AVAILABLE else f1,
        "roc_auc": np.nan,
    }
    if SKLEARN_AVAILABLE and probabilities is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        except Exception:
            metrics["roc_auc"] = np.nan
    elif probabilities is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = _manual_auc(y_true, np.asarray(list(probabilities), dtype=float))
    return metrics


def _manual_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    frame = pd.DataFrame({"y": y_true, "p": probabilities}).dropna()
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _fit_numpy_logistic(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0).replace(0, 1.0)
    train = ((x_train - mean) / std).to_numpy(dtype=float)
    test = ((x_test - mean) / std).to_numpy(dtype=float)
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    y = y_train.to_numpy(dtype=float)
    weights = np.zeros(train.shape[1], dtype=float)
    learning_rate = 0.05
    l2 = 0.01
    for _ in range(600):
        pred = _sigmoid(train @ weights)
        grad = train.T @ (pred - y) / max(len(y), 1)
        grad[1:] += l2 * weights[1:]
        weights -= learning_rate * grad
    return _sigmoid(train @ weights), _sigmoid(test @ weights)


def train_meta_model_benchmark(dataset: pd.DataFrame, config: MetaLabelingConfig | None = None) -> tuple[pd.DataFrame, pd.Series]:
    config = config or MetaLabelingConfig()
    if dataset.empty or dataset["meta_label"].nunique() < 2 or len(dataset) < 30:
        results = pd.DataFrame(
            [
                {
                    "model": "insufficient_data",
                    "train_accuracy": np.nan,
                    "test_accuracy": np.nan,
                    "test_roc_auc": np.nan,
                    "status": "insufficient_data",
                }
            ]
        )
        results.to_csv(OUTPUT_FILES["model_results"], index=False)
        return results, pd.Series(dtype=float)

    x, y, dates = _prepare_model_matrix(dataset)
    split_idx = max(1, min(len(x) - 1, int(len(x) * (1.0 - config.test_size))))
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if not SKLEARN_AVAILABLE:
        train_prob, test_prob = _fit_numpy_logistic(x_train, y_train, x_test)
        probabilities = pd.Series(test_prob, index=x_test.index)
        y_pred = (probabilities >= config.confidence_threshold).astype(int)
        train_pred = (train_prob >= config.confidence_threshold).astype(int)
        train_metrics = _classification_metrics(y_train, train_pred, train_prob)
        metrics = _classification_metrics(y_test, y_pred, probabilities)
        results = pd.DataFrame(
            [
                {
                    "model": "numpy_logistic_regression",
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    "test_accuracy": metrics["accuracy"],
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                    "test_f1": metrics["f1"],
                    "test_roc_auc": metrics["roc_auc"],
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "status": "ok_numpy_fallback",
                },
                {
                    "model": "random_forest_small",
                    "status": "sklearn_unavailable",
                },
                {
                    "model": "gradient_boosting_small",
                    "status": "sklearn_unavailable",
                }
            ]
        )
        results.to_csv(OUTPUT_FILES["model_results"], index=False)
        return results, probabilities

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    models = {
        "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        "random_forest_small": RandomForestClassifier(
            n_estimators=80,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced_subsample",
        ),
        "gradient_boosting_small": GradientBoostingClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    rows = []
    test_probabilities = {}
    for name, model in models.items():
        try:
            train_x = x_train_scaled if name == "logistic_regression" else x_train
            test_x = x_test_scaled if name == "logistic_regression" else x_test
            model.fit(train_x, y_train)
            train_prob = model.predict_proba(train_x)[:, 1]
            test_prob = model.predict_proba(test_x)[:, 1]
            train_pred = (train_prob >= config.confidence_threshold).astype(int)
            test_pred = (test_prob >= config.confidence_threshold).astype(int)
            train_metrics = _classification_metrics(y_train, train_pred, train_prob)
            test_metrics = _classification_metrics(y_test, test_pred, test_prob)
            rows.append(
                {
                    "model": name,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "status": "ok",
                }
            )
            test_probabilities[name] = pd.Series(test_prob, index=x_test.index)
        except Exception as exc:
            rows.append({"model": name, "status": f"failed: {exc}"})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_FILES["model_results"], index=False)
    if results.empty or not test_probabilities:
        return results, pd.Series(dtype=float)
    best_row = results[results["status"].eq("ok")].sort_values(["test_roc_auc", "test_f1"], ascending=False).head(1)
    if best_row.empty:
        return results, pd.Series(dtype=float)
    best_model = str(best_row["model"].iloc[0])
    return results, test_probabilities.get(best_model, pd.Series(dtype=float))


def _performance_metrics(returns: pd.Series, labels: pd.Series | None = None) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {
            "sample_size": 0,
            "average_return": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
            "TP_rate": np.nan,
            "SL_rate": np.nan,
            "hit_rate": np.nan,
        }
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    sharpe = mean_ret / std_ret * np.sqrt(252 / 20) if std_ret > 0 else 0.0
    sortino = mean_ret / downside_std * np.sqrt(252 / 20) if downside_std > 0 else 0.0
    calmar = mean_ret * (252 / 20) / abs(max_dd) if max_dd < 0 else 0.0
    label_series = pd.to_numeric(labels, errors="coerce") if labels is not None else pd.Series(dtype=float)
    return {
        "sample_size": len(returns),
        "average_return": mean_ret,
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "max_drawdown": max_dd,
        "TP_rate": float(label_series.eq(1).mean()) if not label_series.empty else np.nan,
        "SL_rate": float(label_series.eq(-1).mean()) if not label_series.empty else np.nan,
        "hit_rate": float(returns.gt(0).mean()),
    }


def simulate_meta_label_filter(
    dataset: pd.DataFrame,
    probabilities: pd.Series,
    config: MetaLabelingConfig | None = None,
) -> pd.DataFrame:
    config = config or MetaLabelingConfig()
    if dataset.empty:
        comparison = pd.DataFrame()
        comparison.to_csv(OUTPUT_FILES["comparison"], index=False)
        return comparison

    dataset = dataset.sort_values("date").copy()
    split_idx = max(1, min(len(dataset) - 1, int(len(dataset) * (1.0 - config.test_size)))) if len(dataset) > 1 else 0
    test = dataset.iloc[split_idx:].copy()
    returns = _safe_numeric(test["realized_return_for_meta"], np.nan)
    labels = _safe_numeric(test.get("label", pd.Series(np.nan, index=test.index)), np.nan)

    rows = [{"strategy": "current_system", **_performance_metrics(returns, labels)}]
    if not probabilities.empty:
        probabilities = probabilities.reindex(test.index)
        mask = probabilities.ge(config.confidence_threshold).fillna(False)
        rows.append(
            {
                "strategy": "meta_model_filtered",
                **_performance_metrics(returns[mask], labels[mask]),
                "sample_reduction": float(1.0 - mask.mean()) if len(mask) else np.nan,
                "confidence_threshold": config.confidence_threshold,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_FILES["comparison"], index=False)
    return comparison


def run_meta_labeling_framework(config: MetaLabelingConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or MetaLabelingConfig()
    dataset = build_meta_label_dataset(config)
    ranking = compute_meta_feature_ranking(dataset)
    model_results, probabilities = train_meta_model_benchmark(dataset, config)
    comparison = simulate_meta_label_filter(dataset, probabilities, config)
    _print_report(dataset, ranking, model_results, comparison)
    return {
        "dataset": dataset,
        "feature_ranking": ranking,
        "model_results": model_results,
        "comparison": comparison,
    }


def _print_report(
    dataset: pd.DataFrame,
    ranking: pd.DataFrame,
    model_results: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    print("\n===== META LABELING REPORT =====")
    print(f"dataset size: {len(dataset)}")
    positive_pct = float(dataset["meta_label"].mean() * 100) if not dataset.empty else 0.0
    print(f"positive label %: {positive_pct:.2f}%")
    if not ranking.empty:
        print("feature importance ranking:")
        print(ranking[["feature", "information_value", "mutual_information", "spearman_correlation", "feature_score"]].head(10).to_string(index=False))
    if not model_results.empty:
        ok_models = model_results[model_results.get("status", "").eq("ok")] if "status" in model_results.columns else pd.DataFrame()
        best = ok_models.sort_values(["test_roc_auc", "test_f1"], ascending=False).head(1) if not ok_models.empty else model_results.head(1)
        print("best model:")
        print(best.to_string(index=False))

    print("\n===== META LABELING SIMULATION =====")
    if comparison.empty:
        print("No simulation available.")
    else:
        print(comparison.to_string(index=False))

    print("\n===== META LABELING GOVERNANCE =====")
    class_balance = min(positive_pct, 100.0 - positive_pct)
    warnings = []
    if len(dataset) < 150:
        warnings.append("sample_size_below_150")
    if len(dataset) < 500:
        warnings.append("sample_size_below_500")
    if class_balance < 20:
        warnings.append("class_imbalance")
    if not SKLEARN_AVAILABLE:
        warnings.append("sklearn_unavailable")
    recommendation = "research_only"
    if len(dataset) >= 500 and class_balance >= 20 and SKLEARN_AVAILABLE:
        recommendation = "eligible_for_deeper_validation"
    print(f"sample size: {len(dataset)}")
    print(f"class balance minority %: {class_balance:.2f}%")
    print(f"overfitting warnings: {', '.join(warnings) if warnings else 'none'}")
    print(f"promotion recommendation: {recommendation}")


if __name__ == "__main__":
    run_meta_labeling_framework()
