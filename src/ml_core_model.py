from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


OUTPUT_RESULTS = "ml_core_model_results.csv"
OUTPUT_THRESHOLDS = "ml_core_threshold_analysis.csv"
OUTPUT_IMPORTANCE = "ml_core_feature_importance.csv"
OUTPUT_COMPARISON = "ml_core_comparison.csv"


@dataclass
class MLCoreConfig:
    dataset_path: str = "meta_label_dataset.csv"
    selected_features_path: str = "selected_feature_set.json"
    feature_selection_report_path: str = "feature_selection_report.csv"
    previous_meta_results_path: str = "meta_model_results.csv"
    test_size: float = 0.30
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70)
    include_diagnostic_features: bool = False


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    return df


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _load_selected_features(config: MLCoreConfig) -> dict[str, list[str]]:
    path = Path(config.selected_features_path)
    if not path.exists():
        return {"CORE": [], "SUPPORTING": [], "DIAGNOSTIC_ONLY": [], "REMOVE_FROM_ML": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "CORE": list(payload.get("CORE", [])),
        "SUPPORTING": list(payload.get("SUPPORTING", [])),
        "DIAGNOSTIC_ONLY": list(payload.get("DIAGNOSTIC_ONLY", [])),
        "REMOVE_FROM_ML": list(payload.get("REMOVE_FROM_ML", [])),
    }


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
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _fit_numpy_logistic(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0).replace(0, 1.0)
    train = ((x_train - mean) / std).fillna(0.0).to_numpy(dtype=float)
    test = ((x_test - mean) / std).fillna(0.0).to_numpy(dtype=float)
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    y = y_train.to_numpy(dtype=float)
    weights = np.zeros(train.shape[1], dtype=float)
    for _ in range(800):
        pred = _sigmoid(train @ weights)
        grad = train.T @ (pred - y) / max(len(y), 1)
        grad[1:] += 0.01 * weights[1:]
        weights -= 0.05 * grad
    return _sigmoid(train @ weights), _sigmoid(test @ weights), weights[1:]


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.50) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    pred = (probabilities >= threshold).astype(int)
    if SKLEARN_AVAILABLE:
        auc = roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else np.nan
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "positive_prediction_rate": float(pred.mean()),
        }
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _manual_auc(y, probabilities),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "positive_prediction_rate": float(pred.mean()),
    }


def _return_metrics(returns: pd.Series, labels: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    aligned_labels = pd.to_numeric(labels.reindex(returns.index), errors="coerce")
    if returns.empty:
        return {
            "trades_kept": 0,
            "TP_rate": np.nan,
            "SL_rate": np.nan,
            "hit_rate": np.nan,
            "average_realized_return": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
        }
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    annual_factor = np.sqrt(252 / 20)
    return {
        "trades_kept": len(returns),
        "TP_rate": float(aligned_labels.eq(1).mean()) if aligned_labels.notna().any() else np.nan,
        "SL_rate": float(aligned_labels.eq(-1).mean()) if aligned_labels.notna().any() else np.nan,
        "hit_rate": float(returns.gt(0).mean()),
        "average_realized_return": mean_ret,
        "Sharpe": float(mean_ret / std_ret * annual_factor) if std_ret > 0 else 0.0,
        "Sortino": float(mean_ret / downside_std * annual_factor) if downside_std > 0 else 0.0,
        "Calmar": float(mean_ret * (252 / 20) / abs(max_dd)) if max_dd < 0 else 0.0,
        "max_drawdown": max_dd,
    }


def _prepare_dataset(config: MLCoreConfig) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    dataset = _read_csv(config.dataset_path)
    feature_sets = _load_selected_features(config)
    selected = feature_sets["CORE"] + feature_sets["SUPPORTING"]
    if config.include_diagnostic_features:
        selected += feature_sets["DIAGNOSTIC_ONLY"]
    remove = set(feature_sets["REMOVE_FROM_ML"])
    features = [feature for feature in dict.fromkeys(selected) if feature in dataset.columns and feature not in remove]
    dataset = dataset.dropna(subset=["date", "meta_label"]).sort_values("date").copy()
    return dataset, features, feature_sets


def _split_xy(dataset: pd.DataFrame, features: list[str], config: MLCoreConfig):
    x = dataset[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True).fillna(0.0))
    y = pd.to_numeric(dataset["meta_label"], errors="coerce").fillna(0).astype(int)
    split_idx = max(1, min(len(dataset) - 1, int(len(dataset) * (1.0 - config.test_size))))
    return x.iloc[:split_idx], x.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:], dataset.iloc[:split_idx], dataset.iloc[split_idx:]


def _train_models(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    rows = []
    probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}

    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        models = {
            "logistic_regression": (LogisticRegression(max_iter=500, class_weight="balanced", random_state=42), x_train_scaled, x_test_scaled),
            "random_forest_small": (
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
                x_train,
                x_test,
            ),
            "gradient_boosting_small": (
                GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=42),
                x_train,
                x_test,
            ),
        }
        for name, (model, train_x, test_x) in models.items():
            try:
                model.fit(train_x, y_train)
                train_prob = model.predict_proba(train_x)[:, 1]
                test_prob = model.predict_proba(test_x)[:, 1]
                train_metrics = _metrics(y_train, train_prob)
                test_metrics = _metrics(y_test, test_prob)
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
                probabilities[name] = test_prob
                fitted[name] = (model, scaler if name == "logistic_regression" else None)
            except Exception as exc:
                rows.append({"model": name, "status": f"failed: {exc}"})
    else:
        train_prob, test_prob, weights = _fit_numpy_logistic(x_train, y_train, x_test)
        train_metrics = _metrics(y_train, train_prob)
        test_metrics = _metrics(y_test, test_prob)
        rows.append(
            {
                "model": "numpy_logistic_regression",
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "train_size": len(y_train),
                "test_size": len(y_test),
                "status": "ok_numpy_fallback",
            }
        )
        rows.extend(
            [
                {"model": "random_forest_small", "status": "sklearn_unavailable"},
                {"model": "gradient_boosting_small", "status": "sklearn_unavailable"},
            ]
        )
        probabilities["numpy_logistic_regression"] = test_prob
        fitted["numpy_logistic_regression"] = weights

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_RESULTS, index=False)
    return results, probabilities, fitted


def _best_model_name(results: pd.DataFrame) -> str:
    ok = results[results.get("status", "").astype(str).str.startswith("ok")].copy() if not results.empty and "status" in results.columns else pd.DataFrame()
    if ok.empty:
        return ""
    return str(ok.sort_values(["test_roc_auc", "test_f1"], ascending=False)["model"].iloc[0])


def _threshold_analysis(test_dataset: pd.DataFrame, probabilities: np.ndarray, config: MLCoreConfig, model_name: str) -> pd.DataFrame:
    returns = _safe_numeric(test_dataset.get("realized_return_for_meta", pd.Series(np.nan, index=test_dataset.index)), np.nan)
    labels = _safe_numeric(test_dataset.get("label", pd.Series(np.nan, index=test_dataset.index)), np.nan)
    rows = []
    for threshold in config.thresholds:
        mask = pd.Series(probabilities, index=test_dataset.index).ge(threshold)
        metrics = _return_metrics(returns[mask], labels[mask])
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "sample_reduction": float(1.0 - mask.mean()) if len(mask) else np.nan,
                **metrics,
            }
        )
    analysis = pd.DataFrame(rows)
    analysis.to_csv(OUTPUT_THRESHOLDS, index=False)
    return analysis


def _feature_importance(features: list[str], fitted: dict[str, object], results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model_obj in fitted.items():
        if SKLEARN_AVAILABLE:
            model, _ = model_obj
            if hasattr(model, "feature_importances_"):
                values = model.feature_importances_
            elif hasattr(model, "coef_"):
                values = np.abs(model.coef_[0])
            else:
                continue
        else:
            values = np.abs(np.asarray(model_obj, dtype=float))
        if len(values) != len(features):
            continue
        total = float(np.sum(np.abs(values)))
        for feature, value in zip(features, values):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance_raw": float(value),
                    "importance_normalized": float(abs(value) / total) if total > 0 else 0.0,
                }
            )
    importance = pd.DataFrame(rows)
    if not importance.empty:
        importance = importance.sort_values(["model", "importance_normalized"], ascending=[True, False])
    importance.to_csv(OUTPUT_IMPORTANCE, index=False)
    return importance


def _comparison(results: pd.DataFrame, thresholds: pd.DataFrame, config: MLCoreConfig) -> pd.DataFrame:
    previous = _read_csv(config.previous_meta_results_path)
    best_model = _best_model_name(results)
    best = results[results["model"].eq(best_model)].head(1) if best_model else pd.DataFrame()
    prev_ok = previous[previous.get("status", "").astype(str).str.startswith("ok")] if not previous.empty and "status" in previous.columns else pd.DataFrame()
    prev_best = prev_ok.sort_values(["test_roc_auc", "test_f1"], ascending=False).head(1) if not prev_ok.empty else pd.DataFrame()
    best_threshold = thresholds.sort_values(["Sharpe", "average_realized_return", "hit_rate"], ascending=False).head(1) if not thresholds.empty else pd.DataFrame()
    rows = []
    if not best.empty:
        row = best.iloc[0]
        rows.append(
            {
                "item": "ml_core_best_model",
                "model": best_model,
                "roc_auc": row.get("test_roc_auc", np.nan),
                "f1": row.get("test_f1", np.nan),
                "accuracy": row.get("test_accuracy", np.nan),
            }
        )
    if not prev_best.empty:
        row = prev_best.iloc[0]
        rows.append(
            {
                "item": "previous_meta_best_model",
                "model": row.get("model", ""),
                "roc_auc": row.get("test_roc_auc", np.nan),
                "f1": row.get("test_f1", np.nan),
                "accuracy": row.get("test_accuracy", np.nan),
            }
        )
    if not best_threshold.empty:
        row = best_threshold.iloc[0]
        rows.append(
            {
                "item": "ml_core_best_threshold",
                "model": row.get("model", ""),
                "threshold": row.get("threshold", np.nan),
                "Sharpe": row.get("Sharpe", np.nan),
                "average_realized_return": row.get("average_realized_return", np.nan),
                "sample_reduction": row.get("sample_reduction", np.nan),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    return comparison


def _governance(results: pd.DataFrame, thresholds: pd.DataFrame, dataset: pd.DataFrame) -> str:
    best_name = _best_model_name(results)
    if not best_name:
        return "not useful"
    best = results[results["model"].eq(best_name)].iloc[0]
    auc = float(best.get("test_roc_auc", 0.0) or 0.0)
    f1 = float(best.get("test_f1", 0.0) or 0.0)
    sample_size = int(best.get("test_size", 0) or 0)
    best_threshold = thresholds.sort_values(["Sharpe", "average_realized_return"], ascending=False).head(1)
    improvement_ok = False
    if not best_threshold.empty:
        improvement_ok = float(best_threshold.iloc[0].get("sample_reduction", 1.0) or 1.0) < 0.8
    if sample_size < 300 or auc < 0.53:
        return "not useful"
    if auc >= 0.55 and f1 >= 0.55:
        return "useful for research"
    if auc >= 0.60 and improvement_ok and len(dataset) >= 1000:
        return "eligible for deeper validation"
    if auc >= 0.65 and improvement_ok and len(dataset) >= 1500:
        return "candidate for paper trading filter"
    return "useful for research" if auc >= 0.53 else "not useful"


def run_ml_core_model(config: MLCoreConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or MLCoreConfig()
    dataset, features, feature_sets = _prepare_dataset(config)
    if dataset.empty or not features or dataset["meta_label"].nunique() < 2:
        empty = pd.DataFrame()
        for path in [OUTPUT_RESULTS, OUTPUT_THRESHOLDS, OUTPUT_IMPORTANCE, OUTPUT_COMPARISON]:
            empty.to_csv(path, index=False)
        print("\n===== ML CORE MODEL RESULTS =====")
        print("Insufficient data or features.")
        return {"results": empty, "thresholds": empty, "importance": empty, "comparison": empty}

    x_train, x_test, y_train, y_test, _, test_dataset = _split_xy(dataset, features, config)
    results, probabilities, fitted = _train_models(x_train, x_test, y_train, y_test)
    best_model = _best_model_name(results)
    best_prob = probabilities.get(best_model, np.array([]))
    thresholds = _threshold_analysis(test_dataset, best_prob, config, best_model) if len(best_prob) else pd.DataFrame()
    if thresholds.empty:
        thresholds.to_csv(OUTPUT_THRESHOLDS, index=False)
    importance = _feature_importance(features, fitted, results)
    comparison = _comparison(results, thresholds, config)
    governance = _governance(results, thresholds, dataset)
    _print_report(features, feature_sets, results, thresholds, importance, comparison, governance)
    return {
        "results": results,
        "thresholds": thresholds,
        "importance": importance,
        "comparison": comparison,
    }


def _print_report(
    features: list[str],
    feature_sets: dict[str, list[str]],
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    importance: pd.DataFrame,
    comparison: pd.DataFrame,
    governance: str,
) -> None:
    print("\n===== ML CORE MODEL RESULTS =====")
    print(f"selected features used: {', '.join(features)}")
    if results.empty:
        print("No model results.")
    else:
        show_cols = [c for c in ["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_positive_prediction_rate", "status"] if c in results.columns]
        print(results[show_cols].to_string(index=False))

    print("\n===== ML CORE THRESHOLD ANALYSIS =====")
    if thresholds.empty:
        print("No threshold analysis.")
    else:
        print(thresholds.to_string(index=False))

    print("\n===== ML CORE FEATURE IMPORTANCE =====")
    if importance.empty:
        print("No feature importance.")
    else:
        print(importance.head(20).to_string(index=False))

    print("\n===== ML CORE GOVERNANCE =====")
    best_model = _best_model_name(results)
    best_threshold = thresholds.sort_values(["Sharpe", "average_realized_return"], ascending=False).head(1) if not thresholds.empty else pd.DataFrame()
    print(f"CORE: {', '.join(feature_sets.get('CORE', []))}")
    print(f"SUPPORTING: {', '.join(feature_sets.get('SUPPORTING', []))}")
    print(f"best model: {best_model or 'none'}")
    if not best_threshold.empty:
        row = best_threshold.iloc[0]
        print(f"best threshold: {row.get('threshold')}")
        print(f"best threshold Sharpe: {row.get('Sharpe')}")
        print(f"sample reduction: {row.get('sample_reduction')}")
    if not comparison.empty:
        print(comparison.to_string(index=False))
    print(f"governance classification: {governance}")
    print("production behavior changed: False")


if __name__ == "__main__":
    run_ml_core_model()
