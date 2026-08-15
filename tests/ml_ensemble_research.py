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
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


OUTPUT_RESULTS = "ml_ensemble_results.csv"
OUTPUT_THRESHOLDS = "ml_ensemble_threshold_analysis.csv"
OUTPUT_CALIBRATION = "ml_ensemble_calibration.csv"
OUTPUT_COMPARISON = "ml_ensemble_comparison.csv"


@dataclass
class MLEnsembleConfig:
    dataset_path: str = "meta_label_dataset.csv"
    selected_features_path: str = "selected_feature_set.json"
    previous_meta_results_path: str = "meta_model_results.csv"
    ml_core_results_path: str = "ml_core_model_results.csv"
    ml_core_thresholds_path: str = "ml_core_threshold_analysis.csv"
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


def _load_features(config: MLEnsembleConfig) -> list[str]:
    path = Path(config.selected_features_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = list(payload.get("CORE", [])) + list(payload.get("SUPPORTING", []))
    if config.include_diagnostic_features:
        features += list(payload.get("DIAGNOSTIC_ONLY", []))
    remove = set(payload.get("REMOVE_FROM_ML", []))
    return [feature for feature in dict.fromkeys(features) if feature not in remove]


def _prepare_data(config: MLEnsembleConfig):
    dataset = _read_csv(config.dataset_path).sort_values("date")
    features = [feature for feature in _load_features(config) if feature in dataset.columns]
    if dataset.empty or not features:
        return dataset, features, None
    x = dataset[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True).fillna(0.0))
    y = _safe_numeric(dataset["meta_label"], 0.0).astype(int)
    split_idx = max(1, min(len(dataset) - 1, int(len(dataset) * (1.0 - config.test_size))))
    return (
        dataset,
        features,
        (
            x.iloc[:split_idx],
            x.iloc[split_idx:],
            y.iloc[:split_idx],
            y.iloc[split_idx:],
            dataset.iloc[:split_idx],
            dataset.iloc[split_idx:],
        ),
    )


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


def _manual_pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    frame = pd.DataFrame({"y": y_true, "p": probabilities}).dropna().sort_values("p", ascending=False)
    if frame.empty or frame["y"].nunique() < 2:
        return np.nan
    positives = float(frame["y"].sum())
    if positives <= 0:
        return np.nan
    tp = frame["y"].cumsum()
    fp = (1 - frame["y"]).cumsum()
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.trapz(precision, recall)) if len(frame) > 1 else np.nan


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.50) -> dict[str, float]:
    y = y_true.to_numpy(dtype=int)
    pred = (probabilities >= threshold).astype(int)
    if SKLEARN_AVAILABLE:
        roc_auc = roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else np.nan
        pr_auc = average_precision_score(y, probabilities) if len(np.unique(y)) > 1 else np.nan
        brier = brier_score_loss(y, probabilities)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else np.nan,
            "pr_auc": float(pr_auc) if np.isfinite(pr_auc) else np.nan,
            "brier_score": float(brier),
            "positive_prediction_rate": float(pred.mean()),
        }
    tp = float(((y == 1) & (pred == 1)).sum())
    fp = float(((y == 0) & (pred == 1)).sum())
    fn = float(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _manual_auc(y, probabilities),
        "pr_auc": _manual_pr_auc(y, probabilities),
        "brier_score": float(np.mean(np.square(probabilities - y))),
        "positive_prediction_rate": float(pred.mean()),
    }


def _return_metrics(returns: pd.Series, labels: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    labels = pd.to_numeric(labels.reindex(returns.index), errors="coerce")
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
        "TP_rate": float(labels.eq(1).mean()) if labels.notna().any() else np.nan,
        "SL_rate": float(labels.eq(-1).mean()) if labels.notna().any() else np.nan,
        "hit_rate": float(returns.gt(0).mean()),
        "average_realized_return": mean_ret,
        "Sharpe": float(mean_ret / std_ret * annual_factor) if std_ret > 0 else 0.0,
        "Sortino": float(mean_ret / downside_std * annual_factor) if downside_std > 0 else 0.0,
        "Calmar": float(mean_ret * (252 / 20) / abs(max_dd)) if max_dd < 0 else 0.0,
        "max_drawdown": max_dd,
    }


def _fit_models(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series) -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    if not SKLEARN_AVAILABLE:
        return probabilities
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
        model.fit(train_x, y_train)
        probabilities[name] = model.predict_proba(test_x)[:, 1]
    if probabilities:
        stacked = np.column_stack(list(probabilities.values()))
        probabilities["average_probability_ensemble"] = stacked.mean(axis=1)
    return probabilities


def _calibration_rows(y_test: pd.Series, probabilities: dict[str, np.ndarray], bucket_count: int = 5) -> pd.DataFrame:
    rows = []
    y = y_test.reset_index(drop=True)
    for model, probs in probabilities.items():
        frame = pd.DataFrame({"probability": probs, "actual": y})
        frame["bucket"] = pd.qcut(frame["probability"], q=min(bucket_count, frame["probability"].nunique()), duplicates="drop")
        grouped = frame.groupby("bucket", observed=False)
        for bucket, group in grouped:
            rows.append(
                {
                    "model": model,
                    "bucket": str(bucket),
                    "sample_size": len(group),
                    "avg_predicted_probability": float(group["probability"].mean()),
                    "actual_positive_rate": float(group["actual"].mean()),
                    "calibration_error": float(abs(group["probability"].mean() - group["actual"].mean())),
                }
            )
    calibration = pd.DataFrame(rows)
    calibration.to_csv(OUTPUT_CALIBRATION, index=False)
    return calibration


def _threshold_rows(test_dataset: pd.DataFrame, probabilities: dict[str, np.ndarray], config: MLEnsembleConfig) -> pd.DataFrame:
    returns = _safe_numeric(test_dataset.get("realized_return_for_meta", pd.Series(np.nan, index=test_dataset.index)), np.nan)
    labels = _safe_numeric(test_dataset.get("label", pd.Series(np.nan, index=test_dataset.index)), np.nan)
    rows = []
    for model, probs in probabilities.items():
        prob_series = pd.Series(probs, index=test_dataset.index)
        for threshold in config.thresholds:
            mask = prob_series.ge(threshold)
            rows.append(
                {
                    "model": model,
                    "threshold": threshold,
                    "sample_reduction": float(1.0 - mask.mean()) if len(mask) else np.nan,
                    **_return_metrics(returns[mask], labels[mask]),
                }
            )
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(OUTPUT_THRESHOLDS, index=False)
    return threshold_df


def _comparison(results: pd.DataFrame, thresholds: pd.DataFrame, config: MLEnsembleConfig) -> pd.DataFrame:
    previous_meta = _read_csv(config.previous_meta_results_path)
    core_results = _read_csv(config.ml_core_results_path)
    core_thresholds = _read_csv(config.ml_core_thresholds_path)
    rows = []
    if not results.empty:
        best = results.sort_values(["roc_auc", "f1"], ascending=False).head(1).iloc[0]
        rows.append(
            {
                "item": "ensemble_best_model",
                "model": best["model"],
                "roc_auc": best["roc_auc"],
                "f1": best["f1"],
                "accuracy": best["accuracy"],
            }
        )
    if not thresholds.empty:
        best_t = thresholds.sort_values(["Sharpe", "average_realized_return", "hit_rate"], ascending=False).head(1).iloc[0]
        rows.append(
            {
                "item": "ensemble_best_threshold",
                "model": best_t["model"],
                "threshold": best_t["threshold"],
                "Sharpe": best_t["Sharpe"],
                "average_realized_return": best_t["average_realized_return"],
                "sample_reduction": best_t["sample_reduction"],
            }
        )
    if not core_results.empty:
        core = (
            core_results[core_results.get("status", "").astype(str).eq("ok")]
            .sort_values(["test_roc_auc", "test_f1"], ascending=False)
            .head(1)
        )
        if not core.empty:
            row = core.iloc[0]
            rows.append(
                {
                    "item": "ml_core_best_model",
                    "model": row["model"],
                    "roc_auc": row["test_roc_auc"],
                    "f1": row["test_f1"],
                    "accuracy": row["test_accuracy"],
                }
            )
    if not core_thresholds.empty:
        row = core_thresholds.sort_values(["Sharpe", "average_realized_return"], ascending=False).head(1).iloc[0]
        rows.append(
            {
                "item": "ml_core_best_threshold",
                "model": row["model"],
                "threshold": row["threshold"],
                "Sharpe": row["Sharpe"],
                "average_realized_return": row["average_realized_return"],
                "sample_reduction": row["sample_reduction"],
            }
        )
    if not previous_meta.empty:
        prev = (
            previous_meta[previous_meta.get("status", "").astype(str).eq("ok")]
            .sort_values(["test_roc_auc", "test_f1"], ascending=False)
            .head(1)
        )
        if not prev.empty:
            row = prev.iloc[0]
            rows.append(
                {
                    "item": "previous_meta_best_model",
                    "model": row["model"],
                    "roc_auc": row["test_roc_auc"],
                    "f1": row["test_f1"],
                    "accuracy": row["test_accuracy"],
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    return comparison


def _governance(results: pd.DataFrame, thresholds: pd.DataFrame) -> str:
    if results.empty:
        return "not useful"
    best = results.sort_values(["roc_auc", "f1"], ascending=False).head(1).iloc[0]
    auc = float(best.get("roc_auc", 0.0) or 0.0)
    f1 = float(best.get("f1", 0.0) or 0.0)
    threshold_best = (
        thresholds.sort_values(["Sharpe", "average_realized_return"], ascending=False).head(1) if not thresholds.empty else pd.DataFrame()
    )
    sample_ok = False
    if not threshold_best.empty:
        sample_ok = int(threshold_best.iloc[0].get("trades_kept", 0) or 0) >= 100
    if auc < 0.53:
        return "not useful"
    if auc >= 0.53 and f1 >= 0.55:
        return "useful for research"
    if auc >= 0.60 and sample_ok:
        return "eligible for deeper validation"
    if auc >= 0.65 and sample_ok:
        return "candidate for paper trading filter"
    return "useful for research"


def run_ml_ensemble_research(config: MLEnsembleConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or MLEnsembleConfig()
    dataset, features, split = _prepare_data(config)
    if split is None or not SKLEARN_AVAILABLE:
        empty = pd.DataFrame()
        for path in [OUTPUT_RESULTS, OUTPUT_THRESHOLDS, OUTPUT_CALIBRATION, OUTPUT_COMPARISON]:
            empty.to_csv(path, index=False)
        print("\n===== ML ENSEMBLE RESULTS =====")
        print("Insufficient data or sklearn unavailable.")
        return {"results": empty, "thresholds": empty, "calibration": empty, "comparison": empty}

    x_train, x_test, y_train, y_test, _, test_dataset = split
    probabilities = _fit_models(x_train, x_test, y_train)
    rows = []
    for model, probs in probabilities.items():
        rows.append({"model": model, **_metrics(y_test, probs), "test_size": len(y_test), "status": "ok"})
    results = pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)
    results.to_csv(OUTPUT_RESULTS, index=False)
    thresholds = _threshold_rows(test_dataset, probabilities, config)
    calibration = _calibration_rows(y_test, probabilities)
    comparison = _comparison(results, thresholds, config)
    governance = _governance(results, thresholds)
    _print_report(features, results, thresholds, calibration, comparison, governance)
    return {"results": results, "thresholds": thresholds, "calibration": calibration, "comparison": comparison}


def _print_report(
    features: list[str],
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    calibration: pd.DataFrame,
    comparison: pd.DataFrame,
    governance: str,
) -> None:
    print("\n===== ML ENSEMBLE RESULTS =====")
    print(f"features used: {', '.join(features)}")
    print(results.to_string(index=False) if not results.empty else "No results.")

    print("\n===== ML ENSEMBLE THRESHOLD ANALYSIS =====")
    if thresholds.empty:
        print("No threshold analysis.")
    else:
        cols = [
            "model",
            "threshold",
            "trades_kept",
            "sample_reduction",
            "TP_rate",
            "SL_rate",
            "hit_rate",
            "average_realized_return",
            "Sharpe",
            "Sortino",
            "Calmar",
            "max_drawdown",
        ]
        print(thresholds[cols].sort_values(["Sharpe", "average_realized_return"], ascending=False).head(15).to_string(index=False))

    print("\n===== ML ENSEMBLE CALIBRATION =====")
    if calibration.empty:
        print("No calibration.")
    else:
        print(calibration.head(20).to_string(index=False))

    print("\n===== ML ENSEMBLE GOVERNANCE =====")
    best_model = results.iloc[0]["model"] if not results.empty else "none"
    best_threshold = (
        thresholds.sort_values(["Sharpe", "average_realized_return"], ascending=False).head(1) if not thresholds.empty else pd.DataFrame()
    )
    print(f"best model: {best_model}")
    if not best_threshold.empty:
        row = best_threshold.iloc[0]
        print(f"best threshold: {row['threshold']}")
        print(f"best threshold model: {row['model']}")
        print(f"best Sharpe: {row['Sharpe']}")
        print(f"trades kept: {row['trades_kept']}")
        print(f"sample reduction: {row['sample_reduction']}")
    if not comparison.empty:
        print(comparison.to_string(index=False))
    print(f"governance classification: {governance}")
    print("production behavior changed: False")


if __name__ == "__main__":
    run_ml_ensemble_research()
