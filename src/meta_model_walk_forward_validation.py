from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


OUTPUT_RESULTS = "meta_model_walk_forward_results.csv"
OUTPUT_FOLDS = "meta_model_walk_forward_folds.csv"
OUTPUT_COMPARISON = "meta_model_walk_forward_comparison.csv"


@dataclass
class WalkForwardMetaConfig:
    dataset_path: str = "meta_label_dataset.csv"
    selected_features_path: str = "selected_feature_set.json"
    minimum_train_size: int = 500
    calibration_size: int = 200
    test_window_size: int = 100
    candidates: tuple[tuple[str, str, float], ...] = (
        ("logistic_regression", "isotonic", 0.65),
        ("gradient_boosting_small", "platt", 0.60),
    )


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


def _load_features(path: str, dataset: pd.DataFrame) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    features = list(payload.get("CORE", [])) + list(payload.get("SUPPORTING", []))
    remove = set(payload.get("REMOVE_FROM_ML", [])) | set(payload.get("DIAGNOSTIC_ONLY", []))
    return [feature for feature in dict.fromkeys(features) if feature not in remove and feature in dataset.columns]


def _prepare_dataset(config: WalkForwardMetaConfig) -> tuple[pd.DataFrame, list[str]]:
    dataset = _read_csv(config.dataset_path).sort_values("date").reset_index(drop=True)
    features = _load_features(config.selected_features_path, dataset)
    return dataset, features


def _model(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    if model_name == "gradient_boosting_small":
        return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=42)
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_predict_prob(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, x_cal: pd.DataFrame, x_test: pd.DataFrame):
    model = _model(model_name)
    if model_name == "logistic_regression":
        scaler = StandardScaler()
        x_train_used = scaler.fit_transform(x_train)
        x_cal_used = scaler.transform(x_cal)
        x_test_used = scaler.transform(x_test)
    else:
        x_train_used = x_train
        x_cal_used = x_cal
        x_test_used = x_test
    model.fit(x_train_used, y_train)
    return model.predict_proba(x_cal_used)[:, 1], model.predict_proba(x_test_used)[:, 1]


def _clip_prob(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)


def _calibrate(calibration_method: str, cal_prob: np.ndarray, y_cal: pd.Series, test_prob: np.ndarray) -> np.ndarray:
    if calibration_method == "raw":
        return test_prob
    if y_cal.nunique() < 2:
        return test_prob
    if calibration_method == "platt":
        x_cal = np.log(_clip_prob(cal_prob) / (1.0 - _clip_prob(cal_prob))).reshape(-1, 1)
        x_test = np.log(_clip_prob(test_prob) / (1.0 - _clip_prob(test_prob))).reshape(-1, 1)
        calibrator = LogisticRegression(max_iter=300, random_state=42)
        calibrator.fit(x_cal, y_cal)
        return calibrator.predict_proba(x_test)[:, 1]
    if calibration_method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_prob, y_cal)
        return calibrator.predict(test_prob)
    return test_prob


def _risk_metrics(returns: pd.Series, labels: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    labels = pd.to_numeric(labels.reindex(returns.index), errors="coerce")
    if returns.empty:
        return {
            "trades": 0,
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
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    equity = (1.0 + returns).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    ann = np.sqrt(252 / 20)
    return {
        "trades": int(len(returns)),
        "average_return": mean_ret,
        "Sharpe": float(mean_ret / std_ret * ann) if std_ret > 0 else 0.0,
        "Sortino": float(mean_ret / downside_std * ann) if downside_std > 0 else 0.0,
        "Calmar": float(mean_ret * (252 / 20) / abs(max_dd)) if max_dd < 0 else 0.0,
        "max_drawdown": max_dd,
        "TP_rate": float(labels.eq(1).mean()) if labels.notna().any() else np.nan,
        "SL_rate": float(labels.eq(-1).mean()) if labels.notna().any() else np.nan,
        "hit_rate": float(returns.gt(0).mean()),
    }


def _classification_metrics(y_test: pd.Series, probabilities: np.ndarray, mask: pd.Series) -> dict[str, float]:
    y = y_test.to_numpy(dtype=int)
    pred = mask.to_numpy(dtype=int)
    precision = precision_score(y, pred, zero_division=0) if SKLEARN_AVAILABLE else np.nan
    recall = recall_score(y, pred, zero_division=0) if SKLEARN_AVAILABLE else np.nan
    auc = roc_auc_score(y, probabilities) if SKLEARN_AVAILABLE and len(np.unique(y)) > 1 else np.nan
    brier = brier_score_loss(y, probabilities) if SKLEARN_AVAILABLE else float(np.mean(np.square(probabilities - y)))
    return {
        "precision": float(precision) if np.isfinite(precision) else np.nan,
        "recall": float(recall) if np.isfinite(recall) else np.nan,
        "ROC_AUC": float(auc) if np.isfinite(auc) else np.nan,
        "Brier": float(brier),
    }


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True).fillna(0.0))


def _fold_ranges(n: int, config: WalkForwardMetaConfig) -> list[tuple[int, int, int, int]]:
    ranges = []
    test_start = config.minimum_train_size + config.calibration_size
    while test_start < n:
        train_end = test_start - config.calibration_size
        cal_start = train_end
        test_end = min(test_start + config.test_window_size, n)
        if train_end >= config.minimum_train_size and test_end > test_start:
            ranges.append((0, train_end, cal_start, test_start, test_end))
        test_start = test_end
    return ranges


def run_meta_model_walk_forward_validation(config: WalkForwardMetaConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or WalkForwardMetaConfig()
    dataset, features = _prepare_dataset(config)
    if dataset.empty or not features or not SKLEARN_AVAILABLE:
        empty = pd.DataFrame()
        for path in [OUTPUT_RESULTS, OUTPUT_FOLDS, OUTPUT_COMPARISON]:
            empty.to_csv(path, index=False)
        print("\n===== META MODEL WALK-FORWARD VALIDATION =====")
        print("Insufficient data or sklearn unavailable.")
        return {"results": empty, "folds": empty, "comparison": empty}

    fold_rows = []
    result_frames = []
    for fold_id, (train_start, train_end, cal_start, test_start, test_end) in enumerate(_fold_ranges(len(dataset), config), start=1):
        train = dataset.iloc[train_start:train_end].copy()
        cal = dataset.iloc[cal_start:test_start].copy()
        test = dataset.iloc[test_start:test_end].copy()
        x_train = _feature_matrix(train, features)
        y_train = _safe_numeric(train["meta_label"], 0).astype(int)
        x_cal = _feature_matrix(cal, features)
        y_cal = _safe_numeric(cal["meta_label"], 0).astype(int)
        x_test = _feature_matrix(test, features)
        y_test = _safe_numeric(test["meta_label"], 0).astype(int)

        unfiltered_returns = _safe_numeric(test["realized_return_for_meta"], np.nan)
        labels = _safe_numeric(test.get("label", pd.Series(np.nan, index=test.index)), np.nan)
        unfiltered_metrics = _risk_metrics(unfiltered_returns, labels)
        result_frames.append(
            pd.DataFrame(
                [
                    {
                        "fold": fold_id,
                        "candidate": "current_system_unfiltered",
                        "model": "none",
                        "calibration_method": "none",
                        "threshold": np.nan,
                        "sample_reduction": 0.0,
                        **unfiltered_metrics,
                    }
                ]
            )
        )

        for model_name, calibration_method, threshold in config.candidates:
            cal_prob, test_prob = _fit_predict_prob(model_name, x_train, y_train, x_cal, x_test)
            calibrated_prob = _calibrate(calibration_method, cal_prob, y_cal, test_prob)
            mask = pd.Series(calibrated_prob, index=test.index).ge(threshold)
            filtered_metrics = _risk_metrics(unfiltered_returns[mask], labels[mask])
            class_metrics = _classification_metrics(y_test, calibrated_prob, mask)
            result_frames.append(
                pd.DataFrame(
                    [
                        {
                            "fold": fold_id,
                            "candidate": f"{model_name}_{calibration_method}_{threshold}",
                            "model": model_name,
                            "calibration_method": calibration_method,
                            "threshold": threshold,
                            "sample_reduction": float(1.0 - mask.mean()) if len(mask) else np.nan,
                            **filtered_metrics,
                            **class_metrics,
                        }
                    ]
                )
            )

        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "calibration_start": cal["date"].min(),
                "calibration_end": cal["date"].max(),
                "test_start": test["date"].min(),
                "test_end": test["date"].max(),
                "train_size": int(len(train)),
                "calibration_size": int(len(cal)),
                "test_size": int(len(test)),
                "features": ", ".join(features),
                "no_lookahead": True,
            }
        )

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    comparison = _comparison(results)
    results.to_csv(OUTPUT_RESULTS, index=False)
    folds.to_csv(OUTPUT_FOLDS, index=False)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    _print_report(results, folds, comparison)
    return {"results": results, "folds": folds, "comparison": comparison}


def _comparison(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for candidate, group in results.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": candidate,
                "folds": int(group["fold"].nunique()),
                "avg_trades": float(group["trades"].mean()),
                "avg_sample_reduction": float(group.get("sample_reduction", pd.Series(np.nan)).mean()),
                "avg_return": float(group["average_return"].mean()),
                "avg_Sharpe": float(group["Sharpe"].mean()),
                "avg_Sortino": float(group["Sortino"].mean()),
                "avg_Calmar": float(group["Calmar"].mean()),
                "avg_max_drawdown": float(group["max_drawdown"].mean()),
                "avg_TP_rate": float(group["TP_rate"].mean()),
                "avg_SL_rate": float(group["SL_rate"].mean()),
                "avg_hit_rate": float(group["hit_rate"].mean()),
                "avg_precision": float(group.get("precision", pd.Series(np.nan)).mean()),
                "avg_recall": float(group.get("recall", pd.Series(np.nan)).mean()),
                "avg_ROC_AUC": float(group.get("ROC_AUC", pd.Series(np.nan)).mean()),
                "avg_Brier": float(group.get("Brier", pd.Series(np.nan)).mean()),
            }
        )
    comparison = pd.DataFrame(rows)
    baseline = comparison[comparison["candidate"].eq("current_system_unfiltered")]
    if not baseline.empty:
        base = baseline.iloc[0]
        comparison["Sharpe_vs_unfiltered"] = comparison["avg_Sharpe"] - float(base["avg_Sharpe"])
        comparison["return_vs_unfiltered"] = comparison["avg_return"] - float(base["avg_return"])
        comparison["hit_rate_vs_unfiltered"] = comparison["avg_hit_rate"] - float(base["avg_hit_rate"])
    return comparison.sort_values(["avg_Sharpe", "avg_return"], ascending=False)


def _governance(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "failed walk-forward"
    filtered = comparison[~comparison["candidate"].eq("current_system_unfiltered")]
    if filtered.empty:
        return "failed walk-forward"
    best = filtered.iloc[0]
    if best["folds"] < 3 or best["avg_trades"] < 30:
        return "failed walk-forward"
    if best.get("Sharpe_vs_unfiltered", 0.0) <= 0 or best.get("return_vs_unfiltered", 0.0) <= 0:
        return "failed walk-forward"
    if best["avg_sample_reduction"] > 0.85:
        return "useful for research"
    if best["avg_Sharpe"] > 1.0 and best["avg_hit_rate"] > 0.58:
        return "eligible for paper trading filter"
    return "candidate for further validation"


def _print_report(results: pd.DataFrame, folds: pd.DataFrame, comparison: pd.DataFrame) -> None:
    print("\n===== META MODEL WALK-FORWARD VALIDATION =====")
    print(f"number of folds: {len(folds)}")
    if not folds.empty:
        print(f"test range: {folds['test_start'].min()} -> {folds['test_end'].max()}")
        print("strict no-look-ahead: True")

    print("\n===== META FILTER VS CURRENT SYSTEM =====")
    print(comparison.to_string(index=False) if not comparison.empty else "No comparison available.")

    print("\n===== META MODEL FOLD RESULTS =====")
    if results.empty:
        print("No fold results.")
    else:
        show_cols = ["fold", "candidate", "trades", "sample_reduction", "average_return", "Sharpe", "TP_rate", "SL_rate", "hit_rate", "precision", "recall", "ROC_AUC", "Brier"]
        print(results[[c for c in show_cols if c in results.columns]].to_string(index=False))

    print("\n===== META MODEL GOVERNANCE =====")
    print(f"classification: {_governance(comparison)}")
    print("production behavior changed: False")


if __name__ == "__main__":
    run_meta_model_walk_forward_validation()
