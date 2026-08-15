from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


OUTPUT_CALIBRATION = "meta_probability_calibration.csv"
OUTPUT_BUCKETS = "meta_calibration_buckets.csv"
OUTPUT_THRESHOLDS = "calibrated_threshold_analysis.csv"


@dataclass
class ProbabilityCalibrationConfig:
    dataset_path: str = "meta_label_dataset.csv"
    selected_features_path: str = "selected_feature_set.json"
    train_size: float = 0.60
    calibration_size: float = 0.20
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70)
    min_isotonic_samples: int = 150


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


def _load_features(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    features = list(payload.get("CORE", [])) + list(payload.get("SUPPORTING", []))
    remove = set(payload.get("REMOVE_FROM_ML", [])) | set(payload.get("DIAGNOSTIC_ONLY", []))
    return [feature for feature in dict.fromkeys(features) if feature not in remove]


def _prepare_data(config: ProbabilityCalibrationConfig):
    dataset = _read_csv(config.dataset_path).sort_values("date")
    features = [feature for feature in _load_features(config.selected_features_path) if feature in dataset.columns]
    if dataset.empty or not features or "meta_label" not in dataset.columns:
        return dataset, features, None
    x = dataset[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True).fillna(0.0))
    y = _safe_numeric(dataset["meta_label"], 0.0).astype(int)
    n = len(dataset)
    train_end = max(1, min(n - 2, int(n * config.train_size)))
    cal_end = max(train_end + 1, min(n - 1, int(n * (config.train_size + config.calibration_size))))
    split = {
        "x_train": x.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "x_cal": x.iloc[train_end:cal_end],
        "y_cal": y.iloc[train_end:cal_end],
        "x_test": x.iloc[cal_end:],
        "y_test": y.iloc[cal_end:],
        "test_dataset": dataset.iloc[cal_end:],
    }
    return dataset, features, split


def _fit_base_models(split: dict) -> dict[str, dict[str, np.ndarray]]:
    if not SKLEARN_AVAILABLE:
        return {}
    x_train, y_train = split["x_train"], split["y_train"]
    x_cal, x_test = split["x_cal"], split["x_test"]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_cal_scaled = scaler.transform(x_cal)
    x_test_scaled = scaler.transform(x_test)
    models = {
        "logistic_regression": (LogisticRegression(max_iter=500, class_weight="balanced", random_state=42), x_train_scaled, x_cal_scaled, x_test_scaled),
        "random_forest_small": (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            x_train,
            x_cal,
            x_test,
        ),
        "gradient_boosting_small": (
            GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=42),
            x_train,
            x_cal,
            x_test,
        ),
    }
    output = {}
    for name, (model, train_x, cal_x, test_x) in models.items():
        model.fit(train_x, y_train)
        output[name] = {
            "cal": model.predict_proba(cal_x)[:, 1],
            "test": model.predict_proba(test_x)[:, 1],
        }
    if output:
        output["average_probability_ensemble"] = {
            "cal": np.column_stack([v["cal"] for v in output.values()]).mean(axis=1),
            "test": np.column_stack([v["test"] for v in output.values()]).mean(axis=1),
        }
    return output


def _clip_prob(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)


def _platt_calibrate(cal_prob: np.ndarray, cal_y: pd.Series, test_prob: np.ndarray) -> np.ndarray:
    if not SKLEARN_AVAILABLE or cal_y.nunique() < 2:
        return test_prob
    x_cal = np.log(_clip_prob(cal_prob) / (1.0 - _clip_prob(cal_prob))).reshape(-1, 1)
    x_test = np.log(_clip_prob(test_prob) / (1.0 - _clip_prob(test_prob))).reshape(-1, 1)
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(x_cal, cal_y)
    return model.predict_proba(x_test)[:, 1]


def _isotonic_calibrate(cal_prob: np.ndarray, cal_y: pd.Series, test_prob: np.ndarray, min_samples: int) -> np.ndarray:
    if not SKLEARN_AVAILABLE or len(cal_y) < min_samples or cal_y.nunique() < 2:
        return np.full_like(test_prob, np.nan, dtype=float)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(cal_prob, cal_y)
    return model.predict(test_prob)


def _ece(y_true: pd.Series, prob: np.ndarray, buckets: int = 5) -> float:
    frame = pd.DataFrame({"y": y_true.to_numpy(dtype=int), "p": prob}).dropna()
    if frame.empty:
        return np.nan
    try:
        frame["bucket"] = pd.qcut(frame["p"], q=min(buckets, frame["p"].nunique()), duplicates="drop")
    except ValueError:
        return np.nan
    total = len(frame)
    error = 0.0
    for _, group in frame.groupby("bucket", observed=False):
        error += len(group) / total * abs(float(group["p"].mean()) - float(group["y"].mean()))
    return float(error)


def _metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"y": y_true.to_numpy(dtype=int), "p": prob}).dropna()
    if frame.empty:
        return {"brier_score": np.nan, "roc_auc": np.nan, "calibration_error": np.nan, "expected_calibration_error": np.nan}
    roc = roc_auc_score(frame["y"], frame["p"]) if frame["y"].nunique() > 1 and SKLEARN_AVAILABLE else np.nan
    brier = brier_score_loss(frame["y"], frame["p"]) if SKLEARN_AVAILABLE else float(np.mean(np.square(frame["p"] - frame["y"])))
    return {
        "brier_score": float(brier),
        "roc_auc": float(roc) if np.isfinite(roc) else np.nan,
        "calibration_error": float(abs(frame["p"].mean() - frame["y"].mean())),
        "expected_calibration_error": _ece(frame["y"], frame["p"]),
    }


def _bucket_rows(model: str, method: str, y_true: pd.Series, prob: np.ndarray) -> list[dict]:
    frame = pd.DataFrame({"y": y_true.to_numpy(dtype=int), "p": prob}).dropna()
    if frame.empty:
        return []
    try:
        frame["bucket"] = pd.qcut(frame["p"], q=min(5, frame["p"].nunique()), duplicates="drop")
    except ValueError:
        frame["bucket"] = "single_bucket"
    rows = []
    for bucket, group in frame.groupby("bucket", observed=False):
        rows.append(
            {
                "model": model,
                "calibration_method": method,
                "predicted_probability_bucket": str(bucket),
                "sample_size": int(len(group)),
                "avg_predicted_probability": float(group["p"].mean()),
                "actual_positive_rate": float(group["y"].mean()),
                "bucket_calibration_error": float(abs(group["p"].mean() - group["y"].mean())),
            }
        )
    return rows


def _return_metrics(returns: pd.Series, labels: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    labels = pd.to_numeric(labels.reindex(returns.index), errors="coerce")
    if returns.empty:
        return {"trades_kept": 0, "TP_rate": np.nan, "SL_rate": np.nan, "hit_rate": np.nan, "average_return": np.nan, "Sharpe": np.nan, "max_drawdown": np.nan}
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    equity = (1.0 + returns).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    return {
        "trades_kept": int(len(returns)),
        "TP_rate": float(labels.eq(1).mean()) if labels.notna().any() else np.nan,
        "SL_rate": float(labels.eq(-1).mean()) if labels.notna().any() else np.nan,
        "hit_rate": float(returns.gt(0).mean()),
        "average_return": mean_ret,
        "Sharpe": float(mean_ret / std_ret * np.sqrt(252 / 20)) if std_ret > 0 else 0.0,
        "max_drawdown": max_dd,
    }


def _threshold_rows(test_dataset: pd.DataFrame, model: str, method: str, prob: np.ndarray, thresholds: tuple[float, ...]) -> list[dict]:
    returns = _safe_numeric(test_dataset.get("realized_return_for_meta", pd.Series(np.nan, index=test_dataset.index)), np.nan)
    labels = _safe_numeric(test_dataset.get("label", pd.Series(np.nan, index=test_dataset.index)), np.nan)
    prob_series = pd.Series(prob, index=test_dataset.index)
    rows = []
    for threshold in thresholds:
        mask = prob_series.ge(threshold)
        rows.append(
            {
                "model": model,
                "calibration_method": method,
                "threshold": threshold,
                "sample_reduction": float(1.0 - mask.mean()) if len(mask) else np.nan,
                **_return_metrics(returns[mask], labels[mask]),
            }
        )
    return rows


def run_meta_probability_calibration(config: ProbabilityCalibrationConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or ProbabilityCalibrationConfig()
    dataset, features, split = _prepare_data(config)
    if split is None or not SKLEARN_AVAILABLE:
        empty = pd.DataFrame()
        for path in [OUTPUT_CALIBRATION, OUTPUT_BUCKETS, OUTPUT_THRESHOLDS]:
            empty.to_csv(path, index=False)
        print("\n===== META PROBABILITY CALIBRATION =====")
        print("Insufficient data or sklearn unavailable.")
        return {"calibration": empty, "buckets": empty, "thresholds": empty}

    base_probs = _fit_base_models(split)
    rows = []
    bucket_rows = []
    threshold_rows = []
    for model_name, probs in base_probs.items():
        methods = {
            "raw": probs["test"],
            "platt": _platt_calibrate(probs["cal"], split["y_cal"], probs["test"]),
            "isotonic": _isotonic_calibrate(probs["cal"], split["y_cal"], probs["test"], config.min_isotonic_samples),
        }
        for method_name, test_prob in methods.items():
            if np.isnan(test_prob).all():
                continue
            metric = _metrics(split["y_test"], test_prob)
            rows.append(
                {
                    "model": model_name,
                    "calibration_method": method_name,
                    "features_used": ", ".join(features),
                    "train_size": int(len(split["y_train"])),
                    "calibration_size": int(len(split["y_cal"])),
                    "test_size": int(len(split["y_test"])),
                    **metric,
                }
            )
            bucket_rows.extend(_bucket_rows(model_name, method_name, split["y_test"], test_prob))
            threshold_rows.extend(_threshold_rows(split["test_dataset"], model_name, method_name, test_prob, config.thresholds))

    calibration = pd.DataFrame(rows).sort_values(["brier_score", "expected_calibration_error"], ascending=True)
    buckets = pd.DataFrame(bucket_rows)
    thresholds = pd.DataFrame(threshold_rows).sort_values(["Sharpe", "average_return"], ascending=False)
    calibration.to_csv(OUTPUT_CALIBRATION, index=False)
    buckets.to_csv(OUTPUT_BUCKETS, index=False)
    thresholds.to_csv(OUTPUT_THRESHOLDS, index=False)
    _print_report(calibration, buckets, thresholds)
    return {"calibration": calibration, "buckets": buckets, "thresholds": thresholds}


def _governance(calibration: pd.DataFrame, thresholds: pd.DataFrame) -> str:
    if calibration.empty:
        return "not enough data"
    best = calibration.iloc[0]
    brier = float(best.get("brier_score", np.inf))
    ece = float(best.get("expected_calibration_error", np.inf))
    best_threshold = thresholds.head(1)
    trades = int(best_threshold.iloc[0].get("trades_kept", 0)) if not best_threshold.empty else 0
    if trades < 100:
        return "not enough data"
    if brier > 0.25 or ece > 0.15:
        return "probabilities unreliable"
    if brier <= 0.245 and ece <= 0.10:
        return "usable for research"
    if brier <= 0.235 and ece <= 0.08:
        return "eligible for paper trading filter"
    return "usable for research"


def _print_report(calibration: pd.DataFrame, buckets: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    print("\n===== META PROBABILITY CALIBRATION =====")
    print(calibration.head(20).to_string(index=False) if not calibration.empty else "No calibration results.")

    print("\n===== CALIBRATION BUCKETS =====")
    print(buckets.head(30).to_string(index=False) if not buckets.empty else "No calibration buckets.")

    print("\n===== CALIBRATED THRESHOLD ANALYSIS =====")
    cols = ["model", "calibration_method", "threshold", "trades_kept", "sample_reduction", "TP_rate", "SL_rate", "hit_rate", "average_return", "Sharpe", "max_drawdown"]
    print(thresholds[cols].head(20).to_string(index=False) if not thresholds.empty else "No threshold analysis.")

    print("\n===== CALIBRATION GOVERNANCE =====")
    if not calibration.empty:
        best = calibration.iloc[0]
        print(f"best calibrated model: {best['model']}")
        print(f"best calibration method: {best['calibration_method']}")
        print(f"Brier score: {best['brier_score']}")
        print(f"ECE: {best['expected_calibration_error']}")
    if not thresholds.empty:
        best_t = thresholds.iloc[0]
        print(f"best threshold model: {best_t['model']}")
        print(f"best threshold method: {best_t['calibration_method']}")
        print(f"best threshold: {best_t['threshold']}")
        print(f"best threshold Sharpe: {best_t['Sharpe']}")
    print(f"governance classification: {_governance(calibration, thresholds)}")
    print("production behavior changed: False")


if __name__ == "__main__":
    run_meta_probability_calibration()
