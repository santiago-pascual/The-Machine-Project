from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


REPORT_FILE = "paper_meta_filter_report.csv"
FILTERED_ALLOCATION_FILE = "paper_meta_filtered_allocation.csv"


@dataclass
class PaperMetaFilterConfig:
    dataset_path: str = "meta_label_dataset.csv"
    selected_features_path: str = "selected_feature_set.json"
    model: str = "logistic_isotonic"
    threshold: float = 0.65
    min_train_samples: int = 500


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    return df


def _safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _load_features(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    remove = set(payload.get("REMOVE_FROM_ML", [])) | set(payload.get("DIAGNOSTIC_ONLY", []))
    features = list(payload.get("CORE", [])) + list(payload.get("SUPPORTING", []))
    return [feature for feature in dict.fromkeys(features) if feature not in remove]


def _prepare_training_data(
    config: PaperMetaFilterConfig, features: list[str], current_date: object | None
) -> tuple[pd.DataFrame, pd.Series] | None:
    dataset = _read_csv(config.dataset_path)
    if dataset.empty or "meta_label" not in dataset.columns:
        return None
    if current_date is not None and "date" in dataset.columns:
        dataset = dataset[dataset["date"] < pd.Timestamp(current_date)].copy()
    available = [feature for feature in features if feature in dataset.columns]
    if len(available) != len(features):
        return None
    dataset = dataset.sort_values("date").dropna(subset=["meta_label"]).copy()
    if len(dataset) < config.min_train_samples or dataset["meta_label"].nunique() < 2:
        return None
    x = dataset[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True).fillna(0.0))
    y = _safe_numeric(dataset["meta_label"], 0.0).astype(int)
    return x, y


def _current_feature_frame(
    *,
    final_allocation_table: pd.DataFrame,
    diagnostics_df_full: pd.DataFrame,
    expected_daily_returns: pd.Series | None,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    active = final_allocation_table.drop(index="CASH", errors="ignore").copy()
    if active.empty:
        return pd.DataFrame(), ["no_active_positions"]
    tickers = active.index.astype(str).tolist()
    diag = diagnostics_df_full.copy()
    diag.index = diag.index.astype(str)
    frame = pd.DataFrame(index=tickers)
    frame["original_weight"] = _safe_numeric(active.get("final_weight_decimal", pd.Series(index=active.index)), 0.0).reindex(tickers)
    if "signal_strength" in features:
        frame["signal_strength"] = _safe_numeric(diag.get("signal_strength", pd.Series(index=diag.index)), np.nan).reindex(tickers)
    if "expected_daily_return" in features:
        if expected_daily_returns is not None:
            frame["expected_daily_return"] = _safe_numeric(pd.Series(expected_daily_returns), np.nan).reindex(tickers)
        elif "expected_daily_return" in diag.columns:
            frame["expected_daily_return"] = _safe_numeric(diag["expected_daily_return"], np.nan).reindex(tickers)
        elif "expected_return" in diag.columns:
            frame["expected_daily_return"] = _safe_numeric(diag["expected_return"], np.nan).reindex(tickers)
    if "expected_total_return" in features:
        if "expected_total_return" in diag.columns:
            frame["expected_total_return"] = _safe_numeric(diag["expected_total_return"], np.nan).reindex(tickers)
        elif "expected_daily_return" in frame.columns:
            frame["expected_total_return"] = frame["expected_daily_return"] * 20.0
    if "daily_volatility" in features:
        for candidate in ["daily_volatility", "volatility", "kalman_residual_vol"]:
            if candidate in diag.columns:
                frame["daily_volatility"] = _safe_numeric(diag[candidate], np.nan).reindex(tickers)
                break
        if "daily_volatility" not in frame.columns:
            frame["daily_volatility"] = 0.0
        frame["daily_volatility"] = frame["daily_volatility"].fillna(0.0)
    missing = [feature for feature in features if feature not in frame.columns or frame[feature].isna().any()]
    return frame, missing


def _fit_probability_model(x: pd.DataFrame, y: pd.Series, current_x: pd.DataFrame) -> np.ndarray | None:
    if not SKLEARN_AVAILABLE:
        return None
    split = max(1, min(len(x) - 1, int(len(x) * 0.80)))
    x_train, x_cal = x.iloc[:split], x.iloc[split:]
    y_train, y_cal = y.iloc[:split], y.iloc[split:]
    if y_cal.nunique() < 2:
        return None
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_cal_scaled = scaler.transform(x_cal)
    current_scaled = scaler.transform(current_x)
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(x_train_scaled, y_train)
    cal_prob = model.predict_proba(x_cal_scaled)[:, 1]
    current_prob = model.predict_proba(current_scaled)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_prob, y_cal)
    return calibrator.predict(current_prob)


def _filtered_allocation(final_allocation_table: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    filtered = final_allocation_table.copy()
    if filtered.empty or report.empty:
        return filtered
    rejected = report[~report["meta_filter_pass"]]
    rejected_weight = float(pd.to_numeric(rejected["original_weight"], errors="coerce").fillna(0.0).sum())
    for ticker in rejected["ticker"].astype(str):
        if ticker in filtered.index:
            filtered.loc[ticker, ["final_weight_decimal", "final_weight_percent", "allocation_per_1000"]] = 0.0
    if "CASH" not in filtered.index:
        filtered.loc["CASH", ["final_weight_decimal", "final_weight_percent", "allocation_per_1000"]] = 0.0
    filtered.loc["CASH", "final_weight_decimal"] = float(filtered.loc["CASH", "final_weight_decimal"]) + rejected_weight
    filtered.loc["CASH", "final_weight_percent"] = filtered.loc["CASH", "final_weight_decimal"] * 100.0
    filtered.loc["CASH", "allocation_per_1000"] = filtered.loc["CASH", "final_weight_decimal"] * 1000.0
    return filtered.sort_values("final_weight_decimal", ascending=False)


def apply_paper_meta_filter(
    *,
    final_allocation_table: pd.DataFrame,
    diagnostics_df_full: pd.DataFrame,
    expected_daily_returns: pd.Series | None = None,
    current_date: object | None = None,
    config: PaperMetaFilterConfig | None = None,
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or PaperMetaFilterConfig()
    features = _load_features(config.selected_features_path)
    fail_reason = ""
    if not features:
        fail_reason = "selected_feature_set_missing_or_empty"
    elif not SKLEARN_AVAILABLE:
        fail_reason = "sklearn_unavailable"

    report = pd.DataFrame()
    filtered_allocation = final_allocation_table.copy()
    if fail_reason:
        report = _fail_report(final_allocation_table, fail_reason)
    else:
        training = _prepare_training_data(config, features, current_date)
        current_frame, missing = _current_feature_frame(
            final_allocation_table=final_allocation_table,
            diagnostics_df_full=diagnostics_df_full,
            expected_daily_returns=expected_daily_returns,
            features=features,
        )
        if training is None:
            report = _fail_report(final_allocation_table, "insufficient_historical_training_data")
        elif current_frame.empty:
            report = _fail_report(final_allocation_table, "no_current_positions")
        elif missing:
            report = _fail_report(final_allocation_table, f"missing_current_features: {', '.join(missing)}")
        else:
            x_train, y_train = training
            x_current = current_frame[features].apply(pd.to_numeric, errors="coerce")
            probabilities = _fit_probability_model(x_train[features], y_train, x_current)
            if probabilities is None:
                report = _fail_report(final_allocation_table, "model_training_or_calibration_failed")
            else:
                report = current_frame.reset_index(names="ticker")
                report["meta_probability"] = probabilities
                report["meta_filter_pass"] = report["meta_probability"] >= config.threshold
                report["filtered_weight"] = np.where(report["meta_filter_pass"], report["original_weight"], 0.0)
                report["meta_filter_reason"] = np.where(
                    report["meta_filter_pass"],
                    f"pass_probability_gte_{config.threshold}",
                    f"reject_probability_lt_{config.threshold}",
                )
                filtered_allocation = _filtered_allocation(final_allocation_table, report)

    report.to_csv(REPORT_FILE, index=False)
    filtered_allocation.to_csv(FILTERED_ALLOCATION_FILE, index=True)
    if print_report:
        print_paper_meta_filter_report(report, final_allocation_table, filtered_allocation, config)
    return filtered_allocation, report


def _fail_report(final_allocation_table: pd.DataFrame, reason: str) -> pd.DataFrame:
    active = final_allocation_table.drop(index="CASH", errors="ignore").copy()
    if active.empty:
        return pd.DataFrame(
            columns=["ticker", "original_weight", "meta_probability", "meta_filter_pass", "filtered_weight", "meta_filter_reason"]
        )
    return pd.DataFrame(
        {
            "ticker": active.index.astype(str),
            "original_weight": _safe_numeric(active.get("final_weight_decimal", pd.Series(index=active.index)), 0.0).values,
            "meta_probability": np.nan,
            "meta_filter_pass": True,
            "filtered_weight": _safe_numeric(active.get("final_weight_decimal", pd.Series(index=active.index)), 0.0).values,
            "meta_filter_reason": f"fail_safe_no_filter: {reason}",
        }
    )


def print_paper_meta_filter_report(
    report: pd.DataFrame,
    original_allocation: pd.DataFrame,
    filtered_allocation: pd.DataFrame,
    config: PaperMetaFilterConfig,
) -> None:
    print("\n===== PAPER META FILTER =====")
    if report.empty:
        print("No paper meta filter rows.")
    else:
        columns = ["ticker", "original_weight", "meta_probability", "meta_filter_pass", "filtered_weight", "meta_filter_reason"]
        print(report[[c for c in columns if c in report.columns]].to_string(index=False))

    print("\n===== PAPER META FILTER SUMMARY =====")
    original_active = original_allocation.drop(index="CASH", errors="ignore")
    filtered_active = filtered_allocation.drop(index="CASH", errors="ignore")
    before_cash = float(original_allocation.loc["CASH", "final_weight_decimal"]) if "CASH" in original_allocation.index else 0.0
    after_cash = float(filtered_allocation.loc["CASH", "final_weight_decimal"]) if "CASH" in filtered_allocation.index else 0.0
    rejected = (
        report.loc[~report.get("meta_filter_pass", pd.Series(True, index=report.index)).astype(bool), "ticker"].astype(str).tolist()
        if not report.empty
        else []
    )
    print(f"model: {config.model}")
    print(f"threshold: {config.threshold}")
    print(f"positions before filter: {len(original_active)}")
    print(
        f"positions after filter: {int((pd.to_numeric(filtered_active.get('final_weight_decimal', pd.Series(dtype=float)), errors='coerce').fillna(0.0) > 0).sum())}"
    )
    print(f"cash before: {before_cash:.6f}")
    print(f"cash after: {after_cash:.6f}")
    print(f"rejected tickers: {', '.join(rejected) if rejected else 'none'}")
    print("expected impact based on walk-forward validation: Sharpe 1.2495 vs 0.9730 baseline; research only")
    print(f"saved: {REPORT_FILE}, {FILTERED_ALLOCATION_FILE}")


if __name__ == "__main__":
    print("paper_meta_filter.py is a library module. Enable through PAPER_META_FILTER_ENABLED=1 during paper trading.")
