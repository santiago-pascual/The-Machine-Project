from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_AUDIT = "target_engineering_audit.csv"
OUTPUT_REGIME = "forecast_error_by_regime.csv"
OUTPUT_TICKER = "forecast_error_by_ticker.csv"
OUTPUT_VOLATILITY = "forecast_error_by_volatility.csv"
OUTPUT_CONFIDENCE = "confidence_audit.csv"


@dataclass
class TargetAuditConfig:
    snapshots_path: str = "historical_forecast_snapshots.csv"
    realized_returns_path: str = "historical_realized_returns.csv"
    ic_dataset_path: str = "historical_ic_dataset.csv"
    triple_barrier_path: str = "historical_triple_barrier_labels.csv"
    feature_store_path: str = "historical_feature_store.csv"
    horizons: tuple[int, ...] = (5, 10, 20)


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


def _spearman_no_scipy(a: pd.Series, b: pd.Series) -> float:
    frame = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if frame.empty or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return np.nan
    value = frame["a"].rank().corr(frame["b"].rank(), method="pearson")
    return float(value) if np.isfinite(value) else np.nan


def _calibration_slope(forecast: pd.Series, realized: pd.Series) -> float:
    frame = pd.DataFrame({"forecast": forecast, "realized": realized}).dropna()
    if len(frame) < 3:
        return np.nan
    x = frame["forecast"].to_numpy(dtype=float)
    y = frame["realized"].to_numpy(dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x, y) / denom)


def _forecast_metrics(frame: pd.DataFrame, forecast_col: str, realized_col: str) -> dict[str, float]:
    data = frame[[forecast_col, realized_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {
            "sample_size": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "bias": np.nan,
            "forecast_dispersion": np.nan,
            "pearson_corr": np.nan,
            "spearman_ic": np.nan,
            "calibration_slope": np.nan,
        }
    error = data[forecast_col] - data[realized_col]
    denom = data[realized_col].abs().replace(0, np.nan)
    return {
        "sample_size": int(len(data)),
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "MAPE": float((error.abs() / denom).replace([np.inf, -np.inf], np.nan).mean()),
        "bias": float(error.mean()),
        "forecast_dispersion": float(data[forecast_col].std(ddof=0)),
        "pearson_corr": float(data[forecast_col].corr(data[realized_col])) if data[forecast_col].nunique() > 1 and data[realized_col].nunique() > 1 else np.nan,
        "spearman_ic": _spearman_no_scipy(data[forecast_col], data[realized_col]),
        "calibration_slope": _calibration_slope(data[forecast_col], data[realized_col]),
    }


def _prepare_base(config: TargetAuditConfig) -> pd.DataFrame:
    snapshots = _read_csv(config.snapshots_path)
    realized = _read_csv(config.realized_returns_path)
    if snapshots.empty:
        return pd.DataFrame()
    base = snapshots.copy()
    merge_keys = ["date", "ticker", "model_mode"]
    if not realized.empty:
        realized_cols = merge_keys + [c for c in realized.columns if c.startswith("realized_return_")]
        base = base.drop(columns=[c for c in realized_cols if c in base.columns and c not in merge_keys], errors="ignore")
        base = base.merge(realized[realized_cols], on=merge_keys, how="left")
    base["forecast_return"] = _safe_numeric(base.get("expected_total_return", pd.Series(np.nan, index=base.index)), np.nan)
    if "current_price" in base.columns and "target_price" in base.columns:
        current = _safe_numeric(base["current_price"], np.nan)
        target = _safe_numeric(base["target_price"], np.nan)
        target_return = target / current.replace(0, np.nan) - 1.0
        base["target_implied_return"] = target_return.replace([np.inf, -np.inf], np.nan)
    else:
        base["target_implied_return"] = base["forecast_return"]
    return base


def _overall_audit(base: pd.DataFrame, config: TargetAuditConfig) -> pd.DataFrame:
    rows = []
    for horizon in config.horizons:
        realized_col = f"realized_return_{horizon}d"
        if realized_col not in base.columns:
            continue
        for forecast_col in ["forecast_return", "target_implied_return"]:
            metrics = _forecast_metrics(base, forecast_col, realized_col)
            rows.append(
                {
                    "section": "forecast_error_analysis",
                    "horizon": f"{horizon}D",
                    "forecast_source": forecast_col,
                    **metrics,
                    "bias_type": _bias_type(metrics.get("bias", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _bias_type(bias: float) -> str:
    if not np.isfinite(bias):
        return "unknown"
    if bias > 0.002:
        return "optimistic_bias"
    if bias < -0.002:
        return "pessimistic_bias"
    return "well_centered"


def _attribution(base: pd.DataFrame, group_col: str, config: TargetAuditConfig) -> pd.DataFrame:
    rows = []
    if group_col not in base.columns:
        return pd.DataFrame()
    for horizon in config.horizons:
        realized_col = f"realized_return_{horizon}d"
        if realized_col not in base.columns:
            continue
        for group, frame in base.groupby(group_col, dropna=False):
            metrics = _forecast_metrics(frame, "forecast_return", realized_col)
            rows.append({"group_type": group_col, "group": str(group), "horizon": f"{horizon}D", **metrics, "bias_type": _bias_type(metrics.get("bias", np.nan))})
    return pd.DataFrame(rows)


def _quintile_attribution(base: pd.DataFrame, column: str, label: str, config: TargetAuditConfig) -> pd.DataFrame:
    if column not in base.columns:
        return pd.DataFrame()
    frame = base.copy()
    values = _safe_numeric(frame[column], np.nan)
    if values.nunique(dropna=True) < 3:
        return pd.DataFrame()
    try:
        frame[label] = pd.qcut(values, q=5, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    return _attribution(frame, label, config)


def _confidence_audit(base: pd.DataFrame, config: TargetAuditConfig) -> pd.DataFrame:
    if "target_confidence" not in base.columns:
        output = pd.DataFrame()
        output.to_csv(OUTPUT_CONFIDENCE, index=False)
        return output
    frame = base.copy()
    confidence = _safe_numeric(frame["target_confidence"], np.nan)
    try:
        frame["confidence_quintile"] = pd.qcut(confidence, q=5, duplicates="drop")
    except ValueError:
        frame["confidence_quintile"] = "single_bucket"
    rows = []
    for horizon in config.horizons:
        realized_col = f"realized_return_{horizon}d"
        if realized_col not in frame.columns:
            continue
        frame["abs_error"] = (_safe_numeric(frame["forecast_return"], np.nan) - _safe_numeric(frame[realized_col], np.nan)).abs()
        frame["success"] = _safe_numeric(frame[realized_col], np.nan).gt(0)
        corr_accuracy = _spearman_no_scipy(confidence, -frame["abs_error"])
        corr_success = _spearman_no_scipy(confidence, frame["success"].astype(float))
        for bucket, group in frame.groupby("confidence_quintile", observed=False, dropna=False):
            rows.append(
                {
                    "horizon": f"{horizon}D",
                    "confidence_bucket": str(bucket),
                    "sample_size": int(len(group)),
                    "avg_confidence": float(_safe_numeric(group["target_confidence"], np.nan).mean(skipna=True)),
                    "avg_abs_error": float(group["abs_error"].mean(skipna=True)),
                    "success_rate": float(group["success"].mean(skipna=True)),
                    "confidence_vs_accuracy_spearman": corr_accuracy,
                    "confidence_vs_success_spearman": corr_success,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_CONFIDENCE, index=False)
    return output


def _target_quality(base: pd.DataFrame, group_col: str, config: TargetAuditConfig) -> pd.DataFrame:
    attr = _attribution(base, group_col, config)
    if attr.empty:
        return attr
    return attr.sort_values(["MAE", "RMSE"], ascending=True)


def run_target_engineering_audit(config: TargetAuditConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or TargetAuditConfig()
    base = _prepare_base(config)
    if base.empty:
        empty = pd.DataFrame()
        for path in [OUTPUT_AUDIT, OUTPUT_REGIME, OUTPUT_TICKER, OUTPUT_VOLATILITY, OUTPUT_CONFIDENCE]:
            empty.to_csv(path, index=False)
        return {"audit": empty, "regime": empty, "ticker": empty, "volatility": empty, "confidence": empty}

    audit = _overall_audit(base, config)
    by_regime = _target_quality(base, "regime", config)
    by_ticker = _target_quality(base, "ticker", config)
    vol_source = "daily_volatility" if "daily_volatility" in base.columns else "target_implied_return"
    by_volatility = _quintile_attribution(base, vol_source, "volatility_quintile", config)
    signal_quintile = _quintile_attribution(base, "signal_strength", "signal_strength_quintile", config)
    expected_quintile = _quintile_attribution(base, "forecast_return", "expected_return_quintile", config)
    error_attribution = pd.concat([df for df in [by_regime.assign(output="regime"), by_ticker.assign(output="ticker"), by_volatility.assign(output="volatility"), signal_quintile.assign(output="signal_strength"), expected_quintile.assign(output="expected_return")] if not df.empty], ignore_index=True)
    confidence = _confidence_audit(base, config)

    audit.to_csv(OUTPUT_AUDIT, index=False)
    by_regime.to_csv(OUTPUT_REGIME, index=False)
    by_ticker.to_csv(OUTPUT_TICKER, index=False)
    by_volatility.to_csv(OUTPUT_VOLATILITY, index=False)
    _print_report(audit, error_attribution, by_ticker, by_regime, confidence)
    return {"audit": audit, "regime": by_regime, "ticker": by_ticker, "volatility": by_volatility, "confidence": confidence}


def _print_report(audit: pd.DataFrame, attribution: pd.DataFrame, by_ticker: pd.DataFrame, by_regime: pd.DataFrame, confidence: pd.DataFrame) -> None:
    print("\n===== TARGET ENGINEERING AUDIT =====")
    print("research only: True")
    print("production behavior changed: False")

    print("\n===== FORECAST ERROR ANALYSIS =====")
    cols = ["horizon", "forecast_source", "sample_size", "MAE", "RMSE", "MAPE", "bias", "forecast_dispersion", "pearson_corr", "spearman_ic", "calibration_slope", "bias_type"]
    print(audit[[c for c in cols if c in audit.columns]].to_string(index=False) if not audit.empty else "No audit rows.")

    print("\n===== ERROR ATTRIBUTION =====")
    if attribution.empty:
        print("No attribution rows.")
    else:
        print("Best predicted tickers:")
        print(by_ticker[["group", "horizon", "sample_size", "MAE", "bias_type"]].head(10).to_string(index=False))
        print("Worst predicted tickers:")
        print(by_ticker[["group", "horizon", "sample_size", "MAE", "bias_type"]].tail(10).to_string(index=False))
        print("Regime ranking:")
        print(by_regime[["group", "horizon", "sample_size", "MAE", "bias", "bias_type"]].head(15).to_string(index=False))

    print("\n===== CONFIDENCE AUDIT =====")
    if confidence.empty:
        print("No confidence audit.")
    else:
        print(confidence.head(20).to_string(index=False))

    print("\n===== TARGET GOVERNANCE =====")
    if audit.empty:
        print("No governance.")
        return
    optimistic = audit[audit["bias_type"].eq("optimistic_bias")]
    avg_ic = audit["spearman_ic"].mean(skipna=True)
    avg_slope = audit["calibration_slope"].mean(skipna=True)
    print(f"average Spearman IC: {avg_ic:.6f}")
    print(f"average calibration slope: {avg_slope:.6f}")
    print(f"optimistic bias rows: {len(optimistic)}")
    if len(optimistic) > 0 or (np.isfinite(avg_slope) and avg_slope < 0.5):
        print("recommendation: recalibrate target magnitude before using as predictive alpha")
    else:
        print("recommendation: target engine acceptable for research diagnostics")


if __name__ == "__main__":
    run_target_engineering_audit()
