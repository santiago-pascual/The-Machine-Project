from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COEFFICIENTS = "forecast_calibration_coefficients.csv"
OUTPUT_DIAGNOSTICS = "calibrated_forecast_diagnostics.csv"
OUTPUT_CONFIDENCE = "calibrated_confidence_diagnostics.csv"


@dataclass
class ForecastCalibrationConfig:
    snapshots_path: str = "historical_forecast_snapshots.csv"
    realized_returns_path: str = "historical_realized_returns.csv"
    target_audit_path: str = "target_engineering_audit.csv"
    regime_error_path: str = "forecast_error_by_regime.csv"
    ticker_error_path: str = "forecast_error_by_ticker.csv"
    volatility_error_path: str = "forecast_error_by_volatility.csv"
    confidence_audit_path: str = "confidence_audit.csv"
    horizons: tuple[int, ...] = (5, 10, 20)
    train_size: float = 0.70
    min_group_samples: int = 150
    min_ticker_samples: int = 100


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
    corr = frame["a"].rank().corr(frame["b"].rank(), method="pearson")
    return float(corr) if np.isfinite(corr) else np.nan


def _prepare_base(config: ForecastCalibrationConfig) -> pd.DataFrame:
    snapshots = _read_csv(config.snapshots_path)
    realized = _read_csv(config.realized_returns_path)
    if snapshots.empty:
        return pd.DataFrame()
    base = snapshots.copy()
    keys = ["date", "ticker", "model_mode"]
    if not realized.empty:
        realized_cols = keys + [c for c in realized.columns if c.startswith("realized_return_")]
        base = base.drop(columns=[c for c in realized_cols if c in base.columns and c not in keys], errors="ignore")
        base = base.merge(realized[realized_cols], on=keys, how="left")
    base["forecast_return"] = _safe_numeric(base.get("expected_total_return", pd.Series(np.nan, index=base.index)), np.nan)
    base["signal_strength"] = _safe_numeric(base.get("signal_strength", pd.Series(0.0, index=base.index)), 0.0)
    base["target_confidence"] = _safe_numeric(base.get("target_confidence", pd.Series(0.5, index=base.index)), 0.5)
    if "regime" not in base.columns:
        base["regime"] = "unknown"
    if "daily_volatility" not in base.columns:
        base["daily_volatility"] = base["forecast_return"].abs()
    base["daily_volatility"] = _safe_numeric(base["daily_volatility"], np.nan)
    try:
        base["volatility_quintile"] = pd.qcut(base["daily_volatility"], q=5, duplicates="drop").astype(str)
    except ValueError:
        base["volatility_quintile"] = "single_bucket"
    return base.sort_values("date").reset_index(drop=True)


def _fit_linear_calibration(frame: pd.DataFrame, forecast_col: str, realized_col: str) -> dict[str, float]:
    data = frame[[forecast_col, realized_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20 or data[forecast_col].nunique() < 2:
        return {"alpha": 0.0, "beta": 1.0, "r2": 0.0, "sample_size": int(len(data))}
    x = data[forecast_col].to_numpy(dtype=float)
    y = data[realized_col].to_numpy(dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    beta = float(np.dot(x - x_mean, y - y_mean) / max(np.dot(x - x_mean, x - x_mean), 1e-12))
    alpha = y_mean - beta * x_mean
    pred = alpha + beta * x
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - y_mean)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"alpha": alpha, "beta": beta, "r2": float(r2), "sample_size": int(len(data))}


def _error_metrics(forecast: pd.Series, realized: pd.Series) -> dict[str, float]:
    data = pd.DataFrame({"forecast": forecast, "realized": realized}).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "bias": np.nan, "IC": np.nan}
    error = data["forecast"] - data["realized"]
    return {
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "IC": _spearman_no_scipy(data["forecast"], data["realized"]),
    }


def _coeff_row(scope: str, group: str, horizon: int, train: pd.DataFrame, test: pd.DataFrame, forecast_col: str, realized_col: str) -> dict[str, object]:
    coeff = _fit_linear_calibration(train, forecast_col, realized_col)
    calibrated_test = coeff["alpha"] + coeff["beta"] * _safe_numeric(test[forecast_col], np.nan)
    before = _error_metrics(test[forecast_col], test[realized_col])
    after = _error_metrics(calibrated_test, test[realized_col])
    return {
        "scope": scope,
        "group": group,
        "horizon": f"{horizon}D",
        **coeff,
        "MAE_before": before["MAE"],
        "MAE_after": after["MAE"],
        "RMSE_before": before["RMSE"],
        "RMSE_after": after["RMSE"],
        "bias_before": before["bias"],
        "bias_after": after["bias"],
        "IC_before": before["IC"],
        "IC_after": after["IC"],
    }


def _split_train_test(base: pd.DataFrame, train_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = max(1, min(len(base) - 1, int(len(base) * train_size)))
    return base.iloc[:split].copy(), base.iloc[split:].copy()


def _build_coefficients(base: pd.DataFrame, config: ForecastCalibrationConfig) -> pd.DataFrame:
    train, test = _split_train_test(base, config.train_size)
    rows = []
    for horizon in config.horizons:
        realized_col = f"realized_return_{horizon}d"
        if realized_col not in base.columns:
            continue
        rows.append(_coeff_row("global", "all", horizon, train, test, "forecast_return", realized_col))
        for regime, group_train in train.groupby("regime"):
            group_test = test[test["regime"].astype(str).eq(str(regime))]
            if len(group_train) >= config.min_group_samples and len(group_test) >= 20:
                rows.append(_coeff_row("regime", str(regime), horizon, group_train, group_test, "forecast_return", realized_col))
        for bucket, group_train in train.groupby("volatility_quintile"):
            group_test = test[test["volatility_quintile"].astype(str).eq(str(bucket))]
            if len(group_train) >= config.min_group_samples and len(group_test) >= 20:
                rows.append(_coeff_row("volatility_quintile", str(bucket), horizon, group_train, group_test, "forecast_return", realized_col))
        for ticker, group_train in train.groupby("ticker"):
            group_test = test[test["ticker"].astype(str).eq(str(ticker))]
            if len(group_train) >= config.min_ticker_samples and len(group_test) >= 20:
                rows.append(_coeff_row("ticker", str(ticker), horizon, group_train, group_test, "forecast_return", realized_col))
    return pd.DataFrame(rows)


def _coefficient_lookup(coefficients: pd.DataFrame, scope: str, group: str, horizon: int) -> dict[str, float] | None:
    if coefficients.empty:
        return None
    rows = coefficients[
        coefficients["scope"].astype(str).eq(scope)
        & coefficients["group"].astype(str).eq(str(group))
        & coefficients["horizon"].astype(str).eq(f"{horizon}D")
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {"alpha": float(row["alpha"]), "beta": float(row["beta"]), "sample_size": float(row.get("sample_size", 0.0))}


def _apply_calibration(base: pd.DataFrame, coefficients: pd.DataFrame, config: ForecastCalibrationConfig) -> pd.DataFrame:
    output = base.copy()
    warnings: list[str] = []
    methods: list[str] = []
    confidence_parts = []
    for horizon in config.horizons:
        values = []
        methods = []
        warnings = []
        for _, row in output.iterrows():
            coeff = (
                _coefficient_lookup(coefficients, "regime", row.get("regime", "unknown"), horizon)
                or _coefficient_lookup(coefficients, "volatility_quintile", row.get("volatility_quintile", "unknown"), horizon)
                or _coefficient_lookup(coefficients, "global", "all", horizon)
            )
            if coeff is None:
                coeff = {"alpha": 0.0, "beta": 1.0, "sample_size": 0.0}
                methods.append("identity_fallback")
                warnings.append("missing_coefficients")
            else:
                methods.append("regime_or_volatility_or_global")
                warnings.append("" if coeff["sample_size"] >= config.min_group_samples else "low_sample_coefficients")
            values.append(coeff["alpha"] + coeff["beta"] * float(row["forecast_return"]))
        output[f"calibrated_expected_return_{horizon}d"] = values
        if horizon == 20:
            output["calibration_method_used"] = methods
            output["calibration_warning"] = warnings

    reliability = _confidence_from_coefficients(output, coefficients, config)
    output["calibrated_target_confidence"] = reliability
    return output


def _confidence_from_coefficients(base: pd.DataFrame, coefficients: pd.DataFrame, config: ForecastCalibrationConfig) -> pd.Series:
    horizon = 20 if 20 in config.horizons else config.horizons[-1]
    regime_mae = coefficients[(coefficients["scope"].eq("regime")) & (coefficients["horizon"].eq(f"{horizon}D"))].set_index("group")["MAE_after"] if not coefficients.empty else pd.Series(dtype=float)
    global_mae_rows = coefficients[(coefficients["scope"].eq("global")) & (coefficients["horizon"].eq(f"{horizon}D"))]
    global_mae = float(global_mae_rows["MAE_after"].iloc[0]) if not global_mae_rows.empty else 0.10
    confidence = []
    forecast_dispersion = float(base["forecast_return"].std(ddof=0)) if len(base) else 0.0
    for _, row in base.iterrows():
        regime = str(row.get("regime", "unknown"))
        mae = float(regime_mae.get(regime, global_mae)) if not regime_mae.empty else global_mae
        reliability = 1.0 / (1.0 + max(mae, 0.0) / max(global_mae, 1e-6))
        signal = float(np.clip(row.get("signal_strength", 0.0), 0.0, 1.0))
        forecast_strength = min(1.0, abs(float(row.get("forecast_return", 0.0))) / max(forecast_dispersion, 1e-6))
        score = 0.50 * reliability + 0.30 * signal + 0.20 * forecast_strength
        confidence.append(float(np.clip(score, 0.0, 1.0)))
    return pd.Series(confidence, index=base.index)


def _confidence_diagnostics(calibrated: pd.DataFrame, config: ForecastCalibrationConfig) -> pd.DataFrame:
    rows = []
    for horizon in config.horizons:
        realized_col = f"realized_return_{horizon}d"
        calibrated_col = f"calibrated_expected_return_{horizon}d"
        if realized_col not in calibrated.columns or calibrated_col not in calibrated.columns:
            continue
        frame = calibrated.copy()
        frame["abs_error"] = (_safe_numeric(frame[calibrated_col], np.nan) - _safe_numeric(frame[realized_col], np.nan)).abs()
        frame["success"] = _safe_numeric(frame[realized_col], np.nan).gt(0)
        try:
            frame["confidence_bucket"] = pd.qcut(frame["calibrated_target_confidence"], q=5, duplicates="drop")
        except ValueError:
            frame["confidence_bucket"] = "single_bucket"
        corr_acc = _spearman_no_scipy(frame["calibrated_target_confidence"], -frame["abs_error"])
        corr_success = _spearman_no_scipy(frame["calibrated_target_confidence"], frame["success"].astype(float))
        for bucket, group in frame.groupby("confidence_bucket", observed=False, dropna=False):
            rows.append(
                {
                    "horizon": f"{horizon}D",
                    "confidence_bucket": str(bucket),
                    "sample_size": int(len(group)),
                    "avg_confidence": float(group["calibrated_target_confidence"].mean()),
                    "avg_abs_error": float(group["abs_error"].mean(skipna=True)),
                    "success_rate": float(group["success"].mean(skipna=True)),
                    "confidence_vs_accuracy_spearman": corr_acc,
                    "confidence_vs_success_spearman": corr_success,
                }
            )
    return pd.DataFrame(rows)


def run_forecast_calibration_layer(config: ForecastCalibrationConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or ForecastCalibrationConfig()
    base = _prepare_base(config)
    if base.empty:
        empty = pd.DataFrame()
        for path in [OUTPUT_COEFFICIENTS, OUTPUT_DIAGNOSTICS, OUTPUT_CONFIDENCE]:
            empty.to_csv(path, index=False)
        return {"coefficients": empty, "diagnostics": empty, "confidence": empty}
    coefficients = _build_coefficients(base, config)
    calibrated = _apply_calibration(base, coefficients, config)
    confidence = _confidence_diagnostics(calibrated, config)
    coefficients.to_csv(OUTPUT_COEFFICIENTS, index=False)
    calibrated.to_csv(OUTPUT_DIAGNOSTICS, index=False)
    confidence.to_csv(OUTPUT_CONFIDENCE, index=False)
    _print_report(coefficients, calibrated, confidence, config)
    return {"coefficients": coefficients, "diagnostics": calibrated, "confidence": confidence}


def _print_report(coefficients: pd.DataFrame, calibrated: pd.DataFrame, confidence: pd.DataFrame, config: ForecastCalibrationConfig) -> None:
    print("\n===== FORECAST CALIBRATION LAYER =====")
    print("research only: True")
    print("production behavior changed: False")
    print(f"coefficient rows: {len(coefficients)}")
    print(f"diagnostic rows: {len(calibrated)}")

    print("\n===== CALIBRATION BEFORE VS AFTER =====")
    if coefficients.empty:
        print("No coefficients.")
    else:
        cols = ["scope", "group", "horizon", "sample_size", "alpha", "beta", "r2", "MAE_before", "MAE_after", "RMSE_before", "RMSE_after", "bias_before", "bias_after", "IC_before", "IC_after"]
        print(coefficients[coefficients["scope"].eq("global")][cols].to_string(index=False))

    print("\n===== CALIBRATED CONFIDENCE REPORT =====")
    if confidence.empty:
        print("No confidence diagnostics.")
    else:
        print(confidence.head(20).to_string(index=False))

    print("\n===== FORECAST CALIBRATION GOVERNANCE =====")
    global_rows = coefficients[coefficients["scope"].eq("global")] if not coefficients.empty else pd.DataFrame()
    if global_rows.empty:
        print("recommendation: insufficient_data")
        return
    mae_improvement = (global_rows["MAE_before"] - global_rows["MAE_after"]).mean()
    rmse_improvement = (global_rows["RMSE_before"] - global_rows["RMSE_after"]).mean()
    bias_abs_improvement = (global_rows["bias_before"].abs() - global_rows["bias_after"].abs()).mean()
    ic_change = (global_rows["IC_after"] - global_rows["IC_before"]).mean()
    print(f"avg MAE improvement: {mae_improvement:.6f}")
    print(f"avg RMSE improvement: {rmse_improvement:.6f}")
    print(f"avg abs bias improvement: {bias_abs_improvement:.6f}")
    print(f"avg IC change: {ic_change:.6f}")
    if mae_improvement > 0 and bias_abs_improvement > 0:
        print("recommendation: candidate_for_shadow_comparison")
    else:
        print("recommendation: diagnostic_only_until_reworked")


if __name__ == "__main__":
    run_forecast_calibration_layer()
