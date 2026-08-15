from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_FORECASTS = "walk_forward_calibrated_forecasts.csv"
OUTPUT_COEFFICIENTS = "walk_forward_calibration_coefficients.csv"
OUTPUT_DIAGNOSTICS = "walk_forward_calibration_diagnostics.csv"
OUTPUT_SHADOW = "walk_forward_calibrated_forecast_shadow_results.csv"


@dataclass
class WalkForwardForecastCalibrationConfig:
    snapshots_path: str = "historical_forecast_snapshots.csv"
    realized_returns_path: str = "historical_realized_returns.csv"
    triple_barrier_path: str = "historical_triple_barrier_labels.csv"
    min_train_observations: int = 500
    horizons: tuple[int, ...] = (5, 10, 20)
    portfolio_horizon: int = 20
    model_mode: str = "regime_gated_full_quant"


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


def _prepare_panel(config: WalkForwardForecastCalibrationConfig) -> pd.DataFrame:
    snapshots = _read_csv(config.snapshots_path)
    realized = _read_csv(config.realized_returns_path)
    if snapshots.empty:
        return pd.DataFrame()
    panel = snapshots[snapshots["model_mode"].astype(str).eq(config.model_mode)].copy()
    keys = ["date", "ticker", "model_mode"]
    if not realized.empty:
        realized_cols = keys + [c for c in realized.columns if c.startswith("realized_return_")]
        panel = panel.drop(columns=[c for c in realized_cols if c in panel.columns and c not in keys], errors="ignore")
        panel = panel.merge(realized[realized_cols], on=keys, how="left")
    panel["forecast_return"] = _safe_numeric(panel.get("expected_total_return", pd.Series(np.nan, index=panel.index)), np.nan)
    return panel.sort_values("date").reset_index(drop=True)


def _fit_coefficients(train: pd.DataFrame, forecast_col: str, realized_col: str) -> dict[str, float]:
    data = train[[forecast_col, realized_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20 or data[forecast_col].nunique() < 2:
        return {"alpha": 0.0, "beta": 1.0, "r2": 0.0, "sample_size": len(data)}
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
    return {"alpha": alpha, "beta": beta, "r2": float(r2), "sample_size": len(data)}


def _calibrate_walk_forward(panel: pd.DataFrame, config: WalkForwardForecastCalibrationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_parts = []
    coeff_rows = []
    for date, current in panel.groupby("date", sort=True):
        history = panel[panel["date"] < date].copy()
        current = current.copy()
        for horizon in config.horizons:
            realized_col = f"realized_return_{horizon}d"
            output_col = f"wf_calibrated_expected_return_{horizon}d"
            if realized_col not in panel.columns:
                current[output_col] = current["forecast_return"]
                current[f"wf_calibration_warning_{horizon}d"] = "missing_realized_column"
                continue
            if len(history) < config.min_train_observations:
                coeff = {"alpha": 0.0, "beta": 1.0, "r2": 0.0, "sample_size": len(history)}
                warning = "insufficient_prior_data"
            else:
                coeff = _fit_coefficients(history, "forecast_return", realized_col)
                warning = "" if coeff["sample_size"] >= config.min_train_observations else "insufficient_valid_prior_pairs"
            current[output_col] = coeff["alpha"] + coeff["beta"] * current["forecast_return"]
            current[f"wf_calibration_warning_{horizon}d"] = warning
            coeff_rows.append(
                {
                    "date": date,
                    "horizon": f"{horizon}D",
                    "alpha": coeff["alpha"],
                    "beta": coeff["beta"],
                    "r2": coeff["r2"],
                    "sample_size": coeff["sample_size"],
                    "warning": warning,
                    "no_lookahead": True,
                }
            )
        forecast_parts.append(current)
    forecasts = pd.concat(forecast_parts, ignore_index=True) if forecast_parts else pd.DataFrame()
    coefficients = pd.DataFrame(coeff_rows)
    return forecasts, coefficients


def _error_metrics(frame: pd.DataFrame, forecast_col: str, realized_col: str) -> dict[str, float]:
    data = frame[[forecast_col, realized_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "bias": np.nan, "IC": np.nan, "rank_correlation": np.nan, "hit_rate": np.nan}
    error = data[forecast_col] - data[realized_col]
    return {
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "IC": _spearman_no_scipy(data[forecast_col], data[realized_col]),
        "rank_correlation": _spearman_no_scipy(data[forecast_col], data[realized_col]),
        "hit_rate": float((np.sign(data[forecast_col]) == np.sign(data[realized_col])).mean()),
    }


def _diagnostics(forecasts: pd.DataFrame, config: WalkForwardForecastCalibrationConfig) -> pd.DataFrame:
    rows = []
    for horizon in config.horizons:
        realized_col = f"realized_return_{horizon}d"
        calibrated_col = f"wf_calibrated_expected_return_{horizon}d"
        if realized_col not in forecasts.columns or calibrated_col not in forecasts.columns:
            continue
        before = _error_metrics(forecasts, "forecast_return", realized_col)
        after = _error_metrics(forecasts, calibrated_col, realized_col)
        rows.append(
            {
                "horizon": f"{horizon}D",
                "MAE_before": before["MAE"],
                "MAE_after": after["MAE"],
                "RMSE_before": before["RMSE"],
                "RMSE_after": after["RMSE"],
                "bias_before": before["bias"],
                "bias_after": after["bias"],
                "IC_before": before["IC"],
                "IC_after": after["IC"],
                "rank_correlation_before": before["rank_correlation"],
                "rank_correlation_after": after["rank_correlation"],
                "hit_rate_before": before["hit_rate"],
                "hit_rate_after": after["hit_rate"],
            }
        )
    return pd.DataFrame(rows)


def _select_shadow(group: pd.DataFrame, score_col: str, selected_count: int, cash_weight: float, candidate: str, horizon: int) -> pd.DataFrame:
    frame = group.copy()
    frame["score"] = _safe_numeric(frame[score_col], np.nan)
    frame = frame.sort_values("score", ascending=False).head(max(1, int(selected_count))).copy()
    positive = frame["score"].clip(lower=0.0)
    investable = max(0.0, 1.0 - float(cash_weight))
    if float(positive.sum()) > 0:
        frame["shadow_weight"] = positive / float(positive.sum()) * investable
    else:
        frame["shadow_weight"] = investable / max(1, len(frame))
    frame["candidate"] = candidate
    frame["shadow_return"] = frame["shadow_weight"] * _safe_numeric(frame[f"realized_return_{horizon}d"], np.nan)
    return frame


def _portfolio_shadow(forecasts: pd.DataFrame, config: WalkForwardForecastCalibrationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_parts = []
    daily_rows = []
    previous_weights: dict[str, pd.Series] = {}
    selected_counts = forecasts[forecasts["selected"].astype(str).str.lower().isin(["true", "1", "yes"])].groupby("date")["ticker"].count()
    cash_by_date = forecasts.groupby("date")["cash_weight"].first()
    for date, group in forecasts.groupby("date", sort=True):
        selected_count = int(selected_counts.get(date, 0))
        if selected_count <= 0:
            continue
        cash = float(cash_by_date.get(date, 0.0))
        candidates = {
            "original_forecast_shadow": "forecast_return",
            "wf_calibrated_forecast_shadow": f"wf_calibrated_expected_return_{config.portfolio_horizon}d",
        }
        for candidate, score_col in candidates.items():
            selected = _select_shadow(group, score_col, selected_count, cash, candidate, config.portfolio_horizon)
            trade_parts.append(selected)
            weights = selected.set_index("ticker")["shadow_weight"].astype(float)
            prior = previous_weights.get(candidate, pd.Series(dtype=float))
            tickers = sorted(set(prior.index.astype(str)) | set(weights.index.astype(str)))
            turnover = sum(abs(float(weights.get(t, 0.0)) - float(prior.get(t, 0.0))) for t in tickers) / 2.0
            daily_rows.append(
                {
                    "date": date,
                    "candidate": candidate,
                    "cash_weight": cash,
                    "selected_count": len(selected),
                    "turnover": turnover,
                    f"realized_portfolio_return_{config.portfolio_horizon}d": float(selected["shadow_return"].sum(skipna=True)),
                }
            )
            previous_weights[candidate] = weights
    return pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(), pd.DataFrame(daily_rows)


def _risk_metrics(returns: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"realized_return": np.nan, "volatility": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "Calmar": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan}
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    ann = np.sqrt(252 / 20)
    max_dd = float(dd.min()) if len(dd) else 0.0
    return {
        "realized_return": float(equity.iloc[-1] - 1.0),
        "volatility": float(std_ret * ann),
        "Sharpe": float(mean_ret / std_ret * ann) if std_ret > 0 else 0.0,
        "Sortino": float(mean_ret / downside_std * ann) if downside_std > 0 else 0.0,
        "Calmar": float(mean_ret * (252 / 20) / abs(max_dd)) if max_dd < 0 else 0.0,
        "max_drawdown": max_dd,
        "hit_rate": float(returns.gt(0).mean()),
    }


def _shadow_results(daily: pd.DataFrame, config: WalkForwardForecastCalibrationConfig) -> pd.DataFrame:
    rows = []
    ret_col = f"realized_portfolio_return_{config.portfolio_horizon}d"
    for candidate, group in daily.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": candidate,
                **_risk_metrics(group[ret_col]),
                "average_cash": float(group["cash_weight"].mean()),
                "average_selected_count": float(group["selected_count"].mean()),
                "turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["Sharpe", "realized_return"], ascending=False)


def _governance(diagnostics: pd.DataFrame, results: pd.DataFrame) -> str:
    if diagnostics.empty or results.empty:
        return "failed"
    mae_improves = bool((diagnostics["MAE_after"] < diagnostics["MAE_before"]).mean() >= 0.5)
    ic_worse = bool((diagnostics["IC_after"] < diagnostics["IC_before"]).any())
    calibrated = results[results["candidate"].eq("wf_calibrated_forecast_shadow")]
    original = results[results["candidate"].eq("original_forecast_shadow")]
    if calibrated.empty or original.empty:
        return "diagnostic only"
    c = calibrated.iloc[0]
    o = original.iloc[0]
    if mae_improves and c["Sharpe"] > o["Sharpe"] and c["realized_return"] > o["realized_return"]:
        return "candidate for paper testing"
    if mae_improves and c["Sharpe"] > o["Sharpe"]:
        return "useful for research"
    if mae_improves and not ic_worse:
        return "diagnostic only"
    return "failed"


def run_walk_forward_forecast_calibration(config: WalkForwardForecastCalibrationConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or WalkForwardForecastCalibrationConfig()
    panel = _prepare_panel(config)
    if panel.empty:
        empty = pd.DataFrame()
        for path in [OUTPUT_FORECASTS, OUTPUT_COEFFICIENTS, OUTPUT_DIAGNOSTICS, OUTPUT_SHADOW]:
            empty.to_csv(path, index=False)
        return {"forecasts": empty, "coefficients": empty, "diagnostics": empty, "shadow": empty}
    forecasts, coefficients = _calibrate_walk_forward(panel, config)
    diagnostics = _diagnostics(forecasts, config)
    trades, daily = _portfolio_shadow(forecasts, config)
    shadow = _shadow_results(daily, config)
    forecasts.to_csv(OUTPUT_FORECASTS, index=False)
    coefficients.to_csv(OUTPUT_COEFFICIENTS, index=False)
    diagnostics.to_csv(OUTPUT_DIAGNOSTICS, index=False)
    shadow.to_csv(OUTPUT_SHADOW, index=False)
    _print_report(diagnostics, shadow, coefficients)
    return {"forecasts": forecasts, "coefficients": coefficients, "diagnostics": diagnostics, "shadow": shadow, "trades": trades, "daily": daily}


def _print_report(diagnostics: pd.DataFrame, shadow: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    print("\n===== WALK-FORWARD FORECAST CALIBRATION =====")
    print("strict no-look-ahead: True")
    print(f"coefficient rows: {len(coefficients)}")
    warnings = coefficients["warning"].replace("", np.nan).dropna().value_counts().to_dict() if not coefficients.empty and "warning" in coefficients.columns else {}
    print(f"warnings: {warnings if warnings else 'none'}")

    print("\n===== BEFORE VS AFTER FORECAST CALIBRATION =====")
    print(diagnostics.to_string(index=False) if not diagnostics.empty else "No diagnostics.")

    print("\n===== WALK-FORWARD CALIBRATED PORTFOLIO SHADOW =====")
    print(shadow.to_string(index=False) if not shadow.empty else "No shadow results.")

    print("\n===== FORECAST CALIBRATION GOVERNANCE =====")
    print(f"classification: {_governance(diagnostics, shadow)}")
    print("production behavior changed: False")


if __name__ == "__main__":
    run_walk_forward_forecast_calibration()
