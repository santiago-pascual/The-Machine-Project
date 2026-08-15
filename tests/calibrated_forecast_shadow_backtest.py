from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_RESULTS = "calibrated_forecast_shadow_backtest_results.csv"
OUTPUT_DAILY = "calibrated_forecast_shadow_daily_returns.csv"
OUTPUT_TRADES = "calibrated_forecast_shadow_trades.csv"


@dataclass
class CalibratedShadowConfig:
    snapshots_path: str = "historical_forecast_snapshots.csv"
    realized_returns_path: str = "historical_realized_returns.csv"
    calibrated_diagnostics_path: str = "calibrated_forecast_diagnostics.csv"
    confidence_diagnostics_path: str = "calibrated_confidence_diagnostics.csv"
    coefficients_path: str = "forecast_calibration_coefficients.csv"
    historical_portfolio_path: str = "historical_walk_forward_portfolio_returns.csv"
    triple_barrier_path: str = "historical_triple_barrier_labels.csv"
    horizon: int = 20
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


def _error_metrics(frame: pd.DataFrame, forecast_col: str, realized_col: str) -> dict[str, float]:
    data = frame[[forecast_col, realized_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "bias": np.nan, "IC": np.nan}
    error = data[forecast_col] - data[realized_col]
    return {
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "IC": _spearman_no_scipy(data[forecast_col], data[realized_col]),
    }


def _load_shadow_panel(config: CalibratedShadowConfig) -> pd.DataFrame:
    calibrated = _read_csv(config.calibrated_diagnostics_path)
    if calibrated.empty:
        return pd.DataFrame()
    calibrated = calibrated[calibrated["model_mode"].astype(str).eq(config.model_mode)].copy()
    if calibrated.empty:
        return calibrated
    for col in ["forecast_return", f"calibrated_expected_return_{config.horizon}d", "calibrated_target_confidence"]:
        if col not in calibrated.columns:
            calibrated[col] = np.nan
    return calibrated


def _score_and_select(
    group: pd.DataFrame, score_col: str, selected_count: int, cash_weight: float, candidate: str, config: CalibratedShadowConfig
) -> pd.DataFrame:
    frame = group.copy()
    frame["ranking_score"] = _safe_numeric(frame[score_col], np.nan)
    if score_col.startswith("calibrated"):
        frame["ranking_score"] = frame["ranking_score"] * _safe_numeric(
            frame.get("calibrated_target_confidence", pd.Series(1.0, index=frame.index)), 1.0
        )
    frame = frame.sort_values("ranking_score", ascending=False).head(max(1, int(selected_count))).copy()
    positive = frame["ranking_score"].clip(lower=0.0)
    investable = max(0.0, 1.0 - float(cash_weight))
    if float(positive.sum()) > 0:
        weights = positive / float(positive.sum()) * investable
    else:
        weights = pd.Series(investable / max(1, len(frame)), index=frame.index)
    frame["shadow_weight"] = weights
    frame["candidate"] = candidate
    frame["shadow_cash_weight"] = cash_weight
    return frame


def _build_shadow_trades(config: CalibratedShadowConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _load_shadow_panel(config)
    historical = _read_csv(config.historical_portfolio_path)
    if panel.empty or historical.empty:
        return pd.DataFrame(), pd.DataFrame()
    historical = historical[historical["model_mode"].astype(str).eq(config.model_mode)].copy()
    counts = historical.set_index("date")[["cash_weight", "selected_count"]]
    trade_parts = []
    daily_rows = []
    previous_weights: dict[str, pd.Series] = {}
    for date, group in panel.groupby("date", sort=True):
        if date not in counts.index:
            continue
        cash = float(counts.loc[date, "cash_weight"])
        selected_count = int(counts.loc[date, "selected_count"])
        candidates = {
            "original_forecast_rerank": "forecast_return",
            "calibrated_forecast_rerank": f"calibrated_expected_return_{config.horizon}d",
        }
        for candidate, score_col in candidates.items():
            selected = _score_and_select(group, score_col, selected_count, cash, candidate, config)
            trade_parts.append(selected)
            returns = _safe_numeric(selected[f"realized_return_{config.horizon}d"], np.nan)
            weights = _safe_numeric(selected["shadow_weight"], 0.0)
            portfolio_return = float((weights * returns).sum(skipna=True))
            current_weights = selected.set_index("ticker")["shadow_weight"].astype(float)
            prior = previous_weights.get(candidate, pd.Series(dtype=float))
            all_tickers = sorted(set(prior.index.astype(str)) | set(current_weights.index.astype(str)))
            turnover = sum(abs(float(current_weights.get(t, 0.0)) - float(prior.get(t, 0.0))) for t in all_tickers) / 2.0
            daily_rows.append(
                {
                    "date": date,
                    "candidate": candidate,
                    "cash_weight": cash,
                    "selected_count": len(selected),
                    "turnover": turnover,
                    f"realized_portfolio_return_{config.horizon}d": portfolio_return,
                }
            )
            previous_weights[candidate] = current_weights
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    if not trades.empty:
        labels = _read_csv(config.triple_barrier_path)
        if not labels.empty:
            labels = labels[
                labels.get("horizon", pd.Series(dtype=float)).eq(config.horizon) & labels["model_mode"].astype(str).eq(config.model_mode)
            ][["date", "ticker", "model_mode", "label"]].drop_duplicates(["date", "ticker", "model_mode"])
            trades = trades.merge(labels, on=["date", "ticker", "model_mode"], how="left")
    daily = pd.DataFrame(daily_rows)
    production = historical.rename(columns={"model_mode": "candidate"}).copy()
    production["candidate"] = "production_original"
    keep_cols = ["date", "candidate", "cash_weight", "selected_count", "turnover", f"realized_portfolio_return_{config.horizon}d"]
    daily = pd.concat([production[[c for c in keep_cols if c in production.columns]], daily], ignore_index=True)
    return trades, daily


def _risk_metrics(returns: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {
            "realized_return": np.nan,
            "volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }
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


def _label_metrics(trades: pd.DataFrame, candidate: str, config: CalibratedShadowConfig) -> dict[str, float]:
    if trades.empty or "candidate" not in trades.columns:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    frame = trades[trades["candidate"].eq(candidate)].copy()
    if "label" not in frame.columns:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    labels = _safe_numeric(frame["label"], np.nan)
    tp = float(labels.eq(1).mean()) if labels.notna().any() else np.nan
    sl = float(labels.eq(-1).mean()) if labels.notna().any() else np.nan
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan}


def _results(trades: pd.DataFrame, daily: pd.DataFrame, panel: pd.DataFrame, config: CalibratedShadowConfig) -> pd.DataFrame:
    ret_col = f"realized_portfolio_return_{config.horizon}d"
    rows = []
    for candidate, group in daily.groupby("candidate", sort=False):
        metrics = _risk_metrics(group[ret_col])
        row = {
            "candidate": candidate,
            **metrics,
            "average_cash": float(group["cash_weight"].mean()) if "cash_weight" in group.columns else np.nan,
            "average_selected_count": float(group["selected_count"].mean()) if "selected_count" in group.columns else np.nan,
            "turnover": float(group["turnover"].mean()) if "turnover" in group.columns else np.nan,
        }
        row.update(_label_metrics(trades, candidate, config))
        rows.append(row)
    results = pd.DataFrame(rows)
    realized_col = f"realized_return_{config.horizon}d"
    if not panel.empty and realized_col in panel.columns:
        before = _error_metrics(panel, "forecast_return", realized_col)
        after = _error_metrics(panel, f"calibrated_expected_return_{config.horizon}d", realized_col)
        for col in ["MAE", "RMSE", "bias", "IC"]:
            results[f"forecast_{col}_before"] = before[col]
            results[f"forecast_{col}_after"] = after[col]
    results["calibration_type"] = "full_sample_shadow_diagnostic"
    results["walk_forward_valid"] = False
    return results.sort_values(["Sharpe", "realized_return"], ascending=False)


def _governance(results: pd.DataFrame) -> str:
    if results.empty:
        return "diagnostic only"
    calibrated = results[results["candidate"].eq("calibrated_forecast_rerank")]
    production = results[results["candidate"].eq("production_original")]
    if calibrated.empty or production.empty:
        return "diagnostic only"
    c = calibrated.iloc[0]
    p = production.iloc[0]
    if c["Sharpe"] > p["Sharpe"] and c["realized_return"] > p["realized_return"]:
        return "useful but needs walk-forward calibration"
    if c["Sharpe"] > p["Sharpe"]:
        return "candidate for research mode"
    return "diagnostic only"


def run_calibrated_forecast_shadow_backtest(config: CalibratedShadowConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or CalibratedShadowConfig()
    panel = _load_shadow_panel(config)
    trades, daily = _build_shadow_trades(config)
    results = _results(trades, daily, panel, config)
    trades.to_csv(OUTPUT_TRADES, index=False)
    daily.to_csv(OUTPUT_DAILY, index=False)
    results.to_csv(OUTPUT_RESULTS, index=False)
    _print_report(results)
    return {"results": results, "daily": daily, "trades": trades}


def _print_report(results: pd.DataFrame) -> None:
    print("\n===== CALIBRATED FORECAST SHADOW BACKTEST =====")
    if results.empty:
        print("No results.")
    else:
        cols = [
            "candidate",
            "realized_return",
            "volatility",
            "Sharpe",
            "Sortino",
            "Calmar",
            "max_drawdown",
            "TP_rate",
            "SL_rate",
            "TP_minus_SL",
            "hit_rate",
            "average_cash",
            "average_selected_count",
            "turnover",
        ]
        print(results[[c for c in cols if c in results.columns]].to_string(index=False))

    print("\n===== ORIGINAL VS CALIBRATED FORECASTS =====")
    if not results.empty:
        cols = [
            "candidate",
            "forecast_MAE_before",
            "forecast_MAE_after",
            "forecast_RMSE_before",
            "forecast_RMSE_after",
            "forecast_bias_before",
            "forecast_bias_after",
            "forecast_IC_before",
            "forecast_IC_after",
        ]
        print(results[[c for c in cols if c in results.columns]].drop_duplicates().to_string(index=False))

    print("\n===== CALIBRATED FORECAST GOVERNANCE =====")
    print("calibration type: full_sample_shadow_diagnostic")
    print("possible leakage: True")
    print("production behavior changed: False")
    print(f"classification: {_governance(results)}")


if __name__ == "__main__":
    run_calibrated_forecast_shadow_backtest()
