from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from risk_metrics import compute_return_risk_metrics


DEFAULT_OUTPUT_FILE = "regime_performance_attribution.csv"


def _load_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    data = pd.read_csv(file_path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "ticker" in data.columns:
        data["ticker"] = data["ticker"].astype(str)
    return data


def _regime_column(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for col in ("regime", "market_mode", "spy_macro_regime"):
        if col in df.columns:
            return df[col].fillna("unknown").astype(str), col
    return pd.Series("unknown", index=df.index, dtype=str), "missing"


def _max_drawdown(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _portfolio_metrics_by_regime(portfolio_df: pd.DataFrame, regime_by_date: pd.Series) -> pd.DataFrame:
    if portfolio_df.empty:
        return pd.DataFrame()
    portfolio = portfolio_df.copy()
    portfolio["regime"] = portfolio["date"].map(regime_by_date).fillna("unknown")
    rows: list[dict[str, object]] = []

    for regime, group in portfolio.groupby("regime"):
        returns_1d = pd.to_numeric(group.get("realized_portfolio_return_1d"), errors="coerce").dropna()
        risk = compute_return_risk_metrics(returns_1d)
        row = {
            "regime": regime,
            "portfolio_observations": int(len(group)),
            "portfolio_return_5d": _safe_mean(group, "realized_portfolio_return_5d"),
            "portfolio_return_10d": _safe_mean(group, "realized_portfolio_return_10d"),
            "portfolio_return_20d": _safe_mean(group, "realized_portfolio_return_20d"),
            "portfolio_volatility": float(risk["annualized_volatility"]),
            "portfolio_sharpe": float(risk["annualized_return_estimate"] / risk["annualized_volatility"])
            if risk["annualized_volatility"] > 0
            else 0.0,
            "portfolio_sortino": float(risk["sortino_ratio"]),
            "portfolio_max_drawdown": _max_drawdown(returns_1d),
            "average_cash": _safe_mean(group, "cash_weight"),
            "average_selected_count": _safe_mean(group, "selected_count"),
            "average_turnover": _safe_mean(group, "turnover"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _signal_metrics_by_regime(predictions_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    regime_series, regime_source = _regime_column(predictions_df)
    predictions = predictions_df.copy()
    predictions["regime"] = regime_series

    if labels_df.empty:
        merged = predictions.copy()
    else:
        labels = labels_df.copy()
        merged = labels.merge(
            predictions,
            on=["date", "ticker"],
            how="left",
            suffixes=("", "_prediction"),
        )
        if "regime" not in merged.columns:
            merged["regime"] = merged.get("regime_prediction", "unknown")
        merged["regime"] = merged["regime"].fillna("unknown").astype(str)

    rows: list[dict[str, object]] = []
    for regime, group in merged.groupby("regime"):
        first_touch = group.get("first_touch_type", pd.Series(dtype=object)).astype(str)
        label = pd.to_numeric(group.get("label", pd.Series(dtype=float)), errors="coerce")
        realized = pd.to_numeric(group.get("realized_return_at_barrier", pd.Series(dtype=float)), errors="coerce")
        expected = pd.to_numeric(group.get("expected_daily_return", pd.Series(dtype=float)), errors="coerce")
        row = {
            "regime": regime,
            "signal_observations": int(len(group)),
            "regime_label_source": regime_source,
            "TP_rate": float((first_touch == "take_profit").mean()) if len(group) else np.nan,
            "SL_rate": float((first_touch == "stop_loss").mean()) if len(group) else np.nan,
            "timeout_rate": float((first_touch == "vertical_timeout").mean()) if len(group) else np.nan,
            "hit_rate": float((realized > 0).mean()) if realized.notna().any() else np.nan,
            "direction_accuracy": float((np.sign(expected) == np.sign(realized)).mean())
            if expected.notna().any() and realized.notna().any()
            else np.nan,
            "average_signal_realized_return": float(realized.mean()) if realized.notna().any() else np.nan,
            "average_signal_strength": _safe_mean(group, "signal_strength"),
            "average_target_confidence": _safe_mean(group, "target_confidence"),
            "average_quality_score": _safe_mean(group, "quality_score"),
            "average_volatility": _safe_mean(group, "volatility"),
            "average_expected_return": _safe_mean(group, "expected_daily_return"),
        }
        row["TP_minus_SL"] = row["TP_rate"] - row["SL_rate"] if np.isfinite(row["TP_rate"]) and np.isfinite(row["SL_rate"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _weakness_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("TP_rate", np.nan) < row.get("SL_rate", np.nan):
        reasons.append("TP_rate_below_SL_rate")
    if row.get("portfolio_sharpe", np.nan) < 0:
        reasons.append("negative_portfolio_sharpe")
    if row.get("average_signal_realized_return", np.nan) < 0:
        reasons.append("negative_average_signal_return")
    if row.get("average_turnover", 0.0) > 0.60 and row.get("portfolio_return_5d", 0.0) < 0.005:
        reasons.append("high_turnover_low_return")
    if row.get("portfolio_sharpe", 0.0) < 0 and row.get("average_cash", 1.0) < 0.30:
        reasons.append("cash_too_low_bad_regime")
    if row.get("average_target_confidence", 0.0) > 0.65 and row.get("average_signal_realized_return", 0.0) < 0:
        reasons.append("high_confidence_bad_performance")
    return ", ".join(reasons) if reasons else "ok"


def run_regime_performance_attribution(
    predictions_path: str | Path = "walk_forward_predictions.csv",
    portfolio_returns_path: str | Path = "walk_forward_portfolio_returns.csv",
    triple_barrier_labels_path: str | Path = "triple_barrier_labels.csv",
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    predictions = _load_csv(predictions_path)
    portfolio = _load_csv(portfolio_returns_path)
    labels = _load_csv(triple_barrier_labels_path)

    if predictions.empty and portfolio.empty:
        result = pd.DataFrame()
        result.to_csv(output_path, index=False)
        print("\n===== REGIME PERFORMANCE ATTRIBUTION =====")
        print("No walk-forward prediction/portfolio data available.")
        return result

    regime_series, regime_source = _regime_column(predictions) if not predictions.empty else (pd.Series(dtype=str), "missing")
    regime_by_date = pd.Series(regime_series.values, index=predictions["date"]).groupby(level=0).first() if not predictions.empty else pd.Series(dtype=str)
    portfolio_metrics = _portfolio_metrics_by_regime(portfolio, regime_by_date)
    signal_metrics = _signal_metrics_by_regime(predictions, labels)

    if portfolio_metrics.empty:
        result = signal_metrics
    elif signal_metrics.empty:
        result = portfolio_metrics
    else:
        result = portfolio_metrics.merge(signal_metrics, on="regime", how="outer")

    if "regime_label_source" not in result.columns:
        result["regime_label_source"] = regime_source
    result["weakness_flags"] = result.apply(_weakness_reasons, axis=1)
    result.to_csv(output_path, index=False)

    print("\n===== REGIME PERFORMANCE ATTRIBUTION =====")
    display_cols = [
        "regime",
        "signal_observations",
        "portfolio_return_5d",
        "portfolio_sharpe",
        "TP_rate",
        "SL_rate",
        "TP_minus_SL",
        "hit_rate",
        "average_cash",
        "average_turnover",
    ]
    existing = [col for col in display_cols if col in result.columns]
    print(result[existing].to_string(index=False))

    print("\n===== REGIME WEAKNESS REPORT =====")
    weak = result[result["weakness_flags"] != "ok"]
    if weak.empty:
        print("No weak regimes flagged.")
    else:
        print(weak[["regime", "weakness_flags"]].to_string(index=False))
    if regime_source == "missing":
        print("Regime info incomplete: no regime/market_mode/spy_macro_regime column found.")
    return result
