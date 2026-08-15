from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
CALIBRATED_FILE = "walk_forward_calibrated_forecasts.csv"
TRIPLE_BARRIER_FILE = "historical_triple_barrier_labels.csv"
RESULTS_FILE = "calibrated_forecast_research_backtest_results.csv"
DAILY_RETURNS_FILE = "calibrated_forecast_research_backtest_daily_returns.csv"
TRADES_FILE = "calibrated_forecast_research_backtest_trades.csv"
GOVERNANCE_FILE = "calibrated_forecast_research_governance.csv"
TRADING_DAYS = 252


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _max_drawdown(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _sortino(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return np.nan
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0))
    if downside_std <= 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (downside_std * np.sqrt(TRADING_DAYS)))


def _calmar(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    mdd = abs(_max_drawdown(returns))
    if returns.empty or not np.isfinite(mdd) or mdd <= 0:
        return np.nan
    annualized = (1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0
    return float(annualized / mdd)


def _portfolio_rows_from_predictions(predictions: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    previous_assets: set[str] = set()
    selected_df = predictions[_as_bool(predictions["selected"])].copy()
    if selected_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected_df["date"] = selected_df["date"].astype(str)
    for date, group in selected_df.groupby("date", sort=True):
        weights = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        realized = pd.to_numeric(group["realized_return_1d"], errors="coerce")
        valid = weights.gt(0) & realized.notna()
        group = group.loc[valid].copy()
        weights = weights.loc[valid]
        realized = realized.loc[valid]
        selected_assets = set(group["ticker"].astype(str))
        if group.empty or float(weights.sum()) <= 0:
            period_return = 0.0
            cash_weight = 1.0
        else:
            weight_sum = float(weights.sum())
            cash_weight = max(0.0, 1.0 - weight_sum)
            period_return = float((weights * realized).sum())
        rows.append(
            {
                "date": date,
                "model_mode": mode,
                "portfolio_return": period_return,
                "cash_weight": cash_weight,
                "selected_count": len(selected_assets),
                "turnover": len(selected_assets.symmetric_difference(previous_assets)) / max(1, len(selected_assets | previous_assets)),
                "selected_tickers": ",".join(sorted(selected_assets)),
            }
        )
        for _, trade in group.iterrows():
            trades.append(
                {
                    "date": date,
                    "model_mode": mode,
                    "ticker": str(trade["ticker"]),
                    "selected": True,
                    "weight": _safe_float(trade.get("weight"), 0.0),
                    "expected_daily_return": _safe_float(trade.get("expected_daily_return"), 0.0),
                    "realized_return_1d": _safe_float(trade.get("realized_return_1d"), np.nan),
                    "realized_return_5d": _safe_float(trade.get("realized_return_5d"), np.nan),
                    "realized_return_10d": _safe_float(trade.get("realized_return_10d"), np.nan),
                    "realized_return_20d": _safe_float(trade.get("realized_return_20d"), np.nan),
                    "cash_weight": cash_weight,
                }
            )
        previous_assets = selected_assets
    return pd.DataFrame(rows), pd.DataFrame(trades)


def _build_existing_mode(snapshots: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = snapshots[snapshots["model_mode"].astype(str).eq(mode)].copy()
    return _portfolio_rows_from_predictions(data, mode)


def _merge_realized_returns(frame: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or realized.empty:
        return frame
    merge_cols = ["date", "ticker", "model_mode"]
    if not set(merge_cols).issubset(frame.columns) or not set(merge_cols).issubset(realized.columns):
        return frame
    realized_cols = [
        "realized_return_1d",
        "realized_return_5d",
        "realized_return_10d",
        "realized_return_20d",
        "realized_return_30d",
    ]
    available = [col for col in realized_cols if col in realized.columns]
    base = frame.drop(columns=[col for col in available if col in frame.columns], errors="ignore").copy()
    merged = base.merge(realized[merge_cols + available], on=merge_cols, how="left")
    return merged


def _build_calibrated_mode(calibrated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data = calibrated[calibrated["model_mode"].astype(str).eq("regime_gated_full_quant")].copy()
    metadata = {
        "calibrated_rows": len(data),
        "fallback_count": 0,
        "average_calibration_adjustment": np.nan,
        "selection_overlap_avg": np.nan,
        "warning": "none",
    }
    if data.empty:
        metadata["warning"] = "missing_calibrated_rows"
        return pd.DataFrame(), pd.DataFrame(), metadata

    data["date"] = data["date"].astype(str)
    calibrated_total = pd.to_numeric(data["wf_calibrated_expected_return_20d"], errors="coerce")
    original_daily = pd.to_numeric(data["expected_daily_return"], errors="coerce")
    calibrated_daily = np.power(1.0 + calibrated_total.clip(lower=-0.999), 1.0 / 20.0) - 1.0
    data["calibrated_expected_daily_return"] = calibrated_daily
    fallback_mask = data["calibrated_expected_daily_return"].isna()
    data.loc[fallback_mask, "calibrated_expected_daily_return"] = original_daily.loc[fallback_mask]
    metadata["fallback_count"] = int(fallback_mask.sum())
    metadata["average_calibration_adjustment"] = float((data["calibrated_expected_daily_return"] - original_daily).abs().mean())

    selected_rows: list[pd.DataFrame] = []
    overlaps: list[float] = []
    for date, group in data.groupby("date", sort=True):
        original_selected = set(group.loc[_as_bool(group["selected"]), "ticker"].astype(str))
        selected_count = max(2, int(_as_bool(group["selected"]).sum()))
        selected_count = min(4, selected_count)
        candidates = group[
            pd.to_numeric(group["calibrated_expected_daily_return"], errors="coerce").gt(0)
            & pd.to_numeric(group["signal_strength"], errors="coerce").fillna(0.0).gt(0.15)
        ].copy()
        if candidates.empty:
            candidates = group.copy()
        picks = candidates.sort_values("calibrated_expected_daily_return", ascending=False).head(selected_count).copy()
        weights = pd.to_numeric(picks.get("weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(weights.sum()) <= 0:
            cash = (
                float(pd.to_numeric(group.get("cash_weight"), errors="coerce").dropna().iloc[0])
                if "cash_weight" in group.columns and pd.to_numeric(group.get("cash_weight"), errors="coerce").notna().any()
                else 0.5
            )
            active_weight = max(0.0, 1.0 - cash)
            weights = pd.Series(active_weight / len(picks), index=picks.index)
        else:
            original_active = min(1.0, max(0.0, float(pd.to_numeric(group["weight"], errors="coerce").clip(lower=0.0).sum())))
            weights = weights / float(weights.sum()) * original_active
        picks["weight"] = weights
        picks["selected"] = True
        picks["expected_daily_return"] = picks["calibrated_expected_daily_return"]
        selected_rows.append(picks)
        calibrated_selected = set(picks["ticker"].astype(str))
        overlaps.append(
            len(original_selected & calibrated_selected) / len(original_selected | calibrated_selected)
            if (original_selected | calibrated_selected)
            else 1.0
        )

    metadata["selection_overlap_avg"] = float(np.nanmean(overlaps)) if overlaps else np.nan
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    portfolio, trades = _portfolio_rows_from_predictions(selected, "calibrated_forecast_research")
    return portfolio, trades, metadata


def _label_metrics(labels: pd.DataFrame, mode: str) -> dict[str, float]:
    if labels.empty or "model_mode" not in labels.columns:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    subset = labels[
        labels["model_mode"].astype(str).eq(mode) & labels["horizon"].astype(str).eq("20") & _as_bool(labels["selected"])
    ].copy()
    if subset.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    tp = float((subset["first_touch_type"].astype(str) == "take_profit").mean())
    sl = float((subset["first_touch_type"].astype(str) == "stop_loss").mean())
    hit = float((pd.to_numeric(subset["realized_return_at_barrier"], errors="coerce") > 0).mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl, "hit_rate": hit}


def _metrics(portfolio: pd.DataFrame, trades: pd.DataFrame, labels: pd.DataFrame, mode: str) -> dict[str, object]:
    returns = pd.to_numeric(portfolio.get("portfolio_return", pd.Series(dtype=float)), errors="coerce").dropna()
    if returns.empty:
        base = {
            "model_mode": mode,
            "realized_return": np.nan,
            "annualized_volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
            "average_cash": np.nan,
            "average_selected_count": np.nan,
            "turnover": np.nan,
            "direction_accuracy": np.nan,
            "sample_size": 0,
        }
    else:
        total_return = float((1.0 + returns).prod() - 1.0)
        vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan
        ann_return = float((1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0)
        base = {
            "model_mode": mode,
            "realized_return": total_return,
            "annualized_volatility": vol,
            "Sharpe": ann_return / vol if np.isfinite(vol) and vol > 0 else np.nan,
            "Sortino": _sortino(returns),
            "Calmar": _calmar(returns),
            "max_drawdown": _max_drawdown(returns),
            "average_cash": float(pd.to_numeric(portfolio["cash_weight"], errors="coerce").mean())
            if "cash_weight" in portfolio
            else np.nan,
            "average_selected_count": float(pd.to_numeric(portfolio["selected_count"], errors="coerce").mean())
            if "selected_count" in portfolio
            else np.nan,
            "turnover": float(pd.to_numeric(portfolio["turnover"], errors="coerce").mean()) if "turnover" in portfolio else np.nan,
            "direction_accuracy": float((returns > 0).mean()),
            "sample_size": len(trades),
        }
    base.update(_label_metrics(labels, mode))
    if mode == "calibrated_forecast_research" and not trades.empty:
        base["hit_rate"] = float((pd.to_numeric(trades["realized_return_20d"], errors="coerce") > 0).mean())
    return base


def _governance(results: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    row = results.set_index("model_mode") if not results.empty else pd.DataFrame()
    candidate = row.loc["calibrated_forecast_research"] if "calibrated_forecast_research" in row.index else pd.Series(dtype=float)
    baseline = row.loc["baseline"] if "baseline" in row.index else pd.Series(dtype=float)
    gated = row.loc["regime_gated_full_quant"] if "regime_gated_full_quant" in row.index else pd.Series(dtype=float)
    sample = _safe_float(candidate.get("sample_size"), 0.0)
    sharpe_improves = _safe_float(candidate.get("Sharpe"), -999.0) > max(
        _safe_float(baseline.get("Sharpe"), -999.0), _safe_float(gated.get("Sharpe"), -999.0)
    )
    drawdown_ok = _safe_float(candidate.get("max_drawdown"), -1.0) >= min(
        _safe_float(baseline.get("max_drawdown"), -1.0), _safe_float(gated.get("max_drawdown"), -1.0)
    )
    overlap = _safe_float(metadata.get("selection_overlap_avg"), np.nan)
    if sample < 150:
        classification = "research only"
        reason = "sample_size_below_150"
    elif sharpe_improves and drawdown_ok and overlap >= 0.30:
        classification = "eligible for paper testing"
        reason = "sharpe_improved_with_acceptable_drawdown_and_overlap"
    elif sharpe_improves:
        classification = "research only"
        reason = "sharpe_improved_but_drawdown_or_overlap_risk"
    else:
        classification = "reject"
        reason = "does_not_improve_sharpe"
    return pd.DataFrame(
        [
            {
                "candidate": "calibrated_forecast_research",
                "classification": classification,
                "reason": reason,
                "sample_size": sample,
                "selection_overlap_avg": overlap,
                "fallback_count": int(metadata.get("fallback_count", 0)),
                "average_calibration_adjustment": _safe_float(metadata.get("average_calibration_adjustment"), np.nan),
                "warning": metadata.get("warning", "none"),
            }
        ]
    )


def run_calibrated_forecast_research_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshots = _read_csv(SNAPSHOTS_FILE)
    realized = _read_csv(REALIZED_FILE)
    calibrated = _read_csv(CALIBRATED_FILE)
    labels = _read_csv(TRIPLE_BARRIER_FILE)
    if snapshots.empty or calibrated.empty:
        raise ValueError("Missing historical snapshots or walk-forward calibrated forecasts.")

    snapshots = _merge_realized_returns(snapshots, realized)
    calibrated = _merge_realized_returns(calibrated, realized)

    baseline_portfolio, baseline_trades = _build_existing_mode(snapshots, "baseline")
    gated_portfolio, gated_trades = _build_existing_mode(snapshots, "regime_gated_full_quant")
    calibrated_portfolio, calibrated_trades, calibrated_metadata = _build_calibrated_mode(calibrated)

    daily_returns = pd.concat([baseline_portfolio, gated_portfolio, calibrated_portfolio], ignore_index=True)
    trades = pd.concat([baseline_trades, gated_trades, calibrated_trades], ignore_index=True)
    results = pd.DataFrame(
        [
            _metrics(baseline_portfolio, baseline_trades, labels, "baseline"),
            _metrics(gated_portfolio, gated_trades, labels, "regime_gated_full_quant"),
            _metrics(calibrated_portfolio, calibrated_trades, labels, "calibrated_forecast_research"),
        ]
    )
    governance = _governance(results, calibrated_metadata)

    results.to_csv(RESULTS_FILE, index=False)
    daily_returns.to_csv(DAILY_RETURNS_FILE, index=False)
    trades.to_csv(TRADES_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== CALIBRATED FORECAST RESEARCH BACKTEST =====")
    print(f"historical snapshots rows: {len(snapshots)}")
    print(f"historical realized rows: {len(realized)}")
    print(f"calibrated rows: {int(calibrated_metadata.get('calibrated_rows', 0))}")
    print(f"fallback count: {int(calibrated_metadata.get('fallback_count', 0))}")
    print(f"average calibration adjustment: {_safe_float(calibrated_metadata.get('average_calibration_adjustment'), np.nan):.8f}")
    print(f"selection overlap avg vs regime-gated: {_safe_float(calibrated_metadata.get('selection_overlap_avg'), np.nan):.4f}")

    print("\n===== BASELINE VS REGIME GATED VS CALIBRATED =====")
    display_cols = [
        "model_mode",
        "realized_return",
        "annualized_volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_cash",
        "average_selected_count",
        "turnover",
        "TP_rate",
        "SL_rate",
        "TP_minus_SL",
        "hit_rate",
        "direction_accuracy",
        "sample_size",
    ]
    print(results[display_cols].to_string(index=False))

    print("\n===== CALIBRATED FORECAST GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_RETURNS_FILE).resolve()}")
    print(f"Saved: {Path(TRADES_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, daily_returns, trades, governance


if __name__ == "__main__":
    run_calibrated_forecast_research_backtest()
