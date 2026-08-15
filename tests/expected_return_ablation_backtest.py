from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HARMFUL_FILE = "expected_return_harmful_transformations.csv"
STAGE_ATTRIBUTION_FILE = "expected_return_stage_attribution.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
TRIPLE_BARRIER_FILE = "historical_triple_barrier_labels.csv"
FEATURE_STORE_FILE = "historical_feature_store.csv"
DECOMPOSITION_FILE = "expected_return_decomposition.csv"

RESULTS_FILE = "expected_return_ablation_results.csv"
DAILY_RETURNS_FILE = "expected_return_ablation_daily_returns.csv"
TRADES_FILE = "expected_return_ablation_trades.csv"
GOVERNANCE_FILE = "expected_return_ablation_governance.csv"
TRADING_DAYS = 252


VARIANTS = {
    "baseline_current": "final_expected_daily_return",
    "no_signal_strength_adjustment": "constant_penalty_proxy",
    "no_regime_adjustment": "signal_strength_adjusted_proxy",
    "no_signal_strength_and_no_regime_adjustment": "constant_penalty_proxy",
    "raw_target_return_only": "raw_daily_return_proxy",
    "raw_target_return_with_final_scaling": "scaling_final_proxy",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _max_drawdown(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if len(returns) < 2:
        return np.nan
    vol = float(returns.std(ddof=0))
    if vol <= 0:
        return np.nan
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    downside = returns[returns < 0]
    if returns.empty or len(downside) < 2:
        return np.nan
    downside_std = float(downside.std(ddof=0))
    if downside_std <= 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (downside_std * np.sqrt(TRADING_DAYS)))


def _calmar(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    dd = abs(_max_drawdown(returns))
    if returns.empty or not np.isfinite(dd) or dd <= 0:
        return np.nan
    annualized = (1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0
    return float(annualized / dd)


def _monotonicity(signal: pd.Series, realized: pd.Series) -> float:
    frame = pd.DataFrame({"s": _num(signal), "r": _num(realized)}).dropna()
    if len(frame) < 50 or frame["s"].nunique() < 10:
        return np.nan
    try:
        frame["decile"] = pd.qcut(frame["s"].rank(method="first"), 10, labels=False) + 1
    except ValueError:
        return np.nan
    grouped = frame.groupby("decile")["r"].mean().reset_index()
    return _spearman(grouped["decile"], grouped["r"])


def _load_or_build_decomposition() -> pd.DataFrame:
    decomp = _read_csv(DECOMPOSITION_FILE)
    if not decomp.empty:
        decomp["date"] = pd.to_datetime(decomp["date"], errors="coerce").dt.normalize()
        return decomp
    from expected_return_decomposition import run_expected_return_decomposition

    decomp, _, _, _ = run_expected_return_decomposition()
    decomp["date"] = pd.to_datetime(decomp["date"], errors="coerce").dt.normalize()
    return decomp


def _load_snapshots() -> pd.DataFrame:
    snapshots = _read_csv(SNAPSHOTS_FILE)
    realized = _read_csv(REALIZED_FILE)
    if snapshots.empty:
        snapshots = _read_csv(FEATURE_STORE_FILE)
    if snapshots.empty:
        return pd.DataFrame()
    snapshots = snapshots.copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce").dt.normalize()
    snapshots = snapshots.dropna(subset=["date", "ticker"])
    if not realized.empty and {"date", "ticker", "model_mode"}.issubset(realized.columns):
        realized = realized.copy()
        realized["date"] = pd.to_datetime(realized["date"], errors="coerce").dt.normalize()
        cols = [f"realized_return_{h}d" for h in [1, 5, 10, 20, 30] if f"realized_return_{h}d" in realized.columns]
        snapshots = snapshots.drop(columns=[col for col in cols if col in snapshots.columns], errors="ignore")
        snapshots = snapshots.merge(realized[["date", "ticker", "model_mode"] + cols], on=["date", "ticker", "model_mode"], how="left")
    return snapshots


def _prepare_dataset() -> pd.DataFrame:
    snapshots = _load_snapshots()
    decomp = _load_or_build_decomposition()
    if snapshots.empty or decomp.empty:
        return pd.DataFrame()
    keys = ["date", "ticker", "model_mode"]
    stage_cols = [col for col in VARIANTS.values() if col in decomp.columns]
    data = snapshots.merge(decomp[keys + stage_cols].drop_duplicates(keys, keep="last"), on=keys, how="left")
    data = data.loc[:, ~data.columns.duplicated()]
    for col in data.columns:
        if col not in {"date", "ticker", "model_mode", "selected", "regime", "timing_model", "target_model", "covariance_method", "gate_decision", "gate_reason"}:
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0:
                data[col] = converted
    return data


def _select_variant(data: pd.DataFrame, variant: str, score_col: str) -> pd.DataFrame:
    selected_rows: list[pd.DataFrame] = []
    for date, group in data.groupby("date", sort=True):
        current_selected = group[_bool(group.get("selected", pd.Series(False, index=group.index)))]
        selected_count = int(len(current_selected)) if len(current_selected) else 4
        selected_count = min(4, max(2, selected_count))
        candidates = group[_num(group[score_col]).gt(0)].copy()
        if candidates.empty:
            candidates = group.copy()
        picks = candidates.sort_values(score_col, ascending=False).head(selected_count).copy()
        active_weight = float(_num(group.get("weight", pd.Series(dtype=float))).clip(lower=0.0).sum())
        if not np.isfinite(active_weight) or active_weight <= 0:
            active_weight = max(0.0, 1.0 - float(_num(group.get("cash_weight", pd.Series([0.5]))).dropna().iloc[0] if "cash_weight" in group else 0.5))
        picks["ablation_variant"] = variant
        picks["ablation_score"] = picks[score_col]
        picks["ablation_weight"] = active_weight / max(1, len(picks))
        selected_rows.append(picks)
    return pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()


def _portfolio_daily(selected: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prev_assets: set[str] = set()
    for date, group in selected.groupby("date", sort=True):
        weights = _num(group["ablation_weight"]).fillna(0.0)
        returns = _num(group["realized_return_1d"])
        period_return = float((weights * returns).sum()) if returns.notna().any() else np.nan
        assets = set(group["ticker"].astype(str))
        rows.append(
            {
                "date": date,
                "variant": variant,
                "portfolio_return": period_return,
                "selected_count": len(assets),
                "cash_proxy": max(0.0, 1.0 - float(weights.sum())),
                "turnover": len(assets.symmetric_difference(prev_assets)) / max(1, len(assets | prev_assets)),
                "selected_tickers": ",".join(sorted(assets)),
            }
        )
        prev_assets = assets
    return pd.DataFrame(rows)


def _tp_sl(selected: pd.DataFrame) -> dict[str, float]:
    labels = _read_csv(TRIPLE_BARRIER_FILE)
    if labels.empty or selected.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels = labels[labels.get("horizon", pd.Series(index=labels.index)).astype(str).eq("20")] if "horizon" in labels else labels
    merged = selected[["date", "ticker"]].drop_duplicates().merge(labels, on=["date", "ticker"], how="left")
    if merged.empty or "first_touch_type" not in merged:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean())
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _metrics(data: pd.DataFrame, selected: pd.DataFrame, daily: pd.DataFrame, variant: str, score_col: str) -> dict[str, object]:
    returns = _num(daily.get("portfolio_return", pd.Series(dtype=float))).dropna()
    score = _num(data[score_col])
    result = {
        "variant": variant,
        "score_source": score_col,
        "capture_type": "exact" if score_col == "final_expected_daily_return" else "proxy",
        "realized_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
        "volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan,
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "hit_rate": float((_num(selected.get("realized_return_20d", pd.Series(dtype=float))) > 0).mean()) if not selected.empty else np.nan,
        "direction_accuracy": float((returns > 0).mean()) if not returns.empty else np.nan,
        "IC_5D": _spearman(score, data.get("realized_return_5d", pd.Series(index=data.index))),
        "IC_10D": _spearman(score, data.get("realized_return_10d", pd.Series(index=data.index))),
        "IC_20D": _spearman(score, data.get("realized_return_20d", pd.Series(index=data.index))),
        "monotonicity_20d": _monotonicity(score, data.get("realized_return_20d", pd.Series(index=data.index))),
        "average_cash": float(_num(daily.get("cash_proxy", pd.Series(dtype=float))).mean()) if not daily.empty else np.nan,
        "selected_count": float(_num(daily.get("selected_count", pd.Series(dtype=float))).mean()) if not daily.empty else np.nan,
        "turnover": float(_num(daily.get("turnover", pd.Series(dtype=float))).mean()) if not daily.empty else np.nan,
        "sample_size": int(len(selected)),
    }
    result.update(_tp_sl(selected))
    return result


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["variant"].eq("baseline_current")]
    baseline_sharpe = float(baseline["Sharpe"].iloc[0]) if not baseline.empty else np.nan
    baseline_ic = float(baseline["IC_20D"].iloc[0]) if not baseline.empty else np.nan
    rows: list[dict[str, object]] = []
    for _, row in results.iterrows():
        sharpe = float(row.get("Sharpe", np.nan))
        ic = float(row.get("IC_20D", np.nan))
        dd = float(row.get("max_drawdown", np.nan))
        variant = str(row["variant"])
        if variant == "baseline_current":
            classification = "research only"
            reason = "current_reference"
        elif np.isfinite(sharpe) and np.isfinite(baseline_sharpe) and sharpe > baseline_sharpe * 1.05 and ic >= baseline_ic:
            classification = "candidate for shadow mode"
            reason = "improves_sharpe_and_preserves_ic"
        elif np.isfinite(ic) and np.isfinite(baseline_ic) and ic > baseline_ic and (not np.isfinite(sharpe) or sharpe <= baseline_sharpe):
            classification = "research only"
            reason = "improves_ic_but_not_portfolio"
        elif np.isfinite(sharpe) and np.isfinite(baseline_sharpe) and sharpe > baseline_sharpe:
            classification = "research only"
            reason = "marginal_portfolio_improvement"
        else:
            classification = "reject"
            reason = "does_not_improve_current_pipeline"
        rows.append(
            {
                "variant": variant,
                "classification": classification,
                "reason": reason,
                "Sharpe": sharpe,
                "baseline_Sharpe": baseline_sharpe,
                "IC_20D": ic,
                "baseline_IC_20D": baseline_ic,
                "max_drawdown": dd,
                "production_change": "none",
            }
        )
    return pd.DataFrame(rows)


def run_expected_return_ablation_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _prepare_dataset()
    if data.empty:
        raise ValueError("No historical data available for expected return ablation.")
    result_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for variant, score_col in VARIANTS.items():
        if score_col not in data.columns:
            continue
        selected = _select_variant(data, variant, score_col)
        daily = _portfolio_daily(selected, variant)
        result_rows.append(_metrics(data, selected, daily, variant, score_col))
        daily_frames.append(daily)
        trade_frames.append(selected.assign(variant=variant))
    results = pd.DataFrame(result_rows).sort_values("Sharpe", ascending=False)
    daily_returns = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    governance = _governance(results)

    results.to_csv(RESULTS_FILE, index=False)
    daily_returns.to_csv(DAILY_RETURNS_FILE, index=False)
    trades.to_csv(TRADES_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== EXPECTED RETURN ABLATION BACKTEST =====")
    print(f"observations: {len(data)}")
    print("capture note: non-baseline variants use diagnostic proxies from expected_return_decomposition.csv")

    print("\n===== ABLATION PERFORMANCE COMPARISON =====")
    cols = ["variant", "capture_type", "realized_return", "volatility", "Sharpe", "Sortino", "Calmar", "max_drawdown", "TP_rate", "SL_rate", "TP_minus_SL", "hit_rate", "IC_5D", "IC_10D", "IC_20D", "monotonicity_20d", "average_cash", "selected_count", "turnover", "sample_size"]
    print(results[cols].to_string(index=False))

    print("\n===== ABLATION GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_RETURNS_FILE).resolve()}")
    print(f"Saved: {Path(TRADES_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, daily_returns, trades, governance


if __name__ == "__main__":
    run_expected_return_ablation_backtest()
