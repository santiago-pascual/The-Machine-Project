from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ALPHA_REPORT_FILE = "alpha_attribution_report.csv"
IC_RANKING_FILE = "factor_ic_ranking.csv"
INCREMENTAL_FILE = "factor_incremental_alpha.csv"
CORRELATION_FILE = "factor_correlation_matrix.csv"
FEATURE_STORE_FILE = "historical_feature_store.csv"
REALIZED_FILE = "historical_realized_returns.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
TRIPLE_BARRIER_FILE = "historical_triple_barrier_labels.csv"

RESULTS_FILE = "factor_alpha_model_results.csv"
SCORES_FILE = "factor_alpha_candidate_scores.csv"
BACKTEST_FILE = "factor_alpha_backtest.csv"
GOVERNANCE_FILE = "factor_alpha_governance.csv"
TRADING_DAYS = 252


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


def _rank_pct(series: pd.Series) -> pd.Series:
    values = _num(series)
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return values.rank(pct=True).fillna(0.5)


def _zscore(series: pd.Series) -> pd.Series:
    values = _num(series)
    std = float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0
    if std <= 0 or not np.isfinite(std):
        return pd.Series(0.0, index=series.index)
    return ((values - float(values.mean())) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _pearson(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame["x"].corr(frame["y"]))


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
    down_std = float(downside.std(ddof=0))
    if down_std <= 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (down_std * np.sqrt(TRADING_DAYS)))


def _calmar(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    dd = abs(_max_drawdown(returns))
    if returns.empty or not np.isfinite(dd) or dd <= 0:
        return np.nan
    annualized = (1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0
    return float(annualized / dd)


def _prepare_dataset() -> pd.DataFrame:
    snapshots = _read_csv(SNAPSHOTS_FILE)
    realized = _read_csv(REALIZED_FILE)
    if snapshots.empty:
        feature_store = _read_csv(FEATURE_STORE_FILE)
        snapshots = feature_store
    if snapshots.empty:
        return pd.DataFrame()
    data = snapshots.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date", "ticker"])
    if not realized.empty and {"date", "ticker", "model_mode"}.issubset(realized.columns):
        realized = realized.copy()
        realized["date"] = pd.to_datetime(realized["date"], errors="coerce").dt.normalize()
        cols = ["realized_return_1d", "realized_return_5d", "realized_return_10d", "realized_return_20d", "realized_return_30d"]
        available = [col for col in cols if col in realized.columns]
        data = data.drop(columns=[col for col in available if col in data.columns], errors="ignore")
        data = data.merge(realized[["date", "ticker", "model_mode"] + available], on=["date", "ticker", "model_mode"], how="left")
    for col in data.columns:
        if col not in {"date", "ticker", "model_mode", "selected", "regime", "timing_model", "target_model", "covariance_method", "gate_decision", "gate_reason"}:
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0:
                data[col] = converted
    if "daily_volatility" not in data.columns:
        if "realized_return_1d" in data.columns:
            data["daily_volatility"] = data.groupby("ticker")["realized_return_1d"].transform(lambda s: _num(s).rolling(20, min_periods=5).std())
        elif "realized_return_5d" in data.columns:
            data["daily_volatility"] = _num(data["realized_return_5d"]).abs() / np.sqrt(5)
        else:
            data["daily_volatility"] = np.nan
    return data


def _orthogonalize_signal(data: pd.DataFrame) -> pd.Series:
    x = _num(data["expected_daily_return"])
    y = _num(data["signal_strength"])
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    residual = pd.Series(0.0, index=data.index)
    if len(frame) < 30 or frame["x"].var(ddof=0) <= 0:
        return residual
    beta = float(frame["x"].cov(frame["y"]) / frame["x"].var(ddof=0))
    alpha = float(frame["y"].mean() - beta * frame["x"].mean())
    residual.loc[frame.index] = frame["y"] - (alpha + beta * frame["x"])
    return residual.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _candidate_scores(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    expected = _num(out["expected_daily_return"]).fillna(0.0)
    signal = _num(out["signal_strength"]).fillna(0.0)
    volatility = _num(out["daily_volatility"]).fillna(_num(out["daily_volatility"]).median()).fillna(0.0)
    residual_signal = _orthogonalize_signal(out)
    out["residual_signal_strength"] = residual_signal
    out["candidate_A_expected_only"] = expected
    out["candidate_B_expected_plus_orthogonal_signal"] = _zscore(expected) + 0.25 * _zscore(residual_signal)
    out["candidate_C_expected_signal_volatility"] = _zscore(expected) + 0.20 * _zscore(signal) - 0.10 * _zscore(volatility)
    out["candidate_D_expected_volatility_penalty"] = expected / (1.0 + volatility.rank(pct=True).fillna(0.5))
    return out


def _candidate_columns() -> list[str]:
    return [
        "candidate_A_expected_only",
        "candidate_B_expected_plus_orthogonal_signal",
        "candidate_C_expected_signal_volatility",
        "candidate_D_expected_volatility_penalty",
    ]


def _monotonicity(data: pd.DataFrame, score_col: str) -> dict[str, float]:
    target = "realized_return_20d"
    frame = data[[score_col, target]].dropna().copy()
    if len(frame) < 50 or frame[score_col].nunique() < 10:
        return {"decile_corr": np.nan, "top_bottom_spread": np.nan, "monotonicity_score": np.nan}
    try:
        frame["decile"] = pd.qcut(frame[score_col].rank(method="first"), 10, labels=False) + 1
    except ValueError:
        return {"decile_corr": np.nan, "top_bottom_spread": np.nan, "monotonicity_score": np.nan}
    grouped = frame.groupby("decile")[target].mean().reset_index()
    corr = _spearman(grouped["decile"], grouped[target])
    spread = float(grouped.loc[grouped["decile"].eq(10), target].mean() - grouped.loc[grouped["decile"].eq(1), target].mean())
    steps = int((grouped[target].diff().dropna() > 0).sum())
    return {
        "decile_corr": corr,
        "top_bottom_spread": spread,
        "monotonicity_score": float((abs(corr) if np.isfinite(corr) else 0.0) + min(1.0, steps / 9.0)) / 2.0,
    }


def _score_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in _candidate_columns():
        mono = _monotonicity(data, col)
        rows.append(
            {
                "candidate": col,
                "formula": _formula_text(col),
                "pearson_ic_5d": _pearson(data[col], data["realized_return_5d"]),
                "pearson_ic_10d": _pearson(data[col], data["realized_return_10d"]),
                "pearson_ic_20d": _pearson(data[col], data["realized_return_20d"]),
                "spearman_ic_5d": _spearman(data[col], data["realized_return_5d"]),
                "spearman_ic_10d": _spearman(data[col], data["realized_return_10d"]),
                "spearman_ic_20d": _spearman(data[col], data["realized_return_20d"]),
                **mono,
            }
        )
    scores = pd.DataFrame(rows)
    scores["average_abs_rank_ic"] = scores[["spearman_ic_5d", "spearman_ic_10d", "spearman_ic_20d"]].abs().mean(axis=1)
    return scores.sort_values("average_abs_rank_ic", ascending=False)


def _formula_text(candidate: str) -> str:
    return {
        "candidate_A_expected_only": "expected_daily_return",
        "candidate_B_expected_plus_orthogonal_signal": "z(expected_daily_return) + 0.25*z(residual_signal_strength)",
        "candidate_C_expected_signal_volatility": "z(expected_daily_return) + 0.20*z(signal_strength) - 0.10*z(daily_volatility)",
        "candidate_D_expected_volatility_penalty": "expected_daily_return / (1 + volatility_percentile)",
    }.get(candidate, candidate)


def _select_by_candidate(data: pd.DataFrame, score_col: str) -> pd.DataFrame:
    selected_rows: list[pd.DataFrame] = []
    for date, group in data.groupby("date", sort=True):
        current_selected = group[_bool(group.get("selected", pd.Series(False, index=group.index)))]
        selected_count = int(len(current_selected)) if len(current_selected) else 4
        selected_count = min(4, max(2, selected_count))
        candidates = group[_num(group["expected_daily_return"]).gt(0)].copy()
        if candidates.empty:
            candidates = group.copy()
        picks = candidates.sort_values(score_col, ascending=False).head(selected_count).copy()
        active_weight = float(_num(group.get("weight", pd.Series(dtype=float))).clip(lower=0.0).sum())
        if not np.isfinite(active_weight) or active_weight <= 0:
            active_weight = max(0.0, 1.0 - float(_num(group.get("cash_weight", pd.Series([0.5]))).dropna().iloc[0] if "cash_weight" in group else 0.5))
        picks["factor_alpha_weight"] = active_weight / max(1, len(picks))
        selected_rows.append(picks)
    return pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()


def _portfolio_metrics(selected: pd.DataFrame, candidate: str) -> dict[str, object]:
    daily_returns: list[dict[str, object]] = []
    prev_assets: set[str] = set()
    for date, group in selected.groupby("date", sort=True):
        weights = _num(group["factor_alpha_weight"]).fillna(0.0)
        returns = _num(group["realized_return_1d"])
        period_return = float((weights * returns).sum()) if returns.notna().any() else np.nan
        assets = set(group["ticker"].astype(str))
        daily_returns.append(
            {
                "date": date,
                "candidate": candidate,
                "portfolio_return": period_return,
                "selected_count": len(assets),
                "cash_proxy": max(0.0, 1.0 - float(weights.sum())),
                "turnover": len(assets.symmetric_difference(prev_assets)) / max(1, len(assets | prev_assets)),
            }
        )
        prev_assets = assets
    daily = pd.DataFrame(daily_returns)
    returns = _num(daily["portfolio_return"]).dropna() if not daily.empty else pd.Series(dtype=float)
    total_return = float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan
    return {
        "candidate": candidate,
        "realized_return": total_return,
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "hit_rate": float((_num(selected["realized_return_20d"]) > 0).mean()) if not selected.empty else np.nan,
        "average_selected_count": float(daily["selected_count"].mean()) if not daily.empty else np.nan,
        "average_cash_proxy": float(daily["cash_proxy"].mean()) if not daily.empty else np.nan,
        "turnover": float(daily["turnover"].mean()) if not daily.empty else np.nan,
        "sample_size": int(len(selected)),
    }, daily


def _tp_sl_metrics(selected: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    if selected.empty or labels.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels20 = labels[labels["horizon"].astype(str).eq("20")] if "horizon" in labels else labels
    merged = selected[["date", "ticker"]].drop_duplicates().merge(labels20, on=["date", "ticker"], how="left")
    if merged.empty or "first_touch_type" not in merged:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean())
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _backtest_candidates(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = _read_csv(TRIPLE_BARRIER_FILE)
    rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    for candidate in _candidate_columns():
        selected = _select_by_candidate(data, candidate)
        metrics, daily = _portfolio_metrics(selected, candidate)
        metrics.update(_tp_sl_metrics(selected, labels))
        rows.append(metrics)
        daily_frames.append(daily)
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False), pd.concat(daily_frames, ignore_index=True)


def _governance(scores: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_sharpe = float(backtest.loc[backtest["candidate"].eq("candidate_A_expected_only"), "Sharpe"].iloc[0]) if "candidate_A_expected_only" in set(backtest["candidate"]) else np.nan
    for _, row in backtest.iterrows():
        candidate = str(row["candidate"])
        score_row = scores[scores["candidate"].eq(candidate)].iloc[0] if candidate in set(scores["candidate"]) else pd.Series(dtype=float)
        sharpe = float(row.get("Sharpe", np.nan))
        tp_sl = float(row.get("TP_minus_SL", np.nan))
        avg_ic = float(score_row.get("average_abs_rank_ic", np.nan))
        if candidate == "candidate_A_expected_only":
            classification = "useful for research"
            reason = "baseline_alpha_factor"
        elif np.isfinite(sharpe) and np.isfinite(baseline_sharpe) and sharpe > baseline_sharpe * 1.05 and tp_sl > 0 and avg_ic >= 0.03:
            classification = "candidate for shadow mode"
            reason = "beats_expected_return_baseline_with_positive_tp_sl"
        elif np.isfinite(sharpe) and np.isfinite(baseline_sharpe) and sharpe > baseline_sharpe:
            classification = "useful for research"
            reason = "marginal_improvement"
        elif avg_ic >= 0.03:
            classification = "diagnostic only"
            reason = "signal_quality_without_portfolio_improvement"
        else:
            classification = "reject"
            reason = "weak_ic_or_backtest"
        rows.append(
            {
                "candidate": candidate,
                "formula": _formula_text(candidate),
                "classification": classification,
                "reason": reason,
                "Sharpe": sharpe,
                "baseline_sharpe": baseline_sharpe,
                "average_abs_rank_ic": avg_ic,
                "TP_minus_SL": tp_sl,
                "production_change": "none",
            }
        )
    return pd.DataFrame(rows)


def run_factor_alpha_model() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _prepare_dataset()
    if data.empty:
        raise ValueError("No historical data available for factor alpha model.")
    required = {"expected_daily_return", "signal_strength", "realized_return_1d", "realized_return_5d", "realized_return_10d", "realized_return_20d"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = _candidate_scores(data)
    scores = _score_metrics(data)
    backtest, daily = _backtest_candidates(data)
    governance = _governance(scores, backtest)
    results = backtest.merge(scores, on="candidate", how="left", suffixes=("", "_score")).merge(governance[["candidate", "classification", "reason"]], on="candidate", how="left")

    results.to_csv(RESULTS_FILE, index=False)
    scores.to_csv(SCORES_FILE, index=False)
    daily.to_csv(BACKTEST_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== FACTOR ALPHA MODEL =====")
    print(f"observations: {len(data)}")
    print("production change: none")

    print("\n===== FACTOR ALPHA CANDIDATES =====")
    print(scores[["candidate", "formula", "spearman_ic_5d", "spearman_ic_10d", "spearman_ic_20d", "average_abs_rank_ic", "monotonicity_score"]].to_string(index=False))

    print("\n===== FACTOR ALPHA BACKTEST =====")
    print(backtest[["candidate", "realized_return", "Sharpe", "Sortino", "Calmar", "max_drawdown", "TP_rate", "SL_rate", "TP_minus_SL", "hit_rate", "average_selected_count", "average_cash_proxy", "sample_size"]].to_string(index=False))

    print("\n===== FACTOR ALPHA GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(SCORES_FILE).resolve()}")
    print(f"Saved: {Path(BACKTEST_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, scores, backtest, governance


if __name__ == "__main__":
    run_factor_alpha_model()
