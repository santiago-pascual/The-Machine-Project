from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = "2020-01-01"
RAW_DAILY_FILE = "expected_return_ablation_daily_returns.csv"
RAW_TRADES_FILE = "expected_return_ablation_trades.csv"
PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"

RESULTS_FILE = "raw_target_2020_results.csv"
DAILY_FILE = "raw_target_2020_daily_returns.csv"
YEAR_FILE = "raw_target_2020_year_analysis.csv"
STRESS_FILE = "raw_target_2020_stress_test.csv"
BENCHMARK_FILE = "raw_target_2020_vs_benchmark.csv"
GOVERNANCE_FILE = "raw_target_2020_governance.csv"
TRADING_DAYS = 252


STRESS_PERIODS = {
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "covid_recovery": ("2020-03-24", "2020-12-31"),
    "2021_bull_market": ("2021-01-01", "2021-12-31"),
    "2022_bear_market": ("2022-01-01", "2022-12-31"),
    "2023_recovery": ("2023-01-01", "2023-12-31"),
    "2024_ai_bull_market": ("2024-01-01", "2024-12-31"),
    "2025_plus": ("2025-01-01", "2025-12-31"),
    "2026_ytd": ("2026-01-01", "2026-12-31"),
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


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def _max_drawdown(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _recovery_time(returns: pd.Series) -> int | float:
    returns = _num(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough_idx = dd.idxmin()
    prior_peak_value = peak.loc[trough_idx]
    recovered = equity.loc[trough_idx:][equity.loc[trough_idx:] >= prior_peak_value]
    if recovered.empty:
        return np.nan
    return int(recovered.index[0] - trough_idx)


def _sharpe(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if len(returns) < 2:
        return np.nan
    vol = float(returns.std(ddof=0))
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS)) if vol > 0 else np.nan


def _sortino(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    downside = returns[returns < 0]
    if returns.empty or len(downside) < 2:
        return np.nan
    down_std = float(downside.std(ddof=0))
    return float((returns.mean() * TRADING_DAYS) / (down_std * np.sqrt(TRADING_DAYS))) if down_std > 0 else np.nan


def _cagr(returns: pd.Series, dates: pd.Series) -> float:
    returns = _num(returns).dropna()
    if returns.empty or dates.empty:
        return np.nan
    years = max((pd.to_datetime(dates.max()) - pd.to_datetime(dates.min())).days / 365.25, 1 / 365.25)
    return float((1.0 + returns).prod() ** (1.0 / years) - 1.0)


def _metrics(name: str, daily: pd.DataFrame, return_col: str = "return") -> dict[str, object]:
    if daily.empty or return_col not in daily.columns:
        return {"model": name}
    returns = _num(daily[return_col]).dropna()
    dates = daily.loc[returns.index, "date"] if "date" in daily.columns else pd.Series(dtype="datetime64[ns]")
    dd = abs(_max_drawdown(returns))
    return {
        "model": name,
        "start_date_requested": START_DATE,
        "actual_start_date": str(dates.min().date()) if not dates.empty else "missing",
        "end_date": str(dates.max().date()) if not dates.empty else "missing",
        "observations": len(returns),
        "total_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
        "CAGR": _cagr(returns, dates),
        "volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan,
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": (_cagr(returns, dates) / dd) if np.isfinite(dd) and dd > 0 else np.nan,
        "max_drawdown": _max_drawdown(returns),
        "recovery_time_periods": _recovery_time(returns.reset_index(drop=True)),
    }


def _raw_daily() -> pd.DataFrame:
    daily = _prepare_dates(_read_csv(RAW_DAILY_FILE))
    if daily.empty:
        return pd.DataFrame()
    raw = daily[daily["variant"].astype(str).eq("raw_target_return_only")].copy()
    if raw.empty:
        raw = daily[daily["variant"].astype(str).eq("no_signal_strength_adjustment")].copy()
    raw = raw[raw["date"] >= pd.Timestamp(START_DATE)].copy()
    raw["return"] = _num(raw["portfolio_return"]).fillna(0.0)
    raw["cash"] = _num(raw["cash_proxy"]).fillna(0.0)
    raw["exposure"] = (1.0 - raw["cash"]).clip(0.0, 1.0)
    return raw


def _mode_daily(mode: str) -> pd.DataFrame:
    portfolio = _prepare_dates(_read_csv(PORTFOLIO_FILE))
    if portfolio.empty:
        return pd.DataFrame()
    daily = portfolio[(portfolio["model_mode"].astype(str).eq(mode)) & (portfolio["date"] >= pd.Timestamp(START_DATE))].copy()
    daily["return"] = _num(daily["realized_portfolio_return_1d"]).fillna(0.0)
    daily["cash"] = _num(daily["cash_weight"]).fillna(0.0)
    daily["exposure"] = (1.0 - daily["cash"]).clip(0.0, 1.0)
    return daily


def _benchmark_daily(ticker: str) -> pd.DataFrame:
    snapshots = _prepare_dates(_read_csv(SNAPSHOTS_FILE))
    if snapshots.empty or "current_price" not in snapshots.columns:
        return pd.DataFrame()
    data = snapshots[
        snapshots["ticker"].astype(str).eq(ticker)
        & snapshots["model_mode"].astype(str).eq("baseline")
        & (snapshots["date"] >= pd.Timestamp(START_DATE))
    ].copy()
    if data.empty:
        return pd.DataFrame()
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    data["return"] = _num(data["current_price"]).pct_change().fillna(0.0)
    data["cash"] = 0.0
    data["exposure"] = 1.0
    return data[["date", "return", "cash", "exposure"]]


def _trading_quality(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"hit_rate": np.nan, "average_winner": np.nan, "average_loser": np.nan}
    ret = _num(trades.get("realized_return_20d", pd.Series(dtype=float))).dropna()
    winners = ret[ret > 0]
    losers = ret[ret < 0]
    return {
        "hit_rate": float((ret > 0).mean()) if not ret.empty else np.nan,
        "average_winner": float(winners.mean()) if not winners.empty else np.nan,
        "average_loser": float(losers.mean()) if not losers.empty else np.nan,
    }


def _tp_sl(trades: pd.DataFrame) -> dict[str, float]:
    labels = _prepare_dates(_read_csv(LABELS_FILE))
    if labels.empty or trades.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    labels20 = labels[labels["horizon"].astype(str).eq("20")] if "horizon" in labels else labels
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.normalize()
    merged = trades[["date", "ticker"]].drop_duplicates().merge(labels20, on=["date", "ticker"], how="left")
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean()) if "first_touch_type" in merged else np.nan
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean()) if "first_touch_type" in merged else np.nan
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan}


def _raw_trades() -> pd.DataFrame:
    trades = _prepare_dates(_read_csv(RAW_TRADES_FILE))
    if trades.empty:
        return pd.DataFrame()
    raw = trades[trades["variant"].astype(str).eq("raw_target_return_only")].copy()
    if raw.empty:
        raw = trades[trades["variant"].astype(str).eq("no_signal_strength_adjustment")].copy()
    return raw[raw["date"] >= pd.Timestamp(START_DATE)].copy()


def _portfolio_stats(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    weight_col = "ablation_weight" if "ablation_weight" in trades.columns else "weight"
    weights = _num(trades.get(weight_col, pd.Series(dtype=float))).dropna()
    concentration = np.nan
    if not trades.empty and weight_col in trades.columns:
        by_date_hhi = trades.groupby("date")[weight_col].apply(lambda s: float(np.square(_num(s).fillna(0.0)).sum()))
        concentration = float(by_date_hhi.mean()) if not by_date_hhi.empty else np.nan
    return {
        "average_cash": float(_num(daily.get("cash", pd.Series(dtype=float))).mean()) if not daily.empty else np.nan,
        "average_exposure": float(_num(daily.get("exposure", pd.Series(dtype=float))).mean()) if not daily.empty else np.nan,
        "turnover": float(_num(daily.get("turnover", pd.Series(dtype=float))).mean()) if "turnover" in daily.columns else np.nan,
        "number_of_positions": float(trades.groupby("date")["ticker"].nunique().mean()) if not trades.empty else np.nan,
        "average_position_size": float(weights.mean()) if not weights.empty else np.nan,
        "max_position_size": float(weights.max()) if not weights.empty else np.nan,
        "concentration_hhi": concentration,
    }


def _annual_returns(name: str, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    data = daily.copy()
    data["year"] = data["date"].dt.year
    rows = []
    for year, group in data.groupby("year"):
        rows.append({"model": name, "year": year, "annual_return": float((1.0 + _num(group["return"]).fillna(0.0)).prod() - 1.0)})
    return pd.DataFrame(rows)


def _stress(name: str, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in STRESS_PERIODS.items():
        subset = daily[(daily["date"] >= pd.Timestamp(start)) & (daily["date"] <= pd.Timestamp(end))]
        metric = _metrics(name, subset)
        metric["stress_period"] = period
        rows.append(metric)
    return pd.DataFrame(rows)


def _robustness(raw_daily: pd.DataFrame, raw_trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = raw_daily.copy()
    data["year"] = data["date"].dt.year
    annual = data.groupby("year")["return"].apply(lambda s: float((1.0 + _num(s).fillna(0.0)).prod() - 1.0))
    if not annual.empty:
        best_year = annual.idxmax()
        no_best = data[data["year"] != best_year]
        rows.append({"check": "remove_best_year", **_metrics("raw_target_research", no_best)})
    trades = raw_trades.copy()
    trades["trade_return"] = _num(trades.get("realized_return_20d", pd.Series(dtype=float))).fillna(0.0)
    top_trades = set(trades.sort_values("trade_return", ascending=False).head(10).index)
    if top_trades:
        filtered = trades.drop(index=list(top_trades), errors="ignore")
        rows.append(
            {
                "check": "remove_best_10_trades",
                "hit_rate": float((filtered["trade_return"] > 0).mean()),
                "average_trade_return": float(filtered["trade_return"].mean()),
                "sample_size": len(filtered),
            }
        )
    if not trades.empty:
        ticker_perf = trades.groupby("ticker")["trade_return"].sum().sort_values(ascending=False)
        best_ticker = ticker_perf.index[0]
        filtered = trades[trades["ticker"] != best_ticker]
        rows.append(
            {
                "check": "remove_best_ticker",
                "removed_ticker": best_ticker,
                "average_trade_return": float(filtered["trade_return"].mean()),
                "hit_rate": float((filtered["trade_return"] > 0).mean()),
                "sample_size": len(filtered),
            }
        )
    rolling_vol = data["return"].rolling(20, min_periods=5).std()
    high_vol = data[rolling_vol >= rolling_vol.quantile(0.75)]
    rows.append({"check": "high_volatility_only", **_metrics("raw_target_research", high_vol)})
    bear = data[data["return"].rolling(20, min_periods=5).sum() < 0]
    rows.append({"check": "bear_markets_only", **_metrics("raw_target_research", bear)})
    return pd.DataFrame(rows)


def _governance(results: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    raw = results[results["model"].eq("raw_target_research")].iloc[0]
    qqq = results[results["model"].eq("QQQ_buy_hold")]
    qqq_row = qqq.iloc[0] if not qqq.empty else pd.Series(dtype=object)
    reasons = []
    classification = "candidate"
    if raw["Sharpe"] < 1:
        classification = "reject"
        reasons.append("Sharpe < 1")
    if raw["max_drawdown"] < -0.25:
        classification = "reject"
        reasons.append("DD > 25%")
    remove_best_year = robustness[robustness["check"].eq("remove_best_year")]
    if not remove_best_year.empty and float(remove_best_year.iloc[0].get("total_return", 0.0)) < 0:
        classification = "reject"
        reasons.append("performance depends on one year")
    if not qqq_row.empty and raw["Sharpe"] <= qqq_row.get("Sharpe", -np.inf) and raw["max_drawdown"] <= qqq_row.get("max_drawdown", 0):
        classification = "reject"
        reasons.append("loses to QQQ with similar/worse drawdown")
    if classification != "reject":
        if raw["Sharpe"] > 1.2 and raw["max_drawdown"] > -0.20 and (qqq_row.empty or raw["Sharpe"] > qqq_row.get("Sharpe", -np.inf)):
            classification = "candidate for aggressive growth research"
            reasons.append("passes growth thresholds")
        else:
            classification = "research only"
            reasons.append("does not pass all growth candidate thresholds")
    return pd.DataFrame(
        [{"model": "raw_target_research", "classification": classification, "reason": "; ".join(reasons), "production_change": "none"}]
    )


def run_raw_target_extended_backtest_2020() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_daily = _raw_daily()
    raw_trades = _raw_trades()
    if raw_daily.empty:
        raise ValueError("Raw target historical daily returns are required.")
    models = {
        "baseline": _mode_daily("baseline"),
        "regime_gated_full_quant": _mode_daily("regime_gated_full_quant"),
        "raw_target_research": raw_daily,
        "SPY_buy_hold": _benchmark_daily("SPY"),
        "QQQ_buy_hold": _benchmark_daily("QQQ"),
    }
    result_rows = []
    annual_frames = []
    stress_frames = []
    for name, daily in models.items():
        result = _metrics(name, daily)
        if name == "raw_target_research":
            result.update(_trading_quality(raw_trades))
            result.update(_tp_sl(raw_trades))
            result.update(_portfolio_stats(daily, raw_trades))
        result_rows.append(result)
        annual_frames.append(_annual_returns(name, daily))
        stress_frames.append(_stress(name, daily))
    results = pd.DataFrame(result_rows)
    annual = pd.concat(annual_frames, ignore_index=True, sort=False)
    stress = pd.concat(stress_frames, ignore_index=True, sort=False)
    robustness = _robustness(raw_daily, raw_trades)
    governance = _governance(results, robustness)

    raw_daily.to_csv(DAILY_FILE, index=False)
    results.to_csv(RESULTS_FILE, index=False)
    annual.to_csv(YEAR_FILE, index=False)
    stress.to_csv(STRESS_FILE, index=False)
    results.to_csv(BENCHMARK_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== RAW TARGET EXTENDED BACKTEST 2020-PRESENT =====")
    print(results.to_string(index=False))
    actual_start = results.loc[results["model"].eq("raw_target_research"), "actual_start_date"].iloc[0]
    if str(actual_start) > START_DATE:
        print(f"\n[WARNING] requested start {START_DATE}, but available raw target data starts {actual_start}.")
    print("\n===== RAW TARGET 2020 STRESS TEST =====")
    print(
        stress[stress["model"].eq("raw_target_research")][
            ["stress_period", "actual_start_date", "end_date", "observations", "total_return", "Sharpe", "max_drawdown"]
        ].to_string(index=False)
    )
    print("\n===== RAW TARGET 2020 GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_FILE).resolve()}")
    print(f"Saved: {Path(YEAR_FILE).resolve()}")
    print(f"Saved: {Path(STRESS_FILE).resolve()}")
    print(f"Saved: {Path(BENCHMARK_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, raw_daily, stress, governance


if __name__ == "__main__":
    run_raw_target_extended_backtest_2020()
