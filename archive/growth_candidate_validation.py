from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE = "soft_exit_rule_vol_target_22pct"
BASE_VARIANT = "soft_exit_rule"

GROWTH_RESULTS_FILE = "growth_volatility_targeting_results.csv"
GROWTH_DAILY_FILE = "growth_volatility_targeting_daily_returns.csv"
GROWTH_EXPOSURE_FILE = "growth_volatility_targeting_exposure.csv"
EXIT_TRADES_FILE = "exit_rule_walk_forward_trades.csv"
RAW_DAILY_FILE = "raw_target_2020_daily_returns.csv"
BENCHMARK_RESULTS_FILE = "raw_target_2020_vs_benchmark.csv"
REALIZED_FILE = "historical_realized_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"

OUT_RESULTS = "growth_candidate_validation_results.csv"
OUT_BENCHMARKS = "growth_candidate_vs_benchmarks.csv"
OUT_ROBUSTNESS = "growth_candidate_robustness.csv"
OUT_CONCENTRATION = "growth_candidate_concentration.csv"
OUT_GOVERNANCE = "growth_candidate_governance.csv"

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


def _dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"])


def _periods_per_year(dates: pd.Series) -> float:
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return 52.0
    step = np.median(dates.diff().dt.days.dropna())
    return float(365.25 / step) if np.isfinite(step) and step > 0 else 52.0


def _max_drawdown(returns: pd.Series) -> float:
    r = _num(returns).dropna()
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _recovery_time(returns: pd.Series) -> float:
    r = _num(returns).dropna().reset_index(drop=True)
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    trough = int(dd.idxmin())
    prior_peak = equity.cummax().iloc[trough]
    recovered = equity.iloc[trough:][equity.iloc[trough:] >= prior_peak]
    if recovered.empty:
        return np.nan
    return float(int(recovered.index[0]) - trough)


def _sortino(returns: pd.Series, ppy: float) -> float:
    r = _num(returns).dropna()
    downside = r[r < 0].std(ddof=0)
    if r.empty or not np.isfinite(downside) or downside <= 0:
        return np.nan
    return float((r.mean() * ppy) / (downside * np.sqrt(ppy)))


def _metrics(name: str, daily: pd.DataFrame, return_col: str = "return") -> dict:
    if daily.empty or return_col not in daily.columns:
        return {"model": name}
    data = _dates(daily).sort_values("date")
    r = _num(data[return_col]).dropna()
    if r.empty:
        return {"model": name}
    ppy = _periods_per_year(data["date"])
    total = float((1.0 + r).prod() - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(r) / ppy, 1e-9)
    vol = float(r.std(ddof=0) * np.sqrt(ppy))
    sharpe = np.nan if vol <= 0 else float((r.mean() * ppy) / vol)
    mdd = _max_drawdown(r)
    return {
        "model": name,
        "start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "observations": len(r),
        "total_return": total,
        "CAGR": float((1.0 + total) ** (1.0 / years) - 1.0),
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": _sortino(r, ppy),
        "Calmar": np.nan if not np.isfinite(mdd) or mdd >= 0 else float(((1.0 + total) ** (1.0 / years) - 1.0) / abs(mdd)),
        "max_drawdown": mdd,
        "recovery_time": _recovery_time(r),
        "hit_rate": float((r > 0).mean()),
    }


def _candidate_daily() -> pd.DataFrame:
    daily = _dates(_read_csv(GROWTH_DAILY_FILE))
    if daily.empty:
        return pd.DataFrame()
    daily = daily[daily["vol_target_variant"].eq(CANDIDATE) | daily["variant"].eq(CANDIDATE)].copy()
    daily["return"] = _num(daily.get("vol_target_return", daily.get("return", pd.Series(index=daily.index, dtype=float))))
    daily["exposure"] = _num(daily.get("target_exposure", pd.Series(index=daily.index, dtype=float)))
    daily["cash"] = _num(daily.get("cash_weight", 1.0 - daily["exposure"]))
    return daily.sort_values("date")


def _raw_daily() -> pd.DataFrame:
    raw = _dates(_read_csv(RAW_DAILY_FILE))
    if raw.empty:
        return raw
    raw["return"] = _num(raw.get("return", raw.get("portfolio_return", pd.Series(index=raw.index, dtype=float))))
    return raw.sort_values("date")


def _mode_daily(mode: str) -> pd.DataFrame:
    pf = _dates(_read_csv(PORTFOLIO_FILE))
    if pf.empty or "model_mode" not in pf.columns:
        return pd.DataFrame()
    data = pf[pf["model_mode"].eq(mode)].copy()
    for col in ["realized_portfolio_return_5d", "return", "portfolio_return", "realized_portfolio_return_1d"]:
        if col in data.columns:
            data["return"] = _num(data[col])
            break
    return data.sort_values("date")


def _benchmark_daily(ticker: str) -> pd.DataFrame:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty:
        return pd.DataFrame()
    if "model_mode" in snaps.columns:
        baseline = snaps[snaps["model_mode"].eq("baseline")]
        if not baseline.empty:
            snaps = baseline.copy()
    data = snaps[snaps["ticker"].eq(ticker)].drop_duplicates("date").sort_values("date").copy()
    if data.empty:
        return data
    data["return"] = _num(data["current_price"]).pct_change()
    return data.dropna(subset=["return"])


def _candidate_trades(candidate_daily: pd.DataFrame) -> pd.DataFrame:
    trades = _dates(_read_csv(EXIT_TRADES_FILE))
    if trades.empty:
        return trades
    trades = trades[trades["variant"].eq(BASE_VARIANT)].copy()
    exposure = candidate_daily[["date", "exposure", "cash", "return"]].rename(columns={"return": "candidate_period_return"})
    trades = trades.merge(exposure, on="date", how="inner")
    trades["trade_contribution"] = _num(trades.get("weight_proxy", pd.Series(index=trades.index, dtype=float))).fillna(0.0) * _num(trades.get("realized_return_5d", pd.Series(index=trades.index, dtype=float))).fillna(0.0) * _num(trades["exposure"]).fillna(0.0)
    trades["variant"] = CANDIDATE
    return trades


def _labels() -> pd.DataFrame:
    labels = _dates(_read_csv(LABELS_FILE))
    if labels.empty:
        return labels
    if "model_mode" in labels.columns:
        baseline = labels[labels["model_mode"].eq("baseline")]
        if not baseline.empty:
            labels = baseline.copy()
    if "horizon" in labels.columns:
        labels = labels[labels["horizon"].eq(20)].copy()
    return labels.drop_duplicates(["date", "ticker"])


def _tp_sl(trades: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if trades.empty or labels.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    data = trades[["date", "ticker"]].merge(labels[["date", "ticker", "label"]], on=["date", "ticker"], how="left")
    data = data.dropna(subset=["label"])
    if data.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((data["label"] == 1).mean())
    sl = float((data["label"] == -1).mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _add_candidate_exposure(metrics: dict, daily: pd.DataFrame) -> dict:
    metrics = dict(metrics)
    metrics["average_exposure"] = float(_num(daily.get("exposure", pd.Series(index=daily.index, dtype=float))).mean())
    metrics["min_exposure"] = float(_num(daily.get("exposure", pd.Series(index=daily.index, dtype=float))).min())
    metrics["max_exposure"] = float(_num(daily.get("exposure", pd.Series(index=daily.index, dtype=float))).max())
    metrics["time_below_50pct_exposure"] = float((_num(daily.get("exposure", pd.Series(index=daily.index, dtype=float))) < 0.50).mean())
    metrics["turnover"] = float(_num(daily.get("turnover", pd.Series(index=daily.index, dtype=float))).mean())
    return metrics


def _benchmark_comparison(candidate: dict, benchmarks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, bench in benchmarks.iterrows():
        if bench["model"] == CANDIDATE:
            continue
        rows.append({
            "benchmark": bench["model"],
            "return_gap": candidate.get("total_return", np.nan) - bench.get("total_return", np.nan),
            "CAGR_gap": candidate.get("CAGR", np.nan) - bench.get("CAGR", np.nan),
            "Sharpe_gap": candidate.get("Sharpe", np.nan) - bench.get("Sharpe", np.nan),
            "Sortino_gap": candidate.get("Sortino", np.nan) - bench.get("Sortino", np.nan),
            "drawdown_gap": candidate.get("max_drawdown", np.nan) - bench.get("max_drawdown", np.nan),
            "candidate_beats_return": candidate.get("total_return", -np.inf) > bench.get("total_return", np.inf),
            "candidate_beats_sharpe": candidate.get("Sharpe", -np.inf) > bench.get("Sharpe", np.inf),
            "candidate_beats_drawdown": candidate.get("max_drawdown", -np.inf) > bench.get("max_drawdown", np.inf),
        })
    return pd.DataFrame(rows)


def _period_analysis(candidate_daily: pd.DataFrame) -> pd.DataFrame:
    data = candidate_daily.copy()
    data["year"] = data["date"].dt.year
    data["quarter"] = data["date"].dt.to_period("Q").astype(str)
    rows = []
    for year, group in data.groupby("year"):
        row = _metrics(f"year_{year}", group)
        row["period_type"] = "year"
        row["period"] = str(year)
        rows.append(row)
    for quarter, group in data.groupby("quarter"):
        row = _metrics(f"quarter_{quarter}", group)
        row["period_type"] = "quarter"
        row["period"] = quarter
        rows.append(row)
    return pd.DataFrame(rows)


def _robustness(candidate_daily: pd.DataFrame, trades: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = candidate_daily.copy()
    data["year"] = data["date"].dt.year
    yearly = data.groupby("year")["return"].apply(lambda x: (1.0 + _num(x).dropna()).prod() - 1.0)
    if not yearly.empty:
        best_year = int(yearly.idxmax())
        filtered = data[~data["year"].eq(best_year)]
        rows.append({"check": "remove_best_year", "removed": best_year, **_metrics("candidate_without_best_year", filtered)})
    if not trades.empty:
        best_10 = trades.sort_values("trade_contribution", ascending=False).head(10)
        penalty_by_date = best_10.groupby("date")["trade_contribution"].sum()
        filtered = data.copy()
        filtered["return"] = filtered["return"] - filtered["date"].map(penalty_by_date).fillna(0.0)
        rows.append({"check": "remove_best_10_trades", "removed": ",".join(best_10["ticker"].astype(str).tolist()), **_metrics("candidate_without_best_10_trades", filtered)})
        ticker_contrib = trades.groupby("ticker")["trade_contribution"].sum().sort_values(ascending=False)
        if not ticker_contrib.empty:
            best_ticker = ticker_contrib.index[0]
            penalty = trades[trades["ticker"].eq(best_ticker)].groupby("date")["trade_contribution"].sum()
            filtered = data.copy()
            filtered["return"] = filtered["return"] - filtered["date"].map(penalty).fillna(0.0)
            rows.append({"check": "remove_best_ticker", "removed": best_ticker, **_metrics("candidate_without_best_ticker", filtered)})
    rolling_vol = _num(data["return"]).rolling(12, min_periods=4).std()
    rows.append({"check": "high_volatility_periods", "removed": "low_vol_periods", **_metrics("candidate_high_vol_only", data[rolling_vol >= rolling_vol.quantile(0.75)])})
    bear = spy.copy()
    if not bear.empty:
        bear["spy_rolling_return"] = _num(bear["return"]).rolling(12, min_periods=4).sum()
        bear_dates = set(bear[bear["spy_rolling_return"] < 0]["date"])
        rows.append({"check": "bear_market_periods", "removed": "non_bear_periods", **_metrics("candidate_bear_only", data[data["date"].isin(bear_dates)])})
    sideways = spy.copy()
    if not sideways.empty:
        sideways["spy_abs_rolling_return"] = _num(sideways["return"]).rolling(12, min_periods=4).sum().abs()
        side_dates = set(sideways[sideways["spy_abs_rolling_return"] <= sideways["spy_abs_rolling_return"].quantile(0.35)]["date"])
        rows.append({"check": "sideways_periods", "removed": "directional_periods", **_metrics("candidate_sideways_only", data[data["date"].isin(side_dates)])})
    return pd.DataFrame(rows)


def _concentration(trades: pd.DataFrame, candidate_total_return: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    total_contrib = float(_num(trades["trade_contribution"]).sum())
    denom = total_contrib if abs(total_contrib) > 1e-12 else np.nan
    top5 = float(_num(trades["trade_contribution"]).sort_values(ascending=False).head(5).sum())
    top10 = float(_num(trades["trade_contribution"]).sort_values(ascending=False).head(10).sum())
    ticker = trades.groupby("ticker")["trade_contribution"].sum().sort_values(ascending=False)
    rows = [
        {"metric": "top_5_trades_contribution", "value": top5, "share_of_trade_contribution": top5 / denom if np.isfinite(denom) else np.nan},
        {"metric": "top_10_trades_contribution", "value": top10, "share_of_trade_contribution": top10 / denom if np.isfinite(denom) else np.nan},
    ]
    if not ticker.empty:
        rows.append({"metric": "top_ticker_contribution", "ticker": ticker.index[0], "value": float(ticker.iloc[0]), "share_of_trade_contribution": float(ticker.iloc[0] / denom) if np.isfinite(denom) else np.nan})
        rows.append({"metric": "worst_ticker_drag", "ticker": ticker.index[-1], "value": float(ticker.iloc[-1]), "share_of_trade_contribution": float(ticker.iloc[-1] / denom) if np.isfinite(denom) else np.nan})
    rows.append({"metric": "candidate_total_return", "value": candidate_total_return, "share_of_trade_contribution": np.nan})
    return pd.DataFrame(rows)


def _governance(candidate: dict, benchmark_gap: pd.DataFrame, robustness: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    spy_gap = benchmark_gap[benchmark_gap["benchmark"].eq("SPY_buy_hold")]
    qqq_gap = benchmark_gap[benchmark_gap["benchmark"].eq("QQQ_buy_hold")]
    reasons = []
    classification = "research only"
    spy_ok = not spy_gap.empty and bool(spy_gap["candidate_beats_return"].iloc[0]) and bool(spy_gap["candidate_beats_drawdown"].iloc[0])
    sharpe_close = not spy_gap.empty and float(spy_gap["Sharpe_gap"].iloc[0]) > -0.05
    dd_ok = candidate.get("max_drawdown", -np.inf) > -0.20
    depends_one_year = False
    remove_year = robustness[robustness["check"].eq("remove_best_year")]
    if not remove_year.empty:
        depends_one_year = float(remove_year["total_return"].iloc[0]) < 0
    top_conc = concentration[concentration["metric"].eq("top_10_trades_contribution")]
    concentrated = not top_conc.empty and float(top_conc["share_of_trade_contribution"].iloc[0]) > 0.75
    if not spy_ok:
        classification = "reject"
        reasons.append("does not beat SPY return/drawdown")
    if not sharpe_close:
        classification = "reject"
        reasons.append("Sharpe not close to SPY")
    if not dd_ok:
        classification = "reject"
        reasons.append("drawdown above 20%")
    if depends_one_year:
        classification = "reject"
        reasons.append("depends entirely on one year")
    if concentrated:
        classification = "research only"
        reasons.append("trade concentration risk")
    if not reasons:
        if not qqq_gap.empty and bool(qqq_gap["candidate_beats_sharpe"].iloc[0]) and bool(qqq_gap["candidate_beats_drawdown"].iloc[0]):
            classification = "eligible for paper trading"
            reasons.append("beats SPY and has favorable QQQ risk-adjusted comparison")
        else:
            classification = "candidate for shadow mode"
            reasons.append("beats SPY requirements but needs more live validation")
    return pd.DataFrame([{
        "candidate": CANDIDATE,
        "classification": classification,
        "reason": "; ".join(reasons),
        "production_change": "none",
        "uses_hindsight": False,
    }])


def run_growth_candidate_validation() -> dict[str, pd.DataFrame]:
    candidate_daily = _candidate_daily()
    if candidate_daily.empty:
        raise ValueError(f"{CANDIDATE} daily returns are required.")
    trades = _candidate_trades(candidate_daily)
    labels = _labels()
    candidate_metrics = _add_candidate_exposure(_metrics(CANDIDATE, candidate_daily), candidate_daily)
    candidate_metrics.update(_tp_sl(trades, labels))

    comparison_rows = [
        candidate_metrics,
        _metrics("raw_target_research", _raw_daily()),
        _metrics("baseline_old_regime_gate", _mode_daily("regime_gated_full_quant")),
        _metrics("SPY_buy_hold", _benchmark_daily("SPY")),
        _metrics("QQQ_buy_hold", _benchmark_daily("QQQ")),
    ]
    results = pd.DataFrame(comparison_rows)
    benchmark_gap = _benchmark_comparison(candidate_metrics, results)
    robustness = pd.concat([_period_analysis(candidate_daily), _robustness(candidate_daily, trades, _benchmark_daily("SPY"))], ignore_index=True, sort=False)
    concentration = _concentration(trades, candidate_metrics.get("total_return", np.nan))
    governance = _governance(candidate_metrics, benchmark_gap, robustness, concentration)

    results.to_csv(OUT_RESULTS, index=False)
    benchmark_gap.to_csv(OUT_BENCHMARKS, index=False)
    robustness.to_csv(OUT_ROBUSTNESS, index=False)
    concentration.to_csv(OUT_CONCENTRATION, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== GROWTH CANDIDATE VALIDATION =====")
    print(results.to_string(index=False))
    print("\n===== GROWTH CANDIDATE VS BENCHMARKS =====")
    print(benchmark_gap.to_string(index=False))
    print("\n===== GROWTH ROBUSTNESS CHECKS =====")
    show_rob = ["check", "period_type", "period", "removed", "total_return", "CAGR", "Sharpe", "max_drawdown", "observations"]
    print(robustness[[c for c in show_rob if c in robustness.columns]].to_string(index=False))
    print("\n===== GROWTH CONCENTRATION CHECKS =====")
    print(concentration.to_string(index=False))
    print("\n===== GROWTH GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_RESULTS).resolve()}")
    print(f"Saved: {Path(OUT_BENCHMARKS).resolve()}")
    print(f"Saved: {Path(OUT_ROBUSTNESS).resolve()}")
    print(f"Saved: {Path(OUT_CONCENTRATION).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return {
        "results": results,
        "benchmark_gap": benchmark_gap,
        "robustness": robustness,
        "concentration": concentration,
        "governance": governance,
    }


if __name__ == "__main__":
    run_growth_candidate_validation()
