from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE = "soft_exit_rule_vol_target_22pct"
BASE_VARIANT = "soft_exit_rule"

VALIDATION_RESULTS_FILE = "growth_candidate_validation_results.csv"
BENCHMARKS_FILE = "growth_candidate_vs_benchmarks.csv"
ROBUSTNESS_FILE = "growth_candidate_robustness.csv"
CONCENTRATION_FILE = "growth_candidate_concentration.csv"
GROWTH_DAILY_FILE = "growth_volatility_targeting_daily_returns.csv"
GROWTH_EXPOSURE_FILE = "growth_volatility_targeting_exposure.csv"
GROWTH_RESULTS_FILE = "growth_volatility_targeting_results.csv"
EXIT_TRADES_FILE = "exit_rule_walk_forward_trades.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"

OUT_RESULTS = "growth_candidate_deep_validation_results.csv"
OUT_ROLLING = "growth_candidate_rolling_performance.csv"
OUT_DRAWDOWN = "growth_candidate_drawdown_analysis.csv"
OUT_MONTHLY = "growth_candidate_monthly_returns.csv"
OUT_TAIL = "growth_candidate_tail_risk.csv"
OUT_EXPOSURE = "growth_candidate_exposure_analysis.csv"
OUT_RELATIVE = "growth_candidate_relative_performance.csv"
OUT_GOVERNANCE = "growth_candidate_deep_governance.csv"


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


def _sortino(returns: pd.Series, ppy: float) -> float:
    r = _num(returns).dropna()
    downside = r[r < 0].std(ddof=0)
    if r.empty or not np.isfinite(downside) or downside <= 0:
        return np.nan
    return float((r.mean() * ppy) / (downside * np.sqrt(ppy)))


def _sharpe(returns: pd.Series, ppy: float) -> float:
    r = _num(returns).dropna()
    vol = r.std(ddof=0)
    if r.empty or not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float((r.mean() * ppy) / (vol * np.sqrt(ppy)))


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
    mdd = _max_drawdown(r)
    return {
        "model": name,
        "start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "observations": len(r),
        "total_return": total,
        "CAGR": float((1.0 + total) ** (1.0 / years) - 1.0),
        "volatility": vol,
        "Sharpe": _sharpe(r, ppy),
        "Sortino": _sortino(r, ppy),
        "Calmar": np.nan if not np.isfinite(mdd) or mdd >= 0 else float(((1.0 + total) ** (1.0 / years) - 1.0) / abs(mdd)),
        "max_drawdown": mdd,
        "hit_rate": float((r > 0).mean()),
    }


def _candidate_daily() -> pd.DataFrame:
    daily = _dates(_read_csv(GROWTH_DAILY_FILE))
    if daily.empty:
        return daily
    daily = daily[daily.get("vol_target_variant", daily.get("variant", "")).eq(CANDIDATE) | daily.get("variant", "").eq(CANDIDATE)].copy()
    daily["return"] = _num(daily.get("vol_target_return", daily.get("return", pd.Series(index=daily.index, dtype=float))))
    daily["exposure"] = _num(daily.get("target_exposure", pd.Series(index=daily.index, dtype=float)))
    daily["cash"] = _num(daily.get("cash_weight", 1.0 - daily["exposure"]))
    daily = daily.dropna(subset=["date", "return"]).sort_values("date")
    daily["equity"] = (1.0 + daily["return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    return daily


def _benchmark_daily(ticker: str) -> pd.DataFrame:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty:
        return snaps
    if "model_mode" in snaps.columns:
        baseline = snaps[snaps["model_mode"].eq("baseline")]
        if not baseline.empty:
            snaps = baseline.copy()
    data = snaps[snaps["ticker"].eq(ticker)].drop_duplicates("date").sort_values("date").copy()
    if data.empty:
        return data
    data["return"] = _num(data["current_price"]).pct_change()
    data = data.dropna(subset=["return"])
    data["equity"] = (1.0 + data["return"]).cumprod()
    data["drawdown"] = data["equity"] / data["equity"].cummax() - 1.0
    return data


def _candidate_trades(candidate_daily: pd.DataFrame) -> pd.DataFrame:
    trades = _dates(_read_csv(EXIT_TRADES_FILE))
    if trades.empty:
        return trades
    trades = trades[trades["variant"].eq(BASE_VARIANT)].copy()
    exposure = candidate_daily[["date", "exposure", "return"]].rename(columns={"return": "candidate_return"})
    trades = trades.merge(exposure, on="date", how="inner")
    trades["weight_proxy"] = _num(trades.get("weight_proxy", pd.Series(index=trades.index, dtype=float))).fillna(0.0)
    trades["realized_return_5d"] = _num(trades.get("realized_return_5d", pd.Series(index=trades.index, dtype=float))).fillna(0.0)
    trades["trade_contribution"] = trades["weight_proxy"] * trades["realized_return_5d"] * _num(trades["exposure"]).fillna(0.0)
    return trades


def _rolling_performance(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.sort_values("date").copy()
    ppy = _periods_per_year(data["date"])
    windows = {"3m": 13, "6m": 26, "12m": 52}
    rows = []
    for label, window in windows.items():
        rolling_return = (1.0 + data["return"]).rolling(window, min_periods=max(4, window // 3)).apply(np.prod, raw=True) - 1.0
        rolling_vol = data["return"].rolling(window, min_periods=max(4, window // 3)).std(ddof=0) * np.sqrt(ppy)
        rolling_sharpe = (data["return"].rolling(window, min_periods=max(4, window // 3)).mean() * ppy) / rolling_vol
        rolling_dd = data["return"].rolling(window, min_periods=max(4, window // 3)).apply(lambda x: _max_drawdown(pd.Series(x)), raw=False)
        rolling_exposure = data["exposure"].rolling(window, min_periods=max(4, window // 3)).mean()
        for i in range(len(data)):
            rows.append({
                "date": data["date"].iloc[i],
                "window": label,
                "rolling_return": rolling_return.iloc[i],
                "rolling_sharpe": rolling_sharpe.iloc[i],
                "rolling_max_drawdown": rolling_dd.iloc[i],
                "rolling_volatility": rolling_vol.iloc[i],
                "rolling_exposure": rolling_exposure.iloc[i],
            })
    return pd.DataFrame(rows).dropna(subset=["rolling_return"], how="all")


def _drawdown_analysis(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.sort_values("date").copy()
    rows = []
    in_dd = False
    start = None
    trough_date = None
    trough_dd = 0.0
    for _, row in data.iterrows():
        dd = float(row["drawdown"])
        date = row["date"]
        if dd < 0 and not in_dd:
            in_dd = True
            start = date
            trough_date = date
            trough_dd = dd
        elif dd < 0 and in_dd:
            if dd < trough_dd:
                trough_dd = dd
                trough_date = date
        elif dd >= 0 and in_dd:
            rows.append({
                "drawdown_start": start,
                "drawdown_trough": trough_date,
                "drawdown_end": date,
                "max_drawdown": trough_dd,
                "duration_periods": int((data[(data["date"] >= start) & (data["date"] <= date)]).shape[0]),
                "recovered": True,
            })
            in_dd = False
    if in_dd:
        rows.append({
            "drawdown_start": start,
            "drawdown_trough": trough_date,
            "drawdown_end": pd.NaT,
            "max_drawdown": trough_dd,
            "duration_periods": int(data[data["date"] >= start].shape[0]),
            "recovered": False,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["average_recovery_time"] = out.loc[out["recovered"], "duration_periods"].mean()
    out["max_duration_underwater"] = out["duration_periods"].max()
    return out.sort_values("max_drawdown").head(10)


def _monthly_quarterly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.sort_values("date").copy()
    data["month"] = data["date"].dt.to_period("M").astype(str)
    data["quarter"] = data["date"].dt.to_period("Q").astype(str)
    rows = []
    for period_type, col in [("month", "month"), ("quarter", "quarter")]:
        for period, group in data.groupby(col):
            ret = float((1.0 + group["return"]).prod() - 1.0)
            rows.append({
                "period_type": period_type,
                "period": period,
                "return": ret,
                "win_rate": float((_num(group["return"]) > 0).mean()),
                "observations": len(group),
                "best_period_flag": False,
                "worst_period_flag": False,
            })
    out = pd.DataFrame(rows)
    for period_type in ["month", "quarter"]:
        mask = out["period_type"].eq(period_type)
        if mask.any():
            out.loc[mask & out["return"].eq(out.loc[mask, "return"].max()), "best_period_flag"] = True
            out.loc[mask & out["return"].eq(out.loc[mask, "return"].min()), "worst_period_flag"] = True
    return out


def _tail_risk(daily: pd.DataFrame) -> pd.DataFrame:
    r = _num(daily["return"]).dropna()
    if r.empty:
        return pd.DataFrame()
    var95 = float(r.quantile(0.05))
    var99 = float(r.quantile(0.01))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else np.nan
    cvar99 = float(r[r <= var99].mean()) if (r <= var99).any() else np.nan
    rows = [{
        "metric": "summary",
        "worst_period_return": float(r.min()),
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "VaR_95": var95,
        "VaR_99": var99,
        "CVaR_95": cvar95,
        "CVaR_99": cvar99,
    }]
    worst = daily.assign(return_num=_num(daily["return"])).sort_values("return_num").head(5)
    for _, row in worst.iterrows():
        rows.append({
            "metric": "worst_5_periods",
            "date": row["date"],
            "worst_period_return": row["return_num"],
            "exposure": row.get("exposure"),
            "drawdown": row.get("drawdown"),
        })
    return pd.DataFrame(rows)


def _exposure_analysis(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    best_threshold = data["return"].quantile(0.90)
    dd_threshold = data["drawdown"].quantile(0.10)
    rows = [
        {"metric": "average_exposure", "value": float(data["exposure"].mean())},
        {"metric": "min_exposure", "value": float(data["exposure"].min())},
        {"metric": "max_exposure", "value": float(data["exposure"].max())},
        {"metric": "exposure_p25", "value": float(data["exposure"].quantile(0.25))},
        {"metric": "exposure_p50", "value": float(data["exposure"].quantile(0.50))},
        {"metric": "exposure_p75", "value": float(data["exposure"].quantile(0.75))},
        {"metric": "exposure_during_worst_drawdowns", "value": float(data.loc[data["drawdown"] <= dd_threshold, "exposure"].mean())},
        {"metric": "exposure_during_best_periods", "value": float(data.loc[data["return"] >= best_threshold, "exposure"].mean())},
        {"metric": "vol_target_reacts_correctly_proxy", "value": float(data["exposure"].corr(_num(data["return"]).rolling(12, min_periods=4).std() * -1))},
    ]
    return pd.DataFrame(rows)


def _relative_performance(candidate: pd.DataFrame, benchmark: pd.DataFrame, name: str) -> pd.DataFrame:
    merged = candidate[["date", "return", "drawdown"]].merge(
        benchmark[["date", "return", "drawdown"]], on="date", suffixes=("_candidate", f"_{name}")
    ).sort_values("date")
    if merged.empty:
        return pd.DataFrame()
    ppy = _periods_per_year(merged["date"])
    excess = merged["return_candidate"] - merged[f"return_{name}"]
    rolling_window = 26
    out = pd.DataFrame({
        "date": merged["date"],
        "benchmark": name,
        "rolling_excess_return": (1.0 + excess).rolling(rolling_window, min_periods=8).apply(np.prod, raw=True) - 1.0,
        "rolling_tracking_difference": excess.rolling(rolling_window, min_periods=8).mean(),
        "rolling_sharpe_difference": (
            (merged["return_candidate"].rolling(rolling_window, min_periods=8).mean() * ppy)
            / (merged["return_candidate"].rolling(rolling_window, min_periods=8).std(ddof=0) * np.sqrt(ppy))
        ) - (
            (merged[f"return_{name}"].rolling(rolling_window, min_periods=8).mean() * ppy)
            / (merged[f"return_{name}"].rolling(rolling_window, min_periods=8).std(ddof=0) * np.sqrt(ppy))
        ),
        "rolling_drawdown_difference": merged["drawdown_candidate"] - merged[f"drawdown_{name}"],
    })
    return out


def _robustness_extension(daily: pd.DataFrame, trades: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = daily.copy()
    data["year"] = data["date"].dt.year
    yearly = data.groupby("year")["return"].apply(lambda x: (1.0 + _num(x).dropna()).prod() - 1.0)
    if not yearly.empty:
        best_year = int(yearly.idxmax())
        rows.append({"check": "exclude_best_year", "removed": best_year, **_metrics("candidate", data[~data["year"].eq(best_year)])})
    if not trades.empty:
        ticker_contrib = trades.groupby("ticker")["trade_contribution"].sum().sort_values(ascending=False)
        for n, check in [(1, "exclude_best_ticker"), (5, "exclude_top_5_tickers")]:
            removed = ticker_contrib.head(n).index.tolist()
            penalty = trades[trades["ticker"].isin(removed)].groupby("date")["trade_contribution"].sum()
            filtered = data.copy()
            filtered["return"] = filtered["return"] - filtered["date"].map(penalty).fillna(0.0)
            rows.append({"check": check, "removed": ",".join(removed), **_metrics("candidate", filtered)})
    rolling_vol = data["return"].rolling(12, min_periods=4).std()
    rows.append({"check": "high_volatility_regime", "removed": "low_vol", **_metrics("candidate", data[rolling_vol >= rolling_vol.quantile(0.75)])})
    if not spy.empty:
        spy = spy.copy()
        spy["trend_12p"] = spy["return"].rolling(12, min_periods=4).sum()
        up_dates = set(spy[spy["trend_12p"] > 0]["date"])
        down_dates = set(spy[spy["trend_12p"] < 0]["date"])
        rows.append({"check": "market_uptrend_regime", "removed": "non_uptrend", **_metrics("candidate", data[data["date"].isin(up_dates)])})
        rows.append({"check": "market_downtrend_regime", "removed": "non_downtrend", **_metrics("candidate", data[data["date"].isin(down_dates)])})
    return pd.DataFrame(rows)


def _governance(results: pd.DataFrame, drawdowns: pd.DataFrame, rolling: pd.DataFrame, concentration: pd.DataFrame, robustness: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    candidate = results[results["model"].eq(CANDIDATE)].iloc[0]
    reasons = []
    classification = "eligible for paper trading"
    if float(candidate["max_drawdown"]) < -0.20:
        classification = "reject"
        reasons.append("max DD above 20%")
    roll_sharpe = rolling[rolling["window"].eq("12m")]["rolling_sharpe"].dropna()
    if not roll_sharpe.empty and float((roll_sharpe < 0).mean()) > 0.25:
        classification = "research only"
        reasons.append("rolling Sharpe structurally weak in too many 12m windows")
    top_ticker = concentration[concentration["metric"].eq("top_ticker_contribution")]
    if not top_ticker.empty and float(top_ticker["share_of_trade_contribution"].iloc[0]) > 0.35:
        classification = "research only"
        reasons.append("too dependent on one ticker")
    top10 = concentration[concentration["metric"].eq("top_10_trades_contribution")]
    if not top10.empty and float(top10["share_of_trade_contribution"].iloc[0]) > 0.60:
        classification = "research only"
        reasons.append("too dependent on top trades")
    summary_tail = tail[tail["metric"].eq("summary")]
    if not summary_tail.empty and float(summary_tail["CVaR_95"].iloc[0]) < -0.10:
        classification = "research only"
        reasons.append("tail risk high")
    exposure_react = pd.read_csv(OUT_EXPOSURE) if Path(OUT_EXPOSURE).exists() else pd.DataFrame()
    if not reasons:
        classification = "extended paper trading candidate"
        reasons.append("passes deep validation constraints; no production change")
    return pd.DataFrame([{
        "candidate": CANDIDATE,
        "classification": classification,
        "reason": "; ".join(reasons),
        "production_change": "none",
        "uses_hindsight": False,
    }])


def run_growth_candidate_deep_validation() -> dict[str, pd.DataFrame]:
    candidate_daily = _candidate_daily()
    if candidate_daily.empty:
        raise ValueError(f"{CANDIDATE} daily returns are required.")
    spy = _benchmark_daily("SPY")
    qqq = _benchmark_daily("QQQ")
    trades = _candidate_trades(candidate_daily)

    results = pd.DataFrame([
        _metrics(CANDIDATE, candidate_daily),
        _metrics("SPY_buy_hold", spy),
        _metrics("QQQ_buy_hold", qqq),
    ])
    rolling = _rolling_performance(candidate_daily)
    drawdowns = _drawdown_analysis(candidate_daily)
    monthly = _monthly_quarterly_returns(candidate_daily)
    tail = _tail_risk(candidate_daily)
    exposure = _exposure_analysis(candidate_daily)
    relative = pd.concat([
        _relative_performance(candidate_daily, spy, "SPY"),
        _relative_performance(candidate_daily, qqq, "QQQ"),
    ], ignore_index=True, sort=False)
    robustness_extra = _robustness_extension(candidate_daily, trades, spy)
    prior_robustness = _read_csv(ROBUSTNESS_FILE)
    robustness = pd.concat([prior_robustness, robustness_extra], ignore_index=True, sort=False)
    prior_concentration = _read_csv(CONCENTRATION_FILE)
    concentration = prior_concentration if not prior_concentration.empty else _concentration_fallback(trades, candidate_daily)
    governance = _governance(results, drawdowns, rolling, concentration, robustness, tail)

    results.to_csv(OUT_RESULTS, index=False)
    rolling.to_csv(OUT_ROLLING, index=False)
    drawdowns.to_csv(OUT_DRAWDOWN, index=False)
    monthly.to_csv(OUT_MONTHLY, index=False)
    tail.to_csv(OUT_TAIL, index=False)
    exposure.to_csv(OUT_EXPOSURE, index=False)
    relative.to_csv(OUT_RELATIVE, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== GROWTH CANDIDATE DEEP VALIDATION =====")
    print(results.to_string(index=False))
    print("\n===== ROLLING PERFORMANCE =====")
    print(rolling.groupby("window").agg(
        avg_rolling_return=("rolling_return", "mean"),
        min_rolling_return=("rolling_return", "min"),
        avg_rolling_sharpe=("rolling_sharpe", "mean"),
        min_rolling_sharpe=("rolling_sharpe", "min"),
        worst_rolling_dd=("rolling_max_drawdown", "min"),
        avg_rolling_exposure=("rolling_exposure", "mean"),
    ).reset_index().to_string(index=False))
    print("\n===== DRAWDOWN ANALYSIS =====")
    print(drawdowns.to_string(index=False))
    print("\n===== TAIL RISK =====")
    print(tail.to_string(index=False))
    print("\n===== EXPOSURE ANALYSIS =====")
    print(exposure.to_string(index=False))
    print("\n===== BENCHMARK RELATIVE PERFORMANCE =====")
    print(relative.groupby("benchmark").agg(
        avg_rolling_excess_return=("rolling_excess_return", "mean"),
        min_rolling_excess_return=("rolling_excess_return", "min"),
        avg_rolling_sharpe_difference=("rolling_sharpe_difference", "mean"),
        avg_rolling_drawdown_difference=("rolling_drawdown_difference", "mean"),
    ).reset_index().to_string(index=False))
    print("\n===== GROWTH GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_RESULTS).resolve()}")
    print(f"Saved: {Path(OUT_ROLLING).resolve()}")
    print(f"Saved: {Path(OUT_DRAWDOWN).resolve()}")
    print(f"Saved: {Path(OUT_MONTHLY).resolve()}")
    print(f"Saved: {Path(OUT_TAIL).resolve()}")
    print(f"Saved: {Path(OUT_EXPOSURE).resolve()}")
    print(f"Saved: {Path(OUT_RELATIVE).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return {
        "results": results,
        "rolling": rolling,
        "drawdowns": drawdowns,
        "monthly": monthly,
        "tail": tail,
        "exposure": exposure,
        "relative": relative,
        "governance": governance,
    }


def _concentration_fallback(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    contrib = trades.groupby("ticker")["trade_contribution"].sum().sort_values(ascending=False)
    total = float(trades["trade_contribution"].sum())
    denom = total if abs(total) > 1e-12 else np.nan
    return pd.DataFrame([
        {"metric": "top_ticker_contribution", "ticker": contrib.index[0], "value": float(contrib.iloc[0]), "share_of_trade_contribution": float(contrib.iloc[0] / denom) if np.isfinite(denom) else np.nan},
        {"metric": "top_10_trades_contribution", "value": float(trades["trade_contribution"].sort_values(ascending=False).head(10).sum()), "share_of_trade_contribution": float(trades["trade_contribution"].sort_values(ascending=False).head(10).sum() / denom) if np.isfinite(denom) else np.nan},
    ])


if __name__ == "__main__":
    run_growth_candidate_deep_validation()
