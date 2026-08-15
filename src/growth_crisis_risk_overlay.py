from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DAILY = "reconstructed_growth_long_horizon_daily_returns.csv"
BASE_TRADES = "reconstructed_growth_long_horizon_trades.csv"
BASE_RESULTS = "reconstructed_growth_long_horizon_results.csv"
BASE_STRESS = "reconstructed_growth_stress_periods.csv"
BENCHMARK_RESULTS = "reconstructed_growth_benchmark_comparison.csv"
PRICE_CACHE_DIR = Path("yahoo_ohlcv_price_cache")

OUT_RESULTS = "growth_crisis_overlay_results.csv"
OUT_DAILY = "growth_crisis_overlay_daily_returns.csv"
OUT_STRESS = "growth_crisis_overlay_stress_periods.csv"
OUT_GOVERNANCE = "growth_crisis_overlay_governance.csv"

NORMAL_CAP = 0.60

STRESS_PERIODS = {
    "2011_euro_crisis": ("2011-07-01", "2011-12-31"),
    "2018_q4_selloff": ("2018-10-01", "2018-12-31"),
    "covid_crash_2020": ("2020-02-15", "2020-04-30"),
    "2022_bear_market": ("2022-01-01", "2022-12-31"),
    "2024_ai_bull_market": ("2024-01-01", "2024-12-31"),
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out.dropna(subset=[col]).sort_values(col)


def _num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _periods_per_year(dates: pd.Series) -> float:
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return 52.0
    median_step = dates.diff().dt.days.dropna().median()
    return float(365.25 / median_step) if pd.notna(median_step) and median_step > 0 else 52.0


def _metrics(name: str, df: pd.DataFrame, return_col: str = "overlay_return") -> dict[str, object]:
    if df.empty or return_col not in df.columns:
        return {"model": name, "observations": 0}
    data = _dates(df)
    returns = _num(data[return_col]).dropna()
    if returns.empty:
        return {"model": name, "observations": 0}
    ppy = _periods_per_year(data["date"])
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(returns) / ppy, 1e-9)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    vol = float(returns.std(ddof=0) * np.sqrt(ppy))
    sharpe = float((returns.mean() * ppy) / vol) if vol > 0 else np.nan
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(ppy) if (returns < 0).any() else np.nan
    sortino = float((returns.mean() * ppy) / downside) if pd.notna(downside) and downside > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    return {
        "model": name,
        "start_date": data["date"].min().strftime("%Y-%m-%d"),
        "end_date": data["date"].max().strftime("%Y-%m-%d"),
        "observations": int(len(returns)),
        "total_return": total_return,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "hit_rate": float((returns > 0).mean()),
    }


def _load_price(ticker: str) -> pd.Series:
    path = PRICE_CACHE_DIR / f"{ticker}.csv"
    df = _read_csv(path)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if col is None:
        return pd.Series(dtype=float, name=ticker)
    out = pd.to_numeric(df[col], errors="coerce").dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out.rename(ticker)


def _asof(series: pd.Series, dates: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(np.nan, index=dates.index)
    left = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize()})
    right = series.reset_index()
    right.columns = ["date", "value"]
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    merged = pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date", direction="backward")
    merged.index = left.sort_values("date").index
    return merged.reindex(dates.index)["value"]


def _market_features(dates: pd.Series) -> pd.DataFrame:
    spy = _load_price("SPY")
    qqq = _load_price("QQQ")
    features = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize()})
    for ticker, price in {"SPY": spy, "QQQ": qqq}.items():
        ma200 = price.rolling(200, min_periods=150).mean().shift(1)
        close_shifted = price.shift(1)
        features[f"{ticker.lower()}_price"] = _asof(close_shifted, features["date"])
        features[f"{ticker.lower()}_ma200"] = _asof(ma200, features["date"])
        features[f"{ticker.lower()}_below_200dma"] = (
            features[f"{ticker.lower()}_price"] < features[f"{ticker.lower()}_ma200"]
        )
    spy_ret = spy.pct_change()
    spy_vol20 = spy_ret.rolling(20, min_periods=15).std().shift(1) * np.sqrt(252)
    spy_vol90 = spy_vol20.rolling(252, min_periods=126).quantile(0.90).shift(1)
    features["spy_realized_vol20"] = _asof(spy_vol20, features["date"])
    features["spy_vol20_p90"] = _asof(spy_vol90, features["date"])
    features["spy_high_vol"] = features["spy_realized_vol20"] > features["spy_vol20_p90"]
    return features


def _cap_for_overlay(row: pd.Series, overlay: str, drawdown_state: dict[str, object]) -> tuple[float, str]:
    spy_below = row.get("spy_below_200dma", False) is True
    qqq_below = row.get("qqq_below_200dma", False) is True
    if overlay == "base_growth_v2":
        return NORMAL_CAP, "base cap 60%"
    if overlay == "spy_trend_filter":
        return (0.30, "SPY below 200D MA") if spy_below else (NORMAL_CAP, "SPY above 200D MA")
    if overlay == "qqq_trend_filter":
        return (0.30, "QQQ below 200D MA") if qqq_below else (NORMAL_CAP, "QQQ above 200D MA")
    if overlay == "dual_trend_filter":
        if spy_below and qqq_below:
            return 0.25, "SPY and QQQ below 200D MA"
        if spy_below or qqq_below:
            return 0.40, "one benchmark below 200D MA"
        return NORMAL_CAP, "SPY and QQQ above 200D MA"
    if overlay == "realized_volatility_filter":
        high_vol = row.get("spy_high_vol", False) is True
        return (0.30, "SPY 20D vol above trailing 90th percentile") if high_vol else (NORMAL_CAP, "SPY vol normal")
    if overlay == "drawdown_aware_trend_filter":
        if drawdown_state.get("defensive", False):
            if not spy_below:
                drawdown_state["defensive"] = False
                return NORMAL_CAP, "SPY recovered above 200D MA"
            return 0.25, "defensive mode active until SPY recovers"
        current_dd = float(drawdown_state.get("current_drawdown", 0.0))
        if current_dd <= -0.15 and spy_below:
            drawdown_state["defensive"] = True
            return 0.25, "portfolio DD > 15% and SPY below 200D MA"
        return NORMAL_CAP, "drawdown/trend guard inactive"
    return NORMAL_CAP, "unknown overlay fallback"


def _apply_overlay(window_df: pd.DataFrame, overlay: str) -> pd.DataFrame:
    data = _dates(window_df).copy()
    if data.empty:
        return data
    market = _market_features(data["date"]).reset_index(drop=True)
    data = data.reset_index(drop=True).join(market.drop(columns=["date"]))
    returns: list[float] = []
    caps: list[float] = []
    scales: list[float] = []
    reasons: list[str] = []
    equity = 1.0
    peak = 1.0
    state: dict[str, object] = {"defensive": False, "current_drawdown": 0.0}
    for _, row in data.iterrows():
        state["current_drawdown"] = equity / peak - 1.0
        cap, reason = _cap_for_overlay(row, overlay, state)
        base_exposure = float(row.get("target_exposure", NORMAL_CAP))
        if not np.isfinite(base_exposure) or base_exposure <= 0:
            scale = 0.0
        else:
            scale = min(1.0, cap / base_exposure)
        base_return = float(row.get("return", 0.0))
        overlay_return = base_return * scale
        equity *= 1.0 + overlay_return
        peak = max(peak, equity)
        returns.append(overlay_return)
        caps.append(cap)
        scales.append(scale)
        reasons.append(reason)
    data["overlay"] = overlay
    data["overlay_cap"] = caps
    data["overlay_scale"] = scales
    data["overlay_exposure"] = _num(data["target_exposure"]).fillna(0.0) * data["overlay_scale"]
    data["overlay_cash"] = 1.0 - data["overlay_exposure"]
    data["overlay_return"] = returns
    data["overlay_reason"] = reasons
    data["overlay_turnover_proxy"] = _num(data.get("turnover", pd.Series(0.0, index=data.index))).fillna(0.0) * data["overlay_scale"]
    return data


def _base_lookup() -> dict[str, dict[str, float]]:
    base = _read_csv(BASE_RESULTS)
    out: dict[str, dict[str, float]] = {}
    if base.empty or "window_start" not in base.columns:
        return out
    for _, row in base.iterrows():
        window = str(row.get("window_start", ""))
        out[window] = {
            "base_CAGR": float(row.get("CAGR", np.nan)),
            "base_Sharpe": float(row.get("Sharpe", np.nan)),
            "base_max_drawdown": float(row.get("max_drawdown", np.nan)),
        }
    return out


def run_overlay() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_daily = _read_csv(BASE_DAILY)
    if base_daily.empty:
        raise FileNotFoundError(f"Missing {BASE_DAILY}. Run reconstructed backtest first.")
    base_daily = _dates(base_daily)
    overlays = [
        "base_growth_v2",
        "spy_trend_filter",
        "qqq_trend_filter",
        "dual_trend_filter",
        "realized_volatility_filter",
        "drawdown_aware_trend_filter",
    ]
    daily_frames: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    base_by_window = _base_lookup()
    for window, window_df in base_daily.groupby("window_start"):
        window = str(window)
        for overlay in overlays:
            odf = _apply_overlay(window_df, overlay)
            if odf.empty:
                continue
            odf["window_start"] = window
            daily_frames.append(odf)
            metrics = _metrics(f"{window}_{overlay}", odf)
            metrics["window_start"] = window
            metrics["overlay"] = overlay
            metrics["average_exposure"] = float(_num(odf["overlay_exposure"]).mean())
            metrics["average_cash"] = float(_num(odf["overlay_cash"]).mean())
            metrics["average_turnover"] = float(_num(odf["overlay_turnover_proxy"]).mean())
            base_stats = base_by_window.get(window, {})
            metrics.update(base_stats)
            metrics["return_lost_vs_base"] = float(base_stats.get("base_CAGR", np.nan) - metrics.get("CAGR", np.nan))
            metrics["DD_improvement_vs_base"] = float(metrics.get("max_drawdown", np.nan) - base_stats.get("base_max_drawdown", np.nan))
            result_rows.append(metrics)
            for stress_name, (start, end) in STRESS_PERIODS.items():
                mask = odf["date"].between(pd.Timestamp(start), pd.Timestamp(end))
                sdf = odf.loc[mask].copy()
                sm = _metrics(f"{window}_{overlay}_{stress_name}", sdf)
                sm["window_start"] = window
                sm["overlay"] = overlay
                sm["stress_period"] = stress_name
                stress_rows.append(sm)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    results = pd.DataFrame(result_rows)
    stress = pd.DataFrame(stress_rows)
    if not results.empty:
        bench = _read_csv(BENCHMARK_RESULTS)
        if not bench.empty and {"window_start", "benchmark"}.issubset(bench.columns):
            bench_pivot = bench.pivot_table(index="window_start", columns="benchmark", values=["CAGR", "Sharpe", "max_drawdown"], aggfunc="first")
            bench_pivot.columns = [f"{metric}_{benchmark}" for metric, benchmark in bench_pivot.columns]
            results = results.merge(bench_pivot.reset_index(), on="window_start", how="left")
            results["beats_SPY_CAGR"] = results["CAGR"] > results.get("CAGR_SPY", np.nan)
            results["beats_QQQ_CAGR"] = results["CAGR"] > results.get("CAGR_QQQ", np.nan)
            results["beats_SPY_Sharpe"] = results["Sharpe"] > results.get("Sharpe_SPY", np.nan)
            results["beats_QQQ_Sharpe"] = results["Sharpe"] > results.get("Sharpe_QQQ", np.nan)
    daily.to_csv(OUT_DAILY, index=False)
    results.to_csv(OUT_RESULTS, index=False)
    stress.to_csv(OUT_STRESS, index=False)
    governance = _governance(results)
    governance.to_csv(OUT_GOVERNANCE, index=False)
    _print_report(results, stress, governance)
    return results, daily, governance


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if results.empty:
        return pd.DataFrame([{"classification": "research_only", "reason": "No overlay results generated."}])
    for overlay, group in results.groupby("overlay"):
        if overlay == "base_growth_v2":
            continue
        long_windows = group[group["window_start"].isin(["2008-01-01", "2015-01-01", "2020-01-01"])]
        base_group = results[(results["overlay"] == "base_growth_v2") & (results["window_start"].isin(long_windows["window_start"]))]
        if long_windows.empty or base_group.empty:
            classification = "research_only"
            reason = "Insufficient comparable windows."
        else:
            dd_better = float((long_windows["DD_improvement_vs_base"] > 0.05).mean())
            cagr_above_spy = bool(long_windows.get("beats_SPY_CAGR", pd.Series(False)).fillna(False).mean() >= 0.67)
            sharpe_ok = bool((long_windows["Sharpe"] >= base_group["Sharpe"].values - 0.05).mean() >= 0.67)
            calmar_better = bool((long_windows["Calmar"] > base_group["Calmar"].values).mean() >= 0.67)
            bull = group[(group["window_start"].eq("2020-01-01")) | (group["window_start"].eq("2022-01-03"))]
            bull_destroyed = bool((bull["return_lost_vs_base"] > 0.15).any()) if not bull.empty else False
            if dd_better >= 0.67 and cagr_above_spy and (sharpe_ok or calmar_better) and not bull_destroyed:
                classification = "candidate_for_growth_v3"
                reason = "Drawdown improved across long windows while preserving enough return and risk-adjusted quality."
            elif dd_better >= 0.34:
                classification = "research_only"
                reason = "Some drawdown improvement, but return/Sharpe tradeoff is not clearly robust."
            else:
                classification = "reject"
                reason = "Overlay does not materially improve drawdown across key long windows."
        rows.append(
            {
                "overlay": overlay,
                "classification": classification,
                "production_changed": False,
                "paper_changed": False,
                "parameter_tuning": False,
                "hindsight": False,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _pct(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def _print_report(results: pd.DataFrame, stress: pd.DataFrame, governance: pd.DataFrame) -> None:
    print("\n===== GROWTH CRISIS RISK OVERLAY =====")
    if results.empty:
        print("No overlay results generated.")
        return
    print("Mode: research only. Production/paper trading unchanged.")
    print("Assumption: reconstructed base returns are exposure-scaled by causal crisis caps.")

    print("\n===== OVERLAY COMPARISON =====")
    cols = [
        "window_start",
        "overlay",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_exposure",
        "return_lost_vs_base",
        "DD_improvement_vs_base",
    ]
    display = results[cols].copy()
    for col in ["CAGR", "max_drawdown", "average_exposure", "return_lost_vs_base", "DD_improvement_vs_base"]:
        display[col] = display[col].map(_pct)
    for col in ["Sharpe", "Sortino", "Calmar"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")
    print(display.to_string(index=False))

    print("\n===== STRESS PERIOD IMPACT =====")
    if not stress.empty:
        stress_view = stress[stress["stress_period"].isin(["2011_euro_crisis", "2018_q4_selloff", "covid_crash_2020", "2022_bear_market", "2024_ai_bull_market"])]
        stress_view = stress_view[["window_start", "overlay", "stress_period", "total_return", "max_drawdown", "Sharpe"]].copy()
        stress_view["total_return"] = stress_view["total_return"].map(_pct)
        stress_view["max_drawdown"] = stress_view["max_drawdown"].map(_pct)
        stress_view["Sharpe"] = pd.to_numeric(stress_view["Sharpe"], errors="coerce").map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")
        print(stress_view.head(80).to_string(index=False))

    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))


if __name__ == "__main__":
    run_overlay()
