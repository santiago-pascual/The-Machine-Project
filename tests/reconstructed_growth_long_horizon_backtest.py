from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_target_model import generate_targets_advanced

COVERAGE_FILE = "yahoo_historical_ohlcv_coverage.csv"
PRODUCTION_PARITY_FILE = "production_parity_growth_daily_returns.csv"
PRICE_CACHE_DIR = Path("yahoo_ohlcv_price_cache")

OUT_RESULTS = "reconstructed_growth_long_horizon_results.csv"
OUT_DAILY = "reconstructed_growth_long_horizon_daily_returns.csv"
OUT_TRADES = "reconstructed_growth_long_horizon_trades.csv"
OUT_STRESS = "reconstructed_growth_stress_periods.csv"
OUT_PARITY = "reconstructed_vs_production_parity_check.csv"
OUT_GOVERNANCE = "reconstructed_growth_governance.csv"

WINDOWS = ["2008-01-01", "2010-01-01", "2015-01-01", "2020-01-01", "2022-01-03"]
TARGET_VOL = 0.22
EXPOSURE_CAP = 0.60
MIN_EXPOSURE = 0.40
MAX_EXPOSURE = 1.00
MAX_EXPOSURE_CHANGE = 0.15
BASE_POSITIONS = 2
MAX_POSITIONS = 4
LOOKBACK = 252
STEP_DAYS = 5


STRESS_PERIODS = {
    "2008_crisis": ("2008-09-01", "2009-03-31"),
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


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _import_yfinance():
    if importlib.util.find_spec("yfinance") is None:
        return None
    import yfinance as yf  # type: ignore

    cache_dir = Path("yf_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        try:
            yf.set_tz_cache_location(str(cache_dir))
        except Exception:
            pass
    return yf


def _flatten_download(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    out = data.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if ticker in out.columns.get_level_values(-1):
            out = out.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in out.columns.get_level_values(0):
            out = out.xs(ticker, axis=1, level=0, drop_level=True)
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out.dropna(how="all")


def _load_or_download_prices(tickers: list[str], sleep_seconds: float, retries: int) -> pd.DataFrame:
    PRICE_CACHE_DIR.mkdir(exist_ok=True)
    yf = _import_yfinance()
    series = {}
    for idx, ticker in enumerate(tickers, start=1):
        cache = PRICE_CACHE_DIR / f"{ticker}.csv"
        df = pd.DataFrame()
        if cache.exists():
            df = _read_csv(cache)
            if not df.empty and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).set_index("Date")
        if df.empty and yf is not None:
            for attempt in range(retries + 1):
                try:
                    raw = yf.download(
                        ticker,
                        start="2001-01-01",
                        progress=False,
                        auto_adjust=False,
                        actions=False,
                        threads=False,
                        timeout=20,
                    )
                    df = _flatten_download(raw, ticker)
                    if not df.empty:
                        out = df.copy()
                        out.index.name = "Date"
                        out.reset_index().to_csv(cache, index=False)
                        break
                except Exception:
                    if attempt >= retries:
                        df = pd.DataFrame()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        if df.empty:
            continue
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            continue
        prices = pd.to_numeric(df[col], errors="coerce").dropna()
        if not prices.empty:
            series[ticker] = prices.rename(ticker)
        if sleep_seconds > 0 and idx % 20 == 0:
            time.sleep(sleep_seconds)
    if not series:
        return pd.DataFrame()
    prices_df = pd.concat(series.values(), axis=1).sort_index()
    prices_df.index = pd.to_datetime(prices_df.index).tz_localize(None)
    return prices_df


def _tickers_for_window(start: str, universe: str) -> list[str]:
    coverage = _read_csv(COVERAGE_FILE)
    if coverage.empty:
        return []
    if "universe" in coverage.columns:
        coverage = coverage[coverage["universe"].astype(str).eq(universe)]
    start_ts = pd.Timestamp(start)
    year = start[:4]
    col = f"enough_history_from_{year}"
    ok_base = coverage[coverage["download_status"].astype(str).eq("ok")].copy()
    if col in ok_base.columns:
        ok = ok_base[ok_base[col].astype(str).str.lower().isin(["true", "1", "yes"])]
    else:
        first = pd.to_datetime(ok_base.get("first_available_date", pd.Series(dtype=str)), errors="coerce")
        obs = pd.to_numeric(ok_base.get("observations", pd.Series(dtype=float)), errors="coerce").fillna(0)
        ok = ok_base[(first <= start_ts) & (obs >= LOOKBACK + STEP_DAYS + 2)]
    return ok["ticker"].dropna().astype(str).tolist()


def _periods_per_year(dates: pd.Series) -> float:
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return 52.0
    step = np.median(dates.diff().dt.days.dropna())
    return float(365.25 / step) if np.isfinite(step) and step > 0 else 52.0


def _metrics(name: str, df: pd.DataFrame, return_col: str = "return") -> dict[str, object]:
    if df.empty or return_col not in df.columns:
        return {"model": name, "observations": 0}
    data = _dates(df).sort_values("date")
    returns = _num(data[return_col]).dropna()
    if returns.empty:
        return {"model": name, "observations": 0}
    ppy = _periods_per_year(data["date"])
    equity = (1.0 + returns).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(returns) / ppy, 1e-9)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
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
        "observations": len(returns),
        "total_return": total,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "hit_rate": float((returns > 0).mean()),
    }


def _target_exposure(prior_returns: list[float], previous_exposure: float) -> tuple[float, float, float]:
    if len(prior_returns) < 4:
        rolling_vol = np.nan
        raw = previous_exposure
    else:
        recent = pd.Series(prior_returns[-12:], dtype=float)
        rolling_vol = float(recent.std(ddof=0) * np.sqrt(52))
        raw = previous_exposure if not np.isfinite(rolling_vol) or rolling_vol <= 0 else TARGET_VOL / rolling_vol
    raw = float(np.clip(raw, MIN_EXPOSURE, MAX_EXPOSURE))
    change = float(np.clip(raw - previous_exposure, -MAX_EXPOSURE_CHANGE, MAX_EXPOSURE_CHANGE))
    uncapped = float(np.clip(previous_exposure + change, MIN_EXPOSURE, MAX_EXPOSURE))
    final = float(min(uncapped, EXPOSURE_CAP))
    return final, uncapped, rolling_vol


def _raw_targets_from_history(hist: pd.DataFrame) -> tuple[pd.Series, str]:
    try:
        outputs = generate_targets_advanced(hist)
        current = hist.ffill().iloc[-1]
        raw = outputs["target_price"].reindex(hist.columns) / current.reindex(hist.columns) - 1.0
        return raw.replace([np.inf, -np.inf], np.nan), "advanced_target_model.generate_targets_advanced"
    except Exception:
        close = hist.ffill()
        mom20 = close.iloc[-1] / close.shift(20).iloc[-1] - 1.0
        mom60 = close.iloc[-1] / close.shift(60).iloc[-1] - 1.0
        raw = 0.4 * mom20 + 0.6 * mom60
        return raw.replace([np.inf, -np.inf], np.nan), "fallback_momentum_proxy"


def _run_window(start: str, prices: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        return pd.DataFrame(), pd.DataFrame()
    px = prices.reindex(columns=tickers).dropna(axis=1, how="all").ffill()
    px = px[px.index >= pd.Timestamp(start)]
    px = px.dropna(axis=1, thresh=LOOKBACK + 20)
    if px.empty or len(px) < LOOKBACK + STEP_DAYS + 2:
        return pd.DataFrame(), pd.DataFrame()
    dates = px.index[LOOKBACK::STEP_DAYS]
    daily_rows = []
    trade_rows = []
    prior_weights: dict[str, float] = {}
    prior_selected: set[str] = set()
    prior_returns: list[float] = []
    previous_exposure = EXPOSURE_CAP
    for decision_date in dates:
        loc = px.index.get_loc(decision_date)
        if loc + STEP_DAYS + 1 >= len(px):
            break
        entry_date = px.index[loc + 1]
        exit_date = px.index[min(loc + STEP_DAYS + 1, len(px) - 1)]
        hist = px.iloc[: loc + 1].dropna(axis=1, thresh=LOOKBACK)
        if hist.shape[1] < 2:
            continue
        raw, source = _raw_targets_from_history(hist)
        raw = raw.dropna().sort_values(ascending=False)
        positive = raw[raw > 0]
        base = positive.head(BASE_POSITIONS).index.astype(str).tolist()
        soft_keep = [ticker for ticker in prior_selected if ticker in raw.index and raw.loc[ticker] > 0]
        selected = list(dict.fromkeys(base + soft_keep))[:MAX_POSITIONS]
        exposure, uncapped_exposure, rolling_vol = _target_exposure(prior_returns, previous_exposure)
        weights = {ticker: exposure / len(selected) for ticker in selected} if selected else {}
        period_return = 0.0
        for ticker, weight in weights.items():
            if ticker not in px.columns or pd.isna(px.loc[entry_date, ticker]) or pd.isna(px.loc[exit_date, ticker]):
                continue
            asset_return = float(px.loc[exit_date, ticker] / px.loc[entry_date, ticker] - 1.0)
            contribution = weight * asset_return
            period_return += contribution
            trade_rows.append(
                {
                    "window_start": start,
                    "date": decision_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "ticker": ticker,
                    "weight": weight,
                    "previous_weight": prior_weights.get(ticker, 0.0),
                    "raw_target_reconstructed": float(raw.get(ticker, np.nan)),
                    "raw_target_rank": int(raw.index.get_loc(ticker) + 1) if ticker in raw.index else np.nan,
                    "asset_return": asset_return,
                    "trade_contribution": contribution,
                    "soft_exit_status": "retained_positive_raw_target" if ticker in prior_selected else "new_or_reentered",
                    "signal_source": source,
                    "reconstruction_assumption": "not production parity; current target logic rerun on OHLCV truncated at decision date",
                }
            )
        turnover = 0.5 * sum(abs(weights.get(t, 0.0) - prior_weights.get(t, 0.0)) for t in set(weights) | set(prior_weights))
        daily_rows.append(
            {
                "window_start": start,
                "date": decision_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "return": period_return,
                "selected_tickers": ",".join(selected),
                "selected_count": len(selected),
                "target_exposure": exposure,
                "uncapped_exposure": uncapped_exposure,
                "cash_weight": 1.0 - exposure,
                "turnover": turnover,
                "rolling_vol_used": rolling_vol,
                "raw_target_source": source,
                "reconstruction_type": "causal_ohlcv_reconstruction_not_exact_replay",
            }
        )
        prior_returns.append(period_return)
        previous_exposure = exposure
        prior_weights = weights
        prior_selected = set(selected)
    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def _benchmark_returns(prices: pd.DataFrame, start: str, reference_dates: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if ticker not in prices.columns or reference_dates.empty:
        return pd.DataFrame()
    rows = []
    px = prices[ticker].dropna()
    for _, row in reference_dates.iterrows():
        entry = pd.Timestamp(row["entry_date"])
        exit_ = pd.Timestamp(row["exit_date"])
        if entry in px.index and exit_ in px.index:
            rows.append({"date": row["date"], "return": float(px.loc[exit_] / px.loc[entry] - 1.0)})
    return pd.DataFrame(rows)


def _annual_monthly(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = _dates(daily).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    annual = df.groupby(["window_start", "year"])["return"].apply(lambda x: (1.0 + x).prod() - 1.0).reset_index(name="annual_return")
    monthly = df.groupby(["window_start", "month"])["return"].apply(lambda x: (1.0 + x).prod() - 1.0).reset_index(name="monthly_return")
    return annual, monthly


def _stress(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, group in daily.groupby("window_start"):
        group = _dates(group)
        for name, (start, end) in STRESS_PERIODS.items():
            sub = group[group["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            metrics = _metrics(f"{window}_{name}", sub)
            rows.append({"window_start": window, "stress_period": name, **metrics})
    return pd.DataFrame(rows)


def _parity_check(reconstructed: pd.DataFrame) -> pd.DataFrame:
    parity = _dates(_read_csv(PRODUCTION_PARITY_FILE))
    if parity.empty or reconstructed.empty:
        return pd.DataFrame([{"status": "missing_production_parity_or_reconstructed_data"}])
    recon = reconstructed[reconstructed["window_start"].astype(str).eq("2022-01-03")].copy()
    if recon.empty:
        return pd.DataFrame([{"status": "missing_2022_reconstructed_window"}])
    parity_col = "return" if "return" in parity.columns else "portfolio_return"
    left = recon[["date", "return"]].rename(columns={"return": "reconstructed_return"})
    right = parity[["date", parity_col]].rename(columns={parity_col: "production_parity_return"})
    merged = left.merge(right, on="date", how="inner")
    if merged.empty:
        return pd.DataFrame([{"status": "no_overlapping_dates"}])
    corr = merged["reconstructed_return"].corr(merged["production_parity_return"])
    mae = (merged["reconstructed_return"] - merged["production_parity_return"]).abs().mean()
    recon_metrics = _metrics("reconstructed_2022", recon)
    parity_metrics = _metrics("production_parity_2022", parity.rename(columns={parity_col: "return"}))
    return pd.DataFrame(
        [
            {
                "status": "compared",
                "overlap_observations": len(merged),
                "return_correlation": corr,
                "MAE": mae,
                "reconstructed_CAGR": recon_metrics.get("CAGR"),
                "production_parity_CAGR": parity_metrics.get("CAGR"),
                "reconstructed_Sharpe": recon_metrics.get("Sharpe"),
                "production_parity_Sharpe": parity_metrics.get("Sharpe"),
                "reconstructed_max_drawdown": recon_metrics.get("max_drawdown"),
                "production_parity_max_drawdown": parity_metrics.get("max_drawdown"),
                "mismatch_explanation": "Expected: reconstructed target uses current OHLCV target logic, production-parity uses stored generated targets/snapshots.",
            }
        ]
    )


def run(universe: str = "normal", sleep_seconds: float = 0.5, retries: int = 1) -> None:
    coverage = _read_csv("yahoo_historical_ohlcv_coverage.csv")
    if coverage.empty:
        raise ValueError("Run yahoo_historical_reconstruction_feasibility.py first.")
    if "universe" in coverage.columns:
        coverage = coverage[coverage["universe"].astype(str).eq(universe)]
    all_needed = sorted(set(coverage.loc[coverage["download_status"].astype(str).eq("ok"), "ticker"].astype(str)) | {"SPY", "QQQ"})
    prices = _load_or_download_prices(all_needed, sleep_seconds=sleep_seconds, retries=retries)
    if prices.empty:
        governance = pd.DataFrame([{
            "classification": "reconstruction_not_reliable",
            "universe": universe,
            "production_changed": False,
            "parameter_tuning": False,
            "ce_dear_filtering": universe == "cedear",
            "exact_production_parity": False,
            "assumption": "Yahoo OHLCV required but download/cache returned no usable prices",
            "reason": "No OHLCV price panel available; reconstructed backtest not run.",
        }])
        governance.to_csv(OUT_GOVERNANCE, index=False)
        pd.DataFrame().to_csv(OUT_RESULTS, index=False)
        pd.DataFrame().to_csv(OUT_DAILY, index=False)
        pd.DataFrame().to_csv(OUT_TRADES, index=False)
        pd.DataFrame().to_csv(OUT_STRESS, index=False)
        pd.DataFrame([{"status": "not_run_no_prices"}]).to_csv(OUT_PARITY, index=False)
        print("\n===== RECONSTRUCTED LONG-HORIZON GROWTH BACKTEST =====")
        print("No OHLCV price panel available; reconstructed backtest not run.")
        print("\n===== GOVERNANCE =====")
        print(governance.to_string(index=False))
        return
    daily_all = []
    trades_all = []
    result_rows = []
    benchmark_rows = []
    for start in WINDOWS:
        tickers = _tickers_for_window(start, universe)
        if not tickers:
            continue
        daily, trades = _run_window(start, prices, tickers)
        if daily.empty:
            continue
        daily_all.append(daily)
        trades_all.append(trades)
        result_rows.append({**_metrics(f"reconstructed_growth_{start}", daily), "window_start": start, "ticker_coverage": len(tickers), "average_exposure": float(daily["target_exposure"].mean()), "average_cash": float(daily["cash_weight"].mean()), "average_turnover": float(daily["turnover"].mean()), "average_selected_count": float(daily["selected_count"].mean())})
        for bench in ["SPY", "QQQ"]:
            bdf = _benchmark_returns(prices, start, daily, bench)
            if not bdf.empty:
                benchmark_rows.append({**_metrics(f"{bench}_{start}", bdf), "window_start": start, "benchmark": bench})
    daily_out = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()
    trades_out = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    results = pd.DataFrame(result_rows)
    benchmarks = pd.DataFrame(benchmark_rows)
    daily_out.to_csv(OUT_DAILY, index=False)
    trades_out.to_csv(OUT_TRADES, index=False)
    results.to_csv(OUT_RESULTS, index=False)
    _stress(daily_out).to_csv(OUT_STRESS, index=False)
    parity = _parity_check(daily_out)
    parity.to_csv(OUT_PARITY, index=False)
    annual, monthly = _annual_monthly(daily_out)
    if not annual.empty:
        annual.to_csv("reconstructed_growth_annual_returns.csv", index=False)
    if not monthly.empty:
        monthly.to_csv("reconstructed_growth_monthly_returns.csv", index=False)
    if not benchmarks.empty:
        benchmarks.to_csv("reconstructed_growth_benchmark_comparison.csv", index=False)
    governance_class = "research_only"
    reason = "Reconstructed OHLCV stress test; not exact production parity."
    if not results.empty and (results["Sharpe"].dropna() > 0).any():
        governance_class = "passed_long_horizon_stress"
    if not parity.empty and parity.iloc[0].get("status") == "compared":
        corr = pd.to_numeric(parity["return_correlation"], errors="coerce").iloc[0]
        if pd.notna(corr) and corr < 0.3:
            governance_class = "research_only"
            reason = "2022+ reconstruction differs materially from production-parity; use as stress test only."
    governance = pd.DataFrame(
        [
            {
                "classification": governance_class,
                "universe": universe,
                "production_changed": False,
                "parameter_tuning": False,
                "ce_dear_filtering": False,
                "exact_production_parity": False,
                "assumption": "raw target reconstructed by rerunning current advanced target logic on OHLCV truncated at each decision date",
                "reason": reason,
            }
        ]
    )
    governance.to_csv(OUT_GOVERNANCE, index=False)
    print("\n===== RECONSTRUCTED LONG-HORIZON GROWTH BACKTEST =====")
    print(results.to_string(index=False))
    print("\n===== RECONSTRUCTED VS PRODUCTION-PARITY CHECK =====")
    print(parity.to_string(index=False))
    print("\n===== BENCHMARK COMPARISON =====")
    print(benchmarks.head(20).to_string(index=False) if not benchmarks.empty else "benchmark unavailable")
    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstructed long-horizon Growth Champion v2 backtest.")
    parser.add_argument("--universe", choices=["normal", "cedear"], default="normal")
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(universe=args.universe, sleep_seconds=args.sleep_seconds, retries=args.retries)
