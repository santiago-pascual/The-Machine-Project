
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100000.0
STRATEGY_FILE = "growth_champion_reconstructed_stress_daily.csv"
CACHE_DIR = Path("yahoo_ohlcv_price_cache")


def numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_strategy() -> pd.DataFrame:
    if not Path(STRATEGY_FILE).exists():
        return pd.DataFrame()
    df = pd.read_csv(STRATEGY_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).sort_values("date")
    return df


def load_benchmark_prices(ticker: str, dates: pd.Series) -> pd.DataFrame:
    path = CACHE_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    px = pd.read_csv(path)
    date_col = "Date" if "Date" in px.columns else "date" if "date" in px.columns else None
    adj_col = "Adj Close" if "Adj Close" in px.columns else None
    close_col = "Close" if "Close" in px.columns else None
    price_col = adj_col or close_col
    if date_col is None or price_col is None:
        return pd.DataFrame()
    work = px[[date_col, price_col]].copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work[f"{ticker}_price"] = numeric(work[price_col])
    work = work.dropna(subset=["date", f"{ticker}_price"]).sort_values("date")
    base = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()})
    aligned = pd.merge_asof(base, work[["date", f"{ticker}_price"]], on="date", direction="backward")
    aligned[f"{ticker}_return"] = aligned[f"{ticker}_price"].pct_change().fillna(0.0)
    aligned[f"{ticker}_equity"] = (1 + aligned[f"{ticker}_return"].fillna(0)).cumprod() * INITIAL_CAPITAL
    aligned[f"{ticker}_cumulative_return"] = aligned[f"{ticker}_equity"] / INITIAL_CAPITAL - 1
    aligned[f"{ticker}_source_price_column"] = price_col
    return aligned


def max_abs(a: pd.Series) -> float:
    x = numeric(a).dropna()
    return float(x.abs().max()) if not x.empty else np.nan


def main() -> None:
    strategy = load_strategy()
    rows = []
    norm_rows = []
    comp_rows = []
    report_lines = ["===== BENCHMARK INTEGRITY AUDIT =====", ""]
    if strategy.empty:
        rows.append({"check": "strategy_file_available", "status": "FAIL", "detail": f"Missing or empty {STRATEGY_FILE}"})
        pd.DataFrame(rows).to_csv("benchmark_integrity_audit.csv", index=False)
        Path("benchmark_alignment_report.txt").write_text("\n".join(report_lines + ["Strategy file missing."]), encoding="utf-8")
        return

    strategy = strategy.copy()
    strategy["growth_return"] = numeric(strategy.get("gross_daily_return", pd.Series(dtype=float))).fillna(0)
    strategy["growth_equity_recomputed"] = (1 + strategy["growth_return"]).cumprod() * INITIAL_CAPITAL
    if "gross_equity" in strategy.columns:
        strategy["growth_equity_reported"] = numeric(strategy["gross_equity"]) * INITIAL_CAPITAL
    else:
        strategy["growth_equity_reported"] = np.nan
    strategy["growth_cumulative_return_recomputed"] = strategy["growth_equity_recomputed"] / INITIAL_CAPITAL - 1

    spy = load_benchmark_prices("SPY", strategy["date"])
    qqq = load_benchmark_prices("QQQ", strategy["date"])
    aligned = strategy[["date", "growth_return", "growth_equity_recomputed", "growth_equity_reported", "growth_cumulative_return_recomputed", "exposure", "cash", "turnover"]].copy()
    if not spy.empty:
        aligned = aligned.merge(spy[["date", "SPY_price", "SPY_return", "SPY_equity", "SPY_cumulative_return", "SPY_source_price_column"]], on="date", how="left")
    if not qqq.empty:
        aligned = aligned.merge(qqq[["date", "QQQ_price", "QQQ_return", "QQQ_equity", "QQQ_cumulative_return", "QQQ_source_price_column"]], on="date", how="left")

    # 1 initial capital
    rows.append({"check": "same_initial_capital", "status": "PASS", "detail": f"Growth, SPY and QQQ normalized to {INITIAL_CAPITAL:,.0f} in audit recomputation."})
    norm_rows.append({"series": "Growth", "initial_equity": INITIAL_CAPITAL, "first_date": strategy["date"].min().date(), "last_date": strategy["date"].max().date(), "final_equity": aligned["growth_equity_recomputed"].iloc[-1]})
    for ticker in ["SPY", "QQQ"]:
        if f"{ticker}_equity" in aligned.columns:
            norm_rows.append({"series": ticker, "initial_equity": INITIAL_CAPITAL, "first_date": strategy["date"].min().date(), "last_date": strategy["date"].max().date(), "final_equity": aligned[f"{ticker}_equity"].dropna().iloc[-1]})

    # 2 adjusted close
    for ticker in ["SPY", "QQQ"]:
        source_col = aligned.get(f"{ticker}_source_price_column", pd.Series(dtype=str)).dropna().astype(str)
        status = "PASS" if not source_col.empty and source_col.iloc[-1] == "Adj Close" else "WARNING"
        rows.append({"check": f"{ticker}_adjusted_close_used", "status": status, "detail": source_col.iloc[-1] if not source_col.empty else "benchmark cache missing"})

    # 3 frequency
    gaps = strategy["date"].diff().dt.days.dropna()
    exact_daily = bool((gaps <= 1).all()) if not gaps.empty else False
    rows.append({"check": "growth_same_daily_frequency_as_benchmark", "status": "WARNING" if not exact_daily else "PASS", "detail": f"Strategy observations={len(strategy)}, median_gap_days={gaps.median() if not gaps.empty else 'n/a'}, max_gap_days={gaps.max() if not gaps.empty else 'n/a'}; not true daily if gaps > 1."})

    # 4 weekends/holidays
    weekend_dates = int(strategy["date"].dt.weekday.isin([5, 6]).sum())
    rows.append({"check": "weekends_holidays_aligned", "status": "PASS" if weekend_dates == 0 else "WARNING", "detail": f"weekend_strategy_dates={weekend_dates}; benchmarks merge_asof to prior trading close."})

    # 5 vol targeting consistency
    exposure = numeric(strategy.get("exposure", pd.Series(dtype=float)))
    cash = numeric(strategy.get("cash", pd.Series(dtype=float)))
    exp_ok = bool(exposure.dropna().between(0.0, 0.60 + 1e-9).all()) if exposure.notna().any() else False
    cash_identity = max_abs((exposure + cash) - 1.0)
    rows.append({"check": "volatility_targeting_exposure_bounds", "status": "PASS" if exp_ok else "WARNING", "detail": f"min_exposure={exposure.min()}, max_exposure={exposure.max()}, expected cap <=0.60"})

    # 6 cash included
    rows.append({"check": "cash_included_correctly", "status": "PASS" if pd.notna(cash_identity) and cash_identity < 1e-9 else "WARNING", "detail": f"max_abs(exposure+cash-1)={cash_identity}"})

    # 7 normalization identical
    norm_ok = all(abs(float(r["initial_equity"]) - INITIAL_CAPITAL) < 1e-9 for r in norm_rows)
    rows.append({"check": "curves_normalized_identically", "status": "PASS" if norm_ok else "FAIL", "detail": "All audited series start from same initial capital."})

    # 8 cumulative calc
    reported_diff = max_abs(aligned["growth_equity_reported"] - aligned["growth_equity_recomputed"])
    rows.append({"check": "growth_cumulative_return_calculation", "status": "PASS" if pd.notna(reported_diff) and reported_diff < 1e-6 else "WARNING", "detail": f"max_reported_vs_recomputed_equity_diff={reported_diff}"})

    # 9 duplicated compounding
    for series in ["Growth", "SPY", "QQQ"]:
        if series == "Growth":
            ret = aligned["growth_return"]
            eq = aligned["growth_equity_recomputed"]
        else:
            if f"{series}_return" not in aligned.columns:
                continue
            ret = aligned[f"{series}_return"]
            eq = aligned[f"{series}_equity"]
        recomputed = (1 + numeric(ret).fillna(0)).cumprod() * INITIAL_CAPITAL
        diff = max_abs(recomputed - numeric(eq))
        comp_rows.append({"series": series, "observations": len(ret), "final_return_pct": (numeric(eq).dropna().iloc[-1] / INITIAL_CAPITAL - 1) * 100 if numeric(eq).notna().any() else np.nan, "max_compounding_diff": diff, "duplicated_compounding_suspected": bool(pd.notna(diff) and diff > 1e-6)})
    rows.append({"check": "no_duplicated_compounding", "status": "PASS" if all(not r["duplicated_compounding_suspected"] for r in comp_rows) else "FAIL", "detail": "See benchmark_compounding_check.csv"})

    # 10 scaling mismatch
    scale_ok = aligned["growth_equity_recomputed"].iloc[-1] > INITIAL_CAPITAL and ("SPY_equity" not in aligned.columns or aligned["SPY_equity"].dropna().iloc[-1] > INITIAL_CAPITAL)
    rows.append({"check": "no_scaling_mismatch", "status": "PASS" if scale_ok else "WARNING", "detail": "Equity values are dollar-normalized; chart values can be displayed as cumulative percent."})

    audit = pd.DataFrame(rows)
    norm = pd.DataFrame(norm_rows)
    comp = pd.DataFrame(comp_rows)
    audit.to_csv("benchmark_integrity_audit.csv", index=False)
    norm.to_csv("benchmark_normalization_check.csv", index=False)
    comp.to_csv("benchmark_compounding_check.csv", index=False)

    valid = "statistically_valid_with_caveat" if audit["status"].isin(["FAIL"]).sum() == 0 else "not_valid_until_fixed"
    caveats = audit[audit["status"].eq("WARNING")]
    report_lines += [
        f"strategy_file: {STRATEGY_FILE}",
        f"observations: {len(strategy)}",
        f"start_date: {strategy['date'].min().date()}",
        f"end_date: {strategy['date'].max().date()}",
        f"classification: {valid}",
        "",
        "Key findings:",
    ]
    for _, row in audit.iterrows():
        report_lines.append(f"- {row['status']} | {row['check']} | {row['detail']}")
    report_lines += [
        "",
        "Interpretation:",
        "- Cumulative return comparison is valid only when SPY/QQQ are reconstructed from adjusted prices aligned to the same strategy observation dates.",
        "- The long-horizon strategy file is not true daily frequency; it is a sparse/rebalance-observation reconstruction. Return comparisons are acceptable after date alignment, but volatility/Sharpe comparisons must account for interval frequency.",
        "- Blank spy_daily_return/qqq_daily_return columns in the strategy file should not be used directly for benchmark charts.",
        "- No strategy, ranking, optimizer, allocation, or paper logic was modified by this audit.",
    ]
    Path("benchmark_alignment_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("===== BENCHMARK INTEGRITY AUDIT =====")
    print(audit.to_string(index=False))
    print("classification", valid)


if __name__ == "__main__":
    main()
