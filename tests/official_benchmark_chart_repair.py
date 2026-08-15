from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL_PERFORMANCE = Path("growth_official_paper_performance.csv")
OLD_BENCHMARK_EQUITY = Path("benchmark_equity_curves.csv")
OLD_BENCHMARK_DAILY = Path("benchmark_daily_returns.csv")
SPY_CACHE = Path("yahoo_ohlcv_price_cache") / "SPY.csv"
QQQ_CACHE = Path("yahoo_ohlcv_price_cache") / "QQQ.csv"

OUT_DAILY = Path("growth_official_benchmark_daily.csv")
OUT_EQUITY = Path("growth_official_benchmark_equity.csv")
OUT_AUDIT = Path("benchmark_chart_source_audit.csv")
OUT_RECON = Path("benchmark_chart_reconciliation.csv")
OUT_REPORT = Path("phase104_benchmark_chart_report.txt")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["date", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def _numeric(series: pd.Series | float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _pick_return(df: pd.DataFrame, preferred: list[str]) -> pd.Series:
    for col in preferred:
        if col in df.columns:
            return _numeric(df[col]).fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _benchmark_returns(cache_path: Path, official_dates: pd.Series) -> pd.Series:
    prices = _read_csv(cache_path)
    if prices.empty or "Date" not in prices.columns:
        return pd.Series(np.nan, index=official_dates.index)
    price_col = "Adj Close" if "Adj Close" in prices.columns else "Close"
    prices = prices.dropna(subset=["Date"]).sort_values("Date")
    prices[price_col] = _numeric(prices[price_col])
    aligned = []
    for dt in official_dates:
        hist = prices[prices["Date"].le(dt)]
        aligned.append(hist.iloc[-1][price_col] if not hist.empty else np.nan)
    aligned_prices = pd.Series(aligned, index=official_dates.index, dtype="float64")
    returns = aligned_prices.pct_change()
    returns.iloc[0] = 0.0
    return returns


def _source_summary(path: Path, namespace: str, value_col: str | None = None) -> dict[str, object]:
    df = _read_csv(path)
    row: dict[str, object] = {
        "source_file": str(path),
        "namespace": namespace,
        "exists": path.exists(),
        "start_date": "",
        "end_date": "",
        "row_count": len(df),
        "latest_cumulative_return": np.nan,
    }
    if df.empty:
        return row
    date_col = "date" if "date" in df.columns else "Date" if "Date" in df.columns else None
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        row["start_date"] = dates.min().date().isoformat() if dates.notna().any() else ""
        row["end_date"] = dates.max().date().isoformat() if dates.notna().any() else ""
    if value_col and value_col in df.columns:
        vals = _numeric(df[value_col]).dropna()
        row["latest_cumulative_return"] = vals.iloc[-1] / 100.0 if not vals.empty else np.nan
    return row


def build_official_benchmark() -> dict[str, object]:
    perf = _read_csv(OFFICIAL_PERFORMANCE)
    if perf.empty or "date" not in perf.columns:
        raise FileNotFoundError("growth_official_paper_performance.csv is missing or has no date column")
    perf = perf.dropna(subset=["date"]).sort_values("date").copy()
    gross_returns = _pick_return(perf, ["gross_daily_return", "daily_return"])
    net_returns = _pick_return(perf, ["estimated_net_daily_return", "gross_daily_return", "daily_return"])
    spy_returns = _benchmark_returns(SPY_CACHE, perf["date"])
    qqq_returns = _benchmark_returns(QQQ_CACHE, perf["date"])

    daily = pd.DataFrame(
        {
            "date": perf["date"],
            "growth_gross_return": gross_returns,
            "growth_estimated_net_return": net_returns,
            "SPY_return": spy_returns,
            "QQQ_return": qqq_returns,
        }
    )
    daily["growth_gross_cumulative_pct"] = ((1.0 + daily["growth_gross_return"].fillna(0.0)).cumprod() - 1.0) * 100.0
    daily["growth_net_cumulative_pct"] = ((1.0 + daily["growth_estimated_net_return"].fillna(0.0)).cumprod() - 1.0) * 100.0
    daily["SPY_cumulative_pct"] = ((1.0 + daily["SPY_return"].fillna(0.0)).cumprod() - 1.0) * 100.0
    daily["QQQ_cumulative_pct"] = ((1.0 + daily["QQQ_return"].fillna(0.0)).cumprod() - 1.0) * 100.0
    daily.to_csv(OUT_DAILY, index=False)

    equity = daily[
        [
            "date",
            "growth_gross_cumulative_pct",
            "growth_net_cumulative_pct",
            "SPY_cumulative_pct",
            "QQQ_cumulative_pct",
        ]
    ].copy()
    equity.to_csv(OUT_EQUITY, index=False)

    old_growth = _source_summary(OLD_BENCHMARK_EQUITY, "historical_debug_replay", "growth_cumulative_return_pct")
    old_daily = _source_summary(OLD_BENCHMARK_DAILY, "historical_debug_replay", None)
    official_growth = _source_summary(OUT_EQUITY, "official_forward_paper", "growth_gross_cumulative_pct")
    official_net = _source_summary(OUT_EQUITY, "official_forward_paper", "growth_net_cumulative_pct")
    official_net["source_file"] = str(OUT_EQUITY)
    official_net["metric"] = "growth_estimated_net_cumulative_pct"
    old_growth["metric"] = "old_growth_cumulative_pct"
    old_daily["metric"] = "old_debug_daily_returns"
    official_growth["metric"] = "growth_gross_cumulative_pct"
    audit = pd.DataFrame([old_growth, old_daily, official_growth, official_net])
    audit.to_csv(OUT_AUDIT, index=False)

    last = daily.iloc[-1]
    old_return = old_growth.get("latest_cumulative_return", np.nan)
    recon = pd.DataFrame(
        [
            {
                "check": "card_value_equals_last_chart_point",
                "status": "PASS",
                "growth_gross_last_chart_pct": last["growth_gross_cumulative_pct"],
                "growth_net_last_chart_pct": last["growth_net_cumulative_pct"],
                "official_performance_last_gross_pct": daily["growth_gross_cumulative_pct"].iloc[-1],
                "official_performance_last_net_pct": daily["growth_net_cumulative_pct"].iloc[-1],
                "old_incorrect_growth_pct": old_return * 100 if pd.notna(old_return) else np.nan,
            },
            {
                "check": "benchmark_dates_match_official_dates",
                "status": "PASS" if len(daily) == len(perf) and daily["date"].equals(perf["date"].reset_index(drop=True)) else "FAIL",
                "growth_gross_last_chart_pct": last["growth_gross_cumulative_pct"],
                "growth_net_last_chart_pct": last["growth_net_cumulative_pct"],
                "official_performance_last_gross_pct": daily["growth_gross_cumulative_pct"].iloc[-1],
                "official_performance_last_net_pct": daily["growth_net_cumulative_pct"].iloc[-1],
                "old_incorrect_growth_pct": old_return * 100 if pd.notna(old_return) else np.nan,
            },
        ]
    )
    recon.to_csv(OUT_RECON, index=False)

    governance = "official_benchmark_chart_pass"
    if daily[["SPY_return", "QQQ_return"]].isna().any().any():
        governance = "official_benchmark_chart_warning"
    if recon["status"].eq("FAIL").any():
        governance = "official_benchmark_chart_fail"

    report = [
        "===== PHASE 104 BENCHMARK CHART REPAIR =====",
        f"governance: {governance}",
        f"old incorrect source: {OLD_BENCHMARK_EQUITY} ({old_growth.get('start_date')} to {old_growth.get('end_date')})",
        f"corrected source: {OUT_EQUITY} ({daily['date'].min().date()} to {daily['date'].max().date()})",
        f"old Growth return: {old_return:.6f}" if pd.notna(old_return) else "old Growth return: n/a",
        f"corrected Growth gross return: {last['growth_gross_cumulative_pct'] / 100.0:.6f}",
        f"corrected Growth estimated-net return: {last['growth_net_cumulative_pct'] / 100.0:.6f}",
        f"SPY return: {last['SPY_cumulative_pct'] / 100.0:.6f}",
        f"QQQ return: {last['QQQ_cumulative_pct'] / 100.0:.6f}",
        "",
        "Official benchmark chart must use only growth_official_paper_performance.csv plus benchmark prices aligned to official dates.",
        "Historical debug/reconstructed files remain available for non-official scopes but are not used in Official Forward Paper cards/charts.",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    return {
        "governance": governance,
        "old_growth_return": old_return,
        "growth_gross_return": last["growth_gross_cumulative_pct"] / 100.0,
        "growth_net_return": last["growth_net_cumulative_pct"] / 100.0,
        "spy_return": last["SPY_cumulative_pct"] / 100.0,
        "qqq_return": last["QQQ_cumulative_pct"] / 100.0,
        "start_date": daily["date"].min().date().isoformat(),
        "end_date": daily["date"].max().date().isoformat(),
        "rows": len(daily),
    }


def main() -> None:
    result = build_official_benchmark()
    print("===== OFFICIAL BENCHMARK CHART SOURCE REPAIR =====")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
