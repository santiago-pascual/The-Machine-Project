from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = "2020-01-01"
RAW_RESULTS_FILE = "raw_target_2020_results.csv"
RAW_DAILY_FILE = "raw_target_2020_daily_returns.csv"
YEAR_FILE = "raw_target_2020_year_analysis.csv"
STRESS_FILE = "raw_target_2020_stress_test.csv"
BENCHMARK_FILE = "raw_target_2020_vs_benchmark.csv"
RAW_TRADES_FILE = "raw_target_research_backtest_trades.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"

OUT_GAP = "spy_benchmark_gap_analysis.csv"
OUT_YEARLY = "spy_yearly_gap.csv"
OUT_MISSED = "spy_missed_winners.csv"
OUT_SIZING = "spy_position_sizing_gap.csv"
OUT_DRAWDOWN = "spy_drawdown_comparison.csv"
OUT_MAP = "spy_improvement_map.csv"

TRADING_DAYS = 252

REGIME_PERIODS = {
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
    median_days = np.median(dates.diff().dt.days.dropna())
    if not np.isfinite(median_days) or median_days <= 0:
        return 52.0
    return float(365.25 / median_days)


def _max_drawdown(returns: pd.Series) -> float:
    r = _num(returns).dropna()
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _drawdown_frame(returns: pd.Series, dates: pd.Series, label: str) -> pd.DataFrame:
    r = _num(returns)
    dates = pd.to_datetime(dates, errors="coerce")
    data = pd.DataFrame({"date": dates, "return": r}).dropna()
    if data.empty:
        return pd.DataFrame()
    equity = (1.0 + data["return"]).cumprod()
    dd = equity / equity.cummax() - 1.0
    return pd.DataFrame({"date": data["date"], "model": label, "equity": equity, "drawdown": dd})


def _sortino(returns: pd.Series, ppy: float) -> float:
    r = _num(returns).dropna()
    if r.empty:
        return np.nan
    downside = r[r < 0].std(ddof=0)
    if not np.isfinite(downside) or downside <= 0:
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
        "actual_start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "observations": len(r),
        "total_return": total,
        "CAGR": float((1.0 + total) ** (1.0 / years) - 1.0),
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": _sortino(r, ppy),
        "Calmar": np.nan if not np.isfinite(mdd) or mdd >= 0 else float(((1.0 + total) ** (1.0 / years) - 1.0) / abs(mdd)),
        "max_drawdown": mdd,
        "hit_rate": float((r > 0).mean()),
        "average_winner": float(r[r > 0].mean()) if (r > 0).any() else np.nan,
        "average_loser": float(r[r < 0].mean()) if (r < 0).any() else np.nan,
    }


def _raw_daily() -> pd.DataFrame:
    df = _dates(_read_csv(RAW_DAILY_FILE))
    if df.empty:
        return df
    if "return" not in df.columns and "portfolio_return" in df.columns:
        df["return"] = _num(df["portfolio_return"])
    if "cash" not in df.columns and "cash_proxy" in df.columns:
        df["cash"] = _num(df["cash_proxy"])
    if "exposure" not in df.columns and "cash" in df.columns:
        df["exposure"] = 1.0 - _num(df["cash"])
    return df.sort_values("date")


def _raw_trades() -> pd.DataFrame:
    df = _dates(_read_csv(RAW_TRADES_FILE))
    if df.empty:
        return df
    if "ablation_variant" in df.columns:
        df = df[df["ablation_variant"].eq("raw_target_return_only")].copy()
    elif "variant" in df.columns:
        df = df[df["variant"].eq("raw_target_return_only")].copy()
    df = df.drop_duplicates(subset=["date", "ticker"])
    if "trade_return" not in df.columns:
        for col in ("realized_return_20d", "realized_return_10d", "realized_return_5d"):
            if col in df.columns:
                df["trade_return"] = _num(df[col])
                break
    if "raw_weight" not in df.columns:
        weight_col = "ablation_weight" if "ablation_weight" in df.columns else "weight"
        df["raw_weight"] = _num(df.get(weight_col, pd.Series(index=df.index, dtype=float))).fillna(0.0)
    return df


def _historical_mode_daily(mode: str) -> pd.DataFrame:
    pf = _dates(_read_csv("historical_walk_forward_portfolio_returns.csv"))
    if pf.empty or "model_mode" not in pf.columns:
        return pd.DataFrame()
    df = pf[pf["model_mode"].eq(mode)].copy()
    candidates = ["realized_portfolio_return_5d", "return", "portfolio_return", "realized_portfolio_return_1d"]
    for col in candidates:
        if col in df.columns:
            df["return"] = _num(df[col])
            break
    return df.sort_values("date")


def _benchmark_daily(ticker: str) -> pd.DataFrame:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty or "current_price" not in snaps.columns:
        return pd.DataFrame()
    df = snaps[snaps["ticker"].eq(ticker)].copy()
    if "model_mode" in df.columns:
        baseline = df[df["model_mode"].eq("baseline")]
        if not baseline.empty:
            df = baseline
    df = df.drop_duplicates("date").sort_values("date")
    df["return"] = _num(df["current_price"]).pct_change()
    return df.dropna(subset=["return"])


def _yearly_gap(raw: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw = _dates(raw).copy()
    spy = _dates(spy).copy()
    if raw.empty or spy.empty:
        return pd.DataFrame()
    raw["year"] = raw["date"].dt.year
    spy["year"] = spy["date"].dt.year
    years = sorted(set(raw["year"]).intersection(set(spy["year"])))
    for year in years:
        rd = raw[raw["year"].eq(year)]
        sd = spy[spy["year"].eq(year)]
        rm = _metrics("raw", rd)
        sm = _metrics("SPY", sd)
        rows.append(
            {
                "year": year,
                "raw_return": rm.get("total_return"),
                "SPY_return": sm.get("total_return"),
                "raw_minus_SPY": rm.get("total_return") - sm.get("total_return"),
                "raw_Sharpe": rm.get("Sharpe"),
                "SPY_Sharpe": sm.get("Sharpe"),
                "raw_max_drawdown": rm.get("max_drawdown"),
                "SPY_max_drawdown": sm.get("max_drawdown"),
                "raw_beats_SPY": bool(rm.get("total_return", -np.inf) > sm.get("total_return", np.inf)),
            }
        )
    return pd.DataFrame(rows)


def _stress_gap(raw: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (start, end) in REGIME_PERIODS.items():
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
        rd = raw[(raw["date"] >= start_dt) & (raw["date"] <= end_dt)]
        sd = spy[(spy["date"] >= start_dt) & (spy["date"] <= end_dt)]
        rm = _metrics("raw_target_research", rd)
        sm = _metrics("SPY_buy_hold", sd)
        rows.append(
            {
                "regime_period": name,
                "raw_return": rm.get("total_return"),
                "SPY_return": sm.get("total_return"),
                "raw_minus_SPY": rm.get("total_return", np.nan) - sm.get("total_return", np.nan),
                "raw_Sharpe": rm.get("Sharpe"),
                "SPY_Sharpe": sm.get("Sharpe"),
                "raw_drawdown": rm.get("max_drawdown"),
                "SPY_drawdown": sm.get("max_drawdown"),
                "drawdown_protection": sm.get("max_drawdown", np.nan) - rm.get("max_drawdown", np.nan),
                "observations": rm.get("observations"),
            }
        )
    return pd.DataFrame(rows)


def _performance_gap(raw_metrics: dict, spy_metrics: dict, raw_daily: pd.DataFrame, spy_daily: pd.DataFrame) -> pd.DataFrame:
    raw_cash = float(_num(raw_daily.get("cash", pd.Series(dtype=float))).mean()) if "cash" in raw_daily.columns else np.nan
    spy_aligned = spy_daily[spy_daily["date"].isin(set(raw_daily["date"]))].copy()
    missed_upside = np.nan
    if not spy_aligned.empty:
        merged = raw_daily[["date", "return"]].merge(spy_aligned[["date", "return"]], on="date", suffixes=("_raw", "_SPY"))
        up = merged[merged["return_SPY"] > 0]
        if not up.empty:
            missed_upside = float((up["return_SPY"] - up["return_raw"]).mean())
    downside_benefit = np.nan
    if not spy_aligned.empty:
        merged = raw_daily[["date", "return"]].merge(spy_aligned[["date", "return"]], on="date", suffixes=("_raw", "_SPY"))
        down = merged[merged["return_SPY"] < 0]
        if not down.empty:
            downside_benefit = float((down["return_raw"] - down["return_SPY"]).mean())
    return pd.DataFrame(
        [
            {
                "comparison": "raw_target_research_vs_SPY",
                "return_gap": raw_metrics.get("total_return", np.nan) - spy_metrics.get("total_return", np.nan),
                "CAGR_gap": raw_metrics.get("CAGR", np.nan) - spy_metrics.get("CAGR", np.nan),
                "Sharpe_gap": raw_metrics.get("Sharpe", np.nan) - spy_metrics.get("Sharpe", np.nan),
                "max_drawdown_gap": raw_metrics.get("max_drawdown", np.nan) - spy_metrics.get("max_drawdown", np.nan),
                "volatility_gap": raw_metrics.get("volatility", np.nan) - spy_metrics.get("volatility", np.nan),
                "cash_drag_proxy": raw_cash,
                "missed_upside_proxy": missed_upside,
                "downside_avoidance_benefit": downside_benefit,
            }
        ]
    )


def _selected_map(raw_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "selected_tickers" not in raw_daily.columns:
        return pd.DataFrame()
    for _, row in raw_daily.iterrows():
        tickers = str(row.get("selected_tickers", "")).split(",")
        tickers = [t.strip() for t in tickers if t.strip() and t.strip().lower() != "nan"]
        n = len(tickers)
        for ticker in tickers:
            rows.append({"date": row["date"], "ticker": ticker, "selected_by_raw": True, "raw_equal_weight_proxy": 1.0 / n if n else 0.0})
    return pd.DataFrame(rows)


def _missed_winners(raw_daily: pd.DataFrame) -> pd.DataFrame:
    realized = _dates(_read_csv(REALIZED_FILE))
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if realized.empty or snaps.empty:
        return pd.DataFrame()
    if "model_mode" in realized.columns:
        realized = realized[realized["model_mode"].eq("baseline")].copy()
    if "model_mode" in snaps.columns:
        snaps = snaps[snaps["model_mode"].eq("baseline")].copy()
    selected = _selected_map(raw_daily)
    realized["future_return_20d"] = _num(realized.get("realized_return_20d", pd.Series(index=realized.index, dtype=float)))
    winners = realized.sort_values(["date", "future_return_20d"], ascending=[True, False]).groupby("date").head(5)
    cols = ["date", "ticker", "selected", "weight", "expected_daily_return", "signal_strength", "target_confidence", "quality_score"]
    available = [c for c in cols if c in snaps.columns]
    out = winners.merge(snaps[available].drop_duplicates(["date", "ticker"]), on=["date", "ticker"], how="left")
    if not selected.empty:
        out = out.merge(selected, on=["date", "ticker"], how="left")
    else:
        out["selected_by_raw"] = False
        out["raw_equal_weight_proxy"] = np.nan
    out["selected_by_raw"] = out["selected_by_raw"].fillna(False)
    out["raw_weight_proxy"] = out["raw_equal_weight_proxy"].fillna(0.0)
    out["miss_reason"] = np.where(
        out["selected_by_raw"],
        "selected",
        np.where(
            _num(out.get("expected_daily_return", pd.Series(index=out.index))).fillna(-np.inf) <= 0,
            "low_or_negative_expected_return",
            "not_in_raw_selection",
        ),
    )
    return out.rename(columns={"future_return_20d": "winner_forward_return_20d"})


def _position_sizing(raw_daily: pd.DataFrame, raw_trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if raw_daily.empty:
        return pd.DataFrame()
    rows.append(
        {
            "metric": "average_cash",
            "value": float(_num(raw_daily.get("cash", pd.Series(dtype=float))).mean()) if "cash" in raw_daily.columns else np.nan,
            "interpretation": "cash reduces upside capture when benchmark rises",
        }
    )
    rows.append(
        {
            "metric": "average_exposure",
            "value": float(_num(raw_daily.get("exposure", pd.Series(dtype=float))).mean()) if "exposure" in raw_daily.columns else np.nan,
            "interpretation": "growth model is mostly invested",
        }
    )
    rows.append(
        {
            "metric": "average_turnover",
            "value": float(_num(raw_daily.get("turnover", pd.Series(dtype=float))).mean()) if "turnover" in raw_daily.columns else np.nan,
            "interpretation": "high turnover can cut winners early",
        }
    )
    rows.append(
        {
            "metric": "average_selected_count",
            "value": float(_num(raw_daily.get("selected_count", pd.Series(dtype=float))).mean())
            if "selected_count" in raw_daily.columns
            else np.nan,
            "interpretation": "few positions increases idiosyncratic gap versus SPY",
        }
    )
    if not raw_trades.empty and "raw_weight" in raw_trades.columns:
        rows.append(
            {
                "metric": "max_single_position_proxy",
                "value": float(_num(raw_trades["raw_weight"]).max()),
                "interpretation": "cap can underweight major winners if alpha rank is correct",
            }
        )
    return pd.DataFrame(rows)


def _drawdown_comparison(raw_daily: pd.DataFrame, spy_daily: pd.DataFrame) -> pd.DataFrame:
    raw_dd = _drawdown_frame(raw_daily["return"], raw_daily["date"], "raw_target_research")
    spy_dd = _drawdown_frame(spy_daily["return"], spy_daily["date"], "SPY_buy_hold")
    if raw_dd.empty or spy_dd.empty:
        return pd.DataFrame()
    merged = raw_dd[["date", "drawdown"]].merge(spy_dd[["date", "drawdown"]], on="date", suffixes=("_raw", "_SPY"))
    merged["raw_protection_benefit"] = merged["drawdown_raw"] - merged["drawdown_SPY"]
    merged["raw_drawdown_worse"] = merged["drawdown_raw"] < merged["drawdown_SPY"]
    return merged


def _improvement_map(
    gap: pd.DataFrame, yearly: pd.DataFrame, missed: pd.DataFrame, sizing: pd.DataFrame, drawdown: pd.DataFrame
) -> pd.DataFrame:
    gap_row = gap.iloc[0] if not gap.empty else pd.Series(dtype=object)
    missed_mask = pd.Series(dtype=bool)
    if not missed.empty and "selected_by_raw" in missed.columns:
        missed_mask = ~missed["selected_by_raw"].fillna(False).astype(bool)
    missed_count = int(missed_mask.sum()) if not missed_mask.empty else 0
    missed_avg = (
        float(_num(missed.loc[missed_mask, "winner_forward_return_20d"]).mean())
        if not missed_mask.empty and "winner_forward_return_20d" in missed.columns
        else np.nan
    )
    avg_cash = (
        float(sizing.loc[sizing["metric"].eq("average_cash"), "value"].iloc[0])
        if not sizing.empty and sizing["metric"].eq("average_cash").any()
        else np.nan
    )
    avg_turnover = (
        float(sizing.loc[sizing["metric"].eq("average_turnover"), "value"].iloc[0])
        if not sizing.empty and sizing["metric"].eq("average_turnover").any()
        else np.nan
    )
    dd_benefit = float(drawdown["raw_protection_benefit"].min()) if not drawdown.empty else np.nan
    rows = [
        {
            "gap_type": "return_gap",
            "severity": "high" if gap_row.get("return_gap", 0) < 0 else "none",
            "diagnosis": "raw target trails SPY total return" if gap_row.get("return_gap", 0) < 0 else "raw target beats SPY total return",
            "evidence": gap_row.get("return_gap"),
            "do_not_change_yet": True,
        },
        {
            "gap_type": "missed_winners",
            "severity": "high" if missed_count > 0 and missed_avg > 0.05 else "medium",
            "diagnosis": "top forward winners often not selected or underweighted",
            "evidence": f"missed_count={missed_count}, missed_avg_20d={missed_avg}",
            "do_not_change_yet": True,
        },
        {
            "gap_type": "cash_drag",
            "severity": "low" if np.isfinite(avg_cash) and avg_cash < 0.15 else "medium",
            "diagnosis": "cash exists but is not the primary raw target gap",
            "evidence": avg_cash,
            "do_not_change_yet": True,
        },
        {
            "gap_type": "turnover",
            "severity": "medium" if np.isfinite(avg_turnover) and avg_turnover > 0.35 else "low",
            "diagnosis": "turnover may cut winners before full trend capture",
            "evidence": avg_turnover,
            "do_not_change_yet": True,
        },
        {
            "gap_type": "downside_protection",
            "severity": "medium" if np.isfinite(dd_benefit) and dd_benefit < 0 else "supportive",
            "diagnosis": "raw protects drawdown better than SPY on full sample"
            if gap_row.get("max_drawdown_gap", 0) > 0
            else "raw drawdown protection insufficient",
            "evidence": gap_row.get("max_drawdown_gap"),
            "do_not_change_yet": True,
        },
    ]
    if not yearly.empty:
        worst = yearly.sort_values("raw_minus_SPY").head(1)
        best = yearly.sort_values("raw_minus_SPY", ascending=False).head(1)
        rows.append(
            {
                "gap_type": "worst_relative_year",
                "severity": "diagnostic",
                "diagnosis": str(worst.to_dict("records")),
                "evidence": None,
                "do_not_change_yet": True,
            }
        )
        rows.append(
            {
                "gap_type": "best_relative_year",
                "severity": "diagnostic",
                "diagnosis": str(best.to_dict("records")),
                "evidence": None,
                "do_not_change_yet": True,
            }
        )
    return pd.DataFrame(rows)


def run_spy_benchmark_gap_analysis() -> dict[str, pd.DataFrame]:
    results = _read_csv(RAW_RESULTS_FILE)
    raw_daily = _raw_daily()
    raw_trades = _raw_trades()
    spy_daily = _benchmark_daily("SPY")
    qqq_daily = _benchmark_daily("QQQ")
    if raw_daily.empty or spy_daily.empty:
        raise ValueError("raw target daily returns and SPY benchmark data are required.")

    raw_metrics = _metrics("raw_target_research", raw_daily)
    spy_metrics = _metrics("SPY_buy_hold", spy_daily)
    qqq_metrics = _metrics("QQQ_buy_hold", qqq_daily)
    baseline_metrics = _metrics("baseline", _historical_mode_daily("baseline"))
    regime_metrics = _metrics("regime_gated_full_quant", _historical_mode_daily("regime_gated_full_quant"))

    comparison = pd.DataFrame([baseline_metrics, regime_metrics, raw_metrics, spy_metrics, qqq_metrics])
    gap = _performance_gap(raw_metrics, spy_metrics, raw_daily, spy_daily)
    yearly = _yearly_gap(raw_daily, spy_daily)
    stress = _stress_gap(raw_daily, spy_daily)
    missed = _missed_winners(raw_daily)
    sizing = _position_sizing(raw_daily, raw_trades)
    drawdown = _drawdown_comparison(raw_daily, spy_daily)
    improvement = _improvement_map(gap, yearly, missed, sizing, drawdown)

    gap.to_csv(OUT_GAP, index=False)
    yearly.to_csv(OUT_YEARLY, index=False)
    missed.to_csv(OUT_MISSED, index=False)
    sizing.to_csv(OUT_SIZING, index=False)
    drawdown.to_csv(OUT_DRAWDOWN, index=False)
    improvement.to_csv(OUT_MAP, index=False)

    print("\n===== SPY BENCHMARK GAP ANALYSIS =====")
    print(comparison.to_string(index=False))
    print("\n===== RAW TARGET VS SPY GAP =====")
    print(gap.to_string(index=False))
    actual_start = raw_metrics.get("actual_start_date")
    if actual_start and str(actual_start) > START_DATE:
        print(f"\n[WARNING] requested start {START_DATE}, but available benchmark/gap data starts {actual_start}.")

    print("\n===== RAW TARGET VS SPY YEARLY GAP =====")
    print(yearly.to_string(index=False))

    print("\n===== MISSED WINNERS ANALYSIS =====")
    missed_cols = [
        "date",
        "ticker",
        "winner_forward_return_20d",
        "selected_by_raw",
        "raw_weight_proxy",
        "expected_daily_return",
        "signal_strength",
        "miss_reason",
    ]
    print(missed[[c for c in missed_cols if c in missed.columns]].head(25).to_string(index=False))

    print("\n===== POSITION SIZING GAP =====")
    print(sizing.to_string(index=False))

    print("\n===== DRAWDOWN COMPARISON =====")
    draw_cols = ["date", "drawdown_raw", "drawdown_SPY", "raw_protection_benefit", "raw_drawdown_worse"]
    print(drawdown[[c for c in draw_cols if c in drawdown.columns]].sort_values("raw_protection_benefit").head(15).to_string(index=False))

    print("\n===== IMPROVEMENT MAP =====")
    print(improvement.to_string(index=False))

    print(f"\nSaved: {Path(OUT_GAP).resolve()}")
    print(f"Saved: {Path(OUT_YEARLY).resolve()}")
    print(f"Saved: {Path(OUT_MISSED).resolve()}")
    print(f"Saved: {Path(OUT_SIZING).resolve()}")
    print(f"Saved: {Path(OUT_DRAWDOWN).resolve()}")
    print(f"Saved: {Path(OUT_MAP).resolve()}")

    return {
        "comparison": comparison,
        "gap": gap,
        "yearly": yearly,
        "stress": stress,
        "missed": missed,
        "sizing": sizing,
        "drawdown": drawdown,
        "improvement": improvement,
    }


if __name__ == "__main__":
    run_spy_benchmark_gap_analysis()
