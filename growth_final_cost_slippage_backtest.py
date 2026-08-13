from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DAILY_FILE = "growth_final_selection_daily_returns.csv"
RESULTS_FILE = "growth_final_selection_results.csv"
STRESS_FILE = "growth_final_selection_stress_periods.csv"
CONFIG_FILE = "growth_candidate_paper_config.json"

OUT_RESULTS = "growth_final_cost_slippage_results.csv"
OUT_BY_WINDOW = "growth_final_cost_slippage_by_window.csv"
OUT_COST_DRAG = "growth_final_cost_drag_analysis.csv"
OUT_BENCHMARKS = "growth_final_after_costs_vs_benchmarks.csv"
OUT_GOVERNANCE = "growth_final_cost_slippage_governance.csv"

CANDIDATE = "growth_champion_v3"

COST_SCENARIOS = [
    ("zero_cost", 0.0, 0.0),
    ("very_low_cost", 1.0, 2.0),
    ("realistic_us_liquid", 2.0, 5.0),
    ("conservative", 5.0, 10.0),
    ("stress", 10.0, 25.0),
    ("extreme", 25.0, 50.0),
]

WINDOWS = ["2008-01-01", "2015-01-01", "2020-01-01", "2022-01-03"]


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


def _periods_per_year(dates: pd.Series) -> float:
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return 52.0
    step = dates.diff().dt.days.dropna().median()
    return float(365.25 / step) if pd.notna(step) and step > 0 else 52.0


def _metrics(name: str, df: pd.DataFrame, return_col: str = "net_return") -> dict[str, object]:
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
    volatility = float(returns.std(ddof=0) * np.sqrt(ppy))
    sharpe = float((returns.mean() * ppy) / volatility) if volatility > 0 else np.nan
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
        "volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "hit_rate": float((returns > 0).mean()),
    }


def _scenario_daily(base: pd.DataFrame, scenario: str, commission_bps: float, slippage_bps: float) -> pd.DataFrame:
    out = base.copy()
    out["cost_scenario"] = scenario
    out["commission_bps"] = commission_bps
    out["slippage_bps"] = slippage_bps
    out["all_in_cost_bps"] = commission_bps + slippage_bps
    turnover = _num(out.get("candidate_turnover", pd.Series(0.0, index=out.index))).fillna(0.0)
    gross = _num(out["candidate_return"]).fillna(0.0)
    period_cost = turnover * (commission_bps + slippage_bps) / 10000.0
    out["gross_return"] = gross
    out["period_cost"] = period_cost
    out["net_return"] = gross - period_cost
    out["cost_drag_return"] = out["gross_return"] - out["net_return"]
    return out


def _base_daily() -> pd.DataFrame:
    daily = _dates(_read_csv(DAILY_FILE))
    if daily.empty:
        raise FileNotFoundError(f"Missing {DAILY_FILE}. Run growth_final_selection.py first.")
    data = daily[daily["candidate"].astype(str).eq(CANDIDATE)].copy()
    if data.empty:
        raise ValueError(f"{DAILY_FILE} does not contain {CANDIDATE}.")
    return data


def _gross_lookup() -> pd.DataFrame:
    results = _read_csv(RESULTS_FILE)
    if results.empty:
        return pd.DataFrame()
    return results[results["candidate"].astype(str).eq(CANDIDATE)].copy()


def _benchmark_columns(results: pd.DataFrame) -> pd.DataFrame:
    gross = _gross_lookup()
    if gross.empty:
        return results
    cols = [
        "window_start",
        "CAGR_SPY",
        "Sharpe_SPY",
        "max_drawdown_SPY",
        "CAGR_QQQ",
        "Sharpe_QQQ",
        "max_drawdown_QQQ",
        "production_champion_Sharpe",
        "production_champion_max_drawdown",
        "production_champion_return",
    ]
    gross = gross[[c for c in cols if c in gross.columns]].drop_duplicates("window_start")
    out = results.merge(gross, on="window_start", how="left")
    out["return_vs_SPY"] = out["CAGR"] - out.get("CAGR_SPY", np.nan)
    out["Sharpe_vs_SPY"] = out["Sharpe"] - out.get("Sharpe_SPY", np.nan)
    out["drawdown_vs_SPY"] = out["max_drawdown"] - out.get("max_drawdown_SPY", np.nan)
    out["return_vs_QQQ"] = out["CAGR"] - out.get("CAGR_QQQ", np.nan)
    out["Sharpe_vs_QQQ"] = out["Sharpe"] - out.get("Sharpe_QQQ", np.nan)
    out["drawdown_vs_QQQ"] = out["max_drawdown"] - out.get("max_drawdown_QQQ", np.nan)
    out["beats_SPY_CAGR_after_costs"] = out["CAGR"] > out.get("CAGR_SPY", np.nan)
    out["beats_QQQ_CAGR_after_costs"] = out["CAGR"] > out.get("CAGR_QQQ", np.nan)
    out["beats_SPY_Sharpe_after_costs"] = out["Sharpe"] > out.get("Sharpe_SPY", np.nan)
    out["beats_QQQ_Sharpe_after_costs"] = out["Sharpe"] > out.get("Sharpe_QQQ", np.nan)
    return out


def _cost_drag(rows: list[dict[str, object]], scenario_daily: pd.DataFrame) -> pd.DataFrame:
    gross_rows = [r for r in rows if r.get("cost_scenario") == "zero_cost"]
    gross_map = {(r["window_start"], r["model_scope"]): r for r in gross_rows}
    out_rows = []
    for row in rows:
        key = (row["window_start"], row["model_scope"])
        gross = gross_map.get(key, {})
        drag = float(gross.get("CAGR", np.nan)) - float(row.get("CAGR", np.nan))
        subset = scenario_daily[
            (scenario_daily["window_start"].astype(str).eq(str(row["window_start"])))
            & (scenario_daily["cost_scenario"].astype(str).eq(str(row["cost_scenario"])))
        ]
        ppy = _periods_per_year(subset["date"]) if not subset.empty else np.nan
        annual_cost_drag = float(_num(subset["period_cost"]).mean() * ppy) if not subset.empty and pd.notna(ppy) else np.nan
        total_cost_drag = float((1.0 + _num(subset["gross_return"]).fillna(0.0)).prod() - (1.0 + _num(subset["net_return"]).fillna(0.0)).prod()) if not subset.empty else np.nan
        out_rows.append(
            {
                "window_start": row["window_start"],
                "model_scope": row["model_scope"],
                "cost_scenario": row["cost_scenario"],
                "all_in_cost_bps": row["all_in_cost_bps"],
                "average_annual_cost_drag": annual_cost_drag,
                "total_cost_drag": total_cost_drag,
                "CAGR_drag_vs_zero_cost": drag,
                "average_turnover": row.get("average_turnover", np.nan),
            }
        )
    return pd.DataFrame(out_rows)


def _breakeven_cost(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, group in results.groupby("window_start"):
        group = group[group["model_scope"].eq("window")]
        if group.empty:
            continue
        spy = group["CAGR_SPY"].dropna()
        qqq = group["CAGR_QQQ"].dropna()
        spy_level = float(spy.iloc[0]) if not spy.empty else np.nan
        qqq_level = float(qqq.iloc[0]) if not qqq.empty else np.nan
        for benchmark, level in [("SPY", spy_level), ("QQQ", qqq_level)]:
            ordered = group.sort_values("all_in_cost_bps")
            below = ordered[ordered["CAGR"] <= level]
            if below.empty:
                breakeven = ">75bps"
            else:
                breakeven = float(below.iloc[0]["all_in_cost_bps"])
            rows.append({"window_start": window, "benchmark": benchmark, "breakeven_all_in_cost_bps": breakeven})
    return pd.DataFrame(rows)


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    realistic = results[(results["cost_scenario"].eq("realistic_us_liquid")) & (results["model_scope"].eq("window"))]
    conservative = results[(results["cost_scenario"].eq("conservative")) & (results["model_scope"].eq("window"))]
    if realistic.empty:
        classification = "fails_after_costs"
        reason = "No realistic scenario results."
    else:
        realistic_ok = bool(
            (realistic["CAGR"] > realistic["CAGR_SPY"]).mean() >= 0.75
            and (realistic["Sharpe"] > realistic["Sharpe_SPY"]).mean() >= 0.75
            and (realistic["max_drawdown"] > realistic["max_drawdown_SPY"]).mean() >= 0.50
        )
        conservative_ok = False
        if not conservative.empty:
            conservative_ok = bool(
                (conservative["CAGR"] > conservative["CAGR_SPY"]).mean() >= 0.75
                and (conservative["Sharpe"] > conservative["Sharpe_SPY"]).mean() >= 0.75
            )
        institutional = bool(
            conservative_ok
            and (conservative["CAGR"] > conservative["CAGR_QQQ"]).mean() >= 0.50
            and (conservative["Sharpe"] > conservative["Sharpe_QQQ"]).mean() >= 0.50
        ) if not conservative.empty else False
        if institutional:
            classification = "institutional_quality_after_costs"
            reason = "Conservative costs still beat SPY broadly and remain competitive versus QQQ."
        elif conservative_ok:
            classification = "robust_to_conservative_costs"
            reason = "Conservative costs still beat SPY broadly."
        elif realistic_ok:
            classification = "robust_to_realistic_costs"
            reason = "Realistic liquid US equity costs preserve broad SPY outperformance."
        else:
            classification = "fails_after_costs"
            reason = "Realistic costs remove too much benchmark-relative edge."
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "gross_or_net": "net_of_estimated_commission_and_slippage",
                "production_changed": False,
                "paper_changed": False,
                "parameter_tuning": False,
                "reason": reason,
            }
        ]
    )


def run_cost_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _base_daily()
    daily_frames = []
    rows = []
    for scenario, commission_bps, slippage_bps in COST_SCENARIOS:
        scenario_df = _scenario_daily(base, scenario, commission_bps, slippage_bps)
        daily_frames.append(scenario_df)
        for window in WINDOWS:
            subset = scenario_df[scenario_df["window_start"].astype(str).eq(window)].copy()
            metrics = _metrics(f"{CANDIDATE}_{window}_{scenario}", subset)
            metrics["candidate"] = CANDIDATE
            metrics["window_start"] = window
            metrics["model_scope"] = "window"
            metrics["cost_scenario"] = scenario
            metrics["commission_bps"] = commission_bps
            metrics["slippage_bps"] = slippage_bps
            metrics["all_in_cost_bps"] = commission_bps + slippage_bps
            metrics["average_exposure"] = float(_num(subset.get("candidate_exposure", pd.Series(dtype=float))).mean()) if not subset.empty else np.nan
            metrics["average_cash"] = float(_num(subset.get("candidate_cash", pd.Series(dtype=float))).mean()) if not subset.empty else np.nan
            metrics["average_turnover"] = float(_num(subset.get("candidate_turnover", pd.Series(dtype=float))).mean()) if not subset.empty else np.nan
            rows.append(metrics)
    all_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    results = pd.DataFrame(rows)
    results = _benchmark_columns(results)
    drag = _cost_drag(rows, all_daily)
    breakeven = _breakeven_cost(results)
    if not drag.empty and not breakeven.empty:
        breakeven_pivot = breakeven.pivot_table(index="window_start", columns="benchmark", values="breakeven_all_in_cost_bps", aggfunc="first").reset_index()
        breakeven_pivot.columns = ["window_start" if c == "window_start" else f"breakeven_cost_vs_{c}" for c in breakeven_pivot.columns]
        drag = drag.merge(breakeven_pivot, on="window_start", how="left")
    governance = _governance(results)

    all_daily.to_csv("growth_final_cost_slippage_daily_returns.csv", index=False)
    results.to_csv(OUT_RESULTS, index=False)
    results.to_csv(OUT_BY_WINDOW, index=False)
    drag.to_csv(OUT_COST_DRAG, index=False)
    benchmark_cols = [
        "candidate",
        "window_start",
        "cost_scenario",
        "all_in_cost_bps",
        "CAGR",
        "Sharpe",
        "max_drawdown",
        "CAGR_SPY",
        "Sharpe_SPY",
        "max_drawdown_SPY",
        "CAGR_QQQ",
        "Sharpe_QQQ",
        "max_drawdown_QQQ",
        "return_vs_SPY",
        "Sharpe_vs_SPY",
        "drawdown_vs_SPY",
        "return_vs_QQQ",
        "Sharpe_vs_QQQ",
        "drawdown_vs_QQQ",
    ]
    results[[c for c in benchmark_cols if c in results.columns]].to_csv(OUT_BENCHMARKS, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)
    _print_report(results, drag, governance)
    return results, drag, governance


def _pct(x: object) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "n/a"


def _fmt(x: object) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.3f}"
    except Exception:
        return "n/a"


def _print_report(results: pd.DataFrame, drag: pd.DataFrame, governance: pd.DataFrame) -> None:
    print("\n===== GROWTH FINAL COST & SLIPPAGE BACKTEST =====")
    print("Mode: research only. Reported strategy metrics are net of estimated costs.")
    print("Cost model: net_return = gross_return - turnover * (commission_bps + slippage_bps) / 10000.")

    print("\n===== COST SCENARIO COMPARISON =====")
    view = results[results["window_start"].isin(WINDOWS)][
        ["window_start", "cost_scenario", "all_in_cost_bps", "CAGR", "Sharpe", "Sortino", "Calmar", "max_drawdown", "average_turnover"]
    ].copy()
    for col in ["CAGR", "max_drawdown", "average_turnover"]:
        view[col] = view[col].map(_pct)
    for col in ["Sharpe", "Sortino", "Calmar"]:
        view[col] = view[col].map(_fmt)
    print(view.to_string(index=False))

    print("\n===== COST DRAG ANALYSIS =====")
    dview = drag[drag["cost_scenario"].isin(["realistic_us_liquid", "conservative", "stress", "extreme"])].copy()
    cols = ["window_start", "cost_scenario", "all_in_cost_bps", "average_annual_cost_drag", "total_cost_drag", "CAGR_drag_vs_zero_cost", "breakeven_all_in_cost_bps"]
    dview = dview[[c for c in cols if c in dview.columns]]
    for col in ["average_annual_cost_drag", "total_cost_drag", "CAGR_drag_vs_zero_cost"]:
        if col in dview.columns:
            dview[col] = dview[col].map(_pct)
    print(dview.to_string(index=False))

    print("\n===== BENCHMARK AFTER COSTS =====")
    bview = results[results["cost_scenario"].isin(["realistic_us_liquid", "conservative", "stress"])][
        ["window_start", "cost_scenario", "CAGR", "Sharpe", "max_drawdown", "return_vs_SPY", "Sharpe_vs_SPY", "return_vs_QQQ", "Sharpe_vs_QQQ"]
    ].copy()
    for col in ["CAGR", "max_drawdown", "return_vs_SPY", "return_vs_QQQ"]:
        bview[col] = bview[col].map(_pct)
    for col in ["Sharpe", "Sharpe_vs_SPY", "Sharpe_vs_QQQ"]:
        bview[col] = bview[col].map(_fmt)
    print(bview.to_string(index=False))

    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))


if __name__ == "__main__":
    run_cost_backtest()
