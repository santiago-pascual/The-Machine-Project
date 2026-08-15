from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OVERLAY_RESULTS = "growth_crisis_overlay_results.csv"
OVERLAY_DAILY = "growth_crisis_overlay_daily_returns.csv"
OVERLAY_STRESS = "growth_crisis_overlay_stress_periods.csv"
BENCHMARK_RESULTS = "reconstructed_growth_benchmark_comparison.csv"
PRODUCTION_CHAMPION_RESULTS = "final_champion_backtest_results.csv"

OUT_RESULTS = "growth_final_selection_results.csv"
OUT_DAILY = "growth_final_selection_daily_returns.csv"
OUT_DRAWDOWNS = "growth_final_selection_drawdowns.csv"
OUT_STRESS = "growth_final_selection_stress_periods.csv"
OUT_GOVERNANCE = "growth_final_selection_governance.csv"

V2_OVERLAY = "base_growth_v2"
V3_OVERLAY = "dual_trend_filter"
V2_NAME = "growth_champion_v2"
V3_NAME = "growth_champion_v3"
FINAL_NAME = "growth_champion_final"

STRESS_ORDER = [
    "2008_crisis",
    "2011_euro_crisis",
    "2018_q4_selloff",
    "covid_crash_2020",
    "2022_bear_market",
    "2024_ai_bull_market",
]


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


def _candidate_name(overlay: str) -> str:
    if overlay == V2_OVERLAY:
        return V2_NAME
    if overlay == V3_OVERLAY:
        return V3_NAME
    return overlay


def _candidate_daily() -> pd.DataFrame:
    daily = _read_csv(OVERLAY_DAILY)
    if daily.empty:
        return pd.DataFrame()
    daily = _dates(daily)
    daily = daily[daily["overlay"].isin([V2_OVERLAY, V3_OVERLAY])].copy()
    daily["candidate"] = daily["overlay"].map(_candidate_name)
    daily["candidate_return"] = _num(daily["overlay_return"])
    daily["candidate_exposure"] = _num(daily["overlay_exposure"])
    daily["candidate_cash"] = _num(daily["overlay_cash"])
    daily["candidate_turnover"] = _num(daily["overlay_turnover_proxy"])
    keep = [
        "window_start",
        "date",
        "entry_date",
        "exit_date",
        "candidate",
        "candidate_return",
        "candidate_exposure",
        "candidate_cash",
        "candidate_turnover",
        "selected_tickers",
        "selected_count",
        "overlay_cap",
        "overlay_scale",
        "overlay_reason",
    ]
    return daily[[c for c in keep if c in daily.columns]].copy()


def _drawdowns(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if daily.empty:
        return pd.DataFrame()
    for (window, candidate), group in daily.groupby(["window_start", "candidate"]):
        g = _dates(group).copy()
        returns = _num(g["candidate_return"]).fillna(0.0)
        equity = (1.0 + returns).cumprod()
        dd = equity / equity.cummax() - 1.0
        out = g[["window_start", "date", "candidate"]].copy()
        out["equity"] = equity.values
        out["drawdown"] = dd.values
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _results() -> pd.DataFrame:
    results = _read_csv(OVERLAY_RESULTS)
    if results.empty:
        return pd.DataFrame()
    out = results[results["overlay"].isin([V2_OVERLAY, V3_OVERLAY])].copy()
    out["candidate"] = out["overlay"].map(_candidate_name)

    needed_benchmark_cols = {"CAGR_SPY", "Sharpe_SPY", "max_drawdown_SPY", "CAGR_QQQ", "Sharpe_QQQ", "max_drawdown_QQQ"}
    bench = _read_csv(BENCHMARK_RESULTS)
    if not needed_benchmark_cols.issubset(set(out.columns)) and not bench.empty and {"window_start", "benchmark"}.issubset(bench.columns):
        pivot = bench.pivot_table(
            index="window_start",
            columns="benchmark",
            values=["CAGR", "Sharpe", "Sortino", "Calmar", "max_drawdown", "total_return"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{benchmark}" for metric, benchmark in pivot.columns]
        out = out.merge(pivot.reset_index(), on="window_start", how="left")

    prod = _read_csv(PRODUCTION_CHAMPION_RESULTS)
    if not prod.empty:
        row = prod.iloc[0]
        out["production_champion_Sharpe"] = row.get("Sharpe", np.nan)
        out["production_champion_max_drawdown"] = row.get("max_drawdown", np.nan)
        out["production_champion_return"] = row.get("realized_return", np.nan)

    cols = [
        "candidate",
        "window_start",
        "start_date",
        "end_date",
        "observations",
        "total_return",
        "CAGR",
        "volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
        "average_turnover",
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
    return out[[c for c in cols if c in out.columns]].sort_values(["window_start", "candidate"])


def _stress() -> pd.DataFrame:
    stress = _read_csv(OVERLAY_STRESS)
    if stress.empty:
        return pd.DataFrame()
    out = stress[stress["overlay"].isin([V2_OVERLAY, V3_OVERLAY])].copy()
    out["candidate"] = out["overlay"].map(_candidate_name)
    if "stress_period" in out.columns:
        out["stress_order"] = out["stress_period"].map({name: i for i, name in enumerate(STRESS_ORDER)}).fillna(999)
        out = out.sort_values(["window_start", "stress_order", "candidate"])
    cols = [
        "candidate",
        "window_start",
        "stress_period",
        "start_date",
        "end_date",
        "observations",
        "total_return",
        "CAGR",
        "volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "hit_rate",
    ]
    return out[[c for c in cols if c in out.columns]]


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            [
                {
                    "selected_champion": V2_NAME,
                    "classification": "research_only",
                    "reason": "Missing final selection inputs.",
                    "production_changed": False,
                    "paper_changed": False,
                }
            ]
        )
    pivot = results.pivot(index="window_start", columns="candidate")
    rows = []
    pass_all = True
    for window in sorted(results["window_start"].dropna().astype(str).unique()):
        v2 = results[(results["window_start"].eq(window)) & (results["candidate"].eq(V2_NAME))]
        v3 = results[(results["window_start"].eq(window)) & (results["candidate"].eq(V3_NAME))]
        if v2.empty or v3.empty:
            pass_all = False
            rows.append({"window_start": window, "window_pass": False, "reason": "Missing v2 or v3 row."})
            continue
        a = v2.iloc[0]
        b = v3.iloc[0]
        cagr_sacrifice = (float(a["CAGR"]) - float(b["CAGR"])) / abs(float(a["CAGR"])) if float(a["CAGR"]) != 0 else np.inf
        checks = {
            "Sharpe_improved": float(b["Sharpe"]) > float(a["Sharpe"]),
            "Sortino_improved": float(b["Sortino"]) > float(a["Sortino"]),
            "Calmar_improved": float(b["Calmar"]) > float(a["Calmar"]),
            "drawdown_improved": float(b["max_drawdown"]) > float(a["max_drawdown"]),
            "CAGR_sacrifice_le_15pct": cagr_sacrifice <= 0.15,
        }
        window_pass = all(checks.values())
        pass_all = pass_all and window_pass
        rows.append(
            {
                "window_start": window,
                "window_pass": window_pass,
                "v2_CAGR": a["CAGR"],
                "v3_CAGR": b["CAGR"],
                "v2_Sharpe": a["Sharpe"],
                "v3_Sharpe": b["Sharpe"],
                "v2_Sortino": a["Sortino"],
                "v3_Sortino": b["Sortino"],
                "v2_Calmar": a["Calmar"],
                "v3_Calmar": b["Calmar"],
                "v2_max_drawdown": a["max_drawdown"],
                "v3_max_drawdown": b["max_drawdown"],
                "cagr_sacrifice_pct": cagr_sacrifice,
                **checks,
            }
        )
    selected = FINAL_NAME if pass_all else V2_NAME
    classification = "growth_champion_final" if pass_all else "keep_growth_champion_v2"
    reason = (
        "v3 improves Sharpe, Sortino, Calmar and drawdown in every window without >15% CAGR sacrifice."
        if pass_all
        else "v3 does not pass all strict selection checks; keep v2."
    )
    summary = {
        "window_start": "ALL",
        "window_pass": pass_all,
        "selected_champion": selected,
        "classification": classification,
        "production_changed": False,
        "paper_changed": False,
        "parameter_tuning": False,
        "new_overlay_added": False,
        "reason": reason,
    }
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([summary])], ignore_index=True)


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


def _print_report(results: pd.DataFrame, stress: pd.DataFrame, governance: pd.DataFrame) -> None:
    print("\n===== GROWTH FINAL SELECTION =====")
    print("Research only. Production and paper trading unchanged.")
    print("A = growth_champion_v2. B = growth_champion_v3 = v2 + dual_trend_filter.")

    print("\n===== CHAMPION COMPARISON =====")
    view_cols = [
        "window_start",
        "candidate",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
        "average_turnover",
        "CAGR_SPY",
        "Sharpe_SPY",
        "CAGR_QQQ",
        "Sharpe_QQQ",
    ]
    view = results[[c for c in view_cols if c in results.columns]].copy()
    for col in ["CAGR", "max_drawdown", "average_exposure", "average_cash", "average_turnover", "CAGR_SPY", "CAGR_QQQ"]:
        if col in view.columns:
            view[col] = view[col].map(_pct)
    for col in ["Sharpe", "Sortino", "Calmar", "Sharpe_SPY", "Sharpe_QQQ"]:
        if col in view.columns:
            view[col] = view[col].map(_fmt)
    print(view.to_string(index=False))

    print("\n===== STRESS TEST COMPARISON =====")
    if stress.empty:
        print("No stress data available.")
    else:
        sview = stress[["window_start", "candidate", "stress_period", "total_return", "max_drawdown", "Sharpe"]].copy()
        sview["total_return"] = sview["total_return"].map(_pct)
        sview["max_drawdown"] = sview["max_drawdown"].map(_pct)
        sview["Sharpe"] = sview["Sharpe"].map(_fmt)
        print(sview.head(80).to_string(index=False))

    print("\n===== GOVERNANCE =====")
    print(governance.tail(1).to_string(index=False))


def run_final_selection() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = _results()
    daily = _candidate_daily()
    drawdowns = _drawdowns(daily)
    stress = _stress()
    governance = _governance(results)

    results.to_csv(OUT_RESULTS, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    drawdowns.to_csv(OUT_DRAWDOWNS, index=False)
    stress.to_csv(OUT_STRESS, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    _print_report(results, stress, governance)
    return results, daily, governance


if __name__ == "__main__":
    run_final_selection()
