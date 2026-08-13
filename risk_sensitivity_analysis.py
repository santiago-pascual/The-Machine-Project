
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100000.0
MIN_EXPOSURE = 0.40
CURRENT_VOL_TARGET = 0.22
CURRENT_EXPOSURE_CAP = 0.60
DUAL_TREND_GRIDS = {
    "current_60_40_25": (0.60, 0.40, 0.25),
    "disabled_100_100_100": (1.00, 1.00, 1.00),
    "aggressive_50_30_15": (0.50, 0.30, 0.15),
    "conservative_70_50_35": (0.70, 0.50, 0.35),
}


def numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in ["date", "entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def benchmark_returns_from_cache(dates: pd.Series, ticker: str) -> pd.Series:
    path = Path("yahoo_ohlcv_price_cache") / f"{ticker}.csv"
    base = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()})
    if not path.exists() or base.empty:
        return pd.Series(0.0, index=base.index)
    px = pd.read_csv(path)
    date_col = "Date" if "Date" in px.columns else "date" if "date" in px.columns else None
    price_col = "Adj Close" if "Adj Close" in px.columns else "Close" if "Close" in px.columns else None
    if date_col is None or price_col is None:
        return pd.Series(0.0, index=base.index)
    work = px[[date_col, price_col]].copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work["price"] = numeric(work[price_col])
    work = work.dropna(subset=["date", "price"]).sort_values("date")
    aligned = pd.merge_asof(base, work[["date", "price"]], on="date", direction="backward")
    return aligned["price"].pct_change().fillna(0.0)


def periods_per_year(dates: pd.Series) -> float:
    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(d) < 2:
        return 52.0
    years = max((d.iloc[-1] - d.iloc[0]).days / 365.25, 1e-9)
    return len(d) / years


def metrics(df: pd.DataFrame, returns: pd.Series, exposure: pd.Series, cash: pd.Series, turnover: pd.Series, spy: pd.Series, qqq: pd.Series) -> dict:
    r = numeric(returns).fillna(0)
    freq = periods_per_year(df["date"])
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1) if not equity.empty else np.nan
    years = max((df["date"].max() - df["date"].min()).days / 365.25, 1e-9) if not df.empty else np.nan
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if not equity.empty and years > 0 else np.nan
    vol = float(r.std(ddof=0) * np.sqrt(freq)) if len(r) > 1 else np.nan
    sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(freq)) if len(r) > 1 and r.std(ddof=0) else np.nan
    downside = r[r < 0].std(ddof=0)
    sortino = float(r.mean() / downside * np.sqrt(freq)) if len(r) > 1 and downside else np.nan
    dd = equity / equity.cummax() - 1
    maxdd = float(dd.min()) if not dd.empty else np.nan
    calmar = float(cagr / abs(maxdd)) if pd.notna(cagr) and pd.notna(maxdd) and maxdd < 0 else np.nan
    spy_total = float((1 + numeric(spy).fillna(0)).prod() - 1) if len(spy) else np.nan
    qqq_total = float((1 + numeric(qqq).fillna(0)).prod() - 1) if len(qqq) else np.nan
    return {
        "total_return": total, "CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "max_drawdown": maxdd, "annual_volatility": vol, "average_cash": numeric(cash).mean(),
        "median_cash": numeric(cash).median(), "min_cash": numeric(cash).min(), "max_cash": numeric(cash).max(),
        "cash_utilization_pct": 1 - numeric(cash).mean(), "average_exposure": numeric(exposure).mean(),
        "turnover": numeric(turnover).mean(), "alpha_vs_SPY": total - spy_total, "alpha_vs_QQQ": total - qqq_total,
    }


def dual_cap_from_reason(reason: str, config: tuple[float, float, float]) -> float:
    both_above, one_below, both_below = config
    text = str(reason).lower()
    if "both" in text and "below" in text:
        return both_below
    if "only" in text or "one" in text:
        return one_below
    return both_above


def prepare_base() -> pd.DataFrame:
    final = load_csv("growth_final_selection_daily_returns.csv")
    base = load_csv("reconstructed_growth_long_horizon_daily_returns.csv")
    if final.empty:
        raise SystemExit("Missing growth_final_selection_daily_returns.csv")
    v3 = final[(final["window_start"].astype(str).eq("2008-01-01")) & (final["candidate"].astype(str).eq("growth_champion_v3"))].copy().sort_values("date")
    if v3.empty:
        raise SystemExit("Missing 2008 growth_champion_v3 rows")
    base2008 = base[base["window_start"].astype(str).eq("2008-01-01")].copy().sort_values("date") if not base.empty else pd.DataFrame()
    out = v3.merge(base2008[["date", "uncapped_exposure", "rolling_vol_used", "target_exposure"]], on="date", how="left", suffixes=("", "_base"))
    out["current_exposure"] = numeric(out["candidate_exposure"]).fillna(0)
    out["basket_return"] = numeric(out["candidate_return"]) / out["current_exposure"].replace(0, np.nan)
    out["basket_return"] = out["basket_return"].replace([np.inf, -np.inf], np.nan).fillna(0)
    out["current_dual_cap"] = numeric(out["overlay_cap"]).fillna(CURRENT_EXPOSURE_CAP)
    rv = numeric(out.get("rolling_vol_used", pd.Series(dtype=float)))
    out["vol_exposure_current"] = (CURRENT_VOL_TARGET / rv).clip(lower=MIN_EXPOSURE, upper=1.0)
    out.loc[out["vol_exposure_current"].isna(), "vol_exposure_current"] = numeric(out.get("uncapped_exposure", pd.Series(dtype=float))).fillna(CURRENT_EXPOSURE_CAP)
    out["spy_return"] = benchmark_returns_from_cache(out["date"], "SPY").values
    out["qqq_return"] = benchmark_returns_from_cache(out["date"], "QQQ").values
    return out


def simulate(base: pd.DataFrame, experiment_type: str, parameter: str, exposure_cap: float = CURRENT_EXPOSURE_CAP, vol_target: float = CURRENT_VOL_TARGET, dual_config: tuple[float, float, float] = DUAL_TREND_GRIDS["current_60_40_25"]) -> pd.DataFrame:
    out = base.copy()
    if experiment_type == "production":
        out["shadow_exposure"] = out["current_exposure"]
        out["shadow_cash"] = numeric(out.get("candidate_cash", pd.Series(dtype=float))).fillna(1 - out["shadow_exposure"])
        out["shadow_return"] = numeric(out.get("candidate_return", pd.Series(dtype=float))).fillna(0)
        out["experiment_type"] = experiment_type
        out["parameter"] = parameter
        return out
    rv = numeric(out.get("rolling_vol_used", pd.Series(dtype=float)))
    vol_exp = (vol_target / rv).clip(lower=MIN_EXPOSURE, upper=1.0)
    vol_exp = vol_exp.where(vol_exp.notna(), out["vol_exposure_current"])
    dual = out["overlay_reason"].map(lambda x: dual_cap_from_reason(x, dual_config))
    out["shadow_exposure"] = pd.concat([vol_exp, pd.Series(exposure_cap, index=out.index), dual], axis=1).min(axis=1)
    out["shadow_cash"] = 1 - out["shadow_exposure"]
    out["shadow_return"] = out["basket_return"] * out["shadow_exposure"]
    out["experiment_type"] = experiment_type
    out["parameter"] = parameter
    return out


def run_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = prepare_base()
    sims = []
    # production baseline
    sims.append(simulate(base, "production", "current", CURRENT_EXPOSURE_CAP, CURRENT_VOL_TARGET, DUAL_TREND_GRIDS["current_60_40_25"]))
    for cap in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        sims.append(simulate(base, "exposure_cap", f"{int(cap*100)}pct", cap, CURRENT_VOL_TARGET, DUAL_TREND_GRIDS["current_60_40_25"]))
    for vt in [0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
        sims.append(simulate(base, "vol_target", f"{int(vt*100)}pct", CURRENT_EXPOSURE_CAP, vt, DUAL_TREND_GRIDS["current_60_40_25"]))
    for name, cfg in DUAL_TREND_GRIDS.items():
        sims.append(simulate(base, "dual_trend", name, CURRENT_EXPOSURE_CAP, CURRENT_VOL_TARGET, cfg))
    daily = pd.concat(sims, ignore_index=True)
    rows = []
    for (etype, param), g in daily.groupby(["experiment_type", "parameter"], sort=False):
        m = metrics(g, g["shadow_return"], g["shadow_exposure"], g["shadow_cash"], g["candidate_turnover"], g["spy_return"], g["qqq_return"])
        rows.append({
            "experiment_type": etype, "parameter": param,
            "Exposure Cap": np.nan, "Vol Target": np.nan, "Dual Trend": np.nan,
            **m,
        })
        if etype == "exposure_cap": rows[-1]["Exposure Cap"] = float(param.replace("pct", "")) / 100
        if etype == "vol_target": rows[-1]["Vol Target"] = float(param.replace("pct", "")) / 100
        if etype == "dual_trend": rows[-1]["Dual Trend"] = param
    grid = pd.DataFrame(rows)
    prod = grid[grid["experiment_type"].eq("production")].iloc[0]
    for col in ["CAGR", "Sharpe", "Sortino", "Calmar", "max_drawdown", "annual_volatility", "average_cash", "cash_utilization_pct"]:
        grid[col + "_delta_vs_production"] = grid[col] - prod[col]
    return grid, daily


def pareto_flags(grid: pd.DataFrame) -> pd.DataFrame:
    work = grid.copy()
    work = work[~work["experiment_type"].eq("production")].copy()
    higher_is_better = ["CAGR", "Sharpe", "Calmar", "cash_utilization_pct", "max_drawdown"]
    lower_is_better = ["annual_volatility"]
    flags = []
    for i, row in work.iterrows():
        dominated = False
        for j, other in work.iterrows():
            if i == j: continue
            better_or_equal = all(other[c] >= row[c] for c in higher_is_better) and all(other[c] <= row[c] for c in lower_is_better)
            strictly = any(other[c] > row[c] for c in higher_is_better) or any(other[c] < row[c] for c in lower_is_better)
            if better_or_equal and strictly:
                dominated = True; break
        flags.append(not dominated)
    work["pareto_efficient"] = flags
    return work


def frontier_tables(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier = grid[["experiment_type", "parameter", "CAGR", "max_drawdown", "annual_volatility", "average_cash", "cash_utilization_pct", "Sharpe", "Calmar"]].copy()
    cash_frontier = frontier.sort_values(["cash_utilization_pct", "CAGR"], ascending=[False, False])
    pareto = pareto_flags(grid)
    return frontier, cash_frontier, pareto
