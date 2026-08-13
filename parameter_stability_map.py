from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_VOLS = [0.18, 0.20, 0.22, 0.24, 0.26]
EXPOSURE_CAPS = [0.50, 0.55, 0.60, 0.65, 0.70]
MIN_EXPOSURES = [0.20, 0.30, 0.40, None]
VOL_LOOKBACKS = [40, 60, 90]
DUAL_TREND_CAPS = [
    (0.60, 0.40, 0.25),
    (0.60, 0.35, 0.20),
    (0.65, 0.45, 0.25),
]
CURRENT_CONFIG = (0.22, 0.60, 0.40, 60, (0.60, 0.40, 0.25))


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def periods_per_year(dates: pd.Series) -> float:
    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(d) < 3:
        return 52.0
    step = d.diff().dt.days.dropna().median()
    return float(365.25 / step) if np.isfinite(step) and step > 0 else 52.0


def perf_metrics(df: pd.DataFrame) -> dict[str, float]:
    r = pd.to_numeric(df["return"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}
    dates = pd.to_datetime(df.loc[r.index, "date"], errors="coerce")
    ppy = periods_per_year(dates)
    total = float((1 + r).prod() - 1)
    years = max((dates.max() - dates.min()).days / 365.25, 1e-9)
    cagr = float((1 + total) ** (1 / years) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(ppy)) if len(r) > 2 else np.nan
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(ppy)) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std(ddof=1) * math.sqrt(ppy)) if len(downside) > 2 and downside.std(ddof=1) > 0 else np.nan
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    return {
        "observations": int(len(r)),
        "total_return": total,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0).mean()),
        "average_exposure": float(pd.to_numeric(df["exposure"], errors="coerce").mean()),
        "average_cash": float(1 - pd.to_numeric(df["exposure"], errors="coerce").mean()),
        "average_turnover": float(pd.to_numeric(df["turnover"], errors="coerce").mean()),
    }


def load_base() -> pd.DataFrame:
    trades = read_csv("reconstructed_growth_long_horizon_trades.csv")
    if trades.empty:
        return pd.DataFrame()
    if "window_start" in trades.columns:
        ws = pd.to_datetime(trades["window_start"], errors="coerce")
        canonical = ws.dropna().min()
        trades = trades.loc[ws.eq(canonical)].copy()
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
    trades["asset_return"] = pd.to_numeric(trades["asset_return"], errors="coerce")
    grouped = trades.dropna(subset=["date", "asset_return"]).groupby("date").agg(
        basket_return=("asset_return", "mean"),
        selected_count=("ticker", "nunique"),
        selected_tickers=("ticker", lambda s: ",".join(sorted(set(map(str, s))))),
    ).reset_index()
    overlay = read_csv("growth_crisis_overlay_daily_returns.csv")
    if not overlay.empty:
        if "window_start" in overlay.columns:
            ws = pd.to_datetime(overlay["window_start"], errors="coerce")
            canonical = ws.dropna().min()
            overlay = overlay.loc[ws.eq(canonical)].copy()
        overlay = overlay.loc[overlay["overlay"].astype(str).eq("base_growth_v2")].copy()
        overlay["date"] = pd.to_datetime(overlay["date"], errors="coerce")
        grouped = grouped.merge(
            overlay[["date", "spy_below_200dma", "qqq_below_200dma"]],
            on="date",
            how="left",
        )
    grouped["spy_below_200dma"] = grouped["spy_below_200dma"].fillna(False).astype(bool)
    grouped["qqq_below_200dma"] = grouped["qqq_below_200dma"].fillna(False).astype(bool)
    return grouped.sort_values("date").reset_index(drop=True)


def dual_cap(row: pd.Series, caps: tuple[float, float, float]) -> float:
    both_above, one_below, both_below = caps
    spy = bool(row["spy_below_200dma"])
    qqq = bool(row["qqq_below_200dma"])
    if spy and qqq:
        return both_below
    if spy or qqq:
        return one_below
    return both_above


def simulate(base: pd.DataFrame, target_vol: float, exposure_cap: float, min_exposure: float | None, lookback_days: int, caps: tuple[float, float, float]) -> pd.DataFrame:
    df = base.copy()
    ppy = periods_per_year(df["date"])
    median_step = pd.to_datetime(df["date"]).diff().dt.days.dropna().median()
    lookback_periods = max(3, int(round(lookback_days / max(float(median_step), 1.0))))
    rolling_vol = df["basket_return"].rolling(lookback_periods).std() * math.sqrt(ppy)
    uncapped = target_vol / rolling_vol.replace(0, np.nan)
    uncapped = uncapped.replace([np.inf, -np.inf], np.nan).fillna(exposure_cap)
    floor = 0.0 if min_exposure is None else float(min_exposure)
    df["dual_trend_cap"] = df.apply(lambda r: dual_cap(r, caps), axis=1)
    df["uncapped_exposure"] = uncapped
    df["exposure"] = np.minimum(np.maximum(uncapped, floor), np.minimum(exposure_cap, df["dual_trend_cap"]))
    df["return"] = df["basket_return"] * df["exposure"]
    weights_by_date = []
    prev = {}
    for _, row in df.iterrows():
        tickers = [t for t in str(row["selected_tickers"]).split(",") if t]
        w = float(row["exposure"]) / len(tickers) if tickers else 0.0
        cur = {t: w for t in tickers}
        turnover = sum(abs(cur.get(t, 0.0) - prev.get(t, 0.0)) for t in set(cur) | set(prev))
        weights_by_date.append(turnover)
        prev = cur
    df["turnover"] = weights_by_date
    return df


def window_stability(sim: pd.DataFrame) -> dict[str, float]:
    windows = {
        "2008_plus": ("2008-01-01", None),
        "2015_plus": ("2015-01-01", None),
        "2020_plus": ("2020-01-01", None),
        "2022_plus": ("2022-01-03", None),
    }
    sharpes = []
    cagrs = []
    for _, (start, end) in windows.items():
        d = sim.loc[sim["date"] >= pd.Timestamp(start)]
        if end:
            d = d.loc[d["date"] <= pd.Timestamp(end)]
        m = perf_metrics(d)
        if np.isfinite(m.get("Sharpe", np.nan)):
            sharpes.append(m["Sharpe"])
        if np.isfinite(m.get("CAGR", np.nan)):
            cagrs.append(m["CAGR"])
    return {
        "window_sharpe_mean": float(np.mean(sharpes)) if sharpes else np.nan,
        "window_sharpe_std": float(np.std(sharpes)) if sharpes else np.nan,
        "window_cagr_mean": float(np.mean(cagrs)) if cagrs else np.nan,
        "window_cagr_std": float(np.std(cagrs)) if cagrs else np.nan,
    }


def main() -> None:
    base = load_base()
    if base.empty:
        empty = pd.DataFrame([{"status": "missing reconstructed growth trades"}])
        for path in ["parameter_stability_map.csv", "parameter_sensitivity_results.csv", "robustness_plateau_analysis.csv", "parameter_governance.csv"]:
            empty.to_csv(path, index=False)
        print("missing reconstructed growth trades")
        return
    rows = []
    current_key = None
    for target_vol, cap, floor, lookback, dual_caps in itertools.product(TARGET_VOLS, EXPOSURE_CAPS, MIN_EXPOSURES, VOL_LOOKBACKS, DUAL_TREND_CAPS):
        sim = simulate(base, target_vol, cap, floor, lookback, dual_caps)
        metrics = perf_metrics(sim)
        stable = window_stability(sim)
        row = {
            "target_vol": target_vol,
            "exposure_cap": cap,
            "min_exposure": "none" if floor is None else floor,
            "vol_lookback_days": lookback,
            "dual_trend_caps": f"{dual_caps[0]:.2f}/{dual_caps[1]:.2f}/{dual_caps[2]:.2f}",
            "is_current_config": (target_vol, cap, floor, lookback, dual_caps) == CURRENT_CONFIG,
            **metrics,
            **stable,
        }
        rows.append(row)
        if row["is_current_config"]:
            current_key = row
    results = pd.DataFrame(rows)
    if current_key is None:
        current_key = results.iloc[(results["target_vol"].sub(0.22).abs() + results["exposure_cap"].sub(0.60).abs()).idxmin()].to_dict()
    cur_sharpe = float(current_key["Sharpe"])
    cur_cagr = float(current_key["CAGR"])
    cur_dd = float(current_key["max_drawdown"])
    results["Sharpe_delta_vs_current"] = results["Sharpe"] - cur_sharpe
    results["CAGR_delta_vs_current"] = results["CAGR"] - cur_cagr
    results["DD_delta_vs_current"] = results["max_drawdown"] - cur_dd
    plateau = results.loc[
        (results["Sharpe"] >= cur_sharpe * 0.95)
        & (results["CAGR"] >= cur_cagr * 0.90)
        & (results["max_drawdown"] >= cur_dd - 0.05)
    ].copy()
    local = results.loc[
        results["target_vol"].between(0.20, 0.24)
        & results["exposure_cap"].between(0.55, 0.65)
        & results["vol_lookback_days"].isin([40, 60, 90])
    ].copy()
    plateau_summary = pd.DataFrame([
        {
            "total_configs": len(results),
            "plateau_configs": len(plateau),
            "plateau_pct": len(plateau) / len(results),
            "local_configs": len(local),
            "local_sharpe_std": float(local["Sharpe"].std()),
            "local_cagr_std": float(local["CAGR"].std()),
            "current_sharpe": cur_sharpe,
            "current_CAGR": cur_cagr,
            "current_max_drawdown": cur_dd,
            "current_rank_by_sharpe": int(results["Sharpe"].rank(ascending=False, method="min").loc[results["is_current_config"]].iloc[0]),
            "current_rank_by_CAGR": int(results["CAGR"].rank(ascending=False, method="min").loc[results["is_current_config"]].iloc[0]),
            "interpretation": "broad plateau" if len(plateau) / len(results) >= 0.20 and local["Sharpe"].std() < 0.20 else "moderate sensitivity" if len(plateau) / len(results) >= 0.08 else "sharp isolated optimum",
        }
    ])
    interp = plateau_summary.iloc[0]["interpretation"]
    classification = "stable_plateau" if interp == "broad plateau" else "moderately_sensitive" if interp == "moderate sensitivity" else "fragile_configuration"
    governance = pd.DataFrame([
        {
            "classification": classification,
            "active_parameters_changed": False,
            "production_changed": False,
            "paper_changed": False,
            "new_best_selected": False,
            "reason": f"{interp}; plateau_pct={plateau_summary.iloc[0]['plateau_pct']:.2%}; local_sharpe_std={plateau_summary.iloc[0]['local_sharpe_std']:.3f}",
        }
    ])
    results.to_csv("parameter_sensitivity_results.csv", index=False)
    results.sort_values(["target_vol", "exposure_cap", "vol_lookback_days"]).to_csv("parameter_stability_map.csv", index=False)
    plateau_summary.to_csv("robustness_plateau_analysis.csv", index=False)
    governance.to_csv("parameter_governance.csv", index=False)
    print("===== PARAMETER ROBUSTNESS AND STABILITY MAP =====")
    print(f"configs_tested: {len(results)}")
    print(f"current_sharpe: {cur_sharpe:.4f}")
    print(f"current_CAGR: {cur_cagr:.4f}")
    print(f"plateau_pct: {plateau_summary.iloc[0]['plateau_pct']:.2%}")
    print(f"classification: {classification}")
    print("outputs: parameter_stability_map.csv, parameter_sensitivity_results.csv, robustness_plateau_analysis.csv, parameter_governance.csv")


if __name__ == "__main__":
    main()
