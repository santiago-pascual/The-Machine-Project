from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_FILE = "production_parity_growth_results.csv"
DAILY_FILE = "production_parity_growth_daily_returns.csv"
TRADES_FILE = "production_parity_growth_trades.csv"
BENCHMARK_FILE = "production_parity_growth_benchmark_comparison.csv"
ROBUSTNESS_FILE = "production_parity_growth_robustness.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"

OUT_ATTRIBUTION = "production_parity_drawdown_attribution.csv"
OUT_BY_TICKER = "production_parity_drawdown_by_ticker.csv"
OUT_VOL_TARGET = "production_parity_vol_target_behavior.csv"
OUT_OVERLAYS = "production_parity_drawdown_overlay_diagnostics.csv"
OUT_GOVERNANCE = "production_parity_drawdown_governance.csv"

TRADING_DAYS = 252
TARGET_VOL = 0.22


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date"])


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _ppy(dates: pd.Series) -> float:
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return 52.0
    step = np.median(dates.diff().dt.days.dropna())
    return float(365.25 / step) if np.isfinite(step) and step > 0 else 52.0


def _metrics(name: str, daily: pd.DataFrame, return_col: str = "return") -> dict[str, object]:
    data = _dates(daily).sort_values("date")
    if data.empty or return_col not in data.columns:
        return {"variant": name, "observations": 0}
    r = _num(data[return_col]).dropna()
    if r.empty:
        return {"variant": name, "observations": 0}
    ppy = _ppy(data["date"])
    total = float((1.0 + r).prod() - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(r) / ppy, 1e-9)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=0) * np.sqrt(ppy))
    sharpe = np.nan if vol <= 0 else float((r.mean() * ppy) / vol)
    downside = r[r < 0].std(ddof=0)
    sortino = np.nan if not np.isfinite(downside) or downside <= 0 else float((r.mean() * ppy) / (downside * np.sqrt(ppy)))
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    return {
        "variant": name,
        "observations": len(r),
        "total_return": total,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": max_dd,
        "Calmar": np.nan if max_dd >= 0 else cagr / abs(max_dd),
        "average_exposure": float(_num(data.get("target_exposure", pd.Series(index=data.index, dtype=float))).mean()),
    }


def _max_drawdown_window(daily: pd.DataFrame) -> dict[str, object]:
    data = _dates(daily).sort_values("date").copy()
    data["equity"] = (1.0 + _num(data["return"]).fillna(0.0)).cumprod()
    data["running_high"] = data["equity"].cummax()
    data["drawdown"] = data["equity"] / data["running_high"] - 1.0
    trough_idx = int(data["drawdown"].idxmin())
    trough_date = pd.Timestamp(data.loc[trough_idx, "date"])
    peak_slice = data.loc[:trough_idx]
    start_idx = int(peak_slice["equity"].idxmax())
    start_date = pd.Timestamp(data.loc[start_idx, "date"])
    peak_equity = float(data.loc[start_idx, "equity"])
    recovery = data.loc[trough_idx:]
    recovery = recovery[recovery["equity"] >= peak_equity]
    recovery_date = pd.NaT if recovery.empty else pd.Timestamp(recovery.iloc[0]["date"])
    dd_period = data[(data["date"] >= start_date) & (data["date"] <= trough_date)].copy()
    return {
        "start_date": start_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "duration_periods": len(dd_period),
        "portfolio_return": float(data.loc[trough_idx, "equity"] / peak_equity - 1.0),
        "period_frame": dd_period,
    }


def _benchmark_period_return(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty or "current_price" not in snaps.columns:
        return np.nan
    if "model_mode" in snaps.columns:
        base = snaps[snaps["model_mode"].astype(str).eq("baseline")]
        if not base.empty:
            snaps = base
    data = snaps[snaps["ticker"].astype(str).eq(ticker)].drop_duplicates("date").sort_values("date")
    data = data[(data["date"] >= start) & (data["date"] <= end)].copy()
    if len(data) < 2:
        return np.nan
    prices = _num(data["current_price"]).dropna()
    if len(prices) < 2 or prices.iloc[0] <= 0:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def _drawdown_ticker_attribution(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    trades = _dates(trades)
    period = trades[(trades["date"] >= start) & (trades["date"] <= end)].copy()
    if period.empty:
        return pd.DataFrame()
    grouped = period.groupby("ticker").agg(
        contribution=("trade_contribution", "sum"),
        avg_weight=("weight", "mean"),
        max_weight=("weight", "max"),
        avg_raw_target_rank=("raw_target_rank", "mean"),
        avg_raw_target_return=("raw_target_return_exact", "mean"),
        trades=("ticker", "size"),
        avg_realized_return=("realized_return_5d", "mean"),
    ).reset_index()
    total_loss = float(grouped.loc[grouped["contribution"] < 0, "contribution"].sum())
    grouped["loss_share"] = np.where(total_loss != 0, grouped["contribution"] / total_loss, np.nan)
    grouped = grouped.sort_values("contribution")
    return grouped


def _vol_target_behavior(daily: pd.DataFrame, start: pd.Timestamp, trough: pd.Timestamp, recovery: pd.Timestamp | pd.NaT) -> pd.DataFrame:
    data = _dates(daily).sort_values("date").copy()
    windows = {
        "before_drawdown": data[data["date"] < start].tail(8),
        "during_drawdown": data[(data["date"] >= start) & (data["date"] <= trough)],
        "after_trough": data[data["date"] > trough].head(8),
    }
    rows = []
    for name, frame in windows.items():
        if frame.empty:
            rows.append({"period": name})
            continue
        rows.append(
            {
                "period": name,
                "start": frame["date"].min().date().isoformat(),
                "end": frame["date"].max().date().isoformat(),
                "avg_exposure": float(_num(frame["target_exposure"]).mean()),
                "min_exposure": float(_num(frame["target_exposure"]).min()),
                "max_exposure": float(_num(frame["target_exposure"]).max()),
                "avg_cash": float(_num(frame["cash_weight"]).mean()),
                "avg_rolling_vol": float(_num(frame["rolling_vol_used"]).mean()),
                "avg_turnover": float(_num(frame["turnover"]).mean()),
                "period_return": float((1.0 + _num(frame["return"]).fillna(0.0)).prod() - 1.0),
            }
        )
    out = pd.DataFrame(rows)
    during = data[(data["date"] >= start) & (data["date"] <= trough)].copy()
    exposure_high_at_start = bool(not during.empty and float(during.iloc[0]["target_exposure"]) > 0.80)
    reduction_too_late = bool(
        not during.empty
        and (during["target_exposure"].iloc[: max(1, min(3, len(during)))].mean() > 0.80)
        and (during["return"].iloc[: max(1, min(3, len(during)))].sum() < -0.08)
    )
    out["exposure_too_high_at_drawdown_start"] = exposure_high_at_start
    out["exposure_reduction_too_late"] = reduction_too_late
    out["recovery_date"] = "" if pd.isna(recovery) else recovery.date().isoformat()
    return out


def _apply_overlay(daily: pd.DataFrame, overlay: str) -> pd.DataFrame:
    data = _dates(daily).sort_values("date").copy()
    returns = _num(data["return"]).fillna(0.0)
    base_exposure = _num(data["target_exposure"]).fillna(0.0).clip(0.0, 1.0)
    adjusted_exposure = base_exposure.copy()
    if overlay == "exposure_cap_70pct":
        adjusted_exposure = base_exposure.clip(upper=0.70)
    elif overlay == "exposure_cap_60pct":
        adjusted_exposure = base_exposure.clip(upper=0.60)
    elif overlay in {"drawdown_brake_15pct", "drawdown_brake_18pct"}:
        threshold = -0.15 if overlay.endswith("15pct") else -0.18
        equity = 1.0
        high = 1.0
        factors = []
        defensive = False
        for ret, exposure in zip(returns, base_exposure):
            dd = equity / high - 1.0
            if dd <= threshold:
                defensive = True
            if defensive and dd > -0.05:
                defensive = False
            factor = 0.50 if defensive else 1.0
            factors.append(factor)
            equity *= 1.0 + float(ret) * factor
            high = max(high, equity)
        adjusted_exposure = base_exposure * pd.Series(factors, index=data.index)
    elif overlay == "vol_target_faster_reaction":
        adj = []
        prev = float(base_exposure.iloc[0]) if len(base_exposure) else 1.0
        for i, ret in enumerate(returns):
            hist = returns.iloc[max(0, i - 6): i]
            if len(hist) < 3:
                raw = prev
            else:
                vol = float(hist.std(ddof=0) * np.sqrt(52.0))
                raw = prev if vol <= 0 or not np.isfinite(vol) else TARGET_VOL / vol
            raw = float(np.clip(raw, 0.40, 1.00))
            change = float(np.clip(raw - prev, -0.30, 0.30))
            prev = float(np.clip(prev + change, 0.40, 1.00))
            adj.append(prev)
        adjusted_exposure = pd.Series(adj, index=data.index)
    scale = np.where(base_exposure > 0, adjusted_exposure / base_exposure, 0.0)
    data["overlay"] = overlay
    data["base_return"] = returns
    data["return"] = returns * scale
    data["target_exposure"] = adjusted_exposure
    data["cash_weight"] = 1.0 - adjusted_exposure
    return data


def _overlay_diagnostics(daily: pd.DataFrame, original: dict[str, object]) -> pd.DataFrame:
    rows = []
    for overlay in [
        "original_growth",
        "exposure_cap_70pct",
        "exposure_cap_60pct",
        "drawdown_brake_15pct",
        "drawdown_brake_18pct",
        "vol_target_faster_reaction",
    ]:
        frame = daily.copy() if overlay == "original_growth" else _apply_overlay(daily, overlay)
        metrics = _metrics(overlay, frame)
        metrics["return_vs_original_growth"] = metrics.get("total_return", np.nan) - original.get("total_return", np.nan)
        metrics["DD_improvement_vs_original_growth"] = metrics.get("max_drawdown", np.nan) - original.get("max_drawdown", np.nan)
        metrics["shadow_candidate"] = bool(
            metrics.get("max_drawdown", -1) > -0.20
            and metrics.get("CAGR", 0) > 0.30
            and metrics.get("Sharpe", 0) > 1.20
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_production_parity_drawdown_attribution() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = _dates(_read_csv(DAILY_FILE)).sort_values("date")
    trades = _dates(_read_csv(TRADES_FILE)).sort_values("date")
    results = _read_csv(RESULTS_FILE)
    if daily.empty or trades.empty or results.empty:
        raise ValueError("Production-parity growth outputs are required. Run full_production_parity_growth_backtest.py first.")

    window = _max_drawdown_window(daily)
    start = pd.Timestamp(window["start_date"])
    trough = pd.Timestamp(window["trough_date"])
    recovery = pd.Timestamp(window["recovery_date"]) if not pd.isna(window["recovery_date"]) else pd.NaT
    spy_return = _benchmark_period_return("SPY", start, trough)
    qqq_return = _benchmark_period_return("QQQ", start, trough)
    period_frame = window["period_frame"].copy()

    attribution = pd.DataFrame(
        [
            {
                "drawdown_start": start.date().isoformat(),
                "drawdown_trough": trough.date().isoformat(),
                "drawdown_recovery": "" if pd.isna(recovery) else recovery.date().isoformat(),
                "duration_periods": window["duration_periods"],
                "portfolio_return_during_drawdown": window["portfolio_return"],
                "SPY_return_same_period": spy_return,
                "QQQ_return_same_period": qqq_return,
                "avg_exposure_during_drawdown": float(_num(period_frame["target_exposure"]).mean()),
                "avg_cash_during_drawdown": float(_num(period_frame["cash_weight"]).mean()),
                "avg_turnover_during_drawdown": float(_num(period_frame["turnover"]).mean()),
                "avg_rolling_vol_during_drawdown": float(_num(period_frame["rolling_vol_used"]).mean()),
                "avg_concentration_hhi_during_drawdown": float(_num(period_frame["concentration_hhi"]).mean()),
            }
        ]
    )

    by_ticker = _drawdown_ticker_attribution(trades, start, trough)
    vol_behavior = _vol_target_behavior(daily, start, trough, recovery)
    original_metrics = results.iloc[0].to_dict()
    overlays = _overlay_diagnostics(daily, original_metrics)

    worst_ticker = "" if by_ticker.empty else str(by_ticker.iloc[0]["ticker"])
    top5_loss = float(by_ticker.head(5)["contribution"].sum()) if not by_ticker.empty else np.nan
    total_loss = float(by_ticker.loc[by_ticker["contribution"] < 0, "contribution"].sum()) if not by_ticker.empty else np.nan
    one_ticker_damage = bool(not by_ticker.empty and abs(float(by_ticker.iloc[0]["loss_share"])) > 0.50)
    candidates = overlays[overlays["shadow_candidate"].astype(bool)] if "shadow_candidate" in overlays.columns else pd.DataFrame()
    governance = pd.DataFrame(
        [
            {
                "classification": "candidate_overlay_for_shadow_test" if not candidates.empty else "no_overlay_passed_shadow_requirements",
                "max_drawdown_explained_by_ticker": worst_ticker,
                "one_ticker_caused_most_damage": one_ticker_damage,
                "top5_ticker_loss_contribution": top5_loss,
                "total_negative_ticker_contribution": total_loss,
                "exposure_too_high_at_drawdown_start": bool(vol_behavior.get("exposure_too_high_at_drawdown_start", pd.Series([False])).iloc[0]),
                "exposure_reduction_too_late": bool(vol_behavior.get("exposure_reduction_too_late", pd.Series([False])).iloc[0]),
                "candidate_overlays": ",".join(candidates["variant"].astype(str).tolist()) if not candidates.empty else "",
                "production_changed": False,
                "growth_paper_changed": False,
                "reason": "Diagnostic-only attribution and fixed overlay tests; no production or paper settings changed.",
            }
        ]
    )

    attribution.to_csv(OUT_ATTRIBUTION, index=False)
    by_ticker.to_csv(OUT_BY_TICKER, index=False)
    vol_behavior.to_csv(OUT_VOL_TARGET, index=False)
    overlays.to_csv(OUT_OVERLAYS, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== PRODUCTION-PARITY DRAWDOWN ATTRIBUTION =====")
    print(attribution.to_string(index=False))
    print("\n===== MAX DRAWDOWN PERIOD =====")
    print(
        f"{start.date().isoformat()} -> {trough.date().isoformat()} | "
        f"recovery={'' if pd.isna(recovery) else recovery.date().isoformat()} | "
        f"portfolio={float(window['portfolio_return']):.6f} | SPY={spy_return:.6f} | QQQ={qqq_return:.6f}"
    )
    print("\n===== DRAWDOWN CONTRIBUTION BY TICKER =====")
    print(by_ticker.head(12).to_string(index=False))
    print("\n===== VOL TARGETING BEHAVIOR =====")
    print(vol_behavior.to_string(index=False))
    print("\n===== DRAWDOWN OVERLAY DIAGNOSTICS =====")
    print(overlays[["variant", "total_return", "CAGR", "Sharpe", "Sortino", "max_drawdown", "Calmar", "average_exposure", "return_vs_original_growth", "DD_improvement_vs_original_growth", "shadow_candidate"]].to_string(index=False))
    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_ATTRIBUTION).resolve()}")
    print(f"Saved: {Path(OUT_BY_TICKER).resolve()}")
    print(f"Saved: {Path(OUT_VOL_TARGET).resolve()}")
    print(f"Saved: {Path(OUT_OVERLAYS).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return attribution, by_ticker, overlays


if __name__ == "__main__":
    run_production_parity_drawdown_attribution()
