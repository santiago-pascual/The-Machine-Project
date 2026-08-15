from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAILY_FILE = "production_parity_growth_daily_returns.csv"
TRADES_FILE = "production_parity_growth_trades.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"

OUT_RESULTS = "growth_head_to_head_results.csv"
OUT_DAILY = "growth_head_to_head_daily_returns.csv"
OUT_DRAWDOWNS = "growth_head_to_head_drawdowns.csv"
OUT_YEARLY = "growth_head_to_head_yearly.csv"
OUT_ROLLING = "growth_head_to_head_rolling.csv"
OUT_GOVERNANCE = "growth_head_to_head_governance.csv"

TRADING_DAYS = 252


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


def _equity_drawdown(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    r = _num(returns).fillna(0.0)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return equity, dd


def _metrics(name: str, daily: pd.DataFrame) -> dict[str, object]:
    data = _dates(daily).sort_values("date")
    if data.empty or "return" not in data.columns:
        return {"candidate": name, "observations": 0}
    r = _num(data["return"]).dropna()
    if r.empty:
        return {"candidate": name, "observations": 0}
    ppy = _ppy(data["date"])
    total = float((1.0 + r).prod() - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(r) / ppy, 1e-9)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=0) * np.sqrt(ppy))
    sharpe = np.nan if vol <= 0 else float((r.mean() * ppy) / vol)
    downside = r[r < 0].std(ddof=0)
    sortino = np.nan if not np.isfinite(downside) or downside <= 0 else float((r.mean() * ppy) / (downside * np.sqrt(ppy)))
    _, dd = _equity_drawdown(r)
    max_dd = float(dd.min())
    return {
        "candidate": name,
        "start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "observations": len(r),
        "total_return": total,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": np.nan if max_dd >= 0 else cagr / abs(max_dd),
        "max_drawdown": max_dd,
        "average_exposure": float(_num(data.get("target_exposure", pd.Series(index=data.index, dtype=float))).mean()),
        "average_cash": float(_num(data.get("cash_weight", pd.Series(index=data.index, dtype=float))).mean()),
        "turnover": float(_num(data.get("turnover", pd.Series(index=data.index, dtype=float))).mean()),
        "hit_rate": float((r > 0).mean()),
    }


def _candidate_daily() -> dict[str, pd.DataFrame]:
    base = _dates(_read_csv(DAILY_FILE)).sort_values("date")
    if base.empty:
        raise ValueError("production_parity_growth_daily_returns.csv is required.")
    a = base.copy()
    a["candidate"] = "growth_v1"
    a["return"] = _num(a["return"]).fillna(0.0)
    a["target_exposure"] = _num(a["target_exposure"]).fillna(0.0).clip(0.0, 1.0)
    a["cash_weight"] = 1.0 - a["target_exposure"]

    b = a.copy()
    b["candidate"] = "growth_v1_exposure_cap_60"
    capped = b["target_exposure"].clip(upper=0.60)
    scale = np.where(b["target_exposure"] > 0, capped / b["target_exposure"], 0.0)
    b["return"] = b["return"] * scale
    b["target_exposure"] = capped
    b["cash_weight"] = 1.0 - capped
    return {"growth_v1": a, "growth_v1_exposure_cap_60": b}


def _tp_sl() -> dict[str, float]:
    trades = _dates(_read_csv(TRADES_FILE))
    labels = _dates(_read_csv(LABELS_FILE))
    if trades.empty or labels.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    if "model_mode" in labels.columns:
        base = labels[labels["model_mode"].astype(str).eq("baseline")]
        if not base.empty:
            labels = base
    if "horizon" in labels.columns:
        labels = labels[labels["horizon"].eq(20)]
    merged = trades[["date", "ticker"]].merge(labels[["date", "ticker", "label"]], on=["date", "ticker"], how="left")
    merged = merged.dropna(subset=["label"])
    if merged.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((merged["label"] == 1).mean())
    sl = float((merged["label"] == -1).mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _benchmark_daily(ticker: str, dates: pd.Series) -> pd.DataFrame:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty:
        return pd.DataFrame()
    if "model_mode" in snaps.columns:
        base = snaps[snaps["model_mode"].astype(str).eq("baseline")]
        if not base.empty:
            snaps = base
    data = snaps[snaps["ticker"].astype(str).eq(ticker)].drop_duplicates("date").sort_values("date")
    data = data[data["date"].isin(pd.to_datetime(dates).dt.normalize())].copy()
    if data.empty:
        return data
    data["return"] = _num(data["current_price"]).pct_change()
    data["target_exposure"] = 1.0
    data["cash_weight"] = 0.0
    data["turnover"] = 0.0
    return data.dropna(subset=["return"])


def _production_champion_daily(dates: pd.Series) -> pd.DataFrame:
    pf = _dates(_read_csv(PORTFOLIO_FILE))
    if pf.empty or "model_mode" not in pf.columns:
        return pd.DataFrame()
    data = pf[pf["model_mode"].astype(str).eq("regime_gated_full_quant")].copy()
    data = data[data["date"].isin(pd.to_datetime(dates).dt.normalize())]
    data["return"] = _num(data.get("realized_portfolio_return_5d", pd.Series(index=data.index, dtype=float)))
    data["target_exposure"] = 1.0 - _num(data.get("cash_weight", pd.Series(index=data.index, dtype=float))).fillna(0.0)
    data["turnover"] = _num(data.get("turnover", pd.Series(index=data.index, dtype=float))).fillna(0.0)
    return data.dropna(subset=["return"]).sort_values("date")


def _drawdown_summary(name: str, daily: pd.DataFrame) -> pd.DataFrame:
    data = _dates(daily).sort_values("date").copy()
    _, dd = _equity_drawdown(data["return"])
    data["drawdown"] = dd.values
    rows = []
    in_dd = False
    start = None
    trough = None
    trough_value = 0.0
    duration = 0
    for _, row in data.iterrows():
        current_dd = float(row["drawdown"])
        if current_dd < 0 and not in_dd:
            in_dd = True
            start = row["date"]
            trough = row["date"]
            trough_value = current_dd
            duration = 1
        elif current_dd < 0 and in_dd:
            duration += 1
            if current_dd < trough_value:
                trough_value = current_dd
                trough = row["date"]
        elif current_dd >= 0 and in_dd:
            rows.append({"candidate": name, "start": start, "trough": trough, "end": row["date"], "drawdown": trough_value, "duration": duration})
            in_dd = False
    if in_dd:
        rows.append({"candidate": name, "start": start, "trough": trough, "end": pd.NaT, "drawdown": trough_value, "duration": duration})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame([{"candidate": name, "drawdowns_gt_10": 0, "drawdowns_gt_15": 0, "worst_drawdown": 0.0, "average_drawdown_duration": 0.0, "max_underwater_duration": 0.0}])
    return pd.DataFrame(
        [
            {
                "candidate": name,
                "drawdowns_gt_10": int((out["drawdown"] <= -0.10).sum()),
                "drawdowns_gt_15": int((out["drawdown"] <= -0.15).sum()),
                "worst_drawdown": float(out["drawdown"].min()),
                "average_drawdown_duration": float(out["duration"].mean()),
                "max_underwater_duration": float(out["duration"].max()),
            }
        ]
    )


def _yearly(name: str, daily: pd.DataFrame) -> pd.DataFrame:
    data = _dates(daily).sort_values("date").copy()
    data["year"] = data["date"].dt.year
    rows = []
    for year, group in data.groupby("year"):
        metrics = _metrics(name, group)
        rows.append({"candidate": name, "year": int(year), "yearly_return": metrics.get("total_return", np.nan), "yearly_Sharpe": metrics.get("Sharpe", np.nan)})
    return pd.DataFrame(rows)


def _rolling(name: str, daily: pd.DataFrame, spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    data = _dates(daily).sort_values("date").copy()
    data = data[["date", "return"]].rename(columns={"return": "candidate_return"})
    data = data.merge(spy[["date", "return"]].rename(columns={"return": "SPY_return"}), on="date", how="left")
    data = data.merge(qqq[["date", "return"]].rename(columns={"return": "QQQ_return"}), on="date", how="left")
    window = 52
    ppy = _ppy(data["date"])
    data["rolling_12m_return"] = (1.0 + data["candidate_return"]).rolling(window, min_periods=20).apply(np.prod, raw=True) - 1.0
    data["rolling_12m_SPY_return"] = (1.0 + data["SPY_return"].fillna(0.0)).rolling(window, min_periods=20).apply(np.prod, raw=True) - 1.0
    data["rolling_12m_QQQ_return"] = (1.0 + data["QQQ_return"].fillna(0.0)).rolling(window, min_periods=20).apply(np.prod, raw=True) - 1.0
    mean = data["candidate_return"].rolling(window, min_periods=20).mean() * ppy
    vol = data["candidate_return"].rolling(window, min_periods=20).std(ddof=0) * np.sqrt(ppy)
    data["rolling_12m_Sharpe"] = mean / vol.replace(0, np.nan)
    data["rolling_12m_excess_vs_SPY"] = data["rolling_12m_return"] - data["rolling_12m_SPY_return"]
    data["rolling_12m_excess_vs_QQQ"] = data["rolling_12m_return"] - data["rolling_12m_QQQ_return"]
    data["candidate"] = name
    return data


def run_growth_head_to_head() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = _candidate_daily()
    dates = next(iter(candidates.values()))["date"]
    spy = _benchmark_daily("SPY", dates)
    qqq = _benchmark_daily("QQQ", dates)
    champion = _production_champion_daily(dates)
    tp_sl = _tp_sl()

    result_rows = []
    daily_rows = []
    drawdown_rows = []
    yearly_rows = []
    rolling_rows = []
    for name, daily in candidates.items():
        metrics = _metrics(name, daily)
        metrics.update(tp_sl)
        result_rows.append(metrics)
        daily_rows.append(daily.assign(candidate=name))
        drawdown_rows.append(_drawdown_summary(name, daily))
        yearly_rows.append(_yearly(name, daily))
        rolling_rows.append(_rolling(name, daily, spy, qqq))

    for bench_name, frame in {"SPY": spy, "QQQ": qqq, "current_production_champion": champion}.items():
        metrics = _metrics(bench_name, frame)
        result_rows.append(metrics)
        daily_rows.append(frame.assign(candidate=bench_name))
        drawdown_rows.append(_drawdown_summary(bench_name, frame))
        yearly_rows.append(_yearly(bench_name, frame))

    results = pd.DataFrame(result_rows)
    daily_out = pd.concat(daily_rows, ignore_index=True, sort=False)
    drawdowns = pd.concat(drawdown_rows, ignore_index=True, sort=False)
    yearly = pd.concat(yearly_rows, ignore_index=True, sort=False)
    rolling = pd.concat(rolling_rows, ignore_index=True, sort=False)

    a = results[results["candidate"].eq("growth_v1")].iloc[0]
    b = results[results["candidate"].eq("growth_v1_exposure_cap_60")].iloc[0]
    b_dominates = (
        b["Sharpe"] > a["Sharpe"]
        and b["CAGR"] < a["CAGR"]
        and b["Sortino"] > a["Sortino"]
        and b["Calmar"] < a["Calmar"]
        and b["max_drawdown"] > a["max_drawdown"]
    )
    a_dominates = (
        a["Sharpe"] >= b["Sharpe"]
        and a["CAGR"] >= b["CAGR"]
        and a["Sortino"] >= b["Sortino"]
        and a["Calmar"] >= b["Calmar"]
        and a["max_drawdown"] >= b["max_drawdown"] - 0.02
    )
    if a_dominates:
        classification = "growth_champion_v2"
        champion_name = "growth_v1"
        reason = "growth_v1 dominates or nearly matches risk while preserving higher CAGR and Calmar."
    elif b_dominates:
        classification = "growth_champion_v2"
        champion_name = "growth_v1_exposure_cap_60"
        reason = "cap 60 improves risk-adjusted metrics enough to become champion despite lower CAGR."
    else:
        classification = "keep_both_in_paper_testing"
        champion_name = ""
        reason = "No candidate dominates across Sharpe, CAGR, Sortino and Calmar without tradeoffs."
    governance = pd.DataFrame(
        [
            {
                "classification": classification,
                "growth_champion_v2": champion_name,
                "production_changed": False,
                "parameter_tuning": False,
                "reason": reason,
            }
        ]
    )

    results.to_csv(OUT_RESULTS, index=False)
    daily_out.to_csv(OUT_DAILY, index=False)
    drawdowns.to_csv(OUT_DRAWDOWNS, index=False)
    yearly.to_csv(OUT_YEARLY, index=False)
    rolling.to_csv(OUT_ROLLING, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== GROWTH HEAD TO HEAD =====")
    print(results.to_string(index=False))
    print("\n===== CANDIDATE A =====")
    print(results[results["candidate"].eq("growth_v1")].T.to_string(header=False))
    print("\n===== CANDIDATE B =====")
    print(results[results["candidate"].eq("growth_v1_exposure_cap_60")].T.to_string(header=False))
    print("\n===== BENCHMARK COMPARISON =====")
    print(results[results["candidate"].isin(["SPY", "QQQ", "current_production_champion"])].to_string(index=False))
    print("\n===== DRAWDOWN COMPARISON =====")
    print(drawdowns.to_string(index=False))
    print("\n===== ROLLING PERFORMANCE =====")
    print(rolling.groupby("candidate")[["rolling_12m_Sharpe", "rolling_12m_excess_vs_SPY", "rolling_12m_excess_vs_QQQ"]].agg(["mean", "min"]).to_string())
    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_RESULTS).resolve()}")
    print(f"Saved: {Path(OUT_DAILY).resolve()}")
    print(f"Saved: {Path(OUT_DRAWDOWNS).resolve()}")
    print(f"Saved: {Path(OUT_YEARLY).resolve()}")
    print(f"Saved: {Path(OUT_ROLLING).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return results, drawdowns, governance


if __name__ == "__main__":
    run_growth_head_to_head()
