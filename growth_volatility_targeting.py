from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXIT_DAILY_FILE = "exit_rule_walk_forward_daily_returns.csv"
EXIT_TRADES_FILE = "exit_rule_walk_forward_trades.csv"
RAW_DAILY_FILE = "raw_target_2020_daily_returns.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"

OUT_RESULTS = "growth_volatility_targeting_results.csv"
OUT_DAILY = "growth_volatility_targeting_daily_returns.csv"
OUT_EXPOSURE = "growth_volatility_targeting_exposure.csv"
OUT_GOVERNANCE = "growth_volatility_targeting_governance.csv"

BASE_VARIANTS = [
    "raw_target_research",
    "soft_exit_rule",
    "winner_retention_rule",
    "turnover_penalty_overlay",
]
TARGET_VOLS = [0.15, 0.18, 0.20, 0.22]
MIN_EXPOSURE = 0.40
MAX_EXPOSURE = 1.00
MAX_EXPOSURE_CHANGE = 0.15
ROLLING_WINDOW = 12


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
    step = np.median(dates.diff().dt.days.dropna())
    return float(365.25 / step) if np.isfinite(step) and step > 0 else 52.0


def _max_drawdown(returns: pd.Series) -> float:
    r = _num(returns).dropna()
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _sortino(returns: pd.Series, ppy: float) -> float:
    r = _num(returns).dropna()
    downside = r[r < 0].std(ddof=0)
    if r.empty or not np.isfinite(downside) or downside <= 0:
        return np.nan
    return float((r.mean() * ppy) / (downside * np.sqrt(ppy)))


def _metrics(variant: str, daily: pd.DataFrame, return_col: str = "vol_target_return") -> dict:
    if daily.empty or return_col not in daily.columns:
        return {"variant": variant}
    data = _dates(daily).sort_values("date")
    r = _num(data[return_col]).dropna()
    if r.empty:
        return {"variant": variant}
    ppy = _periods_per_year(data["date"])
    total = float((1.0 + r).prod() - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(r) / ppy, 1e-9)
    vol = float(r.std(ddof=0) * np.sqrt(ppy))
    sharpe = np.nan if vol <= 0 else float((r.mean() * ppy) / vol)
    mdd = _max_drawdown(r)
    return {
        "variant": variant,
        "start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "observations": int(len(r)),
        "total_return": total,
        "CAGR": float((1.0 + total) ** (1.0 / years) - 1.0),
        "realized_volatility": vol,
        "Sharpe": sharpe,
        "Sortino": _sortino(r, ppy),
        "Calmar": np.nan if not np.isfinite(mdd) or mdd >= 0 else float(((1.0 + total) ** (1.0 / years) - 1.0) / abs(mdd)),
        "max_drawdown": mdd,
        "hit_rate": float((r > 0).mean()),
        "average_exposure": float(_num(data.get("target_exposure", pd.Series(index=data.index, dtype=float))).mean()),
        "min_exposure": float(_num(data.get("target_exposure", pd.Series(index=data.index, dtype=float))).min()),
        "max_exposure": float(_num(data.get("target_exposure", pd.Series(index=data.index, dtype=float))).max()),
        "time_below_50pct_exposure": float((_num(data.get("target_exposure", pd.Series(index=data.index, dtype=float))) < 0.50).mean()),
        "turnover": float(_num(data.get("turnover", pd.Series(index=data.index, dtype=float))).mean()),
    }


def _base_daily() -> pd.DataFrame:
    exit_daily = _dates(_read_csv(EXIT_DAILY_FILE))
    raw_daily = _dates(_read_csv(RAW_DAILY_FILE))
    frames = []
    if not exit_daily.empty:
        exit_daily["return"] = _num(exit_daily.get("return", pd.Series(index=exit_daily.index, dtype=float)))
        frames.append(exit_daily[exit_daily["variant"].isin([v for v in BASE_VARIANTS if v != "raw_target_research"])].copy())
    if not raw_daily.empty:
        raw = raw_daily.copy()
        raw["variant"] = "raw_target_research"
        raw["return"] = _num(raw.get("return", raw.get("portfolio_return", pd.Series(index=raw.index, dtype=float))))
        if "turnover" not in raw.columns:
            raw["turnover"] = np.nan
        if "selected_count" not in raw.columns:
            raw["selected_count"] = np.nan
        frames.append(raw[["date", "variant", "return", "turnover", "selected_count", "selected_tickers"]].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["return"]).sort_values(["variant", "date"])


def _spy_daily() -> pd.DataFrame:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty:
        return pd.DataFrame()
    if "model_mode" in snaps.columns:
        baseline = snaps[snaps["model_mode"].eq("baseline")]
        if not baseline.empty:
            snaps = baseline.copy()
    spy = snaps[snaps["ticker"].eq("SPY")].drop_duplicates("date").sort_values("date").copy()
    if spy.empty:
        return spy
    spy["vol_target_return"] = _num(spy["current_price"]).pct_change()
    return spy.dropna(subset=["vol_target_return"])


def _apply_vol_target(data: pd.DataFrame, target_vol: float) -> pd.DataFrame:
    data = data.sort_values("date").copy()
    ppy = _periods_per_year(data["date"])
    trailing_vol = _num(data["return"]).rolling(ROLLING_WINDOW, min_periods=4).std().shift(1) * np.sqrt(ppy)
    raw_exposure = (target_vol / trailing_vol).replace([np.inf, -np.inf], np.nan).clip(MIN_EXPOSURE, MAX_EXPOSURE)
    exposures = []
    previous = 1.0
    for value in raw_exposure:
        target = previous if not np.isfinite(value) else float(value)
        target = min(MAX_EXPOSURE, max(MIN_EXPOSURE, target))
        change = min(MAX_EXPOSURE_CHANGE, max(-MAX_EXPOSURE_CHANGE, target - previous))
        smoothed = min(MAX_EXPOSURE, max(MIN_EXPOSURE, previous + change))
        exposures.append(smoothed)
        previous = smoothed
    out = data.copy()
    out["target_volatility"] = target_vol
    out["rolling_vol_used"] = trailing_vol
    out["raw_target_exposure"] = raw_exposure
    out["target_exposure"] = exposures
    out["cash_weight"] = 1.0 - out["target_exposure"]
    out["vol_target_return"] = _num(out["return"]) * out["target_exposure"]
    out["vol_target_variant"] = out["variant"] + "_vol_target_" + str(int(target_vol * 100)) + "pct"
    out["uses_hindsight"] = False
    return out


def _labels() -> pd.DataFrame:
    labels = _dates(_read_csv(LABELS_FILE))
    if labels.empty:
        return labels
    if "model_mode" in labels.columns:
        baseline = labels[labels["model_mode"].eq("baseline")]
        if not baseline.empty:
            labels = baseline.copy()
    if "horizon" in labels.columns:
        labels = labels[labels["horizon"].eq(20)].copy()
    return labels.drop_duplicates(["date", "ticker"])


def _trades() -> pd.DataFrame:
    trades = _dates(_read_csv(EXIT_TRADES_FILE))
    if trades.empty:
        return trades
    return trades[trades["variant"].isin(BASE_VARIANTS)].copy()


def _tp_sl_for_variant(variant: str, trades: pd.DataFrame, labels: pd.DataFrame) -> dict:
    base_variant = variant.rsplit("_vol_target_", 1)[0]
    if trades.empty or labels.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    data = trades[trades["variant"].eq(base_variant)][["date", "ticker"]].merge(labels[["date", "ticker", "label"]], on=["date", "ticker"], how="left")
    data = data.dropna(subset=["label"])
    if data.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((data["label"] == 1).mean())
    sl = float((data["label"] == -1).mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _raw_target_metrics() -> dict:
    raw = _base_daily()
    raw = raw[raw["variant"].eq("raw_target_research")].copy()
    if raw.empty:
        return {}
    raw["vol_target_return"] = raw["return"]
    raw["target_exposure"] = 1.0
    return _metrics("raw_target_research", raw)


def _governance(results: pd.DataFrame, raw_metrics: dict, spy_metrics: dict) -> pd.DataFrame:
    rows = []
    raw_sharpe = raw_metrics.get("Sharpe", np.nan)
    spy_return = spy_metrics.get("total_return", np.nan)
    spy_sharpe = spy_metrics.get("Sharpe", np.nan)
    spy_dd = spy_metrics.get("max_drawdown", np.nan)
    for _, row in results.iterrows():
        classification = "research only"
        reasons = []
        if row.get("uses_hindsight", False):
            classification = "reject"
            reasons.append("uses hindsight")
        if row.get("Sharpe", -np.inf) < raw_sharpe:
            classification = "reject"
            reasons.append("Sharpe below raw target")
        if row.get("total_return", -np.inf) < spy_return:
            classification = "reject"
            reasons.append("return collapses below SPY")
        if row.get("max_drawdown", -np.inf) < -0.30:
            classification = "reject"
            reasons.append("DD still too high")
        if row.get("average_exposure", 1.0) < 0.55 and row.get("total_return", -np.inf) <= raw_metrics.get("total_return", -np.inf):
            classification = "reject"
            reasons.append("improvement only from de-risking")
        if not reasons:
            if (
                row.get("total_return", -np.inf) > spy_return
                and row.get("Sharpe", -np.inf) > spy_sharpe
                and row.get("max_drawdown", -np.inf) > spy_dd
                and row.get("time_below_50pct_exposure", 1.0) <= 0.50
            ):
                classification = "candidate for shadow mode"
                reasons.append("beats SPY return/Sharpe/DD without excessive defensiveness")
            else:
                reasons.append("needs more validation")
        rows.append({
            "variant": row["variant"],
            "classification": classification,
            "reason": "; ".join(reasons),
            "production_change": "none",
        })
    return pd.DataFrame(rows)


def run_growth_volatility_targeting() -> dict[str, pd.DataFrame]:
    base = _base_daily()
    if base.empty:
        raise ValueError("exit_rule_walk_forward_daily_returns.csv or raw_target_2020_daily_returns.csv is required.")
    frames = []
    for variant in BASE_VARIANTS:
        variant_data = base[base["variant"].eq(variant)].copy()
        if variant_data.empty:
            continue
        for target_vol in TARGET_VOLS:
            frames.append(_apply_vol_target(variant_data, target_vol))
    daily = pd.concat(frames, ignore_index=True, sort=False)
    daily["variant"] = daily["vol_target_variant"]

    labels = _labels()
    trades = _trades()
    spy_metrics = _metrics("SPY_buy_hold", _spy_daily())
    raw_metrics = _raw_target_metrics()
    rows = []
    for variant, data in daily.groupby("variant"):
        result = _metrics(variant, data)
        result.update(_tp_sl_for_variant(variant, trades, labels))
        result["return_vs_SPY"] = result.get("total_return", np.nan) - spy_metrics.get("total_return", np.nan)
        result["Sharpe_vs_SPY"] = result.get("Sharpe", np.nan) - spy_metrics.get("Sharpe", np.nan)
        result["DD_vs_SPY"] = result.get("max_drawdown", np.nan) - spy_metrics.get("max_drawdown", np.nan)
        result["uses_hindsight"] = False
        rows.append(result)
    results = pd.DataFrame(rows).sort_values(["Sharpe", "total_return"], ascending=False)
    governance = _governance(results, raw_metrics, spy_metrics)
    exposure = daily[["date", "variant", "target_volatility", "rolling_vol_used", "raw_target_exposure", "target_exposure", "cash_weight", "vol_target_return", "uses_hindsight"]].copy()

    results.to_csv(OUT_RESULTS, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    exposure.to_csv(OUT_EXPOSURE, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== GROWTH VOLATILITY TARGETING =====")
    print(results.to_string(index=False))
    print("\n===== VOL TARGET VARIANT COMPARISON =====")
    show = ["variant", "total_return", "CAGR", "realized_volatility", "Sharpe", "max_drawdown", "average_exposure", "min_exposure", "max_exposure", "time_below_50pct_exposure", "return_vs_SPY", "Sharpe_vs_SPY", "DD_vs_SPY"]
    print(results[[c for c in show if c in results.columns]].to_string(index=False))
    print("\n===== EXPOSURE ANALYSIS =====")
    exposure_summary = exposure.groupby("variant").agg(
        avg_exposure=("target_exposure", "mean"),
        min_exposure=("target_exposure", "min"),
        max_exposure=("target_exposure", "max"),
        periods_below_50=("target_exposure", lambda x: float((x < 0.50).mean())),
    ).reset_index()
    print(exposure_summary.to_string(index=False))
    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_RESULTS).resolve()}")
    print(f"Saved: {Path(OUT_DAILY).resolve()}")
    print(f"Saved: {Path(OUT_EXPOSURE).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return {"results": results, "daily": daily, "exposure": exposure, "governance": governance}


if __name__ == "__main__":
    run_growth_volatility_targeting()
