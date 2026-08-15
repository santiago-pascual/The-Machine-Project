from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXIT_DAILY_FILE = "exit_rule_walk_forward_daily_returns.csv"
EXIT_TRADES_FILE = "exit_rule_walk_forward_trades.csv"
EXIT_RESULTS_FILE = "exit_rule_walk_forward_results.csv"
RAW_DAILY_FILE = "raw_target_2020_daily_returns.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"

OUT_RESULTS = "exit_rule_drawdown_guard_results.csv"
OUT_DAILY = "exit_rule_drawdown_guard_daily_returns.csv"
OUT_TRADES = "exit_rule_drawdown_guard_trades.csv"
OUT_GOVERNANCE = "exit_rule_drawdown_guard_governance.csv"

BASE_VARIANTS = ["soft_exit_rule", "winner_retention_rule", "turnover_penalty_overlay"]
TRADING_DAYS = 252


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


def _metrics(variant: str, daily: pd.DataFrame) -> dict:
    if daily.empty or "guarded_return" not in daily.columns:
        return {"variant": variant}
    data = _dates(daily).sort_values("date")
    r = _num(data["guarded_return"]).dropna()
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
        "observations": len(r),
        "total_return": total,
        "CAGR": float((1.0 + total) ** (1.0 / years) - 1.0),
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": _sortino(r, ppy),
        "Calmar": np.nan if not np.isfinite(mdd) or mdd >= 0 else float(((1.0 + total) ** (1.0 / years) - 1.0) / abs(mdd)),
        "max_drawdown": mdd,
        "hit_rate": float((r > 0).mean()),
        "time_in_defensive_mode": float((data["defensive_mode"] != "normal").mean()) if "defensive_mode" in data.columns else np.nan,
        "average_cash": float(_num(data.get("cash_weight", pd.Series(index=data.index, dtype=float))).mean()),
        "average_exposure": float(_num(data.get("exposure_multiplier", pd.Series(index=data.index, dtype=float))).mean()),
        "turnover": float(_num(data.get("turnover", pd.Series(index=data.index, dtype=float))).mean()),
    }


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
    spy["guarded_return"] = _num(spy["current_price"]).pct_change()
    return spy.dropna(subset=["guarded_return"])


def _raw_baseline_metrics() -> dict:
    raw = _dates(_read_csv(RAW_DAILY_FILE))
    if raw.empty:
        return {}
    if "guarded_return" not in raw.columns:
        raw["guarded_return"] = _num(raw.get("return", raw.get("portfolio_return", pd.Series(index=raw.index, dtype=float))))
    return _metrics("raw_target_research", raw)


def _vol_threshold(all_daily: pd.DataFrame) -> float:
    raw = all_daily[all_daily["variant"].eq("raw_target_research")].copy()
    if raw.empty:
        raw = all_daily.copy()
    vol = _num(raw["return"]).rolling(12, min_periods=4).std()
    threshold = vol.quantile(0.75)
    return float(threshold) if np.isfinite(threshold) else np.inf


def _apply_guard(data: pd.DataFrame, vol_threshold: float, variant_suffix: str) -> pd.DataFrame:
    rows = []
    equity = 1.0
    peak = 1.0
    exposure = 1.0
    prior_returns: list[float] = []
    for _, row in data.sort_values("date").iterrows():
        prior_dd = equity / peak - 1.0
        prior_vol = np.std(prior_returns[-12:], ddof=0) if len(prior_returns) >= 4 else 0.0
        reason = []
        if prior_dd <= -0.12:
            exposure = 0.20
            reason.append("dd_below_minus_12")
        elif prior_dd <= -0.08:
            exposure = 0.50
            reason.append("dd_below_minus_8")
        elif prior_dd > -0.05:
            exposure = min(1.0, exposure + 0.25)
            reason.append("recovery_reenable")
        if prior_vol > vol_threshold:
            exposure = min(exposure, 0.50)
            reason.append("high_vol_stop")
        raw_return = float(row["return"])
        guarded_return = raw_return * exposure
        equity *= 1.0 + guarded_return
        peak = max(peak, equity)
        prior_returns.append(raw_return)
        defensive_mode = "normal"
        if exposure <= 0.25:
            defensive_mode = "high_cash"
        elif exposure <= 0.50:
            defensive_mode = "reduced_exposure"
        out = row.to_dict()
        out.update({
            "variant": f"{row['variant']}+drawdown_guard",
            "base_variant": row["variant"],
            "raw_variant_return": raw_return,
            "guarded_return": guarded_return,
            "equity_after": equity,
            "prior_drawdown_used": prior_dd,
            "prior_volatility_used": prior_vol,
            "exposure_multiplier": exposure,
            "cash_weight": 1.0 - exposure,
            "defensive_mode": defensive_mode,
            "guard_reason": ",".join(reason) if reason else "normal",
            "uses_hindsight": False,
            "variant_suffix": variant_suffix,
        })
        rows.append(out)
    return pd.DataFrame(rows)


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


def _tp_sl(guarded_trades: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if guarded_trades.empty or labels.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    data = guarded_trades[["date", "ticker", "variant"]].merge(labels[["date", "ticker", "label"]], on=["date", "ticker"], how="left")
    data = data.dropna(subset=["label"])
    if data.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((data["label"] == 1).mean())
    sl = float((data["label"] == -1).mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _guard_trades(trades: pd.DataFrame, guarded_daily: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or guarded_daily.empty:
        return pd.DataFrame()
    map_cols = ["date", "base_variant", "variant", "exposure_multiplier", "cash_weight", "defensive_mode", "guard_reason"]
    merged = trades.merge(guarded_daily[map_cols], left_on=["date", "variant"], right_on=["date", "base_variant"], how="inner", suffixes=("", "_guarded"))
    merged["variant"] = merged["variant_guarded"]
    merged["guarded_weight_proxy"] = _num(merged.get("weight_proxy", pd.Series(index=merged.index, dtype=float))).fillna(0.0) * _num(merged["exposure_multiplier"]).fillna(0.0)
    return merged.drop(columns=[c for c in ["variant_guarded", "base_variant"] if c in merged.columns])


def _governance(results: pd.DataFrame, raw_metrics: dict, spy_metrics: dict) -> pd.DataFrame:
    rows = []
    raw_sharpe = raw_metrics.get("Sharpe", np.nan)
    raw_dd = raw_metrics.get("max_drawdown", np.nan)
    spy_return = spy_metrics.get("total_return", np.nan)
    for _, row in results.iterrows():
        classification = "research only"
        reasons = []
        if row.get("max_drawdown", 0) < -0.20:
            classification = "reject"
            reasons.append("DD remains above -20%")
        if row.get("Sharpe", -np.inf) < raw_sharpe:
            classification = "reject"
            reasons.append("Sharpe falls below raw target")
        if row.get("total_return", -np.inf) < spy_return:
            classification = "reject"
            reasons.append("return collapses below SPY")
        if row.get("time_in_defensive_mode", 0) > 0.50:
            classification = "reject"
            reasons.append("defensive mode active too often")
        if not reasons:
            if row.get("Sharpe", -np.inf) > raw_sharpe and row.get("max_drawdown", -np.inf) >= raw_dd - 0.03 and row.get("total_return", -np.inf) > spy_return:
                classification = "candidate for shadow mode"
                reasons.append("improves Sharpe with near-raw drawdown and beats SPY return")
            else:
                reasons.append("improves some metrics but needs more validation")
        rows.append({
            "variant": row["variant"],
            "classification": classification,
            "reason": "; ".join(reasons),
            "production_change": "none",
        })
    return pd.DataFrame(rows)


def run_exit_rule_drawdown_guard() -> dict[str, pd.DataFrame]:
    daily = _dates(_read_csv(EXIT_DAILY_FILE))
    trades = _dates(_read_csv(EXIT_TRADES_FILE))
    if daily.empty:
        raise ValueError("exit_rule_walk_forward_daily_returns.csv is required.")
    daily["return"] = _num(daily["return"])
    test_daily = daily[daily["variant"].isin(BASE_VARIANTS)].copy()
    threshold = _vol_threshold(daily)

    guarded_frames = []
    for variant in BASE_VARIANTS:
        data = test_daily[test_daily["variant"].eq(variant)].copy()
        guarded_frames.append(_apply_guard(data, threshold, "drawdown_guard"))
    guarded_daily = pd.concat(guarded_frames, ignore_index=True, sort=False)
    guarded_trades = _guard_trades(trades, guarded_daily)
    labels = _labels()

    raw_metrics = _raw_baseline_metrics()
    spy_metrics = _metrics("SPY_buy_hold", _spy_daily())
    rows = []
    for variant, data in guarded_daily.groupby("variant"):
        result = _metrics(variant, data)
        result.update(_tp_sl(guarded_trades[guarded_trades["variant"].eq(variant)], labels))
        result["return_vs_SPY"] = result.get("total_return", np.nan) - spy_metrics.get("total_return", np.nan)
        result["Sharpe_vs_SPY"] = result.get("Sharpe", np.nan) - spy_metrics.get("Sharpe", np.nan)
        result["DD_vs_SPY"] = result.get("max_drawdown", np.nan) - spy_metrics.get("max_drawdown", np.nan)
        result["volatility_stop_threshold"] = threshold
        result["uses_hindsight"] = False
        rows.append(result)
    results = pd.DataFrame(rows)
    governance = _governance(results, raw_metrics, spy_metrics)

    results.to_csv(OUT_RESULTS, index=False)
    guarded_daily.to_csv(OUT_DAILY, index=False)
    guarded_trades.to_csv(OUT_TRADES, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== EXIT RULES WITH DRAWDOWN GUARD =====")
    print(results.to_string(index=False))
    print("\n===== DRAWDOWN GUARD PERFORMANCE =====")
    show = ["variant", "total_return", "CAGR", "volatility", "Sharpe", "max_drawdown", "time_in_defensive_mode", "average_cash", "average_exposure", "turnover", "return_vs_SPY", "Sharpe_vs_SPY", "DD_vs_SPY"]
    print(results[[c for c in show if c in results.columns]].to_string(index=False))
    print("\n===== DEFENSIVE MODE ANALYSIS =====")
    defensive = guarded_daily.groupby(["variant", "defensive_mode"]).size().reset_index(name="periods")
    defensive["share"] = defensive["periods"] / defensive.groupby("variant")["periods"].transform("sum")
    print(defensive.to_string(index=False))
    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_RESULTS).resolve()}")
    print(f"Saved: {Path(OUT_DAILY).resolve()}")
    print(f"Saved: {Path(OUT_TRADES).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return {"results": results, "daily": guarded_daily, "trades": guarded_trades, "governance": governance}


if __name__ == "__main__":
    run_exit_rule_drawdown_guard()
