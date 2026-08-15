from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DAILY_FILE = "raw_target_2020_daily_returns.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"
RAW_TRADES_FILE = "raw_target_research_backtest_trades.csv"

OUT_RESULTS = "exit_rule_walk_forward_results.csv"
OUT_DAILY = "exit_rule_walk_forward_daily_returns.csv"
OUT_TRADES = "exit_rule_walk_forward_trades.csv"
OUT_GOVERNANCE = "exit_rule_walk_forward_governance.csv"

VARIANTS = [
    "raw_target_research",
    "winner_retention_rule",
    "soft_exit_rule",
    "turnover_penalty_overlay",
]

MAX_POSITIONS = 4


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
    if daily.empty or "return" not in daily.columns:
        return {"variant": variant}
    data = _dates(daily).sort_values("date")
    r = _num(data["return"]).dropna()
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
        "turnover": float(_num(data.get("turnover", pd.Series(index=data.index, dtype=float))).mean()),
        "average_selected_count": float(_num(data.get("selected_count", pd.Series(index=data.index, dtype=float))).mean()),
    }


def _raw_daily() -> pd.DataFrame:
    df = _dates(_read_csv(RAW_DAILY_FILE)).sort_values("date")
    if df.empty:
        return df
    if "return" not in df.columns and "portfolio_return" in df.columns:
        df["return"] = _num(df["portfolio_return"])
    return df


def _selected_from_text(value: object) -> list[str]:
    return [t.strip() for t in str(value).split(",") if t.strip() and t.strip().lower() != "nan"]


def _raw_selected_by_date(raw_daily: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    return {row["date"]: _selected_from_text(row.get("selected_tickers", "")) for _, row in raw_daily.iterrows()}


def _snapshots() -> pd.DataFrame:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty:
        return snaps
    if "model_mode" in snaps.columns:
        baseline = snaps[snaps["model_mode"].eq("baseline")]
        if not baseline.empty:
            snaps = baseline.copy()
    for col in ["expected_daily_return", "signal_strength", "current_price", "quality_score"]:
        if col in snaps.columns:
            snaps[col] = _num(snaps[col])
    snaps = snaps.sort_values(["ticker", "date"])
    snaps["momentum_20d_proxy"] = snaps.groupby("ticker")["current_price"].pct_change(4)
    snaps["expected_return_rank_pct"] = snaps.groupby("date")["expected_daily_return"].rank(pct=True, ascending=False)
    return snaps.sort_values(["date", "expected_daily_return"], ascending=[True, False])


def _realized() -> pd.DataFrame:
    realized = _dates(_read_csv(REALIZED_FILE))
    if realized.empty:
        return realized
    if "model_mode" in realized.columns:
        baseline = realized[realized["model_mode"].eq("baseline")]
        if not baseline.empty:
            realized = baseline.copy()
    for col in ["realized_return_5d", "realized_return_10d", "realized_return_20d"]:
        if col in realized.columns:
            realized[col] = _num(realized[col])
    return realized.drop_duplicates(["date", "ticker"])


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


def _spy_daily() -> pd.DataFrame:
    snaps = _snapshots()
    if snaps.empty:
        return pd.DataFrame()
    spy = snaps[snaps["ticker"].eq("SPY")].drop_duplicates("date").sort_values("date").copy()
    spy["return"] = _num(spy["current_price"]).pct_change()
    return spy.dropna(subset=["return"])


def _return_for_selection(date: pd.Timestamp, selection: list[str], realized: pd.DataFrame) -> float:
    if not selection:
        return 0.0
    data = realized[(realized["date"].eq(date)) & (realized["ticker"].isin(selection))]
    if data.empty or "realized_return_5d" not in data.columns:
        return np.nan
    data = data.dropna(subset=["realized_return_5d"])
    if data.empty:
        return np.nan
    return float(data["realized_return_5d"].mean())


def _trade_rows(date: pd.Timestamp, variant: str, selection: list[str], previous: list[str], realized: pd.DataFrame) -> list[dict]:
    rows = []
    data = realized[(realized["date"].eq(date)) & (realized["ticker"].isin(selection))]
    for ticker in selection:
        rr = data[data["ticker"].eq(ticker)]
        rows.append({
            "date": date,
            "variant": variant,
            "ticker": ticker,
            "weight_proxy": 1.0 / len(selection) if selection else 0.0,
            "is_new_position": ticker not in previous,
            "is_retained_position": ticker in previous,
            "realized_return_5d": float(rr["realized_return_5d"].iloc[0]) if not rr.empty and "realized_return_5d" in rr.columns else np.nan,
            "realized_return_20d": float(rr["realized_return_20d"].iloc[0]) if not rr.empty and "realized_return_20d" in rr.columns else np.nan,
        })
    return rows


def _choose_variant_selection(
    variant: str,
    date: pd.Timestamp,
    base_selection: list[str],
    previous_selection: list[str],
    snaps: pd.DataFrame,
) -> list[str]:
    today = snaps[snaps["date"].eq(date)].copy()
    if today.empty:
        return base_selection
    today = today.sort_values(["expected_daily_return", "signal_strength"], ascending=False)
    if variant == "raw_target_research":
        return base_selection
    if variant == "winner_retention_rule":
        top_quintile = set(today[today["expected_return_rank_pct"] <= 0.20]["ticker"])
        retained = [t for t in previous_selection if t in top_quintile]
        return list(dict.fromkeys(base_selection + retained))[:MAX_POSITIONS]
    if variant == "soft_exit_rule":
        positive = set(today[today["expected_daily_return"] > 0]["ticker"])
        retained = [t for t in previous_selection if t in positive]
        return list(dict.fromkeys(base_selection + retained))[:MAX_POSITIONS]
    if variant == "turnover_penalty_overlay":
        selection = list(base_selection)
        selected_data = today[today["ticker"].isin(selection)]
        weakest_new_er = selected_data["expected_daily_return"].min() if not selected_data.empty else -np.inf
        for old in previous_selection:
            if old in selection:
                continue
            old_data = today[today["ticker"].eq(old)]
            if old_data.empty:
                continue
            old_er = float(old_data["expected_daily_return"].iloc[0])
            old_mom = float(old_data["momentum_20d_proxy"].iloc[0]) if "momentum_20d_proxy" in old_data else np.nan
            keep_old = old_er > 0 and (old_er >= weakest_new_er * 0.85 or (np.isfinite(old_mom) and old_mom > 0))
            if keep_old:
                selection.append(old)
        return list(dict.fromkeys(selection))[:MAX_POSITIONS]
    return base_selection


def _average_holding_period(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    periods = []
    for _, group in trades.sort_values("date").groupby(["variant", "ticker"]):
        current = 0
        last_date = None
        for _, row in group.iterrows():
            if bool(row.get("is_retained_position")):
                current += 1
            else:
                if current:
                    periods.append(current)
                current = 1
            last_date = row["date"]
        if current:
            periods.append(current)
    return float(np.mean(periods)) if periods else np.nan


def _missed_winner_capture(trades: pd.DataFrame, realized: pd.DataFrame) -> float:
    if trades.empty or realized.empty:
        return np.nan
    top = realized.sort_values(["date", "realized_return_20d"], ascending=[True, False]).groupby("date").head(5)
    top = top[_num(top["realized_return_20d"]) > 0.05]
    if top.empty:
        return np.nan
    selected_pairs = set(zip(trades["date"], trades["ticker"]))
    hits = sum((row["date"], row["ticker"]) in selected_pairs for _, row in top.iterrows())
    return float(hits / len(top))


def _tp_sl(trades: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if trades.empty or labels.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    data = trades[["date", "ticker"]].merge(labels[["date", "ticker", "label"]], on=["date", "ticker"], how="left")
    data = data.dropna(subset=["label"])
    if data.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((data["label"] == 1).mean())
    sl = float((data["label"] == -1).mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def run_exit_rule_walk_forward_validation() -> dict[str, pd.DataFrame]:
    raw_daily = _raw_daily()
    snaps = _snapshots()
    realized = _realized()
    labels = _labels()
    spy = _spy_daily()
    if raw_daily.empty or snaps.empty or realized.empty:
        raise ValueError("Missing required walk-forward validation inputs.")

    base_selected = _raw_selected_by_date(raw_daily)
    previous_by_variant = {variant: [] for variant in VARIANTS}
    daily_rows = []
    trade_rows = []
    for _, row in raw_daily.iterrows():
        date = row["date"]
        base = base_selected.get(date, [])
        for variant in VARIANTS:
            previous = previous_by_variant.get(variant, [])
            selection = _choose_variant_selection(variant, date, base, previous, snaps)
            if variant == "raw_target_research":
                period_return = float(row.get("return", row.get("portfolio_return", np.nan)))
            else:
                period_return = _return_for_selection(date, selection, realized)
            turnover = len(set(selection).symmetric_difference(set(previous))) / max(len(set(selection).union(previous)), 1)
            daily_rows.append({
                "date": date,
                "variant": variant,
                "return": period_return,
                "selected_tickers": ",".join(selection),
                "selected_count": len(selection),
                "turnover": turnover,
                "no_hindsight": True,
            })
            trade_rows.extend(_trade_rows(date, variant, selection, previous, realized))
            previous_by_variant[variant] = selection

    daily = pd.DataFrame(daily_rows).dropna(subset=["return"])
    trades = pd.DataFrame(trade_rows)
    results = []
    spy_metrics = _metrics("SPY_buy_hold", spy)
    raw_metrics = None
    for variant, data in daily.groupby("variant"):
        result = _metrics(variant, data)
        variant_trades = trades[trades["variant"].eq(variant)]
        result["average_holding_period"] = _average_holding_period(variant_trades)
        result["missed_winner_capture"] = _missed_winner_capture(variant_trades, realized)
        result.update(_tp_sl(variant_trades, labels))
        result["return_vs_SPY"] = result.get("total_return", np.nan) - spy_metrics.get("total_return", np.nan)
        result["Sharpe_vs_SPY"] = result.get("Sharpe", np.nan) - spy_metrics.get("Sharpe", np.nan)
        result["DD_vs_SPY"] = result.get("max_drawdown", np.nan) - spy_metrics.get("max_drawdown", np.nan)
        result["uses_hindsight"] = False
        results.append(result)
        if variant == "raw_target_research":
            raw_metrics = result
    results_df = pd.DataFrame(results)
    governance = _governance(results_df, raw_metrics or {})

    results_df.to_csv(OUT_RESULTS, index=False)
    daily.to_csv(OUT_DAILY, index=False)
    trades.to_csv(OUT_TRADES, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== EXIT RULE WALK-FORWARD VALIDATION =====")
    print(results_df.to_string(index=False))
    print("\n===== RAW TARGET EXIT VARIANT COMPARISON =====")
    show_cols = ["variant", "total_return", "CAGR", "volatility", "Sharpe", "max_drawdown", "turnover", "average_holding_period", "missed_winner_capture", "return_vs_SPY", "Sharpe_vs_SPY", "DD_vs_SPY"]
    print(results_df[[c for c in show_cols if c in results_df.columns]].to_string(index=False))
    print("\n===== WINNER RETENTION WALK-FORWARD =====")
    print(results_df[results_df["variant"].isin(["winner_retention_rule", "soft_exit_rule"])].to_string(index=False))
    print("\n===== TURNOVER PENALTY WALK-FORWARD =====")
    print(results_df[results_df["variant"].eq("turnover_penalty_overlay")].to_string(index=False))
    print("\n===== EXIT RULE GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_RESULTS).resolve()}")
    print(f"Saved: {Path(OUT_DAILY).resolve()}")
    print(f"Saved: {Path(OUT_TRADES).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return {"results": results_df, "daily": daily, "trades": trades, "governance": governance}


def _governance(results: pd.DataFrame, raw_metrics: dict) -> pd.DataFrame:
    rows = []
    raw_sharpe = raw_metrics.get("Sharpe", np.nan)
    raw_dd = raw_metrics.get("max_drawdown", np.nan)
    for _, row in results.iterrows():
        classification = "research only"
        reasons = []
        if bool(row.get("uses_hindsight", False)):
            classification = "reject"
            reasons.append("uses hindsight")
        if row["variant"] == "raw_target_research":
            reasons.append("baseline comparator")
        else:
            if row.get("Sharpe", -np.inf) <= raw_sharpe:
                classification = "reject"
                reasons.append("Sharpe does not improve versus raw target")
            if row.get("max_drawdown", -np.inf) < raw_dd - 0.03:
                classification = "reject"
                reasons.append("drawdown increases materially")
            if row.get("total_return", 0) < raw_metrics.get("total_return", 0) * 0.75:
                classification = "reject"
                reasons.append("turnover falls but return collapses")
            if not reasons:
                if row.get("Sharpe_vs_SPY", -np.inf) > 0 and row.get("max_drawdown", -np.inf) >= raw_dd - 0.03:
                    classification = "candidate for shadow mode"
                    reasons.append("beats raw/SPY Sharpe with acceptable drawdown")
                else:
                    reasons.append("needs more validation")
        rows.append({
            "variant": row["variant"],
            "classification": classification,
            "reason": "; ".join(reasons),
            "production_change": "none",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_exit_rule_walk_forward_validation()
