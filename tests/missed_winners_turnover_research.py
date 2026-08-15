from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DAILY_FILE = "raw_target_2020_daily_returns.csv"
RAW_TRADES_FILE = "raw_target_research_backtest_trades.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"
MISSED_FILE = "spy_missed_winners.csv"
SIZING_FILE = "spy_position_sizing_gap.csv"
SPY_GAP_FILE = "spy_benchmark_gap_analysis.csv"

OUT_MISSED = "missed_winners_attribution.csv"
OUT_RETENTION = "winner_retention_analysis.csv"
OUT_TURNOVER = "turnover_drag_analysis.csv"
OUT_VARIANTS = "missed_winners_turnover_variant_results.csv"
OUT_GOVERNANCE = "missed_winners_turnover_governance.csv"

MAX_POSITIONS = 4
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


def _metrics(name: str, daily: pd.DataFrame, return_col: str = "return") -> dict:
    if daily.empty or return_col not in daily.columns:
        return {"variant": name}
    data = _dates(daily).sort_values("date")
    r = _num(data[return_col]).dropna()
    if r.empty:
        return {"variant": name}
    ppy = _periods_per_year(data["date"])
    total = float((1.0 + r).prod() - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(r) / ppy, 1e-9)
    vol = float(r.std(ddof=0) * np.sqrt(ppy))
    sharpe = np.nan if vol <= 0 else float((r.mean() * ppy) / vol)
    mdd = _max_drawdown(r)
    return {
        "variant": name,
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
        "average_winner": float(r[r > 0].mean()) if (r > 0).any() else np.nan,
        "average_loser": float(r[r < 0].mean()) if (r < 0).any() else np.nan,
    }


def _raw_daily() -> pd.DataFrame:
    df = _dates(_read_csv(RAW_DAILY_FILE)).sort_values("date")
    if df.empty:
        return df
    if "return" not in df.columns and "portfolio_return" in df.columns:
        df["return"] = _num(df["portfolio_return"])
    if "cash" not in df.columns and "cash_proxy" in df.columns:
        df["cash"] = _num(df["cash_proxy"])
    if "exposure" not in df.columns:
        df["exposure"] = 1.0 - _num(df.get("cash", pd.Series(0.0, index=df.index))).fillna(0.0)
    return df


def _selected_map(raw_daily: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    selected: dict[pd.Timestamp, list[str]] = {}
    for _, row in raw_daily.iterrows():
        tickers = [t.strip() for t in str(row.get("selected_tickers", "")).split(",") if t.strip() and t.strip().lower() != "nan"]
        selected[row["date"]] = tickers
    return selected


def _snapshot_data() -> pd.DataFrame:
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if snaps.empty:
        return snaps
    if "model_mode" in snaps.columns:
        baseline = snaps[snaps["model_mode"].eq("baseline")].copy()
        if not baseline.empty:
            snaps = baseline
    snaps["expected_daily_return"] = _num(snaps.get("expected_daily_return", pd.Series(index=snaps.index, dtype=float)))
    snaps["signal_strength"] = _num(snaps.get("signal_strength", pd.Series(index=snaps.index, dtype=float)))
    snaps["current_price"] = _num(snaps.get("current_price", pd.Series(index=snaps.index, dtype=float)))
    snaps = snaps.sort_values(["ticker", "date"])
    snaps["momentum_20d_proxy"] = snaps.groupby("ticker")["current_price"].pct_change(4)
    snaps["momentum_60d_proxy"] = snaps.groupby("ticker")["current_price"].pct_change(12)
    snaps["volatility_proxy"] = (
        snaps.groupby("ticker")["current_price"].pct_change().rolling(12, min_periods=4).std().reset_index(level=0, drop=True)
    )
    snaps["expected_return_rank_pct"] = snaps.groupby("date")["expected_daily_return"].rank(pct=True, ascending=False)
    snaps["signal_rank_pct"] = snaps.groupby("date")["signal_strength"].rank(pct=True, ascending=False)
    snaps["volatility_rank_pct"] = snaps.groupby("date")["volatility_proxy"].rank(pct=True, ascending=False)
    return snaps


def _realized_data() -> pd.DataFrame:
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


def _labels_data() -> pd.DataFrame:
    labels = _dates(_read_csv(LABELS_FILE))
    if labels.empty:
        return labels
    if "model_mode" in labels.columns:
        baseline = labels[labels["model_mode"].eq("baseline")]
        if not baseline.empty:
            labels = baseline.copy()
    return labels


def missed_winners_attribution(raw_daily: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    missed = _dates(_read_csv(MISSED_FILE))
    if missed.empty:
        return pd.DataFrame()
    selected = _selected_map(raw_daily)
    rows = []
    rank_cols = [
        "date",
        "ticker",
        "expected_return_rank_pct",
        "signal_rank_pct",
        "volatility_rank_pct",
        "quality_score",
        "target_confidence",
    ]
    rank_data = snaps[[c for c in rank_cols if c in snaps.columns]].drop_duplicates(["date", "ticker"])
    missed = missed.merge(rank_data, on=["date", "ticker"], how="left", suffixes=("", "_ranked"))
    for _, row in missed.iterrows():
        selected_now = row["ticker"] in selected.get(row["date"], [])
        rows.append(
            {
                "date": row["date"],
                "ticker": row["ticker"],
                "period_return_20d": row.get("winner_forward_return_20d", row.get("realized_return_20d")),
                "was_in_universe": True,
                "was_selected": bool(selected_now),
                "was_underweighted": bool(selected_now and row.get("raw_weight_proxy", 0.0) < 0.20),
                "expected_return_rank_pct": row.get("expected_return_rank_pct"),
                "signal_rank_pct": row.get("signal_rank_pct"),
                "volatility_rank_pct": row.get("volatility_rank_pct"),
                "expected_daily_return": row.get("expected_daily_return"),
                "signal_strength": row.get("signal_strength"),
                "quality_score": row.get("quality_score"),
                "target_confidence": row.get("target_confidence"),
                "filter_status": row.get("miss_reason"),
                "diagnosis": "selected" if selected_now else row.get("miss_reason", "not_selected"),
            }
        )
    return pd.DataFrame(rows)


def winner_retention_analysis(raw_daily: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    selected = _selected_map(raw_daily)
    dates = list(raw_daily["date"])
    rows = []
    for ticker in sorted({t for tickers in selected.values() for t in tickers}):
        active_start = None
        active_dates = []
        for date in dates:
            is_selected = ticker in selected.get(date, [])
            if is_selected and active_start is None:
                active_start = date
                active_dates = [date]
            elif is_selected:
                active_dates.append(date)
            elif active_start is not None:
                exit_date = date
                exit_ret = realized[(realized["date"].eq(exit_date)) & (realized["ticker"].eq(ticker))]
                rows.append(
                    {
                        "ticker": ticker,
                        "entry_date": active_start,
                        "exit_date": exit_date,
                        "holding_periods": len(active_dates),
                        "return_after_exit_5d": float(exit_ret["realized_return_5d"].iloc[0])
                        if not exit_ret.empty and "realized_return_5d" in exit_ret
                        else np.nan,
                        "return_after_exit_20d": float(exit_ret["realized_return_20d"].iloc[0])
                        if not exit_ret.empty and "realized_return_20d" in exit_ret
                        else np.nan,
                        "sold_too_early_proxy": bool(not exit_ret.empty and float(exit_ret["realized_return_20d"].iloc[0]) > 0.05),
                        "exit_reason_proxy": "removed_by_ranking_or_optimizer",
                    }
                )
                active_start = None
                active_dates = []
        if active_start is not None:
            rows.append(
                {
                    "ticker": ticker,
                    "entry_date": active_start,
                    "exit_date": pd.NaT,
                    "holding_periods": len(active_dates),
                    "return_after_exit_5d": np.nan,
                    "return_after_exit_20d": np.nan,
                    "sold_too_early_proxy": False,
                    "exit_reason_proxy": "still_active_at_sample_end",
                }
            )
    return pd.DataFrame(rows)


def turnover_drag_analysis(raw_daily: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    selected = _selected_map(raw_daily)
    dates = list(raw_daily["date"])
    rows = []
    previous: set[str] = set()
    for date in dates:
        current = set(selected.get(date, []))
        sold = sorted(previous - current)
        bought = sorted(current - previous)
        retained = sorted(current & previous)
        sold_forward = []
        for ticker in sold:
            rr = realized[(realized["date"].eq(date)) & (realized["ticker"].eq(ticker))]
            if not rr.empty and "realized_return_20d" in rr:
                sold_forward.append(float(rr["realized_return_20d"].iloc[0]))
        rows.append(
            {
                "date": date,
                "turnover_count": len(sold) + len(bought),
                "sold_count": len(sold),
                "bought_count": len(bought),
                "retained_count": len(retained),
                "sold_tickers": ",".join(sold),
                "bought_tickers": ",".join(bought),
                "avg_return_after_sale_20d": float(np.nanmean(sold_forward)) if sold_forward else np.nan,
                "winner_sold_count": int(sum(np.array(sold_forward) > 0.05)) if sold_forward else 0,
                "whipsaw_proxy": len(set(sold) & set(bought)),
            }
        )
        previous = current
    out = pd.DataFrame(rows)
    if not out.empty:
        out["turnover_rate_proxy"] = out["turnover_count"] / (out["retained_count"] + out["turnover_count"]).replace(0, np.nan)
    return out


def _base_candidates_for_date(date: pd.Timestamp, snaps: pd.DataFrame) -> pd.DataFrame:
    data = snaps[snaps["date"].eq(date)].copy()
    return data.sort_values(["expected_daily_return", "signal_strength"], ascending=False)


def _selected_return(
    date: pd.Timestamp, selected: list[str], realized: pd.DataFrame, cash: float = 0.0, weights: dict[str, float] | None = None
) -> float:
    if not selected:
        return 0.0
    rr = realized[(realized["date"].eq(date)) & (realized["ticker"].isin(selected))]
    if rr.empty or "realized_return_5d" not in rr.columns:
        return np.nan
    rr = rr.dropna(subset=["realized_return_5d"])
    if rr.empty:
        return np.nan
    if weights:
        vals = []
        total_w = 0.0
        for _, row in rr.iterrows():
            w = float(weights.get(row["ticker"], 0.0))
            vals.append(w * float(row["realized_return_5d"]))
            total_w += w
        return float(np.nansum(vals)) if total_w > 0 else np.nan
    exposure = max(0.0, 1.0 - cash)
    return float(rr["realized_return_5d"].mean() * exposure)


def _simulate_variants(
    raw_daily: pd.DataFrame, snaps: pd.DataFrame, realized: pd.DataFrame, missed_attr: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_base = _selected_map(raw_daily)
    dates = list(raw_daily["date"])
    previous_by_variant: dict[str, list[str]] = {}
    daily_rows = []
    variant_names = [
        "raw_target_current",
        "winner_retention_rule",
        "soft_exit_rule",
        "momentum_confirmation_overlay",
        "turnover_penalty_overlay",
        "top_winner_rescue_overlay",
    ]
    for _, raw_row in raw_daily.iterrows():
        date = raw_row["date"]
        base = list(selected_base.get(date, []))
        candidate_data = _base_candidates_for_date(date, snaps)
        top_quintile = candidate_data[candidate_data["expected_return_rank_pct"] <= 0.20]["ticker"].tolist()
        positive = candidate_data[candidate_data["expected_daily_return"] > 0]["ticker"].tolist()
        momentum_ok = candidate_data[
            (candidate_data["expected_daily_return"] > 0)
            & (candidate_data["momentum_20d_proxy"] > 0)
            & (candidate_data["momentum_60d_proxy"] > 0)
        ]["ticker"].tolist()
        winner_rescue = (
            missed_attr[
                (missed_attr["date"].eq(date))
                & (missed_attr["expected_daily_return"] > 0)
                & (~missed_attr["was_selected"].fillna(False).astype(bool))
            ]
            .sort_values("period_return_20d", ascending=False)["ticker"]
            .head(1)
            .tolist()
        )

        variant_selection = {"raw_target_current": base}
        prev = previous_by_variant.get("winner_retention_rule", [])
        retained = [t for t in prev if t in top_quintile]
        variant_selection["winner_retention_rule"] = list(dict.fromkeys(base + retained))[:MAX_POSITIONS]

        prev = previous_by_variant.get("soft_exit_rule", [])
        soft = [t for t in prev if t in positive]
        variant_selection["soft_exit_rule"] = list(dict.fromkeys(base + soft))[:MAX_POSITIONS]

        variant_selection["momentum_confirmation_overlay"] = list(dict.fromkeys(base + momentum_ok))[:MAX_POSITIONS]

        prev = previous_by_variant.get("turnover_penalty_overlay", [])
        replacement = list(base)
        for old in prev:
            if old in replacement:
                continue
            old_row = candidate_data[candidate_data["ticker"].eq(old)]
            new_min = candidate_data[candidate_data["ticker"].isin(replacement)]["expected_daily_return"].min()
            old_er = float(old_row["expected_daily_return"].iloc[0]) if not old_row.empty else -np.inf
            if np.isfinite(old_er) and np.isfinite(new_min) and old_er > 0 and new_min < old_er * 1.25:
                replacement.append(old)
        variant_selection["turnover_penalty_overlay"] = list(dict.fromkeys(replacement))[:MAX_POSITIONS]

        variant_selection["top_winner_rescue_overlay"] = list(dict.fromkeys(base + winner_rescue))[:MAX_POSITIONS]

        for variant in variant_names:
            selection = variant_selection[variant]
            if variant == "raw_target_current":
                ret = float(raw_row.get("return", raw_row.get("portfolio_return", np.nan)))
            else:
                ret = _selected_return(date, selection, realized, cash=0.0)
            prev = previous_by_variant.get(variant, [])
            turnover = len(set(selection).symmetric_difference(set(prev))) / max(len(set(selection).union(prev)), 1)
            daily_rows.append(
                {
                    "date": date,
                    "variant": variant,
                    "return": ret,
                    "selected_tickers": ",".join(selection),
                    "selected_count": len(selection),
                    "turnover": turnover,
                    "validity_warning": "benchmark_hindsight_diagnostic_only"
                    if variant == "top_winner_rescue_overlay"
                    else "research_proxy_no_production_change",
                }
            )
            previous_by_variant[variant] = selection
    variant_daily = pd.DataFrame(daily_rows).dropna(subset=["return"])
    results = []
    labels = _labels_data()
    for variant, data in variant_daily.groupby("variant"):
        result = _metrics(variant, data)
        result["turnover"] = float(_num(data["turnover"]).mean())
        result["average_holding_period_proxy"] = np.nan
        result["missed_winner_capture_rate"] = _capture_rate(data, missed_attr)
        if not labels.empty:
            tp_sl = _tp_sl_rate(data, labels)
            result.update(tp_sl)
        result["return_vs_SPY"] = np.nan
        result["Sharpe_vs_SPY"] = np.nan
        result["DD_vs_SPY"] = np.nan
        result["validity_warning"] = data["validity_warning"].iloc[0]
        results.append(result)
    return pd.DataFrame(results), variant_daily


def _capture_rate(variant_daily: pd.DataFrame, missed_attr: pd.DataFrame) -> float:
    if variant_daily.empty or missed_attr.empty:
        return np.nan
    selected = _selected_map(variant_daily)
    major = missed_attr[_num(missed_attr["period_return_20d"]) > 0.05]
    if major.empty:
        return np.nan
    hits = 0
    for _, row in major.iterrows():
        hits += int(row["ticker"] in selected.get(row["date"], []))
    return float(hits / len(major))


def _tp_sl_rate(variant_daily: pd.DataFrame, labels: pd.DataFrame) -> dict:
    selected = _selected_map(variant_daily)
    rows = []
    horizon = labels[labels["horizon"].eq(20)].copy() if "horizon" in labels.columns else labels.copy()
    for date, tickers in selected.items():
        rows.append(horizon[(horizon["date"].eq(date)) & (horizon["ticker"].isin(tickers))])
    if not rows:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    data = pd.concat(rows, ignore_index=True)
    if data.empty or "label" not in data.columns:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((data["label"] == 1).mean())
    sl = float((data["label"] == -1).mean())
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl}


def _add_spy_comparison(results: pd.DataFrame) -> pd.DataFrame:
    spy_daily = _benchmark_daily("SPY")
    if spy_daily.empty:
        return results
    spy_row = _metrics("SPY_buy_hold", spy_daily)
    out = results.copy()
    out["return_vs_SPY"] = out["total_return"] - float(spy_row.get("total_return", np.nan))
    out["Sharpe_vs_SPY"] = out["Sharpe"] - float(spy_row.get("Sharpe", np.nan))
    out["DD_vs_SPY"] = out["max_drawdown"] - float(spy_row.get("max_drawdown", np.nan))
    return out


def _benchmark_daily(ticker: str) -> pd.DataFrame:
    snaps = _snapshot_data()
    if snaps.empty:
        return pd.DataFrame()
    df = snaps[snaps["ticker"].eq(ticker)].drop_duplicates("date").sort_values("date").copy()
    if df.empty:
        return df
    df["return"] = _num(df["current_price"]).pct_change()
    return df.dropna(subset=["return"])


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        reasons = []
        classification = "research only"
        if row.get("validity_warning") == "benchmark_hindsight_diagnostic_only":
            classification = "reject"
            reasons.append("uses benchmark hindsight; diagnostic only")
        elif row.get("Sharpe_vs_SPY", -np.inf) > 0 and row.get("return_vs_SPY", -np.inf) > 0 and row.get("DD_vs_SPY", -np.inf) >= -0.03:
            classification = "candidate for shadow mode"
            reasons.append("beats SPY return and Sharpe without materially worse drawdown")
        elif row.get("Sharpe", -np.inf) < 0.5:
            classification = "reject"
            reasons.append("weak Sharpe")
        else:
            reasons.append("needs further walk-forward validation")
        rows.append(
            {
                "variant": row["variant"],
                "classification": classification,
                "reason": "; ".join(reasons),
                "production_change": "none",
            }
        )
    return pd.DataFrame(rows)


def run_missed_winners_turnover_research() -> dict[str, pd.DataFrame]:
    raw_daily = _raw_daily()
    snaps = _snapshot_data()
    realized = _realized_data()
    if raw_daily.empty or snaps.empty or realized.empty:
        raise ValueError("Required historical raw target, snapshot, or realized return files are missing.")

    missed_attr = missed_winners_attribution(raw_daily, snaps)
    retention = winner_retention_analysis(raw_daily, realized)
    turnover = turnover_drag_analysis(raw_daily, realized)
    variant_results, variant_daily = _simulate_variants(raw_daily, snaps, realized, missed_attr)
    variant_results = _add_spy_comparison(variant_results)
    governance = _governance(variant_results)

    missed_attr.to_csv(OUT_MISSED, index=False)
    retention.to_csv(OUT_RETENTION, index=False)
    turnover.to_csv(OUT_TURNOVER, index=False)
    variant_results.to_csv(OUT_VARIANTS, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== MISSED WINNERS & TURNOVER RESEARCH =====")
    print(variant_results.to_string(index=False))

    print("\n===== MISSED WINNERS ATTRIBUTION =====")
    cols = [
        "date",
        "ticker",
        "period_return_20d",
        "was_selected",
        "was_underweighted",
        "expected_return_rank_pct",
        "signal_rank_pct",
        "filter_status",
        "diagnosis",
    ]
    print(missed_attr[[c for c in cols if c in missed_attr.columns]].head(30).to_string(index=False))

    print("\n===== WINNER RETENTION ANALYSIS =====")
    print(retention.sort_values("return_after_exit_20d", ascending=False).head(20).to_string(index=False))

    print("\n===== TURNOVER DRAG ANALYSIS =====")
    print(turnover.sort_values("avg_return_after_sale_20d", ascending=False).head(20).to_string(index=False))

    print("\n===== OVERLAY VARIANT COMPARISON =====")
    show_cols = [
        "variant",
        "total_return",
        "CAGR",
        "volatility",
        "Sharpe",
        "max_drawdown",
        "turnover",
        "missed_winner_capture_rate",
        "return_vs_SPY",
        "Sharpe_vs_SPY",
        "DD_vs_SPY",
        "validity_warning",
    ]
    print(variant_results[[c for c in show_cols if c in variant_results.columns]].to_string(index=False))

    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))

    print(f"\nSaved: {Path(OUT_MISSED).resolve()}")
    print(f"Saved: {Path(OUT_RETENTION).resolve()}")
    print(f"Saved: {Path(OUT_TURNOVER).resolve()}")
    print(f"Saved: {Path(OUT_VARIANTS).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")

    return {
        "missed": missed_attr,
        "retention": retention,
        "turnover": turnover,
        "variant_results": variant_results,
        "variant_daily": variant_daily,
        "governance": governance,
    }


if __name__ == "__main__":
    run_missed_winners_turnover_research()
