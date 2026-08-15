from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRADES_FILE = "production_parity_growth_trades.csv"
DAILY_FILE = "production_parity_growth_daily_returns.csv"
REALITY_RESULTS_FILE = "growth_reality_check_results.csv"
LIQUIDITY_PRIOR_FILE = "growth_liquidity_audit.csv"
SURVIVORSHIP_PRIOR_FILE = "growth_survivorship_bias_audit.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
SOURCE_FILE = "financial_data_system.py"

OUT_UNIVERSE = "point_in_time_universe_audit.csv"
OUT_AVAILABILITY = "historical_ticker_availability.csv"
OUT_LIQUIDITY_DATA = "growth_liquidity_data.csv"
OUT_CAPACITY = "growth_capacity_analysis.csv"
OUT_LIQUIDITY_ADJUSTED = "growth_liquidity_adjusted_results.csv"
OUT_GOVERNANCE = "point_in_time_liquidity_governance.csv"

CAPITAL_LEVELS = [10_000, 100_000, 1_000_000, 10_000_000]
MAX_PARTICIPATION = 0.05


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
        return {"scenario": name, "observations": 0}
    r = _num(data[return_col]).dropna()
    if r.empty:
        return {"scenario": name, "observations": 0}
    ppy = _ppy(data["date"])
    total = float((1.0 + r).prod() - 1.0)
    years = max((data["date"].max() - data["date"].min()).days / 365.25, len(r) / ppy, 1e-9)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=0) * np.sqrt(ppy))
    sharpe = np.nan if vol <= 0 else float((r.mean() * ppy) / vol)
    equity = (1.0 + r).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "scenario": name,
        "observations": int(len(r)),
        "total_return": total,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def _source_contains(pattern: str) -> bool:
    path = Path(SOURCE_FILE)
    if not path.exists():
        return False
    return pattern in path.read_text(errors="ignore")


def _universe_audit() -> pd.DataFrame:
    has_builder = _source_contains("def build_trading_universe")
    has_nasdaq_fetch = _source_contains("fetch_nasdaq_listed_tickers")
    uses_current_nasdaq = _source_contains("include_full_nasdaq=True")
    has_point_in_time_file = bool(list(Path(".").glob("*point*in*time*"))) or bool(list(Path(".").glob("*constituent*")))
    if has_point_in_time_file:
        method = "local_point_in_time_candidate_file_detected"
        pit_confirmed = False
        risk = "medium"
        reason = "A point-in-time-like file name exists, but this audit did not validate constituent dates."
    elif has_builder and has_nasdaq_fetch and uses_current_nasdaq:
        method = "dynamic_current_nasdaq_list_plus_manual_research_universe"
        pit_confirmed = False
        risk = "high"
        reason = "Universe builder fetches or falls back to current Nasdaq-style ticker lists; no historical constituent membership dates are available."
    else:
        method = "unknown_or_static_research_universe"
        pit_confirmed = False
        risk = "high"
        reason = "Could not verify a point-in-time universe construction method from local files."
    rows = [
        {
            "audit_item": "universe_method",
            "value": method,
            "point_in_time_confirmed": pit_confirmed,
            "survivorship_bias_risk": risk,
            "reason": reason,
        },
        {
            "audit_item": "current_list_bias",
            "value": bool(uses_current_nasdaq),
            "point_in_time_confirmed": pit_confirmed,
            "survivorship_bias_risk": "high" if uses_current_nasdaq else risk,
            "reason": "Current ticker list usage creates future-survivor risk for historical dates." if uses_current_nasdaq else "Current-list usage not detected directly.",
        },
        {
            "audit_item": "delisted_failed_tickers",
            "value": "not_available",
            "point_in_time_confirmed": False,
            "survivorship_bias_risk": "high",
            "reason": "No delisting database or failed-name universe file is present.",
        },
        {
            "audit_item": "ticker_changes_mergers_SPACs",
            "value": "not_handled_point_in_time",
            "point_in_time_confirmed": False,
            "survivorship_bias_risk": "high",
            "reason": "No corporate action mapping file was found for ticker changes, mergers, SPAC conversions, or delistings.",
        },
    ]
    return pd.DataFrame(rows)


def _availability_audit() -> pd.DataFrame:
    trades = _dates(_read_csv(TRADES_FILE))
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if trades.empty:
        return pd.DataFrame()
    if "model_mode" in snaps.columns:
        base = snaps[snaps["model_mode"].astype(str).eq("baseline")]
        if not base.empty:
            snaps = base
    rows = []
    for ticker, tgroup in trades.groupby("ticker"):
        ticker = str(ticker)
        s = snaps[snaps["ticker"].astype(str).eq(ticker)].drop_duplicates("date").sort_values("date").copy()
        trade_dates = pd.to_datetime(tgroup["date"], errors="coerce").dt.normalize().dropna().sort_values()
        if s.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "first_available_price_date": "",
                    "last_available_price_date": "",
                    "trade_count": int(len(tgroup)),
                    "trades_with_available_snapshot": 0,
                    "ticker_existed_on_trade_dates": False,
                    "suspicious_gap_count": np.nan,
                    "max_gap_days": np.nan,
                    "appears_only_after_strong_performance": "not_evaluable",
                    "availability_warning": "ticker_missing_from_historical_snapshots",
                }
            )
            continue
        available_dates = set(s["date"])
        trades_available = int(sum(date in available_dates for date in trade_dates))
        gaps = s["date"].sort_values().diff().dt.days.dropna()
        max_gap = float(gaps.max()) if not gaps.empty else 0.0
        suspicious_gaps = int((gaps > 21).sum()) if not gaps.empty else 0
        first_trade = trade_dates.min() if not trade_dates.empty else pd.NaT
        first_available = pd.Timestamp(s["date"].min())
        first_trade_lag = np.nan if pd.isna(first_trade) else int((first_trade - first_available).days)
        pre_trade_prices = s[s["date"] <= first_trade].copy() if not pd.isna(first_trade) else pd.DataFrame()
        if len(pre_trade_prices) >= 4:
            perf_into_first_trade = float(_num(pre_trade_prices["current_price"]).iloc[-1] / _num(pre_trade_prices["current_price"]).iloc[0] - 1.0)
            appears_after_runup = bool(first_trade_lag <= 45 and perf_into_first_trade > 0.30)
        else:
            perf_into_first_trade = np.nan
            appears_after_runup = False
        rows.append(
            {
                "ticker": ticker,
                "first_available_price_date": first_available.date().isoformat(),
                "last_available_price_date": pd.Timestamp(s["date"].max()).date().isoformat(),
                "trade_count": int(len(tgroup)),
                "trades_with_available_snapshot": trades_available,
                "ticker_existed_on_trade_dates": bool(trades_available == len(tgroup)),
                "suspicious_gap_count": suspicious_gaps,
                "max_gap_days": max_gap,
                "first_trade_lag_days": first_trade_lag,
                "performance_into_first_trade": perf_into_first_trade,
                "appears_only_after_strong_performance": appears_after_runup,
                "availability_warning": "ok" if trades_available == len(tgroup) and suspicious_gaps == 0 else "review",
            }
        )
    return pd.DataFrame(rows).sort_values(["availability_warning", "trade_count"], ascending=[False, False])


def _liquidity_data() -> pd.DataFrame:
    trades = _dates(_read_csv(TRADES_FILE))
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if trades.empty:
        return pd.DataFrame()
    volume_col = next((c for c in snaps.columns if c.lower() in {"volume", "avg_volume", "daily_volume"}), None)
    dollar_col = next((c for c in snaps.columns if c.lower() in {"dollar_volume", "average_dollar_volume", "adv"}), None)
    rows = []
    for ticker, group in trades.groupby("ticker"):
        ticker = str(ticker)
        row = {
            "ticker": ticker,
            "trade_count": int(len(group)),
            "volume_available": bool(volume_col or dollar_col),
            "volume_missing_pct": 1.0,
            "average_daily_volume": np.nan,
            "median_daily_volume": np.nan,
            "average_daily_dollar_volume": np.nan,
            "median_daily_dollar_volume": np.nan,
            "minimum_20d_rolling_dollar_volume": np.nan,
            "liquidity_confidence": "low",
            "liquidity_reason": "No local historical volume or dollar-volume columns available in snapshots.",
        }
        if volume_col or dollar_col:
            s = snaps[snaps["ticker"].astype(str).eq(ticker)].copy().sort_values("date")
            if dollar_col:
                dollar_volume = _num(s[dollar_col])
            else:
                dollar_volume = _num(s["current_price"]) * _num(s[volume_col])
            volume = _num(s[volume_col]) if volume_col else pd.Series(np.nan, index=s.index)
            row.update(
                {
                    "volume_missing_pct": float(dollar_volume.isna().mean()),
                    "average_daily_volume": float(volume.mean()) if volume.notna().any() else np.nan,
                    "median_daily_volume": float(volume.median()) if volume.notna().any() else np.nan,
                    "average_daily_dollar_volume": float(dollar_volume.mean()) if dollar_volume.notna().any() else np.nan,
                    "median_daily_dollar_volume": float(dollar_volume.median()) if dollar_volume.notna().any() else np.nan,
                    "minimum_20d_rolling_dollar_volume": float(dollar_volume.rolling(20, min_periods=5).mean().min()) if dollar_volume.notna().sum() >= 5 else np.nan,
                    "liquidity_confidence": "medium" if dollar_volume.notna().sum() > 20 else "low",
                    "liquidity_reason": "Computed from local snapshot volume fields.",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _capacity_analysis(liquidity: pd.DataFrame) -> pd.DataFrame:
    prior = _read_csv(LIQUIDITY_PRIOR_FILE)
    trades = _dates(_read_csv(TRADES_FILE))
    rows = []
    for ticker, group in trades.groupby("ticker"):
        ticker = str(ticker)
        max_weight = float(_num(group["weight"]).max())
        avg_weight = float(_num(group["weight"]).mean())
        liq_row = liquidity[liquidity["ticker"].astype(str).eq(ticker)]
        adv = float(liq_row["average_daily_dollar_volume"].iloc[0]) if not liq_row.empty else np.nan
        if not np.isfinite(adv) and not prior.empty and "average_daily_dollar_volume" in prior.columns:
            p = prior[prior["ticker"].astype(str).eq(ticker)]
            adv = float(p["average_daily_dollar_volume"].iloc[0]) if not p.empty else np.nan
        for capital in CAPITAL_LEVELS:
            position_size = capital * max_weight
            participation = position_size / adv if np.isfinite(adv) and adv > 0 else np.nan
            if not np.isfinite(participation):
                flag = "unknown"
            elif participation <= 0.01:
                flag = "safe"
            elif participation <= MAX_PARTICIPATION:
                flag = "caution"
            else:
                flag = "illiquid"
            rows.append(
                {
                    "ticker": ticker,
                    "capital": capital,
                    "avg_weight": avg_weight,
                    "max_weight": max_weight,
                    "estimated_position_size": position_size,
                    "average_daily_dollar_volume": adv,
                    "participation_rate": participation,
                    "capacity_flag": flag,
                    "confidence": "low" if not np.isfinite(adv) else "medium",
                }
            )
    return pd.DataFrame(rows)


def _liquidity_adjusted_results(capacity: pd.DataFrame) -> pd.DataFrame:
    daily = _dates(_read_csv(DAILY_FILE)).sort_values("date")
    trades = _dates(_read_csv(TRADES_FILE)).sort_values("date")
    if daily.empty or trades.empty:
        return pd.DataFrame()
    rows = []
    for capital in [10_000, 100_000, 1_000_000]:
        cap = capacity[capacity["capital"].eq(capital)].copy()
        if cap.empty or cap["participation_rate"].notna().sum() == 0:
            row = _metrics(f"liquidity_adjusted_{capital}", daily)
            row.update(
                {
                    "capital": capital,
                    "liquidity_filter_available": False,
                    "skipped_trades": np.nan,
                    "capped_trades": np.nan,
                    "return_drag": np.nan,
                    "reason": "Volume unavailable; liquidity-adjusted backtest cannot be computed reliably.",
                }
            )
            rows.append(row)
            continue
        allowed = set(cap.loc[cap["participation_rate"] <= MAX_PARTICIPATION, "ticker"].astype(str))
        adjusted_trades = trades.copy()
        adjusted_trades["allowed"] = adjusted_trades["ticker"].astype(str).isin(allowed)
        skipped = int((~adjusted_trades["allowed"]).sum())
        contrib = adjusted_trades[adjusted_trades["allowed"]].groupby("date")["trade_contribution"].sum()
        adjusted = daily.copy()
        adjusted["return"] = contrib.reindex(adjusted["date"]).fillna(0.0).to_numpy()
        base_total = _metrics("base", daily).get("total_return", np.nan)
        row = _metrics(f"liquidity_adjusted_{capital}", adjusted)
        row.update(
            {
                "capital": capital,
                "liquidity_filter_available": True,
                "skipped_trades": skipped,
                "capped_trades": 0,
                "return_drag": row.get("total_return", np.nan) - base_total,
                "reason": "Excluded trades above 5% ADV.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _governance(universe: pd.DataFrame, liquidity: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    pit_confirmed = bool(universe["point_in_time_confirmed"].fillna(False).all()) if not universe.empty else False
    survivorship = "high"
    if not universe.empty and "survivorship_bias_risk" in universe.columns:
        risks = set(universe["survivorship_bias_risk"].astype(str))
        survivorship = "high" if "high" in risks else ("medium" if "medium" in risks else "low")
    liq_conf = "low"
    if not liquidity.empty and "liquidity_confidence" in liquidity.columns:
        confs = set(liquidity["liquidity_confidence"].astype(str))
        liq_conf = "low" if "low" in confs else ("medium" if "medium" in confs else "high")
    if survivorship == "high":
        classification = "blocked_by_survivorship_bias"
        reason = "Point-in-time universe cannot be confirmed with local data."
    elif liq_conf == "low":
        classification = "blocked_by_liquidity"
        reason = "Volume/dollar-volume data unavailable or insufficient."
    else:
        classification = "eligible_small_capital_paper"
        reason = "Point-in-time and liquidity risks are not blocking under available evidence."
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "point_in_time_confirmed": pit_confirmed,
                "survivorship_bias_risk": survivorship,
                "liquidity_confidence": liq_conf,
                "liquidity_adjusted_available": bool(not adjusted.empty and adjusted["liquidity_filter_available"].fillna(False).any()),
                "production_changed": False,
                "paper_trading_changed": False,
                "reason": reason,
            }
        ]
    )


def run_point_in_time_universe_liquidity_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = _universe_audit()
    availability = _availability_audit()
    liquidity = _liquidity_data()
    capacity = _capacity_analysis(liquidity)
    adjusted = _liquidity_adjusted_results(capacity)
    governance = _governance(universe, liquidity, adjusted)

    universe.to_csv(OUT_UNIVERSE, index=False)
    availability.to_csv(OUT_AVAILABILITY, index=False)
    liquidity.to_csv(OUT_LIQUIDITY_DATA, index=False)
    capacity.to_csv(OUT_CAPACITY, index=False)
    adjusted.to_csv(OUT_LIQUIDITY_ADJUSTED, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== POINT-IN-TIME UNIVERSE AUDIT =====")
    print(universe.to_string(index=False))
    print("\n===== HISTORICAL TICKER AVAILABILITY =====")
    print(availability.to_string(index=False))
    print("\n===== LIQUIDITY AUDIT =====")
    print(liquidity.to_string(index=False))
    print("\n===== CAPACITY ANALYSIS =====")
    print(capacity.head(80).to_string(index=False))
    print("\n===== LIQUIDITY-ADJUSTED RESULTS =====")
    print(adjusted.to_string(index=False))
    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_UNIVERSE).resolve()}")
    print(f"Saved: {Path(OUT_AVAILABILITY).resolve()}")
    print(f"Saved: {Path(OUT_LIQUIDITY_DATA).resolve()}")
    print(f"Saved: {Path(OUT_CAPACITY).resolve()}")
    print(f"Saved: {Path(OUT_LIQUIDITY_ADJUSTED).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return universe, liquidity, governance


if __name__ == "__main__":
    run_point_in_time_universe_liquidity_audit()
