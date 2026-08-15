from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAILY_FILE = "production_parity_growth_daily_returns.csv"
TRADES_FILE = "production_parity_growth_trades.csv"
RESULTS_FILE = "production_parity_growth_results.csv"
OVERLAY_FILE = "production_parity_drawdown_overlay_diagnostics.csv"
BENCHMARK_FILE = "production_parity_growth_benchmark_comparison.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"

OUT_RESULTS = "growth_reality_check_results.csv"
OUT_COSTS = "growth_transaction_cost_stress.csv"
OUT_LIQUIDITY = "growth_liquidity_audit.csv"
OUT_SURVIVORSHIP = "growth_survivorship_bias_audit.csv"
OUT_ADJUSTED = "growth_reality_adjusted_results.csv"
OUT_GOVERNANCE = "growth_reality_governance.csv"

TRADING_DAYS = 252
TARGET_VOL = 0.22
COST_BPS = [0, 5, 10, 25, 50]
SLIPPAGE_BPS = [0, 5, 10, 25]
CAPITAL_LEVELS = [10_000, 100_000, 1_000_000, 10_000_000]


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
        return {"candidate": name, "observations": 0}
    r = _num(data[return_col]).dropna()
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
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
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
        "max_drawdown": max_dd,
        "Calmar": np.nan if max_dd >= 0 else cagr / abs(max_dd),
    }


def _apply_overlay(daily: pd.DataFrame, candidate: str) -> pd.DataFrame:
    data = _dates(daily).sort_values("date").copy()
    data["gross_return"] = _num(data["return"]).fillna(0.0)
    data["return"] = data["gross_return"]
    data["target_exposure"] = _num(data.get("target_exposure", pd.Series(index=data.index, dtype=float))).fillna(0.0).clip(0, 1)
    if candidate == "growth_candidate_v1":
        data["candidate"] = candidate
        return data
    if candidate == "growth_candidate_v1_exposure_cap_60pct":
        adjusted_exposure = data["target_exposure"].clip(upper=0.60)
        scale = np.where(data["target_exposure"] > 0, adjusted_exposure / data["target_exposure"], 0.0)
        data["return"] = data["gross_return"] * scale
        data["target_exposure"] = adjusted_exposure
    elif candidate == "growth_candidate_v1_drawdown_brake_18pct":
        equity = 1.0
        high = 1.0
        factors: list[float] = []
        defensive = False
        for ret in data["gross_return"]:
            dd = equity / high - 1.0
            if dd <= -0.18:
                defensive = True
            if defensive and dd > -0.05:
                defensive = False
            factor = 0.50 if defensive else 1.0
            factors.append(factor)
            equity *= 1.0 + float(ret) * factor
            high = max(high, equity)
        data["return"] = data["gross_return"] * pd.Series(factors, index=data.index)
        data["target_exposure"] = data["target_exposure"] * pd.Series(factors, index=data.index)
    data["cash_weight"] = 1.0 - data["target_exposure"]
    data["candidate"] = candidate
    return data


def _candidate_daily_sets() -> dict[str, pd.DataFrame]:
    daily = _dates(_read_csv(DAILY_FILE)).sort_values("date")
    if daily.empty:
        raise ValueError("production_parity_growth_daily_returns.csv is required.")
    return {
        "growth_candidate_v1": _apply_overlay(daily, "growth_candidate_v1"),
        "growth_candidate_v1_exposure_cap_60pct": _apply_overlay(daily, "growth_candidate_v1_exposure_cap_60pct"),
        "growth_candidate_v1_drawdown_brake_18pct": _apply_overlay(daily, "growth_candidate_v1_drawdown_brake_18pct"),
    }


def _cost_stress(candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, daily in candidates.items():
        base_total = _metrics(name, daily).get("total_return", np.nan)
        turnover = _num(daily.get("turnover", pd.Series(index=daily.index, dtype=float))).fillna(0.0)
        for cost_bps in COST_BPS:
            for slip_bps in SLIPPAGE_BPS:
                all_in_rate = (cost_bps + slip_bps) / 10_000.0
                adjusted = daily.copy()
                adjusted["transaction_cost_drag"] = turnover * all_in_rate
                adjusted["return"] = _num(adjusted["return"]).fillna(0.0) - adjusted["transaction_cost_drag"]
                metrics = _metrics(name, adjusted)
                metrics.update(
                    {
                        "cost_bps": cost_bps,
                        "slippage_bps": slip_bps,
                        "all_in_bps": cost_bps + slip_bps,
                        "return_drag": metrics.get("total_return", np.nan) - base_total,
                        "avg_period_cost_drag": float(adjusted["transaction_cost_drag"].mean()),
                    }
                )
                rows.append(metrics)
    out = pd.DataFrame(rows)
    breakevens = []
    for name, group in out.groupby("candidate"):
        viable = group[group["total_return"] > 0].sort_values("all_in_bps")
        breakevens.append(
            {
                "candidate": name,
                "breakeven_cost_level_bps_grid": float(viable["all_in_bps"].max()) if not viable.empty else 0.0,
            }
        )
    return out.merge(pd.DataFrame(breakevens), on="candidate", how="left")


def _benchmark_thresholds() -> dict[str, dict[str, float]]:
    bench = _read_csv(BENCHMARK_FILE)
    out: dict[str, dict[str, float]] = {}
    if bench.empty:
        return out
    for _, row in bench.iterrows():
        name = str(row.get("benchmark", ""))
        out[name] = {
            "total_return": float(row.get("benchmark_return", np.nan)),
            "Sharpe": float(row.get("benchmark_Sharpe", np.nan)),
            "max_drawdown": float(row.get("benchmark_max_drawdown", np.nan)),
            "CAGR": float(row.get("benchmark_CAGR", np.nan)),
        }
    return out


def _reality_adjusted(costs: pd.DataFrame) -> pd.DataFrame:
    bench = _benchmark_thresholds()
    spy = bench.get("SPY", {})
    qqq = bench.get("QQQ", {})
    rows = []
    for _, row in costs.iterrows():
        row = row.to_dict()
        row["still_beats_SPY_return"] = bool(row.get("total_return", -np.inf) > spy.get("total_return", np.inf))
        row["still_beats_SPY_sharpe"] = bool(row.get("Sharpe", -np.inf) > spy.get("Sharpe", np.inf))
        row["still_beats_QQQ_return"] = bool(row.get("total_return", -np.inf) > qqq.get("total_return", np.inf))
        row["still_beats_QQQ_sharpe"] = bool(row.get("Sharpe", -np.inf) > qqq.get("Sharpe", np.inf))
        row["passes_growth_criteria"] = bool(
            row.get("CAGR", 0) > 0.25
            and row.get("Sharpe", 0) > 1.0
            and row.get("max_drawdown", -1) > -0.25
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _liquidity_audit() -> pd.DataFrame:
    trades = _dates(_read_csv(TRADES_FILE))
    snaps = _dates(_read_csv(SNAPSHOTS_FILE))
    if trades.empty:
        return pd.DataFrame()
    rows = []
    volume_cols = [c for c in snaps.columns if c.lower() in {"volume", "avg_volume", "dollar_volume", "average_dollar_volume"}]
    has_volume = bool(volume_cols)
    for ticker, group in trades.groupby("ticker"):
        avg_weight = float(_num(group["weight"]).mean())
        max_weight = float(_num(group["weight"]).max())
        row = {
            "ticker": ticker,
            "trade_count": len(group),
            "avg_weight": avg_weight,
            "max_weight": max_weight,
            "average_daily_dollar_volume": np.nan,
            "median_daily_dollar_volume": np.nan,
            "minimum_daily_dollar_volume": np.nan,
            "liquidity_confidence": "low" if not has_volume else "medium",
            "illiquid_flag": "unknown_volume",
            "microcap_flag": "unknown_market_cap",
            "stale_price_flag": "not_evaluable_without_volume",
            "SPAC_like_flag": bool(len(str(ticker)) >= 4 and str(ticker).endswith(("U", "W", "R"))),
        }
        for capital in CAPITAL_LEVELS:
            row[f"estimated_position_size_{capital}"] = capital * max_weight
            row[f"participation_rate_{capital}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["liquidity_confidence", "max_weight", "trade_count"], ascending=[True, False, False])


def _survivorship_bias_audit() -> pd.DataFrame:
    snaps = _read_csv(SNAPSHOTS_FILE)
    tickers = set(snaps.get("ticker", pd.Series(dtype=str)).astype(str)) if not snaps.empty else set()
    dead_like = [t for t in tickers if len(t) >= 4 and t.endswith(("U", "W", "R"))]
    rows = [
        {
            "audit_item": "point_in_time_universe",
            "status": "not_confirmed",
            "risk": "high",
            "explanation": "Historical snapshots appear generated from the available research universe, not a verified point-in-time constituent database.",
        },
        {
            "audit_item": "delisted_failed_names_absent",
            "status": "likely",
            "risk": "high",
            "explanation": "No delisting/point-in-time membership file is available; failed names may be missing from historical candidate selection.",
        },
        {
            "audit_item": "current_universe_future_knowledge",
            "status": "possible",
            "risk": "high",
            "explanation": "If the universe was expanded from current Nasdaq/ticker lists, historical replay can include future survivor knowledge.",
        },
        {
            "audit_item": "SPAC_dead_ticker_presence",
            "status": "mixed",
            "risk": "medium",
            "explanation": f"SPAC-like suffix tickers detected: {len(dead_like)}. Presence helps, but it is not sufficient for point-in-time coverage.",
        },
        {
            "audit_item": "survivorship_bias_risk",
            "status": "high",
            "risk": "high",
            "explanation": "Treat growth backtest as research-only until tested on a point-in-time survivorship-free universe.",
        },
    ]
    return pd.DataFrame(rows)


def _governance(adjusted: pd.DataFrame, liquidity: pd.DataFrame, survivorship: pd.DataFrame) -> pd.DataFrame:
    rows = []
    survivorship_risk = "high" if (not survivorship.empty and "high" in set(survivorship["risk"].astype(str))) else "medium"
    liquidity_confidence = "low" if liquidity.empty or "low" in set(liquidity["liquidity_confidence"].astype(str)) else "medium"
    for candidate, group in adjusted.groupby("candidate"):
        moderate = group[(group["all_in_bps"] <= 20)].copy()
        best_moderate = moderate.sort_values("Sharpe", ascending=False).head(1)
        if best_moderate.empty:
            classification = "fail reality check"
            reason = "No moderate-cost scenario available."
        else:
            row = best_moderate.iloc[0]
            if survivorship_risk == "high" or liquidity_confidence == "low":
                classification = "eligible for small-capital paper trading"
                reason = "Performance survives moderate cost assumptions, but liquidity confidence is low and survivorship risk is high."
            elif bool(row["passes_growth_criteria"]):
                classification = "eligible for extended paper trading"
                reason = "Passes growth criteria under moderate frictions."
            else:
                classification = "research only"
                reason = "Friction-adjusted performance does not pass all growth criteria."
        rows.append(
            {
                "candidate": candidate,
                "classification": classification,
                "liquidity_confidence": liquidity_confidence,
                "survivorship_bias_risk": survivorship_risk,
                "production_changed": False,
                "real_trading": False,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def run_growth_reality_check() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = _candidate_daily_sets()
    base_results = pd.DataFrame([_metrics(name, daily) for name, daily in candidates.items()])
    costs = _cost_stress(candidates)
    liquidity = _liquidity_audit()
    survivorship = _survivorship_bias_audit()
    adjusted = _reality_adjusted(costs)
    governance = _governance(adjusted, liquidity, survivorship)

    base_results.to_csv(OUT_RESULTS, index=False)
    costs.to_csv(OUT_COSTS, index=False)
    liquidity.to_csv(OUT_LIQUIDITY, index=False)
    survivorship.to_csv(OUT_SURVIVORSHIP, index=False)
    adjusted.to_csv(OUT_ADJUSTED, index=False)
    governance.to_csv(OUT_GOVERNANCE, index=False)

    print("\n===== GROWTH REALITY CHECK =====")
    print(base_results.to_string(index=False))
    print("\n===== TRANSACTION COST STRESS TEST =====")
    view = costs[costs["slippage_bps"].isin([0, 10, 25]) & costs["cost_bps"].isin([0, 10, 25, 50])]
    print(view[["candidate", "cost_bps", "slippage_bps", "total_return", "CAGR", "Sharpe", "max_drawdown", "return_drag", "breakeven_cost_level_bps_grid"]].to_string(index=False))
    print("\n===== LIQUIDITY AUDIT =====")
    print(liquidity[["ticker", "trade_count", "avg_weight", "max_weight", "liquidity_confidence", "illiquid_flag", "microcap_flag", "SPAC_like_flag"]].to_string(index=False))
    print("\n===== SURVIVORSHIP BIAS AUDIT =====")
    print(survivorship.to_string(index=False))
    print("\n===== REALITY ADJUSTED RESULTS =====")
    moderate = adjusted[(adjusted["cost_bps"].isin([10, 25, 50])) & (adjusted["slippage_bps"].isin([10, 25]))]
    print(moderate[["candidate", "cost_bps", "slippage_bps", "total_return", "CAGR", "Sharpe", "max_drawdown", "still_beats_SPY_sharpe", "still_beats_QQQ_sharpe", "passes_growth_criteria"]].to_string(index=False))
    print("\n===== GROWTH REALITY GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(OUT_RESULTS).resolve()}")
    print(f"Saved: {Path(OUT_COSTS).resolve()}")
    print(f"Saved: {Path(OUT_LIQUIDITY).resolve()}")
    print(f"Saved: {Path(OUT_SURVIVORSHIP).resolve()}")
    print(f"Saved: {Path(OUT_ADJUSTED).resolve()}")
    print(f"Saved: {Path(OUT_GOVERNANCE).resolve()}")
    return base_results, adjusted, governance


if __name__ == "__main__":
    run_growth_reality_check()
