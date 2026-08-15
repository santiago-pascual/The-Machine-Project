from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 100000.0


def numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def compound_return(returns: pd.Series) -> float:
    r = numeric(returns).fillna(0)
    if r.empty:
        return np.nan
    return float((1 + r).prod() - 1)


def sharpe(returns: pd.Series) -> float:
    r = numeric(returns).dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=0) * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    r = numeric(returns).fillna(0)
    if r.empty:
        return np.nan
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in ["date", "entry_date", "exit_date", "signal_date", "economic_application_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def decile_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "raw_target_rank" not in trades.columns or "asset_return" not in trades.columns:
        return pd.DataFrame()
    work = trades.dropna(subset=["raw_target_rank", "asset_return"]).copy()
    if work.empty:
        return pd.DataFrame()
    # Rank 1 is best, so invert for decile labels: decile 10 = best forecast rank.
    work["rank_percentile"] = 1 - (numeric(work["raw_target_rank"]) - numeric(work["raw_target_rank"]).min()) / max(
        numeric(work["raw_target_rank"]).max() - numeric(work["raw_target_rank"]).min(), 1
    )
    work["forecast_decile"] = pd.qcut(work["rank_percentile"].rank(method="first"), 10, labels=False, duplicates="drop") + 1
    out = (
        work.groupby("forecast_decile")
        .agg(
            observations=("ticker", "count"),
            average_return=("asset_return", "mean"),
            volatility=("asset_return", "std"),
            hit_rate=("asset_return", lambda s: float((numeric(s) > 0).mean())),
            average_rank=("raw_target_rank", "mean"),
        )
        .reset_index()
    )
    out["sharpe_proxy"] = out["average_return"] / out["volatility"].replace(0, np.nan)
    return out


def forecast_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"metric": "forecast_data_available", "value": False}])
    work = trades.dropna(subset=["raw_target_reconstructed", "asset_return", "raw_target_rank"]).copy()
    if work.empty:
        return pd.DataFrame([{"metric": "forecast_data_available", "value": False}])
    ic = numeric(work["raw_target_reconstructed"]).corr(numeric(work["asset_return"]), method="pearson")
    ric = numeric(work["raw_target_reconstructed"]).corr(numeric(work["asset_return"]), method="spearman")
    hit = float((numeric(work["asset_return"]) > 0).mean())
    dec = decile_analysis(work)
    spread = np.nan
    mono = np.nan
    if not dec.empty:
        top = dec.loc[dec["forecast_decile"].eq(dec["forecast_decile"].max()), "average_return"].mean()
        bot = dec.loc[dec["forecast_decile"].eq(dec["forecast_decile"].min()), "average_return"].mean()
        spread = float(top - bot)
        mono = dec["forecast_decile"].corr(dec["average_return"], method="spearman")
    return pd.DataFrame(
        [
            {"metric": "forecast_ic", "value": ic},
            {"metric": "forecast_rank_ic", "value": ric},
            {"metric": "forecast_hit_rate", "value": hit},
            {"metric": "forecast_decile_spread", "value": spread},
            {"metric": "forecast_monotonicity", "value": mono},
            {"metric": "observations", "value": len(work)},
        ]
    )


def ranking_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "raw_target_rank" not in trades.columns:
        return pd.DataFrame()
    work = trades.dropna(subset=["raw_target_rank", "asset_return"]).copy()
    rows = []
    for n in [10, 20, 50]:
        sub = work[numeric(work["raw_target_rank"]) <= n]
        rows.append(
            {
                "rank_bucket": f"top_{n}",
                "observations": len(sub),
                "average_asset_return": numeric(sub.get("asset_return", pd.Series(dtype=float))).mean(),
                "hit_rate": (numeric(sub.get("asset_return", pd.Series(dtype=float))) > 0).mean() if len(sub) else np.nan,
            }
        )
    by_rank = (
        work.groupby("raw_target_rank")
        .agg(observations=("ticker", "count"), average_return=("asset_return", "mean"))
        .reset_index()
        .head(50)
    )
    rows.extend(
        by_rank.assign(rank_bucket=lambda d: "rank_" + d["raw_target_rank"].astype(str))
        .rename(columns={"average_return": "average_asset_return"})
        .to_dict("records")
    )
    out = pd.DataFrame(rows)
    return out


def sizing_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(), np.nan
    work = trades.dropna(subset=["date", "ticker", "weight", "asset_return"]).copy()
    if work.empty:
        return pd.DataFrame(), np.nan
    rows = []
    diffs = []
    for d, g in work.groupby("date"):
        exposure = numeric(g["weight"]).sum()
        current_return = numeric(g.get("trade_contribution", g["weight"] * g["asset_return"])).sum()
        equal_weight = exposure / len(g) if len(g) else 0
        equal_return = (numeric(g["asset_return"]) * equal_weight).sum()
        diff = current_return - equal_return
        diffs.append(diff)
        rows.append(
            {
                "date": d,
                "selected_count": len(g),
                "exposure": exposure,
                "current_weighted_return": current_return,
                "equal_weight_return": equal_return,
                "sizing_alpha": diff,
            }
        )
    return pd.DataFrame(rows), compound_return(pd.Series(diffs))


def cash_drag_analysis(daily: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if daily.empty or "gross_daily_return" not in daily.columns:
        return pd.DataFrame(), np.nan
    work = daily.copy()
    exposure = numeric(work.get("exposure", pd.Series(dtype=float))).replace(0, np.nan)
    work["fully_invested_equivalent_return"] = numeric(work["gross_daily_return"]) / exposure
    work["cash_drag_return"] = numeric(work["gross_daily_return"]) - work["fully_invested_equivalent_return"]
    actual = compound_return(work["gross_daily_return"])
    full = compound_return(work["fully_invested_equivalent_return"].replace([np.inf, -np.inf], np.nan).fillna(0))
    drag = actual - full
    return work[["date", "gross_daily_return", "exposure", "cash", "fully_invested_equivalent_return", "cash_drag_return"]], drag


def turnover_analysis(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["turnover"] = numeric(work.get("turnover", pd.Series(dtype=float))).fillna(0)
    work["abs_return_next"] = numeric(work.get("gross_daily_return", pd.Series(dtype=float))).abs()
    return pd.DataFrame(
        [
            {
                "average_turnover": work["turnover"].mean(),
                "total_turnover": work["turnover"].sum(),
                "turnover_return_correlation": work["turnover"].corr(numeric(work.get("gross_daily_return", pd.Series(dtype=float))))
                if len(work) > 2
                else np.nan,
                "high_turnover_days": int((work["turnover"] > work["turnover"].quantile(0.9)).sum())
                if len(work) > 10
                else int((work["turnover"] > 0).sum()),
            }
        ]
    )


def holding_period_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    if "entry_date" in work.columns and "exit_date" in work.columns:
        work["holding_days"] = (work["exit_date"] - work["entry_date"]).dt.days
    else:
        work["holding_days"] = np.nan
    work["winner"] = numeric(work.get("asset_return", pd.Series(dtype=float))) > 0
    return (
        work.groupby("ticker")
        .agg(
            observations=("ticker", "count"),
            average_holding_days=("holding_days", "mean"),
            average_return=("asset_return", "mean"),
            hit_rate=("winner", "mean"),
            total_contribution=("trade_contribution", "sum"),
        )
        .reset_index()
        .sort_values("total_contribution", ascending=True)
    )


def cost_attribution(daily: pd.DataFrame, official_costs: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    hist_cost_return = (
        numeric(daily.get("net_daily_return", pd.Series(dtype=float))).fillna(0)
        - numeric(daily.get("gross_daily_return", pd.Series(dtype=float))).fillna(0)
        if not daily.empty
        else pd.Series(dtype=float)
    )
    hist_impact = compound_return(hist_cost_return) if not hist_cost_return.empty else 0.0
    rows.append({"source": "historical_reconstruction", "cost_alpha": hist_impact, "notes": "net minus gross daily return"})
    official_total_cost = (
        numeric(official_costs.get("estimated_total_cost", pd.Series(dtype=float))).sum() if not official_costs.empty else 0.0
    )
    rows.append(
        {
            "source": "official_forward_estimated_costs",
            "cost_alpha": -official_total_cost / INITIAL_CAPITAL,
            "notes": "official estimated reporting-only costs / initial capital",
        }
    )
    return pd.DataFrame(rows), float(-official_total_cost / INITIAL_CAPITAL)
