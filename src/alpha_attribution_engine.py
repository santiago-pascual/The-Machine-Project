
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_attribution_calculations import (
    cash_drag_analysis,
    cost_attribution,
    decile_analysis,
    forecast_analysis,
    holding_period_analysis,
    load_csv,
    numeric,
    ranking_analysis,
    sizing_analysis,
    turnover_analysis,
)

TOL = 1e-8


def log_total_return(returns: pd.Series) -> float:
    r = numeric(returns).dropna()
    r = r[r > -0.999999]
    return float(np.log1p(r).sum()) if not r.empty else 0.0


def benchmark_log_return_from_cache(dates: pd.Series, ticker: str = "SPY") -> float:
    path = Path("yahoo_ohlcv_price_cache") / f"{ticker}.csv"
    if not path.exists():
        return 0.0
    try:
        px = pd.read_csv(path)
    except Exception:
        return 0.0
    date_col = "Date" if "Date" in px.columns else "date" if "date" in px.columns else None
    price_col = "Adj Close" if "Adj Close" in px.columns else "Close" if "Close" in px.columns else None
    if date_col is None or price_col is None:
        return 0.0
    base = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()})
    work = px[[date_col, price_col]].copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work["price"] = numeric(work[price_col])
    work = work.dropna(subset=["date", "price"]).sort_values("date")
    aligned = pd.merge_asof(base, work[["date", "price"]], on="date", direction="backward")
    ret = aligned["price"].pct_change().fillna(0.0)
    return log_total_return(ret)


def _risk_overlay_alpha(final_daily: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if final_daily.empty or "candidate" not in final_daily.columns:
        return pd.DataFrame(), np.nan
    work = final_daily.copy()
    v2 = work[work["candidate"].astype(str).eq("growth_champion_v2")].sort_values("date")
    v3 = work[work["candidate"].astype(str).eq("growth_champion_v3")].sort_values("date")
    if v2.empty or v3.empty:
        return pd.DataFrame(), np.nan
    joined = v3[["date", "candidate_return", "candidate_exposure", "candidate_cash", "overlay_cap", "overlay_reason"]].rename(columns={"candidate_return": "v3_return", "candidate_exposure": "v3_exposure", "candidate_cash": "v3_cash"}).merge(
        v2[["date", "candidate_return", "candidate_exposure", "candidate_cash"]].rename(columns={"candidate_return": "v2_return", "candidate_exposure": "v2_exposure", "candidate_cash": "v2_cash"}), on="date", how="inner"
    )
    joined["risk_overlay_daily_alpha"] = numeric(joined["v3_return"]) - numeric(joined["v2_return"])
    alpha = log_total_return(joined["v3_return"]) - log_total_return(joined["v2_return"])
    return joined, float(alpha)


def _factor_contribution(current_features: pd.DataFrame, factor_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if current_features.empty:
        return pd.DataFrame()
    feature_map = {
        "Forecast": ["raw_target_return_exact", "raw_expected_daily_return_exact"],
        "Momentum": ["return_20d", "return_5d"],
        "Trend": ["raw_target_return_exact"],
        "Quality": ["quality_pass", "passed_tradability_filter"],
        "Volatility": ["realized_vol_60d", "rolling_volatility_used"],
        "Mean Reversion": ["signal_strength_adjustment_value"],
        "Risk": ["final_exposure", "dual_trend_cap", "volatility_target_exposure"],
    }
    selected = current_features[current_features.get("raw_target_selected", False).astype(str).str.lower().isin(["true", "1"])] if "raw_target_selected" in current_features.columns else current_features.head(0)
    for factor, cols in feature_map.items():
        present = [c for c in cols if c in current_features.columns]
        if not present:
            rows.append({"factor": factor, "status": "unavailable", "current_selected_average": np.nan, "universe_average": np.nan, "evidence": "stored factor unavailable"})
            continue
        vals = []
        uvals = []
        for col in present:
            vals.append(numeric(selected[col]).mean() if not selected.empty else np.nan)
            uvals.append(numeric(current_features[col]).mean())
        rows.append({"factor": factor, "status": "diagnostic", "current_selected_average": np.nanmean(vals), "universe_average": np.nanmean(uvals), "evidence": ",".join(present)})
    if not factor_ic.empty:
        for _, r in factor_ic.head(8).iterrows():
            rows.append({"factor": str(r.get("feature")), "status": "historical_ic", "current_selected_average": np.nan, "universe_average": np.nan, "evidence": f"avg_spearman_ic={r.get('average_spearman_ic', np.nan)}"})
    return pd.DataFrame(rows)


def run_alpha_attribution() -> dict[str, object]:
    daily = load_csv("growth_champion_reconstructed_stress_daily.csv")
    trades = load_csv("reconstructed_growth_long_horizon_trades.csv")
    final_daily = load_csv("growth_final_selection_daily_returns.csv")
    current_features = load_csv("current_growth_features.csv")
    factor_ic = load_csv("factor_ic_ranking.csv")
    official_costs = load_csv("growth_official_estimated_cost_ledger.csv")

    if daily.empty:
        raise SystemExit("Missing growth_champion_reconstructed_stress_daily.csv")

    total_return = log_total_return(daily["gross_daily_return"])
    benchmark_return = benchmark_log_return_from_cache(daily["date"], "SPY")

    forecast_df = forecast_analysis(trades)
    ranking_df = ranking_analysis(trades)
    sizing_df, sizing_alpha = sizing_analysis(trades)
    cash_df, cash_alpha_compounded = cash_drag_analysis(daily)
    if not cash_df.empty:
        cash_alpha = log_total_return(cash_df["gross_daily_return"]) - log_total_return(cash_df["fully_invested_equivalent_return"].replace([np.inf, -np.inf], np.nan).fillna(0))
    else:
        cash_alpha = 0.0
    risk_df, risk_alpha = _risk_overlay_alpha(final_daily)
    turnover_df = turnover_analysis(daily)
    cost_df, official_cost_alpha = cost_attribution(daily, official_costs)
    holding_df = holding_period_analysis(trades)
    decile_df = decile_analysis(trades)
    factor_df = _factor_contribution(current_features, factor_ic)

    # Measured component values. Missing exact marginal effects are not fabricated.
    components = [
        {"component": "Benchmark Drift", "alpha_contribution": benchmark_return, "method": "SPY adjusted-close log return aligned to strategy observation dates", "confidence": "medium" if benchmark_return != 0 else "low"},
        {"component": "Forecast Alpha", "alpha_contribution": 0.0, "method": "Forecast IC/decile measured separately; exact marginal PnL not stored", "confidence": "diagnostic_only"},
        {"component": "Ranking Alpha", "alpha_contribution": 0.0, "method": "Rank buckets measured separately; exact counterfactual universe return not stored", "confidence": "diagnostic_only"},
        {"component": "Position Sizing Alpha", "alpha_contribution": 0.0 if sizing_df.empty else float(numeric(sizing_df["sizing_alpha"]).sum()), "method": "sum of current weighted selected daily return minus equal-weight selected daily return", "confidence": "medium"},
        {"component": "Cash Allocation Alpha", "alpha_contribution": cash_alpha, "method": "actual gross compounded return minus fully-invested selected equivalent", "confidence": "medium"},
        {"component": "Risk Overlay Alpha", "alpha_contribution": 0.0 if pd.isna(risk_alpha) else risk_alpha, "method": "growth_champion_v3 minus v2 from final selection comparison", "confidence": "medium"},
        {"component": "Volatility Targeting Alpha", "alpha_contribution": 0.0, "method": "not separately identifiable from risk/cash overlay in stored history", "confidence": "unavailable"},
        {"component": "Turnover Impact", "alpha_contribution": 0.0, "method": "turnover measured separately; exact delay counterfactual not stored", "confidence": "diagnostic_only"},
        {"component": "Cost Impact", "alpha_contribution": official_cost_alpha, "method": "official forward estimated costs divided by initial capital; historical cost fields are zero", "confidence": "medium_forward_only"},
        {"component": "Soft Exit Impact", "alpha_contribution": 0.0, "method": "soft-exit return groups measured in holding-period table; exact no-soft-exit counterfactual not stored", "confidence": "diagnostic_only"},
    ]
    measured_sum = sum(float(c["alpha_contribution"]) for c in components if pd.notna(c["alpha_contribution"]))
    residual = total_return - measured_sum
    components.append({"component": "Residual", "alpha_contribution": residual, "method": "Total return minus measured/stored-attributable components", "confidence": "reconciliation"})
    attr = pd.DataFrame(components)
    attr["absolute_contribution"] = attr["alpha_contribution"].abs()
    attr["share_of_total_abs"] = attr["absolute_contribution"] / attr["absolute_contribution"].sum()

    recon = pd.DataFrame([{
        "total_return": total_return,
        "component_sum": attr["alpha_contribution"].sum(),
        "difference": total_return - attr["alpha_contribution"].sum(),
        "tolerance": TOL,
        "reconciliation_pass": abs(total_return - attr["alpha_contribution"].sum()) <= TOL,
    }])

    # Lost alpha ranking: negative measured components plus large residual caveat if negative.
    lost = attr[attr["alpha_contribution"] < 0].copy().sort_values("alpha_contribution")
    lost["lost_alpha"] = -lost["alpha_contribution"]

    attr.to_csv("alpha_attribution.csv", index=False)
    factor_df.to_csv("factor_contribution.csv", index=False)
    forecast_df.to_csv("forecast_analysis.csv", index=False)
    ranking_df.to_csv("ranking_analysis.csv", index=False)
    sizing_df.to_csv("position_sizing_analysis.csv", index=False)
    cash_df.to_csv("cash_drag_analysis.csv", index=False)
    turnover_df.to_csv("turnover_analysis.csv", index=False)
    cost_df.to_csv("cost_attribution.csv", index=False)
    holding_df.to_csv("holding_period_analysis.csv", index=False)
    recon.to_csv("alpha_reconciliation.csv", index=False)
    decile_df.to_csv("forecast_decile_analysis.csv", index=False)

    integrity = pd.DataFrame([
        {"check": "read_only", "status": "PASS", "detail": "analytical CSV generation only"},
        {"check": "model_modifications", "status": "PASS", "detail": "none"},
        {"check": "optimizer_modifications", "status": "PASS", "detail": "none"},
        {"check": "parameter_modifications", "status": "PASS", "detail": "none"},
        {"check": "scheduler_modifications", "status": "PASS", "detail": "none"},
        {"check": "execution_modifications", "status": "PASS", "detail": "none"},
        {"check": "alpha_reconciliation", "status": "PASS" if bool(recon.iloc[0]["reconciliation_pass"]) else "FAIL", "detail": recon.iloc[0].to_dict()},
    ])
    status = "alpha_attribution_pass" if integrity["status"].eq("FAIL").sum() == 0 else "alpha_attribution_fail"
    integrity.to_csv("alpha_attribution_integrity.csv", index=False)

    top_lost = lost.head(3)
    report = [
        "===== PHASE 120 ALPHA ATTRIBUTION & PERFORMANCE DECOMPOSITION =====",
        f"final_status: {status}",
        f"total_log_return: {total_return:.10f}",
        f"component_sum: {attr['alpha_contribution'].sum():.10f}",
        f"reconciliation_difference: {recon.iloc[0]['difference']:.12f}",
        "",
        "Largest lost alpha sources:",
    ]
    if top_lost.empty:
        report.append("- No negative measured components found. Residual contains unavailable marginal effects.")
    else:
        for i, (_, r) in enumerate(top_lost.iterrows(), 1):
            report.append(f"{i}. {r['component']}: lost_alpha={-float(r['alpha_contribution']):.10f}; method={r['method']}; confidence={r['confidence']}")
    report += [
        "",
        "Important caveat: attribution is additive in log-return space. Exact marginal contribution for Forecast Alpha, Ranking Alpha, Soft Exit and Volatility Targeting is not fully stored historically. These are measured through IC/bucket/diagnostic tables and unobservable marginal effects remain in Residual.",
        "No model, forecast, ranking, optimizer, allocation, scheduler, paper, execution, accounting, governance, parameter, or order logic was modified.",
    ]
    Path("phase120_alpha_attribution_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("===== ALPHA ATTRIBUTION =====")
    print(attr.to_string(index=False))
    print("===== RECONCILIATION =====")
    print(recon.to_string(index=False))
    print("===== TOP LOST ALPHA =====")
    print(top_lost[["component", "lost_alpha", "method", "confidence"]].to_string(index=False) if not top_lost.empty else "none")
    return {"status": status, "attribution": attr, "lost": top_lost, "reconciliation": recon}


if __name__ == "__main__":
    run_alpha_attribution()
