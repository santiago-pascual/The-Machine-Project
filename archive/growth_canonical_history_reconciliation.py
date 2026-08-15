from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_DAILY = "growth_champion_canonical_daily.csv"
RECON_DAILY = "growth_champion_reconstructed_stress_daily.csv"
OUT_METRIC_RECON = "growth_metric_reconciliation.csv"
OUT_SOURCE_AUDIT = "growth_source_series_audit.csv"
OUT_RESULTS = "growth_canonical_results.csv"
OUT_GOV = "growth_canonical_governance.csv"
OUT_REPORT = "growth_canonical_report.txt"

PAPER_PERF = "growth_candidate_paper_performance.csv"
PAPER_STATE = "growth_candidate_paper_state.csv"
PAPER_TRADES = "growth_candidate_paper_trades.csv"
BENCH_DAILY = "benchmark_daily_returns.csv"
FORECAST = "forecast_history.csv"
SCHEDULE = "growth_rebalance_schedule.csv"

PRIOR_FILES = [
    "growth_final_selection_daily_returns.csv",
    "growth_crisis_overlay_daily_returns.csv",
    "reconstructed_growth_long_horizon_daily_returns.csv",
    "production_parity_growth_daily_returns.csv",
    "after_costs_equity_curves.csv",
    "parameter_stability_map.csv",
]


def read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def dateify(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out.dropna(subset=[col]).sort_values(col)


def num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def periods_per_year(dates: pd.Series) -> float:
    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(d) < 2:
        return 252.0
    step = d.diff().dt.days.dropna().median()
    return float(365.25 / step) if pd.notna(step) and step > 0 else 252.0


def metrics(name: str, df: pd.DataFrame, return_col: str = "net_daily_return", equity_col: str | None = None) -> dict[str, object]:
    if df.empty:
        return {"series": name, "observations": 0}
    data = dateify(df)
    if equity_col and equity_col in data.columns:
        equity = pd.to_numeric(data[equity_col], errors="coerce").dropna()
        returns = equity.pct_change().dropna()
        total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 and equity.iloc[0] else np.nan
    elif return_col in data.columns:
        returns = pd.to_numeric(data[return_col], errors="coerce").dropna()
        equity = (1 + returns).cumprod()
        total = float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan
    else:
        return {"series": name, "observations": 0}
    if returns.empty:
        return {"series": name, "observations": 0}
    ppy = periods_per_year(data["date"] if "date" in data.columns else pd.Series(dtype=str))
    years = max(len(returns) / ppy, 1e-9)
    cagr = float((1 + total) ** (1 / years) - 1) if pd.notna(total) and total > -1 else np.nan
    vol = float(returns.std(ddof=0) * np.sqrt(ppy))
    sharpe = float((returns.mean() * ppy) / vol) if vol > 0 else np.nan
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(ppy) if (returns < 0).any() else np.nan
    sortino = float((returns.mean() * ppy) / downside) if pd.notna(downside) and downside > 0 else np.nan
    eq = equity.reset_index(drop=True)
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min()) if not dd.empty else np.nan
    return {
        "series": name,
        "observations": len(returns),
        "start_date": data["date"].min().strftime("%Y-%m-%d") if "date" in data.columns and not data.empty else "",
        "end_date": data["date"].max().strftime("%Y-%m-%d") if "date" in data.columns and not data.empty else "",
        "total_return": total,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": cagr / abs(max_dd) if pd.notna(cagr) and max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "mean_return": float(returns.mean()),
    }


def holdings_by_date(state: pd.DataFrame) -> pd.DataFrame:
    if state.empty:
        return pd.DataFrame(columns=["date", "holdings", "weights"])
    st = dateify(state)
    rows = []
    for dt, g in st.groupby("date"):
        noncash = g[g["ticker"].astype(str).ne("CASH")].copy()
        holdings = ",".join(noncash["ticker"].astype(str).tolist())
        weights = ",".join(f"{float(x):.10f}" for x in pd.to_numeric(noncash.get("paper_position_weight", pd.Series(dtype=float)), errors="coerce"))
        rows.append({"date": dt, "holdings": holdings, "weights": weights})
    return pd.DataFrame(rows)


def first_exact_date() -> tuple[pd.Timestamp | None, pd.DataFrame]:
    perf = dateify(read(PAPER_PERF))
    state = dateify(read(PAPER_STATE))
    forecast = dateify(read(FORECAST))
    rows = []
    if perf.empty:
        return None, pd.DataFrame()
    for dt, row in perf.drop_duplicates("date", keep="last").set_index("date").iterrows():
        fh = forecast[forecast["date"].eq(dt)] if not forecast.empty else pd.DataFrame()
        st = state[state["date"].eq(dt)] if not state.empty else pd.DataFrame()
        exact_raw = bool(str(row.get("raw_target_feature_source", "")) == "raw_target_return_exact")
        if not fh.empty and "raw_target_return_exact" in fh.columns:
            exact_raw = exact_raw and pd.to_numeric(fh["raw_target_return_exact"], errors="coerce").notna().any()
        fresh_ohlcv = bool(str(row.get("data_source", "")) == "current_growth_candidate_allocation")
        filters_current = not st.empty
        fresh_vol = bool(pd.notna(row.get("vol_target_exposure", np.nan)))
        scheduler = "rebalance_due" in perf.columns or Path(SCHEDULE).exists()
        ok = all([exact_raw, fresh_ohlcv, filters_current, fresh_vol, scheduler])
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "raw_target_return_exact": exact_raw,
            "fresh_ohlcv": fresh_ohlcv,
            "current_filters_present": filters_current,
            "fresh_volatility_calculation": fresh_vol,
            "exact_rebalance_scheduler": scheduler,
            "all_exact_inputs": ok,
        })
    audit = pd.DataFrame(rows)
    exact_dates = pd.to_datetime(audit.loc[audit["all_exact_inputs"], "date"], errors="coerce") if not audit.empty else pd.Series(dtype="datetime64[ns]")
    return (exact_dates.min().normalize() if not exact_dates.empty else None), audit


def build_exact_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    first_dt, input_audit = first_exact_date()
    perf = dateify(read(PAPER_PERF))
    state = dateify(read(PAPER_STATE))
    trades = dateify(read(PAPER_TRADES))
    bench = dateify(read(BENCH_DAILY))
    if perf.empty or first_dt is None:
        return pd.DataFrame(), input_audit
    perf = perf[perf["date"].ge(first_dt)].drop_duplicates("date", keep="last").copy()
    hold = holdings_by_date(state)
    out = perf.merge(hold, on="date", how="left")
    if not bench.empty:
        cols = [c for c in ["date", "spy_daily_return", "qqq_daily_return"] if c in bench.columns]
        out = out.merge(bench[cols], on="date", how="left")
    else:
        out["spy_daily_return"] = np.nan
        out["qqq_daily_return"] = np.nan
    cost_cols = ["commissions", "spread_cost", "slippage", "market_impact"]
    for c in cost_cols:
        out[c] = 0.0
    out["model_version"] = "growth_champion_final"
    out["data_mode"] = "exact"
    out["signal_date"] = out["date"]
    out["economic_application_date"] = out["date"].shift(-1)
    out["rebalance_due"] = out.get("rebalance_due", False)
    out["gross_daily_return"] = pd.to_numeric(out.get("daily_return", 0.0), errors="coerce").fillna(0.0)
    out["turnover"] = pd.to_numeric(out.get("turnover", 0.0), errors="coerce").fillna(0.0)
    out["net_daily_return"] = out["gross_daily_return"] - out[cost_cols].sum(axis=1)
    out["gross_equity"] = (1 + out["gross_daily_return"]).cumprod()
    out["net_equity"] = (1 + out["net_daily_return"]).cumprod()
    out["drawdown"] = out["net_equity"] / out["net_equity"].cummax() - 1
    out = out.rename(columns={"cash_weight": "cash"})
    canonical_cols = [
        "date", "model_version", "data_mode", "rebalance_due", "signal_date", "economic_application_date",
        "holdings", "weights", "exposure", "cash", "gross_daily_return", "turnover", "commissions",
        "spread_cost", "slippage", "market_impact", "net_daily_return", "gross_equity", "net_equity", "drawdown",
        "spy_daily_return", "qqq_daily_return",
    ]
    return out[[c for c in canonical_cols if c in out.columns]].copy(), input_audit


def build_reconstructed_history() -> pd.DataFrame:
    crisis = dateify(read("growth_crisis_overlay_daily_returns.csv"))
    if crisis.empty:
        recon = dateify(read("reconstructed_growth_long_horizon_daily_returns.csv"))
        if recon.empty:
            return pd.DataFrame()
        sub = recon[recon.get("window_start", "").astype(str).eq("2008-01-01")].copy()
        ret_col = "return"
        exposure_col = "target_exposure"
        cash_col = "cash_weight"
    else:
        sub = crisis[(crisis.get("window_start", "").astype(str).eq("2008-01-01")) & (crisis.get("overlay", "").astype(str).eq("dual_trend_filter"))].copy()
        ret_col = "overlay_return" if "overlay_return" in sub.columns else "return"
        exposure_col = "overlay_exposure" if "overlay_exposure" in sub.columns else "target_exposure"
        cash_col = "overlay_cash" if "overlay_cash" in sub.columns else "cash_weight"
    if sub.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["date"] = sub["date"]
    out["model_version"] = "growth_champion_final_reconstructed_stress"
    out["data_mode"] = "reconstructed"
    out["rebalance_due"] = True
    out["signal_date"] = sub["date"]
    out["economic_application_date"] = sub.get("entry_date", sub["date"])
    out["holdings"] = sub.get("selected_tickers", "")
    out["weights"] = ""
    out["exposure"] = pd.to_numeric(sub.get(exposure_col, np.nan), errors="coerce")
    out["cash"] = pd.to_numeric(sub.get(cash_col, np.nan), errors="coerce")
    out["gross_daily_return"] = pd.to_numeric(sub[ret_col], errors="coerce").fillna(0.0)
    out["turnover"] = pd.to_numeric(sub.get("turnover", 0.0), errors="coerce").fillna(0.0)
    out["commissions"] = 0.0
    out["spread_cost"] = 0.0
    out["slippage"] = 0.0
    out["market_impact"] = 0.0
    out["net_daily_return"] = out["gross_daily_return"]
    out["gross_equity"] = (1 + out["gross_daily_return"]).cumprod()
    out["net_equity"] = (1 + out["net_daily_return"]).cumprod()
    out["drawdown"] = out["net_equity"] / out["net_equity"].cummax() - 1
    out["spy_daily_return"] = np.nan
    out["qqq_daily_return"] = np.nan
    return out


def source_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    audits = []
    metric_rows = []
    for f in PRIOR_FILES:
        df = dateify(read(f))
        if df.empty:
            audits.append({"source_file": f, "exists": Path(f).exists(), "rows": 0, "issue": "missing_or_empty"})
            continue
        cols = df.columns.tolist()
        ret_col = None
        equity_col = None
        scenario = ""
        data_mode = "unknown"
        if f == "growth_final_selection_daily_returns.csv":
            df = df[df.get("candidate", "").astype(str).eq("growth_champion_v3")]
            ret_col = "candidate_return"
            data_mode = "reconstructed_v2_v3_selection"
            scenario = "growth_champion_v3"
        elif f == "growth_crisis_overlay_daily_returns.csv":
            df = df[(df.get("window_start", "").astype(str).eq("2008-01-01")) & (df.get("overlay", "").astype(str).eq("dual_trend_filter"))]
            ret_col = "overlay_return" if "overlay_return" in df.columns else "return"
            data_mode = "reconstructed_stress_overlay"
            scenario = "dual_trend_filter_2008_window"
        elif f == "reconstructed_growth_long_horizon_daily_returns.csv":
            df = df[df.get("window_start", "").astype(str).eq("2008-01-01")]
            ret_col = "return"
            data_mode = "price_only_reconstruction"
            scenario = "2008_window"
        elif f == "production_parity_growth_daily_returns.csv":
            ret_col = "return"
            data_mode = "production_parity_pre_phase94_2022_2026"
            scenario = "pre_scheduler_exact/proxy_mix"
        elif f == "after_costs_equity_curves.csv":
            if "scenario" in df.columns:
                scen = "Y_1.0"
                matches = df[df["scenario"].astype(str).str.contains("Y=1", regex=False, na=False)]
                if matches.empty:
                    matches = df[df["scenario"].astype(str).str.contains("Y_1", regex=False, na=False)]
                df = matches if not matches.empty else df.drop_duplicates("date", keep="first")
            ret_col = "net_return" if "net_return" in df.columns else None
            equity_col = "net_equity" if "net_equity" in df.columns else None
            data_mode = "after_costs_execution_model"
            scenario = "selected_cost_scenario_or_first_available"
        elif f == "parameter_stability_map.csv":
            row = df[df.get("is_current_config", False).astype(str).str.lower().isin(["true", "1"])]
            if row.empty:
                row = df.head(1)
            r = row.iloc[0].to_dict()
            metric_rows.append({"source_file": f, "series": "current_config_reported_row", **{k: r.get(k, np.nan) for k in ["CAGR", "Sharpe", "Sortino", "Calmar", "max_drawdown", "average_exposure", "average_turnover"]}})
            audits.append({"source_file": f, "exists": True, "rows": len(df), "start_date": "n/a", "end_date": "n/a", "data_mode": "parameter_grid_summary_not_return_series", "return_column": "n/a", "scenario": "current_config", "mismatch_reason": "Summary grid over reconstructed actions; not a canonical daily return source."})
            continue
        m = metrics(f, df, return_col=ret_col or "return", equity_col=equity_col)
        metric_rows.append({"source_file": f, **m})
        start = df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns and not df.empty else ""
        end = df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns and not df.empty else ""
        mismatch = explain_mismatch(f, data_mode, ret_col, equity_col)
        audits.append({"source_file": f, "exists": True, "rows": len(df), "start_date": start, "end_date": end, "data_mode": data_mode, "return_column": ret_col or "", "equity_column": equity_col or "", "scenario": scenario, "mismatch_reason": mismatch})
    return pd.DataFrame(audits), pd.DataFrame(metric_rows)


def explain_mismatch(f: str, mode: str, ret_col: str | None, equity_col: str | None) -> str:
    reasons = {
        "growth_final_selection_daily_returns.csv": "Reconstructed v2/v3 comparison; ends 2026-06-10; uses reconstructed OHLCV signal path, not current exact forecast_history nor Phase94 paper cadence history.",
        "growth_crisis_overlay_daily_returns.csv": "Long-horizon reconstructed stress overlay; dual trend rows are 5-session reconstructed returns, not exact current-model forecast snapshots.",
        "reconstructed_growth_long_horizon_daily_returns.csv": "Price-only reconstruction using fallback/current target logic; explicitly non-production-parity.",
        "production_parity_growth_daily_returns.csv": "Older production-parity replay ending 2026-05-15 before Phase94 scheduler/paper reconciliation and institutional filters were finalized.",
        "after_costs_equity_curves.csv": "Execution-cost model over reconstructed trade source; net returns include impact/cost assumptions and use a different window/source series.",
    }
    return reasons.get(f, f"Different mode={mode}, return_col={ret_col}, equity_col={equity_col}")


def add_benchmark_alpha(result_rows: list[dict[str, object]], canonical: pd.DataFrame) -> None:
    if canonical.empty:
        return
    for bench, col in [("SPY", "spy_daily_return"), ("QQQ", "qqq_daily_return")]:
        if col not in canonical.columns or canonical[col].isna().all():
            continue
        aligned = canonical[["net_daily_return", col]].dropna()
        if len(aligned) < 2:
            continue
        excess = aligned["net_daily_return"] - aligned[col]
        ppy = periods_per_year(canonical["date"])
        ir = float((excess.mean() * ppy) / (excess.std(ddof=0) * np.sqrt(ppy))) if excess.std(ddof=0) > 0 else np.nan
        result_rows.append({"series": f"canonical_exact_alpha_vs_{bench}", "observations": len(excess), "mean_excess_return": float(excess.mean()), "annualized_alpha": float(excess.mean()*ppy), "information_ratio": ir})


def main() -> None:
    exact, input_audit = build_exact_history()
    recon = build_reconstructed_history()
    exact.to_csv(CANONICAL_DAILY, index=False)
    recon.to_csv(RECON_DAILY, index=False)
    source, prior_metrics = source_audit()
    source.to_csv(OUT_SOURCE_AUDIT, index=False)
    prior_metrics.to_csv(OUT_METRIC_RECON, index=False)

    result_rows = []
    if not exact.empty:
        result_rows.append(metrics("canonical_exact_history_gross", exact, "gross_daily_return"))
        result_rows.append(metrics("canonical_exact_history_net", exact, "net_daily_return"))
        result_rows[-1]["average_turnover"] = float(pd.to_numeric(exact.get("turnover", pd.Series(dtype=float)), errors="coerce").mean())
        result_rows[-1]["cost_drag"] = float((pd.to_numeric(exact.get("gross_daily_return", 0), errors="coerce") - pd.to_numeric(exact.get("net_daily_return", 0), errors="coerce")).sum())
        add_benchmark_alpha(result_rows, exact)
    if not recon.empty:
        result_rows.append(metrics("reconstructed_stress_history_gross", recon, "gross_daily_return"))
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_RESULTS, index=False)

    exact_ready = not exact.empty and len(exact) >= 2
    recon_ready = not recon.empty
    classification = "canonical_history_ready" if exact_ready and recon_ready else ("exact_history_ready" if exact_ready else ("reconstructed_only" if recon_ready else "metrics_unreconciled"))
    first_exact = exact["date"].min().strftime("%Y-%m-%d") if exact_ready else ""
    gov = pd.DataFrame([{
        "classification": classification,
        "first_exact_input_date": first_exact,
        "canonical_exact_rows": len(exact),
        "reconstructed_stress_rows": len(recon),
        "source_files_audited": len(source),
        "production_changed": False,
        "paper_changed": False,
        "parameters_changed": False,
        "warning": "Exact history is short; reconstructed stress history is non-production-parity and must not be mixed with exact metrics." if exact_ready else "Exact current-model history unavailable or insufficient.",
    }])
    gov.to_csv(OUT_GOV, index=False)

    lines = [
        "===== GROWTH CANONICAL HISTORY REPORT =====",
        f"governance: {classification}",
        f"first exact current-model input date: {first_exact}",
        f"canonical exact rows: {len(exact)}",
        f"reconstructed stress rows: {len(recon)}",
        "",
        "Exact history uses only current paper/forecast rows with raw_target_return_exact and current scheduler fields.",
        "Reconstructed stress history is separate and labelled reconstructed; it is not production parity.",
        "No single final performance number should be quoted across modes; use growth_canonical_results.csv by data_mode.",
        "",
        "Mismatch explanations are saved in growth_source_series_audit.csv.",
        "Prior reported metrics are saved in growth_metric_reconciliation.csv.",
    ]
    if not results.empty:
        lines.append("\nCanonical/reconstructed metric summary:")
        lines.append(results.to_string(index=False))
    Path(OUT_REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    input_audit.to_csv("growth_exact_input_availability_audit.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
