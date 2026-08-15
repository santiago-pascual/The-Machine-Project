
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in ["date", "entry_date", "exit_date", "signal_date", "economic_application_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def classify_primary(row: pd.Series) -> str:
    parts = {
        "volatility targeting": row.get("cash_forced_by_volatility_targeting", 0.0),
        "exposure caps": row.get("cash_forced_by_exposure_cap", 0.0),
        "dual trend": row.get("cash_forced_by_dual_trend", 0.0),
        "insufficient candidates": row.get("cash_from_insufficient_candidates", 0.0),
        "optimizer": row.get("cash_from_optimizer", 0.0),
        "concentration controls": row.get("cash_from_concentration_controls", 0.0),
        "governance": row.get("cash_from_governance", 0.0),
        "residual/unexplained": row.get("cash_residual_unexplained", 0.0),
    }
    return max(parts.items(), key=lambda kv: kv[1])[0]


def main() -> None:
    stress = read_csv("growth_champion_reconstructed_stress_daily.csv")
    base = read_csv("reconstructed_growth_long_horizon_daily_returns.csv")
    final = read_csv("growth_final_selection_daily_returns.csv")
    top20 = read_csv("growth_top20_candidates.csv")

    if stress.empty:
        raise SystemExit("Missing growth_champion_reconstructed_stress_daily.csv")

    work = stress.copy().sort_values("date")
    if "rebalance_due" in work.columns:
        rebalances = work[work["rebalance_due"].astype(str).str.lower().isin(["true", "1"])].copy()
    else:
        rebalances = work.copy()

    # Use the 2008 long-horizon window as the matching reconstructed stress source.
    base_2008 = base[base.get("window_start", "").astype(str).eq("2008-01-01")].copy() if not base.empty and "window_start" in base.columns else pd.DataFrame()
    if not base_2008.empty:
        base_2008 = base_2008.sort_values("date")
    v3 = final[final.get("candidate", "").astype(str).eq("growth_champion_v3")].copy() if not final.empty and "candidate" in final.columns else pd.DataFrame()
    v2 = final[final.get("candidate", "").astype(str).eq("growth_champion_v2")].copy() if not final.empty and "candidate" in final.columns else pd.DataFrame()

    rows = []
    for _, r in rebalances.iterrows():
        d = r["date"]
        actual_exposure = safe_float(r.get("exposure", np.nan), 0.0)
        cash = safe_float(r.get("cash", 1.0 - actual_exposure), 1.0 - actual_exposure)
        selected = str(r.get("holdings", ""))
        selected_count = len([x for x in selected.split(",") if x.strip()])
        base_row = base_2008[base_2008["date"].eq(d)].tail(1) if not base_2008.empty else pd.DataFrame()
        v3_row = v3[v3["date"].eq(d)].tail(1) if not v3.empty else pd.DataFrame()
        v2_row = v2[v2["date"].eq(d)].tail(1) if not v2.empty else pd.DataFrame()

        uncapped = safe_float(base_row.iloc[-1].get("uncapped_exposure", np.nan), np.nan) if not base_row.empty else np.nan
        if pd.isna(uncapped):
            uncapped = safe_float(v2_row.iloc[-1].get("candidate_exposure", np.nan), actual_exposure) if not v2_row.empty else actual_exposure
        vol_target_exposure = min(max(uncapped, 0.0), 1.0)
        after_vol = vol_target_exposure
        exposure_cap = 0.60
        after_cap = min(after_vol, exposure_cap)
        dual_cap = safe_float(v3_row.iloc[-1].get("overlay_cap", np.nan), np.nan) if not v3_row.empty else np.nan
        if pd.isna(dual_cap):
            dual_cap = min(after_cap, actual_exposure)
        after_dual = min(after_cap, dual_cap)

        vol_cash = max(0.0, 1.0 - after_vol)
        cap_cash = max(0.0, after_vol - after_cap)
        dual_cash = max(0.0, after_cap - after_dual)

        # Candidate shortage only gets attribution if final exposure is below all risk caps.
        investable_candidates = np.nan
        if not top20.empty and "date" in top20.columns:
            day_candidates = top20[top20["date"].eq(d)]
            if not day_candidates.empty:
                elig = day_candidates
                if "ranking_eligible" in elig.columns:
                    elig = elig[elig["ranking_eligible"].astype(str).str.lower().isin(["true", "1"])]
                investable_candidates = len(elig)
        insufficient = 0.0
        if selected_count < 4 and actual_exposure + 1e-9 < after_dual:
            insufficient = max(0.0, after_dual - actual_exposure)

        optimizer_cash = 0.0  # Growth paper/backtest bypasses optimizer; audited separately in prior phases.
        concentration_cash = 0.0  # No stored active HHI/concentration cap cash field in historical replay.
        governance_cash = 0.0  # Historical backtest not governance-blocked.
        explained = vol_cash + cap_cash + dual_cash + insufficient + optimizer_cash + concentration_cash + governance_cash
        residual = max(0.0, cash - explained)

        rows.append({
            "date": d.date().isoformat(),
            "available_investable_capital": 1.0,
            "actually_invested_capital": actual_exposure,
            "unused_capital": cash,
            "selected_count": selected_count,
            "investable_candidates_available": investable_candidates,
            "uncapped_volatility_target_exposure": uncapped,
            "after_volatility_targeting_exposure": after_vol,
            "exposure_cap": exposure_cap,
            "after_exposure_cap": after_cap,
            "dual_trend_cap": dual_cap,
            "after_dual_trend_exposure": after_dual,
            "cash_forced_by_volatility_targeting": vol_cash,
            "cash_forced_by_exposure_cap": cap_cash,
            "cash_forced_by_dual_trend": dual_cash,
            "cash_from_insufficient_candidates": insufficient,
            "cash_from_optimizer": optimizer_cash,
            "cash_from_concentration_controls": concentration_cash,
            "cash_from_hhi": concentration_cash,
            "cash_from_governance": governance_cash,
            "cash_residual_unexplained": residual,
            "primary_reason": "",
            "evidence": "sequential decomposition: 100% -> vol target -> cap60 -> dual trend -> actual exposure",
        })

    root = pd.DataFrame(rows)
    if root.empty:
        raise SystemExit("No rebalance rows found")
    root["primary_reason"] = root.apply(classify_primary, axis=1)

    reason_cols = {
        "volatility targeting": "cash_forced_by_volatility_targeting",
        "exposure caps": "cash_forced_by_exposure_cap",
        "dual trend": "cash_forced_by_dual_trend",
        "optimizer": "cash_from_optimizer",
        "insufficient candidates": "cash_from_insufficient_candidates",
        "concentration controls": "cash_from_concentration_controls",
        "HHI": "cash_from_hhi",
        "governance": "cash_from_governance",
        "residual/unexplained": "cash_residual_unexplained",
    }
    total_unused = root["unused_capital"].sum()
    breakdown = []
    for reason, col in reason_cols.items():
        amount = root[col].sum() if col in root.columns else 0.0
        breakdown.append({
            "reason": reason,
            "unused_capital_amount": amount,
            "percent_of_unused_capital": amount / total_unused if total_unused else np.nan,
            "rebalance_days_with_reason": int((root[col] > 1e-12).sum()) if col in root.columns else 0,
        })
    breakdown_df = pd.DataFrame(breakdown).sort_values("percent_of_unused_capital", ascending=False)
    util = root[["date", "available_investable_capital", "actually_invested_capital", "unused_capital", "primary_reason", "selected_count", "dual_trend_cap", "uncapped_volatility_target_exposure"]].copy()

    root.to_csv("cash_root_cause.csv", index=False)
    breakdown_df.to_csv("cash_reason_breakdown.csv", index=False)
    util.to_csv("cash_utilization_history.csv", index=False)

    report = [
        "===== CASH DRAG ROOT CAUSE ANALYSIS =====",
        f"rebalance_observations: {len(root)}",
        f"total_unused_capital_sum: {total_unused:.10f}",
        "",
        "Unused capital by reason:",
    ]
    for _, row in breakdown_df.iterrows():
        report.append(f"- {row['reason']}: {row['percent_of_unused_capital']:.4%} ({row['unused_capital_amount']:.6f})")
    report += [
        "",
        "Notes:",
        "- Sequential read-only attribution: volatility target first, exposure cap second, dual trend third.",
        "- Optimizer is 0 because Growth Champion bypasses optimizer in audited implementation.",
        "- Concentration/HHI is 0 because no active historical cash-forced-by-HHI field is stored.",
        "- Insufficient candidates only receives attribution if actual exposure is below all risk caps and selected count < 4.",
        "- No model, optimizer, allocation, volatility targeting, scheduler, ranking, forecast, parameter, paper, or execution logic modified.",
    ]
    Path("cash_drag_root_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("===== CASH ROOT CAUSE =====")
    print(breakdown_df.to_string(index=False))


if __name__ == "__main__":
    main()
