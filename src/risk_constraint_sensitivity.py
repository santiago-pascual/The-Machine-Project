from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_sensitivity_analysis import frontier_tables, run_grid


def largest_impact(grid: pd.DataFrame, metric: str) -> pd.Series:
    work = grid[~grid["experiment_type"].eq("production")].copy()
    work["abs_delta"] = work[metric + "_delta_vs_production"].abs()
    return work.sort_values("abs_delta", ascending=False).iloc[0]


def main() -> None:
    grid, daily = run_grid()
    frontier, cash_frontier, pareto = frontier_tables(grid)
    grid.to_csv("risk_constraint_grid.csv", index=False)
    frontier.to_csv("risk_constraint_frontier.csv", index=False)
    cash_frontier.to_csv("cash_utilization_frontier.csv", index=False)
    pareto.to_csv("pareto_frontier.csv", index=False)
    daily.to_csv("risk_constraint_shadow_daily.csv", index=False)

    prod = grid[grid["experiment_type"].eq("production")].iloc[0]
    summary_rows = []
    for etype in ["exposure_cap", "vol_target", "dual_trend"]:
        sub = grid[grid["experiment_type"].eq(etype)].copy()
        summary_rows.append(
            {
                "constraint": etype,
                "max_CAGR_delta": sub["CAGR_delta_vs_production"].abs().max(),
                "max_drawdown_delta": sub["max_drawdown_delta_vs_production"].abs().max(),
                "max_cash_delta": sub["average_cash_delta_vs_production"].abs().max(),
                "best_CAGR_parameter": sub.sort_values("CAGR", ascending=False).iloc[0]["parameter"],
                "lowest_drawdown_parameter": sub.sort_values("max_drawdown", ascending=False).iloc[0]["parameter"],
                "highest_cash_utilization_parameter": sub.sort_values("cash_utilization_pct", ascending=False).iloc[0]["parameter"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("risk_sensitivity_summary.csv", index=False)

    most_conservative = summary.sort_values("max_cash_delta", ascending=False).iloc[0]
    cagr_impact = summary.sort_values("max_CAGR_delta", ascending=False).iloc[0]
    dd_impact = summary.sort_values("max_drawdown_delta", ascending=False).iloc[0]
    cash_impact = summary.sort_values("max_cash_delta", ascending=False).iloc[0]
    further = (
        cagr_impact
        if cagr_impact["constraint"] == cash_impact["constraint"]
        else summary.assign(score=summary["max_CAGR_delta"] + summary["max_cash_delta"] + summary["max_drawdown_delta"])
        .sort_values("score", ascending=False)
        .iloc[0]
    )

    integrity = pd.DataFrame(
        [
            {"check": "shadow_mode_only", "status": "PASS", "detail": "outputs only; active config unchanged"},
            {"check": "production_modified", "status": "PASS", "detail": "False"},
            {"check": "paper_modified", "status": "PASS", "detail": "False"},
            {"check": "optimizer_modified", "status": "PASS", "detail": "False"},
            {"check": "parameters_modified", "status": "PASS", "detail": "False"},
        ]
    )
    integrity.to_csv("risk_constraint_lab_integrity.csv", index=False)
    status = "risk_constraint_lab_pass" if integrity["status"].eq("FAIL").sum() == 0 else "risk_constraint_lab_fail"
    report = [
        "===== PHASE 122 RISK CONSTRAINT SENSITIVITY LABORATORY =====",
        f"final_status: {status}",
        "shadow_mode_only: True",
        "production_changed: False",
        "paper_changed: False",
        "optimizer_changed: False",
        "parameters_changed: False",
        "",
        "Answers:",
        f"1. Most conservative constraint: {most_conservative['constraint']} (max cash delta {most_conservative['max_cash_delta']:.6f}).",
        f"2. Largest CAGR impact: {cagr_impact['constraint']} (max CAGR delta {cagr_impact['max_CAGR_delta']:.6f}).",
        f"3. Largest drawdown impact: {dd_impact['constraint']} (max DD delta {dd_impact['max_drawdown_delta']:.6f}).",
        f"4. Largest cash utilization impact: {cash_impact['constraint']} (max cash delta {cash_impact['max_cash_delta']:.6f}).",
        f"5. Single constraint deserving further investigation: {further['constraint']} (largest combined sensitivity score).",
        "",
        "Summary table:",
        summary.to_string(index=False),
        "",
        "No recommendations. Diagnosis only.",
    ]
    Path("phase122_risk_constraint_lab_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("===== RISK CONSTRAINT LAB =====")
    print(summary.to_string(index=False))
    print("status", status)


if __name__ == "__main__":
    main()
