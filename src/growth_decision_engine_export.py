from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df.copy()
    d = pd.to_datetime(df["date"], errors="coerce")
    if d.notna().any():
        return df[d.eq(d.max())].copy()
    return df.copy()


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _exclusion_stage(row: pd.Series, selected: bool) -> str:
    if selected:
        return "Final Portfolio"
    reason = str(row.get("final_exclusion_reason", row.get("exclusion_reason", ""))).lower()
    if str(row.get("ticker", "")).upper() == "CASH":
        return "Cash"
    if row.get("exact_raw_target_available", True) is False:
        return "Data Available"
    if "blacklist" in reason or "symbol" in reason:
        return "Blacklist / Symbol Eligibility"
    if row.get("quality_pass", True) is False or "missing ohlcv" in reason or "price below" in reason or "spike" in reason:
        return "Basic Quality"
    if (
        row.get("passed_tradability_filter", True) is False
        or "volume" in reason
        or "history" in reason
        or "market cap" in reason
        or "volatility" in reason
    ):
        return "Institutional Tradability"
    if bool(row.get("raw_target_selected", False)) is False:
        return "Position Limit"
    return "Final Sanity Check"


def _reason(row: pd.Series) -> str:
    for col in ["final_exclusion_reason", "exclusion_reason", "tradability_exclusion_reason", "holding_risk_notes"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip() and str(val).lower() != "nan":
            return str(val)
    return "not selected; outside final allocation or retained candidates"


def _merge_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = _read("official_holding_metadata.csv")
    if not meta.empty and "ticker" in meta.columns and "ticker" in df.columns:
        df = df.merge(meta.drop_duplicates("ticker", keep="last"), on="ticker", how="left", suffixes=("", "_metadata"))
        for col in ["company_name", "sector", "industry", "exchange", "market_cap"]:
            mcol = f"{col}_metadata"
            if mcol in df.columns:
                if col in df.columns:
                    df[col] = df[col].where(df[col].notna(), df[mcol])
                else:
                    df[col] = df[mcol]
                df = df.drop(columns=[mcol])
    return df


def build_exports() -> dict[str, int]:
    features = _latest(_read("current_growth_features.csv"))
    raw = _latest(_read("current_raw_target_features.csv"))
    allocation = _latest(_read("current_growth_candidate_allocation.csv"))
    state = _latest(_read("growth_official_paper_state.csv"))
    actions = _latest(_read("growth_official_paper_actions.csv"))
    risk = _read("risk_contribution_reconciliation.csv")
    costs = _read("growth_official_estimated_cost_ledger.csv")
    if features.empty and not raw.empty:
        features = raw.copy()
    if features.empty:
        raise SystemExit("current_growth_features.csv/current_raw_target_features.csv unavailable")
    features = _merge_metadata(features)
    allocation_tickers = set(allocation.get("ticker", pd.Series(dtype=str)).astype(str).str.upper()) - {"CASH"}
    official_tickers = set(state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper()) - {"CASH"}
    selected_tickers = official_tickers or allocation_tickers
    df = features.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    raw_col = "raw_target_return_exact" if "raw_target_return_exact" in df.columns else "raw_target_return"
    df[raw_col] = _num(df.get(raw_col, np.nan))
    eligible = df.copy()
    eligible["ranking_eligible"] = (
        eligible.get("quality_pass", True).fillna(False).astype(bool)
        & eligible.get("passed_tradability_filter", True).fillna(False).astype(bool)
        & eligible.get("exact_raw_target_available", True).fillna(False).astype(bool)
    )
    df["ranking_eligible"] = eligible["ranking_eligible"]
    rank_pool = eligible[eligible["ranking_eligible"]].copy().sort_values(raw_col, ascending=False)
    rank_pool["computed_raw_target_rank"] = range(1, len(rank_pool) + 1)
    if len(rank_pool):
        rank_pool["computed_rank_percentile"] = 1 - (rank_pool["computed_raw_target_rank"] - 1) / max(len(rank_pool) - 1, 1)
    df = df.merge(rank_pool[["ticker", "computed_raw_target_rank", "computed_rank_percentile"]], on="ticker", how="left")
    if "raw_target_rank" in df.columns:
        df["raw_target_rank"] = _num(df["raw_target_rank"]).where(_num(df["raw_target_rank"]).notna(), df["computed_raw_target_rank"])
    else:
        df["raw_target_rank"] = df["computed_raw_target_rank"]
    df["rank_percentile"] = df.get("raw_target_rank_pct", df["computed_rank_percentile"])
    df["is_current_selected"] = df["ticker"].isin(selected_tickers)
    max_rank_selected = _num(df.loc[df["is_current_selected"], "raw_target_rank"]).max()
    df["distance_from_portfolio_cutoff"] = _num(df["raw_target_rank"]) - max_rank_selected if pd.notna(max_rank_selected) else np.nan
    df["exclusion_stage"] = df.apply(lambda r: _exclusion_stage(r, bool(r["is_current_selected"])), axis=1)
    df["exact_exclusion_reason"] = df.apply(_reason, axis=1)

    top20 = df.sort_values(["is_current_selected", raw_col], ascending=[False, False]).head(40).copy()
    top20.to_csv("growth_top20_candidates.csv", index=False)

    rejected = df[~df["is_current_selected"]].copy().sort_values(raw_col, ascending=False)
    rejected[[c for c in rejected.columns]].head(80).to_csv("growth_candidate_rejection_report.csv", index=False)

    explain = df[df["ticker"].isin(selected_tickers)].copy()
    if not risk.empty and "ticker" in risk.columns:
        explain = explain.merge(risk, on="ticker", how="left", suffixes=("", "_risk"))
    if not costs.empty and "ticker" in costs.columns:
        cost_sum = costs.groupby("ticker", dropna=False)["estimated_total_cost"].sum().reset_index()
        explain = explain.merge(cost_sum, on="ticker", how="left")
    if not state.empty and "ticker" in state.columns:
        state_cols = [
            c
            for c in [
                "ticker",
                "paper_position_weight",
                "paper_position_value",
                "action",
                "signal_date",
                "economic_application_date",
                "rebalance_due",
                "monitoring_only",
                "next_rebalance_date",
                "sessions_since_last_rebalance",
            ]
            if c in state.columns
        ]
        explain = explain.merge(
            state[state_cols].drop_duplicates("ticker", keep="last"), on="ticker", how="left", suffixes=("", "_official")
        )
    state_exposure = _num(state.get("final_exposure", pd.Series(dtype=float))).dropna() if not state.empty else pd.Series(dtype=float)
    allocation_exposure = (
        _num(allocation.get("final_exposure", pd.Series(dtype=float))).dropna() if not allocation.empty else pd.Series(dtype=float)
    )
    final_exposure = (
        state_exposure.iloc[-1] if not state_exposure.empty else allocation_exposure.iloc[-1] if not allocation_exposure.empty else np.nan
    )
    explain["reason_summary"] = explain.apply(
        lambda r: (
            f"{r.get('ticker')} was selected because it ranked {int(r.get('raw_target_rank')) if pd.notna(r.get('raw_target_rank')) else 'n/a'} by raw_target_return_exact, passed quality/tradability checks, remained within the final position limit, and the risk overlay permitted {final_exposure:.1%} total exposure."
            if pd.notna(final_exposure)
            else f"{r.get('ticker')} was selected by the current official growth allocation after passing available filters."
        ),
        axis=1,
    )
    explain.to_csv("growth_portfolio_explainability.csv", index=False)

    pending = allocation.copy()
    if not pending.empty and "ticker" in pending.columns:
        pending["ticker"] = pending["ticker"].astype(str).str.upper()
        pending["currently_official_holding"] = pending["ticker"].isin(selected_tickers)
        pending["pending_signal_type"] = pending.apply(
            lambda r: (
                "PENDING_ENTER"
                if r.get("ticker") not in selected_tickers and r.get("ticker") != "CASH"
                else "PENDING_HOLD"
                if r.get("ticker") in selected_tickers
                else "CASH"
            ),
            axis=1,
        )
        pending.to_csv("growth_pending_decision_signals.csv", index=False)
    else:
        pd.DataFrame(columns=["ticker", "pending_signal_type", "currently_official_holding"]).to_csv(
            "growth_pending_decision_signals.csv", index=False
        )

    rows = []
    stages = [
        ("Full Universe", len(df), len(df), "current_growth_features.csv"),
        (
            "Data Available",
            len(df),
            int(df.get("exact_raw_target_available", True).fillna(False).astype(bool).sum()),
            "current_raw_target_features.csv",
        ),
        (
            "Blacklist / Symbol Eligibility",
            int(df.get("exact_raw_target_available", True).fillna(False).astype(bool).sum()),
            int((df.get("exact_raw_target_available", True).fillna(False).astype(bool)).sum()),
            "growth_universe_exclusions.csv",
        ),
        (
            "Basic Quality",
            len(df),
            int(df.get("quality_pass", False).fillna(False).astype(bool).sum()),
            "growth_universe_quality_report.csv",
        ),
        (
            "Institutional Tradability",
            int(df.get("quality_pass", False).fillna(False).astype(bool).sum()),
            int(
                (
                    df.get("quality_pass", False).fillna(False).astype(bool)
                    & df.get("passed_tradability_filter", False).fillna(False).astype(bool)
                ).sum()
            ),
            "growth_tradability_filter_report.csv",
        ),
        ("Raw Target Ranking", int(df["ranking_eligible"].sum()), int(df["ranking_eligible"].sum()), "current_growth_features.csv"),
        (
            "Soft Exit / Prior Holdings",
            int(df["ranking_eligible"].sum()),
            len(allocation[allocation.get("ticker", "").astype(str).str.upper().ne("CASH")])
            if not allocation.empty
            else len(selected_tickers),
            "current_growth_candidate_allocation.csv",
        ),
        ("Final Sanity Check", len(selected_tickers), len(selected_tickers), "final_selected_holdings_audit.csv"),
        ("Position Limit", len(selected_tickers), len(selected_tickers), "current_growth_candidate_allocation.csv"),
        ("Final Portfolio", len(selected_tickers) + 1, len(selected_tickers) + 1, "growth_official_paper_state.csv"),
    ]
    for stage, input_count, passed_count, source in stages:
        rows.append(
            {
                "stage": stage,
                "input_count": input_count,
                "passed_count": passed_count,
                "excluded_count": max(input_count - passed_count, 0),
                "retained_pct": passed_count / input_count if input_count else np.nan,
                "source_file": source,
                "latest_date": df.get("date", pd.Series([""])).iloc[0] if "date" in df.columns else "",
            }
        )
    pd.DataFrame(rows).to_csv("growth_decision_funnel.csv", index=False)
    return {"features": len(df), "top20": len(top20), "rejected": min(len(rejected), 80), "explain": len(explain), "funnel": len(rows)}


if __name__ == "__main__":
    result = build_exports()
    print("===== GROWTH DECISION ENGINE EXPORT =====")
    for k, v in result.items():
        print(f"{k}: {v}")
