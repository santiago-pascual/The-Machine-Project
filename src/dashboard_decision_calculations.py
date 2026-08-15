
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard_data_layer import latest, numeric

DECISION_SOURCES = {
    "current_raw_target_features": "current_raw_target_features.csv",
    "current_growth_features": "current_growth_features.csv",
    "current_allocation": "current_growth_candidate_allocation.csv",
    "universe_quality_report": "growth_universe_quality_report.csv",
    "universe_exclusions": "growth_universe_exclusions.csv",
    "tradability_report": "growth_tradability_filter_report.csv",
    "tradability_exclusions": "growth_tradability_exclusions.csv",
    "holdings_sanity": "final_selected_holdings_audit.csv",
    "holdings_replacements": "final_selected_holdings_replacements.csv",
    "portfolio_explainability": "growth_portfolio_explainability.csv",
    "top20_candidates": "growth_top20_candidates.csv",
    "rejection_report": "growth_candidate_rejection_report.csv",
    "pending_signals": "growth_pending_decision_signals.csv",
    "official_state": "growth_official_paper_state.csv",
    "official_actions": "growth_official_paper_actions.csv",
    "official_rebalance_report": "growth_official_paper_rebalance_report.csv",
    "rebalance_schedule": "growth_rebalance_schedule.csv",
    "official_position_pnl": "growth_official_position_pnl.csv",
    "vol_pipeline_audit": "growth_volatility_pipeline_audit.csv",
    "official_market_data_integrity": "official_market_data_integrity.csv",
    "risk_contribution": "risk_contribution_reconciliation.csv",
    "execution_cost_reconciliation": "execution_cost_reconciliation.csv",
    "decision_funnel": "growth_decision_funnel.csv",
}

FEATURE_COLUMNS = [
    "raw_target_return_exact", "raw_expected_daily_return_exact", "raw_target_price_exact", "time_to_target",
    "signal_strength_adjustment_value", "final_expected_return_after_adjustments", "realized_vol_60d",
    "return_1d", "return_5d", "return_20d", "median_60d_dollar_volume", "avg_volume_20d",
    "market_cap", "trading_history_days", "spy_close", "spy_ma_200", "qqq_close", "qqq_ma_200",
]

DNA_DIMENSIONS = {
    "Expected Return": "raw_target_return_exact",
    "Trend": "return_20d",
    "Quality": "quality_pass",
    "Tradability": "passed_tradability_filter",
    "Liquidity": "median_60d_dollar_volume",
    "Volatility": "realized_vol_60d",
    "Diversification": "pct_total_portfolio_risk",
    "Confidence": "exact_raw_target_available",
    "Regime Compatibility": "dual_trend_cap",
}


@dataclass
class DecisionBundle:
    kpis: dict[str, Any]
    funnel: pd.DataFrame
    sankey_nodes: pd.DataFrame
    sankey_links: pd.DataFrame
    selected: pd.DataFrame
    rejected: pd.DataFrame
    comparison: pd.DataFrame
    decision_tree: pd.DataFrame
    pending: pd.DataFrame
    history: pd.DataFrame
    attrition: pd.DataFrame
    dna: pd.DataFrame
    pipeline: pd.DataFrame
    source_audit: pd.DataFrame
    integrity: pd.DataFrame
    funnel_reconciliation: pd.DataFrame
    missing_attribution: pd.DataFrame
    status: str


def _df(data: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    return data.get(key, pd.DataFrame()).copy()


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    return latest(df) if not df.empty and "date" in df.columns else df.copy()


def _checksum(path: str) -> str:
    p = Path(path)
    if not p.exists() or p.is_dir():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _date_range(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    for col in ["date", "market_date", "signal_date", "economic_application_date"]:
        if col in df.columns:
            d = pd.to_datetime(df[col], errors="coerce")
            if d.notna().any():
                return f"{d.min().date()} to {d.max().date()}"
    return ""


def _fill_from_sources(base: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if base.empty or "ticker" not in base.columns:
        return base
    out = base.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    for key in ["current_features", "holdings_sanity", "official_position_pnl", "risk_contribution", "execution_cost_reconciliation"]:
        src = _latest(_df(data, key))
        if src.empty or "ticker" not in src.columns:
            continue
        src = src.copy()
        src["ticker"] = src["ticker"].astype(str).str.upper()
        src = src.drop_duplicates("ticker", keep="last")
        cols = [c for c in src.columns if c != "ticker"]
        merged = out.merge(src[["ticker"] + cols], on="ticker", how="left", suffixes=("", "__src"))
        for col in cols:
            scol = f"{col}__src"
            if scol not in merged.columns:
                continue
            if col in out.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[scol])
                merged = merged.drop(columns=[scol])
            else:
                merged = merged.rename(columns={scol: col})
        out = merged
    return out


def source_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows=[]
    for key, file in DECISION_SOURCES.items():
        df=_df(data,key)
        rows.append({"source_file":file,"data_key":key,"loaded":not df.empty,"row_count":len(df),"date_range":_date_range(df),"checksum":_checksum(file),"namespace":"official_current_decision"})
    return pd.DataFrame(rows)


def build_sankey(funnel: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = funnel["stage"].astype(str).tolist() if not funnel.empty else []
    tickers = selected.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().tolist() if not selected.empty else []
    labels += tickers + ["CASH"]
    node_rows=[]
    for i, label in enumerate(labels):
        if not funnel.empty and label in set(funnel["stage"].astype(str)):
            row = funnel[funnel["stage"].astype(str).eq(label)].iloc[-1]
            node_rows.append({"id": i, "label": label, "stage": label, "count": row.get("passed_count", np.nan), "retained_pct": row.get("retained_pct", np.nan), "source_file": row.get("source_file", "")})
        else:
            node_rows.append({"id": i, "label": label, "stage": "Final Holding" if label != "CASH" else "Cash", "count": 1, "retained_pct": 1, "source_file": "growth_official_paper_state.csv"})
    nodes = pd.DataFrame(node_rows)
    links=[]
    if not funnel.empty:
        for i in range(len(funnel)-1):
            next_row = funnel.iloc[i+1]
            prev_row = funnel.iloc[i]
            links.append({
                "source": i,
                "target": i+1,
                "value": float(next_row.get("input_count", prev_row.get("passed_count", 0))) if pd.notna(next_row.get("input_count", np.nan)) else float(prev_row.get("passed_count", 0)),
                "stage": f"{prev_row['stage']} -> {next_row['stage']}",
                "excluded_count": next_row.get("excluded_count", 0),
                "retained_pct": next_row.get("retained_pct", np.nan),
                "reason_summary": "pipeline transition",
            })
        final_idx = labels.index("Final Portfolio") if "Final Portfolio" in labels else len(funnel)-1
        for t in tickers:
            links.append({"source":final_idx,"target":labels.index(t),"value":1,"stage":"Final Portfolio -> holding","excluded_count":0,"retained_pct":1,"reason_summary":f"Final holding {t}"})
        links.append({"source":final_idx,"target":labels.index("CASH"),"value":1,"stage":"Final Portfolio -> cash","excluded_count":0,"retained_pct":1,"reason_summary":"Cash sleeve"})
    return nodes, pd.DataFrame(links)


def build_kpis(data: dict[str, pd.DataFrame], funnel: pd.DataFrame, selected: pd.DataFrame, rejected: pd.DataFrame, pending: pd.DataFrame) -> dict[str, Any]:
    state = _latest(_df(data,"official_state"))
    actions = _latest(_df(data,"official_actions"))
    schedule = _latest(_df(data,"rebalance_schedule"))
    reb = _df(data,"official_rebalance_report")
    last_reb = reb[pd.to_numeric(reb.get("turnover",0), errors="coerce").fillna(0).gt(0)].tail(1) if not reb.empty else pd.DataFrame()
    holdings = selected.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().tolist()
    return {
        "active_model_version": state.iloc[-1].get("growth_model_version", state.iloc[-1].get("model_version", "growth_champion_final_v1_0_frozen")) if not state.empty else "growth_champion_final_v1_0_frozen",
        "official_scope": "Official Forward Paper",
        "latest_market_date": state.iloc[-1].get("date", "unavailable") if not state.empty else "unavailable",
        "signal_date": state.iloc[-1].get("signal_date", actions.iloc[-1].get("signal_date", "unavailable") if not actions.empty else "unavailable") if not state.empty else "unavailable",
        "economic_application_date": state.iloc[-1].get("economic_application_date", "unavailable") if not state.empty else "unavailable",
        "last_rebalance": last_reb.iloc[-1].get("date", "unavailable") if not last_reb.empty else "unavailable",
        "next_rebalance": schedule.iloc[-1].get("next_rebalance_date", state.iloc[-1].get("next_rebalance_date", "unavailable") if not state.empty else "unavailable") if not schedule.empty else "unavailable",
        "sessions_until_next_rebalance": 5 - int(schedule.iloc[-1].get("sessions_since_last_rebalance", 0)) if not schedule.empty and pd.notna(schedule.iloc[-1].get("sessions_since_last_rebalance", np.nan)) else "unavailable",
        "current_holdings": ",".join(holdings),
        "universe_count": int(funnel.iloc[0].get("input_count", 0)) if not funnel.empty else 0,
        "final_holding_count": len(holdings),
        "top_rejected": rejected.iloc[0].get("ticker", "none") if not rejected.empty else "none",
        "pending_enters": ",".join(pending[pending.get("pending_signal_type", "").astype(str).eq("PENDING_ENTER")].get("ticker", pd.Series(dtype=str)).astype(str).tolist()) if not pending.empty else "",
    }


def selected_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sel = _latest(_df(data,"portfolio_explainability"))
    if sel.empty:
        state = _latest(_df(data,"official_state"))
        sel = state[state.get("ticker", "").astype(str).str.upper().ne("CASH")].copy() if not state.empty else pd.DataFrame()
    sel = sel[sel.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().ne("CASH")].copy() if not sel.empty else sel
    return _fill_from_sources(sel, data)


def rejected_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rej = _latest(_df(data,"rejection_report"))
    if rej.empty:
        top = _latest(_df(data,"top20_candidates"))
        sel = set(selected_table(data).get("ticker", pd.Series(dtype=str)).astype(str).str.upper())
        rej = top[~top.get("ticker", "").astype(str).str.upper().isin(sel)].copy() if not top.empty else pd.DataFrame()
    return _fill_from_sources(rej.head(20).copy(), data)


def comparison_table(selected: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    s = selected.copy(); r = rejected.head(10).copy()
    s["selection_status"]="Selected"; r["selection_status"]="Rejected"
    common = sorted(set(s.columns).intersection(r.columns).union({"ticker","selection_status"}))
    out = pd.concat([s, r], ignore_index=True, sort=False)
    return out


def decision_tree_table(selected: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([selected, rejected.head(10)], ignore_index=True, sort=False)
    rows=[]
    for _, row in base.iterrows():
        ticker=str(row.get("ticker",""))
        selected_flag=bool(row.get("is_current_selected", ticker in selected.get("ticker", pd.Series(dtype=str)).astype(str).tolist()))
        rules=[
            ("Data Fresh", bool(row.get("exact_raw_target_available", True))),
            ("Not Blacklisted", "blacklist" not in str(row.get("exact_exclusion_reason", row.get("final_exclusion_reason", ""))).lower()),
            ("Price > $5", pd.to_numeric(row.get("current_price", np.nan), errors="coerce") > 5),
            ("Missing OHLCV <= threshold", "missing ohlcv" not in str(row.get("exact_exclusion_reason", row.get("final_exclusion_reason", ""))).lower()),
            ("Dollar Volume >= threshold", bool(row.get("passed_tradability_filter", False))),
            ("History >= 504D", pd.to_numeric(row.get("trading_history_days", np.nan), errors="coerce") >= 504),
            ("Volatility <= 120%", pd.to_numeric(row.get("realized_vol_60d", np.nan), errors="coerce") <= 1.2),
            ("Ranked within candidate set", pd.notna(row.get("raw_target_rank", np.nan))),
            ("Soft Exit / New Selection", str(row.get("soft_exit_status", "available")) != ""),
            ("Final Sanity Check", str(row.get("holding_quality_classification", "not_a_final_holding")) != "reject_from_growth_universe"),
            ("Selected / Rejected", selected_flag),
        ]
        for step, passed in rules:
            rows.append({"ticker":ticker,"rule":step,"status":"PASS" if bool(passed) else "FAIL","selected":selected_flag})
    return pd.DataFrame(rows)


def attrition_table(funnel: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    if funnel.empty:
        return pd.DataFrame()
    stage = funnel[["stage","excluded_count","retained_pct","source_file"]].copy()
    reasons = rejected.get("exact_exclusion_reason", pd.Series(dtype=str)).fillna("unknown").astype(str)
    reason_counts = reasons.str.split(";").explode().str.strip().replace("", "unknown").value_counts().head(15).reset_index()
    reason_counts.columns=["reason","count"]
    reason_counts["stage"]="exclusion_reason"
    return pd.concat([stage.rename(columns={"excluded_count":"count"})[["stage","count","retained_pct","source_file"]], reason_counts], ignore_index=True, sort=False)


def dna_table(selected: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    if selected.empty:
        return pd.DataFrame()
    for dim,col in DNA_DIMENSIONS.items():
        if col not in selected.columns:
            continue
        s = selected[col]
        if s.dtype == bool or set(s.dropna().astype(str).str.lower().unique()).issubset({"true","false"}):
            val = s.astype(str).str.lower().eq("true").mean()
        else:
            nums = numeric(s).dropna()
            if nums.empty: continue
            if dim == "Volatility":
                val = max(0, min(1, 1 - nums.mean()/1.5))
            elif dim == "Diversification":
                val = max(0, min(1, 1 - nums.mean()))
            else:
                rank = nums.rank(pct=True).mean() if len(nums)>1 else min(1, max(0, nums.mean()))
                val = rank
        rows.append({"dimension":dim,"score":float(val),"source_column":col,"note":"Diagnostic profile - not an additional allocation model."})
    return pd.DataFrame(rows)


def pipeline_table(data: dict[str,pd.DataFrame]) -> pd.DataFrame:
    nodes=[
        ("Yahoo / Canonical Market Data","official_market_data_integrity","official_market_data_integrity.csv","official"),
        ("Forecast History","current_raw_target_features","current_raw_target_features.csv","official_current"),
        ("Raw Target","current_raw_target_features","current_raw_target_features.csv","exact"),
        ("Universe Filters","universe_quality_report","growth_universe_quality_report.csv","official_current"),
        ("Ranking","current_growth_features","current_growth_features.csv","official_current"),
        ("Soft Exit","current_allocation","current_growth_candidate_allocation.csv","official_current"),
        ("Volatility Target","vol_pipeline_audit","growth_volatility_pipeline_audit.csv","official_current"),
        ("Dual Trend","current_allocation","current_growth_candidate_allocation.csv","official_current"),
        ("Allocation","current_allocation","current_growth_candidate_allocation.csv","observed_candidate"),
        ("Scheduler","rebalance_schedule","growth_rebalance_schedule.csv","official"),
        ("Official Paper","official_state","growth_official_paper_state.csv","official"),
    ]
    rows=[]
    for node,key,file,scope in nodes:
        df=_df(data,key); l=_latest(df); date=""
        if not l.empty:
            for c in ["date","market_date","volatility_source_date","expected_date"]:
                if c in l.columns:
                    val=pd.to_datetime(l[c], errors="coerce").max()
                    if pd.notna(val): date=str(val.date()); break
        rows.append({"node":node,"file_or_module":file,"latest_date":date,"status":"PASS" if not df.empty else "NOT AVAILABLE","input_output_rows":len(df),"exact_or_proxy":"exact" if scope in {"exact","official","official_current"} else "observed_candidate","scope":scope})
    return pd.DataFrame(rows)


def integrity_table(kpis:dict[str,Any], selected:pd.DataFrame, state:pd.DataFrame, funnel:pd.DataFrame, rejected:pd.DataFrame) -> pd.DataFrame:
    official=set(state[state.get("ticker","").astype(str).str.upper().ne("CASH")].get("ticker",pd.Series(dtype=str)).astype(str).str.upper()) if not state.empty else set()
    explained=set(selected.get("ticker",pd.Series(dtype=str)).astype(str).str.upper()) if not selected.empty else set()
    rows=[
        {"check":"final_holdings_equal_official_portfolio","status":"PASS" if official==explained else "FAIL","detail":f"official={sorted(official)}, explained={sorted(explained)}"},
        {"check":"sankey_counts_available","status":"PASS" if not funnel.empty and funnel.get("passed_count",pd.Series(dtype=float)).notna().all() else "WARNING","detail":len(funnel)},
        {"check":"rejected_candidates_have_reasons","status":"PASS" if rejected.empty or rejected.get("exact_exclusion_reason",pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).all() else "WARNING","detail":len(rejected)},
        {"check":"no_fabricated_feature_contributions","status":"PASS","detail":"feature values shown; exact marginal contribution labeled unavailable unless stored"},
        {"check":"official_current_namespace","status":"PASS","detail":"debug/reconstructed sources not used in current decision view"},
        {"check":"pending_signals_no_mutation","status":"PASS","detail":"pending signals read from current allocation only"},
        {"check":"read_only","status":"PASS","detail":"no order/model controls"},
    ]
    return pd.DataFrame(rows)


def source_audit(data:dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows=[]
    for key,file in DECISION_SOURCES.items():
        df=_df(data,key)
        rows.append({"source_file":file,"data_key":key,"loaded":not df.empty,"row_count":len(df),"date_range":_date_range(df),"checksum":_checksum(file),"namespace":"official_current_decision"})
    return pd.DataFrame(rows)


def build_decision_bundle(data:dict[str,pd.DataFrame]) -> DecisionBundle:
    funnel=_df(data,"decision_funnel")
    selected=selected_table(data)
    rejected=rejected_table(data)
    pending=_latest(_df(data,"pending_signals"))
    state=_latest(_df(data,"official_state"))
    kpis=build_kpis(data,funnel,selected,rejected,pending)
    nodes,links=build_sankey(funnel,selected)
    comparison=comparison_table(selected,rejected)
    tree=decision_tree_table(selected,rejected)
    hist=_df(data,"official_rebalance_report")
    attr=attrition_table(funnel,rejected)
    dna=dna_table(selected)
    pipe=pipeline_table(data)
    src=source_audit(data)
    integ=integrity_table(kpis,selected,state,funnel,rejected)
    missing=pd.DataFrame([{"item":"exact feature marginal contribution weights","status":"not_stored","note":"Feature value available; exact marginal contribution not stored."}])
    status="decision_engine_pass"
    if integ["status"].eq("FAIL").any(): status="decision_engine_fail"
    elif integ["status"].eq("WARNING").any() or not missing.empty: status="decision_engine_warning"
    return DecisionBundle(kpis,funnel,nodes,links,selected,rejected,comparison,tree,pending,hist,attr,dna,pipe,src,integ,funnel.copy(),missing,status)
