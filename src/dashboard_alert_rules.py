
from __future__ import annotations

from datetime import datetime
from hashlib import sha1
from typing import Any

import numpy as np
import pandas as pd

from dashboard_data_layer import latest, numeric

ALERT_COLUMNS = [
    "id", "timestamp", "module", "severity", "category", "description", "trigger_value", "threshold",
    "status", "resolved", "acknowledged", "source_file", "first_seen", "last_seen", "occurrences",
    "resolved_date", "acknowledged_date", "duration_days",
]

SEVERITY_ORDER = {"INFO": 1, "NOTICE": 2, "WARNING": 3, "CRITICAL": 4, "BLOCKER": 5}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _alert_id(module: str, category: str, description: str) -> str:
    key = f"{module}|{category}|{description}".lower().encode("utf-8")
    return "ALT-" + sha1(key).hexdigest()[:10].upper()


def _add(rows: list[dict[str, Any]], module: str, severity: str, category: str, description: str, trigger_value: Any, threshold: Any, source_file: str, status: str = "OPEN") -> None:
    rows.append({
        "id": _alert_id(module, category, description),
        "timestamp": _now(),
        "module": module,
        "severity": severity,
        "category": category,
        "description": description,
        "trigger_value": trigger_value,
        "threshold": threshold,
        "status": status,
        "resolved": False,
        "acknowledged": False,
        "source_file": source_file,
        "first_seen": _now(),
        "last_seen": _now(),
        "occurrences": 1,
        "resolved_date": "",
        "acknowledged_date": "",
        "duration_days": 0,
    })


def _latest_row(data: dict[str, pd.DataFrame], key: str) -> pd.Series:
    df = latest(data.get(key, pd.DataFrame()))
    return df.iloc[-1] if not df.empty else pd.Series(dtype=object)


def _latest_df(data: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    return latest(data.get(key, pd.DataFrame()))


def generate_alerts(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    perf = _latest_row(data, "official_performance")
    state = _latest_df(data, "official_state")
    non_cash = state[~state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("CASH")].copy() if not state.empty and "ticker" in state.columns else pd.DataFrame()
    monitor = _latest_row(data, "official_monitor")

    # Risk alerts
    vol = float(pd.to_numeric(pd.Series([perf.get("volatility", np.nan)]), errors="coerce").iloc[0]) if not perf.empty else np.nan
    if pd.notna(vol) and vol > 0.22:
        _add(rows, "Risk", "WARNING", "Risk", "Portfolio volatility exceeds target", vol, "0.22", "growth_official_paper_performance.csv")
    exposure = float(pd.to_numeric(pd.Series([perf.get("exposure", np.nan)]), errors="coerce").iloc[0]) if not perf.empty else np.nan
    if pd.notna(exposure) and exposure > 0.60:
        _add(rows, "Risk", "CRITICAL", "Risk", "Exposure exceeds cap", exposure, "0.60", "growth_official_paper_performance.csv")
    if pd.notna(exposure) and exposure < 0.40:
        _add(rows, "Risk", "NOTICE", "Risk", "Exposure below floor", exposure, "0.40", "growth_official_paper_performance.csv")
    weights = numeric(non_cash.get("paper_position_weight", pd.Series(dtype=float))) if not non_cash.empty else pd.Series(dtype=float)
    if not weights.empty and weights.max() > 0.25:
        _add(rows, "Risk", "WARNING", "Risk", "Largest position exceeds limit", float(weights.max()), "0.25", "growth_official_paper_state.csv")
    hhi = float((weights.dropna() ** 2).sum()) if not weights.empty else np.nan
    if pd.notna(hhi) and hhi > 0.20:
        _add(rows, "Risk", "WARNING", "Risk", "HHI concentration high", hhi, "0.20", "growth_official_paper_state.csv")
    dd = float(pd.to_numeric(pd.Series([perf.get("current_drawdown", perf.get("max_drawdown", np.nan))]), errors="coerce").iloc[0]) if not perf.empty else np.nan
    if pd.notna(dd) and dd < -0.10:
        _add(rows, "Risk", "WARNING", "Risk", "Drawdown exceeds threshold", dd, "-0.10", "growth_official_paper_performance.csv")
    cash = float(pd.to_numeric(pd.Series([perf.get("cash_weight", np.nan)]), errors="coerce").iloc[0]) if not perf.empty else np.nan
    if pd.notna(cash) and cash < 0.05:
        _add(rows, "Risk", "WARNING", "Risk", "Cash below configured minimum", cash, "0.05", "growth_official_paper_performance.csv")
    if pd.notna(cash) and cash > 0.55:
        _add(rows, "Risk", "NOTICE", "Risk", "Cash above configured maximum", cash, "0.55", "growth_official_paper_performance.csv")

    # Portfolio alerts
    actions = _latest_df(data, "official_actions")
    if not actions.empty and "action" in actions.columns:
        for action, sev, desc in [("BUY", "INFO", "New holding enters"), ("SELL", "NOTICE", "Holding exits"), ("INCREASE", "NOTICE", "Weight increases significantly"), ("REDUCE", "NOTICE", "Weight reduces significantly")]:
            sub = actions[actions["action"].astype(str).str.upper().eq(action)]
            if not sub.empty:
                tickers = ",".join(sub.get("ticker", pd.Series(dtype=str)).astype(str).tolist())
                _add(rows, "Portfolio", sev, "Portfolio", desc, tickers, action, "growth_official_paper_actions.csv")
        turnover = float(pd.to_numeric(pd.Series([perf.get("turnover", np.nan)]), errors="coerce").iloc[0]) if not perf.empty else np.nan
        if pd.notna(turnover) and turnover > 0.50:
            _add(rows, "Portfolio", "WARNING", "Portfolio", "Portfolio turnover unusually high", turnover, "0.50", "growth_official_paper_performance.csv")

    # Execution alerts
    costs = _latest_df(data, "official_cost_ledger")
    if costs.empty:
        _add(rows, "Execution", "WARNING", "Execution", "Missing execution cost ledger", "missing", "file required", "growth_official_estimated_cost_ledger.csv")
    else:
        total_cost = numeric(costs.get("estimated_total_cost", pd.Series(dtype=float))).sum()
        if total_cost > 500:
            _add(rows, "Execution", "WARNING", "Execution", "Estimated costs unusually high", float(total_cost), "$500", "growth_official_estimated_cost_ledger.csv")
    if pd.notna(exposure) and pd.notna(vol) and exposure > 0.55 and vol > 0.22:
        _add(rows, "Execution", "NOTICE", "Execution", "Execution/risk stack elevated", f"exposure={exposure}, vol={vol}", "exposure>0.55 and vol>0.22", "growth_official_paper_performance.csv")

    # Market data alerts
    md = data.get("official_market_data_integrity", pd.DataFrame())
    if md.empty:
        _add(rows, "Market Data", "CRITICAL", "Market Data", "Missing market data integrity file", "missing", "file required", "official_market_data_integrity.csv")
    else:
        if "is_fresh" in md.columns and md["is_fresh"].astype(str).str.lower().isin(["false", "0"]).any():
            _add(rows, "Market Data", "CRITICAL", "Market Data", "Stale market data", "one or more tickers stale", "all fresh", "official_market_data_integrity.csv")
        for col, desc in [("adjusted_close_available", "Adjusted prices unavailable"), ("volume_available", "Volume unavailable"), ("exists", "Missing OHLCV cache")]:
            if col in md.columns and md[col].astype(str).str.lower().isin(["false", "0"]).any():
                _add(rows, "Market Data", "WARNING", "Market Data", desc, "one or more false", "all true", "official_market_data_integrity.csv")
    mdgov = _latest_row(data, "official_market_data_governance")
    if not mdgov.empty and str(mdgov.get("classification", "")).upper().startswith("SINGLE_SOURCE"):
        _add(rows, "Market Data", "WARNING", "Market Data", "Missing second provider", mdgov.get("classification"), "multi_source_confirmed", "official_market_data_governance.csv")

    # Governance alerts
    if not monitor.empty:
        promo = str(monitor.get("promotion_status", ""))
        gov = str(monitor.get("governance_status", ""))
        if "blocked" in promo.lower():
            _add(rows, "Governance", "BLOCKER", "Governance", "Real capital blocked", promo, "promotion allowed", "growth_official_paper_monitor.csv")
        if "warmup" in gov.lower():
            _add(rows, "Governance", "NOTICE", "Governance", "Forward history too short", gov, "minimum paper history", "growth_official_paper_monitor.csv")
    anti = _latest_row(data, "anti_overfitting_governance")
    if not anti.empty:
        pbo = float(pd.to_numeric(pd.Series([anti.get("CSCV_PBO", np.nan)]), errors="coerce").iloc[0])
        if pd.notna(pbo) and pbo > 0.50:
            _add(rows, "Governance", "WARNING", "Governance", "PBO elevated", pbo, "0.50", "anti_overfitting_governance.csv")

    # Research alerts
    param = _latest_row(data, "parameter_governance")
    if not param.empty and "fragile" in str(param.get("classification", "")).lower():
        _add(rows, "Research", "WARNING", "Research", "Parameter instability", param.get("classification"), "stable_plateau", "parameter_governance.csv")
    oos = _latest_row(data, "out_of_sample_governance")
    if not oos.empty and any(x in str(oos.get("classification", "")).lower() for x in ["fail", "unstable"]):
        _add(rows, "Research", "WARNING", "Research", "Walk-forward deterioration", oos.get("classification"), "passes_oos", "out_of_sample_governance.csv")

    # Accounting alerts
    acct = _latest_row(data, "official_accounting_audit")
    recon = _latest_row(data, "official_accounting_reconciliation") if "official_accounting_reconciliation" in data else pd.Series(dtype=object)
    dup = _latest_row(data, "official_cost_duplication_audit")
    if not recon.empty:
        for col, desc in [("gross_identity_pass", "Gross ledger mismatch"), ("net_identity_pass", "Net/Gross mismatch"), ("weights_sum_to_one", "Position reconciliation failure")]:
            if col in recon.index and str(recon.get(col)).lower() not in {"true", "1"}:
                _add(rows, "Accounting", "CRITICAL", "Accounting", desc, recon.get(col), "True", "official_accounting_reconciliation.csv")
    if not dup.empty and float(pd.to_numeric(pd.Series([dup.get("duplicate_cost_rows", 0)]), errors="coerce").iloc[0]) > 0:
        _add(rows, "Accounting", "CRITICAL", "Accounting", "Duplicate costs", dup.get("duplicate_cost_rows"), "0", "official_cost_duplication_audit.csv")

    # Dashboard module alerts
    for key, module, source in [("decision_engine_integrity", "Decision Engine", "decision_engine_integrity.csv"), ("historical_replay_integrity", "Historical Replay", "historical_replay_integrity.csv"), ("quant_lab_surface_integrity", "Dashboard", "quant_lab_surface_integrity.csv")]:
        df = data.get(key, pd.DataFrame())
        if df.empty:
            _add(rows, module, "WARNING", module, "Missing dashboard integrity file", "missing", "file required", source)
        elif "status" in df.columns and df["status"].astype(str).str.upper().eq("FAIL").any():
            _add(rows, module, "CRITICAL", module, "Dashboard integrity failure", "FAIL", "PASS", source)

    out = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    return out.drop_duplicates("id") if not out.empty else pd.DataFrame(columns=ALERT_COLUMNS)


def health_score(active_alerts: pd.DataFrame) -> tuple[float, str]:
    if active_alerts.empty:
        return 100.0, "Perfect"
    penalty = 0
    weights = {"INFO": 1, "NOTICE": 2, "WARNING": 5, "CRITICAL": 15, "BLOCKER": 25}
    severities = active_alerts["severity"].astype(str).str.upper()
    for sev in severities:
        penalty += weights.get(sev, 5)
    score = max(0.0, 100.0 - penalty)
    has_critical = severities.eq("CRITICAL").any()
    has_blocker = severities.eq("BLOCKER").any()
    if has_critical:
        label = "Critical"
    elif has_blocker:
        label = "Governance Blocked"
    elif score >= 95:
        label = "Excellent"
    elif score >= 85:
        label = "Healthy"
    elif score >= 70:
        label = "Needs Attention"
    else:
        label = "Watchlist"
    return score, label
