from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import plotly.express as px

from dashboard_alert_history import update_alert_history
from dashboard_alert_rules import SEVERITY_ORDER, generate_alerts, health_score
from dashboard_components import fmt_num, metric_card, status_badge
from dashboard_data_layer import CSV_FILES, latest, latest_market_date
from dashboard_theme import apply_plotly_layout


OUTPUTS = {
    "alert_engine_source_audit": "alert_engine_source_audit.csv",
    "alert_engine_integrity": "alert_engine_integrity.csv",
    "alert_history": "alert_history.csv",
    "active_alerts": "active_alerts.csv",
    "phase117_report": "phase117_alert_engine_report.txt",
}


def _source_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, path in CSV_FILES.items():
        df = data.get(key, pd.DataFrame())
        latest_date = ""
        if not df.empty and "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            latest_date = dates.max().date().isoformat() if not dates.empty else ""
        rows.append({
            "source_key": key,
            "source_file": path,
            "exists": Path(path).exists(),
            "row_count": len(df),
            "column_count": len(df.columns) if not df.empty else 0,
            "latest_date": latest_date,
        })
    return pd.DataFrame(rows)


def _integrity(active: pd.DataFrame, source_audit: pd.DataFrame, runtime_ms: float) -> pd.DataFrame:
    duplicate_ids = int(active["id"].duplicated().sum()) if not active.empty and "id" in active.columns else 0
    blockers = int(active["severity"].astype(str).str.upper().isin(["BLOCKER"]).sum()) if not active.empty else 0
    critical = int(active["severity"].astype(str).str.upper().isin(["CRITICAL"]).sum()) if not active.empty else 0
    warnings = int(active["severity"].astype(str).str.upper().isin(["WARNING"]).sum()) if not active.empty else 0
    status = "alert_engine_fail" if duplicate_ids else "alert_engine_warning" if blockers or critical or warnings else "alert_engine_pass"
    return pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "active_alerts": len(active),
        "blocker_alerts": blockers,
        "critical_alerts": critical,
        "warning_alerts": warnings,
        "duplicate_alert_ids": duplicate_ids,
        "sources_checked": len(source_audit),
        "missing_sources": int((source_audit["exists"] == False).sum()) if not source_audit.empty else 0,
        "runtime_ms": round(runtime_ms, 2),
    }])


def build_alert_engine(data: dict[str, pd.DataFrame], write_outputs: bool = True) -> dict[str, object]:
    started = perf_counter()
    current = generate_alerts(data)
    if write_outputs:
        history, active = update_alert_history(current)
    else:
        history_path = Path("alert_history.csv")
        history = pd.read_csv(history_path) if history_path.exists() else current.copy()
        active = current.copy()
    score, score_label = health_score(active)
    runtime_ms = (perf_counter() - started) * 1000
    source_audit = _source_audit(data)
    integrity = _integrity(active, source_audit, runtime_ms)

    if write_outputs:
        source_audit.to_csv(OUTPUTS["alert_engine_source_audit"], index=False)
        integrity.to_csv(OUTPUTS["alert_engine_integrity"], index=False)
        report = [
            "===== INSTITUTIONAL ALERT & MONITORING ENGINE =====",
            f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"latest_market_date: {latest_market_date(data)}",
            f"health_score: {score:.2f}",
            f"health_label: {score_label}",
            f"active_alerts: {len(active)}",
            f"resolved_alerts: {int(history['resolved'].astype(str).str.lower().isin(['true', '1']).sum()) if not history.empty else 0}",
            f"critical_or_blocker_alerts: {int(active['severity'].astype(str).str.upper().isin(['CRITICAL', 'BLOCKER']).sum()) if not active.empty else 0}",
            f"runtime_ms: {runtime_ms:.2f}",
            f"final_status: {integrity.iloc[-1]['status']}",
            "",
            "Rules are deterministic and read-only. No model, paper, optimizer, scheduler, accounting or order logic changed.",
        ]
        Path(OUTPUTS["phase117_report"]).write_text("\n".join(report), encoding="utf-8")

    return {
        "current": current,
        "history": history,
        "active": active,
        "health_score": score,
        "health_label": score_label,
        "source_audit": source_audit,
        "integrity": integrity,
        "runtime_ms": runtime_ms,
    }


def _severity_counts(active: pd.DataFrame) -> pd.DataFrame:
    if active.empty:
        return pd.DataFrame(columns=["severity", "count", "order"])
    out = active.groupby("severity").size().reset_index(name="count")
    out["order"] = out["severity"].astype(str).str.upper().map(SEVERITY_ORDER).fillna(0)
    return out.sort_values("order")


def render_alert_center(st, data: dict[str, pd.DataFrame]) -> None:
    result = build_alert_engine(data, write_outputs=True)
    active = result["active"]
    history = result["history"]
    score = result["health_score"]
    score_label = result["health_label"]

    st.subheader("Institutional Alert Center")
    cols = st.columns(5)
    with cols[0]:
        metric_card(st, "Health Score", f"{score:.1f}%", score_label)
    with cols[1]:
        metric_card(st, "Active Alerts", str(len(active)), "open alerts")
    with cols[2]:
        metric_card(st, "Critical / Blocker", str(int(active["severity"].astype(str).str.upper().isin(["CRITICAL", "BLOCKER"]).sum())) if not active.empty else "0", "blocking severity")
    with cols[3]:
        metric_card(st, "Warnings", str(int((active["severity"].astype(str).str.upper() == "WARNING").sum())) if not active.empty else "0", "watch items")
    with cols[4]:
        metric_card(st, "Runtime", f"{result['runtime_ms']:.0f} ms", "alert engine")

    st.markdown("#### Active Alerts")
    if active.empty:
        st.success("No active alerts.")
    else:
        modules = sorted(active["module"].dropna().astype(str).unique().tolist())
        severities = sorted(active["severity"].dropna().astype(str).unique().tolist(), key=lambda x: SEVERITY_ORDER.get(x, 0), reverse=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_modules = st.multiselect("Module", modules, default=modules)
        with c2:
            selected_sev = st.multiselect("Severity", severities, default=severities)
        with c3:
            show_ack = st.checkbox("Show acknowledged", value=True)
        view = active[active["module"].astype(str).isin(selected_modules) & active["severity"].astype(str).isin(selected_sev)].copy()
        if not show_ack and "acknowledged" in view.columns:
            view = view[~view["acknowledged"].astype(str).str.lower().isin(["true", "1"])]
        cols = ["severity", "module", "category", "description", "trigger_value", "threshold", "status", "first_seen", "last_seen", "occurrences", "source_file"]
        st.dataframe(view[[c for c in cols if c in view.columns]], width="stretch", height=360)

    c1, c2 = st.columns(2)
    with c1:
        counts = _severity_counts(active)
        if counts.empty:
            st.info("No active severity distribution.")
        else:
            fig = px.bar(counts, x="severity", y="count", color="severity", title="Active Alerts by Severity")
            fig = apply_plotly_layout(fig, "Active Alerts by Severity")
            st.plotly_chart(fig, width="stretch")
    with c2:
        if active.empty:
            st.info("No active module distribution.")
        else:
            mod = active.groupby("module").size().reset_index(name="count").sort_values("count", ascending=False)
            fig = px.bar(mod, x="module", y="count", color="module", title="Active Alerts by Module")
            fig = apply_plotly_layout(fig, "Active Alerts by Module")
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Alert History")
    if history.empty:
        st.info("Alert history unavailable.")
    else:
        hist = history.sort_values("last_seen", ascending=False).copy()
        st.dataframe(hist, width="stretch", height=420)

    with st.expander("Alert Engine Source Audit"):
        st.dataframe(result["source_audit"], width="stretch")
    with st.expander("Alert Engine Integrity"):
        st.dataframe(result["integrity"], width="stretch")
