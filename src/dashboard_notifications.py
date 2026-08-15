from __future__ import annotations

import pandas as pd

from dashboard_alert_engine import build_alert_engine
from dashboard_components import status_badge


def render_live_status_bar(st, data: dict[str, pd.DataFrame]) -> None:
    result = build_alert_engine(data, write_outputs=False)
    active = result["active"]
    score = float(result["health_score"])
    blockers = int(active["severity"].astype(str).str.upper().isin(["BLOCKER"]).sum()) if not active.empty else 0
    critical = int(active["severity"].astype(str).str.upper().isin(["CRITICAL"]).sum()) if not active.empty else 0
    warnings = int(active["severity"].astype(str).str.upper().isin(["WARNING"]).sum()) if not active.empty else 0
    notices = int(active["severity"].astype(str).str.upper().isin(["NOTICE"]).sum()) if not active.empty else 0
    state = "pass" if score >= 95 else "warning" if score >= 75 else "failed"
    st.markdown(
        f"""
        <div class='status-strip'>
          <span><b>System Health:</b> {score:.1f}% {status_badge(result["health_label"], state)}</span>
          <span>{status_badge(f"BLOCKER {blockers}", "blocked" if blockers else "pass")}</span>
          <span>{status_badge(f"CRITICAL {critical}", "failed" if critical else "pass")}</span>
          <span>{status_badge(f"WARNING {warnings}", "warning" if warnings else "pass")}</span>
          <span>{status_badge(f"NOTICE {notices}", "diagnostic" if notices else "pass")}</span>
          <span class='small-muted'>Read-only alert monitor · no orders</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
