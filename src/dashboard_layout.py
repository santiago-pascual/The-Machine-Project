from __future__ import annotations

import pandas as pd

from dashboard_components import status_badge
from dashboard_data_layer import (
    MODEL_VERSION,
    VARIANT,
    get_scope_namespace,
    latest,
    latest_market_date,
    next_rebalance_date,
    official_start_date,
)

NAV_ITEMS = [
    "Mission Control",
    "Alert Center",
    "Executive",
    "Portfolio",
    "Decision Engine",
    "Performance",
    "Risk",
    "Execution",
    "Research",
    "Alpha Attribution",
    "Governance",
    "Reports",
    "Historical Replay",
    "Quant Lab 3D",
    "Diagnostics",
]

PAGE_SUBTITLES = {
    "Mission Control": "Official system health, pipeline status, alerts, incidents and next actions.",
    "Alert Center": "Deterministic active alerts, alert history, severity filters and system health score.",
    "Executive": "Official forward track, benchmark context, exposure and rebalance status.",
    "Portfolio": "Current holdings, position PnL, quality notes and official allocation state.",
    "Decision Engine": "Universe funnel, selected holding explainability, rejected candidates and pending rebalance signals.",
    "Performance": "Gross/net equity, benchmark comparison, rolling returns and drawdowns.",
    "Risk": "Volatility targeting, concentration, correlation and diagnostic regime context.",
    "Execution": "Rebalance ledger, estimated costs, capacity and execution diagnostics.",
    "Research": "Backtest, reconstructed stress and research-only validation outputs.",
    "Alpha Attribution": "Performance decomposition, forecast/ranking diagnostics and alpha leakage analysis.",
    "Governance": "Live validation, integrity gates, market data confidence and promotion blocks.",
    "Reports": "Institutional PDF, HTML and Markdown reports generated from official namespace files only.",
    "Historical Replay": "Official historical state reconstruction and time-machine review.",
    "Quant Lab 3D": "Interactive diagnostic surfaces. Research-only unless explicitly marked official.",
    "Diagnostics": "Data source audit, namespace integrity and missing-file diagnostics.",
}


def sidebar(st, data: dict[str, pd.DataFrame]) -> tuple[str, str]:
    st.sidebar.markdown("<div style='padding:8px 4px 14px 4px'><div style='font-size:1.22rem;font-weight:850;color:#FF9A2E;letter-spacing:-.03em'>La Máquina</div><div style='color:#8B98A5;font-size:.78rem'>Institutional Quant Terminal</div></div>", unsafe_allow_html=True)
    scope = st.sidebar.selectbox("Data scope", ["Official Forward Paper", "Historical 2008 Backtest"], index=0)
    nav = st.sidebar.radio("Navigation", NAV_ITEMS, index=0)
    monitor = latest(data.get("official_monitor", pd.DataFrame()))
    status = monitor.iloc[-1].get("governance_status", "WARMUP") if not monitor.empty else "WARMUP"
    integrity = monitor.iloc[-1].get("integrity_status", "unavailable") if not monitor.empty else "unavailable"
    next_rebalance = next_rebalance_date(data)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model**  \\n`{MODEL_VERSION}`")
    st.sidebar.markdown(f"**Latest market date**  \\n`{latest_market_date(data)}`")
    st.sidebar.markdown(f"**Official start**  \\n`{official_start_date(data)}`")
    st.sidebar.markdown(f"**Paper status**  \\n{status_badge(str(status), str(status))}", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Integrity**  \\n{status_badge(str(integrity), str(integrity))}", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Next rebalance**  \\n`{next_rebalance}`")
    st.sidebar.markdown("---")
    st.sidebar.caption("Read-only. No broker. No real orders.")
    return scope, nav


def hero(st, data: dict[str, pd.DataFrame], scope: str) -> None:
    monitor = latest(data.get("official_monitor", pd.DataFrame()))
    integrity = monitor.iloc[-1].get("integrity_status", "unavailable") if not monitor.empty else "unavailable"
    next_rebalance = next_rebalance_date(data)
    st.markdown(
        f"""
        <div class='hero'>
          <h1>Growth Champion Final</h1>
          <p>{MODEL_VERSION} · {VARIANT} · official paper terminal · real capital blocked</p>
          <p>{status_badge(scope, 'official' if scope == 'Official Forward Paper' else 'diagnostic')} {status_badge(str(integrity), str(integrity))} <span class='small-muted'>Namespace: {get_scope_namespace(scope)} · Latest: {latest_market_date(data)} · Next rebalance: {next_rebalance}</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(st, page: str, scope: str, data: dict[str, pd.DataFrame]) -> None:
    subtitle = PAGE_SUBTITLES.get(page, "")
    badge_status = "official" if scope == "Official Forward Paper" else "diagnostic"
    st.markdown(
        f"""
        <div class='page-head'>
          <div>
            <h2 class='page-title'>{page} {status_badge(scope, badge_status)}</h2>
            <div class='page-subtitle'>{subtitle}<br>{scope} | {latest_market_date(data)} | {MODEL_VERSION}</div>
          </div>
          <div>{status_badge('READ ONLY', 'diagnostic')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
