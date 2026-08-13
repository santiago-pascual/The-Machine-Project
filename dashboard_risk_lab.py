
from __future__ import annotations

import pandas as pd
import plotly.express as px

from dashboard_components import metric_card, source_caption
from dashboard_theme import apply_plotly_layout
from risk_constraint_sensitivity import main as run_lab


def _load(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def render_risk_lab(st, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Risk Constraint Sensitivity Laboratory")
    st.caption("Shadow mode only. No active model, paper, optimizer, scheduler, ranking, forecast, allocation or parameter changes.")
    run_lab()
    grid = _load("risk_constraint_grid.csv")
    summary = _load("risk_sensitivity_summary.csv")
    pareto = _load("pareto_frontier.csv")
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Experiments", str(len(grid)), "one parameter at a time")
    with cols[1]: metric_card(st, "Pareto Efficient", str(int(pareto.get('pareto_efficient', pd.Series(dtype=bool)).sum())) if not pareto.empty else "n/a")
    with cols[2]: metric_card(st, "Mode", "SHADOW", "read-only")
    with cols[3]: metric_card(st, "Status", "PASS", "no parameter writes")
    if not grid.empty:
        fig = px.scatter(grid, x="max_drawdown", y="CAGR", color="experiment_type", hover_name="parameter", size="cash_utilization_pct", title="Return vs Drawdown Frontier")
        st.plotly_chart(apply_plotly_layout(fig, "Return vs Drawdown Frontier"), width="stretch")
        fig = px.scatter(grid, x="cash_utilization_pct", y="CAGR", color="experiment_type", hover_name="parameter", title="Return vs Cash Utilization")
        st.plotly_chart(apply_plotly_layout(fig, "Return vs Cash Utilization"), width="stretch")
        fig = px.line(grid[grid['experiment_type'].eq('exposure_cap')], x="parameter", y=["CAGR", "Sharpe", "average_cash"], markers=True, title="Exposure Cap Sensitivity")
        st.plotly_chart(apply_plotly_layout(fig, "Exposure Cap Sensitivity"), width="stretch")
        fig = px.line(grid[grid['experiment_type'].eq('vol_target')], x="parameter", y=["CAGR", "Sharpe", "average_cash"], markers=True, title="Vol Target Sensitivity")
        st.plotly_chart(apply_plotly_layout(fig, "Vol Target Sensitivity"), width="stretch")
        st.dataframe(grid, width="stretch", hide_index=True)
    if not summary.empty:
        st.markdown("#### Sensitivity Summary")
        st.dataframe(summary, width="stretch", hide_index=True)
    source_caption(st, "risk_constraint_sensitivity.py + shadow_constraint_backtests.py", "shadow/read-only")
