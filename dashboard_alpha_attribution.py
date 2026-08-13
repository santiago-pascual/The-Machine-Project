
from __future__ import annotations

import pandas as pd
import plotly.express as px

from alpha_attribution_engine import run_alpha_attribution
from dashboard_components import metric_card, source_caption
from dashboard_theme import apply_plotly_layout


def _load(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def render_alpha_attribution(st, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Alpha Attribution & Performance Decomposition")
    st.caption("Strictly analytical. Additive attribution is reported in log-return space; unavailable marginal effects remain in Residual.")
    result = run_alpha_attribution()
    attr = result["attribution"]
    recon = result["reconciliation"]
    lost = result["lost"]
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Status", result["status"], "read-only")
    with cols[1]: metric_card(st, "Total Log Return", f"{float(recon.iloc[-1]['total_return']):.4f}" if not recon.empty else "n/a")
    with cols[2]: metric_card(st, "Recon Diff", f"{float(recon.iloc[-1]['difference']):.2e}" if not recon.empty else "n/a")
    with cols[3]: metric_card(st, "Recon", str(recon.iloc[-1].get("reconciliation_pass", "n/a")) if not recon.empty else "n/a")
    if not attr.empty:
        fig = px.bar(attr, x="component", y="alpha_contribution", color="alpha_contribution", color_continuous_scale="Oranges", title="Alpha Attribution Waterfall Inputs")
        fig = apply_plotly_layout(fig, "Alpha Attribution Components")
        st.plotly_chart(fig, width="stretch")
        pie = px.pie(attr[attr["absolute_contribution"] > 0], names="component", values="absolute_contribution", title="Absolute Attribution Share")
        pie = apply_plotly_layout(pie, "Absolute Attribution Share")
        st.plotly_chart(pie, width="stretch")
        st.dataframe(attr, width="stretch", hide_index=True)
    st.markdown("#### Largest Lost Alpha Sources")
    if lost.empty:
        st.info("No negative measured components found.")
    else:
        st.dataframe(lost[["component", "lost_alpha", "method", "confidence"]], width="stretch", hide_index=True)
    tabs = st.tabs(["Forecast", "Ranking", "Sizing", "Cash", "Turnover", "Costs", "Holdings", "Factors", "Integrity"])
    files = ["forecast_analysis.csv", "ranking_analysis.csv", "position_sizing_analysis.csv", "cash_drag_analysis.csv", "turnover_analysis.csv", "cost_attribution.csv", "holding_period_analysis.csv", "factor_contribution.csv", "alpha_attribution_integrity.csv"]
    for tab, file in zip(tabs, files):
        with tab:
            df = _load(file)
            if df.empty:
                st.warning(f"{file} unavailable")
            else:
                st.dataframe(df, width="stretch", hide_index=True)
    source_caption(st, "alpha_attribution_engine.py + official/reconstructed research CSVs", "read-only")
