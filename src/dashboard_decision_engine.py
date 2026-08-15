
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_components import (
    alert_box,
    fmt_num,
    fmt_pct,
    metric_card,
    section_header,
    source_caption,
)
from dashboard_data_layer import MODEL_VERSION, VARIANT, numeric
from dashboard_decision_calculations import FEATURE_COLUMNS, build_decision_bundle
from dashboard_theme import (
    AMBER,
    BRIGHT_ORANGE,
    CHART_COLORS,
    MUTED_ORANGE,
    ORANGE,
    apply_plotly_layout,
)

DECISION_SEQUENCE = [ORANGE, BRIGHT_ORANGE, AMBER, MUTED_ORANGE, "#FFB25C", "#D96B00"]


def _safe_df(df: pd.DataFrame, cols: Iterable[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    view = df.copy()
    if cols is not None:
        available = [c for c in cols if c in view.columns]
        view = view[available]
    for col in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[col]):
            view[col] = view[col].dt.strftime("%Y-%m-%d")
        elif view[col].dtype == object:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    return view


def _safe_chart(st, fig, height: int = 430) -> None:
    title_text = getattr(getattr(fig.layout, "title", None), "text", None)
    apply_plotly_layout(fig, title=str(title_text) if title_text else None)
    fig.update_layout(height=height, title_text=str(title_text) if title_text else "")
    st.plotly_chart(fig, width="stretch")


def _metric_options(df: pd.DataFrame) -> list[str]:
    requested = [
        "raw_target_return_exact",
        "realized_vol_60d",
        "median_60d_dollar_volume",
        "market_cap",
        "quality_score",
        "health_score",
        "pct_total_portfolio_risk",
        "rank_percentile",
    ]
    return [c for c in requested if c in df.columns and numeric(df[c]).notna().any()]


def _format_reason_list(series: pd.Series, n: int = 4) -> str:
    if series.empty:
        return "unavailable"
    values = series.fillna("unknown").astype(str).str.split(";").explode().str.strip()
    values = values[values.ne("") & values.str.lower().ne("nan")]
    if values.empty:
        return "unavailable"
    return ", ".join(values.value_counts().head(n).index.tolist())


def _header(st, bundle) -> None:
    k = bundle.kpis
    cols = st.columns(4)
    with cols[0]:
        metric_card(st, "Active Model Version", MODEL_VERSION, VARIANT, badge="READ ONLY")
    with cols[1]:
        metric_card(st, "Latest Market Date", str(k.get("latest_market_date", "unavailable")), "official/current decision scope")
    with cols[2]:
        metric_card(st, "Signal / Application", f"{k.get('signal_date', 'n/a')} -> {k.get('economic_application_date', 'n/a')}", "t+1 economic application")
    with cols[3]:
        metric_card(st, "Current Holdings", str(k.get("current_holdings", "unavailable")), "official forward paper")
    cols = st.columns(4)
    with cols[0]:
        metric_card(st, "Last Rebalance", str(k.get("last_rebalance", "unavailable")))
    with cols[1]:
        metric_card(st, "Next Rebalance", str(k.get("next_rebalance", "unavailable")))
    with cols[2]:
        metric_card(st, "Universe Count", fmt_num(k.get("universe_count", np.nan), 0), "full current feature universe")
    with cols[3]:
        metric_card(st, "Decision Status", bundle.status.replace("decision_engine_", ""), "dashboard/export validation", badge=bundle.status)


def _sankey_and_funnel(st, bundle) -> None:
    section_header(st, "Decision Pipeline", "Full universe to official holdings. Counts come from current pipeline reports.")
    left, right = st.columns([1.35, 1.0])
    with left:
        nodes, links = bundle.sankey_nodes, bundle.sankey_links
        if nodes.empty or links.empty:
            alert_box(st, "Sankey unavailable: missing funnel or selected holding data.", "warning")
        else:
            fig = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    label=nodes["label"].astype(str).tolist(),
                    pad=18,
                    thickness=18,
                    color=[DECISION_SEQUENCE[i % len(DECISION_SEQUENCE)] for i in range(len(nodes))],
                    customdata=nodes[["stage", "count", "retained_pct", "source_file"]].astype(str).values,
                    hovertemplate="%{customdata[0]}<br>count=%{customdata[1]}<br>retained=%{customdata[2]}<br>source=%{customdata[3]}<extra></extra>",
                ),
                link=dict(
                    source=links["source"].astype(int).tolist(),
                    target=links["target"].astype(int).tolist(),
                    value=links["value"].astype(float).clip(lower=0.01).tolist(),
                    color="rgba(255,138,42,0.30)",
                    customdata=links[["stage", "excluded_count", "retained_pct", "reason_summary"]].astype(str).values,
                    hovertemplate="%{customdata[0]}<br>flow=%{value}<br>excluded=%{customdata[1]}<br>retained=%{customdata[2]}<br>reasons=%{customdata[3]}<extra></extra>",
                ),
            )])
            fig.update_layout(title_text="Decision Pipeline Sankey")
            _safe_chart(st, fig, 560)
    with right:
        f = bundle.funnel.copy()
        if f.empty or "stage" not in f.columns:
            alert_box(st, "Funnel unavailable: missing growth_decision_funnel.csv.", "warning")
        else:
            fig = px.funnel(f, y="stage", x="passed_count", color="stage", color_discrete_sequence=DECISION_SEQUENCE, title="Current Decision Funnel")
            _safe_chart(st, fig, 560)
            biggest = f.sort_values("excluded_count", ascending=False).head(1)
            reason = _format_reason_list(bundle.rejected.get("exact_exclusion_reason", pd.Series(dtype=str))) if not bundle.rejected.empty else "unavailable"
            if not biggest.empty:
                row = biggest.iloc[0]
                metric_card(st, "Largest Exclusion Stage", str(row.get("stage", "unavailable")), f"excluded {fmt_num(row.get('excluded_count', np.nan), 0)} · reasons: {reason}")


def _selected_explainability(st, bundle) -> None:
    section_header(st, "Selected Holding Explainability", "Official holdings only. No generated advice, deterministic rule summaries.")
    selected = bundle.selected.copy()
    if selected.empty:
        alert_box(st, "No official selected holdings available.", "warning")
        return
    tickers = selected["ticker"].astype(str).tolist()
    ticker = st.selectbox("Selected official holding", tickers, key="decision_selected_ticker")
    row = selected[selected["ticker"].astype(str).eq(ticker)].iloc[-1]
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Company", str(row.get("company_name", ticker)), str(row.get("exchange", "exchange unavailable")))
    with cols[1]: metric_card(st, "Sector / Industry", str(row.get("sector", "unavailable")), str(row.get("industry", "industry unavailable")))
    with cols[2]: metric_card(st, "Raw Target Rank", fmt_num(row.get("raw_target_rank", np.nan), 0), f"percentile {fmt_pct(row.get('rank_percentile', np.nan))}")
    with cols[3]: metric_card(st, "Official Weight", fmt_pct(row.get("current_official_weight", row.get("paper_position_weight", np.nan))), f"target {fmt_pct(row.get('final_target_weight', np.nan))}")
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Raw Target Return", fmt_pct(row.get("raw_target_return_exact", np.nan)), "exact source when available")
    with cols[1]: metric_card(st, "Distance From Cutoff", fmt_num(row.get("distance_from_portfolio_cutoff", np.nan), 0), "rank distance")
    with cols[2]: metric_card(st, "Volatility", fmt_pct(row.get("realized_vol_60d", np.nan)), "60D annualized")
    with cols[3]: metric_card(st, "Risk Contribution", fmt_pct(row.get("pct_total_portfolio_risk", np.nan)), f"beta {fmt_num(row.get('beta_to_portfolio', row.get('beta_to_spy', np.nan)), 2)}")
    alert_box(st, str(row.get("reason_summary", "Selection reason unavailable.")), "info")

    filter_cols = ["ticker", "quality_pass", "passed_tradability_filter", "holding_quality_classification", "soft_exit_status", "exact_raw_target_available", "median_60d_dollar_volume", "market_cap", "estimated_total_cost"]
    st.dataframe(_safe_df(pd.DataFrame([row]), filter_cols), width="stretch", hide_index=True)

    feature_rows = []
    for col in FEATURE_COLUMNS:
        if col not in selected.columns:
            continue
        value = row.get(col, np.nan)
        if pd.isna(value):
            continue
        vals = numeric(selected[col]).dropna()
        percentile = vals.rank(pct=True).iloc[-1] if len(vals) and pd.notna(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]) else np.nan
        feature_rows.append({
            "feature": col,
            "current_value": value,
            "percentile_within_selected": percentile,
            "interpretation": "diagnostic value available",
            "status": "active" if col in {"raw_target_return_exact", "raw_expected_daily_return_exact"} else "diagnostic",
            "source": "current_growth_features.csv/current_raw_target_features.csv",
            "contribution_note": "Feature value available; exact marginal contribution not stored.",
        })
    st.markdown("#### Feature Contribution Panel")
    if feature_rows:
        st.dataframe(_safe_df(pd.DataFrame(feature_rows)), width="stretch", hide_index=True)
    else:
        alert_box(st, "Feature panel unavailable: no stored feature values for selected ticker.", "warning")


def _rejected_and_compare(st, bundle) -> None:
    section_header(st, "Nearest Rejected Candidates", "Top non-selected candidates with exact exclusion stage/reason when stored.")
    rejected = bundle.rejected.head(10).copy()
    if rejected.empty:
        alert_box(st, "Rejected candidate table unavailable.", "warning")
    else:
        cols = ["ticker", "raw_target_rank", "raw_target_return_exact", "quality_pass", "passed_tradability_filter", "soft_exit_status", "holding_quality_classification", "exclusion_stage", "exact_exclusion_reason", "distance_from_portfolio_cutoff"]
        st.dataframe(_safe_df(rejected, cols), width="stretch", hide_index=True)

    comp = bundle.comparison.copy()
    opts = _metric_options(comp)
    if not comp.empty and opts:
        metric = st.selectbox("Selected vs rejected metric", opts, key="decision_compare_metric")
        group_col = "group" if "group" in comp.columns else "selection_status"
        c = comp[["ticker", group_col, metric]].copy().rename(columns={group_col: "group"})
        c[metric] = numeric(c[metric])
        fig = px.bar(c.dropna(subset=[metric]), x="ticker", y=metric, color="group", barmode="group", color_discrete_sequence=DECISION_SEQUENCE, title=f"Selected vs Top Rejected — {metric}")
        _safe_chart(st, fig, 430)
    else:
        alert_box(st, "Comparison chart unavailable: no comparable numeric metric columns found.", "warning")


def _rule_path_and_pending(st, bundle) -> None:
    section_header(st, "Rule Path and Pending Signals", "Observed candidates are signal-only until scheduled rebalance.")
    left, right = st.columns(2)
    with left:
        tree = bundle.decision_tree.copy()
        if tree.empty:
            alert_box(st, "Decision tree unavailable.", "warning")
        else:
            ticker = st.selectbox("Rule path ticker", sorted(tree["ticker"].dropna().astype(str).unique()), key="decision_tree_ticker")
            view = tree[tree["ticker"].astype(str).eq(ticker)].copy()
            fig = px.bar(view, y="rule", x=[1] * len(view), color="status", orientation="h", color_discrete_map={"PASS": CHART_COLORS.get("positive", "#18C48F"), "FAIL": CHART_COLORS.get("negative", "#FF4D4D")}, title=f"Rule Path — {ticker}")
            fig.update_xaxes(visible=False)
            _safe_chart(st, fig, 470)
            st.dataframe(_safe_df(view), width="stretch", hide_index=True)
    with right:
        pending = bundle.pending.copy()
        if pending.empty:
            alert_box(st, "Pending signal table unavailable.", "warning")
        else:
            official = set(bundle.selected.get("ticker", pd.Series(dtype=str)).astype(str))
            candidates = set(pending[pending.get("ticker", "").astype(str).str.upper().ne("CASH")]["ticker"].astype(str))
            enter = sorted(candidates - official)
            exit_ = sorted(official - candidates)
            metric_card(st, "Sessions Until Next Rebalance", fmt_num(bundle.kpis.get("sessions_until_next_rebalance", np.nan), 0), "Signal only — not executed until scheduled rebalance.")
            metric_card(st, "Would Enter If Rebalance Today", ", ".join(enter) if enter else "none", "pending only")
            metric_card(st, "Would Exit If Rebalance Today", ", ".join(exit_) if exit_ else "none", "pending only")
            st.dataframe(_safe_df(pending), width="stretch", hide_index=True)


def _history_attrition_dna(st, bundle) -> None:
    section_header(st, "Decision History and Filter Attrition", "Official rebalance history plus current exclusion analysis.")
    hcol, acol = st.columns(2)
    with hcol:
        hist = bundle.history.copy()
        if hist.empty:
            alert_box(st, "Decision history unavailable: official rebalance report missing.", "warning")
        else:
            cols = ["date", "selected_holdings", "new_entrants", "exits", "retained_holdings", "final_exposure", "cash_weight", "turnover", "estimated_total_cost", "governance_status"]
            st.dataframe(_safe_df(hist, cols), width="stretch", hide_index=True)
            if "turnover" in hist.columns and "date" in hist.columns:
                fig = px.line(hist, x="date", y="turnover", markers=True, title="Turnover by Rebalance Date")
                _safe_chart(st, fig, 330)
    with acol:
        attr = bundle.attrition.copy()
        if attr.empty:
            alert_box(st, "Attrition analysis unavailable.", "warning")
        else:
            stage_rows = attr[attr["stage"].ne("exclusion_reason")].copy()
            if not stage_rows.empty:
                fig = px.bar(stage_rows, x="stage", y="count", color="stage", color_discrete_sequence=DECISION_SEQUENCE, title="Exclusions by Stage")
                fig.update_layout(xaxis_tickangle=-30)
                _safe_chart(st, fig, 330)
            reason_rows = attr[attr["stage"].eq("exclusion_reason")].copy()
            if not reason_rows.empty:
                fig = px.bar(reason_rows.head(12), x="count", y="reason", orientation="h", color="count", color_continuous_scale="Oranges", title="Top Exclusion Reasons")
                _safe_chart(st, fig, 380)
    section_header(st, "Model Decision DNA", "Diagnostic profile — not an additional allocation model.")
    dna = bundle.dna.copy()
    if dna.empty:
        alert_box(st, "Decision DNA unavailable: supported dimensions missing.", "warning")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=dna["score"], theta=dna["dimension"], fill="toself", name="Current Portfolio", line_color=CHART_COLORS.get("orange", "#FF8A2A")))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        _safe_chart(st, fig, 430)
        st.dataframe(_safe_df(dna), width="stretch", hide_index=True)


def _pipeline_sources(st, bundle) -> None:
    section_header(st, "Pipeline Explainability and Sources", "Official/current source audit. No debug namespace mixing in current decision view.")
    st.dataframe(_safe_df(bundle.pipeline), width="stretch", hide_index=True)
    st.markdown("#### Integrity Checks")
    st.dataframe(_safe_df(bundle.integrity), width="stretch", hide_index=True)
    st.markdown("#### Source Audit")
    st.dataframe(_safe_df(bundle.source_audit), width="stretch", hide_index=True)
    if not bundle.missing_attribution.empty:
        alert_box(st, "Missing attribution data: exact marginal contribution weights are not stored. The terminal shows stored feature values only.", "warning")
        st.dataframe(_safe_df(bundle.missing_attribution), width="stretch", hide_index=True)


def render_decision_engine(st, data: dict[str, pd.DataFrame]) -> None:
    bundle = build_decision_bundle(data)
    _header(st, bundle)
    if bundle.status == "decision_engine_fail":
        alert_box(st, "Decision Engine integrity failed. See checks below before relying on the view.", "danger")
    elif bundle.status == "decision_engine_warning":
        alert_box(st, "Decision Engine loaded with warnings. Missing marginal feature contributions are labeled, not inferred.", "warning")
    else:
        alert_box(st, "Decision Engine pass: current official holdings reconcile with explainability exports.", "success")
    _sankey_and_funnel(st, bundle)
    _selected_explainability(st, bundle)
    _rejected_and_compare(st, bundle)
    _rule_path_and_pending(st, bundle)
    _history_attrition_dna(st, bundle)
    _pipeline_sources(st, bundle)
    source_caption(st, "growth_decision_engine_export.py + official/current growth CSVs", "read-only")
