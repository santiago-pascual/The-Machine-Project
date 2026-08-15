
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_components import alert_box, fmt_num, fmt_pct, metric_card, section_header, source_caption
from dashboard_research_calculations import build_research_bundle
from dashboard_theme import AMBER, BRIGHT_ORANGE, CHART_COLORS, CYAN, GREEN, INFO, MUTED_ORANGE, ORANGE, RED, apply_plotly_layout

RESEARCH_COLORS = [ORANGE, BRIGHT_ORANGE, AMBER, MUTED_ORANGE, INFO, CYAN, GREEN, RED]


def _arrow_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_object_dtype(s):
            sample = s.dropna().head(200).tolist()
            types = {type(v).__name__ for v in sample}
            if len(types) > 1 or any(t in types for t in {"str", "dict", "list", "tuple"}):
                out[col] = s.map(lambda v: "" if pd.isna(v) else str(v))
    return out


def _safe_df(st, df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    if df.empty:
        st.info("No data available for this section.")
        return
    view = df.copy()
    if columns:
        keep = [c for c in columns if c in view.columns]
        view = view[keep] if keep else view
    kwargs = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(_arrow_safe_frame(view), **kwargs)


def _metric_state(value: Any, good_low: bool = False, warn: float | None = None, fail: float | None = None) -> str:
    try:
        x = float(value)
    except Exception:
        return "neutral"
    if warn is None:
        return "neutral"
    if good_low:
        if fail is not None and x >= fail:
            return "danger"
        if x >= warn:
            return "warning"
    else:
        if fail is not None and x <= fail:
            return "danger"
        if x <= warn:
            return "warning"
    return "neutral"


def _header(st, bundle) -> None:
    k = bundle.kpis
    section_header(
        st,
        "Research Terminal",
        "Statistical validation, anti-overfitting, OOS evidence, feature decay, parameter robustness and lifecycle status. Read-only.",
        bundle.status,
    )
    cols = st.columns(4)
    with cols[0]:
        metric_card(st, "Active Model", str(k.get("active_model")), str(k.get("lifecycle_status")), badge="frozen")
    with cols[1]:
        metric_card(st, "Official Paper", str(k.get("official_paper_status")), str(k.get("real_capital_status")), badge="blocked")
    with cols[2]:
        metric_card(st, "Research Update", str(k.get("latest_research_update")), "latest available CSV date", badge="diagnostic")
    with cols[3]:
        metric_card(st, "Config Hash", str(k.get("frozen_configuration_hash", "unavailable")), "frozen registry", badge="read only")


def _top_cards(st, bundle) -> None:
    k = bundle.kpis
    cards = [
        ("All-Time Trials", fmt_num(k.get("all_time_trials"), 0), "anti_overfitting_governance.csv", "warning"),
        ("Effective Trials", fmt_num(k.get("effective_independent_trials"), 0), "effective_trial_count.csv", "neutral"),
        ("CSCV PBO", fmt_pct(k.get("CSCV_PBO")), "pbo_distribution.csv", _metric_state(k.get("CSCV_PBO"), True, 0.5, 0.7)),
        ("Exact DSR", fmt_num(k.get("exact_deflated_sharpe")), "deflated_sharpe_exact.csv", "neutral"),
        ("DSR p-value", fmt_num(k.get("DSR_p_value"), 4), "deflated_sharpe_exact.csv", _metric_state(k.get("DSR_p_value"), True, 0.05, 0.20)),
        ("Reality/SPA p", fmt_num(k.get("reality_check_p_value"), 4), "reality_check_results.csv", "neutral"),
        ("Mean OOS Sharpe", fmt_num(k.get("mean_oos_sharpe")), "out_of_sample_governance.csv", _metric_state(k.get("mean_oos_sharpe"), False, 0.7, 0.2)),
        ("Positive OOS Folds", fmt_pct(k.get("positive_oos_folds_pct")), "purged folds", "neutral"),
        ("Holdout Sharpe", fmt_num(k.get("locked_holdout_sharpe")), "locked_holdout_results.csv", "neutral"),
        ("Holdout CAGR", fmt_pct(k.get("locked_holdout_CAGR")), "locked holdout", "neutral"),
        ("Holdout Max DD", fmt_pct(k.get("locked_holdout_max_drawdown")), "locked holdout", "warning"),
        ("Parameter Stability", str(k.get("parameter_stability_classification")), "parameter_governance.csv", "neutral"),
    ]
    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (label, value, note, state) in zip(cols, cards[i:i+4]):
            with col:
                metric_card(st, label, value, note, state=state, badge="research")


def _anti_panel(st, bundle) -> None:
    st.markdown("### Anti-Overfitting Panel")
    anti = bundle.anti_overfitting
    c1, c2 = st.columns(2)
    with c1:
        pbo = anti["pbo"]
        if not pbo.empty and "S" in pbo.columns and "PBO" in pbo.columns:
            fig = px.bar(pbo, x="S", y="PBO", color="S", color_discrete_sequence=RESEARCH_COLORS)
            fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(apply_plotly_layout(fig, "CSCV PBO by S"), width="stretch")
        _safe_df(st, pbo)
    with c2:
        cscv = anti["cscv"]
        if not cscv.empty and "lambda_logit" in cscv.columns:
            fig = px.histogram(cscv, x="lambda_logit", color="overfit_fold" if "overfit_fold" in cscv.columns else None, color_discrete_sequence=RESEARCH_COLORS)
            st.plotly_chart(apply_plotly_layout(fig, "PBO lambda distribution"), width="stretch")
        _safe_df(st, anti["effective_trials"])
    with st.expander("Deflated Sharpe diagnostics"):
        _safe_df(st, anti["dsr"], height=360)
    with st.expander("Reality Check / SPA"):
        _safe_df(st, anti["reality_check"])
    alert_box(st, "Multiple-testing-adjusted Sharpe is visible alongside CSCV PBO. High PBO is not suppressed by strong DSR.", "warning")


def _walk_forward(st, bundle) -> None:
    st.markdown("### Walk-Forward Terminal")
    res = bundle.walk_forward["results"]
    folds = bundle.walk_forward["folds"]
    if not res.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.bar(res, x="fold_id", y="Sharpe", color="fold_type", color_discrete_sequence=RESEARCH_COLORS)
            st.plotly_chart(apply_plotly_layout(fig, "OOS Sharpe by fold"), width="stretch")
        with c2:
            fig = px.bar(res, x="fold_id", y="CAGR", color="fold_type", color_discrete_sequence=RESEARCH_COLORS)
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(apply_plotly_layout(fig, "OOS CAGR by fold"), width="stretch")
        with c3:
            fig = px.bar(res, x="fold_id", y="max_drawdown", color="fold_type", color_discrete_sequence=RESEARCH_COLORS)
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(apply_plotly_layout(fig, "OOS Max Drawdown by fold"), width="stretch")
        heat_cols = [c for c in ["Sharpe", "CAGR", "max_drawdown", "hit_rate", "average_exposure"] if c in res.columns]
        if heat_cols:
            heat = res.set_index("fold_id")[heat_cols].apply(pd.to_numeric, errors="coerce")
            fig = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Oranges"))
            st.plotly_chart(apply_plotly_layout(fig, "Fold consistency heatmap"), width="stretch")
    _safe_df(st, folds, height=320)
    with st.expander("Purged walk-forward results"):
        _safe_df(st, res, height=420)


def _holdout(st, bundle) -> None:
    st.markdown("### Locked Holdout Panel")
    hold = bundle.walk_forward["holdout"]
    _safe_df(st, hold)
    alert_box(st, "Locked holdout was evaluated once in Phase 87, but later development may reduce its status as a permanently untouched sample.", "warning")


def _parameter(st, bundle) -> None:
    st.markdown("### Parameter Stability")
    stab = bundle.parameter["stability"]
    plateau = bundle.parameter["plateau"]
    _safe_df(st, plateau)
    if stab.empty:
        st.warning("Parameter surface unavailable.")
        return
    metric = st.selectbox("3D surface metric", [c for c in ["Sharpe", "CAGR", "max_drawdown", "Calmar"] if c in stab.columns], index=0)
    surf = stab.groupby(["target_vol", "exposure_cap"], as_index=False)[metric].mean()
    pivot = surf.pivot(index="target_vol", columns="exposure_cap", values=metric)
    if not pivot.empty:
        fig = go.Figure(data=[go.Surface(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Oranges")])
        current = stab[stab.get("is_current_config", False).astype(bool)] if "is_current_config" in stab.columns else pd.DataFrame()
        if not current.empty:
            row = current.iloc[0]
            fig.add_trace(go.Scatter3d(x=[row.get("exposure_cap")], y=[row.get("target_vol")], z=[row.get(metric)], mode="markers", marker=dict(size=8, color=RED), name="active config"))
        fig.update_layout(scene=dict(xaxis_title="Exposure cap", yaxis_title="Target vol", zaxis_title=metric))
        st.plotly_chart(apply_plotly_layout(fig, "Parameter robustness surface"), width="stretch")
    heat_cols = ["min_exposure", "vol_lookback_days", "dual_trend_caps", "Sharpe"]
    _safe_df(st, stab[heat_cols + ["is_current_config"]].head(300) if all(c in stab.columns for c in heat_cols) else stab.head(300), height=360)


def _features(st, bundle) -> None:
    st.markdown("### Feature Evidence")
    _safe_df(st, bundle.feature_evidence, height=360)
    raw = bundle.feature_evidence[bundle.feature_evidence["feature"].eq("raw_target_return_exact")]
    sig = bundle.feature_evidence[bundle.feature_evidence["feature"].eq("signal_strength")]
    c1, c2 = st.columns(2)
    with c1:
        if not raw.empty:
            metric_card(st, "Raw Target Evidence", fmt_num(raw.iloc[0].get("rank_ic")), raw.iloc[0].get("evidence_note", ""), badge="active")
    with c2:
        if not sig.empty:
            metric_card(st, "Signal Strength", str(sig.iloc[0].get("role")), sig.iloc[0].get("evidence_note", ""), badge="diagnostic")


def _ic_decay(st, bundle) -> None:
    st.markdown("### IC Decay Terminal")
    alpha = bundle.ic_decay
    rolling = bundle.feature_evidence
    raw_alpha = alpha.copy()
    if not raw_alpha.empty and "horizon" in raw_alpha.columns:
        y = "mean_rank_ic" if "mean_rank_ic" in raw_alpha.columns else "mean_spearman_rank_ic"
        if y in raw_alpha.columns:
            fig = px.line(raw_alpha, x="horizon", y=y, color="feature" if "feature" in raw_alpha.columns else None, markers=True, color_discrete_sequence=RESEARCH_COLORS)
            st.plotly_chart(apply_plotly_layout(fig, "Rank IC by horizon"), width="stretch")
    _safe_df(st, alpha, height=360)


def _models(st, bundle) -> None:
    st.markdown("### Model Comparison")
    _safe_df(st, bundle.model_comparison, height=420)


def _registry_lifecycle(st, bundle) -> None:
    st.markdown("### Research Registry")
    reg = bundle.registry
    t1, t2, t3 = st.tabs(["Governed", "Frozen Champion", "Lifecycle"])
    with t1:
        _safe_df(st, reg["governed"], height=360)
    with t2:
        _safe_df(st, reg["frozen"], height=360)
    with t3:
        life = reg["lifecycle"]
        _safe_df(st, life, height=420)
        stages = ["research", "candidate", "shadow", "paper", "operational paper production", "real-capital blocked"]
        fig = go.Figure(go.Scatter(x=list(range(len(stages))), y=[1]*len(stages), mode="lines+markers+text", text=stages, textposition="top center", marker=dict(size=16, color=RESEARCH_COLORS[:len(stages)])))
        fig.update_yaxes(visible=False)
        st.plotly_chart(apply_plotly_layout(fig, "Lifecycle timeline"), width="stretch")


def _warnings(st, bundle) -> None:
    st.markdown("### Research Warnings")
    _safe_df(st, bundle.warnings)
    alert_box(st, bundle.commentary, "warning")


def _sources(st, bundle) -> None:
    st.markdown("### Research Data Sources")
    _safe_df(st, bundle.source_audit, height=420)
    st.markdown("### Integrity")
    _safe_df(st, bundle.integrity)
    st.markdown("### Evidence Reconciliation")
    _safe_df(st, bundle.evidence_reconciliation)


def render_research_terminal(st, data: dict[str, pd.DataFrame]) -> None:
    bundle = build_research_bundle(data)
    _header(st, bundle)
    source_caption(st, "research governance and validation CSVs", bundle.status)
    _top_cards(st, bundle)
    _warnings(st, bundle)
    tabs = st.tabs([
        "Anti-Overfitting",
        "Walk-Forward",
        "Locked Holdout",
        "Parameters",
        "Feature Evidence",
        "IC Decay",
        "Model Comparison",
        "Registry & Lifecycle",
        "Sources",
    ])
    with tabs[0]:
        _anti_panel(st, bundle)
    with tabs[1]:
        _walk_forward(st, bundle)
    with tabs[2]:
        _holdout(st, bundle)
    with tabs[3]:
        _parameter(st, bundle)
    with tabs[4]:
        _features(st, bundle)
    with tabs[5]:
        _ic_decay(st, bundle)
    with tabs[6]:
        _models(st, bundle)
    with tabs[7]:
        _registry_lifecycle(st, bundle)
    with tabs[8]:
        _sources(st, bundle)
