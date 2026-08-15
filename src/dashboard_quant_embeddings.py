from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from dashboard_components import alert_box, metric_card, source_caption
from dashboard_data_layer import latest, numeric
from dashboard_theme import apply_plotly_layout


def _numeric_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exclude = {"date", "ticker"}
    cols = []
    seen = set()
    for c in df.columns:
        if c in exclude or c in seen:
            continue
        try:
            ok = numeric(df[c]).notna().sum() >= 8
        except Exception:
            ok = False
        if ok:
            cols.append(c)
            seen.add(c)
    if not cols:
        return pd.DataFrame(), []
    x = df.loc[:, cols].copy()
    x = x.loc[:, ~x.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    cols = list(x.columns)
    keep = x.notna().mean(axis=1) >= 0.55
    x = x[keep].copy()
    labels = df.loc[
        x.index, [c for c in ["ticker", "sector", "holding_quality_classification", "raw_target_selected"] if c in df.columns]
    ].copy()
    x = x.fillna(x.median(numeric_only=True)).replace([np.inf, -np.inf], np.nan).fillna(0)
    std = x.std(ddof=0).replace(0, 1)
    x = (x - x.mean()) / std
    return pd.concat([labels, x], axis=1), cols


def render_pca_3d(st, data: dict[str, pd.DataFrame]) -> dict[str, object]:
    st.markdown("#### PCA / Dimensionality Reduction")
    df = latest(data.get("current_features", pd.DataFrame()))
    if df.empty:
        alert_box(st, "PCA unavailable: current_growth_features.csv missing.", "warning")
        return {"surface": "pca_3d", "status": "warning", "detail": "missing current features"}
    matrix, cols = _numeric_feature_matrix(df)
    if matrix.empty or len(cols) < 3:
        alert_box(st, "PCA unavailable: not enough numeric feature columns.", "warning")
        return {"surface": "pca_3d", "status": "warning", "detail": "insufficient numeric columns"}
    x = matrix[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        scores = u[:, :3] * s[:3]
        explained = (s**2) / max(np.sum(s**2), 1e-12)
    except Exception as exc:
        alert_box(st, f"PCA unavailable: {exc}", "warning")
        return {"surface": "pca_3d", "status": "warning", "detail": str(exc)}
    plot = pd.DataFrame(scores, columns=["PC1", "PC2", "PC3"])
    for c in ["ticker", "sector", "holding_quality_classification", "raw_target_selected"]:
        if c in matrix.columns:
            values = matrix[c]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        plot[c] = values.values
    color = st.selectbox(
        "PCA color",
        [c for c in ["raw_target_selected", "sector", "holding_quality_classification"] if c in plot.columns] or ["PC1"],
        key="qlab_pca_color",
    )
    fig = px.scatter_3d(
        plot,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color,
        hover_name="ticker" if "ticker" in plot.columns else None,
        title="PCA 3D Feature Embedding",
    )
    apply_plotly_layout(fig, "PCA 3D Feature Embedding")
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")
    cols_ui = st.columns(3)
    with cols_ui[0]:
        metric_card(st, "Sample Count", str(len(plot)))
    with cols_ui[1]:
        metric_card(st, "Features Used", str(len(cols)))
    with cols_ui[2]:
        metric_card(st, "Variance PC1-3", f"{explained[:3].sum():.1%}")
    with st.expander("Feature list"):
        st.write(", ".join(cols))
    source_caption(st, "current_growth_features.csv", "diagnostic embedding — not used in allocation")
    return {"surface": "pca_3d", "status": "available", "detail": f"samples={len(plot)}, features={len(cols)}"}
