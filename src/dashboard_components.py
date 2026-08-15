from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_data_layer import numeric
from dashboard_theme import (
    CHART_COLORS,
    apply_plotly_layout,
)


def fmt_pct(x: Any, d: int = 2) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x) * 100:.{d}f}%"
    except Exception:
        return "n/a"


def fmt_pct_points(x: Any, d: int = 2) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.{d}f}%"
    except Exception:
        return "n/a"


def fmt_money(x: Any) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"${float(x):,.2f}"
    except Exception:
        return "n/a"


def fmt_num(x: Any, d: int = 3) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.{d}f}"
    except Exception:
        return "n/a"


def _safe_html(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def status_badge(label: str, status: str = "neutral") -> str:
    key = str(status or label).lower().replace(" ", "_").replace("-", "_")
    if "pass" in key or "confirmed" in key:
        key = "pass"
    elif "warmup" in key:
        key = "warmup"
    elif "warn" in key or "single_source" in key:
        key = "warning" if "single_source" not in key else "single_source_fresh"
    elif "block" in key:
        key = "blocked"
    elif "fail" in key:
        key = "failed"
    elif "official" in key:
        key = "official"
    elif "diagnostic" in key:
        key = "diagnostic"
    elif "reconstruct" in key:
        key = "reconstructed"
    return f"<span class='badge {key}'>{_safe_html(label)}</span>"


def metric_card(st, label: str, value: str, note: str = "", delta: str = "", state: str = "neutral", badge: str = "") -> None:
    badge_html = status_badge(badge, badge) if badge else ""
    delta_html = f"<span class='small-muted'>{_safe_html(delta)}</span>" if delta else ""
    st.markdown(
        f"<div class='kpi {state}'><div class='kpi-label'>{_safe_html(label)} {badge_html}</div><div class='kpi-value'>{_safe_html(value)}</div><div class='kpi-note'>{_safe_html(note)} {delta_html}</div></div>",
        unsafe_allow_html=True,
    )


def section_header(st, title: str, subtitle: str = "", badge: str = "") -> None:
    badge_html = status_badge(badge, badge) if badge else ""
    st.markdown(
        f"<div class='page-head'><div><h2 class='page-title'>{_safe_html(title)} {badge_html}</h2><div class='page-subtitle'>{_safe_html(subtitle)}</div></div></div>",
        unsafe_allow_html=True,
    )


def alert_box(st, message: str, state: str = "info") -> None:
    st.markdown(f"<div class='alert-box {state}'>{_safe_html(message)}</div>", unsafe_allow_html=True)


def source_caption(st, source: str, status: str = "official") -> None:
    st.caption(f"Source: `{source}` · scope: {status} · read-only")


def line_chart(st, df: pd.DataFrame, x: str, y: list[str] | str, title: str) -> None:
    if df.empty or x not in df.columns:
        st.warning(f"Chart unavailable: missing {x}")
        return
    ys = y if isinstance(y, list) else [y]
    missing = [col for col in ys if col not in df.columns]
    if missing:
        st.warning(f"Chart unavailable: missing columns {', '.join(missing)}")
        return
    fig = px.line(df, x=x, y=ys, markers=True)
    color_map = {
        "Growth Gross": CHART_COLORS["growth"],
        "Growth Estimated Net": CHART_COLORS["growth_net"],
        "Growth Champion Final": CHART_COLORS["growth"],
        "SPY": CHART_COLORS["spy"],
        "QQQ": CHART_COLORS["qqq"],
        "Gross Equity": CHART_COLORS["growth"],
        "Estimated Net Equity": CHART_COLORS["growth_net"],
    }
    for trace in fig.data:
        if trace.name in color_map:
            trace.line.color = color_map[trace.name]
            trace.marker.color = color_map[trace.name]
    fig = apply_plotly_layout(fig, title)
    st.plotly_chart(fig, width="stretch")


def bar_chart(st, df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.warning(f"Chart unavailable: missing columns {x}, {y}")
        return
    fig = px.bar(df, x=x, y=y, color=color if color in df.columns else None)
    fig = apply_plotly_layout(fig, title)
    st.plotly_chart(fig, width="stretch")


def heatmap(st, matrix: pd.DataFrame, title: str) -> None:
    if matrix.empty:
        st.warning("Insufficient data for this heatmap.")
        return
    fig = go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale="Oranges"))
    fig = apply_plotly_layout(fig, title)
    st.plotly_chart(fig, width="stretch")


def drawdown_frame(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty or "date" not in perf.columns:
        return pd.DataFrame()
    out = perf.sort_values("date").copy()
    if "gross_equity" in out.columns:
        equity = numeric(out["gross_equity"])
    elif "gross_portfolio_value" in out.columns:
        equity = numeric(out["gross_portfolio_value"])
    elif "portfolio_value" in out.columns:
        equity = numeric(out["portfolio_value"])
    elif "gross_daily_return" in out.columns:
        equity = (1 + numeric(out["gross_daily_return"]).fillna(0)).cumprod() * 100000
    else:
        return pd.DataFrame()
    out["drawdown"] = equity / equity.cummax() - 1
    return out[["date", "drawdown"]]


def monthly_return_table(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty or "date" not in perf.columns:
        return pd.DataFrame()
    ret_col = "gross_daily_return" if "gross_daily_return" in perf.columns else "daily_return" if "daily_return" in perf.columns else None
    if ret_col is None:
        return pd.DataFrame()
    work = perf.dropna(subset=["date"]).copy()
    work["year"] = work["date"].dt.year
    work["month"] = work["date"].dt.month
    out = work.groupby(["year", "month"])[ret_col].apply(lambda s: (1 + numeric(s).fillna(0)).prod() - 1).reset_index(name="return")
    return out.pivot(index="year", columns="month", values="return")


def action_counts(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty or "date" not in actions.columns or "action" not in actions.columns:
        return pd.DataFrame()
    out = actions.copy()
    out["action"] = out["action"].astype(str).str.upper()
    return out.groupby(["date", "action"]).size().reset_index(name="count")


def latest_value(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    vals = numeric(df[col]).dropna()
    return float(vals.iloc[-1]) if not vals.empty else np.nan
