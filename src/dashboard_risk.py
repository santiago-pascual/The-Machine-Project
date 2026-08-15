from __future__ import annotations

from typing import Any

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
from dashboard_risk_calculations import (
    MAX_EXPOSURE,
    MIN_EXPOSURE,
    TARGET_VOL,
    build_risk_bundle,
    risk_commentary,
)
from dashboard_theme import (
    AMBER,
    BRIGHT_ORANGE,
    CHART_COLORS,
    CYAN,
    GREEN,
    INFO,
    MUTED_ORANGE,
    ORANGE,
    RED,
    apply_plotly_layout,
)

RISK_COLOR_SEQUENCE = [ORANGE, BRIGHT_ORANGE, AMBER, MUTED_ORANGE, INFO, CYAN, GREEN, RED]


def _value(metrics: dict[str, Any], key: str, default: Any = np.nan) -> Any:
    return metrics.get(key, default)


def _status_state(status: str) -> str:
    text = str(status).lower()
    if "fail" in text or "breach" in text:
        return "danger"
    if "warn" in text:
        return "warning"
    return "neutral"


def _arrow_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            sample = non_null.head(200).tolist()
            type_names = {type(v).__name__ for v in sample}
            has_text_like = any(name in type_names for name in ("str", "dict", "list", "tuple"))
            if len(type_names) > 1 or has_text_like:
                out[col] = series.map(lambda v: "" if pd.isna(v) else str(v))
    return out


def _safe_df(st, df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    if df.empty:
        st.info("No hay datos suficientes para esta tabla.")
        return
    view = df.copy()
    if columns:
        existing = [c for c in columns if c in view.columns]
        view = view[existing] if existing else view
    view = _arrow_safe_frame(view)
    kwargs = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(view, **kwargs)


def _risk_contribution_chart(st, contrib: pd.DataFrame) -> None:
    if contrib.empty or "ticker" not in contrib.columns or "pct_total_portfolio_risk" not in contrib.columns:
        st.warning("Risk contribution unavailable: missing ticker or contribution columns.")
        return
    fig = px.bar(
        contrib.sort_values("pct_total_portfolio_risk", ascending=True),
        x="pct_total_portfolio_risk",
        y="ticker",
        orientation="h",
        color="ticker",
        color_discrete_sequence=RISK_COLOR_SEQUENCE,
        text=contrib.sort_values("pct_total_portfolio_risk", ascending=True)["pct_total_portfolio_risk"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        ),
    )
    fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
    st.plotly_chart(apply_plotly_layout(fig, "Risk contribution by holding"), width="stretch")


def _exposure_waterfall(st, metrics: dict[str, Any]) -> None:
    raw = _value(metrics, "uncapped_vol_target_exposure")
    final = _value(metrics, "final_exposure")
    dual = _value(metrics, "dual_trend_cap")
    rows = pd.DataFrame(
        [
            {"stage": "Uncapped vol target", "exposure": raw, "limit": "target_vol / realized_vol"},
            {"stage": "Minimum floor", "exposure": max(raw, MIN_EXPOSURE) if pd.notna(raw) else np.nan, "limit": "min 40%"},
            {
                "stage": "Exposure cap",
                "exposure": min(max(raw, MIN_EXPOSURE), MAX_EXPOSURE) if pd.notna(raw) else np.nan,
                "limit": "max 60%",
            },
            {
                "stage": "Dual trend cap",
                "exposure": min(max(raw, MIN_EXPOSURE), MAX_EXPOSURE, dual) if pd.notna(raw) and pd.notna(dual) else np.nan,
                "limit": "60/40/25",
            },
            {"stage": "Final exposure", "exposure": final, "limit": "applied"},
        ]
    )
    if rows["exposure"].notna().sum() == 0:
        st.warning("Exposure waterfall unavailable: missing volatility target diagnostics.")
        return
    fig = px.bar(
        rows,
        x="stage",
        y="exposure",
        color="stage",
        color_discrete_sequence=RISK_COLOR_SEQUENCE,
        text=rows["exposure"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a"),
    )
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(apply_plotly_layout(fig, "Exposure limit stack"), width="stretch")
    _safe_df(st, rows, height=210)


def _heatmap(st, matrix: pd.DataFrame, title: str) -> None:
    if matrix.empty:
        st.warning(f"{title} unavailable: insufficient return history.")
        return
    fig = go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale="Oranges", zmid=0))
    st.plotly_chart(apply_plotly_layout(fig, title), width="stretch")


def _surface(st, matrix: pd.DataFrame, title: str) -> None:
    if matrix.empty or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        st.warning(f"{title} unavailable: need at least two holdings.")
        return
    fig = go.Figure(
        data=[go.Surface(z=matrix.values, x=list(range(len(matrix.columns))), y=list(range(len(matrix.index))), colorscale="Oranges")]
    )
    fig.update_layout(scene={"xaxis_title": "Asset", "yaxis_title": "Asset", "zaxis_title": "Value"})
    st.plotly_chart(apply_plotly_layout(fig, title), width="stretch")


def _network_graph(st, corr: pd.DataFrame, contrib: pd.DataFrame) -> None:
    if corr.empty or corr.shape[0] < 2:
        st.warning("Correlation network unavailable: insufficient holdings.")
        return
    tickers = list(corr.columns)
    angles = np.linspace(0, 2 * np.pi, len(tickers), endpoint=False)
    coords = {ticker: (np.cos(a), np.sin(a)) for ticker, a in zip(tickers, angles)}
    risk_map = {}
    if not contrib.empty and "ticker" in contrib.columns and "pct_total_portfolio_risk" in contrib.columns:
        risk_map = contrib.set_index("ticker")["pct_total_portfolio_risk"].to_dict()
    fig = go.Figure()
    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            val = corr.loc[a, b]
            if pd.isna(val):
                continue
            x0, y0 = coords[a]
            x1, y1 = coords[b]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={"width": max(1, abs(val) * 7), "color": f"rgba(255,132,44,{0.15 + abs(val) * 0.55})"},
                    hoverinfo="text",
                    text=f"{a}-{b}: corr {val:.2f}",
                    showlegend=False,
                )
            )
    fig.add_trace(
        go.Scatter(
            x=[coords[t][0] for t in tickers],
            y=[coords[t][1] for t in tickers],
            mode="markers+text",
            text=tickers,
            textposition="top center",
            marker={
                "size": [18 + 45 * float(risk_map.get(t, 0.05)) for t in tickers],
                "color": CHART_COLORS["growth"],
                "line": {"color": "#ffd29a", "width": 1},
            },
            showlegend=False,
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(apply_plotly_layout(fig, "Correlation network"), width="stretch")


def _drawdown_chart(st, dd: pd.DataFrame) -> None:
    if dd.empty or "date" not in dd.columns or "drawdown" not in dd.columns:
        st.warning("Drawdown chart unavailable: missing official performance history.")
        return
    fig = px.area(dd, x="date", y="drawdown")
    fig.update_traces(line_color=CHART_COLORS["negative"], fillcolor="rgba(255,73,73,0.22)")
    fig.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(apply_plotly_layout(fig, "Official forward drawdown"), width="stretch")


def _var_distribution(st, bundle) -> None:
    series = bundle.metrics.get("portfolio_return_observations", 0)
    returns = bundle.returns
    if returns.empty or bundle.holdings.empty:
        st.warning("Return distribution unavailable: insufficient price cache data.")
        return
    from dashboard_risk_calculations import portfolio_return_series

    port = portfolio_return_series(returns, bundle.holdings)
    if port.empty:
        st.warning("Return distribution unavailable: no portfolio return series.")
        return
    fig = px.histogram(port.rename("daily_return"), nbins=30, color_discrete_sequence=[CHART_COLORS["growth"]])
    if pd.notna(bundle.metrics.get("var95")):
        fig.add_vline(x=bundle.metrics["var95"], line_color=CHART_COLORS["negative"], annotation_text="VaR 95")
    if pd.notna(bundle.metrics.get("cvar95")):
        fig.add_vline(x=bundle.metrics["cvar95"], line_color="#ffb366", annotation_text="CVaR 95")
    st.plotly_chart(apply_plotly_layout(fig, f"Portfolio return distribution ({series} obs)"), width="stretch")


def _risk_vs_backtest(st, data: dict[str, pd.DataFrame], bundle) -> None:
    rows = [
        {
            "source": "Official Forward Paper",
            "namespace": "official_forward_paper",
            "volatility": bundle.metrics.get("realized_portfolio_volatility"),
            "sharpe": bundle.metrics.get("official_sharpe"),
            "max_drawdown": bundle.metrics.get("max_drawdown"),
            "exposure": bundle.metrics.get("final_exposure"),
            "note": "current official namespace only",
        }
    ]
    for key, label in [
        ("final_results", "Growth final selection backtest"),
        ("reconstructed_results", "Reconstructed stress history"),
        ("cost_results", "After-cost scenarios"),
    ]:
        df = data.get(key, pd.DataFrame())
        if df.empty:
            continue
        row = df.iloc[-1]
        rows.append(
            {
                "source": label,
                "namespace": "diagnostic_comparison",
                "volatility": row.get("volatility", row.get("annualized_volatility", np.nan)),
                "sharpe": row.get("sharpe", row.get("Sharpe", np.nan)),
                "max_drawdown": row.get("max_drawdown", row.get("max_dd", np.nan)),
                "exposure": row.get("average_exposure", row.get("avg_exposure", np.nan)),
                "note": "not mixed with official paper metrics",
            }
        )
    table = pd.DataFrame(rows)
    _safe_df(st, table)


def render_risk_terminal(st, data: dict[str, pd.DataFrame]) -> None:
    section_header(
        st,
        "Institutional Risk Terminal",
        "Official Forward Paper only by default. Diagnostic/reconstructed history is shown separately and never mixed into current risk.",
        "read only",
    )
    lookback = st.select_slider("Risk lookback", options=[20, 60, 126, 252], value=126)
    bundle = build_risk_bundle(data, lookback=lookback)
    source_caption(st, "growth_official_* + yahoo_ohlcv_price_cache", bundle.status)

    m = bundle.metrics
    cards = st.columns(4)
    with cards[0]:
        metric_card(
            st,
            "Portfolio Vol",
            fmt_pct(m.get("realized_portfolio_volatility")),
            f"target {TARGET_VOL:.0%}",
            state=_status_state(bundle.status),
        )
    with cards[1]:
        metric_card(st, "Final Exposure", fmt_pct(m.get("final_exposure")), f"uncapped {fmt_pct(m.get('uncapped_vol_target_exposure'))}")
    with cards[2]:
        metric_card(st, "Current DD", fmt_pct(m.get("current_drawdown")), f"max {fmt_pct(m.get('max_drawdown'))}")
    with cards[3]:
        metric_card(st, "HHI Concentration", fmt_num(m.get("hhi")), f"top position {fmt_pct(m.get('largest_weight'))}")

    cards = st.columns(4)
    with cards[0]:
        metric_card(st, "Beta vs SPY", fmt_num(m.get("beta_vs_spy")), "holdings based")
    with cards[1]:
        metric_card(st, "TE vs SPY", fmt_pct(m.get("tracking_error_vs_spy")), "annualized")
    with cards[2]:
        metric_card(st, "VaR 95", fmt_pct(m.get("var95")), "daily")
    with cards[3]:
        metric_card(st, "CVaR 95", fmt_pct(m.get("cvar95")), "daily")

    alert_box(st, risk_commentary(bundle), "info")

    tabs = st.tabs(
        [
            "Exposure",
            "Concentration",
            "Correlation",
            "Drawdown",
            "VaR & Tail",
            "Stress",
            "Limits",
            "Backtest Compare",
            "Data Integrity",
        ]
    )

    with tabs[0]:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            _exposure_waterfall(st, m)
        with c2:
            _safe_df(
                st,
                bundle.holdings,
                [
                    "ticker",
                    "weight",
                    "cash_weight",
                    "current_price",
                    "market_cap",
                    "median_60d_dollar_volume",
                    "holding_quality_classification",
                ],
            )

    with tabs[1]:
        c1, c2 = st.columns([1, 1])
        with c1:
            _risk_contribution_chart(st, bundle.contributions)
        with c2:
            _safe_df(
                st,
                bundle.contributions,
                [
                    "ticker",
                    "weight",
                    "standalone_volatility",
                    "pct_total_portfolio_risk",
                    "beta_vs_spy",
                    "correlation_with_portfolio",
                    "drawdown_contribution",
                    "liquidity_note",
                ],
            )

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            _heatmap(st, bundle.corr, "Correlation matrix")
            _network_graph(st, bundle.corr, bundle.contributions)
        with c2:
            _heatmap(st, bundle.cov, "Covariance matrix")
            _surface(st, bundle.corr, "3D correlation surface")

    with tabs[3]:
        _drawdown_chart(st, bundle.drawdown)
        _safe_df(st, bundle.drawdown.tail(30), height=260)

    with tabs[4]:
        c1, c2 = st.columns([1, 1])
        with c1:
            _safe_df(st, bundle.var_table)
            tail_rows = pd.DataFrame([{"metric": k, "value": v} for k, v in bundle.tail.items()])
            _safe_df(st, tail_rows)
        with c2:
            _var_distribution(st, bundle)

    with tabs[5]:
        _safe_df(st, bundle.stress)
        if not bundle.stress.empty:
            fig = px.bar(
                bundle.stress, x="scenario", y="estimated_portfolio_loss", color="scenario", color_discrete_sequence=RISK_COLOR_SEQUENCE
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".1%")
            st.plotly_chart(apply_plotly_layout(fig, "Scenario loss diagnostics"), width="stretch")

    with tabs[6]:
        _safe_df(st, bundle.limits)

    with tabs[7]:
        _risk_vs_backtest(st, data, bundle)

    with tabs[8]:
        c1, c2 = st.columns(2)
        with c1:
            _safe_df(st, bundle.source_audit)
        with c2:
            _safe_df(st, bundle.integrity)
