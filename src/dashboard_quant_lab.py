from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from dashboard_components import (
    alert_box,
    fmt_pct,
    metric_card,
    section_header,
    source_caption,
    status_badge,
)
from dashboard_data_layer import (
    CSV_FILES,
    MODEL_VERSION,
    VARIANT,
    current_holdings,
    latest_market_date,
    numeric,
)
from dashboard_quant_embeddings import render_pca_3d
from dashboard_quant_equations import render_equation_explorer
from dashboard_quant_monte_carlo import render_monte_carlo_lab
from dashboard_quant_surfaces import (
    render_black_litterman_lab,
    render_correlation_surface,
    render_covariance_surface,
    render_efficient_frontier,
    render_feature_space_3d,
    render_hmm_regime_space,
    render_parameter_stability_surface,
    render_stress_cube,
    render_volatility_target_surface,
)

LAB_SECTIONS: dict[str, str] = {
    "Portfolio Geometry": "Efficient Frontier 3D and portfolio geometry diagnostics.",
    "Correlation & Covariance": "Correlation/covariance surfaces, heatmaps and matrix diagnostics.",
    "Volatility Targeting": "Exact clipped policy surface and unclipped target-vol surface.",
    "Parameter Stability": "Phase 92 robustness plateau surfaces. No retuning.",
    "Black-Litterman": "Research-only posterior surfaces when source data exists.",
    "Regime Space": "HMM and regime diagnostics. Diagnostic only.",
    "Feature Space": "Current feature scatter and PCA embedding.",
    "Stress Surfaces": "Shock/correlation stress cube diagnostics.",
    "Monte Carlo": "Distributional simulations using current portfolio estimates.",
    "Model Equations": "Equation library and active/diagnostic role labels.",
}


def _safe_date(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    for c in ["date", "Date", "signal_date", "market_date"]:
        if c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().any():
                return str(d.max().date())
    return ""


def _source_rows(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = [
        "official_state",
        "official_performance",
        "current_features",
        "current_raw_target_features",
        "current_allocation",
        "vol_fresh",
        "parameter_stability",
        "parameter_stability_map",
        "hmm_oos",
        "hmm_model_comparison",
        "official_benchmark_daily",
        "canonical_price_history",
        "factor_correlation_matrix",
        "garch_model_comparison",
    ]
    rows = []
    for key in keys:
        df = data.get(key, pd.DataFrame())
        rows.append(
            {
                "data_key": key,
                "source_file": CSV_FILES.get(key, f"{key}.csv"),
                "loaded": not df.empty,
                "row_count": len(df),
                "latest_date": _safe_date(df),
                "scope": "official" if key.startswith("official") or key.startswith("current") or key == "vol_fresh" else "diagnostic",
                "active_model_impact": "active"
                if key in {"official_state", "current_features", "current_raw_target_features", "vol_fresh"}
                else "diagnostic only",
            }
        )
    return pd.DataFrame(rows)


def _surface_plan(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks = [
        ("efficient_frontier", "yahoo_ohlcv_price_cache", "official holdings price cache", bool(_official_tickers(data))),
        ("correlation_surface", "yahoo_ohlcv_price_cache", "official/top candidate price cache", bool(_official_tickers(data))),
        ("covariance_surface", "yahoo_ohlcv_price_cache", "official holdings price cache", bool(_official_tickers(data))),
        (
            "volatility_target_surface",
            "growth_volatility_targeting_fresh.csv",
            "fresh vol target audit",
            not data.get("vol_fresh", pd.DataFrame()).empty,
        ),
        (
            "parameter_stability_surface",
            "parameter_stability_map.csv",
            "Phase 92 grid",
            not data.get("parameter_stability", pd.DataFrame()).empty or not data.get("parameter_stability_map", pd.DataFrame()).empty,
        ),
        ("black_litterman_lab", "black_litterman_*", "BL grids", False),
        (
            "hmm_regime_space",
            "hmm_out_of_sample_results.csv",
            "HMM diagnostics",
            not data.get("hmm_oos", pd.DataFrame()).empty or not data.get("hmm_model_comparison", pd.DataFrame()).empty,
        ),
        (
            "feature_space_3d",
            "current_growth_features.csv",
            "current feature vectors",
            not data.get("current_features", pd.DataFrame()).empty,
        ),
        ("pca_3d", "current_growth_features.csv", "numeric feature matrix", not data.get("current_features", pd.DataFrame()).empty),
        ("stress_cube", "growth_official_paper_state.csv", "current official holdings", bool(_official_tickers(data))),
        ("monte_carlo", "yahoo_ohlcv_price_cache", "official holdings returns", bool(_official_tickers(data))),
        ("equation_explorer", "code/formula registry", "static equation library", True),
    ]
    return pd.DataFrame(
        [
            {
                "surface": n,
                "source": s,
                "required_input": r,
                "status": "available" if ok else "warning",
                "detail": "input present" if ok else "input missing or diagnostic unavailable",
            }
            for n, s, r, ok in checks
        ]
    )


def _official_tickers(data: dict[str, pd.DataFrame]) -> list[str]:
    holdings = current_holdings(data, "Official Forward Paper")
    if holdings.empty or "ticker" not in holdings.columns:
        return []
    return holdings[~holdings["ticker"].astype(str).str.upper().eq("CASH")]["ticker"].astype(str).str.upper().tolist()


def build_quant_lab_audits(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    source = _source_rows(data)
    surfaces = _surface_plan(data)
    perf = pd.DataFrame(
        [
            {"item": "lazy_rendering", "status": "PASS", "detail": "section selected before render"},
            {
                "item": "cache_policy",
                "status": "PASS",
                "detail": "st.cache_data wrappers for price matrices and Monte Carlo when Streamlit is available",
            },
            {"item": "simulation_controls", "status": "PASS", "detail": "simulation count/horizon controlled by user"},
            {"item": "namespace_mixing", "status": "PASS", "detail": "official/current and diagnostic sources are labelled"},
            {"item": "model_mutation", "status": "PASS", "detail": "read-only dashboard diagnostics"},
        ]
    )
    status = "quant_lab_pass" if surfaces["status"].eq("available").all() else "quant_lab_warning"
    return source, surfaces, perf, status


def render_header(st, data: dict[str, pd.DataFrame]) -> None:
    holdings = current_holdings(data, "Official Forward Paper")
    tickers = _official_tickers(data)
    if not holdings.empty and "paper_position_weight" in holdings.columns and "ticker" in holdings.columns:
        non_cash = holdings[~holdings["ticker"].astype(str).str.upper().eq("CASH")]
        exposure = numeric(non_cash.get("paper_position_weight", pd.Series(dtype=float))).sum()
    else:
        exposure = np.nan
    lifecycle = "operational_paper_production / real_capital_blocked"
    cols = st.columns(4)
    with cols[0]:
        metric_card(st, "Active Model", MODEL_VERSION, VARIANT, badge="READ ONLY")
    with cols[1]:
        metric_card(st, "Latest Market Date", latest_market_date(data), "official data scope")
    with cols[2]:
        metric_card(st, "Current Holdings", ", ".join(tickers) if tickers else "unavailable")
    with cols[3]:
        metric_card(st, "Exposure", fmt_pct(exposure), lifecycle, badge="diagnostic")
    alert_box(
        st,
        "Diagnostic visualization only. Some models shown here are research-only and do not influence the official Growth allocation.",
        "warning",
    )


def _render_section(st, data: dict[str, pd.DataFrame], section: str) -> dict[str, object]:
    start = perf_counter()
    result: dict[str, object]
    if section == "Portfolio Geometry":
        result = render_efficient_frontier(st, data).__dict__
    elif section == "Correlation & Covariance":
        tab1, tab2 = st.tabs(["Correlation", "Covariance"])
        with tab1:
            r1 = render_correlation_surface(st, data).__dict__
        with tab2:
            r2 = render_covariance_surface(st, data).__dict__
        result = {
            "surface": "correlation_covariance",
            "status": "available" if r1.get("status") == "available" or r2.get("status") == "available" else "warning",
            "detail": f"{r1.get('status')}/{r2.get('status')}",
        }
    elif section == "Volatility Targeting":
        result = render_volatility_target_surface(st, data).__dict__
    elif section == "Parameter Stability":
        result = render_parameter_stability_surface(st, data).__dict__
    elif section == "Black-Litterman":
        result = render_black_litterman_lab(st, data).__dict__
    elif section == "Regime Space":
        result = render_hmm_regime_space(st, data).__dict__
    elif section == "Feature Space":
        tab1, tab2 = st.tabs(["Feature 3D", "PCA"])
        with tab1:
            r1_obj = render_feature_space_3d(st, data)
        with tab2:
            r2 = render_pca_3d(st, data)
        r1 = r1_obj.__dict__ if hasattr(r1_obj, "__dict__") else dict(r1_obj)
        result = {
            "surface": "feature_space",
            "status": "available" if r1.get("status") == "available" or r2.get("status") == "available" else "warning",
            "detail": f"{r1.get('detail')}; {r2.get('detail')}",
        }
    elif section == "Stress Surfaces":
        result = render_stress_cube(st, data).__dict__
    elif section == "Monte Carlo":
        result = render_monte_carlo_lab(st, data)
    else:
        result = render_equation_explorer(st, data)
    result["render_seconds"] = round(perf_counter() - start, 3)
    return result


def render_quant_lab(st, data: dict[str, pd.DataFrame]) -> None:
    render_header(st, data)
    section_header(st, "Institutional Quant Lab 3D", "Mathematical structure, risk surfaces, stability maps and research diagnostics.")
    section = st.selectbox("Lab section", list(LAB_SECTIONS.keys()), key="quant_lab_section")
    st.caption(LAB_SECTIONS[section])
    result = _render_section(st, data, section)
    st.caption(
        f"Render status: {status_badge(str(result.get('status', 'unknown')), str(result.get('status', 'diagnostic')))} · {result.get('render_seconds', 'n/a')}s",
        unsafe_allow_html=True,
    )
    with st.expander("Data Source Transparency"):
        source, surfaces, perf, status = build_quant_lab_audits(data)
        st.dataframe(source, width="stretch", hide_index=True)
        st.dataframe(surfaces, width="stretch", hide_index=True)
        st.dataframe(perf, width="stretch", hide_index=True)
        st.caption(f"Quant Lab status: {status}")
    source_caption(st, "dashboard_quant_lab.py", "read-only diagnostic")
