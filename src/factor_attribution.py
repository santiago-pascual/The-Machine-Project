from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_value(series: pd.Series, key: str, default: float = np.nan) -> float:
    try:
        value = float(series.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _direction(value: float, tolerance: float = 1e-9) -> str:
    if not np.isfinite(value) or abs(value) <= tolerance:
        return "neutral"
    return "positive" if value > 0 else "negative"


def _normalize_asset_contributions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = np.array([float(row["raw_value"]) if np.isfinite(float(row["raw_value"])) else 0.0 for row in rows])
    denominator = float(np.sum(np.abs(values)))
    if denominator <= 0:
        for row in rows:
            row["normalized_contribution"] = 0.0
        return rows
    for row in rows:
        raw = float(row["raw_value"]) if np.isfinite(float(row["raw_value"])) else 0.0
        row["normalized_contribution"] = raw / denominator
    return rows


def _factor_row(
    *,
    ticker: str,
    factor: str,
    raw_value: float,
    attribution_type: str,
    explanation: str,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "factor": factor,
        "raw_value": raw_value,
        "normalized_contribution": np.nan,
        "direction": _direction(raw_value),
        "attribution_type": attribution_type,
        "explanation": explanation,
    }


def build_factor_attribution(
    *,
    selected_tickers: list[str],
    diagnostics_df_full: pd.DataFrame,
    final_expected_returns: pd.Series,
    timing_df: pd.DataFrame | None = None,
    regime_score: float = 0.0,
    regime_type: str = "neutral",
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if diagnostics_df_full.empty:
        warnings.append("diagnostics_df_full is empty; attribution unavailable.")
        return pd.DataFrame(), warnings

    timing = timing_df if timing_df is not None else pd.DataFrame()
    rows: list[dict[str, object]] = []

    for ticker in selected_tickers:
        if ticker not in diagnostics_df_full.index:
            warnings.append(f"{ticker}: missing diagnostics row.")
            continue

        diag = diagnostics_df_full.loc[ticker]
        timing_row = timing.loc[ticker] if ticker in timing.index else pd.Series(dtype=float)
        asset_rows: list[dict[str, object]] = []

        total_return = _safe_value(diag, "total_return", default=np.nan)
        expected_return = float(pd.to_numeric(final_expected_returns, errors="coerce").get(ticker, np.nan))
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="target_return_component",
                raw_value=total_return if np.isfinite(total_return) else expected_return,
                attribution_type="proxy attribution",
                explanation="Target/current price gap from diagnostics; proxy for target-driven edge.",
            )
        )

        signal_strength = _safe_value(diag, "signal_strength", default=np.nan)
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="signal_strength_component",
                raw_value=(signal_strength - 0.5) if np.isfinite(signal_strength) else 0.0,
                attribution_type="proxy attribution",
                explanation="Signal strength centered around 0.5; higher values support selection.",
            )
        )

        quality_score = _safe_value(diag, "quality_score", default=np.nan)
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="quality_score_component",
                raw_value=(quality_score - 0.5) if np.isfinite(quality_score) else 0.0,
                attribution_type="proxy attribution",
                explanation="Quality score centered around 0.5; captures return quality and stability.",
            )
        )

        timing_score = _safe_value(timing_row, "ema_timing_score", default=np.nan)
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="timing_component",
                raw_value=(timing_score - 0.5) if np.isfinite(timing_score) else 0.0,
                attribution_type="proxy attribution",
                explanation="EMA timing score centered around 0.5; entry/timing quality.",
            )
        )
        if not np.isfinite(timing_score):
            warnings.append(f"{ticker}: timing score missing; timing attribution set to neutral.")

        downside_ratio = _safe_value(diag, "downside_ratio", default=np.nan)
        recent_drawdown = abs(_safe_value(diag, "recent_drawdown", default=0.0))
        downside_raw = -0.5 * (downside_ratio if np.isfinite(downside_ratio) else 0.0) - 0.5 * recent_drawdown
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="downside_component",
                raw_value=downside_raw,
                attribution_type="proxy attribution",
                explanation="Penalty proxy from downside ratio and recent drawdown.",
            )
        )

        regime_raw = float(regime_score)
        if str(regime_type).lower() in {"risk_off", "defensive"}:
            regime_raw = -abs(regime_raw)
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="regime_component",
                raw_value=regime_raw,
                attribution_type="proxy attribution",
                explanation="Market regime score applied as broad macro context, not asset-specific.",
            )
        )

        quant_confidence = _safe_value(diag, "target_confidence_quant", default=np.nan)
        if not np.isfinite(quant_confidence):
            quant_confidence = _safe_value(diag, "target_confidence", default=np.nan)
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="quant_confidence_component",
                raw_value=(quant_confidence - 0.5) if np.isfinite(quant_confidence) else 0.0,
                attribution_type="proxy attribution",
                explanation="Target confidence centered around 0.5; confidence in target quality.",
            )
        )

        volatility = _safe_value(diag, "volatility", default=np.nan)
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="volatility_component",
                raw_value=-(volatility if np.isfinite(volatility) else 0.0),
                attribution_type="proxy attribution",
                explanation="Volatility is treated as risk drag for attribution.",
            )
        )

        momentum = _safe_value(diag, "momentum", default=0.0)
        trend_alignment = _safe_value(diag, "trend_alignment", default=0.0)
        trend_score = _safe_value(timing_row, "trend_score", default=np.nan)
        trend_raw = 0.5 * momentum + 0.25 * trend_alignment + 0.25 * ((trend_score - 0.5) if np.isfinite(trend_score) else 0.0)
        asset_rows.append(
            _factor_row(
                ticker=ticker,
                factor="momentum_trend_component",
                raw_value=trend_raw,
                attribution_type="proxy attribution",
                explanation="Composite proxy from momentum, trend alignment, and EMA trend score when available.",
            )
        )

        rows.extend(_normalize_asset_contributions(asset_rows))

    attribution = pd.DataFrame(rows)
    if attribution.empty:
        warnings.append("No selected assets could be attributed.")
    else:
        exact_count = int((attribution["attribution_type"] == "exact attribution").sum())
        if exact_count == 0:
            warnings.append("All factor rows are proxy attribution because exact internal marginal contributions are not stored.")
    return attribution, warnings


def print_factor_attribution(
    *,
    selected_tickers: list[str],
    diagnostics_df_full: pd.DataFrame,
    final_expected_returns: pd.Series,
    timing_df: pd.DataFrame | None = None,
    regime_score: float = 0.0,
    regime_type: str = "neutral",
) -> pd.DataFrame:
    attribution, warnings = build_factor_attribution(
        selected_tickers=selected_tickers,
        diagnostics_df_full=diagnostics_df_full,
        final_expected_returns=final_expected_returns,
        timing_df=timing_df,
        regime_score=regime_score,
        regime_type=regime_type,
    )

    print("\n===== FACTOR ATTRIBUTION =====")
    if attribution.empty:
        print("No attribution available.")
    else:
        display_cols = [
            "ticker",
            "factor",
            "raw_value",
            "normalized_contribution",
            "direction",
            "attribution_type",
            "explanation",
        ]
        print(attribution[display_cols].sort_values(["ticker", "normalized_contribution"], ascending=[True, False]))

    print("\n===== TOP POSITIVE FACTORS =====")
    if attribution.empty:
        print("None")
    else:
        print(attribution.sort_values("normalized_contribution", ascending=False).head(10))

    print("\n===== TOP NEGATIVE FACTORS =====")
    if attribution.empty:
        print("None")
    else:
        print(attribution.sort_values("normalized_contribution", ascending=True).head(10))

    if warnings:
        print("\nFactor attribution warnings:")
        for warning in warnings:
            print(f"- {warning}")

    return attribution
