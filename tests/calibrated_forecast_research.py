from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CALIBRATED_FORECAST_FILE = "walk_forward_calibrated_forecasts.csv"


@dataclass(frozen=True)
class CalibratedForecastConfig:
    forecast_file: str = DEFAULT_CALIBRATED_FORECAST_FILE
    horizon_days: int = 20
    max_staleness_days: int = 45
    source_model_mode: str = "regime_gated_full_quant"


def _safe_date(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _total_to_daily(total_return: pd.Series, horizon_days: int) -> pd.Series:
    horizon = max(1, int(horizon_days))
    values = pd.to_numeric(total_return, errors="coerce").replace([np.inf, -np.inf], np.nan)
    geometric = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values > -0.999
    geometric.loc[valid] = np.power(1.0 + values.loc[valid], 1.0 / horizon) - 1.0
    geometric.loc[~valid] = values.loc[~valid] / horizon
    return geometric.replace([np.inf, -np.inf], np.nan)


def _load_calibrated_slice(
    current_date: pd.Timestamp,
    tickers: list[str],
    config: CalibratedForecastConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(config.forecast_file)
    metadata: dict[str, object] = {
        "file_exists": path.exists(),
        "stale": False,
        "latest_date": None,
        "selected_date": None,
        "staleness_days": np.nan,
        "failure_reason": "none",
    }
    if not path.exists():
        metadata["failure_reason"] = "missing_calibration_file"
        return pd.DataFrame(), metadata

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        metadata["failure_reason"] = f"read_error: {exc}"
        return pd.DataFrame(), metadata

    required = {"date", "ticker", f"wf_calibrated_expected_return_{config.horizon_days}d"}
    missing = sorted(required - set(df.columns))
    if missing:
        metadata["failure_reason"] = f"missing_columns: {missing}"
        return pd.DataFrame(), metadata

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    df = df.dropna(subset=["date", "ticker"])
    if "model_mode" in df.columns:
        mode_df = df[df["model_mode"].astype(str).str.lower().eq(config.source_model_mode.lower())]
        if not mode_df.empty:
            df = mode_df

    df = df[df["ticker"].isin(tickers)]
    df = df[df["date"] <= current_date]
    if df.empty:
        metadata["failure_reason"] = "no_calibrated_rows_for_date_or_tickers"
        return pd.DataFrame(), metadata

    latest_date = pd.Timestamp(df["date"].max()).normalize()
    staleness_days = int((current_date - latest_date).days)
    metadata["latest_date"] = latest_date.strftime("%Y-%m-%d")
    metadata["selected_date"] = latest_date.strftime("%Y-%m-%d")
    metadata["staleness_days"] = staleness_days
    if staleness_days > int(config.max_staleness_days):
        metadata["stale"] = True
        metadata["failure_reason"] = f"stale_calibration_file_{staleness_days}d"
        return pd.DataFrame(), metadata

    latest_rows = (
        df[df["date"].eq(latest_date)]
        .sort_values(["ticker"])
        .drop_duplicates(subset=["ticker"], keep="last")
        .set_index("ticker")
    )
    return latest_rows.reindex(tickers), metadata


def apply_walk_forward_calibrated_forecasts(
    expected_daily_returns: pd.Series,
    diagnostics_df: pd.DataFrame,
    current_date: object,
    config: CalibratedForecastConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cfg = config or CalibratedForecastConfig()
    tickers = [str(ticker) for ticker in expected_daily_returns.index]
    decision_date = _safe_date(current_date) or pd.Timestamp.today().normalize()
    original = pd.to_numeric(expected_daily_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diagnostics = diagnostics_df.copy().reindex(original.index)

    rows, metadata = _load_calibrated_slice(decision_date, tickers, cfg)
    horizon_col = f"wf_calibrated_expected_return_{cfg.horizon_days}d"
    if rows.empty or horizon_col not in rows.columns:
        report = pd.DataFrame(
            {
                "original_expected_daily_return": original,
                "calibrated_expected_daily_return": original,
                "calibration_adjustment": 0.0,
                "used_calibrated_forecast": False,
                "fallback_original_forecast": True,
            }
        )
        metadata.update(
            {
                "calibrated_forecasts_used": 0,
                "fallback_original_forecasts": len(original),
                "calibrated_confidence_used": 0,
                "average_calibration_adjustment": 0.0,
            }
        )
        return original, diagnostics, report, metadata

    calibrated_total = pd.to_numeric(rows[horizon_col], errors="coerce").reindex(original.index)
    calibrated_daily = _total_to_daily(calibrated_total, cfg.horizon_days)
    use_mask = calibrated_daily.notna()
    adjusted = original.copy()
    adjusted.loc[use_mask] = calibrated_daily.loc[use_mask].astype(float)

    confidence_used = 0
    if "calibrated_target_confidence" in rows.columns:
        calibrated_confidence = pd.to_numeric(rows["calibrated_target_confidence"], errors="coerce").reindex(original.index)
        confidence_mask = calibrated_confidence.notna()
        if confidence_mask.any():
            diagnostics.loc[confidence_mask, "target_confidence"] = calibrated_confidence.loc[confidence_mask].clip(0.0, 1.0)
            diagnostics.loc[confidence_mask, "calibrated_target_confidence"] = calibrated_confidence.loc[confidence_mask].clip(0.0, 1.0)
            confidence_used = int(confidence_mask.sum())

    diagnostics["original_expected_daily_return"] = original.reindex(diagnostics.index)
    diagnostics["calibrated_expected_daily_return"] = adjusted.reindex(diagnostics.index)
    diagnostics["walk_forward_calibration_used"] = use_mask.reindex(diagnostics.index).fillna(False)

    report = pd.DataFrame(
        {
            "original_expected_daily_return": original,
            "calibrated_expected_daily_return": adjusted,
            "calibration_adjustment": adjusted - original,
            "used_calibrated_forecast": use_mask.reindex(original.index).fillna(False),
            "fallback_original_forecast": ~use_mask.reindex(original.index).fillna(False),
        }
    )
    metadata.update(
        {
            "calibrated_forecasts_used": int(use_mask.sum()),
            "fallback_original_forecasts": int((~use_mask).sum()),
            "calibrated_confidence_used": confidence_used,
            "average_calibration_adjustment": float((adjusted - original).abs().mean()) if len(adjusted) else 0.0,
        }
    )
    return adjusted.replace([np.inf, -np.inf], np.nan).fillna(0.0), diagnostics, report, metadata


def print_calibrated_forecast_research_report(
    report: pd.DataFrame,
    metadata: dict[str, object],
    top_n: int = 10,
) -> None:
    print("\n===== CALIBRATED FORECAST RESEARCH MODE =====")
    print(f"calibrated forecasts used: {int(metadata.get('calibrated_forecasts_used', 0))}")
    print(f"fallback original forecasts: {int(metadata.get('fallback_original_forecasts', 0))}")
    print(f"calibrated confidence used: {int(metadata.get('calibrated_confidence_used', 0))}")
    print(f"average calibration adjustment: {float(metadata.get('average_calibration_adjustment', 0.0)):.8f}")
    print(f"calibration file latest date: {metadata.get('latest_date', 'missing')}")
    print(f"governance status: {'research/paper only' if int(metadata.get('calibrated_forecasts_used', 0)) else 'fallback_to_original'}")
    if metadata.get("failure_reason") not in {None, "none"}:
        print(f"fallback reason: {metadata.get('failure_reason')}")
    if report.empty:
        return
    adjusted = report.copy()
    adjusted["abs_adjustment"] = pd.to_numeric(adjusted["calibration_adjustment"], errors="coerce").abs()
    columns = [
        "original_expected_daily_return",
        "calibrated_expected_daily_return",
        "calibration_adjustment",
        "used_calibrated_forecast",
    ]
    print("\ntickers most adjusted:")
    print(adjusted.sort_values("abs_adjustment", ascending=False).head(top_n)[columns].to_string())
