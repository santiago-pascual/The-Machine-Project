from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IC_HISTORY_FILE = "ic_history.csv"
DEFAULT_HORIZONS = (5, 10, 20)


FEATURE_ALIASES = {
    "target_confidence_quant": "quant_confidence",
    "entropy": "shannon_entropy",
    "expected_daily_return": "expected_return",
    "total_return": "target_gap_pct",
    "ema_timing_score": "ema_timing_score",
    "downside_ratio": "downside_risk",
}


def _clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _build_available_feature_frame(
    diagnostics_df: pd.DataFrame,
    timing_df: pd.DataFrame | None = None,
    regime_score: float | None = None,
    regime_confidence: float | None = None,
) -> pd.DataFrame:
    features = diagnostics_df.copy()
    if timing_df is not None and not timing_df.empty:
        timing_numeric = timing_df.copy()
        for column in timing_numeric.columns:
            if column not in features.columns:
                features[column] = timing_numeric[column]

    for source, alias in FEATURE_ALIASES.items():
        if source in features.columns and alias not in features.columns:
            features[alias] = features[source]

    if regime_score is not None:
        features["regime_score"] = float(regime_score)
    if regime_confidence is not None:
        features["regime_confidence"] = float(regime_confidence)

    numeric = _clean_numeric_frame(features)
    return numeric.dropna(axis=1, how="all")


def _classify_ic(value: float) -> str:
    absolute = abs(float(value)) if np.isfinite(value) else 0.0
    if absolute > 0.10:
        return "Excellent"
    if absolute > 0.05:
        return "Useful"
    if absolute > 0.02:
        return "Weak"
    return "Noise"


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 5:
        return np.nan
    if data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return np.nan
    if method == "spearman":
        left = data.iloc[:, 0].rank(method="average")
        right = data.iloc[:, 1].rank(method="average")
        return float(left.corr(right, method="pearson"))
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method="pearson"))


def compute_information_coefficient_snapshot(
    *,
    prices_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    timing_df: pd.DataFrame | None = None,
    regime_score: float | None = None,
    regime_confidence: float | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Cross-sectional IC snapshot.
    Uses current feature values and realized forward returns only if the supplied
    prices_df already contains enough future rows after the feature date. In a
    normal production run this usually has no future rows, so use the stored
    history functions for real ex-post IC over time.
    """
    if prices_df.empty or diagnostics_df.empty:
        return pd.DataFrame()

    features = _build_available_feature_frame(
        diagnostics_df=diagnostics_df.reindex(prices_df.columns),
        timing_df=timing_df.reindex(prices_df.columns) if timing_df is not None and not timing_df.empty else None,
        regime_score=regime_score,
        regime_confidence=regime_confidence,
    )
    latest_prices = prices_df.ffill().iloc[-1].replace(0, np.nan)
    rows: list[dict[str, object]] = []

    for feature in features.columns:
        row: dict[str, object] = {"feature": feature}
        sample_sizes = []
        for horizon in horizons:
            # Snapshot mode is intentionally conservative: no future rows are
            # assumed available after the current production date.
            if len(prices_df) <= horizon:
                future_return = pd.Series(index=prices_df.columns, dtype=float)
            else:
                future_return = pd.Series(index=prices_df.columns, dtype=float)
            ic = _safe_corr(features[feature], future_return.reindex(features.index), method="spearman")
            row[f"IC_{horizon}D"] = ic
            row[f"Pearson_{horizon}D"] = _safe_corr(features[feature], future_return.reindex(features.index), method="pearson")
            sample_sizes.append(int(pd.concat([features[feature], future_return.reindex(features.index)], axis=1).dropna().shape[0]))
        ic_values = [row[f"IC_{horizon}D"] for horizon in horizons]
        row["Average_IC"] = float(np.nanmean(ic_values)) if any(np.isfinite(v) for v in ic_values) else np.nan
        row["sample_size"] = int(max(sample_sizes) if sample_sizes else 0)
        row["classification"] = _classify_ic(row["Average_IC"])
        row["predictive_direction"] = "positive" if row["Average_IC"] > 0 else "negative" if row["Average_IC"] < 0 else "neutral"
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Average_IC", ascending=False, na_position="last")


def compute_information_coefficient_from_history(
    forecast_history_path: str | Path = "forecast_history_evaluated.csv",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    path = Path(forecast_history_path)
    if not path.exists():
        return pd.DataFrame()

    history = pd.read_csv(path)
    if history.empty:
        return pd.DataFrame()

    candidate_features = [
        col
        for col in history.columns
        if col
        not in {
            "date",
            "ticker",
            "current_price",
            "target_price",
            "selected",
            "regime",
            "final_weight_percent",
            *[f"realized_return_{h}d" for h in horizons],
        }
    ]

    rows: list[dict[str, object]] = []
    for feature in candidate_features:
        feature_values = pd.to_numeric(history[feature], errors="coerce")
        if feature_values.notna().sum() < 10 or feature_values.nunique(dropna=True) < 2:
            continue
        row: dict[str, object] = {"feature": feature}
        sample_sizes = []
        for horizon in horizons:
            realized_col = f"realized_return_{horizon}d"
            if realized_col not in history.columns:
                row[f"IC_{horizon}D"] = np.nan
                row[f"Pearson_{horizon}D"] = np.nan
                sample_sizes.append(0)
                continue
            realized = pd.to_numeric(history[realized_col], errors="coerce")
            data = pd.concat([feature_values, realized], axis=1).dropna()
            sample_sizes.append(len(data))
            row[f"IC_{horizon}D"] = _safe_corr(feature_values, realized, method="spearman")
            row[f"Pearson_{horizon}D"] = _safe_corr(feature_values, realized, method="pearson")
        ic_values = [row[f"IC_{horizon}D"] for horizon in horizons]
        row["Average_IC"] = float(np.nanmean(ic_values)) if any(np.isfinite(v) for v in ic_values) else np.nan
        row["sample_size"] = int(max(sample_sizes) if sample_sizes else 0)
        row["classification"] = _classify_ic(row["Average_IC"])
        row["predictive_direction"] = "positive" if row["Average_IC"] > 0 else "negative" if row["Average_IC"] < 0 else "neutral"
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Average_IC", ascending=False, na_position="last")


def append_ic_history(
    ic_report: pd.DataFrame,
    history_path: str | Path = IC_HISTORY_FILE,
) -> None:
    if ic_report.empty:
        return
    path = Path(history_path)
    output = ic_report.copy()
    output.insert(0, "date", pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))
    columns = ["date", "feature", "IC_5D", "IC_10D", "IC_20D", "Average_IC"]
    existing_cols = [col for col in columns if col in output.columns]
    output[existing_cols].to_csv(path, mode="a", header=not path.exists(), index=False)


def print_information_coefficient_report(
    *,
    prices_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    timing_df: pd.DataFrame | None = None,
    regime_score: float | None = None,
    regime_confidence: float | None = None,
    forecast_history_path: str | Path = "forecast_history_evaluated.csv",
    ic_history_path: str | Path = IC_HISTORY_FILE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    history_report = compute_information_coefficient_from_history(forecast_history_path, horizons=horizons)
    if history_report.empty:
        report = compute_information_coefficient_snapshot(
            prices_df=prices_df,
            diagnostics_df=diagnostics_df,
            timing_df=timing_df,
            regime_score=regime_score,
            regime_confidence=regime_confidence,
            horizons=horizons,
        )
        source = "snapshot_no_forward_labels"
    else:
        report = history_report
        source = "forecast_history_evaluated"
        append_ic_history(report, history_path=ic_history_path)

    print("\n===== INFORMATION COEFFICIENT REPORT =====")
    print(f"source: {source}")
    if report.empty:
        print("Not enough samples to compute IC yet.")
        return report

    display_cols = [
        "feature",
        "IC_5D",
        "IC_10D",
        "IC_20D",
        "Average_IC",
        "sample_size",
        "classification",
        "predictive_direction",
    ]
    existing = [col for col in display_cols if col in report.columns]
    print(report[existing].to_string(index=False))

    print("\n===== FEATURE RANKING =====")
    valid = report.dropna(subset=["Average_IC"])
    print("\nTop 10 predictive features:")
    print(valid.sort_values("Average_IC", ascending=False).head(10)[existing].to_string(index=False) if not valid.empty else "None")
    print("\nTop 10 anti-predictive features:")
    print(valid.sort_values("Average_IC", ascending=True).head(10)[existing].to_string(index=False) if not valid.empty else "None")

    if source == "snapshot_no_forward_labels":
        print("\nIC warning: no forward realized labels available yet; report will become meaningful after forecast_history_evaluated.csv has future returns.")
    return report
