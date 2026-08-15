from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_FILE = "triple_barrier_feature_validation.csv"
EXCLUDE_COLUMNS = {
    "date",
    "ticker",
    "horizon",
    "label",
    "selected",
    "first_touch_type",
    "first_touch_date",
    "vertical_barrier_date",
    "current_price",
    "take_profit_price",
    "stop_loss_price",
    "realized_return_at_barrier",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "time_to_first_touch",
}


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 10 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(data.iloc[:, 0].rank(method="average").corr(data.iloc[:, 1].rank(method="average")))
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method="pearson"))


def _classification(spearman: float, tp_minus_sl: float) -> str:
    score = max(abs(spearman) if np.isfinite(spearman) else 0.0, abs(tp_minus_sl) if np.isfinite(tp_minus_sl) else 0.0)
    if score > 0.10:
        return "strong"
    if score > 0.05:
        return "useful"
    if score > 0.02:
        return "weak"
    return "noise"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= 20 and converted.nunique(dropna=True) >= 2:
            numeric_cols.append(col)
    return numeric_cols


def _analyze_subset(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()

    for horizon, horizon_df in df.groupby("horizon"):
        labels = pd.to_numeric(horizon_df["label"], errors="coerce")
        realized = pd.to_numeric(horizon_df["realized_return_at_barrier"], errors="coerce")
        for feature in _feature_columns(horizon_df):
            values = pd.to_numeric(horizon_df[feature], errors="coerce")
            valid = pd.concat([values, labels, realized, horizon_df["first_touch_type"]], axis=1).dropna()
            if len(valid) < 20:
                continue

            feature_values = valid.iloc[:, 0]
            label_values = valid.iloc[:, 1]
            realized_values = valid.iloc[:, 2]
            touch_type = valid.iloc[:, 3]

            spearman = _safe_corr(feature_values, label_values, method="spearman")
            pearson = _safe_corr(feature_values, label_values, method="pearson")

            avg_tp = float(feature_values[touch_type == "take_profit"].mean()) if (touch_type == "take_profit").any() else np.nan
            avg_sl = float(feature_values[touch_type == "stop_loss"].mean()) if (touch_type == "stop_loss").any() else np.nan
            avg_timeout = (
                float(feature_values[touch_type == "vertical_timeout"].mean()) if (touch_type == "vertical_timeout").any() else np.nan
            )

            try:
                quintile = pd.qcut(feature_values, q=5, labels=False, duplicates="drop")
                quintile_df = pd.DataFrame(
                    {
                        "quintile": quintile,
                        "touch_type": touch_type,
                        "realized": realized_values,
                    }
                ).dropna()
                top_q = int(quintile_df["quintile"].max()) if not quintile_df.empty else -1
                top = quintile_df[quintile_df["quintile"] == top_q]
                tp_rate_top = float((top["touch_type"] == "take_profit").mean()) if not top.empty else np.nan
                sl_rate_top = float((top["touch_type"] == "stop_loss").mean()) if not top.empty else np.nan
                timeout_rate_top = float((top["touch_type"] == "vertical_timeout").mean()) if not top.empty else np.nan
                avg_return_top = float(top["realized"].mean()) if not top.empty else np.nan
            except ValueError:
                tp_rate_top = np.nan
                sl_rate_top = np.nan
                timeout_rate_top = np.nan
                avg_return_top = np.nan

            tp_minus_sl = tp_rate_top - sl_rate_top if np.isfinite(tp_rate_top) and np.isfinite(sl_rate_top) else np.nan
            rows.append(
                {
                    "subset": subset_name,
                    "feature": feature,
                    "horizon": int(horizon),
                    "sample_size": len(valid),
                    "spearman_label_corr": spearman,
                    "pearson_label_corr": pearson,
                    "avg_feature_TP": avg_tp,
                    "avg_feature_SL": avg_sl,
                    "avg_feature_timeout": avg_timeout,
                    "TP_rate_top_quintile": tp_rate_top,
                    "SL_rate_top_quintile": sl_rate_top,
                    "timeout_rate_top_quintile": timeout_rate_top,
                    "TP_minus_SL_top_quintile": tp_minus_sl,
                    "avg_return_top_quintile": avg_return_top,
                    "classification": _classification(spearman, tp_minus_sl),
                }
            )

    return pd.DataFrame(rows)


def load_and_merge_triple_barrier_features(
    labels_path: str | Path = "triple_barrier_labels.csv",
    predictions_path: str | Path = "walk_forward_predictions.csv",
) -> pd.DataFrame:
    labels_file = Path(labels_path)
    predictions_file = Path(predictions_path)
    if not labels_file.exists() or not predictions_file.exists():
        return pd.DataFrame()

    labels = pd.read_csv(labels_file)
    predictions = pd.read_csv(predictions_file)
    if labels.empty or predictions.empty:
        return pd.DataFrame()

    for df in (labels, predictions):
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["ticker"] = df["ticker"].astype(str)

    prediction_cols = [
        col for col in predictions.columns if col not in {"realized_return_5d", "realized_return_10d", "realized_return_20d"}
    ]
    merged = labels.merge(
        predictions[prediction_cols],
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_prediction"),
    )
    if "selected_prediction" in merged.columns and "selected" in merged.columns:
        merged["selected"] = merged["selected"].fillna(merged["selected_prediction"])
    return merged.replace([np.inf, -np.inf], np.nan)


def run_triple_barrier_feature_validation(
    labels_path: str | Path = "triple_barrier_labels.csv",
    predictions_path: str | Path = "walk_forward_predictions.csv",
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    merged = load_and_merge_triple_barrier_features(labels_path, predictions_path)
    if merged.empty:
        result = pd.DataFrame()
        result.to_csv(output_path, index=False)
        print("\n===== TRIPLE BARRIER FEATURE VALIDATION =====")
        print("No merged triple-barrier feature data available.")
        return result

    all_obs = _analyze_subset(merged, "universe")
    selected = merged[merged.get("selected", False).astype(bool)] if "selected" in merged.columns else merged.iloc[0:0]
    selected_obs = _analyze_subset(selected, "selected_only")
    result = pd.concat([all_obs, selected_obs], ignore_index=True)
    result = result.sort_values(["classification", "spearman_label_corr"], ascending=[True, False])
    result.to_csv(output_path, index=False)

    print("\n===== TRIPLE BARRIER FEATURE VALIDATION =====")
    if result.empty:
        print("Not enough samples for feature validation.")
        return result

    display_cols = [
        "subset",
        "feature",
        "horizon",
        "sample_size",
        "spearman_label_corr",
        "pearson_label_corr",
        "TP_rate_top_quintile",
        "SL_rate_top_quintile",
        "TP_minus_SL_top_quintile",
        "avg_return_top_quintile",
        "classification",
    ]
    print(result[display_cols].to_string(index=False))

    print("\n===== TOP TRIPLE-BARRIER PREDICTIVE FEATURES =====")
    predictive = result.sort_values(["spearman_label_corr", "TP_minus_SL_top_quintile"], ascending=False).head(10)
    print(predictive[display_cols].to_string(index=False))

    print("\n===== WORST TRIPLE-BARRIER FEATURES =====")
    worst = result.sort_values(["spearman_label_corr", "TP_minus_SL_top_quintile"], ascending=True).head(10)
    print(worst[display_cols].to_string(index=False))
    return result
