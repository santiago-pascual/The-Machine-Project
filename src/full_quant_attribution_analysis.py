from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import pandas as pd


OUTPUT_FILE = "full_quant_attribution_analysis.csv"
ROBUSTNESS_FILE = "full_quant_robustness_walk_forward.csv"


def _safe_read(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    data = pd.read_csv(file_path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "ticker" in data.columns:
        data["ticker"] = data["ticker"].astype(str)
    return data.replace([np.inf, -np.inf], np.nan)


def _window_to_safe_name(window: str) -> str:
    return window.replace("-", "").replace("_to_", "__")


def _paths(window: str, mode: str) -> dict[str, str]:
    safe = _window_to_safe_name(window)
    return {
        "predictions": f"full_quant_robustness__{safe}__{mode}_predictions.csv",
        "portfolio": f"full_quant_robustness__{safe}__{mode}_portfolio_returns.csv",
        "labels": f"full_quant_robustness__{safe}__{mode}_triple_barrier_labels.csv",
    }


def _selected_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "selected" not in predictions.columns:
        return predictions.iloc[0:0].copy()
    return predictions[predictions["selected"].astype(bool)].copy()


def _avg_col(data: pd.DataFrame, column: str) -> float:
    if data.empty or column not in data.columns:
        return np.nan
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _label_metrics(labels: pd.DataFrame) -> dict[str, float]:
    selected = labels[labels["selected"].astype(bool)] if not labels.empty and "selected" in labels.columns else labels.iloc[0:0]
    if selected.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "timeout_rate": np.nan, "TP_minus_SL": np.nan}
    tp = float((selected["first_touch_type"] == "take_profit").mean())
    sl = float((selected["first_touch_type"] == "stop_loss").mean())
    timeout = float((selected["first_touch_type"] == "vertical_timeout").mean())
    return {"TP_rate": tp, "SL_rate": sl, "timeout_rate": timeout, "TP_minus_SL": tp - sl}


def _mode_metrics(predictions: pd.DataFrame, portfolio: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    selected = _selected_predictions(predictions)
    label_stats = _label_metrics(labels)
    unique_selected = sorted(selected["ticker"].dropna().astype(str).unique().tolist()) if "ticker" in selected.columns else []
    return {
        "selected_tickers": unique_selected,
        "average_cash": _avg_col(portfolio, "cash_weight"),
        "turnover": _avg_col(portfolio, "turnover"),
        "average_volatility": _avg_col(portfolio, "portfolio_expected_volatility"),
        "average_expected_return": _avg_col(portfolio, "portfolio_expected_return"),
        "average_signal_strength": _avg_col(selected, "signal_strength"),
        "average_target_confidence": _avg_col(selected, "target_confidence"),
        "average_trend_persistence_score": _avg_col(selected, "trend_persistence_score"),
        "average_ema_score": _avg_col(selected, "ema_timing_score"),
        "average_regime_confidence": _avg_col(selected, "regime_confidence"),
        "max_drawdown": np.nan,
        **label_stats,
    }


def _selection_group_performance(
    baseline_predictions: pd.DataFrame,
    full_predictions: pd.DataFrame,
) -> dict[str, float | list[str]]:
    merged_dates = sorted(set(baseline_predictions.get("date", [])) | set(full_predictions.get("date", [])))
    baseline_only_returns = []
    full_only_returns = []
    overlaps = []
    baseline_only_all: set[str] = set()
    full_only_all: set[str] = set()
    for date in merged_dates:
        b_sel = baseline_predictions[(baseline_predictions["date"] == date) & baseline_predictions["selected"].astype(bool)]
        f_sel = full_predictions[(full_predictions["date"] == date) & full_predictions["selected"].astype(bool)]
        b_set = set(b_sel["ticker"].astype(str))
        f_set = set(f_sel["ticker"].astype(str))
        baseline_only = b_set - f_set
        full_only = f_set - b_set
        overlap = b_set & f_set
        baseline_only_all.update(baseline_only)
        full_only_all.update(full_only)
        overlaps.append(len(overlap) / len(b_set | f_set) if (b_set | f_set) else 1.0)
        if baseline_only:
            baseline_only_returns.extend(
                pd.to_numeric(b_sel[b_sel["ticker"].isin(baseline_only)]["realized_return_20d"], errors="coerce").dropna().tolist()
            )
        if full_only:
            full_only_returns.extend(
                pd.to_numeric(f_sel[f_sel["ticker"].isin(full_only)]["realized_return_20d"], errors="coerce").dropna().tolist()
            )
    return {
        "average_overlap": float(np.mean(overlaps)) if overlaps else np.nan,
        "baseline_only_tickers": sorted(baseline_only_all),
        "full_quant_only_tickers": sorted(full_only_all),
        "baseline_only_avg_20d_return": float(np.mean(baseline_only_returns)) if baseline_only_returns else np.nan,
        "full_quant_only_avg_20d_return": float(np.mean(full_only_returns)) if full_only_returns else np.nan,
    }


def _diagnose(row: pd.Series) -> list[str]:
    causes: list[str] = []
    if float(row.get("return_difference", 0.0)) < 0:
        if float(row.get("full_quant_cash", 0.0)) - float(row.get("baseline_cash", 0.0)) > 0.08:
            causes.append("too much cash")
        if float(row.get("full_quant_cash", 0.0)) - float(row.get("baseline_cash", 0.0)) < -0.08:
            causes.append("too little cash")
        if float(row.get("full_quant_TP_minus_SL", 0.0)) < float(row.get("baseline_TP_minus_SL", 0.0)):
            causes.append("selected different tickers with worse TP/SL")
        if float(row.get("full_quant_volatility", 0.0)) > float(row.get("baseline_volatility", 0.0)) * 1.10:
            causes.append("volatility filter failure")
        if float(row.get("average_overlap", 1.0)) < 0.45:
            causes.append("trend persistence too permissive")
        if float(row.get("full_quant_average_expected_return", 0.0)) > float(row.get("baseline_average_expected_return", 0.0)) * 1.5:
            causes.append("quant target too aggressive")
        if float(row.get("full_quant_average_expected_return", 0.0)) < float(row.get("baseline_average_expected_return", 0.0)) * 0.75:
            causes.append("quant target too conservative")
    else:
        if float(row.get("full_quant_cash", 0.0)) > float(row.get("baseline_cash", 0.0)):
            causes.append("higher cash reduced drawdown")
        if float(row.get("full_quant_SL_rate", 0.0)) < float(row.get("baseline_SL_rate", 0.0)):
            causes.append("lower stop-loss rate")
        if float(row.get("full_quant_turnover", 0.0)) < float(row.get("baseline_turnover", 0.0)):
            causes.append("lower turnover")
        if float(row.get("full_quant_TP_minus_SL", 0.0)) > float(row.get("baseline_TP_minus_SL", 0.0)):
            causes.append("better TP/SL spread")
    if float(row.get("full_quant_turnover", 0.0)) > 0.80:
        causes.append("high turnover")
    if not causes:
        causes.append("mixed factor contribution")
    return causes


def _verdict(analysis: pd.DataFrame) -> str:
    if analysis.empty:
        return "keep full quant as research mode"
    improves = int((analysis["return_difference"] > 0).sum())
    worsens = int((analysis["return_difference"] <= 0).sum())
    common_causes = " ".join(",".join(x) for x in analysis["diagnosed_causes"].tolist())
    if worsens >= improves and "trend persistence too permissive" in common_causes:
        return "calibrate trend persistence"
    if worsens >= improves and ("quant target too aggressive" in common_causes or "quant target too conservative" in common_causes):
        return "calibrate quant target"
    if improves >= 3 and worsens <= 1:
        return "add regime gate"
    if worsens >= 3:
        return "discard current full quant variant"
    return "keep full quant as research mode"


def run_full_quant_attribution_analysis(
    robustness_path: str | Path = ROBUSTNESS_FILE,
    output_path: str | Path = OUTPUT_FILE,
) -> pd.DataFrame:
    robustness = _safe_read(robustness_path)
    if robustness.empty:
        raise ValueError("full_quant_robustness_walk_forward.csv is required.")

    rows = []
    for _, robust_row in robustness.iterrows():
        window = str(robust_row["window"])
        b_paths = _paths(window, "baseline")
        f_paths = _paths(window, "full_quant_research")
        b_pred = _safe_read(b_paths["predictions"])
        f_pred = _safe_read(f_paths["predictions"])
        b_port = _safe_read(b_paths["portfolio"])
        f_port = _safe_read(f_paths["portfolio"])
        b_labels = _safe_read(b_paths["labels"])
        f_labels = _safe_read(f_paths["labels"])

        b = _mode_metrics(b_pred, b_port, b_labels)
        f = _mode_metrics(f_pred, f_port, f_labels)
        selection = _selection_group_performance(b_pred, f_pred)
        row = {
            "window": window,
            "baseline_sharpe": robust_row.get("baseline_sharpe", np.nan),
            "full_quant_sharpe": robust_row.get("full_quant_sharpe", np.nan),
            "sharpe_difference": robust_row.get("sharpe_difference", np.nan),
            "baseline_return": robust_row.get("baseline_return", np.nan),
            "full_quant_return": robust_row.get("full_quant_return", np.nan),
            "return_difference": robust_row.get("return_difference", np.nan),
            "baseline_volatility": robust_row.get("baseline_volatility", np.nan),
            "full_quant_volatility": robust_row.get("full_quant_volatility", np.nan),
            "baseline_max_drawdown": robust_row.get("baseline_max_drawdown", np.nan),
            "full_quant_max_drawdown": robust_row.get("full_quant_max_drawdown", np.nan),
            "baseline_cash": b["average_cash"],
            "full_quant_cash": f["average_cash"],
            "baseline_turnover": b["turnover"],
            "full_quant_turnover": f["turnover"],
            "baseline_average_expected_return": b["average_expected_return"],
            "full_quant_average_expected_return": f["average_expected_return"],
            "baseline_average_signal_strength": b["average_signal_strength"],
            "full_quant_average_signal_strength": f["average_signal_strength"],
            "baseline_average_target_confidence": b["average_target_confidence"],
            "full_quant_average_target_confidence": f["average_target_confidence"],
            "baseline_average_ema_score": b["average_ema_score"],
            "full_quant_average_trend_persistence_score": f["average_trend_persistence_score"],
            "baseline_TP_rate": b["TP_rate"],
            "full_quant_TP_rate": f["TP_rate"],
            "baseline_SL_rate": b["SL_rate"],
            "full_quant_SL_rate": f["SL_rate"],
            "baseline_TP_minus_SL": b["TP_minus_SL"],
            "full_quant_TP_minus_SL": f["TP_minus_SL"],
            **selection,
        }
        row["diagnosed_causes"] = ", ".join(_diagnose(pd.Series(row)))
        rows.append(row)

    analysis = pd.DataFrame(rows)
    analysis.to_csv(output_path, index=False)

    helped = analysis[analysis["return_difference"] > 0].copy()
    hurt = analysis[analysis["return_difference"] <= 0].copy()
    display_cols = [
        "window",
        "return_difference",
        "sharpe_difference",
        "baseline_cash",
        "full_quant_cash",
        "baseline_TP_minus_SL",
        "full_quant_TP_minus_SL",
        "average_overlap",
        "diagnosed_causes",
    ]
    print("\n===== FULL QUANT HELPED WHEN =====")
    print(helped[display_cols].to_string(index=False) if not helped.empty else "None")
    print("\n===== FULL QUANT HURT WHEN =====")
    print(hurt[display_cols].to_string(index=False) if not hurt.empty else "None")
    print("\n===== FULL QUANT SELECTION OVERLAP =====")
    overlap_cols = [
        "window",
        "average_overlap",
        "baseline_only_avg_20d_return",
        "full_quant_only_avg_20d_return",
        "baseline_only_tickers",
        "full_quant_only_tickers",
    ]
    print(analysis[overlap_cols].to_string(index=False))
    print("\n===== FULL QUANT ATTRIBUTION VERDICT =====")
    print(_verdict(analysis))
    print(f"\nSaved: {Path(output_path).resolve()}")
    return analysis


if __name__ == "__main__":
    run_full_quant_attribution_analysis()
