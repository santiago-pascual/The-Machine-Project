from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_FILE = "meta_filter_failure_attribution.csv"


@dataclass
class MetaFilterFailureConfig:
    results_path: str = "final_candidate_backtest_results.csv"
    trades_path: str = "final_candidate_backtest_trades.csv"
    daily_returns_path: str = "final_candidate_backtest_daily_returns.csv"
    walk_forward_results_path: str = "meta_model_walk_forward_results.csv"
    meta_label_dataset_path: str = "meta_label_dataset.csv"
    triple_barrier_path: str = "historical_triple_barrier_labels.csv"
    realized_returns_path: str = "historical_realized_returns.csv"
    horizon: int = 20


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    return df


def _safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _risk_metrics(returns: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"average_realized_return": np.nan, "median_realized_return": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "hit_rate": np.nan}
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    return {
        "average_realized_return": mean_ret,
        "median_realized_return": float(returns.median()),
        "Sharpe": float(mean_ret / std_ret * np.sqrt(252 / 20)) if std_ret > 0 else 0.0,
        "Sortino": float(mean_ret / downside_std * np.sqrt(252 / 20)) if downside_std > 0 else 0.0,
        "hit_rate": float(returns.gt(0).mean()),
    }


def _drawdown_contribution(weighted_returns: pd.Series) -> float:
    returns = pd.to_numeric(weighted_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min()) if len(dd) else np.nan


def _group_quality(group_name: str, trades: pd.DataFrame, config: MetaFilterFailureConfig) -> dict[str, object]:
    ret_col = f"realized_return_{config.horizon}d"
    returns = _safe_numeric(trades.get(ret_col, pd.Series(np.nan, index=trades.index)), np.nan)
    weights = _safe_numeric(trades.get("original_weight", pd.Series(0.0, index=trades.index)), 0.0)
    labels = _safe_numeric(trades.get("label", pd.Series(np.nan, index=trades.index)), np.nan)
    weighted = returns * weights
    metrics = _risk_metrics(returns)
    horizon_cols = [c for c in trades.columns if c.startswith("realized_return_") and c.endswith("d")]
    output = {
        "section": "kept_vs_rejected_trade_quality",
        "group": group_name,
        "sample_size": len(trades),
        **metrics,
        "TP_rate": float(labels.eq(1).mean()) if labels.notna().any() else np.nan,
        "SL_rate": float(labels.eq(-1).mean()) if labels.notna().any() else np.nan,
        "portfolio_return_contribution": float(weighted.sum(skipna=True)),
        "average_weighted_return_contribution": float(weighted.mean(skipna=True)),
        "drawdown_contribution_proxy": _drawdown_contribution(weighted),
        "avg_signal_strength": float(_safe_numeric(trades.get("signal_strength", pd.Series(dtype=float)), np.nan).mean(skipna=True)),
        "avg_meta_probability": float(_safe_numeric(trades.get("meta_probability", pd.Series(dtype=float)), np.nan).mean(skipna=True)),
        "avg_weight": float(weights.mean(skipna=True)),
    }
    for col in horizon_cols:
        output[f"avg_{col}"] = float(_safe_numeric(trades[col], np.nan).mean(skipna=True))
    return output


def _attach_labels(candidate: pd.DataFrame, config: MetaFilterFailureConfig) -> pd.DataFrame:
    labels = _read_csv(config.triple_barrier_path)
    if labels.empty:
        candidate["label"] = np.nan
        return candidate
    labels = labels[
        labels.get("horizon", pd.Series(dtype=float)).eq(config.horizon)
        & labels["model_mode"].astype(str).eq("regime_gated_full_quant")
    ][["date", "ticker", "model_mode", "label", "first_touch_type"]].drop_duplicates(["date", "ticker", "model_mode"])
    return candidate.merge(labels, on=["date", "ticker", "model_mode"], how="left")


def _error_type_rows(candidate: pd.DataFrame, config: MetaFilterFailureConfig) -> pd.DataFrame:
    ret_col = f"realized_return_{config.horizon}d"
    rows = []
    if candidate.empty or ret_col not in candidate.columns:
        return pd.DataFrame()
    frame = candidate.copy()
    frame["future_return"] = _safe_numeric(frame[ret_col], np.nan)
    frame["meta_filter_pass"] = frame["meta_filter_pass"].astype(bool)
    conditions = {
        "bad_trades_correctly_filtered": (~frame["meta_filter_pass"]) & frame["future_return"].le(0),
        "good_trades_incorrectly_filtered": (~frame["meta_filter_pass"]) & frame["future_return"].gt(0),
        "high_volatility_winners_rejected": (~frame["meta_filter_pass"]) & frame["future_return"].gt(0) & (_safe_numeric(frame.get("daily_volatility", pd.Series(0, index=frame.index)), 0.0) > _safe_numeric(frame.get("daily_volatility", pd.Series(0, index=frame.index)), 0.0).median()),
        "low_confidence_winners_rejected": (~frame["meta_filter_pass"]) & frame["future_return"].gt(0) & (_safe_numeric(frame.get("target_confidence", pd.Series(0.5, index=frame.index)), 0.5) < 0.5),
        "kept_winners": frame["meta_filter_pass"] & frame["future_return"].gt(0),
        "kept_losers": frame["meta_filter_pass"] & frame["future_return"].le(0),
    }
    for name, mask in conditions.items():
        subset = frame[mask]
        rows.append(
            {
                "section": "meta_filter_error_types",
                "group": name,
                "sample_size": len(subset),
                "avg_return": float(subset["future_return"].mean(skipna=True)) if not subset.empty else np.nan,
                "weighted_return_lost_or_kept": float((subset["future_return"] * _safe_numeric(subset.get("original_weight", pd.Series(0, index=subset.index)), 0.0)).sum(skipna=True)) if not subset.empty else 0.0,
                "top_tickers": ", ".join(subset["ticker"].astype(str).value_counts().head(5).index.tolist()) if not subset.empty else "",
                "top_regimes": ", ".join(subset.get("regime", pd.Series(dtype=str)).astype(str).value_counts().head(5).index.tolist()) if not subset.empty and "regime" in subset.columns else "",
            }
        )
    return pd.DataFrame(rows)


def _concentration_rows(candidate: pd.DataFrame, config: MetaFilterFailureConfig) -> pd.DataFrame:
    ret_col = f"realized_return_{config.horizon}d"
    frame = candidate.copy()
    frame["future_return"] = _safe_numeric(frame.get(ret_col, pd.Series(np.nan, index=frame.index)), np.nan)
    frame["weighted_return"] = frame["future_return"] * _safe_numeric(frame.get("original_weight", pd.Series(0, index=frame.index)), 0.0)
    rows = []
    rejected = frame[~frame["meta_filter_pass"].astype(bool)].copy()
    for key, label in [("ticker", "rejected_by_ticker"), ("regime", "rejected_by_regime")]:
        if key not in rejected.columns:
            continue
        grouped = rejected.groupby(key, dropna=False)
        for value, group in grouped:
            rows.append(
                {
                    "section": label,
                    "group": str(value),
                    "sample_size": len(group),
                    "average_realized_return": float(group["future_return"].mean(skipna=True)),
                    "portfolio_return_contribution_lost": float(group["weighted_return"].sum(skipna=True)),
                }
            )
    frame["year"] = frame["date"].dt.year
    rejected = frame[~frame["meta_filter_pass"].astype(bool)].copy()
    for year, group in rejected.groupby("year"):
        rows.append(
            {
                "section": "rejected_by_year",
                "group": str(year),
                "sample_size": len(group),
                "average_realized_return": float(group["future_return"].mean(skipna=True)),
                "portfolio_return_contribution_lost": float(group["weighted_return"].sum(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def _cash_drag_rows(candidate: pd.DataFrame, daily: pd.DataFrame, results: pd.DataFrame, config: MetaFilterFailureConfig) -> pd.DataFrame:
    ret_col = f"realized_return_{config.horizon}d"
    rejected = candidate[~candidate["meta_filter_pass"].astype(bool)].copy()
    rejected_returns = _safe_numeric(rejected.get(ret_col, pd.Series(np.nan, index=rejected.index)), np.nan)
    rejected_weights = _safe_numeric(rejected.get("original_weight", pd.Series(0, index=rejected.index)), 0.0)
    lost_return = float((rejected_returns * rejected_weights).sum(skipna=True))
    baseline = results[results["candidate"].eq("regime_gated_full_quant")]
    filtered = results[results["candidate"].eq("candidate_meta_filtered")]
    vol_reduction = np.nan
    sharpe_change = np.nan
    if not baseline.empty and not filtered.empty:
        vol_reduction = float(baseline.iloc[0]["annualized_volatility"] - filtered.iloc[0]["annualized_volatility"])
        sharpe_change = float(filtered.iloc[0]["Sharpe"] - baseline.iloc[0]["Sharpe"])
    return pd.DataFrame(
        [
            {
                "section": "cash_drag_analysis",
                "group": "rejected_to_cash",
                "sample_size": len(rejected),
                "return_lost_to_cash": lost_return,
                "avg_rejected_return": float(rejected_returns.mean(skipna=True)),
                "avg_rejected_weight": float(rejected_weights.mean(skipna=True)),
                "cash_increase": float(results.loc[results["candidate"].eq("candidate_meta_filtered"), "average_cash"].iloc[0] - results.loc[results["candidate"].eq("regime_gated_full_quant"), "average_cash"].iloc[0]) if {"candidate", "average_cash"}.issubset(results.columns) and not baseline.empty and not filtered.empty else np.nan,
                "volatility_reduction": vol_reduction,
                "sharpe_change": sharpe_change,
                "volatility_reduction_compensated_lost_return": bool(sharpe_change > 0) if np.isfinite(sharpe_change) else False,
                "filter_too_conservative": bool(lost_return > 0 and (not np.isfinite(sharpe_change) or sharpe_change < 0)),
            }
        ]
    )


def _verdict(attribution: pd.DataFrame) -> pd.DataFrame:
    cash = attribution[(attribution["section"].eq("cash_drag_analysis")) & (attribution["group"].eq("rejected_to_cash"))]
    errors = attribution[attribution["section"].eq("meta_filter_error_types")]
    verdicts = []
    if not cash.empty:
        row = cash.iloc[0]
        if bool(row.get("filter_too_conservative", False)):
            verdicts.append("filter too conservative")
        if float(row.get("return_lost_to_cash", 0.0) or 0.0) > 0:
            verdicts.append("cash drag from rejected winners")
    good_rejected = errors[errors["group"].eq("good_trades_incorrectly_filtered")]
    bad_rejected = errors[errors["group"].eq("bad_trades_correctly_filtered")]
    if not good_rejected.empty and not bad_rejected.empty:
        if int(good_rejected.iloc[0]["sample_size"]) > int(bad_rejected.iloc[0]["sample_size"]):
            verdicts.append("threshold too high")
    if not verdicts:
        verdicts.append("regime-specific usefulness")
    return pd.DataFrame(
        [
            {
                "section": "verdict",
                "group": "final",
                "sample_size": np.nan,
                "verdict": " | ".join(dict.fromkeys(verdicts)),
                "recommendation": "do_not_promote; test lower threshold or regime-gated meta-filter in research only",
            }
        ]
    )


def run_meta_filter_failure_attribution(config: MetaFilterFailureConfig | None = None) -> pd.DataFrame:
    config = config or MetaFilterFailureConfig()
    trades = _read_csv(config.trades_path)
    daily = _read_csv(config.daily_returns_path)
    results = _read_csv(config.results_path)
    if trades.empty:
        output = pd.DataFrame()
        output.to_csv(OUTPUT_FILE, index=False)
        return output
    candidate = trades[trades["candidate"].astype(str).eq("candidate_meta_filtered")].copy()
    candidate = _attach_labels(candidate, config)
    kept = candidate[candidate["meta_filter_pass"].astype(bool)].copy()
    rejected = candidate[~candidate["meta_filter_pass"].astype(bool)].copy()

    rows = [
        _group_quality("kept_by_meta_filter", kept, config),
        _group_quality("rejected_by_meta_filter", rejected, config),
    ]
    parts = [pd.DataFrame(rows), _error_type_rows(candidate, config), _cash_drag_rows(candidate, daily, results, config), _concentration_rows(candidate, config)]
    attribution = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    attribution = pd.concat([attribution, _verdict(attribution)], ignore_index=True)
    attribution.to_csv(OUTPUT_FILE, index=False)
    _print_report(attribution)
    return attribution


def _print_report(attribution: pd.DataFrame) -> None:
    print("\n===== META FILTER FAILURE ATTRIBUTION =====")
    if attribution.empty:
        print("No attribution available.")
        return
    summary = attribution[attribution["section"].eq("verdict")]
    if not summary.empty:
        print(summary[["verdict", "recommendation"]].to_string(index=False))

    print("\n===== KEPT VS REJECTED TRADE QUALITY =====")
    quality = attribution[attribution["section"].eq("kept_vs_rejected_trade_quality")]
    show_cols = ["group", "sample_size", "average_realized_return", "median_realized_return", "Sharpe", "Sortino", "TP_rate", "SL_rate", "hit_rate", "portfolio_return_contribution"]
    print(quality[[c for c in show_cols if c in quality.columns]].to_string(index=False) if not quality.empty else "No quality rows.")

    print("\n===== CASH DRAG ANALYSIS =====")
    cash = attribution[attribution["section"].eq("cash_drag_analysis")]
    cash_cols = [
        "group",
        "sample_size",
        "return_lost_to_cash",
        "avg_rejected_return",
        "avg_rejected_weight",
        "cash_increase",
        "volatility_reduction",
        "sharpe_change",
        "volatility_reduction_compensated_lost_return",
        "filter_too_conservative",
    ]
    print(cash[[c for c in cash_cols if c in cash.columns]].to_string(index=False) if not cash.empty else "No cash drag rows.")

    print("\n===== META FILTER ERROR TYPES =====")
    errors = attribution[attribution["section"].eq("meta_filter_error_types")]
    error_cols = ["group", "sample_size", "avg_return", "weighted_return_lost_or_kept", "top_tickers", "top_regimes"]
    print(errors[[c for c in error_cols if c in errors.columns]].to_string(index=False) if not errors.empty else "No error type rows.")


if __name__ == "__main__":
    run_meta_filter_failure_attribution()
