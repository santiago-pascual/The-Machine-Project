from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_REGIME = "regime_conditional_meta_filter.csv"
OUTPUT_COMPARISON = "regime_conditional_meta_filter_comparison.csv"


@dataclass
class RegimeConditionalMetaConfig:
    trades_path: str = "final_candidate_backtest_trades.csv"
    triple_barrier_path: str = "historical_triple_barrier_labels.csv"
    failure_attribution_path: str = "meta_filter_failure_attribution.csv"
    threshold: float = 0.65
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


def _attach_labels(trades: pd.DataFrame, config: RegimeConditionalMetaConfig) -> pd.DataFrame:
    labels = _read_csv(config.triple_barrier_path)
    if labels.empty:
        trades["label"] = np.nan
        return trades
    labels = labels[
        labels.get("horizon", pd.Series(dtype=float)).eq(config.horizon)
        & labels["model_mode"].astype(str).eq("regime_gated_full_quant")
    ][["date", "ticker", "model_mode", "label"]].drop_duplicates(["date", "ticker", "model_mode"])
    return trades.merge(labels, on=["date", "ticker", "model_mode"], how="left")


def _selected_candidate_trades(config: RegimeConditionalMetaConfig) -> pd.DataFrame:
    trades = _read_csv(config.trades_path)
    if trades.empty:
        return pd.DataFrame()
    trades = trades[trades["candidate"].astype(str).eq("candidate_meta_filtered")].copy()
    trades = _attach_labels(trades, config)
    ret_col = f"realized_return_{config.horizon}d"
    trades["future_return"] = _safe_numeric(trades.get(ret_col, pd.Series(np.nan, index=trades.index)), np.nan)
    trades["original_weight"] = _safe_numeric(trades.get("original_weight", pd.Series(0.0, index=trades.index)), 0.0)
    trades["meta_filter_pass"] = trades["meta_filter_pass"].astype(bool)
    trades["regime"] = trades.get("regime", pd.Series("unknown", index=trades.index)).astype(str).fillna("unknown")
    trades["daily_volatility"] = _safe_numeric(trades.get("daily_volatility", pd.Series(0.0, index=trades.index)), 0.0)
    return trades


def _risk_metrics(returns: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"realized_return": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan}
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    return {
        "realized_return": float(equity.iloc[-1] - 1.0),
        "Sharpe": float(mean_ret / std_ret * np.sqrt(252 / 20)) if std_ret > 0 else 0.0,
        "Sortino": float(mean_ret / downside_std * np.sqrt(252 / 20)) if downside_std > 0 else 0.0,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "hit_rate": float(returns.gt(0).mean()),
    }


def _regime_quality(trades: pd.DataFrame, config: RegimeConditionalMetaConfig) -> pd.DataFrame:
    rows = []
    for regime, group in trades.groupby("regime"):
        kept = group[group["meta_filter_pass"]]
        rejected = group[~group["meta_filter_pass"]]
        kept_return = float(kept["future_return"].mean(skipna=True)) if not kept.empty else np.nan
        rejected_return = float(rejected["future_return"].mean(skipna=True)) if not rejected.empty else np.nan
        labels_kept = _safe_numeric(kept.get("label", pd.Series(np.nan, index=kept.index)), np.nan)
        weighted_lost = float((rejected["future_return"] * rejected["original_weight"]).sum(skipna=True)) if not rejected.empty else 0.0
        filter_helps = bool(np.isfinite(kept_return) and np.isfinite(rejected_return) and kept_return > rejected_return and weighted_lost <= 0)
        rows.append(
            {
                "regime": regime,
                "kept_trades": len(kept),
                "rejected_trades": len(rejected),
                "avg_return_kept": kept_return,
                "avg_return_rejected": rejected_return,
                "TP_rate_kept": float(labels_kept.eq(1).mean()) if labels_kept.notna().any() else np.nan,
                "SL_rate_kept": float(labels_kept.eq(-1).mean()) if labels_kept.notna().any() else np.nan,
                "hit_rate_kept": float(kept["future_return"].gt(0).mean()) if not kept.empty else np.nan,
                "cash_drag": weighted_lost,
                "filter_helps": filter_helps,
                "decision": "apply_filter" if filter_helps else "fallback_unfiltered",
            }
        )
    return pd.DataFrame(rows).sort_values("regime")


def _daily_returns(trades: pd.DataFrame, candidate: str, mode: str, apply_regimes: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    output_trades = []
    previous = pd.Series(dtype=float)
    apply_regimes = apply_regimes or set()
    for date, group in trades.groupby("date", sort=True):
        group = group.copy()
        if mode == "unfiltered":
            group["effective_weight_conditional"] = group["original_weight"]
            group["conditional_pass"] = True
        elif mode == "global":
            group["effective_weight_conditional"] = np.where(group["meta_filter_pass"], group["original_weight"], 0.0)
            group["conditional_pass"] = group["meta_filter_pass"]
        elif mode == "regime_conditional":
            use_filter = group["regime"].isin(apply_regimes)
            group["conditional_pass"] = np.where(use_filter, group["meta_filter_pass"], True)
            group["effective_weight_conditional"] = np.where(group["conditional_pass"], group["original_weight"], 0.0)
        else:
            raise ValueError(mode)
        group["candidate_eval"] = candidate
        output_trades.append(group)
        weights = group.set_index("ticker")["effective_weight_conditional"].astype(float)
        tickers = sorted(set(previous.index.astype(str)) | set(weights.index.astype(str)))
        turnover = sum(abs(float(weights.get(t, 0.0)) - float(previous.get(t, 0.0))) for t in tickers) / 2.0
        row = {
            "date": date,
            "candidate": candidate,
            "cash_weight": max(0.0, 1.0 - float(weights.sum())),
            "selected_count": int((weights > 0).sum()),
            "turnover": turnover,
            "trades_kept": int(group["conditional_pass"].sum()),
            "sample_reduction": float(1.0 - group["conditional_pass"].mean()) if len(group) else np.nan,
        }
        for horizon in [1, 5, 10, 20, 30]:
            col = f"realized_return_{horizon}d"
            row[f"realized_portfolio_return_{horizon}d"] = float((group["effective_weight_conditional"] * _safe_numeric(group.get(col, pd.Series(np.nan, index=group.index)), np.nan)).sum(skipna=True))
        rows.append(row)
        previous = weights
    return pd.DataFrame(rows), pd.concat(output_trades, ignore_index=True) if output_trades else pd.DataFrame()


def _comparison_rows(trades: pd.DataFrame, regime_report: pd.DataFrame, config: RegimeConditionalMetaConfig) -> pd.DataFrame:
    apply_regimes = set(regime_report.loc[regime_report["decision"].eq("apply_filter"), "regime"].astype(str))
    candidates = [
        ("regime_gated_full_quant", "unfiltered", set()),
        ("global_meta_filter", "global", set()),
        ("regime_conditional_meta_filter", "regime_conditional", apply_regimes),
        ("neutral_only_meta_filter", "regime_conditional", {"neutral"}),
        ("risk_off_only_meta_filter", "regime_conditional", {"risk_off"}),
    ]
    rows = []
    trade_outputs = []
    for name, mode, regimes in candidates:
        daily, evaluated_trades = _daily_returns(trades, name, mode, regimes)
        trade_outputs.append(evaluated_trades)
        metrics = _risk_metrics(daily[f"realized_portfolio_return_{config.horizon}d"])
        labels = _safe_numeric(evaluated_trades.loc[evaluated_trades["conditional_pass"], "label"], np.nan) if not evaluated_trades.empty else pd.Series(dtype=float)
        rows.append(
            {
                "candidate": name,
                **metrics,
                "TP_rate": float(labels.eq(1).mean()) if labels.notna().any() else np.nan,
                "SL_rate": float(labels.eq(-1).mean()) if labels.notna().any() else np.nan,
                "TP_minus_SL": float(labels.eq(1).mean() - labels.eq(-1).mean()) if labels.notna().any() else np.nan,
                "average_cash": float(daily["cash_weight"].mean()) if not daily.empty else np.nan,
                "average_selected_count": float(daily["selected_count"].mean()) if not daily.empty else np.nan,
                "turnover": float(daily["turnover"].mean()) if not daily.empty else np.nan,
                "trades_kept": int(evaluated_trades["conditional_pass"].sum()) if not evaluated_trades.empty else 0,
                "sample_reduction": float(1.0 - evaluated_trades["conditional_pass"].mean()) if not evaluated_trades.empty else np.nan,
                "filter_regimes": ", ".join(sorted(regimes)) if regimes else "none",
            }
        )
    return pd.DataFrame(rows).sort_values(["Sharpe", "realized_return"], ascending=False)


def _governance(comparison: pd.DataFrame, regime_report: pd.DataFrame) -> pd.DataFrame:
    base = comparison[comparison["candidate"].eq("regime_gated_full_quant")]
    conditional = comparison[comparison["candidate"].eq("regime_conditional_meta_filter")]
    classification = "not useful"
    reason = "missing_comparison"
    if not base.empty and not conditional.empty:
        b = base.iloc[0]
        c = conditional.iloc[0]
        if c["Sharpe"] > b["Sharpe"] and c["realized_return"] >= b["realized_return"]:
            classification = "useful for research"
            reason = "beats_unfiltered_on_sharpe_and_return"
        elif c["Sharpe"] > b["Sharpe"]:
            classification = "regime-specific usefulness"
            reason = "improves_sharpe_but_not_total_return"
        else:
            classification = "not useful"
            reason = "does_not_beat_unfiltered_regime_gated"
    harmful = regime_report[regime_report["decision"].eq("fallback_unfiltered")]["regime"].astype(str).tolist() if not regime_report.empty else []
    helpful = regime_report[regime_report["decision"].eq("apply_filter")]["regime"].astype(str).tolist() if not regime_report.empty else []
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "reason": reason,
                "helpful_regimes": ", ".join(helpful) if helpful else "none",
                "harmful_regimes": ", ".join(harmful) if harmful else "none",
                "production_behavior_changed": False,
                "paper_activation": False,
            }
        ]
    )


def run_regime_conditional_meta_filter(config: RegimeConditionalMetaConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or RegimeConditionalMetaConfig()
    trades = _selected_candidate_trades(config)
    if trades.empty:
        empty = pd.DataFrame()
        empty.to_csv(OUTPUT_REGIME, index=False)
        empty.to_csv(OUTPUT_COMPARISON, index=False)
        return {"regime": empty, "comparison": empty, "governance": empty}
    regime_report = _regime_quality(trades, config)
    comparison = _comparison_rows(trades, regime_report, config)
    governance = _governance(comparison, regime_report)
    regime_report.to_csv(OUTPUT_REGIME, index=False)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    _print_report(regime_report, comparison, governance)
    return {"regime": regime_report, "comparison": comparison, "governance": governance}


def _print_report(regime_report: pd.DataFrame, comparison: pd.DataFrame, governance: pd.DataFrame) -> None:
    print("\n===== REGIME CONDITIONAL META FILTER =====")
    print(governance.to_string(index=False) if not governance.empty else "No governance.")

    print("\n===== META FILTER BY REGIME =====")
    print(regime_report.to_string(index=False) if not regime_report.empty else "No regime report.")

    print("\n===== CONDITIONAL FILTER COMPARISON =====")
    print(comparison.to_string(index=False) if not comparison.empty else "No comparison.")

    print("\n===== REGIME CONDITIONAL GOVERNANCE =====")
    if not governance.empty:
        print(f"classification: {governance.iloc[0]['classification']}")
        print(f"reason: {governance.iloc[0]['reason']}")
    print("production behavior changed: False")


if __name__ == "__main__":
    run_regime_conditional_meta_filter()
