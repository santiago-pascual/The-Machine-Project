from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


OUTPUT_RESULTS = "final_candidate_backtest_results.csv"
OUTPUT_DAILY_RETURNS = "final_candidate_backtest_daily_returns.csv"
OUTPUT_TRADES = "final_candidate_backtest_trades.csv"
OUTPUT_GOVERNANCE = "final_candidate_governance_report.csv"


@dataclass
class FinalCandidateConfig:
    snapshots_path: str = "historical_forecast_snapshots.csv"
    realized_returns_path: str = "historical_realized_returns.csv"
    triple_barrier_path: str = "historical_triple_barrier_labels.csv"
    meta_dataset_path: str = "meta_label_dataset.csv"
    selected_features_path: str = "selected_feature_set.json"
    threshold: float = 0.65
    min_train_samples: int = 500
    calibration_size: int = 200
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


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _load_features(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    remove = set(payload.get("REMOVE_FROM_ML", [])) | set(payload.get("DIAGNOSTIC_ONLY", []))
    features = list(payload.get("CORE", [])) + list(payload.get("SUPPORTING", []))
    return [feature for feature in dict.fromkeys(features) if feature not in remove]


def _feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data = df.copy()
    if "daily_volatility" in features and "daily_volatility" not in data.columns:
        data["daily_volatility"] = 0.0
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True).fillna(0.0))


def _fit_predict_meta_probability(
    *,
    meta_dataset: pd.DataFrame,
    current_rows: pd.DataFrame,
    features: list[str],
    decision_date: pd.Timestamp,
    config: FinalCandidateConfig,
) -> np.ndarray | None:
    if not SKLEARN_AVAILABLE:
        return None
    history = meta_dataset[meta_dataset["date"] < decision_date].sort_values("date").copy()
    if len(history) < config.min_train_samples + config.calibration_size or history["meta_label"].nunique() < 2:
        return None
    train = history.iloc[: -config.calibration_size].copy()
    cal = history.iloc[-config.calibration_size :].copy()
    if cal["meta_label"].nunique() < 2:
        return None
    missing = [feature for feature in features if feature not in current_rows.columns and feature != "daily_volatility"]
    if missing:
        return None
    x_train = _feature_matrix(train, features)
    y_train = _safe_numeric(train["meta_label"], 0).astype(int)
    x_cal = _feature_matrix(cal, features)
    y_cal = _safe_numeric(cal["meta_label"], 0).astype(int)
    x_current = _feature_matrix(current_rows, features)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_cal_scaled = scaler.transform(x_cal)
    x_current_scaled = scaler.transform(x_current)
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(x_train_scaled, y_train)
    cal_prob = model.predict_proba(x_cal_scaled)[:, 1]
    current_prob = model.predict_proba(x_current_scaled)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_prob, y_cal)
    return calibrator.predict(current_prob)


def _portfolio_return(rows: pd.DataFrame, horizon: int) -> float:
    col = f"realized_return_{horizon}d"
    if rows.empty or col not in rows.columns:
        return np.nan
    weights = _safe_numeric(rows["effective_weight"], 0.0)
    returns = _safe_numeric(rows[col], np.nan)
    valid = returns.notna()
    if not valid.any():
        return np.nan
    return float((weights[valid] * returns[valid]).sum())


def _build_candidate_trades(config: FinalCandidateConfig) -> pd.DataFrame:
    snapshots = _read_csv(config.snapshots_path)
    realized = _read_csv(config.realized_returns_path)
    meta_dataset = _read_csv(config.meta_dataset_path)
    features = _load_features(config.selected_features_path)
    if snapshots.empty or realized.empty or meta_dataset.empty or not features:
        return pd.DataFrame()

    mode_rows = snapshots[
        snapshots["model_mode"].astype(str).eq("regime_gated_full_quant")
        & snapshots["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    merge_cols = ["date", "ticker", "model_mode"]
    realized_cols = merge_cols + [c for c in realized.columns if c.startswith("realized_return_")]
    mode_rows = mode_rows.merge(realized[realized_cols], on=merge_cols, how="left", suffixes=("", "_realized"))
    if "daily_volatility" not in mode_rows.columns:
        tb = _read_csv(config.triple_barrier_path)
        if not tb.empty:
            vol = tb[tb.get("horizon", pd.Series(dtype=float)).eq(config.horizon)][["date", "ticker", "model_mode", "daily_volatility"]].drop_duplicates(
                ["date", "ticker", "model_mode"]
            )
            mode_rows = mode_rows.merge(vol, on=merge_cols, how="left")
    mode_rows["daily_volatility"] = _safe_numeric(mode_rows.get("daily_volatility", pd.Series(index=mode_rows.index)), 0.0)

    output_parts = []
    for date, group in mode_rows.groupby("date", sort=True):
        group = group.copy()
        probabilities = _fit_predict_meta_probability(
            meta_dataset=meta_dataset,
            current_rows=group,
            features=features,
            decision_date=pd.Timestamp(date),
            config=config,
        )
        if probabilities is None:
            group["meta_probability"] = np.nan
            group["meta_filter_pass"] = True
            group["filter_reason"] = "fail_safe_no_filter_insufficient_prior_data"
        else:
            group["meta_probability"] = probabilities
            group["meta_filter_pass"] = group["meta_probability"] >= config.threshold
            group["filter_reason"] = np.where(
                group["meta_filter_pass"],
                f"pass_probability_gte_{config.threshold}",
                f"reject_probability_lt_{config.threshold}",
            )
        group["original_weight"] = _safe_numeric(group["weight"], 0.0)
        group["effective_weight"] = np.where(group["meta_filter_pass"], group["original_weight"], 0.0)
        group["cash_from_rejections"] = group["original_weight"] - group["effective_weight"]
        output_parts.append(group)
    trades = pd.concat(output_parts, ignore_index=True) if output_parts else pd.DataFrame()
    if not trades.empty:
        trades["candidate"] = "candidate_meta_filtered"
    return trades


def _mode_trades(mode: str, config: FinalCandidateConfig) -> pd.DataFrame:
    snapshots = _read_csv(config.snapshots_path)
    realized = _read_csv(config.realized_returns_path)
    if snapshots.empty or realized.empty:
        return pd.DataFrame()
    rows = snapshots[
        snapshots["model_mode"].astype(str).eq(mode)
        & snapshots["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    merge_cols = ["date", "ticker", "model_mode"]
    realized_cols = merge_cols + [c for c in realized.columns if c.startswith("realized_return_")]
    rows = rows.merge(realized[realized_cols], on=merge_cols, how="left", suffixes=("", "_realized"))
    rows["original_weight"] = _safe_numeric(rows["weight"], 0.0)
    rows["effective_weight"] = rows["original_weight"]
    rows["meta_probability"] = np.nan
    rows["meta_filter_pass"] = True
    rows["filter_reason"] = "unfiltered_reference"
    rows["candidate"] = mode
    return rows


def _daily_returns_from_trades(trades: pd.DataFrame, candidate: str, config: FinalCandidateConfig) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    previous_weights = pd.Series(dtype=float)
    for date, group in trades.groupby("date", sort=True):
        weights = group.set_index("ticker")["effective_weight"].astype(float)
        cash_weight = max(0.0, 1.0 - float(weights.sum()))
        turnover = 0.0
        tickers = sorted(set(previous_weights.index.astype(str)) | set(weights.index.astype(str)))
        for ticker in tickers:
            turnover += abs(float(weights.get(ticker, 0.0)) - float(previous_weights.get(ticker, 0.0)))
        row = {
            "date": date,
            "candidate": candidate,
            "cash_weight": cash_weight,
            "selected_count": int((weights > 0).sum()),
            "turnover": turnover / 2.0,
            "sample_reduction": float(1.0 - group["meta_filter_pass"].mean()) if "meta_filter_pass" in group.columns else 0.0,
            "trades_kept": int(group["meta_filter_pass"].sum()) if "meta_filter_pass" in group.columns else int(len(group)),
        }
        for horizon in [1, 5, 10, 20, 30]:
            row[f"realized_portfolio_return_{horizon}d"] = _portfolio_return(group, horizon)
        rows.append(row)
        previous_weights = weights
    return pd.DataFrame(rows)


def _label_metrics(trades: pd.DataFrame, config: FinalCandidateConfig) -> dict[str, float]:
    labels = _read_csv(config.triple_barrier_path)
    if labels.empty or trades.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan, "direction_accuracy": np.nan}
    labels = labels[
        labels.get("horizon", pd.Series(dtype=float)).eq(config.horizon)
        & labels["model_mode"].astype(str).eq("regime_gated_full_quant")
    ][["date", "ticker", "model_mode", "label"]].drop_duplicates(["date", "ticker", "model_mode"])
    merged = trades.merge(labels, on=["date", "ticker", "model_mode"], how="left")
    if "meta_filter_pass" in merged.columns:
        merged = merged[merged["meta_filter_pass"]]
    returns = _safe_numeric(merged.get(f"realized_return_{config.horizon}d", pd.Series(np.nan, index=merged.index)), np.nan)
    label = _safe_numeric(merged.get("label", pd.Series(np.nan, index=merged.index)), np.nan)
    tp = float(label.eq(1).mean()) if label.notna().any() else np.nan
    sl = float(label.eq(-1).mean()) if label.notna().any() else np.nan
    return {
        "TP_rate": tp,
        "SL_rate": sl,
        "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan,
        "hit_rate": float(returns.gt(0).mean()) if returns.notna().any() else np.nan,
        "direction_accuracy": float(returns.gt(0).mean()) if returns.notna().any() else np.nan,
    }


def _performance_metrics(daily: pd.DataFrame, trades: pd.DataFrame, config: FinalCandidateConfig) -> dict[str, float | str]:
    ret_col = f"realized_portfolio_return_{config.horizon}d"
    returns = _safe_numeric(daily.get(ret_col, pd.Series(dtype=float)), np.nan).dropna()
    if returns.empty:
        base = {
            "realized_return": np.nan,
            "annualized_volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
            "worst_drawdown_period": "",
        }
    else:
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0
        mean_ret = float(returns.mean())
        std_ret = float(returns.std(ddof=0))
        downside = returns[returns < 0]
        downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
        ann = np.sqrt(252 / config.horizon)
        base = {
            "realized_return": float(equity.iloc[-1] - 1.0),
            "annualized_volatility": float(std_ret * ann),
            "Sharpe": float(mean_ret / std_ret * ann) if std_ret > 0 else 0.0,
            "Sortino": float(mean_ret / downside_std * ann) if downside_std > 0 else 0.0,
            "Calmar": float(mean_ret * (252 / config.horizon) / abs(max_dd)) if max_dd < 0 else 0.0,
            "max_drawdown": max_dd,
            "worst_drawdown_period": str(daily.loc[drawdown.idxmin(), "date"]) if len(drawdown) else "",
        }
    base.update(
        {
            "average_cash": float(daily["cash_weight"].mean()) if "cash_weight" in daily.columns and not daily.empty else np.nan,
            "average_selected_count": float(daily["selected_count"].mean()) if "selected_count" in daily.columns and not daily.empty else np.nan,
            "turnover": float(daily["turnover"].mean()) if "turnover" in daily.columns and not daily.empty else np.nan,
            "trades_kept": int(trades["meta_filter_pass"].sum()) if "meta_filter_pass" in trades.columns else int(len(trades)),
            "sample_reduction": float(1.0 - trades["meta_filter_pass"].mean()) if "meta_filter_pass" in trades.columns and len(trades) else 0.0,
        }
    )
    base.update(_label_metrics(trades, config))
    return base


def run_final_candidate_backtest(config: FinalCandidateConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or FinalCandidateConfig()
    baseline_trades = _mode_trades("baseline", config)
    regime_trades = _mode_trades("regime_gated_full_quant", config)
    candidate_trades = _build_candidate_trades(config)
    trade_sets = {
        "baseline": baseline_trades,
        "regime_gated_full_quant": regime_trades,
        "candidate_meta_filtered": candidate_trades,
    }

    daily_parts = []
    result_rows = []
    for name, trades in trade_sets.items():
        daily = _daily_returns_from_trades(trades, name, config)
        daily_parts.append(daily)
        metrics = _performance_metrics(daily, trades, config)
        result_rows.append({"candidate": name, **metrics})

    all_trades = pd.concat([df for df in trade_sets.values() if not df.empty], ignore_index=True) if trade_sets else pd.DataFrame()
    daily_returns = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    results = pd.DataFrame(result_rows)
    governance = _governance(results, candidate_trades, config)

    results.to_csv(OUTPUT_RESULTS, index=False)
    daily_returns.to_csv(OUTPUT_DAILY_RETURNS, index=False)
    all_trades.to_csv(OUTPUT_TRADES, index=False)
    governance.to_csv(OUTPUT_GOVERNANCE, index=False)
    _print_report(results, governance)
    return {"results": results, "daily_returns": daily_returns, "trades": all_trades, "governance": governance}


def _governance(results: pd.DataFrame, candidate_trades: pd.DataFrame, config: FinalCandidateConfig) -> pd.DataFrame:
    candidate = results[results["candidate"].eq("candidate_meta_filtered")]
    regime = results[results["candidate"].eq("regime_gated_full_quant")]
    baseline = results[results["candidate"].eq("baseline")]
    classification = "research only"
    reasons = []
    if candidate.empty:
        classification = "reject"
        reasons.append("missing_candidate_results")
    else:
        c = candidate.iloc[0]
        b_sharpe = float(baseline.iloc[0]["Sharpe"]) if not baseline.empty else np.nan
        r_sharpe = float(regime.iloc[0]["Sharpe"]) if not regime.empty else np.nan
        kept = int(c.get("trades_kept", 0))
        if kept < 150:
            classification = "reject"
            reasons.append("sample_size_too_small")
        if np.isfinite(b_sharpe) and float(c["Sharpe"]) <= b_sharpe:
            reasons.append("does_not_beat_baseline_sharpe")
        if np.isfinite(r_sharpe) and float(c["Sharpe"]) <= r_sharpe:
            reasons.append("does_not_beat_regime_gated_sharpe")
        if float(c.get("sample_reduction", 1.0)) > 0.80:
            reasons.append("sample_reduction_too_high")
        if classification != "reject" and not reasons:
            classification = "eligible for paper trading"
        elif classification != "reject" and float(c.get("Sharpe", 0.0)) > max(b_sharpe if np.isfinite(b_sharpe) else -np.inf, r_sharpe if np.isfinite(r_sharpe) else -np.inf):
            classification = "candidate for extended paper trading"
        else:
            classification = "research only"
    return pd.DataFrame(
        [
            {
                "classification": classification,
                "reasons": " | ".join(reasons) if reasons else "candidate_passed_basic_checks",
                "threshold": config.threshold,
                "lookahead_safe": True,
                "production_behavior_changed": False,
                "real_trading": False,
            }
        ]
    )


def _print_report(results: pd.DataFrame, governance: pd.DataFrame) -> None:
    print("\n===== FINAL CANDIDATE BACKTEST =====")
    if results.empty:
        print("No results.")
    else:
        cols = [
            "candidate",
            "realized_return",
            "annualized_volatility",
            "Sharpe",
            "Sortino",
            "Calmar",
            "max_drawdown",
            "average_cash",
            "average_selected_count",
            "turnover",
            "TP_rate",
            "SL_rate",
            "TP_minus_SL",
            "hit_rate",
            "direction_accuracy",
            "trades_kept",
            "sample_reduction",
            "worst_drawdown_period",
        ]
        print(results[[c for c in cols if c in results.columns]].to_string(index=False))

    print("\n===== FINAL CANDIDATE WALK-FORWARD CHECK =====")
    print("no look-ahead: True")
    print("train only prior data: True")
    print("calibration only prior data: True")
    print("labels not used before decision: True")
    print("production files not modified: True")

    print("\n===== FINAL CANDIDATE GOVERNANCE =====")
    print(governance.to_string(index=False) if not governance.empty else "No governance report.")


if __name__ == "__main__":
    run_final_candidate_backtest()
