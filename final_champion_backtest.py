from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"
PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"
MODEL_COMPARISON_FILE = "historical_model_mode_comparison.csv"
LIFECYCLE_FILE = "model_lifecycle_status.csv"
SUMMARY_TEXT_FILE = "final_research_summary.txt"
CLEAN_RESEARCH_FILE = "clean_research_evaluation.csv"
TRIAL_LOG_FILE = "strategy_trial_log.csv"

RESULTS_FILE = "final_champion_backtest_results.csv"
DAILY_FILE = "final_champion_backtest_daily_returns.csv"
STRESS_FILE = "final_champion_stress_test.csv"
ROBUSTNESS_FILE = "final_champion_robustness.csv"
GOVERNANCE_FILE = "final_champion_governance.csv"
REPORT_FILE = "final_champion_report.txt"
CHAMPION_MODE = "regime_gated_full_quant"
TRADING_DAYS = 252


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def _max_drawdown(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _drawdown_period(daily: pd.DataFrame) -> dict[str, object]:
    returns = _num(daily["realized_portfolio_return_1d"]).fillna(0.0)
    if returns.empty:
        return {"worst_drawdown_start": None, "worst_drawdown_end": None}
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    end_idx = dd.idxmin()
    start_idx = equity.loc[:end_idx].idxmax()
    return {
        "worst_drawdown_start": daily.loc[start_idx, "date"],
        "worst_drawdown_end": daily.loc[end_idx, "date"],
        "worst_drawdown_value": float(dd.loc[end_idx]),
    }


def _sharpe(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if len(returns) < 2:
        return np.nan
    vol = float(returns.std(ddof=0))
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS)) if vol > 0 else np.nan


def _sortino(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    downside = returns[returns < 0]
    if returns.empty or len(downside) < 2:
        return np.nan
    down_std = float(downside.std(ddof=0))
    return float((returns.mean() * TRADING_DAYS) / (down_std * np.sqrt(TRADING_DAYS))) if down_std > 0 else np.nan


def _calmar(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    dd = abs(_max_drawdown(returns))
    if returns.empty or not np.isfinite(dd) or dd <= 0:
        return np.nan
    ann = (1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0
    return float(ann / dd)


def _load_champion_daily() -> pd.DataFrame:
    portfolio = _prepare_dates(_read_csv(PORTFOLIO_FILE))
    if portfolio.empty:
        return pd.DataFrame()
    daily = portfolio[portfolio["model_mode"].astype(str).eq(CHAMPION_MODE)].copy()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["cash_weight"] = _num(daily["cash_weight"]).fillna(0.0)
    daily["exposure"] = (1.0 - daily["cash_weight"]).clip(lower=0.0, upper=1.0)
    daily["realized_portfolio_return_1d"] = _num(daily["realized_portfolio_return_1d"]).fillna(0.0)
    daily["year"] = daily["date"].dt.year
    daily["quarter"] = daily["date"].dt.to_period("Q").astype(str)
    daily["cash_bucket"] = np.where(daily["cash_weight"] >= daily["cash_weight"].median(), "high_cash", "low_cash")
    rolling_vol = daily["realized_portfolio_return_1d"].rolling(20, min_periods=5).std()
    daily["volatility_regime"] = np.where(rolling_vol >= rolling_vol.quantile(0.75), "high_volatility", np.where(rolling_vol <= rolling_vol.quantile(0.25), "low_volatility", "normal_volatility"))
    rolling_abs = daily["realized_portfolio_return_1d"].abs().rolling(20, min_periods=5).mean()
    daily["correlation_proxy_regime"] = np.where(rolling_abs >= rolling_abs.quantile(0.75), "high_correlation_proxy", "normal_correlation_proxy")
    return daily


def _load_champion_trades() -> pd.DataFrame:
    snapshots = _prepare_dates(_read_csv(SNAPSHOTS_FILE))
    realized = _prepare_dates(_read_csv(REALIZED_FILE))
    if snapshots.empty:
        return pd.DataFrame()
    if not realized.empty:
        cols = [f"realized_return_{h}d" for h in [1, 5, 10, 20, 30] if f"realized_return_{h}d" in realized.columns]
        snapshots = snapshots.drop(columns=[col for col in cols if col in snapshots.columns], errors="ignore")
        snapshots = snapshots.merge(realized[["date", "ticker", "model_mode"] + cols], on=["date", "ticker", "model_mode"], how="left")
    trades = snapshots[
        snapshots["model_mode"].astype(str).eq(CHAMPION_MODE)
        & _bool(snapshots.get("selected", pd.Series(False, index=snapshots.index)))
    ].copy()
    return trades


def _labels_metrics(trades: pd.DataFrame) -> dict[str, float]:
    labels = _prepare_dates(_read_csv(LABELS_FILE))
    if labels.empty or trades.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    labels20 = labels[labels["horizon"].astype(str).eq("20")] if "horizon" in labels else labels
    merged = trades[["date", "ticker"]].drop_duplicates().merge(labels20, on=["date", "ticker"], how="left")
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean()) if "first_touch_type" in merged else np.nan
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean()) if "first_touch_type" in merged else np.nan
    hit = float((_num(merged.get("realized_return_at_barrier", pd.Series(dtype=float))) > 0).mean()) if "realized_return_at_barrier" in merged else np.nan
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan, "hit_rate": hit}


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    returns = _num(daily["realized_portfolio_return_1d"]).dropna()
    best = daily.loc[daily["realized_portfolio_return_1d"].idxmax()]
    worst = daily.loc[daily["realized_portfolio_return_1d"].idxmin()]
    dd_info = _drawdown_period(daily)
    row = {
        "model": "baseline + old regime gate",
        "realized_return": float((1.0 + returns).prod() - 1.0),
        "annualized_volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)),
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "average_cash": float(_num(daily["cash_weight"]).mean()),
        "average_exposure": float(_num(daily["exposure"]).mean()),
        "average_selected_count": float(_num(daily["selected_count"]).mean()),
        "turnover": float(_num(daily["turnover"]).mean()),
        "direction_accuracy": float((returns > 0).mean()),
        "sample_size": int(len(trades)),
        "best_period_date": best["date"],
        "best_period_return": float(best["realized_portfolio_return_1d"]),
        "worst_period_date": worst["date"],
        "worst_period_return": float(worst["realized_portfolio_return_1d"]),
        **dd_info,
        **_labels_metrics(trades),
    }
    return pd.DataFrame([row])


def _group_metrics(daily: pd.DataFrame, group_col: str, section: str) -> pd.DataFrame:
    rows = []
    for key, group in daily.groupby(group_col):
        returns = _num(group["realized_portfolio_return_1d"]).dropna()
        rows.append(
            {
                "section": section,
                "group": key,
                "observations": int(len(group)),
                "realized_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
                "volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan,
                "Sharpe": _sharpe(returns),
                "max_drawdown": _max_drawdown(returns),
                "average_cash": float(_num(group["cash_weight"]).mean()),
                "turnover": float(_num(group["turnover"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _stress_test(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    frames = [
        _group_metrics(daily, "year", "year"),
        _group_metrics(daily, "quarter", "quarter"),
        _group_metrics(daily, "volatility_regime", "volatility_regime"),
        _group_metrics(daily, "cash_bucket", "cash_bucket"),
        _group_metrics(daily, "correlation_proxy_regime", "correlation_proxy"),
    ]
    if not trades.empty and "regime" in trades.columns:
        regime_dates = trades.groupby("date")["regime"].agg(lambda x: x.astype(str).mode().iloc[0]).reset_index()
        regime_daily = daily.merge(regime_dates, on="date", how="left")
        frames.append(_group_metrics(regime_daily.dropna(subset=["regime"]), "regime", "regime"))
    return pd.concat(frames, ignore_index=True, sort=False)


def _robustness(daily: pd.DataFrame, trades: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    returns = _num(daily["realized_portfolio_return_1d"]).dropna()
    year_perf = stress[stress["section"].eq("year")].copy()
    best_year_share = float(year_perf["realized_return"].max() / max(1e-12, year_perf["realized_return"].sum())) if not year_perf.empty and year_perf["realized_return"].sum() != 0 else np.nan
    ticker_contrib = pd.DataFrame()
    if not trades.empty and {"ticker", "weight", "realized_return_1d"}.issubset(trades.columns):
        ticker_contrib = trades.copy()
        ticker_contrib["contribution"] = _num(ticker_contrib["weight"]).fillna(0.0) * _num(ticker_contrib["realized_return_1d"]).fillna(0.0)
        ticker_contrib = ticker_contrib.groupby("ticker")["contribution"].sum().sort_values(ascending=False)
    top_ticker_share = float(ticker_contrib.iloc[0] / ticker_contrib.sum()) if not ticker_contrib.empty and ticker_contrib.sum() != 0 else np.nan
    regime_perf = stress[stress["section"].eq("regime")]
    top_regime_share = float(regime_perf["realized_return"].max() / max(1e-12, regime_perf["realized_return"].sum())) if not regime_perf.empty and regime_perf["realized_return"].sum() != 0 else np.nan
    sorted_trade_returns = _num(trades.get("realized_return_20d", pd.Series(dtype=float))).dropna().sort_values(ascending=False)
    top_trade_share = float(sorted_trade_returns.head(max(1, int(len(sorted_trade_returns) * 0.05))).sum() / sorted_trade_returns.sum()) if len(sorted_trade_returns) and sorted_trade_returns.sum() != 0 else np.nan
    rows = [
        {"check": "depends_on_one_year", "value": best_year_share, "flag": bool(np.isfinite(best_year_share) and best_year_share > 0.70)},
        {"check": "depends_on_one_ticker", "value": top_ticker_share, "flag": bool(np.isfinite(top_ticker_share) and top_ticker_share > 0.35)},
        {"check": "depends_on_one_regime", "value": top_regime_share, "flag": bool(np.isfinite(top_regime_share) and top_regime_share > 0.70)},
        {"check": "returns_concentrated_in_few_trades", "value": top_trade_share, "flag": bool(np.isfinite(top_trade_share) and top_trade_share > 0.60)},
        {"check": "drawdown_acceptable", "value": _max_drawdown(returns), "flag": bool(_max_drawdown(returns) > -0.10)},
    ]
    return pd.DataFrame(rows)


def _anti_overfit_summary() -> dict[str, object]:
    clean = _read_csv(CLEAN_RESEARCH_FILE)
    trials = _read_csv(TRIAL_LOG_FILE)
    governed = clean[clean["trial_group"].astype(str).eq("governed_trials")].iloc[0] if not clean.empty and "trial_group" in clean.columns and (clean["trial_group"].astype(str).eq("governed_trials")).any() else pd.Series(dtype=object)
    exploratory = clean[clean["trial_group"].astype(str).eq("exploratory_trials")].iloc[0] if not clean.empty and "trial_group" in clean.columns and (clean["trial_group"].astype(str).eq("exploratory_trials")).any() else pd.Series(dtype=object)
    return {
        "all_time_overfitting_warning": exploratory.get("overfitting_warning", "missing"),
        "governed_overfitting_warning": governed.get("overfitting_warning", "missing"),
        "PBO_proxy": governed.get("PBO_proxy", np.nan),
        "deflated_Sharpe": governed.get("deflated_sharpe", np.nan),
        "promotion_status": governed.get("promotion_classification", "blocked"),
        "total_trials_logged": int(trials["number_of_trials"].sum()) if not trials.empty and "number_of_trials" in trials.columns else 0,
    }


def _governance(results: pd.DataFrame, robustness: pd.DataFrame, anti: dict[str, object]) -> pd.DataFrame:
    row = results.iloc[0]
    flags = robustness[robustness["flag"].astype(bool)]
    paper_short = _read_csv("paper_performance.csv")
    paper_too_short = len(paper_short) < 60
    if row["Sharpe"] >= 1.5 and row["max_drawdown"] > -0.10 and len(flags) <= 2:
        classification = "eligible for extended paper trading" if paper_too_short else "production review candidate"
        reason = "strong historical validation but paper history too short" if paper_too_short else "historical and paper evidence acceptable"
    elif row["Sharpe"] >= 1.0:
        classification = "eligible for paper trading"
        reason = "positive historical validation with monitoring required"
    else:
        classification = "research only"
        reason = "insufficient historical robustness"
    if str(anti.get("all_time_overfitting_warning", "")).lower() in {"high", "extreme"}:
        reason += "; all-time research overfitting warning remains high"
    return pd.DataFrame(
        [
            {
                "champion": "baseline + old regime gate",
                "classification": classification,
                "reason": reason,
                "Sharpe": row["Sharpe"],
                "max_drawdown": row["max_drawdown"],
                **anti,
                "production_change": "none",
            }
        ]
    )


def _write_report(results: pd.DataFrame, stress: pd.DataFrame, robustness: pd.DataFrame, governance: pd.DataFrame) -> str:
    text = "\n".join(
        [
            "===== FINAL CHAMPION BACKTEST =====",
            results.to_string(index=False),
            "",
            "===== FINAL CHAMPION STRESS TEST =====",
            stress.to_string(index=False),
            "",
            "===== FINAL CHAMPION ROBUSTNESS =====",
            robustness.to_string(index=False),
            "",
            "===== FINAL CHAMPION GOVERNANCE =====",
            governance.to_string(index=False),
        ]
    )
    Path(REPORT_FILE).write_text(text, encoding="utf-8")
    return text


def run_final_champion_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = _load_champion_daily()
    trades = _load_champion_trades()
    if daily.empty:
        raise ValueError("historical_walk_forward_portfolio_returns.csv has no champion rows.")
    results = _metrics(daily, trades)
    stress = _stress_test(daily, trades)
    robustness = _robustness(daily, trades, stress)
    governance = _governance(results, robustness, _anti_overfit_summary())

    results.to_csv(RESULTS_FILE, index=False)
    daily.to_csv(DAILY_FILE, index=False)
    stress.to_csv(STRESS_FILE, index=False)
    robustness.to_csv(ROBUSTNESS_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)
    report = _write_report(results, stress, robustness, governance)
    print(report)
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_FILE).resolve()}")
    print(f"Saved: {Path(STRESS_FILE).resolve()}")
    print(f"Saved: {Path(ROBUSTNESS_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    print(f"Saved: {Path(REPORT_FILE).resolve()}")
    return results, stress, robustness, governance


if __name__ == "__main__":
    run_final_champion_backtest()
