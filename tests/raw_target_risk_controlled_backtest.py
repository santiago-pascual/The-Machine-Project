from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_RESULTS_FILE = "raw_target_research_backtest_results.csv"
RAW_TRADES_FILE = "expected_return_ablation_trades.csv"
RAW_DAILY_FILE = "expected_return_ablation_daily_returns.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"
PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"

RESULTS_FILE = "raw_target_risk_controlled_results.csv"
DAILY_FILE = "raw_target_risk_controlled_daily_returns.csv"
TRADES_FILE = "raw_target_risk_controlled_trades.csv"
GOVERNANCE_FILE = "raw_target_risk_controlled_governance.csv"
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


def _max_drawdown(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    if len(returns) < 2:
        return np.nan
    vol = float(returns.std(ddof=0))
    if vol <= 0:
        return np.nan
    return float((returns.mean() / vol) * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    downside = returns[returns < 0]
    if returns.empty or len(downside) < 2:
        return np.nan
    downside_std = float(downside.std(ddof=0))
    if downside_std <= 0:
        return np.nan
    return float((returns.mean() * TRADING_DAYS) / (downside_std * np.sqrt(TRADING_DAYS)))


def _calmar(returns: pd.Series) -> float:
    returns = _num(returns).dropna()
    dd = abs(_max_drawdown(returns))
    if returns.empty or not np.isfinite(dd) or dd <= 0:
        return np.nan
    annualized = (1.0 + returns).prod() ** (TRADING_DAYS / max(1, len(returns))) - 1.0
    return float(annualized / dd)


def _prepare_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.empty and "date" in frame.columns:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame


def _raw_daily() -> pd.DataFrame:
    daily = _prepare_dates(_read_csv(RAW_DAILY_FILE))
    if daily.empty:
        return pd.DataFrame()
    raw = daily[daily["variant"].astype(str).eq("raw_target_return_only")].copy()
    if raw.empty:
        raw = daily[daily["variant"].astype(str).eq("no_signal_strength_adjustment")].copy()
    raw["raw_return"] = _num(raw["portfolio_return"])
    raw["raw_cash"] = _num(raw["cash_proxy"]).fillna(0.0)
    raw["raw_exposure"] = (1.0 - raw["raw_cash"]).clip(lower=0.0, upper=1.0)
    return raw[["date", "raw_return", "raw_cash", "raw_exposure", "selected_count", "turnover", "selected_tickers"]]


def _reference_daily(mode: str) -> pd.DataFrame:
    portfolio = _prepare_dates(_read_csv(PORTFOLIO_FILE))
    if portfolio.empty:
        return pd.DataFrame()
    ref = portfolio[portfolio["model_mode"].astype(str).eq(mode)].copy()
    ref[f"{mode}_return"] = _num(ref["realized_portfolio_return_1d"])
    ref[f"{mode}_cash"] = _num(ref["cash_weight"]).fillna(0.5)
    ref[f"{mode}_exposure"] = (1.0 - ref[f"{mode}_cash"]).clip(lower=0.0, upper=1.0)
    return ref[["date", f"{mode}_return", f"{mode}_cash", f"{mode}_exposure", "selected_count", "turnover"]]


def _scale_variant(raw: pd.DataFrame, baseline: pd.DataFrame, gated: pd.DataFrame, variant: str) -> pd.DataFrame:
    data = raw.merge(baseline, on="date", how="left").merge(gated, on="date", how="left", suffixes=("", "_gated"))
    raw_exposure = _num(data["raw_exposure"]).replace(0.0, np.nan)
    if variant == "raw_target_with_baseline_cash_floor":
        target_exposure = np.minimum(_num(data["raw_exposure"]), _num(data["baseline_exposure"]).fillna(0.5))
        reason = "baseline_cash_floor"
    elif variant == "raw_target_with_regime_gate_cash":
        target_exposure = np.minimum(_num(data["raw_exposure"]), _num(data["regime_gated_full_quant_exposure"]).fillna(0.5))
        reason = "regime_gate_cash_floor"
    elif variant == "raw_target_with_volatility_targeting":
        raw_vol = _num(data["raw_return"]).rolling(20, min_periods=5).std().shift(1)
        base_vol = _num(data["baseline_return"]).rolling(20, min_periods=5).std().shift(1)
        scale = (base_vol / raw_vol.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0).fillna(1.0)
        target_exposure = (_num(data["raw_exposure"]) * scale).clip(lower=0.0, upper=1.0)
        reason = "rolling_vol_target_to_baseline"
    elif variant == "raw_target_with_max_exposure_cap":
        target_exposure = np.minimum(_num(data["raw_exposure"]), 0.50)
        reason = "max_exposure_cap_50pct"
    elif variant == "raw_target_blend_50_50_with_baseline_expected_return":
        data["portfolio_return"] = 0.5 * _num(data["raw_return"]) + 0.5 * _num(data["baseline_return"])
        data["cash_weight"] = 0.5 * _num(data["raw_cash"]) + 0.5 * _num(data["baseline_cash"])
        data["exposure"] = (1.0 - data["cash_weight"]).clip(lower=0.0, upper=1.0)
        data["variant"] = variant
        data["scale_factor"] = 0.5
        data["risk_control_reason"] = "50_50_return_blend_with_baseline"
        return data
    else:
        target_exposure = _num(data["raw_exposure"])
        reason = "none"

    scale = (target_exposure / raw_exposure).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    data["portfolio_return"] = _num(data["raw_return"]) * scale
    data["exposure"] = target_exposure.fillna(0.0).clip(lower=0.0, upper=1.0)
    data["cash_weight"] = 1.0 - data["exposure"]
    data["variant"] = variant
    data["scale_factor"] = scale
    data["risk_control_reason"] = reason
    return data


def _labels_for_trades(trades: pd.DataFrame) -> dict[str, float]:
    labels = _prepare_dates(_read_csv(LABELS_FILE))
    if labels.empty or trades.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    labels20 = labels[labels["horizon"].astype(str).eq("20")] if "horizon" in labels else labels
    merged = trades[["date", "ticker"]].drop_duplicates().merge(labels20, on=["date", "ticker"], how="left")
    if merged.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean()) if "first_touch_type" in merged else np.nan
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean()) if "first_touch_type" in merged else np.nan
    hit = float((_num(merged.get("realized_return_at_barrier", pd.Series(dtype=float))) > 0).mean()) if "realized_return_at_barrier" in merged else np.nan
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan, "hit_rate": hit}


def _raw_trades() -> pd.DataFrame:
    trades = _prepare_dates(_read_csv(RAW_TRADES_FILE))
    if trades.empty:
        return pd.DataFrame()
    raw = trades[trades["variant"].astype(str).eq("raw_target_return_only")].copy()
    if raw.empty:
        raw = trades[trades["variant"].astype(str).eq("no_signal_strength_adjustment")].copy()
    return raw


def _scale_trades(raw_trades: pd.DataFrame, daily: pd.DataFrame, variant: str) -> pd.DataFrame:
    if raw_trades.empty or daily.empty:
        return pd.DataFrame()
    scale = daily[["date", "scale_factor", "cash_weight", "variant", "risk_control_reason"]].drop_duplicates("date")
    out = raw_trades.merge(scale, on="date", how="left", suffixes=("", "_variant"))
    weight_col = "ablation_weight" if "ablation_weight" in out.columns else "weight"
    out["controlled_weight"] = _num(out[weight_col]).fillna(0.0) * _num(out["scale_factor"]).fillna(0.0)
    out["variant"] = variant
    return out


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame, variant: str) -> dict[str, object]:
    returns = _num(daily.get("portfolio_return", pd.Series(dtype=float))).dropna()
    exposure = _num(daily.get("exposure", 1.0 - _num(daily.get("cash_weight", pd.Series(dtype=float)))))
    base = {
        "variant": variant,
        "realized_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
        "volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan,
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "average_cash": float(_num(daily.get("cash_weight", pd.Series(dtype=float))).mean()),
        "average_exposure": float(exposure.mean()),
        "turnover": float(_num(daily.get("turnover", pd.Series(dtype=float))).mean()),
    }
    base.update(_labels_for_trades(trades))
    return base


def _reference_metrics(mode: str) -> dict[str, object]:
    portfolio = _prepare_dates(_read_csv(PORTFOLIO_FILE))
    snapshots = _prepare_dates(_read_csv(SNAPSHOTS_FILE))
    labels = _prepare_dates(_read_csv(LABELS_FILE))
    daily = portfolio[portfolio["model_mode"].astype(str).eq(mode)].copy()
    trades = snapshots[snapshots["model_mode"].astype(str).eq(mode) & _bool(snapshots.get("selected", pd.Series(False, index=snapshots.index)))].copy()
    returns = _num(daily.get("realized_portfolio_return_1d", pd.Series(dtype=float))).dropna()
    out = {
        "variant": mode,
        "realized_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
        "volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan,
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "average_cash": float(_num(daily.get("cash_weight", pd.Series(dtype=float))).mean()),
        "average_exposure": float((1.0 - _num(daily.get("cash_weight", pd.Series(dtype=float)))).mean()),
        "turnover": float(_num(daily.get("turnover", pd.Series(dtype=float))).mean()),
    }
    out.update(_labels_for_trades(trades))
    return out


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    refs = results[results["variant"].isin(["baseline", "regime_gated_full_quant"])]
    best_ref_sharpe = float(refs["Sharpe"].max()) if not refs.empty else np.nan
    best_ref_dd = float(refs["max_drawdown"].min()) if not refs.empty else np.nan
    rows = []
    for _, row in results.iterrows():
        variant = str(row["variant"])
        if variant in {"baseline", "regime_gated_full_quant"}:
            continue
        sharpe = float(row.get("Sharpe", np.nan))
        dd = float(row.get("max_drawdown", np.nan))
        ret = float(row.get("realized_return", np.nan))
        exposure = float(row.get("average_exposure", np.nan))
        if sharpe > best_ref_sharpe and dd >= best_ref_dd * 1.5 and exposure <= 0.75:
            classification = "eligible for paper testing"
            reason = "sharpe_improves_with_controlled_drawdown_and_exposure"
        elif sharpe > best_ref_sharpe:
            classification = "candidate for shadow mode"
            reason = "sharpe_improves_but_risk_needs_review"
        elif ret > float(refs["realized_return"].max()) and exposure > 0.75:
            classification = "reject"
            reason = "improvement_only_from_higher_exposure"
        elif ret > float(refs["realized_return"].max()):
            classification = "research only"
            reason = "higher_return_without_sharpe_improvement"
        else:
            classification = "reject"
            reason = "does_not_improve_reference"
        rows.append(
            {
                "variant": variant,
                "classification": classification,
                "reason": reason,
                "Sharpe": sharpe,
                "reference_best_Sharpe": best_ref_sharpe,
                "max_drawdown": dd,
                "reference_worst_drawdown": best_ref_dd,
                "average_exposure": exposure,
                "production_change": "none",
            }
        )
    return pd.DataFrame(rows)


def run_raw_target_risk_controlled_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _raw_daily()
    baseline = _reference_daily("baseline")
    gated = _reference_daily("regime_gated_full_quant")
    raw_trades = _raw_trades()
    if raw.empty:
        raise ValueError("Raw target ablation daily returns are required.")

    variants = [
        "raw_target_with_baseline_cash_floor",
        "raw_target_with_regime_gate_cash",
        "raw_target_with_volatility_targeting",
        "raw_target_with_max_exposure_cap",
        "raw_target_blend_50_50_with_baseline_expected_return",
    ]
    daily_frames = []
    trade_frames = []
    result_rows = [_reference_metrics("baseline"), _reference_metrics("regime_gated_full_quant")]
    for variant in variants:
        daily = _scale_variant(raw, baseline, gated, variant)
        trades = _scale_trades(raw_trades, daily, variant)
        result_rows.append(_metrics(daily, trades, variant))
        daily_frames.append(daily)
        trade_frames.append(trades)

    results = pd.DataFrame(result_rows).sort_values("Sharpe", ascending=False)
    daily_out = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    trades_out = pd.concat(trade_frames, ignore_index=True, sort=False) if trade_frames else pd.DataFrame()
    governance = _governance(results)

    results.to_csv(RESULTS_FILE, index=False)
    daily_out.to_csv(DAILY_FILE, index=False)
    trades_out.to_csv(TRADES_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== RAW TARGET RISK-CONTROLLED BACKTEST =====")
    print(f"variants tested: {len(variants)}")
    print("production change: none")

    print("\n===== RAW TARGET VARIANT COMPARISON =====")
    cols = [
        "variant",
        "realized_return",
        "volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_cash",
        "average_exposure",
        "TP_rate",
        "SL_rate",
        "TP_minus_SL",
        "hit_rate",
        "turnover",
    ]
    print(results[cols].to_string(index=False))

    print("\n===== RAW TARGET RISK GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_FILE).resolve()}")
    print(f"Saved: {Path(TRADES_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, daily_out, trades_out, governance


if __name__ == "__main__":
    run_raw_target_risk_controlled_backtest()
