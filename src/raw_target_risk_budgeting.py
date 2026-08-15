from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_TRADES_FILE = "expected_return_ablation_trades.csv"
BASELINE_PORTFOLIO_FILE = "historical_walk_forward_portfolio_returns.csv"
LABELS_FILE = "historical_triple_barrier_labels.csv"

RESULTS_FILE = "raw_target_risk_budgeting_results.csv"
DAILY_FILE = "raw_target_risk_budgeting_daily_returns.csv"
TRADES_FILE = "raw_target_risk_budgeting_trades.csv"
GOVERNANCE_FILE = "raw_target_risk_budgeting_governance.csv"
TRADING_DAYS = 252


VARIANTS = [
    "raw_target_vol_target_15pct",
    "raw_target_vol_target_12pct",
    "raw_target_drawdown_guard",
    "raw_target_equal_risk_contribution",
    "raw_target_max_position_10pct",
    "raw_target_cash_floor_40pct",
    "raw_target_cash_floor_50pct",
    "raw_target_dynamic_cash_by_volatility",
]


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


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def _raw_trades() -> pd.DataFrame:
    trades = _prepare_dates(_read_csv(RAW_TRADES_FILE))
    if trades.empty:
        return pd.DataFrame()
    raw = trades[trades["variant"].astype(str).eq("raw_target_return_only")].copy()
    if raw.empty:
        raw = trades[trades["variant"].astype(str).eq("no_signal_strength_adjustment")].copy()
    weight_col = "ablation_weight" if "ablation_weight" in raw.columns else "weight"
    raw["base_weight"] = _num(raw[weight_col]).fillna(0.0).clip(lower=0.0)
    raw["asset_return_1d"] = _num(raw["realized_return_1d"]).fillna(0.0)
    raw["asset_vol_proxy"] = _num(raw.get("realized_return_20d", raw["asset_return_1d"])).abs().fillna(0.02) / np.sqrt(20)
    return raw


def _baseline_daily() -> pd.DataFrame:
    daily = _prepare_dates(_read_csv(BASELINE_PORTFOLIO_FILE))
    if daily.empty:
        return pd.DataFrame()
    baseline = daily[daily["model_mode"].astype(str).eq("baseline")].copy()
    baseline["baseline_return"] = _num(baseline["realized_portfolio_return_1d"]).fillna(0.0)
    baseline["baseline_cash"] = _num(baseline["cash_weight"]).fillna(0.5)
    return baseline[["date", "baseline_return", "baseline_cash"]]


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


def _apply_position_cap(weights: pd.Series, cap: float) -> pd.Series:
    weights = weights.clip(lower=0.0, upper=cap)
    total = float(weights.sum())
    return weights if total <= 1.0 else weights / total


def _erc_weights(group: pd.DataFrame, target_exposure: float) -> pd.Series:
    vol = _num(group["asset_vol_proxy"]).replace(0.0, np.nan).fillna(_num(group["asset_vol_proxy"]).median()).fillna(0.02)
    inv_vol = 1.0 / vol
    weights = (
        inv_vol / float(inv_vol.sum()) * target_exposure
        if float(inv_vol.sum()) > 0
        else pd.Series(target_exposure / len(group), index=group.index)
    )
    return weights.clip(lower=0.0)


def _variant_trades(raw: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    prev_returns: list[float] = []
    peak_equity = 1.0
    equity = 1.0
    for date, group in raw.groupby("date", sort=True):
        group = group.copy()
        base_weights = _num(group["base_weight"]).fillna(0.0)
        base_exposure = min(1.0, max(0.0, float(base_weights.sum())))
        target_exposure = base_exposure
        if variant == "raw_target_vol_target_15pct":
            realized_vol = np.std(prev_returns[-20:]) * np.sqrt(TRADING_DAYS) if len(prev_returns) >= 5 else 0.15
            target_exposure = min(base_exposure, 0.15 / max(realized_vol, 1e-6))
        elif variant == "raw_target_vol_target_12pct":
            realized_vol = np.std(prev_returns[-20:]) * np.sqrt(TRADING_DAYS) if len(prev_returns) >= 5 else 0.12
            target_exposure = min(base_exposure, 0.12 / max(realized_vol, 1e-6))
        elif variant == "raw_target_drawdown_guard":
            drawdown = equity / max(peak_equity, 1e-12) - 1.0
            target_exposure = base_exposure * (0.50 if drawdown < -0.08 else 0.75 if drawdown < -0.05 else 1.0)
        elif variant == "raw_target_equal_risk_contribution":
            target_exposure = min(base_exposure, 0.60)
        elif variant == "raw_target_max_position_10pct":
            target_exposure = base_exposure
        elif variant == "raw_target_cash_floor_40pct":
            target_exposure = min(base_exposure, 0.60)
        elif variant == "raw_target_cash_floor_50pct":
            target_exposure = min(base_exposure, 0.50)
        elif variant == "raw_target_dynamic_cash_by_volatility":
            realized_vol = np.std(prev_returns[-20:]) * np.sqrt(TRADING_DAYS) if len(prev_returns) >= 5 else 0.20
            target_exposure = min(base_exposure, 0.40 if realized_vol > 0.25 else 0.55 if realized_vol > 0.18 else 0.70)

        if variant == "raw_target_equal_risk_contribution":
            weights = _erc_weights(group, target_exposure)
        elif variant == "raw_target_max_position_10pct":
            weights = _apply_position_cap(base_weights, 0.10)
        else:
            weights = base_weights / max(base_exposure, 1e-12) * target_exposure
        group["budgeted_weight"] = weights.reindex(group.index).fillna(0.0)
        group["variant"] = variant
        period_return = float((group["budgeted_weight"] * group["asset_return_1d"]).sum())
        prev_returns.append(period_return)
        equity *= 1.0 + period_return
        peak_equity = max(peak_equity, equity)
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _daily_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous_assets: set[str] = set()
    for (variant, date), group in trades.groupby(["variant", "date"], sort=True):
        weights = _num(group["budgeted_weight"]).fillna(0.0)
        assets = set(group.loc[weights > 0, "ticker"].astype(str))
        rows.append(
            {
                "date": date,
                "variant": variant,
                "portfolio_return": float((weights * _num(group["asset_return_1d"]).fillna(0.0)).sum()),
                "cash_weight": max(0.0, 1.0 - float(weights.sum())),
                "exposure": min(1.0, float(weights.sum())),
                "average_position_size": float(weights[weights > 0].mean()) if (weights > 0).any() else 0.0,
                "max_position_size": float(weights.max()) if len(weights) else 0.0,
                "turnover": len(assets.symmetric_difference(previous_assets)) / max(1, len(assets | previous_assets)),
            }
        )
        previous_assets = assets
    return pd.DataFrame(rows)


def _label_metrics(trades: pd.DataFrame) -> dict[str, float]:
    labels = _prepare_dates(_read_csv(LABELS_FILE))
    if labels.empty or trades.empty:
        return {"TP_rate": np.nan, "SL_rate": np.nan, "TP_minus_SL": np.nan, "hit_rate": np.nan}
    labels20 = labels[labels["horizon"].astype(str).eq("20")] if "horizon" in labels else labels
    merged = trades[["date", "ticker"]].drop_duplicates().merge(labels20, on=["date", "ticker"], how="left")
    tp = float((merged["first_touch_type"].astype(str) == "take_profit").mean()) if "first_touch_type" in merged else np.nan
    sl = float((merged["first_touch_type"].astype(str) == "stop_loss").mean()) if "first_touch_type" in merged else np.nan
    hit = (
        float((_num(merged.get("realized_return_at_barrier", pd.Series(dtype=float))) > 0).mean())
        if "realized_return_at_barrier" in merged
        else np.nan
    )
    return {"TP_rate": tp, "SL_rate": sl, "TP_minus_SL": tp - sl if np.isfinite(tp) and np.isfinite(sl) else np.nan, "hit_rate": hit}


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame, variant: str) -> dict[str, object]:
    returns = _num(daily["portfolio_return"]).dropna()
    out = {
        "variant": variant,
        "realized_return": float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan,
        "annualized_volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else np.nan,
        "Sharpe": _sharpe(returns),
        "Sortino": _sortino(returns),
        "Calmar": _calmar(returns),
        "max_drawdown": _max_drawdown(returns),
        "average_cash": float(_num(daily["cash_weight"]).mean()),
        "average_exposure": float(_num(daily["exposure"]).mean()),
        "average_position_size": float(_num(daily["average_position_size"]).mean()),
        "max_position_size": float(_num(daily["max_position_size"]).max()),
        "turnover": float(_num(daily["turnover"]).mean()),
    }
    out.update(_label_metrics(trades))
    return out


def _reference_metrics() -> pd.DataFrame:
    baseline = _baseline_daily()
    if baseline.empty:
        return pd.DataFrame()
    returns = _num(baseline["baseline_return"]).dropna()
    return pd.DataFrame(
        [
            {
                "variant": "baseline",
                "realized_return": float((1.0 + returns).prod() - 1.0),
                "annualized_volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)),
                "Sharpe": _sharpe(returns),
                "Sortino": _sortino(returns),
                "Calmar": _calmar(returns),
                "max_drawdown": _max_drawdown(returns),
                "average_cash": float(_num(baseline["baseline_cash"]).mean()),
                "average_exposure": float((1.0 - _num(baseline["baseline_cash"])).mean()),
                "average_position_size": np.nan,
                "max_position_size": np.nan,
                "turnover": np.nan,
                "TP_rate": np.nan,
                "SL_rate": np.nan,
                "TP_minus_SL": np.nan,
                "hit_rate": np.nan,
            }
        ]
    )


def _governance(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["variant"].eq("baseline")]
    ref_sharpe = float(baseline["Sharpe"].iloc[0]) if not baseline.empty else np.nan
    ref_dd = float(baseline["max_drawdown"].iloc[0]) if not baseline.empty else np.nan
    ref_ret = float(baseline["realized_return"].iloc[0]) if not baseline.empty else np.nan
    rows = []
    for _, row in results.iterrows():
        variant = str(row["variant"])
        if variant == "baseline":
            continue
        sharpe = float(row["Sharpe"])
        dd = float(row["max_drawdown"])
        ret = float(row["realized_return"])
        exposure = float(row["average_exposure"])
        if sharpe > ref_sharpe and dd >= ref_dd * 1.35 and ret > ref_ret:
            classification = "eligible for paper testing"
            reason = "improves_sharpe_return_and_controls_drawdown"
        elif ret > ref_ret and exposure > 0.75:
            classification = "reject"
            reason = "improvement_only_from_higher_exposure"
        elif ret > ref_ret:
            classification = "candidate for shadow mode"
            reason = "higher_return_but_sharpe_or_drawdown_not_enough"
        else:
            classification = "reject"
            reason = "does_not_improve_baseline"
        rows.append(
            {
                "variant": variant,
                "classification": classification,
                "reason": reason,
                "Sharpe": sharpe,
                "baseline_Sharpe": ref_sharpe,
                "max_drawdown": dd,
                "baseline_max_drawdown": ref_dd,
                "average_exposure": exposure,
                "production_change": "none",
            }
        )
    return pd.DataFrame(rows)


def run_raw_target_risk_budgeting() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _raw_trades()
    if raw.empty:
        raise ValueError("Raw target trades are required.")
    all_trades = []
    all_daily = []
    metrics = []
    for variant in VARIANTS:
        trades = _variant_trades(raw, variant)
        daily = _daily_from_trades(trades)
        all_trades.append(trades)
        all_daily.append(daily)
        metrics.append(_metrics(daily, trades, variant))
    results = pd.concat([_reference_metrics(), pd.DataFrame(metrics)], ignore_index=True).sort_values("Sharpe", ascending=False)
    daily_out = pd.concat(all_daily, ignore_index=True)
    trades_out = pd.concat(all_trades, ignore_index=True)
    governance = _governance(results)

    results.to_csv(RESULTS_FILE, index=False)
    daily_out.to_csv(DAILY_FILE, index=False)
    trades_out.to_csv(TRADES_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)

    print("\n===== RAW TARGET RISK BUDGETING =====")
    print(f"variants tested: {len(VARIANTS)}")
    print("production change: none")

    print("\n===== RISK BUDGET VARIANT COMPARISON =====")
    cols = [
        "variant",
        "realized_return",
        "annualized_volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "max_drawdown",
        "average_cash",
        "average_exposure",
        "average_position_size",
        "max_position_size",
        "turnover",
        "TP_rate",
        "SL_rate",
        "TP_minus_SL",
        "hit_rate",
    ]
    print(results[cols].to_string(index=False))

    print("\n===== RAW TARGET RISK BUDGET GOVERNANCE =====")
    print(governance.to_string(index=False))
    print(f"\nSaved: {Path(RESULTS_FILE).resolve()}")
    print(f"Saved: {Path(DAILY_FILE).resolve()}")
    print(f"Saved: {Path(TRADES_FILE).resolve()}")
    print(f"Saved: {Path(GOVERNANCE_FILE).resolve()}")
    return results, daily_out, trades_out, governance


if __name__ == "__main__":
    run_raw_target_risk_budgeting()
