from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from covariance_estimation import calculate_covariance
from ema_timing_model import apply_timing_to_expected_returns, compute_asset_timing
from expected_returns_model import compute_expected_returns
from exposure_control import compute_net_exposure
from full_quant_regime_gate import (
    average_entropy_from_diagnostics,
    average_trend_score,
    evaluate_full_quant_regime_gate,
)
from market_regime_model import compute_market_regime_model
from portfolio_optimizer import PortfolioOptimizer
from quant_target_model import generate_quant_targets
from risk_metrics import compute_return_risk_metrics
from trend_persistence_engine import (
    apply_trend_persistence_to_expected_returns,
    compute_trend_persistence,
)

DEFAULT_REDUCED_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "AVGO",
    "AMD",
    "NFLX",
    "COST",
    "ADBE",
    "PEP",
    "CSCO",
    "INTC",
    "AMAT",
    "QCOM",
    "TXN",
    "CCJ",
    "YPF",
    "VIST",
]

HORIZONS = (5, 10, 20)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class WalkForwardConfig:
    start_date: str | None = None
    end_date: str | None = None
    step_size_days: int = 5
    max_test_dates: int = 20
    reduced_universe: list[str] | None = None
    optimizer_generations_backtest: int = 50
    disable_live_prices: bool = True
    lookback_window: int = 252
    min_history_required: int = 252
    output_predictions: str = "walk_forward_predictions.csv"
    output_portfolio_returns: str = "walk_forward_portfolio_returns.csv"
    output_summary: str = "walk_forward_summary.csv"
    timing_model: str = "ema"
    target_model: str = "basic"
    model_mode: str = "baseline"


def _generate_basic_target_prices(prices_df: pd.DataFrame) -> pd.Series:
    returns = prices_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    current = prices_df.ffill().iloc[-1]
    momentum = prices_df.ffill().pct_change(20, fill_method=None).iloc[-1].fillna(0.0)
    volatility = returns.tail(60).std().fillna(0.0)
    target_pct = (0.04 + 1.5 * volatility + 0.25 * momentum.clip(lower=-0.10, upper=0.20)).clip(0.01, 0.25)
    return (current * (1.0 + target_pct)).replace([np.inf, -np.inf], np.nan).fillna(current * 1.01)


def _safe_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    clean = prices_df.replace([np.inf, -np.inf], np.nan).ffill().dropna(axis=1, how="any")
    returns = clean.pct_change(fill_method=None).dropna(how="any")
    returns = returns.loc[:, returns.std() > 0]
    return returns


def _diagnostic_series(diagnostics_df: pd.DataFrame, column: str, index: pd.Index, default: float) -> pd.Series:
    if column not in diagnostics_df.columns:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(diagnostics_df[column], errors="coerce").reindex(index).fillna(default)


def _select_assets(
    timing_adjusted_returns: pd.Series,
    signal_strength: pd.Series,
    max_assets: int = 4,
    min_assets: int = 2,
) -> list[str]:
    signal = signal_strength.reindex(timing_adjusted_returns.index).fillna(0.0)
    candidates = timing_adjusted_returns[(timing_adjusted_returns > 0) & (signal > 0.15)]
    if candidates.empty:
        selected = timing_adjusted_returns.sort_values(ascending=False).head(max_assets).index.tolist()
    else:
        selected = candidates.sort_values(ascending=False).head(min(max_assets, len(candidates))).index.tolist()
    if len(selected) < min_assets:
        selected = timing_adjusted_returns.sort_values(ascending=False).head(min_assets).index.tolist()
    return [ticker for ticker in selected if ticker in timing_adjusted_returns.index]


def _future_asset_return(prices_df: pd.DataFrame, ticker: str, t_pos: int, horizon: int) -> float:
    future_pos = t_pos + horizon
    if ticker not in prices_df.columns or future_pos >= len(prices_df):
        return np.nan
    start_price = prices_df.iloc[t_pos][ticker]
    end_price = prices_df.iloc[future_pos][ticker]
    if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0:
        return np.nan
    return float(end_price / start_price - 1.0)


def _run_shadow_production_pipeline(
    historical_prices: pd.DataFrame,
    rf_daily: float,
    optimizer_generations: int,
    timing_model: str = "ema",
    target_model: str = "basic",
    model_mode: str = "baseline",
) -> dict[str, object]:
    np.random.seed(42)
    returns_df = _safe_returns(historical_prices)
    if returns_df.shape[0] < 60 or returns_df.shape[1] < 2:
        raise ValueError("Insufficient return history for walk-forward date.")
    prices = historical_prices[returns_df.columns].ffill().dropna(how="any")
    returns_df = returns_df[prices.columns]

    old_target_prices = _generate_basic_target_prices(prices)
    regime = compute_market_regime_model(
        prices_df=prices,
        returns_df=returns_df,
        spy_close=prices["SPY"] if "SPY" in prices.columns else pd.Series(dtype=float),
        vix_close=pd.Series(dtype=float),
        strict_market_data=False,
    )
    regime_score = float(regime["risk_score"])
    regime_type = str(regime["regime"])
    regime_confidence = float(regime.get("regime_confidence", 0.3))

    gate_info: dict[str, object] = {
        "gate_decision": "not_applicable",
        "allow_full_quant": False,
        "reason": "mode_not_regime_gated",
    }
    active_timing_model = timing_model
    active_target_model = target_model
    if model_mode == "full_quant_research":
        active_timing_model = "trend_persistence"
        active_target_model = "quant"
    elif model_mode == "regime_gated_full_quant":
        gate_info = evaluate_full_quant_regime_gate(
            regime=regime_type,
            market_mode="aggressive" if regime_type == "risk_on" and regime_confidence > 0.5 else "neutral",
            regime_confidence=regime_confidence,
            regime_score=regime_score,
            volatility_condition=regime.get("vol_regime_ratio", np.nan),
            vix_z=regime.get("vix_z", np.nan),
            breadth=regime.get("breadth", np.nan),
            spy_momentum_20d=regime.get("spy_momentum_20d", np.nan),
        )
        if bool(gate_info["allow_full_quant"]):
            active_timing_model = "trend_persistence"
            active_target_model = "quant"
        else:
            active_timing_model = "ema"
            active_target_model = "basic"

    if active_target_model == "quant":
        quant_targets = generate_quant_targets(
            prices_df=prices,
            old_target_prices=old_target_prices,
            regime_type=regime_type,
            horizon_days=20,
            blend_weight=0.15,
        )
        target_prices = quant_targets["quant_target_price"].reindex(prices.columns)
    else:
        target_prices = old_target_prices

    expected_returns, diagnostics = compute_expected_returns(
        prices_df=prices,
        target_prices=target_prices,
        use_live_prices=False,
        risk_free_daily=rf_daily,
        center_distribution=False,
        dead_zone=0.001,
        regime_score=regime_score,
        regime_type=regime_type,
        regime_signal_threshold=0.2,
    )
    diagnostics_df = pd.DataFrame(diagnostics).T.reindex(prices.columns)
    signal_strength = _diagnostic_series(diagnostics_df, "signal_strength", prices.columns, 0.0)
    quality_score = _diagnostic_series(diagnostics_df, "quality_score", prices.columns, 0.5)
    target_confidence = _diagnostic_series(diagnostics_df, "target_confidence", prices.columns, 0.5)

    timing_df = compute_asset_timing(prices, timing_mode="daily")
    trend_persistence_df = compute_trend_persistence(
        prices_df=prices,
        returns_df=returns_df,
        selected_tickers=list(prices.columns),
        diagnostics_df=diagnostics_df,
        market_regime=regime,
    )
    if model_mode == "regime_gated_full_quant":
        refined_gate = evaluate_full_quant_regime_gate(
            regime=regime_type,
            market_mode="aggressive" if regime_type == "risk_on" and regime_confidence > 0.5 else "neutral",
            regime_confidence=regime_confidence,
            regime_score=regime_score,
            volatility_condition=regime.get("vol_regime_ratio", np.nan),
            vix_z=regime.get("vix_z", np.nan),
            breadth=regime.get("breadth", np.nan),
            spy_momentum_20d=regime.get("spy_momentum_20d", np.nan),
            average_entropy=average_entropy_from_diagnostics(diagnostics_df),
            average_trend_persistence_score=average_trend_score(trend_persistence_df),
        )
        if refined_gate["gate_decision"] != gate_info.get("gate_decision"):
            gate_info = refined_gate
        active_timing_model = "trend_persistence" if bool(gate_info["allow_full_quant"]) else "ema"
        desired_target_model = "quant" if bool(gate_info["allow_full_quant"]) else "basic"
        if active_target_model != desired_target_model:
            active_target_model = desired_target_model
            if active_target_model == "quant":
                quant_targets = generate_quant_targets(
                    prices_df=prices,
                    old_target_prices=old_target_prices,
                    regime_type=regime_type,
                    horizon_days=20,
                    blend_weight=0.15,
                )
                target_prices = quant_targets["quant_target_price"].reindex(prices.columns)
            else:
                target_prices = old_target_prices
            expected_returns, diagnostics = compute_expected_returns(
                prices_df=prices,
                target_prices=target_prices,
                use_live_prices=False,
                risk_free_daily=rf_daily,
                center_distribution=False,
                dead_zone=0.001,
                regime_score=regime_score,
                regime_type=regime_type,
                regime_signal_threshold=0.2,
            )
            diagnostics_df = pd.DataFrame(diagnostics).T.reindex(prices.columns)
            signal_strength = _diagnostic_series(diagnostics_df, "signal_strength", prices.columns, 0.0)
            quality_score = _diagnostic_series(diagnostics_df, "quality_score", prices.columns, 0.5)
            target_confidence = _diagnostic_series(diagnostics_df, "target_confidence", prices.columns, 0.5)

    if active_timing_model == "trend_persistence":
        timing_adjusted_returns = apply_trend_persistence_to_expected_returns(
            adjusted_expected_returns=expected_returns,
            trend_persistence_df=trend_persistence_df,
        )
    else:
        timing_adjusted_returns = apply_timing_to_expected_returns(expected_returns, timing_df)
    selected_assets = _select_assets(timing_adjusted_returns, signal_strength)
    selected_assets = [ticker for ticker in selected_assets if ticker in returns_df.columns]
    if len(selected_assets) < 2:
        raise ValueError("Insufficient selected assets for optimization.")

    selected_returns = returns_df[selected_assets]
    selected_expected = timing_adjusted_returns.reindex(selected_assets).fillna(0.0)
    covariance_matrix = calculate_covariance(selected_returns, method="ledoit_wolf", shrinkage_intensity=0.10)
    optimizer = PortfolioOptimizer(
        returns_df=selected_returns,
        rf_daily=rf_daily,
        expected_daily_returns=selected_expected,
        use_expected_returns=True,
        alpha=0.5,
        no_opportunity=False,
        defensive_mode=True,
        max_weight=0.50,
        regime_score=regime_score,
        regime_type=regime_type,
        regime_confidence=regime_confidence,
        n_generations=optimizer_generations,
        random_seed=42,
        covariance_matrix=covariance_matrix,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        weights, sharpe, best_return, best_volatility, _ = optimizer.optimize()

    raw_weights = pd.Series(weights, index=selected_assets, dtype=float).clip(lower=0.0)
    raw_weights = raw_weights / float(raw_weights.sum()) if float(raw_weights.sum()) > 0 else pd.Series(1.0 / len(selected_assets), index=selected_assets)
    exposure_info = compute_net_exposure(
        regime_score=regime_score,
        regime_confidence=regime_confidence,
        expected_returns=timing_adjusted_returns,
        signal_strengths=signal_strength,
        timeframe="daily",
    )
    net_exposure = float(exposure_info["net_exposure"])
    final_weights = raw_weights * net_exposure
    cash_weight = max(0.0, 1.0 - float(final_weights.sum()))

    return {
        "prices": prices,
        "returns_df": returns_df,
        "target_prices": target_prices.reindex(prices.columns),
        "expected_returns": timing_adjusted_returns.reindex(prices.columns).fillna(0.0),
        "diagnostics_df": diagnostics_df,
        "signal_strength": signal_strength,
        "quality_score": quality_score,
        "target_confidence": target_confidence,
        "selected_assets": selected_assets,
        "weights": final_weights,
        "cash_weight": cash_weight,
        "regime": regime_type,
        "portfolio_expected_return": float(best_return),
        "portfolio_expected_volatility": float(best_volatility),
        "portfolio_expected_sharpe": float(sharpe),
        "timing_model": active_timing_model,
        "target_model": active_target_model,
        "model_mode": model_mode,
        "gate_info": gate_info,
    }


def _portfolio_forward_return(
    prices_df: pd.DataFrame,
    weights: pd.Series,
    t_pos: int,
    horizon: int,
) -> float:
    asset_returns = {
        ticker: _future_asset_return(prices_df, ticker, t_pos, horizon)
        for ticker in weights.index
        if ticker in prices_df.columns
    }
    clean = pd.Series(asset_returns, dtype=float).dropna()
    if clean.empty:
        return np.nan
    aligned_weights = weights.reindex(clean.index).fillna(0.0)
    return float(np.dot(aligned_weights, clean))


def _max_drawdown_from_returns(returns: pd.Series) -> tuple[float, str]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0, ""
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    worst_date = str(drawdown.idxmin().date()) if hasattr(drawdown.idxmin(), "date") else str(drawdown.idxmin())
    return float(drawdown.min()), worst_date


def _build_summary(
    prediction_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    realized_1d = pd.to_numeric(portfolio_df["realized_portfolio_return_1d"], errors="coerce").dropna()
    if not realized_1d.empty and "date" in portfolio_df.columns:
        realized_1d.index = pd.to_datetime(portfolio_df.loc[realized_1d.index, "date"], errors="coerce")
    risk = compute_return_risk_metrics(realized_1d)
    max_drawdown, worst_period = _max_drawdown_from_returns(realized_1d)
    rows = {
        "test_start": str(portfolio_df["date"].iloc[0]) if not portfolio_df.empty else "",
        "test_end": str(portfolio_df["date"].iloc[-1]) if not portfolio_df.empty else "",
        "number_of_test_dates": len(portfolio_df),
        "average_cash": float(portfolio_df["cash_weight"].mean()) if not portfolio_df.empty else 0.0,
        "average_selected_count": float(portfolio_df["selected_count"].mean()) if not portfolio_df.empty else 0.0,
        "realized_return": float((1.0 + realized_1d).prod() - 1.0) if not realized_1d.empty else 0.0,
        "realized_volatility": float(risk["annualized_volatility"]),
        "realized_sharpe": float(risk["annualized_return_estimate"] / risk["annualized_volatility"]) if risk["annualized_volatility"] > 0 else 0.0,
        "max_drawdown": max_drawdown,
        "Sortino": float(risk["sortino_ratio"]),
        "Calmar": float(risk["calmar_ratio"]),
        "average_turnover": float(portfolio_df["turnover"].mean()) if not portfolio_df.empty else 0.0,
        "worst_drawdown_period": worst_period,
    }
    for horizon in HORIZONS:
        realized = pd.to_numeric(portfolio_df[f"realized_portfolio_return_{horizon}d"], errors="coerce")
        expected = pd.to_numeric(portfolio_df["portfolio_expected_return"], errors="coerce")
        expected_h = (1.0 + expected).pow(horizon) - 1.0
        valid = pd.concat([expected_h, realized], axis=1).dropna()
        rows[f"hit_rate_{horizon}d"] = float((realized > 0).mean()) if realized.notna().any() else np.nan
        if valid.empty:
            rows[f"direction_accuracy_{horizon}d"] = np.nan
            rows[f"MAE_{horizon}d"] = np.nan
            rows[f"RMSE_{horizon}d"] = np.nan
        else:
            err = valid.iloc[:, 0] - valid.iloc[:, 1]
            rows[f"direction_accuracy_{horizon}d"] = float((np.sign(valid.iloc[:, 0]) == np.sign(valid.iloc[:, 1])).mean())
            rows[f"MAE_{horizon}d"] = float(err.abs().mean())
            rows[f"RMSE_{horizon}d"] = float(np.sqrt(np.square(err).mean()))
    return pd.DataFrame([rows])


def run_walk_forward_backtest(
    prices_df: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or WalkForwardConfig()
    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    data = prices_df.copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index().ffill()
    universe = cfg.reduced_universe or DEFAULT_REDUCED_UNIVERSE
    available = [ticker for ticker in universe if ticker in data.columns]
    data = data[available].dropna(axis=1, thresh=max(60, int(len(data) * 0.50)))
    if data.shape[1] < 2:
        raise ValueError("Not enough valid tickers for walk-forward backtest.")
    if cfg.start_date:
        start_bound = pd.Timestamp(cfg.start_date)
    else:
        start_bound = data.index[min(max(cfg.min_history_required, cfg.lookback_window), len(data) - max(HORIZONS) - 1)]
    end_bound = pd.Timestamp(cfg.end_date) if cfg.end_date else data.index[-max(HORIZONS) - 1]

    eligible_positions = [
        pos
        for pos, date in enumerate(data.index)
        if pos >= max(cfg.min_history_required, cfg.lookback_window)
        and date >= start_bound
        and date <= end_bound
        and pos + max(HORIZONS) < len(data)
    ]
    eligible_positions = eligible_positions[:: max(1, cfg.step_size_days)]
    if cfg.max_test_dates:
        eligible_positions = eligible_positions[: cfg.max_test_dates]

    prediction_rows: list[dict[str, object]] = []
    portfolio_rows: list[dict[str, object]] = []
    previous_selected: set[str] = set()
    skipped = 0
    skip_reasons: list[str] = []
    rf_daily = 0.0

    print("\n===== WALK-FORWARD LOOK-AHEAD CHECK =====")
    print("every run uses data only up to t: True")
    print(f"live prices disabled: {cfg.disable_live_prices}")
    print("target generation uses truncated data: True")
    print("regime uses truncated data: True")
    print("covariance uses truncated data: True")
    print("realized returns computed only after predictions are saved: True")

    for t_pos in eligible_positions:
        date = data.index[t_pos]
        historical = data.iloc[: t_pos + 1].tail(cfg.lookback_window)
        try:
            result = _run_shadow_production_pipeline(
                historical_prices=historical,
                rf_daily=rf_daily,
                optimizer_generations=cfg.optimizer_generations_backtest,
                timing_model=cfg.timing_model,
                target_model=cfg.target_model,
                model_mode=cfg.model_mode,
            )
        except Exception as exc:
            skipped += 1
            skip_reasons.append(f"{date.strftime('%Y-%m-%d')}: {exc}")
            continue

        selected_assets = list(result["selected_assets"])
        weights = pd.Series(result["weights"], dtype=float)
        selected_set = set(selected_assets)
        union = selected_set | previous_selected
        turnover = len(selected_set ^ previous_selected) / len(union) if union else 0.0
        previous_selected = selected_set

        for ticker in result["prices"].columns:
            selected = ticker in selected_set
            weight = float(weights.get(ticker, 0.0))
            row = {
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "selected": bool(selected),
                "weight": weight,
                "expected_daily_return": float(result["expected_returns"].get(ticker, 0.0)),
                "expected_total_return": float((1.0 + result["expected_returns"].get(ticker, 0.0)) ** 20 - 1.0),
                "target_price": float(result["target_prices"].get(ticker, np.nan)),
                "current_price": float(data.iloc[t_pos].get(ticker, np.nan)),
                "target_confidence": float(result["target_confidence"].get(ticker, np.nan)),
                "signal_strength": float(result["signal_strength"].get(ticker, np.nan)),
                "quality_score": float(result["quality_score"].get(ticker, np.nan)),
                "regime": str(result["regime"]),
                "timing_model": str(result["timing_model"]),
                "target_model": str(result["target_model"]),
                "model_mode": str(result.get("model_mode", cfg.model_mode)),
                "gate_decision": str(result.get("gate_info", {}).get("gate_decision", "not_applicable")),
                "gate_reason": str(result.get("gate_info", {}).get("reason", "")),
            }
            for horizon in HORIZONS:
                row[f"realized_return_{horizon}d"] = _future_asset_return(data, ticker, t_pos, horizon)
            prediction_rows.append(row)

        portfolio_row = {
            "date": date.strftime("%Y-%m-%d"),
            "realized_portfolio_return_1d": _portfolio_forward_return(data, weights, t_pos, 1),
            "cash_weight": float(result["cash_weight"]),
            "selected_count": len(selected_assets),
            "turnover": float(turnover),
            "max_weight": float(weights.max()) if len(weights) else 0.0,
            "portfolio_expected_return": float(result["portfolio_expected_return"]),
            "portfolio_expected_volatility": float(result["portfolio_expected_volatility"]),
            "portfolio_expected_sharpe": float(result["portfolio_expected_sharpe"]),
            "timing_model": str(result["timing_model"]),
            "target_model": str(result["target_model"]),
            "model_mode": str(result.get("model_mode", cfg.model_mode)),
            "gate_decision": str(result.get("gate_info", {}).get("gate_decision", "not_applicable")),
            "gate_reason": str(result.get("gate_info", {}).get("reason", "")),
        }
        for horizon in HORIZONS:
            portfolio_row[f"realized_portfolio_return_{horizon}d"] = _portfolio_forward_return(data, weights, t_pos, horizon)
        portfolio_rows.append(portfolio_row)

    prediction_df = pd.DataFrame(prediction_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)
    summary_df = _build_summary(prediction_df, portfolio_df, cfg) if not portfolio_df.empty else pd.DataFrame()

    prediction_df.to_csv(cfg.output_predictions, index=False)
    portfolio_df.to_csv(cfg.output_portfolio_returns, index=False)
    summary_df.to_csv(cfg.output_summary, index=False)

    print("\n===== WALK-FORWARD BACKTEST REPORT =====")
    if summary_df.empty:
        print("No walk-forward test dates completed.")
        print(f"skipped periods: {skipped}")
        if skip_reasons:
            print("skip reasons:")
            for reason in skip_reasons[:10]:
                print(f"- {reason}")
        return prediction_df, portfolio_df, summary_df
    summary = summary_df.iloc[0]
    print(f"test date range: {summary['test_start']} -> {summary['test_end']}")
    print(f"number of test dates: {summary['number_of_test_dates']}")
    print(f"average selected assets: {summary['average_selected_count']:.2f}")
    print(f"average cash: {summary['average_cash']:.4f}")
    print(f"realized return: {summary['realized_return']:.6f}")
    print(f"realized volatility: {summary['realized_volatility']:.6f}")
    print(f"realized Sharpe: {summary['realized_sharpe']:.6f}")
    print(f"max drawdown: {summary['max_drawdown']:.6f}")
    print(f"Sortino: {summary['Sortino']:.6f}")
    print(f"Calmar: {summary['Calmar']:.6f}")
    for horizon in HORIZONS:
        print(f"hit rate {horizon}d: {summary[f'hit_rate_{horizon}d']:.4f}")
    for horizon in HORIZONS:
        print(f"direction accuracy {horizon}d: {summary[f'direction_accuracy_{horizon}d']:.4f}")
    for horizon in HORIZONS:
        print(f"MAE/RMSE {horizon}d: {summary[f'MAE_{horizon}d']:.6f} / {summary[f'RMSE_{horizon}d']:.6f}")
    print(f"average turnover: {summary['average_turnover']:.4f}")
    print(f"worst drawdown period: {summary['worst_drawdown_period']}")
    print(f"skipped periods: {skipped}")
    if skip_reasons:
        print("skip reasons:")
        for reason in skip_reasons[:10]:
            print(f"- {reason}")
    return prediction_df, portfolio_df, summary_df
