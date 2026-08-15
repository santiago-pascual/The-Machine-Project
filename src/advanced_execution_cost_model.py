from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CAPITAL_SCENARIOS = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000]
PARTICIPATION_LIMITS = [0.01, 0.025, 0.05, 0.10]
IMPACT_Y_VALUES = [0.5, 1.0, 1.5]
EPS = 1e-12


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError, ValueError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_ohlcv(ticker: str) -> pd.DataFrame:
    path = Path("yahoo_ohlcv_price_cache") / f"{ticker.upper()}.csv"
    df = read_csv(path)
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["price"] = df[price_col]
    df["daily_return"] = df["price"].pct_change(fill_method=None)
    df["dollar_volume"] = df["Close"] * df["Volume"] if {"Close", "Volume"}.issubset(df.columns) else np.nan
    df["adv20"] = df["dollar_volume"].rolling(20).mean()
    df["adv60"] = df["dollar_volume"].rolling(60).mean()
    df["sigma20"] = df["daily_return"].rolling(20).std()
    if {"High", "Low"}.issubset(df.columns):
        hl = np.log(df["High"] / df["Low"]).replace([np.inf, -np.inf], np.nan)
        df["high_low_spread_proxy"] = (2.0 * (np.exp(hl.abs()) - 1.0) / (1.0 + np.exp(hl.abs()))).clip(lower=0.0, upper=0.25)
        beta = (np.log(df["High"] / df["Low"]) ** 2).rolling(2).sum()
        high2 = df["High"].rolling(2).max()
        low2 = df["Low"].rolling(2).min()
        gamma = np.log(high2 / low2) ** 2
        alpha = ((np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        cs_spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        df["corwin_schultz_spread"] = cs_spread.clip(lower=0.0, upper=0.25)
    else:
        df["high_low_spread_proxy"] = np.nan
        df["corwin_schultz_spread"] = np.nan
    return df.dropna(subset=["date"]).sort_values("date")


def get_metric_on_or_before(ohlcv: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    if ohlcv.empty:
        return {
            "adv20": np.nan,
            "adv60": np.nan,
            "sigma20": np.nan,
            "spread_proxy": np.nan,
            "price": np.nan,
            "volume": np.nan,
        }
    hist = ohlcv.loc[ohlcv["date"] <= date]
    if hist.empty:
        return {
            "adv20": np.nan,
            "adv60": np.nan,
            "sigma20": np.nan,
            "spread_proxy": np.nan,
            "price": np.nan,
            "volume": np.nan,
        }
    row = hist.iloc[-1]
    hist_tail = hist.tail(20)
    spread = hist_tail.get("corwin_schultz_spread", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().median()
    if not np.isfinite(spread) or spread <= 0:
        spread = hist_tail.get("high_low_spread_proxy", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().median()
    # Daily high-low ranges are not quoted bid-ask spreads; cap the proxy to avoid
    # treating volatile single-name ranges as executable spread.
    if np.isfinite(spread):
        spread = min(max(float(spread), 0.0002), 0.02)
    return {
        "adv20": float(row.get("adv20", np.nan)),
        "adv60": float(row.get("adv60", np.nan)),
        "sigma20": float(row.get("sigma20", np.nan)),
        "spread_proxy": float(spread) if np.isfinite(spread) else np.nan,
        "price": float(row.get("price", np.nan)),
        "volume": float(row.get("Volume", np.nan)),
    }


def load_trade_source() -> tuple[pd.DataFrame, str]:
    final = read_csv("growth_final_selection_daily_returns.csv")
    if not final.empty and {"date", "candidate", "candidate_exposure", "selected_tickers"}.issubset(final.columns):
        d = final.loc[final["candidate"].astype(str).eq("growth_champion_v3")].copy()
        if "window_start" in d.columns:
            ws = pd.to_datetime(d["window_start"], errors="coerce")
            canonical = ws.dropna().min()
            d = d.loc[ws.eq(canonical)].copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date")
        prev_weights: dict[str, float] = {}
        rows = []
        for _, row in d.iterrows():
            tickers = [t.strip().upper() for t in str(row.get("selected_tickers", "")).split(",") if t.strip()]
            exposure_value = pd.to_numeric(pd.Series([row.get("candidate_exposure", 0.0)]), errors="coerce").iloc[0]
            exposure = float(exposure_value) if np.isfinite(exposure_value) else 0.0
            target_weight = exposure / len(tickers) if tickers else 0.0
            target = {t: target_weight for t in tickers}
            for ticker in sorted(set(prev_weights) | set(target)):
                old = float(prev_weights.get(ticker, 0.0))
                new = float(target.get(ticker, 0.0))
                change = new - old
                if abs(change) <= 1e-10:
                    action = "HOLD"
                elif old == 0 and new > 0:
                    action = "BUY"
                elif old > 0 and new == 0:
                    action = "SELL"
                elif change > 0:
                    action = "INCREASE"
                else:
                    action = "REDUCE"
                rows.append(
                    {
                        "date": row["date"],
                        "ticker": ticker,
                        "weight": new,
                        "previous_weight": old,
                        "weight_change": change,
                        "action": action,
                    }
                )
            prev_weights = target
        return pd.DataFrame(rows), "growth_final_selection_daily_returns.csv::growth_champion_v3_reconstructed_actions"

    rec = read_csv("reconstructed_growth_long_horizon_trades.csv")
    if not rec.empty and {"date", "ticker", "weight", "previous_weight"}.issubset(rec.columns):
        df = rec.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["weight_change"] = pd.to_numeric(df["weight"], errors="coerce") - pd.to_numeric(df["previous_weight"], errors="coerce").fillna(
            0.0
        )
        df["action"] = np.where(df["weight_change"] > 0, "BUY_OR_INCREASE", np.where(df["weight_change"] < 0, "SELL_OR_REDUCE", "HOLD"))
        return df.dropna(subset=["date", "ticker"]), "reconstructed_growth_long_horizon_trades.csv"
    live = read_csv("growth_candidate_action_signals.csv")
    if not live.empty and {"date", "ticker", "weight_change"}.issubset(live.columns):
        live["date"] = pd.to_datetime(live["date"], errors="coerce")
        return live.dropna(subset=["date", "ticker"]), "growth_candidate_action_signals.csv"
    return pd.DataFrame(), "missing"


def load_return_source() -> tuple[pd.DataFrame, str]:
    df = read_csv("growth_crisis_overlay_daily_returns.csv")
    if not df.empty and {"date", "overlay", "overlay_return"}.issubset(df.columns):
        d = df.loc[df["overlay"].astype(str).eq("dual_trend_filter")].copy()
        if "window_start" in d.columns:
            ws = pd.to_datetime(d["window_start"], errors="coerce")
            canonical = ws.dropna().min()
            d = d.loc[ws.eq(canonical)].copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["gross_return"] = pd.to_numeric(d["overlay_return"], errors="coerce")
        return d[["date", "gross_return"]].dropna(), "growth_crisis_overlay_daily_returns.csv::dual_trend_filter"
    rec = read_csv("reconstructed_growth_long_horizon_daily_returns.csv")
    if not rec.empty and {"date", "return"}.issubset(rec.columns):
        rec["date"] = pd.to_datetime(rec["date"], errors="coerce")
        rec["gross_return"] = pd.to_numeric(rec["return"], errors="coerce")
        return rec[["date", "gross_return"]].dropna(), "reconstructed_growth_long_horizon_daily_returns.csv"
    return pd.DataFrame(), "missing"


def compute_trade_costs(
    trades: pd.DataFrame, portfolio_value: float, y_value: float, commission_bps: float, min_fee: float
) -> pd.DataFrame:
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    active = trades.loc[~trades["ticker"].astype(str).str.upper().eq("CASH")].copy()
    active["weight_change"] = pd.to_numeric(active["weight_change"], errors="coerce").fillna(0.0)
    active = active.loc[active["weight_change"].abs() > 1e-9]
    for _, trade in active.iterrows():
        ticker = str(trade["ticker"]).upper()
        if ticker not in cache:
            cache[ticker] = load_ohlcv(ticker)
        metrics = get_metric_on_or_before(cache[ticker], pd.Timestamp(trade["date"]))
        order_value = abs(float(trade["weight_change"])) * portfolio_value
        adv = metrics["adv60"] if np.isfinite(metrics["adv60"]) and metrics["adv60"] > 0 else metrics["adv20"]
        sigma = metrics["sigma20"] if np.isfinite(metrics["sigma20"]) and metrics["sigma20"] > 0 else 0.02
        spread = metrics["spread_proxy"] if np.isfinite(metrics["spread_proxy"]) and metrics["spread_proxy"] > 0 else 0.002
        participation = order_value / max(adv, EPS) if np.isfinite(adv) else np.nan
        commission = max(min_fee, order_value * commission_bps / 10000.0)
        spread_cost = order_value * spread * 0.5
        slippage_rate = 0.25 * sigma * math.sqrt(max(participation, 0.0)) if np.isfinite(participation) else np.nan
        slippage_cost = order_value * slippage_rate if np.isfinite(slippage_rate) else np.nan
        impact_rate = y_value * sigma * math.sqrt(max(participation, 0.0)) if np.isfinite(participation) else np.nan
        impact_cost = order_value * impact_rate if np.isfinite(impact_rate) else np.nan
        total_cost = np.nansum([commission, spread_cost, slippage_cost, impact_cost])
        rows.append(
            {
                "date": pd.Timestamp(trade["date"]).date().isoformat(),
                "ticker": ticker,
                "action": trade.get("action", ""),
                "weight_change": float(trade["weight_change"]),
                "order_value": order_value,
                "portfolio_value": portfolio_value,
                "adv20": metrics["adv20"],
                "adv60": metrics["adv60"],
                "participation_rate": participation,
                "daily_volatility": sigma,
                "spread_proxy": spread,
                "commission_cost": commission,
                "spread_cost": spread_cost,
                "slippage_rate": slippage_rate,
                "slippage_cost": slippage_cost,
                "impact_Y": y_value,
                "impact_rate": impact_rate,
                "impact_cost": impact_cost,
                "total_cost": total_cost,
                "total_cost_bps_of_order": total_cost / max(order_value, EPS) * 10000.0,
                "missing_liquidity_data": not np.isfinite(participation),
            }
        )
    return pd.DataFrame(rows)


def perf_metrics(returns: pd.Series, dates: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"total_return": np.nan, "CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "max_drawdown": np.nan}
    d = pd.to_datetime(dates.loc[r.index], errors="coerce")
    years = max((d.max() - d.min()).days / 365.25, 1e-9) if len(d.dropna()) > 1 else 1.0
    total = float((1 + r).prod() - 1)
    cagr = float((1 + total) ** (1 / years) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 2 else np.nan
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(downside) > 2 and downside.std(ddof=1) > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {"total_return": total, "CAGR": cagr, "volatility": vol, "Sharpe": sharpe, "Sortino": sortino, "max_drawdown": float(dd.min())}


def apply_costs_to_returns(returns: pd.DataFrame, costs: pd.DataFrame, portfolio_value: float) -> pd.DataFrame:
    daily_cost = costs.groupby("date", as_index=False)["total_cost"].sum()
    daily_cost["date"] = pd.to_datetime(daily_cost["date"], errors="coerce")
    out = returns.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.merge(daily_cost, on="date", how="left")
    out["total_cost"] = out["total_cost"].fillna(0.0)
    out["cost_return_drag"] = out["total_cost"] / portfolio_value
    out["net_return"] = out["gross_return"] - out["cost_return_drag"]
    out["net_equity"] = (1 + out["net_return"].fillna(0)).cumprod()
    out["gross_equity"] = (1 + out["gross_return"].fillna(0)).cumprod()
    return out


def capacity_analysis(costs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if costs.empty:
        return pd.DataFrame()
    for capital in CAPITAL_SCENARIOS:
        scaled = costs.copy()
        base_value = scaled["portfolio_value"].replace(0, np.nan).median()
        scale = capital / base_value if np.isfinite(base_value) and base_value > 0 else 1.0
        scaled_participation = scaled["participation_rate"] * scale
        for limit in PARTICIPATION_LIMITS:
            rows.append(
                {
                    "capital": capital,
                    "participation_limit": limit,
                    "max_participation": float(scaled_participation.max()),
                    "median_participation": float(scaled_participation.median()),
                    "trades_over_limit": int((scaled_participation > limit).sum()),
                    "pct_trades_over_limit": float((scaled_participation > limit).mean()),
                    "capacity_status": "safe"
                    if (scaled_participation > limit).mean() < 0.01
                    else "caution"
                    if (scaled_participation > limit).mean() < 0.10
                    else "capacity_limited",
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    trades, trade_source = load_trade_source()
    returns, return_source = load_return_source()
    if trades.empty or returns.empty:
        empty = pd.DataFrame([{"status": "missing trades or returns", "trade_source": trade_source, "return_source": return_source}])
        for path in [
            "advanced_execution_costs.csv",
            "market_impact_results.csv",
            "capacity_analysis.csv",
            "after_costs_equity_curves.csv",
            "execution_governance.csv",
        ]:
            empty.to_csv(path, index=False)
        print("===== ADVANCED EXECUTION COST MODEL =====")
        print("status: missing trades or returns")
        return

    all_costs = []
    all_equity = []
    market_rows = []
    for y in IMPACT_Y_VALUES:
        costs = compute_trade_costs(trades, args.portfolio_value, y, args.commission_bps, args.minimum_fee)
        costs["scenario"] = f"impact_Y_{y}"
        all_costs.append(costs)
        eq = apply_costs_to_returns(returns, costs, args.portfolio_value)
        eq["scenario"] = f"impact_Y_{y}"
        all_equity.append(eq)
        gross = perf_metrics(eq["gross_return"], eq["date"])
        net = perf_metrics(eq["net_return"], eq["date"])
        total_cost = float(costs["total_cost"].sum()) if not costs.empty else 0.0
        years = max((eq["date"].max() - eq["date"].min()).days / 365.25, 1e-9)
        market_rows.append(
            {
                "scenario": f"impact_Y_{y}",
                "impact_Y": y,
                "commission_bps": args.commission_bps,
                "minimum_fee": args.minimum_fee,
                "trade_source": trade_source,
                "return_source": return_source,
                "trades": (len(costs)),
                "gross_CAGR": gross["CAGR"],
                "net_CAGR": net["CAGR"],
                "gross_Sharpe": gross["Sharpe"],
                "net_Sharpe": net["Sharpe"],
                "gross_Sortino": gross["Sortino"],
                "net_Sortino": net["Sortino"],
                "gross_max_drawdown": gross["max_drawdown"],
                "net_max_drawdown": net["max_drawdown"],
                "total_cost_drag_dollars": total_cost,
                "annual_cost_drag_dollars": total_cost / years,
                "annual_cost_drag_pct_of_aum": total_cost / years / args.portfolio_value,
                "average_turnover_proxy": float(
                    trades.assign(abs_wc=pd.to_numeric(trades["weight_change"], errors="coerce").abs())
                    .groupby("date")["abs_wc"]
                    .sum()
                    .mean()
                ),
                "missing_liquidity_rate": float(costs["missing_liquidity_data"].mean()) if not costs.empty else np.nan,
            }
        )

    advanced_costs = pd.concat(all_costs, ignore_index=True, sort=False)
    equity = pd.concat(all_equity, ignore_index=True, sort=False)
    market = pd.DataFrame(market_rows)
    capacity = capacity_analysis(advanced_costs.loc[advanced_costs["impact_Y"].eq(1.0)].copy())

    safe_500k = capacity.loc[(capacity["capital"].eq(500_000)) & (capacity["participation_limit"].eq(0.05))]
    net_realistic = market.loc[market["impact_Y"].eq(1.0)]
    if (
        not safe_500k.empty
        and safe_500k.iloc[0]["capacity_status"] == "safe"
        and not net_realistic.empty
        and net_realistic.iloc[0]["net_Sharpe"] > 1.0
    ):
        classification = "robust_medium_capital"
    elif not capacity.loc[(capacity["capital"].eq(100_000)) & (capacity["participation_limit"].eq(0.05))].empty and capacity.loc[
        (capacity["capital"].eq(100_000)) & (capacity["participation_limit"].eq(0.05))
    ].iloc[0]["capacity_status"] in {"safe", "caution"}:
        classification = "robust_small_capital"
    elif not safe_500k.empty and safe_500k.iloc[0]["capacity_status"] == "capacity_limited":
        classification = "capacity_limited"
    else:
        classification = "fails_execution_reality"

    gov = pd.DataFrame(
        [
            {
                "classification": classification,
                "portfolio_value_assumption": args.portfolio_value,
                "commission_bps": args.commission_bps,
                "minimum_fee": args.minimum_fee,
                "best_realistic_net_sharpe_Y1": float(net_realistic.iloc[0]["net_Sharpe"]) if not net_realistic.empty else np.nan,
                "annual_cost_drag_pct_Y1": float(net_realistic.iloc[0]["annual_cost_drag_pct_of_aum"])
                if not net_realistic.empty
                else np.nan,
                "production_changed": False,
                "paper_changed": False,
                "reason": "Advanced costs applied to historical action changes with spread/slippage/square-root impact.",
            }
        ]
    )

    advanced_costs.to_csv("advanced_execution_costs.csv", index=False)
    market.to_csv("market_impact_results.csv", index=False)
    capacity.to_csv("capacity_analysis.csv", index=False)
    equity.to_csv("after_costs_equity_curves.csv", index=False)
    gov.to_csv("execution_governance.csv", index=False)

    print("===== ADVANCED EXECUTION COST MODEL =====")
    print(f"trade_source: {trade_source}")
    print(f"return_source: {return_source}")
    print(f"cost_rows: {len(advanced_costs)}")
    print(f"classification: {classification}")
    if not net_realistic.empty:
        print(f"net_CAGR_Y1: {net_realistic.iloc[0]['net_CAGR']}")
        print(f"net_Sharpe_Y1: {net_realistic.iloc[0]['net_Sharpe']}")
        print(f"annual_cost_drag_pct_Y1: {net_realistic.iloc[0]['annual_cost_drag_pct_of_aum']}")
    print(
        "outputs: advanced_execution_costs.csv, market_impact_results.csv, capacity_analysis.csv, after_costs_equity_curves.csv, execution_governance.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only advanced execution cost and capacity model.")
    parser.add_argument("--portfolio-value", type=float, default=100_000.0)
    parser.add_argument("--commission-bps", type=float, default=2.0)
    parser.add_argument("--minimum-fee", type=float, default=0.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
