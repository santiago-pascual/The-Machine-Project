from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from quant_research_features import egarch11_forecast_variance, garch11_forecast_variance


TRADING_DAYS = 252
EPS = 1e-12
DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",
    "RBLX",
    "INTC",
    "BABA",
    "SE",
]


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_price_series(ticker: str, cache_dir: str = "yahoo_ohlcv_price_cache") -> pd.Series:
    path = Path(cache_dir) / f"{ticker.upper()}.csv"
    df = read_csv(path)
    if df.empty or "Date" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close" if "Close" in df.columns else None
    if price_col is None:
        return pd.Series(dtype=float, name=ticker)
    dates = pd.to_datetime(df["Date"], errors="coerce")
    prices = pd.to_numeric(df[price_col], errors="coerce")
    out = pd.Series(prices.to_numpy(), index=dates, name=ticker).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def clean_returns(prices: pd.Series) -> pd.Series:
    r = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    return r.loc[r.abs() < 2.0]


def normal_nll_from_h(r: np.ndarray, h: np.ndarray) -> float:
    h = np.maximum(h, EPS)
    return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(h) + (r * r) / h))


def student_t_nll_from_h(r: np.ndarray, h: np.ndarray, nu: float) -> float:
    if nu <= 2.05:
        return 1e12
    h = np.maximum(h, EPS)
    z2 = (r * r) / h
    # Standardized Student-t with unit variance.
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi) - 0.5 * np.log(h)
    ll = c - ((nu + 1.0) / 2.0) * np.log1p(z2 / (nu - 2.0))
    return float(-np.sum(ll))


def garch_variances(params: np.ndarray, r: np.ndarray) -> np.ndarray:
    omega, alpha, beta = params[:3]
    h = np.empty_like(r, dtype=float)
    h[0] = max(float(np.var(r)), EPS)
    for i in range(1, len(r)):
        h[i] = omega + alpha * r[i - 1] ** 2 + beta * h[i - 1]
        h[i] = max(h[i], EPS)
    return h


def egarch_variances(params: np.ndarray, r: np.ndarray) -> np.ndarray:
    omega, alpha, gamma, beta = params[:4]
    centered = r - float(np.mean(r))
    log_h = np.empty_like(centered, dtype=float)
    log_h[0] = np.log(max(float(np.var(centered)), EPS))
    expected_abs_z = math.sqrt(2.0 / math.pi)
    for i in range(1, len(centered)):
        z = centered[i - 1] / math.sqrt(max(math.exp(log_h[i - 1]), EPS))
        log_h[i] = omega + beta * log_h[i - 1] + alpha * (abs(z) - expected_abs_z) + gamma * z
        log_h[i] = float(np.clip(log_h[i], -30.0, 5.0))
    return np.exp(log_h)


def fit_garch_mle(r: pd.Series, dist: str) -> dict[str, float | bool | str]:
    arr = r.to_numpy(dtype=float)
    var = max(float(np.var(arr)), EPS)

    def objective(x: np.ndarray) -> float:
        omega, alpha, beta = x[:3]
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        h = garch_variances(x, arr)
        return student_t_nll_from_h(arr, h, x[3]) if dist == "student_t" else normal_nll_from_h(arr, h)

    x0 = np.array([var * 0.05, 0.08, 0.88, 8.0]) if dist == "student_t" else np.array([var * 0.05, 0.08, 0.88])
    bounds = [(EPS, var * 10.0), (1e-6, 0.60), (1e-6, 0.998)] + ([(2.1, 80.0)] if dist == "student_t" else [])
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400, "ftol": 1e-9})
    x = res.x
    h = garch_variances(x, arr)
    forecast = float(max(x[0] + x[1] * arr[-1] ** 2 + x[2] * h[-1], EPS))
    k = len(x)
    ll = -float(objective(x))
    return {
        "model": "GARCH",
        "distribution": dist,
        "omega": float(x[0]),
        "alpha": float(x[1]),
        "beta": float(x[2]),
        "gamma": np.nan,
        "nu": float(x[3]) if dist == "student_t" else np.nan,
        "forecast_variance": forecast,
        "converged": bool(res.success),
        "optimizer_message": str(res.message),
        "stationary": bool(x[1] + x[2] < 0.999),
        "positive_constraints": bool(x[0] > 0 and x[1] >= 0 and x[2] >= 0),
        "persistence": float(x[1] + x[2]),
        "log_likelihood": ll,
        "AIC": float(2 * k - 2 * ll),
        "BIC": float(k * math.log(len(arr)) - 2 * ll),
    }


def fit_egarch_mle(r: pd.Series, dist: str) -> dict[str, float | bool | str]:
    arr = r.to_numpy(dtype=float)
    var = max(float(np.var(arr)), EPS)

    def objective(x: np.ndarray) -> float:
        beta = x[3]
        if abs(beta) >= 0.999:
            return 1e12
        h = egarch_variances(x, arr)
        return student_t_nll_from_h(arr - float(np.mean(arr)), h, x[4]) if dist == "student_t" else normal_nll_from_h(arr - float(np.mean(arr)), h)

    x0 = np.array([math.log(var) * 0.05, 0.10, -0.05, 0.92, 8.0]) if dist == "student_t" else np.array([math.log(var) * 0.05, 0.10, -0.05, 0.92])
    bounds = [(-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0), (-0.998, 0.998)] + ([(2.1, 80.0)] if dist == "student_t" else [])
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-9})
    x = res.x
    h = egarch_variances(x, arr)
    centered = arr - float(np.mean(arr))
    z_last = centered[-1] / math.sqrt(max(h[-1], EPS))
    expected_abs_z = math.sqrt(2.0 / math.pi)
    next_log_h = x[0] + x[3] * math.log(max(h[-1], EPS)) + x[1] * (abs(z_last) - expected_abs_z) + x[2] * z_last
    forecast = float(max(math.exp(float(np.clip(next_log_h, -30.0, 5.0))), EPS))
    k = len(x)
    ll = -float(objective(x))
    return {
        "model": "EGARCH",
        "distribution": dist,
        "omega": float(x[0]),
        "alpha": float(x[1]),
        "beta": float(x[3]),
        "gamma": float(x[2]),
        "nu": float(x[4]) if dist == "student_t" else np.nan,
        "forecast_variance": forecast,
        "converged": bool(res.success),
        "optimizer_message": str(res.message),
        "stationary": bool(abs(x[3]) < 0.999),
        "positive_constraints": True,
        "persistence": float(abs(x[3])),
        "log_likelihood": ll,
        "AIC": float(2 * k - 2 * ll),
        "BIC": float(k * math.log(len(arr)) - 2 * ll),
    }


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_sf_1df(x: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - norm_cdf(math.sqrt(max(x, 0.0))))))


def kupiec_test(exceedances: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    x = int(exceedances.sum())
    n = int(exceedances.count())
    if n <= 0:
        return np.nan, np.nan
    phat = min(max(x / n, EPS), 1.0 - EPS)
    p = alpha
    lr = -2.0 * ((n - x) * math.log((1.0 - p) / (1.0 - phat)) + x * math.log(p / phat))
    return float(lr), float(chi2_sf_1df(lr))


def christoffersen_independence_test(exceedances: pd.Series) -> tuple[float, float]:
    e = exceedances.astype(int).dropna().to_numpy()
    if len(e) < 3:
        return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(e[:-1], e[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    pi01 = n01 / max(1, n00 + n01)
    pi11 = n11 / max(1, n10 + n11)

    def ll(p01: float, p11: float) -> float:
        p01 = min(max(p01, EPS), 1.0 - EPS)
        p11 = min(max(p11, EPS), 1.0 - EPS)
        return n00 * math.log(1.0 - p01) + n01 * math.log(p01) + n10 * math.log(1.0 - p11) + n11 * math.log(p11)

    lr = -2.0 * (ll(pi, pi) - ll(pi01, pi11))
    return float(lr), float(chi2_sf_1df(lr))


def validation_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in forecasts.groupby("model_name"):
        g = group.dropna(subset=["forecast_variance", "realized_variance_next"])
        if g.empty:
            continue
        h = np.maximum(g["forecast_variance"].to_numpy(dtype=float), EPS)
        rv = np.maximum(g["realized_variance_next"].to_numpy(dtype=float), EPS)
        errors = h - rv
        var95 = -1.6448536269514722 * np.sqrt(h)
        exceed = g["realized_return_next"].to_numpy(dtype=float) < var95
        kupiec_lr, kupiec_p = kupiec_test(pd.Series(exceed), 0.05)
        christ_lr, christ_p = christoffersen_independence_test(pd.Series(exceed))
        rows.append(
            {
                "model_name": model_name,
                "observations": int(len(g)),
                "QLIKE": float(np.mean(np.log(h) + rv / h)),
                "MSE": float(np.mean(errors**2)),
                "MAE": float(np.mean(np.abs(errors))),
                "forecast_realized_corr": float(pd.Series(h).corr(pd.Series(rv))),
                "VaR_95_exceedance_rate": float(np.mean(exceed)),
                "Kupiec_LR": kupiec_lr,
                "Kupiec_p_value": kupiec_p,
                "Christoffersen_independence_LR": christ_lr,
                "Christoffersen_independence_p_value": christ_p,
            }
        )
    return pd.DataFrame(rows).sort_values(["QLIKE", "MAE"], ascending=[True, True])


def choose_tickers(limit: int) -> list[str]:
    tickers = []
    alloc = read_csv("current_growth_candidate_allocation.csv")
    if not alloc.empty and "ticker" in alloc.columns:
        tickers.extend([t for t in alloc["ticker"].dropna().astype(str).str.upper().tolist() if t != "CASH"])
    tickers.extend(DEFAULT_TICKERS)
    seen = []
    for t in tickers:
        if t not in seen and (Path("yahoo_ohlcv_price_cache") / f"{t}.csv").exists():
            seen.append(t)
    return seen[:limit]


def walk_forward(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    param_rows = []
    forecast_rows = []
    tickers = args.tickers.split(",") if args.tickers else choose_tickers(args.max_tickers)
    for ticker in tickers:
        prices = load_price_series(ticker)
        returns = clean_returns(prices)
        if len(returns) < args.train_window + 30:
            continue
        eval_dates = returns.index[args.train_window :: args.step_days]
        if args.max_eval_dates > 0:
            eval_dates = eval_dates[-args.max_eval_dates :]
        for dt in eval_dates:
            loc = returns.index.get_loc(dt)
            train = returns.iloc[loc - args.train_window : loc]
            if len(train) < args.min_train_observations:
                continue
            realized_next = float(returns.iloc[loc])
            realized_var = realized_next * realized_next
            base_models = {
                "grid_garch": garch11_forecast_variance(train),
                "grid_egarch": egarch11_forecast_variance(train),
                "realized_vol_60d": float(train.tail(60).var()),
            }
            for model_name, forecast_var in base_models.items():
                forecast_rows.append(
                    {
                        "date": dt.date().isoformat(),
                        "ticker": ticker,
                        "model_name": model_name,
                        "forecast_variance": float(max(forecast_var, EPS)),
                        "forecast_volatility": float(math.sqrt(max(forecast_var, EPS))),
                        "realized_return_next": realized_next,
                        "realized_variance_next": realized_var,
                        "estimation_method": "existing_grid_or_realized_benchmark",
                    }
                )
            fitters: list[tuple[str, Callable[[pd.Series, str], dict[str, float | bool | str]], str]] = [
                ("mle_garch_normal", fit_garch_mle, "normal"),
                ("mle_garch_student_t", fit_garch_mle, "student_t"),
                ("mle_egarch_normal", fit_egarch_mle, "normal"),
                ("mle_egarch_student_t", fit_egarch_mle, "student_t"),
            ]
            for model_name, fitter, dist in fitters:
                try:
                    fit = fitter(train, dist)
                    fit.update({"date": dt.date().isoformat(), "ticker": ticker, "model_name": model_name, "train_observations": len(train)})
                    param_rows.append(fit)
                    forecast_rows.append(
                        {
                            "date": dt.date().isoformat(),
                            "ticker": ticker,
                            "model_name": model_name,
                            "forecast_variance": float(fit["forecast_variance"]),
                            "forecast_volatility": float(math.sqrt(float(fit["forecast_variance"]))),
                            "realized_return_next": realized_next,
                            "realized_variance_next": realized_var,
                            "estimation_method": "maximum_likelihood_scipy",
                            "converged": fit["converged"],
                            "stationary": fit["stationary"],
                            "persistence": fit["persistence"],
                            "AIC": fit["AIC"],
                            "BIC": fit["BIC"],
                        }
                    )
                except Exception as exc:
                    param_rows.append(
                        {
                            "date": dt.date().isoformat(),
                            "ticker": ticker,
                            "model_name": model_name,
                            "model": model_name,
                            "distribution": dist,
                            "converged": False,
                            "optimizer_message": f"fit_error: {exc}",
                            "train_observations": len(train),
                        }
                    )
    return pd.DataFrame(param_rows), pd.DataFrame(forecast_rows)


def comparison_table(validation: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rows = []
    convergence = params.groupby("model_name")["converged"].mean() if not params.empty and "converged" in params.columns else pd.Series(dtype=float)
    for _, row in validation.iterrows():
        model_name = row["model_name"]
        rows.append(
            {
                **row.to_dict(),
                "convergence_rate": float(convergence.get(model_name, np.nan)),
                "model_family": "existing_grid" if str(model_name).startswith("grid") else "realized_vol" if model_name == "realized_vol_60d" else "mle",
            }
        )
    return pd.DataFrame(rows).sort_values(["QLIKE", "MAE"], ascending=[True, True])


def governance(comparison: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        classification = "retain_as_diagnostic_only"
        reason = "no validation observations"
        best = ""
    else:
        best = str(comparison.iloc[0]["model_name"])
        grid = comparison.loc[comparison["model_name"].eq("grid_garch")]
        best_row = comparison.iloc[0]
        grid_qlike = float(grid["QLIKE"].iloc[0]) if not grid.empty else np.nan
        improvement = (grid_qlike - float(best_row["QLIKE"])) / abs(grid_qlike) if np.isfinite(grid_qlike) and grid_qlike != 0 else np.nan
        mle_best = str(best).startswith("mle")
        convergence = float(best_row.get("convergence_rate", np.nan))
        if mle_best and np.isfinite(improvement) and improvement > 0.02 and (np.isnan(convergence) or convergence > 0.80):
            classification = "replace_with_mle"
            reason = f"best MLE improves QLIKE vs grid by {improvement:.2%}"
        elif mle_best:
            classification = "retain_as_diagnostic_only"
            reason = f"MLE best but improvement/convergence not material enough; improvement={improvement}"
        else:
            classification = "keep_grid_model"
            reason = f"best model is {best}; MLE does not materially dominate grid"
    return pd.DataFrame(
        [{
            "classification": classification,
            "best_model": best,
            "production_changed": False,
            "paper_changed": False,
            "parameter_tuning": False,
            "walk_forward_only": True,
            "arch_package_used": False,
            "mle_backend": "scipy.optimize",
            "reason": reason,
        }]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only GARCH/EGARCH MLE walk-forward validation.")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers. Default uses current growth holdings plus liquid defaults.")
    parser.add_argument("--max-tickers", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=756)
    parser.add_argument("--min-train-observations", type=int, default=500)
    parser.add_argument("--step-days", type=int, default=21)
    parser.add_argument("--max-eval-dates", type=int, default=8)
    args = parser.parse_args()

    params, forecasts = walk_forward(args)
    validation = validation_metrics(forecasts)
    comparison = comparison_table(validation, params)
    gov = governance(comparison, params)

    params.to_csv("garch_mle_parameters.csv", index=False)
    forecasts.to_csv("garch_mle_forecasts.csv", index=False)
    comparison.to_csv("garch_model_comparison.csv", index=False)
    validation.to_csv("volatility_forecast_validation.csv", index=False)
    gov.to_csv("garch_governance.csv", index=False)

    print("===== GARCH/EGARCH MLE UPGRADE =====")
    print(f"parameter_rows: {len(params)}")
    print(f"forecast_rows: {len(forecasts)}")
    print(f"best_model: {gov.iloc[0]['best_model'] if not gov.empty else ''}")
    print(f"classification: {gov.iloc[0]['classification'] if not gov.empty else ''}")
    print(f"reason: {gov.iloc[0]['reason'] if not gov.empty else ''}")
    print("outputs: garch_mle_parameters.csv, garch_mle_forecasts.csv, garch_model_comparison.csv, volatility_forecast_validation.csv, garch_governance.csv")


if __name__ == "__main__":
    main()

