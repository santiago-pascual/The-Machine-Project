from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None

EPS = 1e-12
TRADING_DAYS = 252


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_close(ticker: str) -> pd.Series:
    path = Path("yahoo_ohlcv_price_cache") / f"{ticker}.csv"
    df = read_csv(path)
    if df.empty or "Date" not in df.columns:
        return pd.Series(dtype=float, name=ticker)
    col = "Adj Close" if "Adj Close" in df.columns else "Close" if "Close" in df.columns else None
    if col is None:
        return pd.Series(dtype=float, name=ticker)
    out = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=pd.to_datetime(df["Date"], errors="coerce"), name=ticker)
    return out.dropna().sort_index()


def build_feature_data(start_date: str = "2008-01-01") -> pd.DataFrame:
    spy = load_close("SPY")
    qqq = load_close("QQQ")
    if spy.empty or qqq.empty:
        return pd.DataFrame()
    prices = pd.concat({"spy": spy, "qqq": qqq}, axis=1).dropna()
    ret = prices.pct_change(fill_method=None).dropna()
    df = pd.DataFrame(index=ret.index)
    df["spy_return"] = ret["spy"]
    df["qqq_return"] = ret["qqq"]
    df["spy_vol20"] = ret["spy"].rolling(20).std() * math.sqrt(TRADING_DAYS)
    df["qqq_vol20"] = ret["qqq"].rolling(20).std() * math.sqrt(TRADING_DAYS)
    df["spy_mom20"] = prices["spy"].pct_change(20)
    df["qqq_mom20"] = prices["qqq"].pct_change(20)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()
    return df


def standardize(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    mean = df.mean()
    std = df.std(ddof=0).replace(0, 1.0)
    x = ((df - mean) / std).to_numpy(dtype=float)
    return x, mean, std


@dataclass
class HMMFit:
    n_states: int
    seed: int
    converged: bool
    iterations: int
    log_likelihood: float
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    gamma: np.ndarray
    labels: list[str]
    aic: float
    bic: float


def log_gaussian_diag(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    var = np.maximum(variances, EPS)
    d = x.shape[1]
    out = np.empty((x.shape[0], means.shape[0]), dtype=float)
    for k in range(means.shape[0]):
        diff = x - means[k]
        out[:, k] = -0.5 * (d * math.log(2.0 * math.pi) + np.log(var[k]).sum() + (diff * diff / var[k]).sum(axis=1))
    return out


def forward_backward(log_emission: np.ndarray, startprob: np.ndarray, transmat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t_len, k = log_emission.shape
    log_start = np.log(np.maximum(startprob, EPS))
    log_trans = np.log(np.maximum(transmat, EPS))
    log_alpha = np.empty((t_len, k), dtype=float)
    log_beta = np.empty((t_len, k), dtype=float)
    log_alpha[0] = log_start + log_emission[0]
    for t in range(1, t_len):
        log_alpha[t] = log_emission[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
    log_likelihood = float(logsumexp(log_alpha[-1]))
    log_beta[-1] = 0.0
    for t in range(t_len - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + log_emission[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True) + EPS
    xi_sum = np.zeros((k, k), dtype=float)
    for t in range(t_len - 1):
        log_xi = log_alpha[t][:, None] + log_trans + log_emission[t + 1][None, :] + log_beta[t + 1][None, :] - log_likelihood
        xi = np.exp(log_xi)
        xi_sum += xi / (xi.sum() + EPS)
    return gamma, xi_sum, log_alpha, log_likelihood


def initialize_params(x: np.ndarray, n_states: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, d = x.shape
    idx = rng.choice(n, size=n_states, replace=False)
    means = x[idx].copy()
    # Sort rough states by first component for deterministic-ish label order.
    means = means[np.argsort(means[:, 0])]
    variances = np.tile(np.var(x, axis=0) + 1e-3, (n_states, 1))
    trans = np.full((n_states, n_states), 0.05 / max(1, n_states - 1))
    np.fill_diagonal(trans, 0.95)
    start = np.full(n_states, 1.0 / n_states)
    return start, trans, means, variances


def fit_hmm(x: np.ndarray, n_states: int, seed: int, max_iter: int = 500, tol: float = 1e-4) -> HMMFit:
    if GaussianHMM is not None:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=max_iter,
            tol=tol,
            random_state=seed,
            min_covar=1e-5,
            implementation="log",
        )
        model.fit(x)
        ll = float(model.score(x))
        gamma = model.predict_proba(x)
        variances = np.asarray(model.covars_, dtype=float)
        if variances.ndim == 3:
            variances = np.array([np.diag(v) for v in variances], dtype=float)
        n_params = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * x.shape[1]
        aic = 2 * n_params - 2 * ll
        bic = n_params * math.log(len(x)) - 2 * ll
        return HMMFit(
            n_states=n_states,
            seed=seed,
            converged=bool(model.monitor_.converged),
            iterations=int(model.monitor_.iter),
            log_likelihood=ll,
            startprob=np.asarray(model.startprob_, dtype=float),
            transmat=np.asarray(model.transmat_, dtype=float),
            means=np.asarray(model.means_, dtype=float),
            variances=np.maximum(variances, 1e-5),
            gamma=np.asarray(gamma, dtype=float),
            labels=[],
            aic=float(aic),
            bic=float(bic),
        )

    rng = np.random.default_rng(seed)
    start, trans, means, variances = initialize_params(x, n_states, rng)
    prev_ll = -np.inf
    converged = False
    gamma = np.full((len(x), n_states), 1.0 / n_states)
    ll = -np.inf
    iteration = 0
    for iteration in range(1, max_iter + 1):
        log_emit = log_gaussian_diag(x, means, variances)
        gamma, xi_sum, _, ll = forward_backward(log_emit, start, trans)
        weights = gamma.sum(axis=0) + EPS
        start = gamma[0] + EPS
        start /= start.sum()
        trans = xi_sum + EPS
        trans /= trans.sum(axis=1, keepdims=True)
        means = (gamma.T @ x) / weights[:, None]
        for k in range(n_states):
            diff = x - means[k]
            variances[k] = (gamma[:, k][:, None] * diff * diff).sum(axis=0) / weights[k]
        variances = np.maximum(variances, 1e-5)
        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll
    n_params = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * x.shape[1]
    aic = 2 * n_params - 2 * ll
    bic = n_params * math.log(len(x)) - 2 * ll
    return HMMFit(n_states, seed, converged, iteration, ll, start, trans, means, variances, gamma, [], float(aic), float(bic))


def best_fit_for_k(x: np.ndarray, n_states: int, seeds: int, max_iter: int, tol: float) -> tuple[HMMFit, list[HMMFit]]:
    fits = [fit_hmm(x, n_states, seed, max_iter=max_iter, tol=tol) for seed in range(seeds)]
    best = max(fits, key=lambda f: f.log_likelihood)
    return best, fits


def label_states(fit: HMMFit, feature_df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> list[str]:
    original_means = pd.DataFrame(fit.means, columns=feature_df.columns) * std + mean
    score = original_means["spy_return"].to_numpy() + 0.5 * original_means["qqq_return"].to_numpy() - 0.5 * original_means["spy_vol20"].to_numpy()
    vol = original_means["spy_vol20"].to_numpy()
    labels = ["neutral"] * fit.n_states
    risk_off = int(np.argmax(vol - score))
    risk_on = int(np.argmax(score - 0.25 * vol))
    labels[risk_off] = "risk_off"
    if risk_on != risk_off:
        labels[risk_on] = "risk_on"
    for i in range(fit.n_states):
        if labels[i] == "neutral" and fit.n_states > 3:
            labels[i] = "neutral"
    return labels


def decode_states(x: np.ndarray, fit: HMMFit) -> np.ndarray:
    log_emit = log_gaussian_diag(x, fit.means, fit.variances)
    gamma, _, _, _ = forward_backward(log_emit, fit.startprob, fit.transmat)
    return gamma.argmax(axis=1)


def model_summary_rows(fits_by_k: dict[int, tuple[HMMFit, list[HMMFit]]], feature_df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> list[dict[str, object]]:
    rows = []
    for k, (best, fits) in fits_by_k.items():
        labels = label_states(best, feature_df, mean, std)
        original_means = pd.DataFrame(best.means, columns=feature_df.columns) * std + mean
        rows.append(
            {
                "n_states": k,
                "best_seed": best.seed,
                "converged": best.converged,
                "converged_runs": int(sum(f.converged for f in fits)),
                "total_runs": len(fits),
                "best_log_likelihood": best.log_likelihood,
                "AIC": best.aic,
                "BIC": best.bic,
                "mean_state_persistence": float(np.diag(best.transmat).mean()),
                "min_state_persistence": float(np.diag(best.transmat).min()),
                "transition_matrix_stability": float(np.std([np.diag(f.transmat).mean() for f in fits])),
                "state_labels": ",".join(labels),
                "risk_on_state_count": labels.count("risk_on"),
                "neutral_state_count": labels.count("neutral"),
                "risk_off_state_count": labels.count("risk_off"),
                "state_interpretability": "interpretable" if "risk_on" in labels and "risk_off" in labels else "weak",
                "risk_off_spy_vol20": float(original_means.loc[labels.index("risk_off"), "spy_vol20"]) if "risk_off" in labels else np.nan,
                "risk_on_spy_return": float(original_means.loc[labels.index("risk_on"), "spy_return"]) if "risk_on" in labels else np.nan,
            }
        )
    return rows


def transition_rows(fits_by_k: dict[int, tuple[HMMFit, list[HMMFit]]], feature_df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> list[dict[str, object]]:
    rows = []
    for k, (fit, _) in fits_by_k.items():
        labels = label_states(fit, feature_df, mean, std)
        for i in range(k):
            for j in range(k):
                rows.append(
                    {
                        "n_states": k,
                        "from_state": i,
                        "to_state": j,
                        "from_label": labels[i],
                        "to_label": labels[j],
                        "transition_probability": float(fit.transmat[i, j]),
                    }
                )
    return rows


def load_growth_returns() -> pd.DataFrame:
    df = read_csv("growth_crisis_overlay_daily_returns.csv")
    if df.empty or "overlay" not in df.columns:
        return pd.DataFrame()
    df = df.loc[df["overlay"].astype(str).eq("dual_trend_filter")].copy()
    if "window_start" in df.columns:
        ws = pd.to_datetime(df["window_start"], errors="coerce")
        canonical = ws.dropna().min()
        df = df.loc[ws.eq(canonical)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["growth_return"] = pd.to_numeric(df.get("overlay_return"), errors="coerce")
    return df[["date", "growth_return"]].dropna()


def regime_usefulness(feature_df: pd.DataFrame, fit: HMMFit, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    x, _, _ = standardize(feature_df)
    states = decode_states(x, fit)
    labels = label_states(fit, feature_df, mean, std)
    out = feature_df.copy()
    out["state"] = states
    out["regime_label"] = [labels[s] for s in states]
    growth = load_growth_returns()
    if not growth.empty:
        out = out.reset_index(names="date").merge(growth, on="date", how="left").set_index("date")
    out["future_spy_return_20d"] = (1.0 + out["spy_return"]).rolling(20).apply(np.prod, raw=True).shift(-20) - 1.0
    out["future_spy_vol20"] = out["spy_return"].rolling(20).std().shift(-20) * math.sqrt(TRADING_DAYS)
    rows = []
    for (state, label), group in out.groupby(["state", "regime_label"]):
        rows.append(
            {
                "n_states": fit.n_states,
                "state": int(state),
                "regime_label": label,
                "observations": int(len(group)),
                "avg_spy_return": float(group["spy_return"].mean()),
                "avg_spy_vol20": float(group["spy_vol20"].mean()),
                "future_20d_return": float(group["future_spy_return_20d"].mean()),
                "future_20d_vol": float(group["future_spy_vol20"].mean()),
                "growth_avg_return": float(group["growth_return"].mean()) if "growth_return" in group.columns else np.nan,
                "growth_hit_rate": float((group["growth_return"] > 0).mean()) if "growth_return" in group.columns else np.nan,
                "incremental_alpha_beyond_dual_trend_proxy": float(group["growth_return"].mean() - out["growth_return"].mean()) if "growth_return" in group.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def out_of_sample_results(feature_df: pd.DataFrame, n_states: int, seeds: int, max_iter: int, tol: float) -> pd.DataFrame:
    rows = []
    start = feature_df.index.min() + pd.DateOffset(years=3)
    end = feature_df.index.max()
    fold = 1
    while start < end:
        test_end = min(start + pd.DateOffset(months=6), end)
        train = feature_df.loc[feature_df.index < start]
        test = feature_df.loc[(feature_df.index >= start) & (feature_df.index < test_end)]
        if len(train) < 500 or len(test) < 30:
            start = start + pd.DateOffset(months=6)
            continue
        x_train, mean, std = standardize(train)
        x_test = ((test - mean) / std).to_numpy(dtype=float)
        fit, _ = best_fit_for_k(x_train, n_states, seeds, max_iter, tol)
        labels = label_states(fit, train, mean, std)
        states = decode_states(x_test, fit)
        test_tmp = test.copy()
        test_tmp["regime_label"] = [labels[s] for s in states]
        risk_off_rate = float((test_tmp["regime_label"] == "risk_off").mean())
        future_vol_corr = test_tmp["spy_vol20"].corr(test_tmp["spy_return"].rolling(20).std().shift(-20) * math.sqrt(TRADING_DAYS))
        rows.append(
            {
                "fold": fold,
                "n_states": n_states,
                "train_start": train.index.min().date().isoformat(),
                "train_end": train.index.max().date().isoformat(),
                "test_start": test.index.min().date().isoformat(),
                "test_end": test.index.max().date().isoformat(),
                "train_observations": len(train),
                "test_observations": len(test),
                "converged": fit.converged,
                "test_risk_off_rate": risk_off_rate,
                "test_state_switch_rate": float(pd.Series(states).diff().ne(0).mean()),
                "future_volatility_corr_proxy": float(future_vol_corr) if pd.notna(future_vol_corr) else np.nan,
                "classification_stability_proxy": float(np.diag(fit.transmat).mean()),
            }
        )
        fold += 1
        start = start + pd.DateOffset(months=6)
    return pd.DataFrame(rows)


def bootstrap_stability(feature_df: pd.DataFrame, base_fit: HMMFit, mean: pd.Series, std: pd.Series, seeds: int, bootstraps: int, block_size: int, max_iter: int, tol: float) -> pd.DataFrame:
    x, _, _ = standardize(feature_df)
    base_states = decode_states(x, base_fit)
    base_labels = label_states(base_fit, feature_df, mean, std)
    base_label_seq = np.array([base_labels[s] for s in base_states])
    rng = np.random.default_rng(123)
    rows = []
    n = len(feature_df)
    starts = np.arange(0, max(1, n - block_size))
    for b in range(bootstraps):
        idx = []
        while len(idx) < n:
            st = int(rng.choice(starts))
            idx.extend(range(st, min(st + block_size, n)))
        idx = np.array(idx[:n])
        sample = feature_df.iloc[idx].reset_index(drop=True)
        x_boot, boot_mean, boot_std = standardize(sample)
        fit, _ = best_fit_for_k(x_boot, base_fit.n_states, seeds, max_iter, tol)
        x_original_scaled = ((feature_df - boot_mean) / boot_std).to_numpy(dtype=float)
        states = decode_states(x_original_scaled, fit)
        labels = label_states(fit, sample, boot_mean, boot_std)
        label_seq = np.array([labels[s] for s in states])
        agreement = float((label_seq == base_label_seq).mean())
        rows.append(
            {
                "bootstrap_id": b + 1,
                "n_states": base_fit.n_states,
                "state_assignment_agreement": agreement,
                "mean_transition_persistence": float(np.diag(fit.transmat).mean()),
                "converged": fit.converged,
                "log_likelihood": fit.log_likelihood,
            }
        )
    return pd.DataFrame(rows)


def governance(model_cmp: pd.DataFrame, stability: pd.DataFrame, oos: pd.DataFrame, usefulness: pd.DataFrame) -> pd.DataFrame:
    if model_cmp.empty:
        return pd.DataFrame([{"classification": "diagnostic_only", "reason": "missing model comparison"}])
    best_bic = model_cmp.sort_values("BIC").iloc[0]
    best_k = int(best_bic["n_states"])
    best_stab = stability.loc[stability["n_states"].eq(best_k)]
    agreement = float(best_stab["state_assignment_agreement"].mean()) if not best_stab.empty else np.nan
    oos_best = oos.loc[oos["n_states"].eq(best_k)]
    oos_stability = float(oos_best["classification_stability_proxy"].mean()) if not oos_best.empty else np.nan
    useful = usefulness.loc[usefulness["n_states"].eq(best_k)]
    alpha_spread = float(useful["incremental_alpha_beyond_dual_trend_proxy"].max() - useful["incremental_alpha_beyond_dual_trend_proxy"].min()) if "incremental_alpha_beyond_dual_trend_proxy" in useful.columns and not useful.empty else np.nan
    if not np.isfinite(agreement) or agreement < 0.55:
        classification = "unstable"
        reason = f"bootstrap state agreement too low: {agreement}"
    elif best_k == 2 and oos_stability > 0.80 and (not np.isfinite(alpha_spread) or abs(alpha_spread) < 0.002):
        classification = "stable_but_redundant"
        reason = "2-state HMM is stable but adds little incremental alpha beyond dual trend proxy"
    elif oos_stability > 0.75 and np.isfinite(alpha_spread) and abs(alpha_spread) >= 0.002:
        classification = "stable_and_useful"
        reason = f"best BIC states={best_k}; bootstrap agreement={agreement:.3f}; alpha spread={alpha_spread:.5f}"
    else:
        classification = "diagnostic_only"
        reason = f"mixed usefulness/stability; best_k={best_k}; agreement={agreement}; oos_stability={oos_stability}"
    return pd.DataFrame(
        [{
            "classification": classification,
            "best_n_states_by_BIC": best_k,
            "best_n_states_by_AIC": int(model_cmp.sort_values("AIC").iloc[0]["n_states"]),
            "bootstrap_assignment_agreement": agreement,
            "oos_transition_persistence_proxy": oos_stability,
            "incremental_alpha_spread_proxy": alpha_spread,
            "production_changed": False,
            "paper_changed": False,
            "automatic_promotion": False,
            "reason": reason,
        }]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only robust Gaussian HMM regime validation.")
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=20)
    parser.add_argument("--bootstrap-seeds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    feature_df = build_feature_data(args.start_date)
    if feature_df.empty:
        empty = pd.DataFrame([{"classification": "diagnostic_only", "reason": "missing SPY/QQQ OHLCV feature data"}])
        empty.to_csv("hmm_governance.csv", index=False)
        for path in ["hmm_model_comparison.csv", "hmm_transition_matrices.csv", "hmm_state_stability.csv", "hmm_out_of_sample_results.csv"]:
            empty.to_csv(path, index=False)
        print("===== ROBUST HMM REGIME VALIDATION =====")
        print("status: missing SPY/QQQ OHLCV feature data")
        return

    x, mean, std = standardize(feature_df)
    fits_by_k: dict[int, tuple[HMMFit, list[HMMFit]]] = {}
    for k in [2, 3, 4]:
        fits_by_k[k] = best_fit_for_k(x, k, args.seeds, args.max_iter, args.tol)

    model_cmp = pd.DataFrame(model_summary_rows(fits_by_k, feature_df, mean, std))
    trans = pd.DataFrame(transition_rows(fits_by_k, feature_df, mean, std))
    useful_rows = []
    oos_rows = []
    stability_rows = []
    for k, (fit, _) in fits_by_k.items():
        useful_rows.append(regime_usefulness(feature_df, fit, mean, std))
        oos_rows.append(out_of_sample_results(feature_df, k, max(5, args.seeds // 4), args.max_iter, args.tol))
        stability_rows.append(bootstrap_stability(feature_df, fit, mean, std, args.bootstrap_seeds, args.bootstrap_samples, 63, args.max_iter, args.tol))
    usefulness = pd.concat(useful_rows, ignore_index=True, sort=False) if useful_rows else pd.DataFrame()
    oos = pd.concat(oos_rows, ignore_index=True, sort=False) if oos_rows else pd.DataFrame()
    stability = pd.concat(stability_rows, ignore_index=True, sort=False) if stability_rows else pd.DataFrame()
    gov = governance(model_cmp, stability, oos, usefulness)

    model_cmp.to_csv("hmm_model_comparison.csv", index=False)
    trans.to_csv("hmm_transition_matrices.csv", index=False)
    stability.to_csv("hmm_state_stability.csv", index=False)
    oos.to_csv("hmm_out_of_sample_results.csv", index=False)
    usefulness.to_csv("hmm_regime_usefulness.csv", index=False)
    gov.to_csv("hmm_governance.csv", index=False)

    print("===== ROBUST HMM REGIME VALIDATION =====")
    print(f"observations: {len(feature_df)}")
    print(f"best_n_states_by_BIC: {gov.iloc[0]['best_n_states_by_BIC']}")
    print(f"classification: {gov.iloc[0]['classification']}")
    print(f"reason: {gov.iloc[0]['reason']}")
    print("outputs: hmm_model_comparison.csv, hmm_transition_matrices.csv, hmm_state_stability.csv, hmm_out_of_sample_results.csv, hmm_governance.csv")


if __name__ == "__main__":
    main()

