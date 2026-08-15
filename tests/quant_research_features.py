from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def _clean_series(series: pd.Series) -> pd.Series:
    return pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def _sigmoid(value: float) -> float:
    x = float(np.clip(value, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def kalman_local_level(series: pd.Series, process_var: float = 1e-5, measurement_var: float | None = None) -> pd.Series:
    values = _clean_series(series)
    if values.empty:
        return pd.Series(dtype=float)
    if measurement_var is None:
        returns = values.pct_change(fill_method=None).dropna()
        measurement_var = float(max(returns.var(), 1e-6))

    x = float(values.iloc[0])
    p = 1.0
    filtered = []
    for obs in values:
        p = p + process_var
        k = p / (p + measurement_var)
        x = x + k * (float(obs) - x)
        p = (1.0 - k) * p
        filtered.append(x)
    return pd.Series(filtered, index=values.index, dtype=float)


def savitzky_golay_last_slope(series: pd.Series, window: int = 11, poly_order: int = 2) -> float:
    values = _clean_series(series).tail(window)
    if len(values) < poly_order + 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    coeffs = np.polyfit(x, values.to_numpy(dtype=float), deg=poly_order)
    derivative = np.polyder(coeffs)
    slope = float(np.polyval(derivative, x[-1]))
    scale = float(abs(values.iloc[-1])) + EPS
    return float(np.tanh((slope / scale) * 100.0))


def fft_low_frequency_energy(series: pd.Series, keep_ratio: float = 0.15) -> float:
    values = _clean_series(series)
    if len(values) < 16:
        return 0.5
    demeaned = values.to_numpy(dtype=float) - float(values.mean())
    spectrum = np.fft.rfft(demeaned)
    power = np.square(np.abs(spectrum))
    if float(power.sum()) <= EPS:
        return 0.5
    keep_n = max(1, int(len(power) * keep_ratio))
    return float(np.clip(power[:keep_n].sum() / power.sum(), 0.0, 1.0))


def haar_wavelet_energy_ratio(series: pd.Series, levels: int = 3) -> float:
    values = _clean_series(series)
    if len(values) < 2 ** (levels + 1):
        return 0.5
    arr = values.to_numpy(dtype=float)
    n = 2 ** int(np.floor(np.log2(len(arr))))
    coeff = arr[-n:] - float(np.mean(arr[-n:]))
    detail_energy = 0.0
    total_energy = float(np.sum(coeff**2)) + EPS
    for _ in range(levels):
        if len(coeff) < 2:
            break
        avg = (coeff[0::2] + coeff[1::2]) / np.sqrt(2.0)
        detail = (coeff[0::2] - coeff[1::2]) / np.sqrt(2.0)
        detail_energy += float(np.sum(detail**2))
        coeff = avg
    return float(np.clip(detail_energy / total_energy, 0.0, 1.0))


def garch11_forecast_variance(returns: pd.Series) -> float:
    r = _clean_series(returns)
    if len(r) < 30:
        return float(r.var()) if len(r) > 1 else 0.0

    arr = r.to_numpy(dtype=float)
    var = float(np.var(arr))
    best_ll = float("inf")
    best_forecast = var
    alpha_grid = np.array([0.03, 0.05, 0.08, 0.10, 0.12])
    beta_grid = np.array([0.80, 0.85, 0.88, 0.90, 0.94])

    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha + beta >= 0.995:
                continue
            omega = max(var * (1.0 - alpha - beta), EPS)
            h = np.empty_like(arr)
            h[0] = max(var, EPS)
            for i in range(1, len(arr)):
                h[i] = omega + alpha * arr[i - 1] ** 2 + beta * h[i - 1]
                h[i] = max(h[i], EPS)
            ll = float(np.sum(np.log(h) + arr**2 / h))
            if ll < best_ll:
                best_ll = ll
                best_forecast = float(omega + alpha * arr[-1] ** 2 + beta * h[-1])
    return float(max(best_forecast, 0.0))


def egarch11_forecast_variance(returns: pd.Series) -> float:
    r = _clean_series(returns)
    if len(r) < 30:
        return float(r.var()) if len(r) > 1 else 0.0
    arr = r.to_numpy(dtype=float)
    centered = arr - float(np.mean(arr))
    unconditional_var = max(float(np.var(centered)), EPS)
    expected_abs_z = np.sqrt(2.0 / np.pi)
    best_ll = float("inf")
    best_forecast = unconditional_var
    alpha_grid = np.array([0.05, 0.08, 0.12, 0.16])
    gamma_grid = np.array([-0.15, -0.08, 0.0])
    beta_grid = np.array([0.85, 0.90, 0.94])

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            for beta in beta_grid:
                log_h = np.empty_like(centered)
                log_h[0] = np.log(unconditional_var)
                omega = np.log(unconditional_var) * (1.0 - beta)
                for i in range(1, len(centered)):
                    prev_h = np.exp(log_h[i - 1])
                    z = centered[i - 1] / np.sqrt(max(prev_h, EPS))
                    log_h[i] = omega + beta * log_h[i - 1] + alpha * (abs(z) - expected_abs_z) + gamma * z
                    log_h[i] = float(np.clip(log_h[i], -30.0, 5.0))
                h = np.exp(log_h)
                ll = float(np.sum(np.log(h) + centered**2 / h))
                if ll < best_ll:
                    best_ll = ll
                    z_last = centered[-1] / np.sqrt(max(h[-1], EPS))
                    next_log_h = omega + beta * log_h[-1] + alpha * (abs(z_last) - expected_abs_z) + gamma * z_last
                    best_forecast = float(np.exp(np.clip(next_log_h, -30.0, 5.0)))
    return float(max(best_forecast, 0.0))


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 40) -> float:
    values = _clean_series(series)
    if len(values) < max_lag + 5:
        return 0.5
    lags = np.arange(min_lag, max_lag + 1)
    tau = []
    arr = values.to_numpy(dtype=float)
    for lag in lags:
        diffs = arr[lag:] - arr[:-lag]
        std = float(np.std(diffs))
        if std > EPS:
            tau.append(std)
        else:
            tau.append(EPS)
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def ou_half_life(series: pd.Series) -> float:
    values = _clean_series(series)
    if len(values) < 20:
        return 252.0
    y = np.log(values / values.shift(1)).dropna()
    if len(y) < 20:
        return 252.0
    lagged = y.shift(1).dropna()
    delta = (y - y.shift(1)).dropna().reindex(lagged.index)
    x = lagged.to_numpy(dtype=float)
    z = delta.to_numpy(dtype=float)
    denom = float(np.dot(x, x))
    if denom <= EPS:
        return 252.0
    beta = float(np.dot(x, z) / denom)
    if beta >= 0:
        return 252.0
    return float(np.clip(-np.log(2.0) / beta, 1.0, 252.0))


def hawkes_downside_intensity(returns: pd.Series, decay: float = 0.15) -> float:
    r = _clean_series(returns)
    if len(r) < 20:
        return 0.0
    threshold = -2.0 * float(r.std())
    events = (r < threshold).astype(float).to_numpy(dtype=float)
    intensity = 0.0
    for event in events:
        intensity = intensity * np.exp(-decay) + event
    baseline = float(events.mean())
    return float(np.clip(baseline + intensity / max(len(events), 1), 0.0, 1.0))


def shannon_entropy(returns: pd.Series, bins: int = 20) -> float:
    r = _clean_series(returns)
    if len(r) < bins:
        return 0.5
    counts, _ = np.histogram(r.to_numpy(dtype=float), bins=bins)
    probs = counts[counts > 0] / max(counts.sum(), 1)
    entropy = -float(np.sum(probs * np.log(probs + EPS)))
    max_entropy = np.log(bins)
    return float(np.clip(entropy / max(max_entropy, EPS), 0.0, 1.0))


def higuchi_fractal_dimension(series: pd.Series, kmax: int = 8) -> float:
    values = _clean_series(series)
    if len(values) < kmax * 3:
        return 1.5
    x = values.to_numpy(dtype=float)
    lengths = []
    ks = np.arange(1, kmax + 1)
    n = len(x)
    for k in ks:
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            dist = np.abs(np.diff(x[idx])).sum()
            norm = (n - 1) / (len(idx) * k)
            lm.append(dist * norm)
        lengths.append(np.mean(lm) if lm else EPS)
    slope = np.polyfit(np.log(1.0 / ks), np.log(np.maximum(lengths, EPS)), 1)[0]
    return float(np.clip(slope, 1.0, 2.0))


def lyapunov_proxy(series: pd.Series, lag: int = 1) -> float:
    values = _clean_series(series)
    if len(values) < 40:
        return 0.0
    arr = np.log(values / values.shift(1)).dropna().to_numpy(dtype=float)
    if len(arr) < 30:
        return 0.0
    x = arr[:-lag]
    y = arr[lag:]
    dist_now = np.abs(x[:, None] - x[None, :]) + EPS
    np.fill_diagonal(dist_now, np.inf)
    nearest = np.argmin(dist_now, axis=1)
    valid = nearest + lag < len(arr)
    if valid.sum() < 5:
        return 0.0
    div = np.abs(y[valid] - arr[nearest[valid] + lag]) + EPS
    base = dist_now[np.arange(len(x))[valid], nearest[valid]]
    return float(np.clip(np.mean(np.log(div / base)), -5.0, 5.0))


def hill_tail_index_abs(returns: pd.Series, tail_fraction: float = 0.10) -> float:
    r = np.abs(_clean_series(returns).to_numpy(dtype=float))
    if len(r) < 30:
        return 0.0
    r = np.sort(r[r > EPS])
    if len(r) < 10:
        return 0.0
    k = max(5, int(len(r) * tail_fraction))
    tail = r[-k:]
    xmin = max(tail[0], EPS)
    hill = float(np.mean(np.log(tail / xmin + EPS)))
    return float(1.0 / max(hill, EPS))


def gaussian_hmm_two_state(returns: pd.Series, n_iter: int = 25) -> dict[str, float]:
    r = _clean_series(returns)
    if len(r) < 60:
        return {"high_vol_probability": 0.5, "low_vol_mean": 0.0, "high_vol_mean": 0.0}
    x = r.to_numpy(dtype=float)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    means = np.array([q25, q75], dtype=float)
    vars_ = np.array([np.var(x[x <= np.median(x)]), np.var(x[x > np.median(x)])], dtype=float)
    vars_ = np.maximum(vars_, np.var(x) * 0.2 + EPS)
    trans = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)

    for _ in range(n_iter):
        emission = np.vstack([
            np.exp(-0.5 * (x - means[s]) ** 2 / vars_[s]) / np.sqrt(2.0 * np.pi * vars_[s])
            for s in range(2)
        ]).T + EPS
        alpha = np.zeros((len(x), 2), dtype=float)
        scale = np.zeros(len(x), dtype=float)
        alpha[0] = pi * emission[0]
        scale[0] = alpha[0].sum() + EPS
        alpha[0] /= scale[0]
        for t in range(1, len(x)):
            alpha[t] = emission[t] * (alpha[t - 1] @ trans)
            scale[t] = alpha[t].sum() + EPS
            alpha[t] /= scale[t]
        beta = np.ones((len(x), 2), dtype=float)
        for t in range(len(x) - 2, -1, -1):
            beta[t] = trans @ (emission[t + 1] * beta[t + 1])
            beta[t] /= beta[t].sum() + EPS
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + EPS
        weights = gamma.sum(axis=0) + EPS
        means = (gamma.T @ x) / weights
        vars_ = (gamma.T @ ((x[:, None] - means) ** 2)).diagonal() / weights
        vars_ = np.maximum(vars_, EPS)

    high_state = int(np.argmax(vars_))
    return {
        "high_vol_probability": float(gamma[-1, high_state]),
        "low_vol_mean": float(means[int(np.argmin(vars_))]),
        "high_vol_mean": float(means[high_state]),
    }


def compute_asset_quant_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for ticker in prices_df.columns:
        prices = _clean_series(prices_df[ticker])
        returns = np.log(prices / prices.shift(1)).dropna()
        kalman = kalman_local_level(prices)
        if len(kalman) > 2:
            aligned_prices = prices.reindex(kalman.index)
            kalman_residual = ((aligned_prices - kalman) / aligned_prices).replace([np.inf, -np.inf], np.nan).dropna()
            kalman_residual_vol = float(kalman_residual.std()) if len(kalman_residual) > 1 else 0.0
        else:
            kalman_residual_vol = 0.0
        garch_var = garch11_forecast_variance(returns)
        garch_vol = float(np.sqrt(max(garch_var, 0.0)))
        egarch_var = egarch11_forecast_variance(returns)
        egarch_vol = float(np.sqrt(max(egarch_var, 0.0)))
        hurst = hurst_exponent(prices)
        half_life = ou_half_life(prices)
        entropy = shannon_entropy(returns)
        fractal_dim = higuchi_fractal_dimension(prices)
        lyap = lyapunov_proxy(prices)
        hawkes = hawkes_downside_intensity(returns)
        tail_index = hill_tail_index_abs(returns)
        fft_energy = fft_low_frequency_energy(prices)
        wavelet_energy = haar_wavelet_energy_ratio(prices)
        sg_slope = savitzky_golay_last_slope(prices)

        market_quality = (
            0.20 * (1.0 - entropy)
            + 0.20 * (1.0 - np.clip((fractal_dim - 1.0), 0.0, 1.0))
            + 0.20 * (1.0 - _sigmoid(lyap))
            + 0.20 * (1.0 - hawkes)
            + 0.20 * fft_energy
        )
        market_quality = float(np.clip(market_quality, 0.0, 1.0))

        rows.append(
            {
                "ticker": ticker,
                "kalman_residual_vol": kalman_residual_vol,
                "savgol_slope": sg_slope,
                "fft_low_freq_energy": fft_energy,
                "haar_wavelet_energy": wavelet_energy,
                "garch_volatility": garch_vol,
                "egarch_volatility": egarch_vol,
                "hurst_exponent": hurst,
                "ou_half_life": half_life,
                "hawkes_downside_intensity": hawkes,
                "entropy": entropy,
                "fractal_dimension": fractal_dim,
                "lyapunov_proxy": lyap,
                "hill_tail_index": tail_index,
                "quant_market_quality": market_quality,
            }
        )
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
