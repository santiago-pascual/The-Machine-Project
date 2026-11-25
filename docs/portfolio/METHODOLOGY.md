# Methodology audit — source-level findings

Scope: static inspection of representative implementations, not exhaustive verification of every invocation. No operational algorithm changes. No method-specific original unit coverage established; existing tests are primarily import/CLI smoke checks. New tests cover independent artifact metrics only.

| Method | Implementation and assumptions | Limitation / selected-model role |
|---|---|---|
| Kalman | `src/quant_research_features.py:kalman_local_level`, recursive scalar filter | Default measurement noise estimated over entire supplied series; historical intermediate outputs are not prefix invariant. Price and return variance units differ. Target feature dependency needs tracing. |
| GARCH | Same module, grid Gaussian likelihood with variance targeting and alpha+beta < .995 | Discrete approximation, not unconstrained MLE; configured grid model, exact influence on target unverified. |
| EGARCH | Same module, grid log-variance recursion with centered returns and clipped log variance | No parameter intervals; MLE upgrade diagnostic only per config. |
| HMM | Two-state Gaussian forward/backward iterations; terminal gamma; fixed transition matrix and initial probabilities | Not full Baum–Welch (transitions not updated); smoothed intermediate states would leak future information if reused. Four-state model diagnostic only. |
| FFT / Haar | Demeaned spectrum low-bin ratio; dyadic trailing window detail energy | Descriptive features, not predictors by construction; price nonstationarity and boundary alignment need validation. |
| Hurst | Log-lag vs log standard deviation of price differences, clipped [0,1] | Not rescaled-range as older README claims. |
| OU | No-intercept regression of log-return increments on lagged log returns | Discrete approximation on returns, not price-level OU; clipped fallback. |
| Hawkes | Tail events below -2 sample SD, fixed exponential decay and scaled event intensity | Not fitted point process; threshold depends on full supplied window. |
| Entropy / fractal / tails | Histogram Shannon entropy, Higuchi-style scaling, nearest-neighbor Lyapunov proxy and Hill absolute-tail index | Finite-sample, bin/scale dependence, defaults/clipping; no demonstrated causal edge. |
| Triple barrier | `src/triple_barrier_labeling.py`: future close path, volatility-scaled barriers and vertical expiry | Close-based, not intraday fill simulation; future label availability must be respected; excursions include full horizon. |
| Meta-labeling / calibration | `src/ml_core_model.py`, `src/meta_model_walk_forward_validation.py`; logistic, RF, GBM; Platt/isotonic calibration | Full-dataset median before split in core; calibration/test partitions separately imputed in walk-forward; row-index splits may divide dates and overlapping labels. Model selection uses test AUC/F1. |
| Covariance / BL | `src/covariance_estimation.py`, `src/black_litterman_model.py` | Shrinkage and equilibrium-view diagnostics; not direct final Growth weights. |
| Genetic optimizer | `src/portfolio_optimizer.py`: fitness, concentration and constraints, crossover/mutation | Heuristic not proven global optimum; bypassed by Growth allocation. |
| Growth sizing | `src/current_growth_feature_generation.py` ranks raw target, equal weights; min(vol sizing, configured exposure cap, dual trend cap) | 22% target is not guarantee; trend caps can override minimum exposure; no independent end-to-end timing replay performed. |
| Costs | Archived cost/slippage tapes; official estimated ledger | Estimated costs not necessarily booked to gross accounting; no fill or capacity certification. |
| DSR / PBO | `src/anti_overfitting_framework.py`, `archive/full_anti_overfitting_validation.py` | Legacy Sharpe difference and heuristic PBO; later CSCV and approximate p-value require comparable data and dependence controls. |
| Purged OOS | `archive/purged_oos_validation.py` | 30 calendar-day purge/embargo, slices fixed precomputed reconstruction; no fold refit. Cannot guarantee upstream feature/label isolation. |

Research ablation summary (`data/expected_return_ablation_results.csv`) reports raw-target-only return 60.46% vs baseline 38.27%; these are selected historical results, not untouched holdout. Actual selected-model direct dependencies are narrower than full research inventory; indirect target-feature contributions remain unverified.
