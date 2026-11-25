# The Machine Project — Executive research summary

**25 November 2026 | Independent, AI-assisted quantitative equity research and engineering | Santiago Pascual**

## Research question

Can concentrated cross-sectional equity selection retain historical performance while limiting exposure, drawdown and research-selection risk? The project explores this question with a Python research platform and simulated execution infrastructure, not a claim of validated investment alpha or institutional affiliation.

## Architecture and selected configuration

The base engine explores return estimates, statistical features, regime classification, meta-labeling, covariance shrinkage, Black-Litterman diagnostics and genetic optimization. The separate frozen Growth paper configuration uses raw-target ranking, eligibility and retention rules, up to four holdings, equal weights within exposure, a 22% annualized volatility target with 60-session lookback, a 60% exposure ceiling and SPY/QQQ 200-session trend caps of 25%, 40% or 60%. Scheduled rebalances occur every five sessions. The original genetic optimizer does not determine final Growth weights. The committed configuration disables broker connectivity and real orders; the running environment was not inspected.

## Historical portfolio comparison

Recomputed from `data/growth_head_to_head_daily_returns.csv` in the November audit, aligned on 219 approximately weekly observations (2022-01-10 to 2026-05-15). Annualization uses 52 observations/year, sample standard deviation and zero risk-free rate. Figures are historical research, not confirmed net returns or independent untouched validation.

| Variant | Total return | Annualized geometric | Annualized volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Growth v1 uncapped | 391.12% | 45.92% | 29.97% | 1.414 | -24.23% |
| Growth v1 cap 60 | 235.49% | 33.30% | 22.19% | 1.409 | -19.76% |
| Legacy comparator | 79.59% | 14.92% | 13.29% | 1.114 | -17.18% |
| SPY | 54.73% | 10.92% | 16.15% | 0.723 | -25.36% |
| QQQ | 76.49% | 14.44% | 21.34% | 0.739 | -34.61% |

The capped candidate was selected historically to reduce risk at the expense of return. The later dual-trend v3 reconstruction reduced drawdown relative to v2 (-24.66% versus -34.68% over 878 recorded observations) with lower cumulative return. These are approximate reconstructions, not production parity; overlapping start windows are dependent. The July 21.32% result refers to the legacy baseline plus old regime gate and one-session return compounding across sparse dates, not the frozen Growth model's continuously invested performance.

## Governance and evidence limits

The historical 14,649 logged trials are not independent trials. The old deflated Sharpe field is a Sharpe difference, not a DSR probability. The heuristic PBO score is not an estimated probability. Later CSCV partition flags show 77.14%, 67.86%, and 63.20% below-median OOS frequencies for 8, 10 and 12 blocks, respectively; mixed frequencies and unequal coverage constrain interpretation. Later purged validation slices precomputed returns rather than refitting the entire model with label-aware splits. Confirmed research-path preprocessing leakage and test-set model selection, unverified point-in-time universe coverage, transaction costs and execution timing prevent scientific certification. The isolated audit verifies stored arithmetic, not the whole historical pipeline.

## Paper trading and next steps

The archived official forward snapshot has ten sparse records, July 10–August 13, 2026: gross compounded +7.91%, estimated net +7.38%; the terminal net field conflicts with the net return series. No annualized forward metric or current live performance claim is made. Research and paper simulation remain distinct from real capital. A future isolated phase should correct leakage, establish point-in-time data and execution parity, reconcile accounting and conduct pre-registered forward validation without altering running services.

See [audit](../portfolio/AUDIT.md), [methodology](../portfolio/METHODOLOGY.md), [metrics](../portfolio/portfolio_metrics.csv) and [historical document review](REVIEW.md). This is an independent project, not investment advice.
