# July whitepaper review — 25 November 2026

Scope: compare the two supplied historical documents with the November repository audit. No production logic, state, or historical evidence changed. This is not a line-by-line revalidation or exhaustive bibliography verification.

| Historical section | Finding / current interpretation |
|---|---|
| Comprehensive §1.1 | Misidentifies legacy baseline + old regime gate as frozen Growth. 847 reported sample count is not 847 independent portfolio returns; tape has 220 dates. One-session compounding yields 21.32%, five-session 78.99%. |
| §1.2 | Momentum-proxy reconstruction explicitly lacks exact parity; causal and survivorship-free claims not certified. Overlapping windows cannot establish independent robustness. Text describing -29.6% drawdown as smaller than SPY -22.7% is numerically incorrect in magnitude. |
| §4.1 | Full supplied-window default measurement-noise estimate; historical causality depends on invocation window. |
| §4.6 | Lagged-difference scaling, not classical rescaled-range Hurst estimation. |
| §4.9 | Histogram entropy measures marginal dispersion, not temporal predictability. |
| §4.13 | Fixed transitions in audited two-state implementation; not fully fitted Baum-Welch. |
| §7 | Chronological ordering alone does not prevent leakage. Full-sample median imputation, independent partition imputation and test-set model selection found in research paths. |
| §§9–13, 21 | Base optimizer and Growth must be separated. Volume I §§6–7 correctly documents Growth equal weighting and non-use of original optimizer. |
| §17 | Historical drawdown overlay must not be conflated with operational soft retention rule. |
| §18 | Legacy Sharpe difference and heuristic PBO are not standard probability estimates; later CSCV and purged-return studies are distinct and limited. |
| §22 / Volume I §16 | Core DSR and PBO references checked against publication records below; other entries not independently verified. |
| Volume I §§10–11 | Historical PASS labels are snapshot assertions, not current accounting or operational certification. |

## Reference corrections

- Bailey & López de Prado (2014), The Deflated Sharpe Ratio, Journal of Portfolio Management 40(5), 94–107. https://doi.org/10.3905/jpm.2014.40.5.094 . The standard DSR uses a probability transformation; the legacy code's Sharpe difference is not that statistic.
- Bailey, Borwein, López de Prado & Zhu, The Probability of Backtest Overfitting, Journal of Computational Finance, DOI https://doi.org/10.21314/JCF.2016.322 . Publication record: https://scholarworks.wmich.edu/math_pubs/42/ (2017; online/DOI record 2016). Volume I lists 2014 with volume 20(4); distinguish preprint from final publication.

## Publication and provenance

Both originals are marked internal and use institutional-style titles; these are historical labels, not affiliation. Author approval is needed before public release. Text pattern scan found no common GitHub/AWS/private-key patterns, email addresses or Windows user paths; embedded images and all metadata have not been exhaustively cleared. DOCX has author/last-editor metadata; PDF has producer metadata. Preserve originals privately if disclosure is not desired. The PDF hash matches the user's local SHA-256. The uploaded DOCX hash (f4b058bf8e0b2f0ced951b0a708a4ceb08f992e3572ecf1e9ebb9e1e27e7c1d0) differs from the local reported 643a435ce36d91c59a20dc3a6b3a0592b14dcd847a3fe5c867bb65b260d49e01; reason unknown. Verify the intended file before publishing. No claim of complete security clearance or bibliography validation.
