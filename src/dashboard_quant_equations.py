
from __future__ import annotations

import pandas as pd

from dashboard_components import alert_box, source_caption

EQUATIONS = {
    "Expected return": (r"r_e = \frac{P_{target}/P_0 - 1}{T}", "target price, current price, time-to-target", "expected_returns_model.py", "active raw target diagnostic"),
    "Volatility targeting": (r"E_t = \min(\max(\sigma^*/\hat\sigma_t, 0.40), 0.60, C_{trend})", "target volatility, realized portfolio volatility, exposure caps", "growth_volatility_targeting_fresh.csv", "active official risk overlay"),
    "Sharpe": (r"S = \frac{\bar r - r_f}{\sigma_r}\sqrt{252}", "mean return, risk-free rate, volatility", "performance reports", "diagnostic/governance"),
    "Sortino": (r"Sortino = \frac{\bar r - r_f}{\sigma_{down}}\sqrt{252}", "downside deviation", "performance reports", "diagnostic/governance"),
    "Calmar": (r"Calmar = \frac{CAGR}{|MaxDD|}", "CAGR and maximum drawdown", "performance reports", "diagnostic/governance"),
    "VaR / CVaR": (r"VaR_\alpha = q_\alpha(r), \quad CVaR_\alpha = E[r | r \le VaR_\alpha]", "return quantiles and tail mean", "risk/monte carlo", "diagnostic"),
    "HHI": (r"HHI = \sum_i w_i^2", "portfolio weights", "official holdings", "diagnostic concentration"),
    "Turnover": (r"TO_t = \frac{1}{2}\sum_i |w_{i,t} - w_{i,t-1}|", "old and new weights", "official actions", "execution diagnostic"),
    "Black-Litterman": (r"\mu_{BL} = \pi + \tau\Sigma P^T(P\tau\Sigma P^T + \Omega)^{-1}(q-P\pi)", "prior, covariance, views, uncertainty", "black_litterman_model.py", "research diagnostic only"),
    "GARCH": (r"\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2", "volatility recursion", "garch diagnostics", "diagnostic"),
    "EGARCH": (r"\log\sigma_t^2 = \omega + \beta\log\sigma_{t-1}^2 + \alpha(|z_{t-1}|-E|z|)+\gamma z_{t-1}", "log volatility recursion", "garch diagnostics", "diagnostic"),
    "Hurst": (r"E[R/S] \propto n^H", "rescaled range scaling", "feature store", "diagnostic"),
    "Entropy": (r"H(X) = -\sum_i p_i\log p_i", "state probabilities", "feature store", "diagnostic"),
    "Ornstein-Uhlenbeck": (r"dX_t = \theta(\mu-X_t)dt + \sigma dW_t", "mean reversion speed and level", "feature store", "diagnostic"),
}


def render_equation_explorer(st, data: dict[str, pd.DataFrame]) -> dict[str, object]:
    st.markdown("#### Model Equations Explorer")
    module = st.selectbox("Equation module", list(EQUATIONS.keys()), key="qlab_equation")
    eq, variables, source, role = EQUATIONS[module]
    st.latex(eq)
    rows = pd.DataFrame([
        {"field": "variables", "value": variables},
        {"field": "input file/module", "value": source},
        {"field": "role", "value": role},
        {"field": "interpretation", "value": "Mathematical diagnostic display. It does not imply active allocation impact unless role says active."},
    ])
    st.dataframe(rows, width="stretch", hide_index=True)
    if "diagnostic" in role and "active" not in role:
        alert_box(st, "Research/diagnostic equation: not part of official Growth allocation logic unless explicitly marked active.", "info")
    source_caption(st, source, role)
    return {"surface": "equation_explorer", "status": "available", "detail": module}
