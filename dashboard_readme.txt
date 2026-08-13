Growth Champion Final Dashboard
================================

Purpose
-------
Local read-only dashboard for Growth Champion Final:
- raw_target_return_exact
- soft_exit_rule
- volatility_target_22pct
- exposure_cap_60
- dual_trend_filter

Run
---
Preferred:
streamlit run dashboard_app.py

If Streamlit or Plotly are missing:
pip install streamlit plotly

Fallback:
python dashboard_app.py

If Flask is installed, the fallback starts a simple read-only web dashboard on:
http://127.0.0.1:8501

Optional fallback dependency:
pip install flask

Files Read
----------
- growth_candidate_paper_performance.csv
- growth_candidate_paper_state.csv
- growth_candidate_paper_trades.csv
- growth_candidate_paper_monitor.csv
- growth_paper_governance_report.csv
- current_growth_features.csv
- current_growth_candidate_allocation.csv
- cedear_growth_universe.csv
- model_ticker_to_cedear_map.csv
- growth_final_selection_daily_returns.csv
- growth_final_selection_results.csv
- growth_final_cost_slippage_results.csv
- growth_final_after_costs_vs_benchmarks.csv
- reconstructed_growth_long_horizon_daily_returns.csv
- production_parity_growth_daily_returns.csv
- production_parity_growth_results.csv

Safety
------
The dashboard is read-only.
It does not place orders.
It does not connect to a broker.
It does not modify production configuration.
It does not modify paper trading state.

The only optional file write is:
- growth_strategy_animation.html

This is created only from the Visualizer export action when Plotly is available.

Known Limitations
-----------------
- If Streamlit is not installed, `streamlit run dashboard_app.py` will fail before the script starts.
- If Plotly is not installed, charts fall back to Streamlit native charts.
- If neither Streamlit nor Flask is installed, the script prints installation instructions.
- SPY/QQQ live benchmark overlay depends on available local benchmark files.
- Paper trading history is currently short, so governance may remain WARMUP.
