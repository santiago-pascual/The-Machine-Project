from __future__ import annotations

from pathlib import Path
import contextlib
import io

import numpy as np
import pandas as pd

from risk_metrics import compute_return_risk_metrics
from triple_barrier_labeling import generate_triple_barrier_labels


DEFAULT_OUTPUT_FILE = "barrier_parameter_optimization.csv"
DEFAULT_HORIZONS = [5, 10, 20, 30]
DEFAULT_TP_MULTIPLES = [0.75, 1.0, 1.25, 1.5, 2.0]
DEFAULT_SL_MULTIPLES = [0.75, 1.0, 1.25, 1.5, 2.0]


def _load_predictions(path: str | Path = "walk_forward_predictions.csv") -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    data = pd.read_csv(file_path)
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["ticker"] = data["ticker"].astype(str)
    return data.dropna(subset=["date", "ticker"])


def _performance_block(labels: pd.DataFrame, subset_name: str) -> dict[str, float | str]:
    if labels.empty:
        return {
            "subset": subset_name,
            "sample_size": 0,
            "TP_rate": np.nan,
            "SL_rate": np.nan,
            "timeout_rate": np.nan,
            "TP_minus_SL": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "max_drawdown_proxy": np.nan,
            "avg_time_to_exit": np.nan,
        }

    realized = pd.to_numeric(labels["realized_return_at_barrier"], errors="coerce").dropna()
    risk = compute_return_risk_metrics(realized)
    tp_rate = float((labels["first_touch_type"] == "take_profit").mean())
    sl_rate = float((labels["first_touch_type"] == "stop_loss").mean())
    timeout_rate = float((labels["first_touch_type"] == "vertical_timeout").mean())
    sharpe = float(realized.mean() / realized.std(ddof=1)) if len(realized) > 1 and float(realized.std(ddof=1)) > 0 else 0.0
    return {
        "subset": subset_name,
        "sample_size": int(len(labels)),
        "TP_rate": tp_rate,
        "SL_rate": sl_rate,
        "timeout_rate": timeout_rate,
        "TP_minus_SL": tp_rate - sl_rate,
        "avg_return": float(realized.mean()) if not realized.empty else np.nan,
        "median_return": float(realized.median()) if not realized.empty else np.nan,
        "Sharpe": sharpe,
        "Sortino": float(risk["sortino_ratio"]),
        "max_drawdown_proxy": float(risk["max_drawdown"]),
        "avg_time_to_exit": float(labels["time_to_first_touch"].mean()),
    }


def run_barrier_parameter_optimization(
    *,
    prices_df: pd.DataFrame,
    predictions_path: str | Path = "walk_forward_predictions.csv",
    horizons: list[int] | None = None,
    tp_multiples: list[float] | None = None,
    sl_multiples: list[float] | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
    min_sample_warning: int = 100,
) -> pd.DataFrame:
    predictions = _load_predictions(predictions_path)
    if prices_df.empty or predictions.empty:
        result = pd.DataFrame()
        result.to_csv(output_path, index=False)
        print("\n===== BARRIER PARAMETER OPTIMIZATION =====")
        print("No prices or walk-forward predictions available.")
        return result

    horizons = horizons or DEFAULT_HORIZONS
    tp_multiples = tp_multiples or DEFAULT_TP_MULTIPLES
    sl_multiples = sl_multiples or DEFAULT_SL_MULTIPLES
    rows: list[dict[str, object]] = []

    for horizon in horizons:
        for tp_multiple in tp_multiples:
            for sl_multiple in sl_multiples:
                with contextlib.redirect_stdout(io.StringIO()):
                    labels = generate_triple_barrier_labels(
                        prices_df=prices_df,
                        predictions_df=predictions,
                        horizons=(int(horizon),),
                        tp_multiple=float(tp_multiple),
                        sl_multiple=float(sl_multiple),
                        output_path=Path(output_path).with_suffix(".tmp.csv"),
                    )
                selected_labels = labels[labels["selected"].astype(bool)] if not labels.empty and "selected" in labels.columns else labels.iloc[0:0]
                for subset_name, subset in [("universe", labels), ("selected_only", selected_labels)]:
                    metrics = _performance_block(subset, subset_name=subset_name)
                    rows.append(
                        {
                            "horizon": int(horizon),
                            "tp_multiple": float(tp_multiple),
                            "sl_multiple": float(sl_multiple),
                            **metrics,
                        }
                    )

    tmp_path = Path(output_path).with_suffix(".tmp.csv")
    tmp_path.unlink(missing_ok=True)
    result = pd.DataFrame(rows)
    if result.empty:
        result.to_csv(output_path, index=False)
        return result

    selected_rank = result[result["subset"] == "selected_only"].copy()
    selected_rank = selected_rank.sort_values(
        ["Sharpe", "avg_return", "TP_minus_SL", "avg_time_to_exit", "SL_rate"],
        ascending=[False, False, False, True, True],
    )
    result = result.sort_values(
        ["subset", "Sharpe", "avg_return", "TP_minus_SL", "avg_time_to_exit", "SL_rate"],
        ascending=[True, False, False, False, True, True],
    )
    result.to_csv(output_path, index=False)

    print("\n===== BARRIER PARAMETER OPTIMIZATION =====")
    print(f"grid searched: horizons={horizons}, tp_multiples={tp_multiples}, sl_multiples={sl_multiples}")
    print(f"total configurations: {len(horizons) * len(tp_multiples) * len(sl_multiples)}")
    if int(selected_rank["sample_size"].max()) < min_sample_warning:
        print(f"[WARNING] Sample size is small: max selected-only sample = {int(selected_rank['sample_size'].max())}")
    display_cols = [
        "horizon",
        "tp_multiple",
        "sl_multiple",
        "sample_size",
        "TP_rate",
        "SL_rate",
        "timeout_rate",
        "TP_minus_SL",
        "avg_return",
        "median_return",
        "Sharpe",
        "Sortino",
        "avg_time_to_exit",
    ]
    print("\n===== BEST BARRIER CONFIGURATIONS =====")
    print(selected_rank[display_cols].head(20).to_string(index=False))
    return result
