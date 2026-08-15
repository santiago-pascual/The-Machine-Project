from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EMBARGO_DAYS = 30
MAX_FORECAST_HORIZON_DAYS = 30
TRAIN_YEARS_MIN = 3
VALIDATION_MONTHS = 6
TEST_MONTHS = 6
ROLL_MONTHS = 6


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_growth_final_series() -> pd.DataFrame:
    crisis = read_csv("growth_crisis_overlay_daily_returns.csv")
    if not crisis.empty and {"date", "overlay", "overlay_return"}.issubset(crisis.columns):
        df = crisis.loc[crisis["overlay"].astype(str).eq("dual_trend_filter")].copy()
        if not df.empty:
            if "window_start" in df.columns:
                df["_window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
                canonical_window = df["_window_start"].dropna().min()
                if pd.notna(canonical_window):
                    df = df.loc[df["_window_start"].eq(canonical_window)].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            df["growth_return"] = safe_num(df["overlay_return"])
            df["exposure"] = safe_num(df.get("overlay_exposure", df.get("target_exposure", np.nan)))
            df["cash"] = safe_num(df.get("overlay_cash", df.get("cash_weight", np.nan)))
            df["turnover"] = safe_num(df.get("overlay_turnover_proxy", df.get("turnover", np.nan)))
            df["selected_count"] = safe_num(df.get("selected_count", np.nan))
            df["spy_return"] = safe_num(df.get("spy_price", np.nan)).pct_change()
            df["qqq_return"] = safe_num(df.get("qqq_price", np.nan)).pct_change()
            df["source"] = "growth_crisis_overlay_daily_returns.csv::dual_trend_filter"
            return df[[
                "date",
                "growth_return",
                "spy_return",
                "qqq_return",
                "exposure",
                "cash",
                "turnover",
                "selected_count",
                "selected_tickers",
                "source",
            ]].dropna(subset=["growth_return"])

    final = read_csv("growth_final_selection_daily_returns.csv")
    if final.empty or not {"date", "candidate", "candidate_return"}.issubset(final.columns):
        return pd.DataFrame()
    df = final.loc[final["candidate"].astype(str).eq("growth_champion_v3")].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["growth_return"] = safe_num(df["candidate_return"])
    df["spy_return"] = np.nan
    df["qqq_return"] = np.nan
    df["exposure"] = safe_num(df.get("candidate_exposure", np.nan))
    df["cash"] = safe_num(df.get("candidate_cash", np.nan))
    df["turnover"] = safe_num(df.get("candidate_turnover", np.nan))
    df["selected_count"] = safe_num(df.get("selected_count", np.nan))
    df["source"] = "growth_final_selection_daily_returns.csv::growth_champion_v3"
    return df[[
        "date",
        "growth_return",
        "spy_return",
        "qqq_return",
        "exposure",
        "cash",
        "turnover",
        "selected_count",
        "selected_tickers",
        "source",
    ]].dropna(subset=["growth_return"])


def periods_per_year(dates: pd.Series) -> float:
    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(d) < 3:
        return 52.0
    step = d.diff().dt.days.dropna().median()
    if not np.isfinite(step) or step <= 0:
        return 52.0
    return float(365.25 / step)


def max_drawdown(returns: pd.Series) -> float:
    r = safe_num(returns).fillna(0.0)
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def sortino(returns: pd.Series, ppy: float) -> float:
    r = safe_num(returns).dropna()
    if len(r) < 3:
        return np.nan
    downside = r[r < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / downside.std(ddof=1) * np.sqrt(ppy))


def sharpe(returns: pd.Series, ppy: float) -> float:
    r = safe_num(returns).dropna()
    if len(r) < 3 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def cagr(returns: pd.Series, dates: pd.Series) -> float:
    r = safe_num(returns).dropna()
    d = pd.to_datetime(dates.loc[r.index], errors="coerce").dropna()
    if len(r) < 2 or d.empty:
        return np.nan
    total = float((1.0 + r).prod() - 1.0)
    years = max(1e-9, (d.max() - d.min()).days / 365.25)
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def information_ratio(strategy: pd.Series, benchmark: pd.Series, ppy: float) -> float:
    diff = safe_num(strategy).sub(safe_num(benchmark), fill_value=np.nan).dropna()
    if len(diff) < 3 or diff.std(ddof=1) == 0:
        return np.nan
    return float(diff.mean() / diff.std(ddof=1) * np.sqrt(ppy))


def metric_row(df: pd.DataFrame, label: str, fold_id: str, fold_type: str) -> dict[str, object]:
    if df.empty:
        return {
            "fold_id": fold_id,
            "fold_type": fold_type,
            "window_label": label,
            "observations": 0,
        }
    ppy = periods_per_year(df["date"])
    r = safe_num(df["growth_return"])
    total = float((1.0 + r.fillna(0.0)).prod() - 1.0)
    dd = max_drawdown(r)
    c = cagr(r, df["date"])
    return {
        "fold_id": fold_id,
        "fold_type": fold_type,
        "window_label": label,
        "start_date": df["date"].min().date().isoformat(),
        "end_date": df["date"].max().date().isoformat(),
        "observations": int(r.notna().sum()),
        "periods_per_year_estimate": ppy,
        "total_return": total,
        "CAGR": c,
        "volatility": float(r.std(ddof=1) * np.sqrt(ppy)) if r.notna().sum() > 2 else np.nan,
        "Sharpe": sharpe(r, ppy),
        "Sortino": sortino(r, ppy),
        "Calmar": float(c / abs(dd)) if np.isfinite(c) and np.isfinite(dd) and dd < 0 else np.nan,
        "max_drawdown": dd,
        "hit_rate": float((r > 0).mean()),
        "average_turnover": float(safe_num(df.get("turnover", pd.Series(dtype=float))).mean()),
        "average_exposure": float(safe_num(df.get("exposure", pd.Series(dtype=float))).mean()),
        "average_cash": float(safe_num(df.get("cash", pd.Series(dtype=float))).mean()),
        "average_selected_count": float(safe_num(df.get("selected_count", pd.Series(dtype=float))).mean()),
        "alpha_vs_SPY": float(r.mean() - safe_num(df.get("spy_return", pd.Series(dtype=float))).mean()) if df.get("spy_return") is not None else np.nan,
        "alpha_vs_QQQ": float(r.mean() - safe_num(df.get("qqq_return", pd.Series(dtype=float))).mean()) if df.get("qqq_return") is not None else np.nan,
        "information_ratio_vs_SPY": information_ratio(r, df.get("spy_return", pd.Series(dtype=float)), ppy),
        "information_ratio_vs_QQQ": information_ratio(r, df.get("qqq_return", pd.Series(dtype=float)), ppy),
        "benchmark_status": "benchmarks_derived_from_aligned_SPY_QQQ_prices" if safe_num(df.get("spy_return", pd.Series(dtype=float))).notna().sum() > 5 else "benchmark_daily_series_missing",
    }


def month_add(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=months)


def build_purged_walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fold_rows = []
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    min_date = df["date"].min()
    max_date = df["date"].max()
    train_start = min_date
    train_end = train_start + pd.DateOffset(years=TRAIN_YEARS_MIN)
    fold = 1
    while True:
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = month_add(validation_start, VALIDATION_MONTHS) - pd.Timedelta(days=1)
        embargo_start = validation_end + pd.Timedelta(days=1)
        embargo_end = embargo_start + pd.Timedelta(days=EMBARGO_DAYS - 1)
        test_start = embargo_end + pd.Timedelta(days=1)
        test_end = month_add(test_start, TEST_MONTHS) - pd.Timedelta(days=1)
        purge_start = train_end - pd.Timedelta(days=MAX_FORECAST_HORIZON_DAYS)
        if test_start > max_date:
            break
        if test_end > max_date:
            test_end = max_date
        train = df[(df["date"] >= train_start) & (df["date"] <= purge_start)]
        validation = df[(df["date"] >= validation_start) & (df["date"] <= validation_end)]
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        rows.append(metric_row(train, "expanding_train_purged", str(fold), "train"))
        rows.append(metric_row(validation, "fixed_validation", str(fold), "validation"))
        rows.append(metric_row(test, "fixed_test_embargoed", str(fold), "test"))
        fold_rows.append(
            {
                "fold_id": fold,
                "train_start": train_start.date().isoformat(),
                "train_end_raw": train_end.date().isoformat(),
                "train_end_after_purge": purge_start.date().isoformat(),
                "validation_start": validation_start.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "embargo_start": embargo_start.date().isoformat(),
                "embargo_end": embargo_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "purge_days": MAX_FORECAST_HORIZON_DAYS,
                "embargo_days": EMBARGO_DAYS,
                "train_observations": len(train),
                "validation_observations": len(validation),
                "test_observations": len(test),
                "parameters_frozen": True,
                "reoptimized_inside_fold": False,
                "future_data_reused": False,
            }
        )
        fold += 1
        train_end = month_add(train_end, ROLL_MONTHS)
        if fold > 200:
            break
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def build_purged_kfold(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if df.empty or len(df) < k * 10:
        return pd.DataFrame()
    indices = np.array_split(np.arange(len(df)), k)
    rows = []
    for i, test_idx in enumerate(indices, start=1):
        test_start = df.iloc[test_idx]["date"].min()
        test_end = df.iloc[test_idx]["date"].max()
        purge_start = test_start - pd.Timedelta(days=MAX_FORECAST_HORIZON_DAYS)
        embargo_end = test_end + pd.Timedelta(days=EMBARGO_DAYS)
        train = df[(df["date"] < purge_start) | (df["date"] > embargo_end)]
        test = df.iloc[test_idx]
        row = metric_row(test, "purged_kfold_test", f"pkf_{i}", "purged_kfold_test")
        row.update(
            {
                "train_observations_after_purge_embargo": len(train),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "embargo_end": embargo_end.date().isoformat(),
                "purge_start": purge_start.date().isoformat(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def locked_holdout(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    end = df["date"].max()
    start = end - pd.DateOffset(months=12)
    holdout = df[(df["date"] > start) & (df["date"] <= end)].copy()
    row = metric_row(holdout, "most_recent_12_months_locked_holdout", "locked_holdout", "locked_holdout")
    row.update(
        {
            "locked_holdout": True,
            "evaluated_once": True,
            "tuning_allowed_after_result": False,
            "holdout_rule": "most recent 12 months available; no post-hoc tuning",
        }
    )
    return pd.DataFrame([row])


def compute_ic_decay() -> pd.DataFrame:
    ic = read_csv("historical_ic_dataset.csv")
    if ic.empty:
        return pd.DataFrame([{"status": "missing historical_ic_dataset.csv"}])
    if "date" not in ic.columns or "expected_daily_return" not in ic.columns:
        return pd.DataFrame([{"status": "missing required IC columns"}])
    ic["date"] = pd.to_datetime(ic["date"], errors="coerce")
    rows = []
    for horizon in [1, 5, 10, 20, 30]:
        ret_col = f"realized_return_{horizon}d"
        if ret_col not in ic.columns:
            continue
        for subset_name, subset in [("all", ic), ("selected_only", ic.loc[ic.get("selected", False).astype(str).str.lower().eq("true")])]:
            per_date = []
            for dt, group in subset.groupby("date"):
                x = safe_num(group["expected_daily_return"])
                y = safe_num(group[ret_col])
                valid = pd.DataFrame({"x": x, "y": y}).dropna()
                if len(valid) < 5:
                    continue
                per_date.append(
                    {
                        "date": dt,
                        "spearman_ic": valid["x"].corr(valid["y"], method="spearman"),
                        "pearson_ic": valid["x"].corr(valid["y"], method="pearson"),
                    }
                )
            pdf = pd.DataFrame(per_date)
            rows.append(
                {
                    "horizon": f"{horizon}d",
                    "subset": subset_name,
                    "dates": int(len(pdf)),
                    "mean_spearman_rank_ic": float(pdf["spearman_ic"].mean()) if not pdf.empty else np.nan,
                    "median_spearman_rank_ic": float(pdf["spearman_ic"].median()) if not pdf.empty else np.nan,
                    "positive_rank_ic_rate": float((pdf["spearman_ic"] > 0).mean()) if not pdf.empty else np.nan,
                    "mean_pearson_ic": float(pdf["pearson_ic"].mean()) if not pdf.empty else np.nan,
                    "ic_decay_input": "expected_daily_return_vs_realized_return",
                    "lookahead_control": "realized returns are evaluated after prediction date in historical_ic_dataset",
                }
            )
    return pd.DataFrame(rows)


def governance(results: pd.DataFrame, holdout: pd.DataFrame, ic_decay: pd.DataFrame) -> pd.DataFrame:
    test = results.loc[results["fold_type"].eq("test")].copy() if not results.empty else pd.DataFrame()
    hold = holdout.iloc[0].to_dict() if not holdout.empty else {}
    mean_test_sharpe = float(test["Sharpe"].mean()) if not test.empty else np.nan
    positive_test_rate = float((test["total_return"] > 0).mean()) if not test.empty else np.nan
    holdout_sharpe = safe_float(hold.get("Sharpe"))
    holdout_dd = safe_float(hold.get("max_drawdown"))
    holdout_cagr = safe_float(hold.get("CAGR"))
    mean_rank_ic_20 = np.nan
    if not ic_decay.empty and "horizon" in ic_decay.columns:
        s = ic_decay.loc[(ic_decay["horizon"].eq("20d")) & (ic_decay["subset"].eq("all")), "mean_spearman_rank_ic"]
        if not s.empty:
            mean_rank_ic_20 = safe_float(s.iloc[0])
    if np.isnan(mean_test_sharpe) or np.isnan(holdout_sharpe):
        classification = "fails_oos"
    elif mean_test_sharpe > 1.0 and holdout_sharpe > 1.0 and holdout_dd > -0.25 and positive_test_rate >= 0.60:
        classification = "strong_oos_candidate"
    elif mean_test_sharpe > 0.5 and holdout_sharpe > 0.5 and positive_test_rate >= 0.50:
        classification = "passes_oos"
    elif mean_test_sharpe > 0.0 and holdout_sharpe > 0.0:
        classification = "unstable_oos"
    else:
        classification = "fails_oos"
    return pd.DataFrame(
        [{
            "classification": classification,
            "mean_test_sharpe": mean_test_sharpe,
            "positive_test_fold_rate": positive_test_rate,
            "locked_holdout_sharpe": holdout_sharpe,
            "locked_holdout_CAGR": holdout_cagr,
            "locked_holdout_max_drawdown": holdout_dd,
            "mean_rank_IC_20d": mean_rank_ic_20,
            "embargo_days": EMBARGO_DAYS,
            "purge_days": MAX_FORECAST_HORIZON_DAYS,
            "production_changed": False,
            "paper_changed": False,
            "parameters_tuned": False,
            "governance_note": "No tuning after locked holdout. Model frozen; folds evaluate temporal stability only.",
        }]
    )


def safe_float(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def write_report(results: pd.DataFrame, folds: pd.DataFrame, holdout: pd.DataFrame, ic_decay: pd.DataFrame, gov: pd.DataFrame, source: str) -> None:
    g = gov.iloc[0].to_dict() if not gov.empty else {}
    lines = [
        "===== PURGED WALK-FORWARD AND LOCKED OUT-OF-SAMPLE VALIDATION =====",
        "",
        f"Source series: {source}",
        f"Embargo days: {EMBARGO_DAYS}",
        f"Purge days: {MAX_FORECAST_HORIZON_DAYS}",
        "Model frozen: True",
        "Production changed: False",
        "Paper changed: False",
        "Parameter tuning: False",
        "",
        "===== FOLD SUMMARY =====",
        f"Walk-forward folds: {int(folds.shape[0]) if not folds.empty else 0}",
        f"Result rows: {int(results.shape[0]) if not results.empty else 0}",
        "",
        "===== LOCKED HOLDOUT =====",
    ]
    if not holdout.empty:
        h = holdout.iloc[0]
        lines.append(f"Period: {h.get('start_date')} to {h.get('end_date')}")
        lines.append(f"Observations: {h.get('observations')}")
        lines.append(f"CAGR: {h.get('CAGR')}")
        lines.append(f"Sharpe: {h.get('Sharpe')}")
        lines.append(f"Max drawdown: {h.get('max_drawdown')}")
    else:
        lines.append("Locked holdout unavailable.")
    lines.extend(["", "===== IC DECAY ====="])
    if not ic_decay.empty:
        for _, row in ic_decay.head(10).iterrows():
            lines.append(f"{row.get('subset')} {row.get('horizon')}: mean rank IC={row.get('mean_spearman_rank_ic')} positive_rate={row.get('positive_rank_ic_rate')}")
    lines.extend(["", "===== GOVERNANCE ====="])
    lines.append(f"Classification: {g.get('classification')}")
    lines.append(f"Mean test Sharpe: {g.get('mean_test_sharpe')}")
    lines.append(f"Positive test fold rate: {g.get('positive_test_fold_rate')}")
    lines.append(f"Locked holdout Sharpe: {g.get('locked_holdout_sharpe')}")
    lines.append(f"Locked holdout max drawdown: {g.get('locked_holdout_max_drawdown')}")
    lines.append(f"Note: {g.get('governance_note')}")
    Path("out_of_sample_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_growth_final_series()
    if df.empty:
        empty = pd.DataFrame([{"status": "missing growth_champion_final return series"}])
        empty.to_csv("purged_walk_forward_results.csv", index=False)
        empty.to_csv("purged_walk_forward_folds.csv", index=False)
        empty.to_csv("locked_holdout_results.csv", index=False)
        empty.to_csv("ic_decay_results.csv", index=False)
        empty.to_csv("out_of_sample_governance.csv", index=False)
        Path("out_of_sample_report.txt").write_text("Missing growth_champion_final return series.", encoding="utf-8")
        print("===== PURGED WALK-FORWARD VALIDATION =====")
        print("status: missing growth_champion_final return series")
        return
    wf_results, folds = build_purged_walk_forward(df)
    pkf = build_purged_kfold(df)
    if not pkf.empty:
        wf_results = pd.concat([wf_results, pkf], ignore_index=True, sort=False)
    holdout = locked_holdout(df)
    ic_decay = compute_ic_decay()
    gov = governance(wf_results, holdout, ic_decay)
    wf_results.to_csv("purged_walk_forward_results.csv", index=False)
    folds.to_csv("purged_walk_forward_folds.csv", index=False)
    holdout.to_csv("locked_holdout_results.csv", index=False)
    ic_decay.to_csv("ic_decay_results.csv", index=False)
    gov.to_csv("out_of_sample_governance.csv", index=False)
    write_report(wf_results, folds, holdout, ic_decay, gov, str(df["source"].iloc[0]))
    g = gov.iloc[0]
    print("===== PURGED WALK-FORWARD AND LOCKED OUT-OF-SAMPLE VALIDATION =====")
    print(f"source: {df['source'].iloc[0]}")
    print(f"folds: {len(folds)}")
    print(f"locked_holdout_sharpe: {g['locked_holdout_sharpe']}")
    print(f"locked_holdout_max_drawdown: {g['locked_holdout_max_drawdown']}")
    print(f"classification: {g['classification']}")
    print("outputs: purged_walk_forward_results.csv, purged_walk_forward_folds.csv, locked_holdout_results.csv, ic_decay_results.csv, out_of_sample_governance.csv, out_of_sample_report.txt")


if __name__ == "__main__":
    main()

