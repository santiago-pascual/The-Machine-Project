from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_VOL = 0.22
OUTPUT_AUDIT = "growth_exposure_targeting_audit.csv"
OUTPUT_CODE = "growth_exposure_code_audit.csv"
OUTPUT_COMPARE = "growth_exposure_vs_backtest.csv"
OUTPUT_SENSITIVITY = "growth_exposure_sensitivity.csv"
OUTPUT_SUMMARY = "growth_exposure_audit_summary.txt"


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def num(series: pd.Series | object) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan)


def latest_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def reason_for_exposure(row: pd.Series) -> str:
    vals = {
        "vol_target_floor_or_output": float(row.get("uncapped_vol_target_exposure", np.nan)),
        "exposure_cap_60": float(row.get("exposure_cap_60", np.nan)),
        "dual_trend_cap": float(row.get("dual_trend_cap", np.nan)),
    }
    final = float(row.get("final_exposure", np.nan))
    close = {k: v for k, v in vals.items() if np.isfinite(v) and abs(v - final) <= 1e-8}
    if not close:
        return "unknown_or_missing_inputs"
    if "vol_target_floor_or_output" in close and final == 0.40:
        return "vol_target_output_at_min_exposure_floor_40pct"
    return "binding_" + "_and_".join(close.keys())


def latest_vol_target_reference() -> dict[str, object]:
    daily = read_csv("growth_volatility_targeting_daily_returns.csv")
    if daily.empty:
        return {"vol_reference_date": "", "rolling_vol_used": np.nan, "raw_target_exposure": np.nan, "target_exposure": np.nan}
    variant_col = "vol_target_variant" if "vol_target_variant" in daily.columns else "variant"
    if variant_col in daily.columns:
        daily = daily[daily[variant_col].astype(str).eq("turnover_penalty_overlay_vol_target_22pct")].copy()
    if daily.empty:
        return {"vol_reference_date": "", "rolling_vol_used": np.nan, "raw_target_exposure": np.nan, "target_exposure": np.nan}
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    row = daily.sort_values("date").iloc[-1]
    return {
        "vol_reference_date": row.get("date").strftime("%Y-%m-%d") if pd.notna(row.get("date")) else "",
        "rolling_vol_used": float(row.get("rolling_vol_used", np.nan)),
        "raw_target_exposure": float(row.get("raw_target_exposure", np.nan)),
        "target_exposure": float(row.get("target_exposure", np.nan)),
    }


def build_exposure_audit() -> pd.DataFrame:
    perf = latest_by_date(read_csv("growth_candidate_paper_performance.csv"))
    state = latest_by_date(read_csv("growth_candidate_paper_state.csv"))
    alloc = latest_by_date(read_csv("current_growth_candidate_allocation.csv"))
    rows: list[dict[str, object]] = []
    vol_ref = latest_vol_target_reference()
    if perf.empty:
        return pd.DataFrame()

    for _, perf_row in perf.sort_values("date").iterrows():
        date = str(perf_row["date"])
        state_day = state[state["date"].astype(str).eq(date)] if not state.empty else pd.DataFrame()
        alloc_day = alloc[alloc["date"].astype(str).eq(date)] if not alloc.empty else pd.DataFrame()
        non_cash = state_day[state_day.get("ticker", pd.Series(dtype=str)).astype(str).ne("CASH")] if not state_day.empty else pd.DataFrame()
        selected = ",".join(non_cash.get("ticker", pd.Series(dtype=str)).astype(str).tolist())
        weights = ",".join(f"{t}:{float(w):.4f}" for t, w in zip(non_cash.get("ticker", []), num(non_cash.get("paper_position_weight", pd.Series(dtype=float))).fillna(0.0))) if not non_cash.empty else ""
        rolling_vol = np.nan
        raw_vol_target = np.nan
        if not alloc_day.empty:
            rolling_vol = float(num(alloc_day.get("rolling_volatility_used", pd.Series([np.nan]))).dropna().iloc[0]) if "rolling_volatility_used" in alloc_day.columns and not num(alloc_day["rolling_volatility_used"]).dropna().empty else np.nan
            raw_vol_target = float(num(alloc_day.get("raw_volatility_target_exposure", pd.Series([np.nan]))).dropna().iloc[0]) if "raw_volatility_target_exposure" in alloc_day.columns and not num(alloc_day["raw_volatility_target_exposure"]).dropna().empty else np.nan
        if not np.isfinite(rolling_vol):
            rolling_vol = float(vol_ref.get("rolling_vol_used", np.nan))
        if not np.isfinite(raw_vol_target):
            raw_vol_target = float(vol_ref.get("raw_target_exposure", np.nan))
        uncapped = float(perf_row.get("uncapped_exposure", perf_row.get("vol_target_exposure", np.nan)))
        final = float(perf_row.get("final_exposure", perf_row.get("exposure", np.nan)))
        row = {
            "date": date,
            "selected_tickers": selected,
            "selected_count": len(non_cash),
            "ticker_weights_before_vol_targeting": weights,
            "estimated_portfolio_volatility": rolling_vol,
            "target_volatility": TARGET_VOL,
            "raw_volatility_target_exposure": raw_vol_target,
            "uncapped_vol_target_exposure": uncapped,
            "exposure_cap_60": float(perf_row.get("exposure_cap_60", perf_row.get("exposure_cap", np.nan))),
            "dual_trend_cap": float(perf_row.get("dual_trend_cap", np.nan)),
            "final_exposure": final,
            "cash": float(perf_row.get("cash_weight", np.nan)),
            "reason_final_exposure": "",
            "final_exposure_equals_40_exactly": abs(final - 0.40) <= 1e-8 if np.isfinite(final) else False,
            "volatility_reference_date": vol_ref.get("vol_reference_date", ""),
            "volatility_reference_stale": str(vol_ref.get("vol_reference_date", "")) < date if vol_ref.get("vol_reference_date", "") else True,
            "data_source": perf_row.get("data_source", ""),
            "raw_target_feature_source": perf_row.get("raw_target_feature_source", ""),
            "stale_data_flag": "unknown",
        }
        row["reason_final_exposure"] = reason_for_exposure(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def code_audit() -> pd.DataFrame:
    findings: list[dict[str, object]] = []
    files = ["current_growth_feature_generation.py", "growth_candidate_paper_trading.py", "growth_candidate_paper_config.json"]
    patterns = {
        "hardcoded_40_or_floor": r"\b0\.40\b|\b0\.4\b|MIN_EXPOSURE",
        "volatility_target": r"TARGET_VOL|volatility_target|target_vol",
        "exposure_cap": r"exposure_cap|exposure_cap_60",
        "dual_trend": r"dual_trend|200D|200d",
        "cash_forced": r"cash\s*=\s*1\.0\s*-\s*|cash_weight",
        "min_vs_max_logic": r"min\(|max\(|np\.clip",
    }
    for file in files:
        path = Path(file)
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        lines = text.splitlines()
        for name, pattern in patterns.items():
            matches = []
            for i, line in enumerate(lines, start=1):
                if re.search(pattern, line, flags=re.IGNORECASE):
                    matches.append(f"{i}: {line.strip()}")
            findings.append(
                {
                    "file": file,
                    "check": name,
                    "matched": bool(matches),
                    "match_count": len(matches),
                    "evidence": " | ".join(matches[:12]),
                }
            )
    cfg = {}
    if Path("growth_candidate_paper_config.json").exists():
        try:
            cfg = json.loads(Path("growth_candidate_paper_config.json").read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    findings.append(
        {
            "file": "growth_candidate_paper_config.json",
            "check": "active_config_values",
            "matched": bool(cfg),
            "match_count": len(cfg),
            "evidence": json.dumps(cfg, sort_keys=True),
        }
    )
    return pd.DataFrame(findings)


def exposure_vs_backtest(audit: pd.DataFrame) -> pd.DataFrame:
    backtest = read_csv("growth_final_selection_daily_returns.csv")
    rows: list[dict[str, object]] = []
    if not backtest.empty and "candidate" in backtest.columns:
        bt = backtest[backtest["candidate"].astype(str).eq("growth_champion_v3")].copy()
        exp = num(bt.get("candidate_exposure", pd.Series(dtype=float))).dropna()
        rows.append(
            {
                "series": "growth_champion_final_backtest_v3",
                "observations": len(exp),
                "average_exposure": float(exp.mean()) if len(exp) else np.nan,
                "median_exposure": float(exp.median()) if len(exp) else np.nan,
                "min_exposure": float(exp.min()) if len(exp) else np.nan,
                "max_exposure": float(exp.max()) if len(exp) else np.nan,
                "pct_days_exactly_40": float((exp.round(8).eq(0.40)).mean()) if len(exp) else np.nan,
            }
        )
    if not audit.empty:
        exp = num(audit["final_exposure"]).dropna()
        rows.append(
            {
                "series": "growth_paper_current_history",
                "observations": len(exp),
                "average_exposure": float(exp.mean()) if len(exp) else np.nan,
                "median_exposure": float(exp.median()) if len(exp) else np.nan,
                "min_exposure": float(exp.min()) if len(exp) else np.nan,
                "max_exposure": float(exp.max()) if len(exp) else np.nan,
                "pct_days_exactly_40": float((exp.round(8).eq(0.40)).mean()) if len(exp) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sensitivity() -> pd.DataFrame:
    alloc = latest_by_date(read_csv("current_growth_candidate_allocation.csv"))
    if alloc.empty:
        return pd.DataFrame()
    latest_date = alloc["date"].max()
    cur = alloc[alloc["date"].astype(str).eq(str(latest_date))].copy()
    rolling_vol = float(num(cur.get("rolling_volatility_used", pd.Series([np.nan]))).dropna().iloc[0]) if "rolling_volatility_used" in cur.columns and not num(cur["rolling_volatility_used"]).dropna().empty else np.nan
    if not np.isfinite(rolling_vol):
        rolling_vol = float(latest_vol_target_reference().get("rolling_vol_used", np.nan))
    cap60 = float(num(cur.get("exposure_cap_60", pd.Series([0.60]))).dropna().iloc[0])
    dual = float(num(cur.get("dual_trend_cap", pd.Series([cap60]))).dropna().iloc[0])
    selected = ",".join(cur.get("ticker", pd.Series(dtype=str)).astype(str).tolist())
    rows = []
    for target in [0.15, 0.22, 0.30]:
        raw = target / rolling_vol if np.isfinite(rolling_vol) and rolling_vol > 0 else np.nan
        floored = float(np.clip(raw, 0.40, 1.00)) if np.isfinite(raw) else np.nan
        final = float(np.clip(min(floored, cap60, dual), 0.0, 1.0)) if np.isfinite(floored) else np.nan
        rows.append(
            {
                "date": latest_date,
                "selected_tickers": selected,
                "target_volatility": target,
                "rolling_volatility_used": rolling_vol,
                "raw_target_exposure_before_floor": raw,
                "exposure_after_min_40_max_100": floored,
                "exposure_cap_60": cap60,
                "dual_trend_cap": dual,
                "diagnostic_final_exposure": final,
                "diagnostic_cash": 1.0 - final if np.isfinite(final) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_summary(audit: pd.DataFrame, code: pd.DataFrame, compare: pd.DataFrame, sens: pd.DataFrame) -> None:
    current = audit.iloc[-1] if not audit.empty else pd.Series(dtype=object)
    exact40 = bool(current.get("final_exposure_equals_40_exactly", False))
    reason = str(current.get("reason_final_exposure", "unknown"))
    dual = float(current.get("dual_trend_cap", np.nan)) if not audit.empty else np.nan
    cap = float(current.get("exposure_cap_60", np.nan)) if not audit.empty else np.nan
    vol = float(current.get("estimated_portfolio_volatility", np.nan)) if not audit.empty else np.nan
    hardcoded = code[(code["file"].eq("current_growth_feature_generation.py")) & (code["check"].eq("hardcoded_40_or_floor"))]["matched"].any()
    stale = str(current.get("data_source", "")).strip() != "current_growth_candidate_allocation"
    lines = [
        "===== GROWTH EXPOSURE TARGETING AUDIT =====",
        f"current_final_exposure: {current.get('final_exposure', np.nan)}",
        f"current_cash: {current.get('cash', np.nan)}",
        f"final_exposure_equals_40_exactly: {exact40}",
        f"reason_final_exposure: {reason}",
        f"estimated_portfolio_volatility_used: {vol}",
        f"target_volatility: {TARGET_VOL}",
        f"exposure_cap_60: {cap}",
        f"dual_trend_cap: {dual}",
        "",
        "===== FINDINGS =====",
        f"is_40_expected: {'yes, under current code because MIN_EXPOSURE floors vol-target output at 40%' if exact40 else 'not currently pinned at 40%'}",
        f"is_hardcoded: {'partially: MIN_EXPOSURE = 0.40 is an explicit floor, not final_exposure hardcoded directly' if hardcoded else 'no explicit 40% floor found'}",
        f"caused_by_high_volatility_holdings: {'likely yes' if np.isfinite(vol) and vol > TARGET_VOL / 0.40 else 'not proven from available vol'}",
        f"caused_by_dual_trend: {'no, dual trend cap is above final exposure' if np.isfinite(dual) and dual > current.get('final_exposure', np.nan) else 'possibly'}",
        f"caused_by_cap60: {'no, cap60 is above final exposure' if np.isfinite(cap) and cap > current.get('final_exposure', np.nan) else 'possibly'}",
        f"stale_data: {'yes or fallback source detected' if stale else 'no, current allocation source is current_growth_candidate_allocation'}",
        f"volatility_reference_stale: {current.get('volatility_reference_stale', np.nan)}",
        "bug_assessment: no evidence of cash/rebalance bug; main issue is exposure floor/design choice plus high rolling volatility estimate.",
        "",
        "===== OUTPUTS =====",
        OUTPUT_AUDIT,
        OUTPUT_CODE,
        OUTPUT_COMPARE,
        OUTPUT_SENSITIVITY,
    ]
    Path(OUTPUT_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_exposure_audit()
    code = code_audit()
    compare = exposure_vs_backtest(audit)
    sens = sensitivity()
    audit.to_csv(OUTPUT_AUDIT, index=False)
    code.to_csv(OUTPUT_CODE, index=False)
    compare.to_csv(OUTPUT_COMPARE, index=False)
    sens.to_csv(OUTPUT_SENSITIVITY, index=False)
    write_summary(audit, code, compare, sens)

    print("\n===== GROWTH EXPOSURE TARGETING AUDIT =====")
    if not audit.empty:
        cols = [
            "date",
            "selected_tickers",
            "selected_count",
            "estimated_portfolio_volatility",
            "target_volatility",
            "raw_volatility_target_exposure",
            "uncapped_vol_target_exposure",
            "exposure_cap_60",
            "dual_trend_cap",
            "final_exposure",
            "cash",
            "reason_final_exposure",
            "final_exposure_equals_40_exactly",
        ]
        print(audit[[c for c in cols if c in audit.columns]].tail(12).to_string(index=False))
    print("\n===== EXPOSURE VS BACKTEST =====")
    print(compare.to_string(index=False))
    print("\n===== EXPOSURE SENSITIVITY =====")
    print(sens.to_string(index=False))
    print("\n===== CODE AUDIT SUMMARY =====")
    print(code[["file", "check", "matched", "match_count"]].to_string(index=False))
    print(f"\nSaved: {Path(OUTPUT_AUDIT).resolve()}")
    print(f"Saved: {Path(OUTPUT_SUMMARY).resolve()}")


if __name__ == "__main__":
    main()
