from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from canonical_market_data_manager import (
    refresh_tickers,
    validate_freshness,
)

LOG_FILE = "daily_research_run_log.csv"
PRICE_CACHE_DIR = Path("yahoo_ohlcv_price_cache")



def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _row_count(path: str | Path) -> int:
    df = _read_csv(path)
    return len(df) if not df.empty else 0


def _latest_date_in_csv(path: str | Path, date_col: str = "date") -> pd.Timestamp | pd.NaT:
    df = _read_csv(path)
    if df.empty or date_col not in df.columns:
        return pd.NaT
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    return dates.max().normalize() if not dates.empty else pd.NaT


def _latest_cache_date() -> pd.Timestamp | pd.NaT:
    if not PRICE_CACHE_DIR.exists():
        return pd.NaT
    dates: list[pd.Timestamp] = []
    for path in PRICE_CACHE_DIR.glob("*.csv"):
        df = _read_csv(path)
        if df.empty:
            continue
        col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
        if col is None:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce").dropna()
        if not parsed.empty:
            dates.append(parsed.max().normalize())
    return max(dates) if dates else pd.NaT


def _date_text(value: pd.Timestamp | pd.NaT) -> str:
    return value.date().isoformat() if pd.notna(value) else "missing"


def _freshness_report(label: str) -> dict[str, pd.Timestamp | pd.NaT]:
    cache_latest = _latest_cache_date()
    forecast_latest = _latest_date_in_csv("forecast_history.csv")
    growth_latest = _latest_date_in_csv("current_growth_features.csv")
    allocation_latest = _latest_date_in_csv("current_growth_candidate_allocation.csv")
    paper_state_latest = _latest_date_in_csv("growth_candidate_paper_state.csv")
    print(f"\n===== FORECAST HISTORY FRESHNESS GATE ({label}) =====")
    print(f"yahoo/cache latest date: {_date_text(cache_latest)}")
    print(f"forecast_history latest date: {_date_text(forecast_latest)}")
    print(f"current_growth_features latest date: {_date_text(growth_latest)}")
    print(f"current_growth_candidate_allocation latest date: {_date_text(allocation_latest)}")
    print(f"growth paper state latest date: {_date_text(paper_state_latest)}")
    return {
        "cache_latest": cache_latest,
        "forecast_latest": forecast_latest,
        "growth_latest": growth_latest,
        "allocation_latest": allocation_latest,
        "paper_state_latest": paper_state_latest,
    }




def _growth_gate_tickers() -> list[str]:
    tickers: list[str] = []
    # Official freshness validates only tickers that can affect official state:
    # current filtered allocation, existing official/candidate holdings, and
    # benchmarks. Raw forecast top-N names are pre-filter candidates and must not
    # block the official pipeline before quality/tradability filters run.
    for path in ["current_growth_candidate_allocation.csv", "growth_official_paper_state.csv", "growth_candidate_paper_state.csv"]:
        df = _read_csv(path)
        if df.empty or "ticker" not in df.columns:
            continue
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["date"].notna().any():
                df = df[df["date"].eq(df["date"].max())]
        tickers.extend([t for t in df["ticker"].dropna().astype(str).str.upper().str.strip() if t and t != "CASH"])
    tickers.extend(["SPY", "QQQ"])
    return list(dict.fromkeys(tickers))


def _official_market_data_gate(expected_date: pd.Timestamp | pd.NaT) -> dict[str, object]:
    if pd.isna(expected_date):
        gov = pd.DataFrame([{
            "expected_signal_date": "missing",
            "classification": "STALE_DATA_BLOCKED",
            "paper_may_run": False,
            "block_new_rebalance": True,
            "do_not_advance_official_paper": True,
            "real_capital_blocked": True,
            "reason": "missing forecast/signal date for official paper gate",
        }])
        gov.to_csv("official_market_data_governance.csv", index=False)
        return gov.iloc[0].to_dict()
    tickers = _growth_gate_tickers()
    print("\n===== OFFICIAL PAPER MARKET DATA FRESHNESS GATE =====")
    print(f"expected signal date: {_date_text(expected_date)}")
    print(f"tickers checked/refreshed: {','.join(tickers)}")
    try:
        refresh = refresh_tickers(tickers, expected_date)
        print(f"canonical refresh attempted rows: {len(refresh)}")
    except Exception as exc:  # noqa: BLE001
        print(f"canonical refresh failed: {exc}")
    integrity, gov = validate_freshness(expected_date, tickers)
    row = gov.iloc[0].to_dict() if not gov.empty else {"paper_may_run": False, "classification": "STALE_DATA_BLOCKED", "reason": "governance missing"}
    print(f"market data classification: {row.get('classification')}")
    print(f"canonical market date: {row.get('canonical_market_date', 'missing')}")
    print(f"paper may run: {row.get('paper_may_run')}")
    print(f"reason: {row.get('reason')}")
    return row

def _dashboard_value(metric: str, default: str = "missing") -> str:
    dashboard = _read_csv("research_dashboard_summary.csv")
    if dashboard.empty or not {"metric", "value"}.issubset(dashboard.columns):
        return default
    rows = dashboard[dashboard["metric"].astype(str).eq(metric)]
    if rows.empty:
        return default
    return str(rows.iloc[-1]["value"])


def _extract_int(pattern: str, text: str, default: int = 0) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _run_python(script: str, env: dict[str, str]) -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).resolve().parent,
        env=env,
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    return int(completed.returncode), output


def _append_log(row: dict[str, object]) -> None:
    existing = _read_csv(LOG_FILE)
    output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) if not existing.empty else pd.DataFrame([row])
    output.to_csv(LOG_FILE, index=False)


def build_daily_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    # The main system run must generate daily forecasts first. Research/dashboard
    # entry-point flags are cleared here because those modes return early.
    for key in [
        "RUN_RESEARCH_DASHBOARD",
        "RUN_WALK_FORWARD_BACKTEST",
        "RUN_TRIPLE_BARRIER_LABELING",
        "RUN_THRESHOLD_OPTIMIZATION",
        "RUN_BARRIER_OPTIMIZATION",
        "RUN_FULL_QUANT_ROBUSTNESS",
        "RUN_HISTORICAL_BACKFILL",
    ]:
        env[key] = "0"
    env.update(
        {
            "MODEL_MODE": args.model_mode,
            "YFINANCE_END_DATE_OFFSET_DAYS": "1",
            "PAPER_TRADING_ENABLED": "1" if args.paper_trading_enabled else "0",
            "PAPER_MODEL_MODE": args.paper_model_mode,
            "PAPER_OVERWRITE_SAME_DAY": "1" if args.overwrite_same_day else "0",
            "PAPER_META_FILTER_ENABLED": "1" if args.paper_meta_filter_enabled else os.environ.get("PAPER_META_FILTER_ENABLED", "0"),
            "PAPER_META_FILTER_MODEL": args.paper_meta_filter_model,
            "PAPER_META_FILTER_THRESHOLD": str(args.paper_meta_filter_threshold),
            "USE_WALK_FORWARD_CALIBRATED_FORECASTS": (
                "1"
                if (
                    args.use_walk_forward_calibrated_forecasts
                    or args.model_mode == "calibrated_forecast_research"
                    or args.paper_model_mode == "calibrated_forecast_research"
                )
                else os.environ.get("USE_WALK_FORWARD_CALIBRATED_FORECASTS", "0")
            ),
            "USE_RAW_TARGET_RETURN": (
                "1"
                if (
                    args.use_raw_target_return
                    or args.model_mode == "raw_target_research"
                    or args.paper_model_mode == "raw_target_research"
                )
                else os.environ.get("USE_RAW_TARGET_RETURN", "0")
            ),
            "WALK_FORWARD_CALIBRATED_FORECASTS_FILE": args.walk_forward_calibrated_forecasts_file,
            "COMPACT_REPORT_MODE": "1",
            "REPORT_SECTIONS_TO_SHOW": ",".join(
                [
                    "MODEL_MODE",
                    "CALIBRATED_FORECAST_RESEARCH",
                    "RAW_TARGET_RESEARCH",
                    "FINAL_ALLOCATION",
                    "ACTION_SIGNALS",
                    "PAPER_META_FILTER",
                    "PAPER_TRADING",
                    "FORECAST_CALIBRATION",
                    "INFORMATION_COEFFICIENT",
                ]
            ),
        }
    )
    return env


def run_daily_research(args: argparse.Namespace) -> dict[str, object]:
    run_date = datetime.now().strftime("%Y-%m-%d")
    run_time = datetime.now().strftime("%H:%M:%S")
    before_forecast_rows = _row_count("forecast_history.csv")
    before_paper_perf_rows = _row_count("paper_performance.csv")
    before_growth_paper_perf_rows = _row_count("growth_candidate_paper_performance.csv")
    before_ic_rows = _row_count("ic_history.csv")
    before_meta_filter_rows = _row_count("paper_meta_filter_report.csv")
    errors: list[str] = []

    print("\n===== DAILY RESEARCH RUN =====")
    print(f"date: {run_date}")
    print(f"model mode: {args.model_mode}")
    print(f"paper model mode: {args.paper_model_mode}")
    print(f"paper meta filter: {'enabled' if args.paper_meta_filter_enabled else os.environ.get('PAPER_META_FILTER_ENABLED', '0')}")
    print(f"growth paper: {'enabled' if args.growth_paper else 'disabled'}")
    print(
        "calibrated forecasts: "
        f"{'enabled' if (args.use_walk_forward_calibrated_forecasts or args.model_mode == 'calibrated_forecast_research' or args.paper_model_mode == 'calibrated_forecast_research') else os.environ.get('USE_WALK_FORWARD_CALIBRATED_FORECASTS', '0')}"
    )
    print(
        "raw target return: "
        f"{'enabled' if (args.use_raw_target_return or args.model_mode == 'raw_target_research' or args.paper_model_mode == 'raw_target_research') else os.environ.get('USE_RAW_TARGET_RETURN', '0')}"
    )
    print("heavy backtests: disabled")
    print("optimizations: disabled")
    print("real trading: disabled")

    env = build_daily_env(args)
    system_status = "skipped"
    forecast_rows_added = 0
    duplicate_forecast_rows_skipped = 0
    paper_updated = False
    paper_meta_filter_status = "disabled"
    ic_updated = False
    ic_rows_added = 0
    ic_status = "skipped"
    freshness_before = _freshness_report("before system run")

    if not args.skip_system:
        code, output = _run_python("financial_data_system.py", env)
        print(output)
        system_status = "ok" if code == 0 else "error"
        if code != 0:
            errors.append(f"financial_data_system.py exited with code {code}")
        forecast_rows_added = _extract_int(r"new rows added:\s*(\d+)", output, default=max(0, _row_count("forecast_history.csv") - before_forecast_rows))
        duplicate_forecast_rows_skipped = _extract_int(r"duplicate rows skipped:\s*(\d+)", output, default=0)
        if "PAPER TRADING SIMULATION" in output and "already exists. Skipped" not in output:
            paper_updated = True
        if args.paper_meta_filter_enabled or env.get("PAPER_META_FILTER_ENABLED", "0") == "1":
            paper_meta_filter_status = "ok" if "PAPER META FILTER" in output else "not_printed_or_skipped"
        after_ic_rows = _row_count("ic_history.csv")
        ic_rows_added = max(0, after_ic_rows - before_ic_rows)
        if "INFORMATION COEFFICIENT REPORT" in output or "FEATURE RANKING" in output:
            ic_status = "ok"
        elif code != 0:
            ic_status = "error"
        else:
            ic_status = "not_available"
        ic_updated = ic_rows_added > 0

    freshness_after_system = _freshness_report("after system run")
    if args.growth_paper:
        cache_latest = freshness_after_system.get("cache_latest", pd.NaT)
        forecast_latest = freshness_after_system.get("forecast_latest", pd.NaT)
        if pd.notna(cache_latest) and (pd.isna(forecast_latest) or forecast_latest < cache_latest):
            print("\n===== FORECAST HISTORY FRESHNESS REFRESH REQUESTED =====")
            print(f"forecast_history={_date_text(forecast_latest)} < yahoo/cache={_date_text(cache_latest)}")
            refresh_code, refresh_output = _run_python("forecast_history_freshness_refresh.py", env)
            print(refresh_output)
            if refresh_code != 0:
                errors.append(f"forecast_history_freshness_refresh.py exited with code {refresh_code}")
            freshness_after_system = _freshness_report("after forecast refresh")

    dashboard_status = "skipped"
    monitor_status = "skipped"
    growth_paper_status = "skipped"
    growth_paper_governance_status = "skipped"
    dashboard_output = ""
    monitor_output = ""
    growth_paper_output = ""
    growth_paper_governance_output = ""
    official_paper_output = ""
    official_paper_status = "not_run"
    official_benchmark_output = ""
    official_benchmark_status = "not_run"

    if args.run_dashboard:
        code, dashboard_output = _run_python("research_dashboard.py", env)
        print(dashboard_output)
        dashboard_status = "ok" if code == 0 else "error"
        if code != 0:
            errors.append(f"research_dashboard.py exited with code {code}")

    if args.run_monitor:
        code, monitor_output = _run_python("paper_trading_monitor.py", env)
        print(monitor_output)
        monitor_status = "ok" if code == 0 else "error"
        if code != 0:
            errors.append(f"paper_trading_monitor.py exited with code {code}")

    if args.growth_paper:
        gate_cache_latest = freshness_after_system.get("cache_latest", pd.NaT)
        gate_forecast_latest = freshness_after_system.get("forecast_latest", pd.NaT)
        market_gate = _official_market_data_gate(gate_forecast_latest)
        official_market_data_blocked = not bool(market_gate.get("paper_may_run", False))
        stale_growth_data = pd.notna(gate_cache_latest) and (pd.isna(gate_forecast_latest) or gate_forecast_latest < gate_cache_latest)
        if official_market_data_blocked:
            msg = (
                "OFFICIAL_PAPER_BLOCKED_STALE_DATA: "
                f"classification={market_gate.get('classification')}; "
                f"canonical_market_date={market_gate.get('canonical_market_date', 'missing')}; "
                f"expected_signal_date={_date_text(gate_forecast_latest)}; "
                f"reason={market_gate.get('reason')}"
            )
            print("\n===== OFFICIAL PAPER MARKET DATA BLOCK =====")
            print(msg)
            errors.append(msg)
            growth_paper_status = "OFFICIAL_PAPER_BLOCKED_STALE_DATA"
            growth_paper_governance_status = str(market_gate.get("classification", "STALE_DATA_BLOCKED"))
        elif stale_growth_data and not args.allow_stale_growth_data:
            msg = (
                "growth paper blocked by stale forecast_history: "
                f"forecast_history={_date_text(gate_forecast_latest)} < yahoo/cache={_date_text(gate_cache_latest)}. "
                "Use --allow-stale-growth-data to override."
            )
            print("\n===== GROWTH PAPER FRESHNESS BLOCK =====")
            print(msg)
            errors.append(msg)
            growth_paper_status = "blocked_stale_forecast_history"
            growth_paper_governance_status = "skipped_stale_forecast_history"
        else:
            if stale_growth_data:
                print("\n===== GROWTH PAPER STALE DATA OVERRIDE =====")
                print(f"forecast_history={_date_text(gate_forecast_latest)} < yahoo/cache={_date_text(gate_cache_latest)}; override enabled")
            feature_args = [sys.executable, "current_growth_feature_generation.py"]
            if args.overwrite_same_day:
                feature_args.append("--overwrite-same-day")
            if args.allow_stale_growth_volatility:
                feature_args.append("--allow-stale-growth-volatility")
            feature_completed = subprocess.run(
                feature_args,
                cwd=Path(__file__).resolve().parent,
                env=env,
                text=True,
                capture_output=True,
            )
            feature_output = "\n".join(part for part in [feature_completed.stdout, feature_completed.stderr] if part)
            print(feature_output)
            if feature_completed.returncode != 0:
                errors.append(f"current_growth_feature_generation.py exited with code {feature_completed.returncode}")

            growth_args = [sys.executable, "growth_candidate_paper_trading.py"]
            if args.overwrite_same_day:
                growth_args.append("--overwrite-same-day")
            if args.allow_growth_proxy_fallback:
                growth_args.append("--allow-proxy-fallback")
            completed = subprocess.run(
                growth_args,
                cwd=Path(__file__).resolve().parent,
                env=env,
                text=True,
                capture_output=True,
            )
            growth_paper_output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
            print(growth_paper_output)
            growth_paper_status = "ok" if completed.returncode == 0 else "error"
            if completed.returncode != 0:
                errors.append(f"growth_candidate_paper_trading.py exited with code {completed.returncode}")

            if completed.returncode == 0:
                official_args = [sys.executable, "growth_official_paper_lifecycle.py"]
                if args.overwrite_same_day:
                    official_args.append("--overwrite-same-day")
                official_completed = subprocess.run(
                    official_args,
                    cwd=Path(__file__).resolve().parent,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                official_paper_output = "\n".join(part for part in [official_completed.stdout, official_completed.stderr] if part)
                print(official_paper_output)
                official_paper_status = "ok" if official_completed.returncode == 0 else "error"
                if official_completed.returncode != 0:
                    errors.append(f"growth_official_paper_lifecycle.py exited with code {official_completed.returncode}")
                else:
                    benchmark_completed = subprocess.run(
                        [sys.executable, "official_benchmark_chart_repair.py"],
                        cwd=Path(__file__).resolve().parent,
                        env=env,
                        text=True,
                        capture_output=True,
                    )
                    official_benchmark_output = "\n".join(part for part in [benchmark_completed.stdout, benchmark_completed.stderr] if part)
                    print(official_benchmark_output)
                    official_benchmark_status = "ok" if benchmark_completed.returncode == 0 else "error"
                    if benchmark_completed.returncode != 0:
                        errors.append(f"official_benchmark_chart_repair.py exited with code {benchmark_completed.returncode}")

            governance_completed = subprocess.run(
                [sys.executable, "growth_paper_governance.py"],
                cwd=Path(__file__).resolve().parent,
                env=env,
                text=True,
                capture_output=True,
            )
            growth_paper_governance_output = "\n".join(part for part in [governance_completed.stdout, governance_completed.stderr] if part)
            print(growth_paper_governance_output)
            growth_paper_governance_status = "ok" if governance_completed.returncode == 0 else "error"
            if governance_completed.returncode != 0:
                errors.append(f"growth_paper_governance.py exited with code {governance_completed.returncode}")

    freshness_after_growth = _freshness_report("after growth paper")

    after_paper_perf_rows = _row_count("paper_performance.csv")
    paper_updated = paper_updated or after_paper_perf_rows > before_paper_perf_rows
    promotion_status = _dashboard_value("promotion_status", "blocked")

    row = {
        "date": run_date,
        "run_time": run_time,
        "model_mode": args.model_mode,
        "paper_model_mode": args.paper_model_mode,
        "use_walk_forward_calibrated_forecasts": env.get("USE_WALK_FORWARD_CALIBRATED_FORECASTS", "0") == "1",
        "use_raw_target_return": env.get("USE_RAW_TARGET_RETURN", "0") == "1",
        "forecast_rows_added": forecast_rows_added,
        "duplicate_forecast_rows_skipped": duplicate_forecast_rows_skipped,
        "paper_updated": bool(paper_updated),
        "growth_paper_enabled": bool(args.growth_paper),
        "growth_paper_updated": _row_count("growth_candidate_paper_performance.csv") > before_growth_paper_perf_rows,
        "growth_paper_status": growth_paper_status,
        "official_paper_status": official_paper_status,
        "official_benchmark_status": official_benchmark_status,
        "growth_paper_governance_status": growth_paper_governance_status,
        "paper_meta_filter_enabled": env.get("PAPER_META_FILTER_ENABLED", "0") == "1",
        "paper_meta_filter_status": paper_meta_filter_status,
        "paper_meta_filter_rows_added": max(0, _row_count("paper_meta_filter_report.csv") - before_meta_filter_rows),
        "ic_updated": bool(ic_updated),
        "ic_rows_added": int(ic_rows_added),
        "ic_status": ic_status,
        "dashboard_updated": dashboard_status == "ok",
        "monitor_updated": monitor_status == "ok",
        "promotion_status": promotion_status,
        "system_status": system_status,
        "dashboard_status": dashboard_status,
        "monitor_status": monitor_status,
        "growth_paper_output_seen": "GROWTH CANDIDATE PAPER TRADING" in growth_paper_output,
        "cache_latest_date": _date_text(freshness_after_growth.get("cache_latest", pd.NaT)),
        "forecast_history_latest_date": _date_text(freshness_after_growth.get("forecast_latest", pd.NaT)),
        "current_growth_features_latest_date": _date_text(freshness_after_growth.get("growth_latest", pd.NaT)),
        "current_growth_allocation_latest_date": _date_text(freshness_after_growth.get("allocation_latest", pd.NaT)),
        "growth_paper_state_latest_date": _date_text(freshness_after_growth.get("paper_state_latest", pd.NaT)),
        "errors": " | ".join(errors),
    }
    _append_log(row)

    print("\n===== DAILY RESEARCH RUN SUMMARY =====")
    print(f"forecast snapshot status: rows_added={forecast_rows_added}, duplicates_skipped={duplicate_forecast_rows_skipped}")
    print(f"paper trading status: {'updated' if paper_updated else 'not updated/skipped'}")
    print(f"growth paper status: {growth_paper_status}")
    print(f"official paper status: {official_paper_status}")
    print(f"official benchmark status: {official_benchmark_status}")
    print(f"growth paper governance status: {growth_paper_governance_status}")
    print(f"paper meta filter status: {paper_meta_filter_status}")
    print(f"IC status: {ic_status}, rows_added={ic_rows_added}")
    print(f"dashboard status: {dashboard_status}")
    print(f"monitor status: {monitor_status}")
    print(f"promotion status: {promotion_status}")
    print(f"errors: {'none' if not errors else ' | '.join(errors)}")
    print(f"Saved: {Path(LOG_FILE).resolve()}")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled daily research/paper workflow.")
    parser.add_argument("--model-mode", default="baseline")
    parser.add_argument("--paper-model-mode", default="regime_gated_full_quant")
    parser.add_argument("--paper-meta-filter", dest="paper_meta_filter_enabled", action="store_true")
    parser.add_argument("--paper-meta-filter-model", default="logistic_isotonic")
    parser.add_argument("--paper-meta-filter-threshold", type=float, default=0.65)
    parser.add_argument("--use-walk-forward-calibrated-forecasts", action="store_true")
    parser.add_argument("--use-raw-target-return", action="store_true")
    parser.add_argument("--walk-forward-calibrated-forecasts-file", default="walk_forward_calibrated_forecasts.csv")
    parser.add_argument("--overwrite-same-day", action="store_true")
    parser.add_argument("--growth-paper", action="store_true", help="Run Growth Candidate v1 paper trading separately from defensive paper.")
    parser.add_argument("--allow-growth-proxy-fallback", action="store_true", help="Allow growth paper to use forecast_history proxy if exact current growth features fail.")
    parser.add_argument("--allow-stale-growth-data", action="store_true", help="Allow growth paper to run even if forecast_history is older than Yahoo/cache latest date.")
    parser.add_argument("--allow-stale-growth-volatility", action="store_true", help="Allow growth paper to use stale volatility fallback if fresh OHLCV volatility cannot be computed.")
    parser.add_argument("--skip-system", action="store_true", help="Only refresh dashboard/monitor/log; do not run main system.")
    parser.add_argument("--no-paper", dest="paper_trading_enabled", action="store_false")
    parser.add_argument("--no-dashboard", dest="run_dashboard", action="store_false")
    parser.add_argument("--no-monitor", dest="run_monitor", action="store_false")
    parser.set_defaults(paper_trading_enabled=True, run_dashboard=True, run_monitor=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_daily_research(parse_args())
