from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import time
from pathlib import Path

import pandas as pd

MODEL_MAP_FILE = "model_ticker_to_cedear_map.csv"
SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
FDS_FILE = "financial_data_system.py"
OUT_OHLCV = "yahoo_historical_ohlcv_coverage.csv"
OUT_RECONSTRUCT = "growth_feature_reconstructability.csv"
OUT_PLAN = "growth_reconstructed_backtest_plan.csv"
OUT_GOVERNANCE = "growth_historical_feasibility_governance.csv"

START_DATES = ["2001-01-01", "2008-01-01", "2010-01-01", "2015-01-01", "2020-01-01"]
MIN_OBSERVATIONS = 252
DEFAULT_BATCH_SIZE = 20
DEFAULT_SLEEP_SECONDS = 3.0
DEFAULT_RETRIES = 1


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _clean_ticker(value: object) -> str:
    text = str(value).strip().upper().replace(".", "-")
    return "".join(ch for ch in text if ch.isalnum() or ch == "-")


def _dedupe(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = _clean_ticker(value)
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def _extract_list_constant(source: str, name: str) -> list[str]:
    match = re.search(rf"{name}\s*=\s*(\[[\s\S]*?\])", source)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        return []
    return [_clean_ticker(x) for x in parsed]


def _system_uses_yfinance() -> tuple[bool, str]:
    path = Path(FDS_FILE)
    if not path.exists():
        return False, "financial_data_system.py missing"
    text = path.read_text(encoding="utf-8", errors="ignore")
    uses = "import yfinance as yf" in text and "yf.download" in text
    return (
        uses,
        "financial_data_system.py imports yfinance as yf and calls yf.download" if uses else "No direct yfinance download call detected",
    )


def _normal_universe() -> list[str]:
    tickers: list[str] = []
    if Path(FDS_FILE).exists():
        src = Path(FDS_FILE).read_text(encoding="utf-8", errors="ignore")
        for name in ["CORE_TICKERS", "GLOBAL_IMPORTANT_TICKERS", "NASDAQ_FALLBACK_TICKERS"]:
            tickers.extend(_extract_list_constant(src, name))
    snapshots = _read_csv(SNAPSHOTS_FILE)
    if not snapshots.empty and "ticker" in snapshots.columns:
        tickers.extend(snapshots["ticker"].dropna().map(_clean_ticker).tolist())
    model_map = _read_csv(MODEL_MAP_FILE)
    if not model_map.empty and "model_ticker" in model_map.columns:
        tickers.extend(model_map["model_ticker"].dropna().map(_clean_ticker).tolist())
    return _dedupe(tickers)


def _cedear_universe() -> list[str]:
    model_map = _read_csv(MODEL_MAP_FILE)
    if model_map.empty or "model_ticker" not in model_map.columns:
        return []
    if "available_as_cedear" in model_map.columns:
        available = model_map[model_map["available_as_cedear"].astype(str).str.lower().isin(["true", "1", "yes"])]
    else:
        available = model_map
    return _dedupe(available["model_ticker"].dropna().tolist())


def _select_universe(universe: str) -> list[str]:
    if universe == "cedear":
        return _cedear_universe()
    return _normal_universe()


def _import_yfinance():
    if importlib.util.find_spec("yfinance") is None:
        return None
    import yfinance as yf  # type: ignore

    cache_dir = Path("yf_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        try:
            yf.set_tz_cache_location(str(cache_dir))
        except Exception:
            pass
    return yf


def _flatten_download(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    out = data.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if ticker in out.columns.get_level_values(-1):
            out = out.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in out.columns.get_level_values(0):
            out = out.xs(ticker, axis=1, level=0, drop_level=True)
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out.dropna(how="all")


def _empty_row(ticker: str, status: str, error: str = "") -> dict[str, object]:
    row = {
        "ticker": ticker,
        "download_status": status,
        "first_available_date": "",
        "last_available_date": "",
        "observations": 0,
        "volume_available": False,
        "adjusted_close_available": False,
        "download_error": error,
    }
    for start in START_DATES:
        year = start[:4]
        row[f"existed_at_{year}"] = False
        row[f"enough_history_from_{year}"] = False
        row[f"observations_from_{year}"] = 0
    return row


def _coverage_from_df(ticker: str, df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return _empty_row(ticker, "no_data")
    first = df.index.min()
    last = df.index.max()
    row = {
        "ticker": ticker,
        "download_status": "ok",
        "first_available_date": first.strftime("%Y-%m-%d") if pd.notna(first) else "",
        "last_available_date": last.strftime("%Y-%m-%d") if pd.notna(last) else "",
        "observations": len(df),
        "volume_available": bool("Volume" in df.columns and pd.to_numeric(df["Volume"], errors="coerce").notna().any()),
        "adjusted_close_available": bool("Adj Close" in df.columns and pd.to_numeric(df["Adj Close"], errors="coerce").notna().any()),
        "download_error": "",
    }
    for start in START_DATES:
        year = start[:4]
        start_ts = pd.Timestamp(start)
        after = df[df.index >= start_ts]
        existed = bool(pd.notna(first) and first <= start_ts)
        enough = existed and len(after.dropna(how="all")) >= MIN_OBSERVATIONS
        row[f"existed_at_{year}"] = existed
        row[f"enough_history_from_{year}"] = bool(enough)
        row[f"observations_from_{year}"] = len(after.dropna(how="all"))
    return row


def _download_one(yf, ticker: str) -> dict[str, object]:
    try:
        data = yf.download(ticker, start="2001-01-01", progress=False, auto_adjust=False, actions=False, threads=False, timeout=20)
        return _coverage_from_df(ticker, _flatten_download(data, ticker))
    except Exception as exc:  # noqa: BLE001
        return _empty_row(ticker, "error", str(exc))


def _write_progress(rows: list[dict[str, object]], universe: str, uses_yf: bool, source_note: str) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if not out.empty:
        out["universe"] = universe
        out["data_source_confirmed"] = uses_yf
        out["data_source_note"] = source_note
        out["survivorship_warning"] = "high_current_universe_not_point_in_time"
        out = out.drop_duplicates(subset=["universe", "ticker"], keep="last").sort_values(["universe", "ticker"])
    out.to_csv(OUT_OHLCV, index=False)
    return out


def yahoo_ohlcv_coverage(
    universe: str, batch_size: int, sleep_seconds: float, retries: int, reset: bool, max_tickers: int | None
) -> pd.DataFrame:
    uses_yf, source_note = _system_uses_yfinance()
    tickers = _select_universe(universe)
    total_universe = len(tickers)
    if max_tickers is not None and max_tickers > 0:
        tickers = tickers[:max_tickers]
    yf = _import_yfinance()
    existing = _read_csv(OUT_OHLCV)
    rows: list[dict[str, object]] = []
    done: set[str] = set()
    if not reset and not existing.empty and {"ticker", "download_status", "universe"}.issubset(existing.columns):
        same = existing[existing["universe"].astype(str).eq(universe)].copy()
        successful = same[same["download_status"].astype(str).eq("ok")]
        rows.extend(same.to_dict("records"))
        done = set(successful["ticker"].astype(str))
    pending = [ticker for ticker in tickers if ticker not in done]
    if yf is None:
        rows.append(_empty_row("", "yfinance_not_installed"))
        out = _write_progress(rows, universe, uses_yf, source_note)
        out.attrs["total_universe_tickers"] = total_universe
        return out
    for attempt in range(max(1, retries + 1)):
        current_pending = pending if attempt == 0 else [r["ticker"] for r in rows if r.get("download_status") != "ok" and r.get("ticker")]
        if not current_pending:
            break
        for i in range(0, len(current_pending), max(1, batch_size)):
            batch = current_pending[i : i + max(1, batch_size)]
            for ticker in batch:
                row = _download_one(yf, ticker)
                rows = [r for r in rows if not (r.get("ticker") == ticker and r.get("universe", universe) == universe)]
                rows.append(row)
            _write_progress(rows, universe, uses_yf, source_note)
            if sleep_seconds > 0 and i + batch_size < len(current_pending):
                time.sleep(sleep_seconds)
    out = _write_progress(rows, universe, uses_yf, source_note)
    out.attrs["total_universe_tickers"] = total_universe
    return out


def feature_reconstructability() -> pd.DataFrame:
    rows = [
        [
            "raw_target_return_exact",
            "C_requires_historical_forecast_snapshots_for_exact",
            False,
            "approximate_if_full_target_engine_rerun",
            True,
            True,
            "Exact value is target_price/current_price generated by model; OHLCV alone cannot recover stored pre-2022 targets.",
        ],
        [
            "raw_target_rank",
            "B_reconstructable_from_current_model_logic_after_raw_target",
            False,
            True,
            True,
            True,
            "Deterministic only after raw_target_return exists.",
        ],
        [
            "soft_exit_rule",
            "B_reconstructable_from_current_model_logic_after_signals",
            False,
            True,
            False,
            True,
            "Needs prior selected positions and current raw target sign.",
        ],
        [
            "volatility_target_22pct",
            "A_reconstructable_from_strategy_return_path",
            "after_candidate_returns_exist",
            True,
            False,
            False,
            "Causal once candidate returns are computed.",
        ],
        ["exposure_cap_60", "A_reconstructable_from_rule", True, True, False, False, "Fixed cap; no market data dependency."],
        [
            "final_allocation",
            "B_reconstructable_from_current_model_logic_after_selection",
            False,
            True,
            "for_exact_replay",
            True,
            "Needs selected tickers, soft exit state, volatility target and cap.",
        ],
        [
            "current_optimizer_constraints",
            "not_active_in_growth_v2_replay",
            "not_required",
            "not_required",
            False,
            False,
            "Growth v2 paper construction is overlay-based, not optimizer-driven.",
        ],
        [
            "benchmark_SPY_QQQ_comparison",
            "A_reconstructable_from_ohlcv",
            True,
            True,
            False,
            False,
            "Yahoo OHLCV can provide benchmark returns if download succeeds.",
        ],
    ]
    out = pd.DataFrame(
        rows,
        columns=[
            "component",
            "classification",
            "reconstructable_from_ohlcv",
            "reconstructable_from_current_model_logic",
            "requires_historical_forecast_snapshots",
            "not_reconstructable_before_2022_exact",
            "notes",
        ],
    )
    out.to_csv(OUT_RECONSTRUCT, index=False)
    return out


def reconstructed_backtest_plan(coverage: pd.DataFrame, total_universe: int, universe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = coverage[coverage.get("download_status", pd.Series(dtype=str)).astype(str).eq("ok")] if not coverage.empty else pd.DataFrame()
    counts = {}
    for start in START_DATES:
        col = f"enough_history_from_{start[:4]}"
        counts[start] = int(ok[col].sum()) if col in ok.columns else 0
    plan = pd.DataFrame(
        [
            {
                "plan_step": 1,
                "name": "batched_price_data_download",
                "description": "Download adjusted OHLCV from Yahoo in batches with retry and partial progress.",
                "exact_or_approx": "input_preparation",
            },
            {
                "plan_step": 2,
                "name": "causal_model_rerun",
                "description": "For each date t, truncate OHLCV to t and rerun current target engine to approximate raw target.",
                "exact_or_approx": "approximation_not_snapshot_replay",
            },
            {
                "plan_step": 3,
                "name": "growth_overlay",
                "description": "Apply raw target rank, soft_exit_rule, volatility_target_22pct and exposure_cap_60 causally.",
                "exact_or_approx": "causal_reconstruction",
            },
            {
                "plan_step": 4,
                "name": "label_and_benchmark",
                "description": "Compute future returns only after decisions and compare to SPY/QQQ.",
                "exact_or_approx": "walk_forward_evaluation",
            },
        ]
    )
    plan.to_csv(OUT_PLAN, index=False)
    if ok.empty:
        classification = "not_feasible"
        reason = "Yahoo/yfinance OHLCV download failed or no usable data returned."
    elif counts.get("2008-01-01", 0) > 0:
        classification = "reconstructed_backtest_possible"
        reason = (
            "Yahoo OHLCV coverage exists for normal universe tickers before 2008; exact replay still impossible without pre-2022 snapshots."
        )
    elif counts.get("2020-01-01", 0) > 0:
        classification = "price_only_reconstruction_possible"
        reason = "Yahoo OHLCV coverage exists before 2020 for some tickers; exact raw target snapshots remain unavailable."
    else:
        classification = "not_feasible"
        reason = "Insufficient Yahoo OHLCV coverage for requested starts."
    governance = pd.DataFrame(
        [
            {
                "classification": classification,
                "universe": universe,
                "total_universe_tickers": total_universe,
                "tickers_downloaded_successfully": len(ok),
                "tickers_failed": int(len(coverage) - len(ok)) if not coverage.empty else total_universe,
                "exact_pre2022_replay_possible": False,
                "reconstructed_backtest_possible": classification
                in {"reconstructed_backtest_possible", "price_only_reconstruction_possible"},
                "price_only_reconstruction_possible": classification
                in {"reconstructed_backtest_possible", "price_only_reconstruction_possible"},
                "tickers_enough_from_2001": counts.get("2001-01-01", 0),
                "tickers_enough_from_2008": counts.get("2008-01-01", 0),
                "tickers_enough_from_2010": counts.get("2010-01-01", 0),
                "tickers_enough_from_2015": counts.get("2015-01-01", 0),
                "tickers_enough_from_2020": counts.get("2020-01-01", 0),
                "survivorship_warning": "high_current_universe_not_point_in_time",
                "reason": reason,
                "production_changed": False,
                "parameter_tuning": False,
            }
        ]
    )
    governance.to_csv(OUT_GOVERNANCE, index=False)
    return plan, governance


def run(universe: str, batch_size: int, sleep_seconds: float, retries: int, reset: bool, max_tickers: int | None) -> None:
    uses_yf, note = _system_uses_yfinance()
    total_universe = len(_select_universe(universe))
    coverage = yahoo_ohlcv_coverage(universe, batch_size, sleep_seconds, retries, reset, max_tickers)
    reconstruct = feature_reconstructability()
    plan, governance = reconstructed_backtest_plan(coverage, total_universe, universe)
    ok_count = int((coverage.get("download_status", pd.Series(dtype=str)).astype(str) == "ok").sum()) if not coverage.empty else 0
    failed_count = int(len(coverage) - ok_count) if not coverage.empty else total_universe
    print("\n===== YAHOO FINANCE HISTORICAL FEASIBILITY =====")
    print(f"system uses yfinance/Yahoo Finance: {uses_yf}")
    print(f"data source note: {note}")
    print(f"universe: {universe}")
    print(
        f"total normal universe tickers: {total_universe}" if universe == "normal" else f"total CEDEAR universe tickers: {total_universe}"
    )
    print(f"tickers attempted in this run/file: {len(coverage)}")
    print(f"tickers downloaded successfully: {ok_count}")
    print(f"tickers failed: {failed_count}")
    print("\n===== OHLCV COVERAGE =====")
    if not coverage.empty:
        print(f"download status counts: {coverage['download_status'].value_counts(dropna=False).to_dict()}")
        cols = [
            "ticker",
            "download_status",
            "first_available_date",
            "last_available_date",
            "observations",
            "volume_available",
            "adjusted_close_available",
        ]
        print(coverage[[c for c in cols if c in coverage.columns]].head(35).to_string(index=False))
    print("\n===== FEATURE RECONSTRUCTABILITY =====")
    print(
        reconstruct[
            ["component", "classification", "requires_historical_forecast_snapshots", "not_reconstructable_before_2022_exact"]
        ].to_string(index=False)
    )
    print("\n===== RECONSTRUCTED BACKTEST PLAN =====")
    print(plan.to_string(index=False))
    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))
    print("\nFiles generated:")
    for p in [OUT_OHLCV, OUT_RECONSTRUCT, OUT_PLAN, OUT_GOVERNANCE]:
        print(f"- {Path(p).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yahoo Finance OHLCV feasibility for Growth Champion v2 reconstruction.")
    parser.add_argument("--universe", choices=["normal", "cedear"], default="normal")
    parser.add_argument("--max-tickers", type=int, default=0, help="0 means all tickers in selected universe.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--reset", action="store_true", help="Ignore existing successful rows and redownload.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        universe=args.universe,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        reset=args.reset,
        max_tickers=None if args.max_tickers == 0 else args.max_tickers,
    )
