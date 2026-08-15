from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

CEDEAR_UNIVERSE_FILE = "cedear_universe.csv"
MODEL_MAP_FILE = "model_ticker_to_cedear_map.csv"
CEDEAR_GROWTH_UNIVERSE_FILE = "cedear_growth_universe.csv"
FEASIBILITY_FILE = "extended_growth_backtest_feasibility.csv"
EXTENDED_RESULTS_FILE = "extended_growth_backtest_results.csv"
EXTENDED_BENCHMARK_FILE = "extended_growth_backtest_vs_benchmarks.csv"
GOVERNANCE_FILE = "extended_growth_backtest_governance.csv"
ALIAS_MAP_FILE = "cedear_alias_map.csv"
MATCHING_AUDIT_FILE = "cedear_matching_audit.csv"

SNAPSHOTS_FILE = "historical_forecast_snapshots.csv"
REALIZED_FILE = "historical_realized_returns.csv"
GROWTH_DAILY_FILE = "growth_head_to_head_daily_returns.csv"
GROWTH_RESULTS_FILE = "growth_head_to_head_results.csv"
GROWTH_BENCHMARK_FILE = "growth_head_to_head_governance.csv"
PRODUCTION_BENCHMARK_FILE = "production_parity_growth_benchmark_comparison.csv"
FDS_FILE = "financial_data_system.py"

START_DATES = ["2008-01-01", "2010-01-01", "2015-01-01", "2020-01-01", "2022-01-01"]

NASDAQ_HINTS = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "AVGO",
    "TSLA",
    "COST",
    "NFLX",
    "ASML",
    "TMUS",
    "CSCO",
    "PEP",
    "AMD",
    "AZN",
    "LIN",
    "INTU",
    "QCOM",
    "TXN",
    "AMGN",
    "ISRG",
    "BKNG",
    "AMAT",
    "ADBE",
    "PDD",
    "ARM",
    "HON",
    "GILD",
    "CMCSA",
    "PANW",
    "MU",
    "MELI",
    "ADP",
    "ADI",
    "LRCX",
    "KLAC",
    "SBUX",
    "MDLZ",
    "REGN",
    "VRTX",
    "INTC",
    "ABNB",
    "CRWD",
    "DASH",
    "MAR",
    "CEG",
    "PYPL",
    "CDNS",
    "SNPS",
}
NYSE_HINTS = {
    "MSTR",
    "SNAP",
    "OKLO",
    "JMIA",
    "XYZ",
    "RBLX",
    "TWLO",
    "SPOT",
    "TEAM",
    "SPCE",
    "SNOW",
    "CCJ",
    "YPF",
    "VIST",
    "TSM",
    "BABA",
    "TM",
    "SONY",
    "NVO",
    "SAP",
    "SHEL",
    "BP",
    "RIO",
    "BHP",
    "VALE",
    "SHOP",
    "SE",
    "NU",
    "PBR",
    "EC",
    "GLOB",
    "UBER",
    "COIN",
    "PLTR",
    "SMCI",
    "NET",
    "DDOG",
    "U",
    "AI",
    "PATH",
    "HIMS",
}
ETF_HINTS = {"SPY", "QQQ", "IWM", "DIA", "EEM", "EWZ", "ARKK", "GLD", "SLV", "TLT"}

MANUAL_ALIAS_RULES = [
    {
        "model_ticker": "GOOG",
        "cedear_underlying_ticker": "GOOGL",
        "match_type": "alias_match",
        "confidence": "documented_share_class_alias",
        "notes": "GOOG class C mapped to available GOOGL Alphabet CEDEAR only when GOOG is missing.",
        "requires_manual_confirmation": False,
    },
    {
        "model_ticker": "NU",
        "cedear_underlying_ticker": "UN",
        "match_type": "alias_match",
        "confidence": "documented_byma_code_alias",
        "notes": "BYMA CEDEAR code UN maps to NU Holdings Ltd/Cayman Islands.",
        "requires_manual_confirmation": False,
    },
    {
        "model_ticker": "YPF",
        "cedear_underlying_ticker": "YPF",
        "match_type": "manual_allow_if_present",
        "confidence": "not_matched_in_current_source",
        "notes": "Only allow if YPF is present in the CEDEAR source; current pasted list search did not confirm it.",
        "requires_manual_confirmation": True,
    },
    {
        "model_ticker": "KEEL",
        "cedear_underlying_ticker": "KEEL",
        "match_type": "exact_if_present",
        "confidence": "verified_in_source_as_bitfarms_line",
        "notes": "Source line: Bitfarms Ltd. KEEL NASDAQ GM 1:5. No BITF substitution needed.",
        "requires_manual_confirmation": False,
    },
    {
        "model_ticker": "RGTI",
        "cedear_underlying_ticker": "RGTI",
        "match_type": "exact_if_present",
        "confidence": "verified_in_source",
        "notes": "Source line: RIGETTI COMPUTING INC RGTI NASDAQ CM 2:1.",
        "requires_manual_confirmation": False,
    },
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out.dropna(subset=[col]).sort_values(col)


def _clean_ticker(value: object) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    text = text.replace(".", "-")
    text = re.sub(r"[^A-Z0-9-]", "", text)
    return text


def _discover_cedear_file(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    patterns = [
        "*cedear*.txt",
        "*CEDEAR*.txt",
        "*Cedear*.txt",
        "*byma*.txt",
        "*BYMA*.txt",
        "*cedear*.csv",
        "*CEDEAR*.csv",
        "*Cedear*.csv",
        "*byma*.csv",
        "*BYMA*.csv",
        "*cedear*.xlsx",
        "*CEDEAR*.xlsx",
        "*byma*.xlsx",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))
    candidates = [p for p in candidates if p.name not in {CEDEAR_UNIVERSE_FILE, MODEL_MAP_FILE, CEDEAR_GROWTH_UNIVERSE_FILE}]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _parse_txt_cedear_file(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    useful = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not useful:
        return pd.DataFrame()

    joined = "\n".join(useful)
    for sep in ["\t", ";", ",", "|"]:
        if sep in joined:
            from io import StringIO

            try:
                df = pd.read_csv(StringIO(joined), sep=sep)
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass

    market_pattern = r"(?:NASDAQ(?:\s+(?:GS|GM|CM))?|NYSE(?:\s+Arca)?|B3|XETRA|LSE|SIX|EURONEXT|OTC|AMEX)"
    full_pattern = re.compile(
        rf"^(?P<company>.*?)\s+(?P<ticker>[A-Z][A-Z0-9.\-]{{0,7}})\s+(?P<exchange>{market_pattern})\s+(?P<ratio>\d+\s*:\s*\d+)\s*$",
        flags=re.IGNORECASE,
    )
    rows = []
    for line in useful:
        low = line.lower()
        if "cedears negociables" in low or "nombre de la" in low or low in {"byma", "mercado donde", "cotiza ratio (*)"}:
            continue
        match = full_pattern.match(line)
        if not match:
            continue
        company = re.sub(r"\s+", " ", match.group("company")).strip()
        ticker = _clean_ticker(match.group("ticker"))
        exchange = re.sub(r"\s+", " ", match.group("exchange")).upper().strip()
        ratio = re.sub(r"\s+", "", match.group("ratio"))
        rows.append(
            {
                "company_name": company,
                "ticker": ticker,
                "underlying": ticker,
                "exchange": exchange,
                "ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


def _load_raw_cedear_list(path: Path | None) -> tuple[pd.DataFrame, str]:
    if path is None:
        return pd.DataFrame(), "missing_cedear_source"
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path), str(path)
        if path.suffix.lower() == ".txt":
            return _parse_txt_cedear_file(path), str(path)
        return pd.read_csv(path), str(path)
    except Exception as exc:
        return pd.DataFrame(), f"failed_to_read:{path}:{exc}"


def _best_column(columns: Iterable[str], names: Iterable[str]) -> str | None:
    normalized = {str(c).lower().strip(): c for c in columns}
    for name in names:
        key = name.lower().strip()
        if key in normalized:
            return normalized[key]
    for col in columns:
        low = str(col).lower()
        if any(name.lower() in low for name in names):
            return col
    return None


def parse_cedear_list(path: Path | None) -> pd.DataFrame:
    raw, source = _load_raw_cedear_list(path)
    if raw.empty:
        out = pd.DataFrame(
            columns=[
                "company_name",
                "byma_ticker",
                "underlying_ticker",
                "exchange",
                "ratio",
                "is_us_listed",
                "is_nasdaq",
                "is_nyse",
                "is_etf",
                "tradable_from_argentina",
                "source_file",
                "parse_status",
            ]
        )
        out.to_csv(CEDEAR_UNIVERSE_FILE, index=False)
        return out

    cols = list(raw.columns)
    company_col = _best_column(cols, ["company_name", "company", "empresa", "descripcion", "description", "nombre", "denominacion"])
    byma_col = _best_column(cols, ["byma_ticker", "ticker byma", "simbolo", "símbolo", "ticker", "especie", "cedear"])
    underlying_col = _best_column(cols, ["underlying_ticker", "underlying", "subyacente", "ticker usa", "ticker_us", "symbol"])
    exchange_col = _best_column(cols, ["exchange", "mercado", "bolsa", "listing"])
    ratio_col = _best_column(cols, ["ratio", "conversion", "conversión", "paridad"])

    rows = []
    for _, row in raw.iterrows():
        byma = _clean_ticker(row.get(byma_col, "")) if byma_col else ""
        underlying = _clean_ticker(row.get(underlying_col, "")) if underlying_col else ""
        if not underlying and byma:
            # Many CEDEAR lists use the underlying as the only ticker column.
            underlying = byma
        exchange = str(row.get(exchange_col, "") if exchange_col else "").upper().strip()
        if not exchange:
            if underlying in NASDAQ_HINTS:
                exchange = "NASDAQ"
            elif underlying in NYSE_HINTS:
                exchange = "NYSE"
        is_etf = underlying in ETF_HINTS or "ETF" in str(row.to_dict()).upper()
        rows.append(
            {
                "company_name": str(row.get(company_col, "") if company_col else "").strip(),
                "byma_ticker": byma,
                "underlying_ticker": underlying,
                "exchange": exchange,
                "ratio": str(row.get(ratio_col, "") if ratio_col else "").strip(),
                "is_us_listed": bool(underlying),
                "is_nasdaq": exchange == "NASDAQ" or underlying in NASDAQ_HINTS,
                "is_nyse": exchange == "NYSE" or underlying in NYSE_HINTS,
                "is_etf": bool(is_etf),
                "tradable_from_argentina": bool(byma or underlying),
                "source_file": source,
                "parse_status": "parsed",
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["underlying_ticker", "byma_ticker"], keep="first")
    out.to_csv(CEDEAR_UNIVERSE_FILE, index=False)
    return out


def _extract_list_constant(source: str, name: str) -> list[str]:
    pattern = rf"{name}\s*=\s*(\[[\s\S]*?\])"
    match = re.search(pattern, source)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        return []
    return [_clean_ticker(x) for x in parsed if _clean_ticker(x)]


def current_model_universe() -> list[str]:
    source = Path(FDS_FILE).read_text(encoding="utf-8", errors="ignore") if Path(FDS_FILE).exists() else ""
    tickers = []
    for name in ["CORE_TICKERS", "GLOBAL_IMPORTANT_TICKERS", "NASDAQ_FALLBACK_TICKERS"]:
        tickers.extend(_extract_list_constant(source, name))
    snapshots = _read_csv(SNAPSHOTS_FILE)
    if not snapshots.empty and "ticker" in snapshots.columns:
        tickers.extend(_clean_ticker(t) for t in snapshots["ticker"].dropna().unique())
    seen: set[str] = set()
    out = []
    for ticker in tickers:
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def _write_alias_map() -> pd.DataFrame:
    alias_df = pd.DataFrame(MANUAL_ALIAS_RULES)
    alias_df.to_csv(ALIAS_MAP_FILE, index=False)
    return alias_df


def _cedear_lookup(cedears: pd.DataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    if cedears.empty:
        return lookup
    for _, row in cedears.iterrows():
        for col in ["underlying_ticker", "byma_ticker"]:
            ticker = _clean_ticker(row.get(col, ""))
            if ticker and ticker not in lookup:
                lookup[ticker] = row
    return lookup


def _alias_lookup_row(ticker: str, cedear_lookup: dict[str, pd.Series]) -> tuple[pd.Series | None, dict[str, object] | None]:
    ticker = _clean_ticker(ticker)
    for rule in MANUAL_ALIAS_RULES:
        if _clean_ticker(rule.get("model_ticker", "")) != ticker:
            continue
        alias = _clean_ticker(rule.get("cedear_underlying_ticker", ""))
        if alias in cedear_lookup:
            return cedear_lookup[alias], rule
    return None, None


def _matching_audit_rows(model_tickers: list[str], map_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in model_tickers:
        match = map_df[map_df["model_ticker"].astype(str).eq(ticker)]
        if match.empty:
            rows.append({"model_ticker": ticker, "match_status": "still_missing", "notes": "not evaluated"})
            continue
        row = match.iloc[0]
        if bool(row.get("available_as_cedear", False)):
            note = str(row.get("notes", ""))
            if "alias" in note:
                status = "alias_match"
            elif "questionable" in note or "manual confirmation" in note:
                status = "questionable_mapping"
            else:
                status = "exact_match"
        else:
            status = "still_missing"
        rows.append(
            {
                "model_ticker": ticker,
                "match_status": status,
                "available_as_cedear": bool(row.get("available_as_cedear", False)),
                "byma_ticker": row.get("byma_ticker", ""),
                "underlying_ticker": row.get("underlying_ticker", ""),
                "exchange": row.get("exchange", ""),
                "ratio": row.get("ratio", ""),
                "notes": row.get("notes", ""),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(MATCHING_AUDIT_FILE, index=False)
    return audit


def map_model_tickers_to_cedears(model_tickers: list[str], cedears: pd.DataFrame) -> pd.DataFrame:
    _write_alias_map()
    if cedears.empty:
        rows = [
            {
                "model_ticker": ticker,
                "available_as_cedear": False,
                "byma_ticker": "",
                "underlying_ticker": "",
                "exchange": "",
                "ratio": "",
                "match_type": "missing_source",
                "requires_manual_confirmation": False,
                "notes": "CEDEAR source list missing; availability not confirmed",
            }
            for ticker in model_tickers
        ]
    else:
        lookup = _cedear_lookup(cedears)
        rows = []
        for ticker in model_tickers:
            clean = _clean_ticker(ticker)
            row = lookup.get(clean)
            if row is not None:
                rows.append(
                    {
                        "model_ticker": ticker,
                        "available_as_cedear": bool(row.get("tradable_from_argentina", True)),
                        "byma_ticker": row.get("byma_ticker", ""),
                        "underlying_ticker": row.get("underlying_ticker", clean),
                        "exchange": row.get("exchange", ""),
                        "ratio": row.get("ratio", ""),
                        "match_type": "exact_match",
                        "requires_manual_confirmation": False,
                        "notes": "matched_by_underlying_or_byma_ticker",
                    }
                )
                continue

            alias_row, alias_rule = _alias_lookup_row(clean, lookup)
            if alias_row is not None and alias_rule is not None:
                rows.append(
                    {
                        "model_ticker": ticker,
                        "available_as_cedear": bool(alias_row.get("tradable_from_argentina", True)),
                        "byma_ticker": alias_row.get("byma_ticker", ""),
                        "underlying_ticker": alias_row.get("underlying_ticker", alias_rule.get("cedear_underlying_ticker", "")),
                        "exchange": alias_row.get("exchange", ""),
                        "ratio": alias_row.get("ratio", ""),
                        "match_type": alias_rule.get("match_type", "alias_match"),
                        "requires_manual_confirmation": bool(alias_rule.get("requires_manual_confirmation", False)),
                        "notes": f"alias:{alias_rule.get('notes', '')}",
                    }
                )
                continue

            rule = next((r for r in MANUAL_ALIAS_RULES if _clean_ticker(r.get("model_ticker", "")) == clean), None)
            if rule is not None and bool(rule.get("requires_manual_confirmation", False)):
                notes = f"questionable_mapping_requires_manual_confirmation:{rule.get('notes', '')}"
            else:
                notes = "not_found_in_cedear_list"
            rows.append(
                {
                    "model_ticker": ticker,
                    "available_as_cedear": False,
                    "byma_ticker": "",
                    "underlying_ticker": "",
                    "exchange": "",
                    "ratio": "",
                    "match_type": "still_missing",
                    "requires_manual_confirmation": bool(rule.get("requires_manual_confirmation", False)) if rule else False,
                    "notes": notes,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(MODEL_MAP_FILE, index=False)
    _matching_audit_rows(model_tickers, out)
    growth = out[out["available_as_cedear"].astype(bool)].copy()
    growth.to_csv(CEDEAR_GROWTH_UNIVERSE_FILE, index=False)
    return out


def _price_coverage_from_snapshots() -> pd.DataFrame:
    snap = _dates(_read_csv(SNAPSHOTS_FILE))
    if snap.empty or "ticker" not in snap.columns:
        return pd.DataFrame(columns=["ticker", "first_date", "last_date", "observations"])
    grouped = snap.groupby("ticker")["date"].agg(["min", "max", "count"]).reset_index()
    grouped.columns = ["ticker", "first_date", "last_date", "observations"]
    return grouped


def feasibility_report(model_map: pd.DataFrame) -> pd.DataFrame:
    snap = _dates(_read_csv(SNAPSHOTS_FILE))
    coverage = _price_coverage_from_snapshots()
    raw_available = "raw_target_return_exact" in snap.columns if not snap.empty else False
    exact_formula_available = {"current_price", "target_price"}.issubset(snap.columns) if not snap.empty else False
    min_date = pd.to_datetime(snap["date"].min()) if not snap.empty and "date" in snap.columns else pd.NaT
    max_date = pd.to_datetime(snap["date"].max()) if not snap.empty and "date" in snap.columns else pd.NaT
    cedear_tickers = (
        set(model_map.loc[model_map["available_as_cedear"].astype(bool), "model_ticker"].astype(str)) if not model_map.empty else set()
    )
    available_snapshot_tickers = set(coverage["ticker"].astype(str)) if not coverage.empty else set()

    rows = []
    for start in START_DATES:
        requested_start_ts = pd.Timestamp(start)
        actual_start_ts = max(requested_start_ts, min_date) if pd.notna(min_date) else requested_start_ts
        tickers_with_enough_history = sorted(
            ticker
            for ticker in cedear_tickers
            if ticker in available_snapshot_tickers
            and pd.to_datetime(coverage.loc[coverage["ticker"].eq(ticker), "first_date"].iloc[0]) <= actual_start_ts
        )
        missing_tickers = sorted(cedear_tickers - set(tickers_with_enough_history))
        exact_possible = (
            pd.notna(min_date)
            and pd.notna(max_date)
            and bool(cedear_tickers)
            and requested_start_ts <= max_date
            and (raw_available or exact_formula_available)
        )
        if not bool(cedear_tickers):
            warning = "high_no_cedear_source"
        elif requested_start_ts < min_date:
            warning = "high_requested_start_before_local_snapshot_history"
        elif not exact_possible:
            warning = "high_exact_replay_not_available_for_start"
        else:
            warning = "medium_current_cedear_list_not_point_in_time"
        rows.append(
            {
                "start_date": start,
                "actual_replay_start_date": actual_start_ts.strftime("%Y-%m-%d") if pd.notna(actual_start_ts) else "",
                "historical_snapshot_start": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "",
                "historical_snapshot_end": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "",
                "cedear_model_tickers": len(cedear_tickers),
                "tickers_with_enough_history": len(tickers_with_enough_history),
                "tickers_with_enough_history_list": ",".join(tickers_with_enough_history),
                "missing_tickers": ",".join(missing_tickers[:100]),
                "price_coverage": f"{min_date.date()} to {max_date.date()}" if pd.notna(min_date) and pd.notna(max_date) else "missing",
                "raw_target_features_available": bool(raw_available or exact_formula_available),
                "raw_target_feature_source": "raw_target_return_exact"
                if raw_available
                else ("target_price/current_price exact formula" if exact_formula_available else "missing"),
                "exact_production_parity_replay_possible": bool(exact_possible),
                "survivorship_warning_level": warning,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FEASIBILITY_FILE, index=False)
    return out


def _metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "total_return": np.nan,
            "CAGR": np.nan,
            "volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Calmar": np.nan,
            "max_drawdown": np.nan,
        }
    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(r) / 52.0, 1e-9)  # weekly-ish historical decision tape
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=0) * np.sqrt(52)) if len(r) > 1 else 0.0
    sharpe = float((r.mean() * 52) / vol) if vol > 0 else np.nan
    downside = r[r < 0].std(ddof=0) * np.sqrt(52) if (r < 0).any() else np.nan
    sortino = float((r.mean() * 52) / downside) if pd.notna(downside) and downside > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "total_return": total_return,
        "CAGR": cagr,
        "volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "max_drawdown": max_dd,
    }


def _split_tickers(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [_clean_ticker(part) for part in str(value).split(",") if _clean_ticker(part)]


def _cedear_filtered_candidate_metrics(candidate: pd.DataFrame, model_map: pd.DataFrame) -> dict[str, object] | None:
    if model_map.empty or "available_as_cedear" not in model_map.columns:
        return None
    cedear_tickers = set(model_map.loc[model_map["available_as_cedear"].astype(bool), "model_ticker"].astype(str).map(_clean_ticker))
    if not cedear_tickers:
        return None

    realized = _dates(_read_csv(REALIZED_FILE))
    if realized.empty or not {"date", "ticker", "realized_return_5d"}.issubset(realized.columns):
        return {
            "model": "growth_champion_v2_cedear_filtered",
            "status": "not_run",
            "reason": "historical_realized_returns_missing",
        }
    realized = realized.copy()
    realized["ticker"] = realized["ticker"].astype(str).map(_clean_ticker)
    realized["realized_return_5d"] = _num(realized["realized_return_5d"])
    realized_lookup = realized.drop_duplicates(subset=["date", "ticker"]).set_index(["date", "ticker"])["realized_return_5d"]

    returns = []
    selected_counts = []
    cash_values = []
    coverage_hits = 0
    for _, row in candidate.iterrows():
        date = pd.Timestamp(row["date"])
        selected = _split_tickers(row.get("selected_tickers", ""))
        cedear_selected = [ticker for ticker in selected if ticker in cedear_tickers]
        exposure = float(_num(pd.Series([row.get("target_exposure", row.get("exposure", 0.0))])).fillna(0.0).iloc[0])
        exposure = float(np.clip(exposure, 0.0, 1.0))
        asset_returns = []
        for ticker in cedear_selected:
            key = (date, ticker)
            if key in realized_lookup.index:
                val = realized_lookup.loc[key]
                if pd.notna(val):
                    asset_returns.append(float(val))
        if asset_returns:
            period_return = exposure * float(np.mean(asset_returns))
            coverage_hits += len(asset_returns)
            cash = 1.0 - exposure
        else:
            period_return = 0.0
            cash = 1.0
        returns.append(period_return)
        selected_counts.append(len(asset_returns))
        cash_values.append(cash)

    metrics = _metrics_from_returns(pd.Series(returns, dtype=float))
    return {
        "model": "growth_champion_v2_cedear_filtered",
        "status": "cedear_post_selection_filtered_not_reoptimized",
        "start_date": candidate["date"].min().strftime("%Y-%m-%d"),
        "end_date": candidate["date"].max().strftime("%Y-%m-%d"),
        "observations": len(candidate),
        "cedear_available_tickers": len(cedear_tickers),
        "average_cedear_selected_count": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "average_cash": float(np.mean(cash_values)) if cash_values else 1.0,
        "realized_return_observations_used": int(coverage_hits),
        "reason": "filtered existing Growth Champion v2 selections to CEDEAR-available tickers; no reoptimization; not point-in-time",
        **metrics,
    }


def run_extended_backtest_if_feasible(
    feasibility: pd.DataFrame, model_map: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = _dates(_read_csv(GROWTH_DAILY_FILE))
    if daily.empty:
        reason = "missing_growth_head_to_head_daily_returns"
        results = pd.DataFrame([{"model": "growth_champion_v2", "status": "not_run", "reason": reason}])
        benchmarks = pd.DataFrame(
            [{"benchmark": "SPY", "status": "not_run", "reason": reason}, {"benchmark": "QQQ", "status": "not_run", "reason": reason}]
        )
        governance = pd.DataFrame([{"classification": "blocked", "reason": reason}])
    else:
        selector = daily.get("candidate", pd.Series(index=daily.index, dtype=str)).astype(str).eq("growth_v1_exposure_cap_60")
        if "growth_paper_variant" in daily.columns:
            selector = selector | daily["growth_paper_variant"].astype(str).eq("growth_v1_exposure_cap_60")
        candidate = daily[selector].copy()
        if candidate.empty:
            reason = "growth_champion_v2_daily_series_missing"
            results = pd.DataFrame([{"model": "growth_champion_v2", "status": "not_run", "reason": reason}])
            benchmarks = pd.DataFrame(
                [{"benchmark": "SPY", "status": "not_run", "reason": reason}, {"benchmark": "QQQ", "status": "not_run", "reason": reason}]
            )
            governance = pd.DataFrame([{"classification": "blocked", "reason": reason}])
        else:
            ret_col = "return" if "return" in candidate.columns else "vol_target_return"
            metrics = _metrics_from_returns(_num(candidate[ret_col]))
            start = candidate["date"].min().strftime("%Y-%m-%d")
            end = candidate["date"].max().strftime("%Y-%m-%d")
            base_row = {
                "model": "growth_champion_v2",
                "status": "reference_only_existing_non_cedear_filtered_series",
                "start_date": start,
                "end_date": end,
                "observations": len(candidate),
                **metrics,
            }
            result_rows = [base_row]
            if model_map is not None:
                filtered_row = _cedear_filtered_candidate_metrics(candidate, model_map)
                if filtered_row is not None:
                    result_rows.append(filtered_row)
            results = pd.DataFrame(result_rows)
            prod_bench = _read_csv(PRODUCTION_BENCHMARK_FILE)
            rows = []
            if not prod_bench.empty:
                for _, row in prod_bench.iterrows():
                    rows.append(
                        {
                            "benchmark": row.get("benchmark", ""),
                            "candidate_return": row.get("candidate_return", metrics["total_return"]),
                            "benchmark_return": row.get("benchmark_return", np.nan),
                            "return_gap": row.get("return_gap", np.nan),
                            "candidate_Sharpe": row.get("candidate_Sharpe", metrics["Sharpe"]),
                            "benchmark_Sharpe": row.get("benchmark_Sharpe", np.nan),
                            "Sharpe_gap": row.get("Sharpe_gap", np.nan),
                            "candidate_max_drawdown": row.get("candidate_max_drawdown", metrics["max_drawdown"]),
                            "benchmark_max_drawdown": row.get("benchmark_max_drawdown", np.nan),
                            "DD_gap": row.get("DD_gap", np.nan),
                            "note": "benchmark comparison from existing production parity benchmark file",
                        }
                    )
            benchmarks = pd.DataFrame(rows) if rows else pd.DataFrame([{"benchmark": "SPY/QQQ", "status": "benchmark_series_missing"}])
            has_cedear_universe = bool(not feasibility.empty and feasibility["cedear_model_tickers"].fillna(0).astype(int).max() > 0)
            feasible_2022 = bool(
                has_cedear_universe
                and not feasibility.empty
                and feasibility.loc[feasibility["start_date"].eq("2022-01-01"), "exact_production_parity_replay_possible"]
                .astype(bool)
                .any()
            )
            if not has_cedear_universe:
                classification = "blocked_missing_cedear_source"
                reason = "CEDEAR source list missing; generated growth result is reference-only and not CEDEAR-filtered"
            elif feasible_2022:
                classification = "cedear_filtered_research_possible_from_2022_actual_start"
                reason = "CEDEAR-filtered reference replay possible from first local snapshot date; current CEDEAR list is not point-in-time and starts before 2022 are not feasible locally"
            else:
                classification = "blocked_extended_replay_not_feasible"
                reason = "CEDEAR list available but exact production-parity replay not feasible for requested starts"
            governance = pd.DataFrame(
                [
                    {
                        "classification": classification,
                        "reason": reason,
                        "production_changed": False,
                        "parameter_tuning": False,
                        "point_in_time_universe_confirmed": False,
                    }
                ]
            )
    results.to_csv(EXTENDED_RESULTS_FILE, index=False)
    benchmarks.to_csv(EXTENDED_BENCHMARK_FILE, index=False)
    governance.to_csv(GOVERNANCE_FILE, index=False)
    return results, benchmarks, governance


def run_phase_67(cedear_file: str | None = None) -> None:
    source = _discover_cedear_file(cedear_file)
    cedears = parse_cedear_list(source)
    model_tickers = current_model_universe()
    model_map = map_model_tickers_to_cedears(model_tickers, cedears)
    feasibility = feasibility_report(model_map)
    results, benchmarks, governance = run_extended_backtest_if_feasible(feasibility, model_map)

    print("\n===== CEDEAR UNIVERSE AUDIT =====")
    print(f"cedear source: {source if source else 'missing_cedear_source'}")
    print(f"cedear rows parsed: {len(cedears)}")
    print(
        f"tradable_from_argentina rows: {int(cedears['tradable_from_argentina'].sum()) if 'tradable_from_argentina' in cedears.columns and not cedears.empty else 0}"
    )

    print("\n===== MODEL TICKER TO CEDEAR MAP =====")
    available = int(model_map["available_as_cedear"].sum()) if not model_map.empty else 0
    print(f"model tickers: {len(model_map)}")
    print(f"available as CEDEAR: {available}")
    print(model_map.head(20).to_string(index=False))

    print("\n===== EXTENDED BACKTEST FEASIBILITY =====")
    print(feasibility.to_string(index=False))

    print("\n===== GROWTH CHAMPION V2 EXTENDED BACKTEST =====")
    print(results.to_string(index=False))

    print("\n===== GOVERNANCE =====")
    print(governance.to_string(index=False))
    print("\nFiles generated:")
    for path in [
        CEDEAR_UNIVERSE_FILE,
        MODEL_MAP_FILE,
        CEDEAR_GROWTH_UNIVERSE_FILE,
        FEASIBILITY_FILE,
        EXTENDED_RESULTS_FILE,
        EXTENDED_BENCHMARK_FILE,
        GOVERNANCE_FILE,
    ]:
        print(f"- {Path(path).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CEDEAR tradable universe and extended growth backtest feasibility.")
    parser.add_argument("--cedear-file", default=None, help="Optional path to the uploaded CEDEAR list CSV/XLSX.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_phase_67(args.cedear_file)
