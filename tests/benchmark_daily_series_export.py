from __future__ import annotations

from pathlib import Path

import pandas as pd

PAPER_FILE = Path("growth_candidate_paper_performance.csv")
CACHE_DIR = Path("yahoo_ohlcv_price_cache")
RETURNS_OUT = Path("benchmark_daily_returns.csv")
EQUITY_OUT = Path("benchmark_equity_curves.csv")


def _normalize_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    return out.dropna(subset=[column]).sort_values(column)


def _read_paper_tracking() -> pd.DataFrame:
    if not PAPER_FILE.exists():
        raise FileNotFoundError(f"Missing paper tracking file: {PAPER_FILE}")
    paper = pd.read_csv(PAPER_FILE)
    if "date" not in paper.columns or "daily_return" not in paper.columns:
        raise ValueError("growth_candidate_paper_performance.csv requires date and daily_return columns")
    paper = _normalize_date(paper, "date")
    paper["growth_daily_return"] = pd.to_numeric(paper["daily_return"], errors="coerce").fillna(0.0)
    return paper[["date", "growth_daily_return"]].drop_duplicates("date", keep="last")


def _download_yahoo(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return pd.DataFrame()
    try:
        yf_cache = Path(".yfinance_cache")
        yf_cache.mkdir(exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(yf_cache))
    except Exception:
        pass
    try:
        # Yahoo end is exclusive; add a few days so the last paper date can be included.
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    return df


def _load_benchmark_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    cache_path = CACHE_DIR / f"{ticker}.csv"
    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
    downloaded = _download_yahoo(ticker, start, end)
    if not downloaded.empty:
        combined = pd.concat([cached, downloaded], ignore_index=True) if not cached.empty else downloaded
        if "Date" in combined.columns:
            combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.normalize()
            combined = combined.dropna(subset=["Date"]).drop_duplicates("Date", keep="last").sort_values("Date")
            CACHE_DIR.mkdir(exist_ok=True)
            combined.to_csv(cache_path, index=False)
        source = "yfinance_updated_cache"
    else:
        combined = cached
        source = "local_cache"
    if combined.empty or "Date" not in combined.columns:
        return pd.DataFrame(), "missing"
    price_col = "Adj Close" if "Adj Close" in combined.columns else "Close"
    prices = combined[["Date", price_col]].rename(columns={"Date": "date", price_col: ticker.lower() + "_price"})
    prices = _normalize_date(prices, "date")
    prices[ticker.lower() + "_price"] = pd.to_numeric(prices[ticker.lower() + "_price"], errors="coerce")
    return prices.dropna(subset=[ticker.lower() + "_price"]), source


def build_benchmark_exports() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paper = _read_paper_tracking()
    start, end = paper["date"].min(), paper["date"].max()
    sources: dict[str, str] = {}
    aligned = paper.copy()
    for ticker in ["SPY", "QQQ"]:
        prices, source = _load_benchmark_prices(ticker, start, end)
        sources[ticker] = source
        price_col = ticker.lower() + "_price"
        if prices.empty:
            aligned[price_col] = pd.NA
            continue
        # Align benchmark closes to the same paper dates. If the newest date is not yet
        # available from Yahoo/cache, it remains NaN rather than inventing a return.
        aligned = aligned.merge(prices[["date", price_col]], on="date", how="left")

    returns = aligned[["date", "growth_daily_return"]].copy()
    for ticker in ["spy", "qqq"]:
        price_col = ticker + "_price"
        return_col = ticker + "_daily_return"
        if price_col in aligned.columns:
            returns[return_col] = pd.to_numeric(aligned[price_col], errors="coerce").pct_change().fillna(0.0)
            returns.loc[aligned[price_col].isna(), return_col] = pd.NA
        else:
            returns[return_col] = pd.NA

    equity = returns[["date"]].copy()
    for source_col, out_col in [
        ("growth_daily_return", "growth_cumulative_return_pct"),
        ("spy_daily_return", "spy_cumulative_return_pct"),
        ("qqq_daily_return", "qqq_cumulative_return_pct"),
    ]:
        vals = pd.to_numeric(returns[source_col], errors="coerce")
        equity[out_col] = ((1 + vals.fillna(0.0)).cumprod() - 1) * 100
        equity.loc[vals.isna(), out_col] = pd.NA

    returns.to_csv(RETURNS_OUT, index=False)
    equity.to_csv(EQUITY_OUT, index=False)
    return returns, equity, sources


def main() -> None:
    returns, equity, sources = build_benchmark_exports()
    print("===== BENCHMARK DAILY SERIES EXPORT =====")
    print(f"paper dates: {len(returns)}")
    print(f"date range: {returns['date'].min().date()} to {returns['date'].max().date()}")
    print(f"SPY source: {sources.get('SPY')}")
    print(f"QQQ source: {sources.get('QQQ')}")
    print(f"saved: {RETURNS_OUT}")
    print(f"saved: {EQUITY_OUT}")
    missing = [c for c in ["spy_daily_return", "qqq_daily_return"] if returns[c].isna().any()]
    if missing:
        print(f"warning: missing benchmark values in columns: {missing}")


if __name__ == "__main__":
    main()
