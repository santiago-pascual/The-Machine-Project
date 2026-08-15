from __future__ import annotations

import argparse

from historical_research_backfill import (
    HistoricalBackfillConfig,
    run_historical_research_backfill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical research backfill without touching daily/production files.")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--step-size-days", type=int, default=5)
    parser.add_argument("--max-test-dates", type=int, default=None)
    parser.add_argument("--period", default="7y")
    parser.add_argument("--full-universe", action="store_true")
    parser.add_argument("--include-full-quant", action="store_true")
    parser.add_argument("--optimizer-generations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_modes = ["baseline", "regime_gated_full_quant"]
    if args.include_full_quant:
        model_modes.append("full_quant_research")
    config = HistoricalBackfillConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        step_size_days=args.step_size_days,
        max_test_dates=args.max_test_dates,
        model_modes=model_modes,
        reduced_universe=not args.full_universe,
        optimizer_generations_backtest=args.optimizer_generations,
        period=args.period,
    )
    run_historical_research_backfill(config)


if __name__ == "__main__":
    main()
