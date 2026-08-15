#!/usr/bin/env python3
"""Project entry point for La Máquina.

This provides a small, stable command surface for the trading/research workflow
without requiring direct imports from the legacy flat-script layout.
"""

from __future__ import annotations

import argparse
import importlib
import sys


def _smoke_test() -> int:
    modules = [
        "src.quant_research_features",
        "src.portfolio_optimizer",
        "src.market_regime_model",
        "src.expected_returns_model",
        "src.quant_target_model",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
    print("smoke-ok")
    return 0


def _run_module(module_name: str) -> int:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise ValueError(f"Module '{module_name}' does not expose a main() entry point.")
    main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="La Máquina project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="Validate the core project modules import cleanly").set_defaults(func=lambda _args: _smoke_test())
    subparsers.add_parser("research", help="Run the main research engine").set_defaults(func=lambda _args: _run_module("financial_data_system"))
    subparsers.add_parser("daily", help="Run the daily orchestrator").set_defaults(func=lambda _args: _run_module("daily_research_run"))
    subparsers.add_parser("dashboard", help="Launch the dashboard app").set_defaults(func=lambda _args: _run_module("dashboard_app"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - CLI surface; we want a clean failure message.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
