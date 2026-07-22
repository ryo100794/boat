#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from boatrace_ai.listwise.market_policy_diagnostics import (
    forward_policy_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forward-only diagnostics for real-odds bankroll policy selection."
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cached = joblib.load(args.cache)
    races = cached.get("races") if isinstance(cached, dict) else None
    if not isinstance(races, list):
        raise ValueError("cache does not contain a races list")
    result = forward_policy_diagnostics(
        races,
        regularization=args.regularization,
        daily_budget_yen=args.daily_budget_yen,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
