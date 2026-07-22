#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from boatrace_ai.listwise.market_diagnostics import calibrator_stability_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report day-level stability of market calibration candidates."
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache = joblib.load(args.cache)
    races = cache.get("races") if isinstance(cache, dict) else None
    if not isinstance(races, list):
        raise ValueError("cache does not contain a races list")
    rows = calibrator_stability_rows(races)
    payload = {
        "cache": str(args.cache),
        "races": len(races),
        "days": len({str(race["race_date"]) for race in races}),
        "best_pooled": min(rows, key=lambda row: row["pooled_log_loss"]),
        "best_worst_day": min(
            rows,
            key=lambda row: (
                row["worst_daily_market_regret"],
                row["mean_daily_market_regret"],
                row["pooled_log_loss"],
            ),
        ),
        "candidates": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
