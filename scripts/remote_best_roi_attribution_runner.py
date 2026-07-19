#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boatrace_ai.adaptive_bankroll_pastlog_v7 import adaptive_bankroll_streaming
from boatrace_ai.db import connection, init_db


DEFAULT_CANDIDATES = (
    "data/models/m6_norm_ev105_top80_19d5c35.json",
    "data/models/m6_norm_ev110_top40_19d5c35.json",
    "data/models/m6_norm_ev120_top30_19d5c35.json",
    "data/models/m6_norm_ev150_top20_19d5c35.json",
    "data/models/m6_norm_ev200_top10_19d5c35.json",
)


def wait_for_results(
    paths: list[Path],
    *,
    poll_seconds: int,
    timeout_seconds: int,
) -> list[tuple[Path, dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    last_ready = -1
    while True:
        rows = load_results(paths)
        if len(rows) != last_ready:
            print(
                json.dumps(
                    {
                        "at": now(),
                        "status": "waiting" if len(rows) < len(paths) else "ready",
                        "ready": len(rows),
                        "expected": len(paths),
                    }
                ),
                flush=True,
            )
            last_ready = len(rows)
        if len(rows) == len(paths):
            return rows
        if time.monotonic() >= deadline:
            if rows:
                print(json.dumps({"at": now(), "status": "timeout_partial", "ready": len(rows)}), flush=True)
                return rows
            raise TimeoutError("no completed bankroll sweep results before timeout")
        time.sleep(max(10, poll_seconds))


def load_results(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            float(payload["roi"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        rows.append((path, payload))
    return rows


def select_best(rows: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any]]:
    if not rows:
        raise ValueError("no valid sweep results")
    return max(
        rows,
        key=lambda item: (
            float(item[1].get("roi") or 0.0),
            float(item[1].get("profit_yen") or 0.0),
            -float(item[1].get("max_drawdown_yen") or 0.0),
        ),
    )


def policy_kwargs(policy: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "daily_budget_yen",
        "ev_threshold",
        "payout_prior_weight",
        "require_real_odds",
        "fractional_kelly",
        "max_daily_exposure_fraction",
        "min_daily_exposure_fraction",
        "race_cap_fraction",
        "ticket_cap_fraction",
        "max_daily_tickets",
        "allocation_mode",
        "stake_granularity_yen",
        "min_stake_yen",
    )
    return {key: policy[key] for key in keys if policy.get(key) is not None}


def supported_policy_kwargs(policy: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    kwargs = policy_kwargs(policy)
    supported = inspect.signature(adaptive_bankroll_streaming).parameters
    dropped = sorted(key for key in kwargs if key not in supported)
    return {key: value for key, value in kwargs.items() if key in supported}, dropped

def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ROI attribution using the best completed normalized-Kelly sweep.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/m6_best_roi_attribution.json")
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=86_400)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=24_000)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists() and output.stat().st_size:
        print(json.dumps({"at": now(), "status": "already_complete", "output": str(output)}), flush=True)
        return 0

    candidate_paths = [Path(value) for value in (args.candidates or DEFAULT_CANDIDATES)]
    rows = wait_for_results(
        candidate_paths,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    source_path, source = select_best(rows)
    kwargs, dropped_kwargs = supported_policy_kwargs(source.get("policy") or {})
    print(
        json.dumps(
            {
                "at": now(),
                "status": "selected",
                "source": str(source_path),
                "source_roi": source.get("roi"),
                "source_profit_yen": source.get("profit_yen"),
                "policy": kwargs,
                "dropped_unsupported_policy": dropped_kwargs,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    init_db(args.db)
    with connection(args.db) as conn:
        result = adaptive_bankroll_streaming(
            conn,
            output_path=output,
            folds=args.folds,
            min_train_races=args.min_train_races,
            batch_size=args.batch_size,
            epochs=args.epochs,
            **kwargs,
        )
    print(
        json.dumps(
            {
                "at": now(),
                "status": "complete",
                "output": str(output),
                "roi": result.get("roi"),
                "profit_yen": result.get("profit_yen"),
                "stable_signals": ((result.get("ticket_roi_attribution") or {}).get("fold_stability") or {}).get("stable_signals"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
