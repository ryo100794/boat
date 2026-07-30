from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..listwise.market_calibration import write_json_atomic
from ..listwise.market_promotion import promote_best_candidate
from .promotion_candidates import discover_market_evaluation_candidates
from .prospective_market_promotion import write_prospective_candidate


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    try:
        candidates = list(args.candidate)
        queue_dir = getattr(args, "evaluation_queue_dir", None)
        if queue_dir:
            queued = discover_market_evaluation_candidates(queue_dir)
            candidates.extend(queued)
            derived_dir = (
                Path(args.state).resolve().parent
                / "market_promotion_candidates"
            )
            for source_path in queued:
                derived = write_prospective_candidate(
                    source_path,
                    output_dir=derived_dir,
                )
                if derived is not None:
                    candidates.append(derived)
        candidates = list(dict.fromkeys(candidates))
        result = promote_best_candidate(
            candidates,
            output_path=args.output,
        )
        status = "ok"
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        status = "error"
    event = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "promotion": result,
    }
    write_json_atomic(Path(args.state), event)
    return event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously verify market shadow promotion artifacts."
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--evaluation-queue-dir")
    parser.add_argument("--output", default="data/models/active_market_model.json")
    parser.add_argument(
        "--state", default="data/models/market_promotion_cycle_state.json"
    )
    parser.add_argument("--interval", type=float, default=3600.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    while True:
        event = run_once(args)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if args.once:
            return 0 if event["status"] == "ok" else 1
        time.sleep(max(60.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
