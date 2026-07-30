from __future__ import annotations

import argparse
import json
from pathlib import Path

from boatrace_ai.web.v21_historical_backtest import build_projection, write_projection


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize one V21 walk-forward dashboard day")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    payload = build_projection(result, race_date=args.date, source_job_id=args.job_id)
    write_projection(args.output, payload)
    print(json.dumps(payload["stats"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
