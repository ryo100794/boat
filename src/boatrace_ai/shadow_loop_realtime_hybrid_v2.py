from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from .db import connection, init_db
from .modeling_realtime_hybrid_v2 import predict_open_races


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Periodic realtime hybrid v2 shadow prediction writer.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_realtime_hybrid_v2.joblib")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output", default="data/models/shadow_realtime_hybrid_v2_latest.json")
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--max-loops", type=int)
    args = parser.parse_args(argv)

    init_db(args.db)
    target_date = date.fromisoformat(args.date)
    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loop = 0
    while True:
        event = {
            "loop": loop,
            "model": str(model_path),
            "output": str(output_path),
            "role": "shadow_realtime_hybrid_v2",
            "predicted": 0,
            "failed": 0,
        }
        try:
            if model_path.exists():
                with connection(args.db) as conn:
                    payload = predict_open_races(conn, model_path=model_path, race_date=target_date)
                output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                event.update({key: payload.get(key) for key in ("generated_at", "predicted", "failed", "feature_set")})
            else:
                event["error"] = "model file does not exist"
                output_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            event["error"] = str(exc)
            output_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
