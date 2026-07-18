from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from .db import connection, init_db
from .modeling_no_odds_v8_modelrank import predict_open_races


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Periodic no-odds v8 model-probability prediction updater.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v8.joblib")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--max-loops", type=int)
    args = parser.parse_args(argv)

    init_db(args.db)
    target_date = date.fromisoformat(args.date)
    model_path = Path(args.model)
    loop = 0
    while True:
        event = {"loop": loop, "predicted": 0, "failed": 0, "model": str(model_path), "rank_basis": "model_probability"}
        try:
            if model_path.exists():
                with connection(args.db) as conn:
                    event.update(predict_open_races(conn, model_path=model_path, race_date=target_date))
            else:
                event["error"] = "model file does not exist"
        except Exception as exc:
            event["error"] = str(exc)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

