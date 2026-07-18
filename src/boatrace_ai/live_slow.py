from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from .constants import RACES_PER_DAY, VENUES
from .db import connection, init_db
from .live import collect_beforeinfo, collect_odds, collect_racelist, collect_result
from .modeling import predict_open_races


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Slow all-venue live collection and prediction loop."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model.joblib")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--sleep-page", type=float, default=2.0)
    parser.add_argument("--sleep-loop", type=float, default=120.0)
    parser.add_argument("--max-loops", type=int)
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args(argv)

    init_db(args.db)
    target_date = date.fromisoformat(args.date)
    raw_dir = Path(args.raw_dir)
    model_path = Path(args.model)
    loop = 0
    while True:
        counters = {
            "loop": loop,
            "racelist": 0,
            "beforeinfo": 0,
            "odds": 0,
            "result_rows": 0,
            "predicted": 0,
            "prediction_failed": 0,
        }
        with connection(args.db) as conn:
            for venue in VENUES:
                for rno in RACES_PER_DAY:
                    if collect_racelist(
                        conn,
                        race_date=target_date,
                        jcd=venue.code,
                        rno=rno,
                        raw_dir=raw_dir,
                    ):
                        counters["racelist"] += 1
                    conn.commit()
                    time.sleep(args.sleep_page)

                    if collect_beforeinfo(
                        conn,
                        race_date=target_date,
                        jcd=venue.code,
                        rno=rno,
                        raw_dir=raw_dir,
                    ):
                        counters["beforeinfo"] += 1
                    conn.commit()
                    time.sleep(args.sleep_page)

                    if collect_odds(
                        conn,
                        race_date=target_date,
                        jcd=venue.code,
                        rno=rno,
                        raw_dir=raw_dir,
                    ):
                        counters["odds"] += 1
                    conn.commit()
                    time.sleep(args.sleep_page)

                    counters["result_rows"] += collect_result(
                        conn,
                        race_date=target_date,
                        jcd=venue.code,
                        rno=rno,
                        raw_dir=raw_dir,
                    )
                    conn.commit()
                    time.sleep(args.sleep_page)

            if args.predict and model_path.exists():
                predicted = predict_open_races(
                    conn,
                    model_path=model_path,
                    race_date=target_date,
                )
                counters["predicted"] = predicted["predicted"]
                counters["prediction_failed"] = predicted["failed"]
                conn.commit()
        print(json.dumps(counters, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.sleep_loop)


if __name__ == "__main__":
    raise SystemExit(main())
