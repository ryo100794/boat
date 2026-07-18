from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from .db import connection, init_db
from .modeling import backtest_model, train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Periodically run historical-only backtest and model training."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model.joblib")
    parser.add_argument("--backtest", default="data/models/backtest.json")
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--min-train-races", type=int, default=50)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-loops", type=int)
    args = parser.parse_args(argv)

    init_db(args.db)
    loop = 0
    while True:
        event = {"loop": loop, "trained": False, "backtested": False}
        try:
            with connection(args.db) as conn:
                counts = dataset_counts(conn)
                event["counts"] = counts
                if counts["examples"] >= args.min_examples:
                    train_meta = train_model(
                        conn,
                        model_path=Path(args.model),
                        include_odds=False,
                        min_examples=args.min_examples,
                    )
                    event["trained"] = True
                    event["train"] = train_meta
                if counts["races"] >= args.min_train_races + args.folds:
                    bt = backtest_model(
                        conn,
                        output_path=Path(args.backtest),
                        folds=args.folds,
                        include_odds=False,
                        min_train_races=args.min_train_races,
                    )
                    event["backtested"] = True
                    event["backtest"] = {
                        "evaluated_races": bt.get("evaluated_races"),
                        "winner_top1_accuracy": bt.get("winner_top1_accuracy"),
                        "trifecta_top5_hit_rate": bt.get("trifecta_top5_hit_rate"),
                    }
        except Exception as exc:
            event["error"] = str(exc)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.interval)


def dataset_counts(conn: sqlite3.Connection) -> dict[str, int]:
    examples = conn.execute(
        """
        SELECT COUNT(*)
        FROM entries e
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE rr.rank IS NOT NULL
        """
    ).fetchone()[0]
    races = conn.execute(
        """
        SELECT COUNT(DISTINCT e.race_id)
        FROM entries e
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE rr.rank IS NOT NULL
        """
    ).fetchone()[0]
    return {"examples": int(examples), "races": int(races)}


if __name__ == "__main__":
    raise SystemExit(main())
