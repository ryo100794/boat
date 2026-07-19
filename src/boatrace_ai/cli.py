from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

from .db import connection, init_db, race_id
from .runtime.time_semantics import operational_race_date


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boat-ai",
        description="BOAT RACE historical backtest, realtime collection, prediction, and web UI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="Create or migrate the SQLite database.")
    add_db_arg(p)
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("backfill", help="Fetch historical official LZH data.")
    add_db_arg(p)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--kind", choices=["program", "result", "both"], default="both")
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--no-skip-existing", action="store_true")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("fetch-racer-stats", help="Fetch official racer period stats.")
    add_db_arg(p)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--from-year", type=int, required=True)
    p.add_argument("--to-year", type=int, required=True)
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--no-skip-existing", action="store_true")
    p.set_defaults(func=cmd_fetch_racer_stats)

    p = sub.add_parser("backtest", help="Backtest using historical data only by default.")
    add_db_arg(p)
    p.add_argument("--output", default="data/models/backtest.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-train-races", type=int, default=500)
    p.add_argument("--include-odds", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("train", help="Train the lane winner model.")
    add_db_arg(p)
    p.add_argument("--model", default="data/models/win_model.joblib")
    p.add_argument("--through-date")
    p.add_argument("--include-odds", action="store_true")
    p.add_argument("--min-examples", type=int, default=100)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("collect-live-once", help="Collect current race pages once.")
    add_db_arg(p)
    add_live_args(p)
    p.set_defaults(func=cmd_collect_live_once)

    p = sub.add_parser(
        "monitor",
        help="Collect realtime odds, update predictions, and optionally retrain.",
    )
    add_db_arg(p)
    add_live_args(p)
    p.add_argument("--model", default="data/models/win_model.joblib")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--max-loops", type=int)
    p.add_argument("--retrain-every", type=int, default=0)
    p.add_argument("--include-odds", action="store_true")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("predict", help="Predict one race and store the result.")
    add_db_arg(p)
    p.add_argument("--model", default="data/models/win_model.joblib")
    p.add_argument("--date", required=True)
    p.add_argument("--jcd", required=True)
    p.add_argument("--rno", type=int, required=True)
    p.add_argument("--top-n", type=int, default=30)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("serve", help="Start the realtime prediction web UI.")
    add_db_arg(p)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=10001)
    p.add_argument("--backtest", default="data/models/backtest.json")
    p.set_defaults(func=cmd_serve)

    return parser


def add_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/boatrace.sqlite")


def add_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date", help="Fix one race date; omit to follow the current JST date automatically.")
    parser.add_argument("--jcd")
    parser.add_argument("--rno", type=int)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--sleep", type=float, default=0.4)


def cmd_init_db(args: argparse.Namespace) -> int:
    init_db(args.db)
    print(json.dumps({"db": args.db, "status": "initialized"}, ensure_ascii=False))
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    init_db(args.db)
    end = _parse_date(args.end) if args.end else date.today()
    start = _parse_date(args.start) if args.start else end - timedelta(days=365 * args.years)
    from .ingestion.backfill import backfill_historical

    with connection(args.db) as conn:
        stats = backfill_historical(
            conn,
            start=start,
            end=end,
            kind=args.kind,
            raw_dir=Path(args.raw_dir),
            sleep_seconds=args.sleep,
            skip_existing=not args.no_skip_existing,
        )
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
    return 0


def cmd_fetch_racer_stats(args: argparse.Namespace) -> int:
    init_db(args.db)
    from .ingestion.historical import fetch_racer_stats

    with connection(args.db) as conn:
        stored = fetch_racer_stats(
            conn,
            from_year=args.from_year,
            to_year=args.to_year,
            raw_dir=Path(args.raw_dir),
            sleep_seconds=args.sleep,
            skip_existing=not args.no_skip_existing,
        )
    print(json.dumps({"stored": stored}, ensure_ascii=False, indent=2))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    init_db(args.db)
    from .modeling import backtest_model

    with connection(args.db) as conn:
        result = backtest_model(
            conn,
            output_path=Path(args.output),
            folds=args.folds,
            include_odds=args.include_odds,
            min_train_races=args.min_train_races,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    init_db(args.db)
    from .modeling import train_model

    with connection(args.db) as conn:
        result = train_model(
            conn,
            model_path=Path(args.model),
            include_odds=args.include_odds,
            through_date=args.through_date,
            min_examples=args.min_examples,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_collect_live_once(args: argparse.Namespace) -> int:
    init_db(args.db)
    from .ingestion.live import collect_live_once

    with connection(args.db) as conn:
        result = collect_live_once(
            conn,
            race_date=operational_race_date(_parse_date(args.date) if args.date else None),
            raw_dir=Path(args.raw_dir),
            sleep_seconds=args.sleep,
            jcd=args.jcd,
            rno=args.rno,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    init_db(args.db)
    from .ingestion.live import collect_live_once
    from .modeling import predict_open_races, train_model

    fixed_date = _parse_date(args.date) if args.date else None
    model_path = Path(args.model)
    loop = 0
    while True:
        race_date = operational_race_date(fixed_date)
        with connection(args.db) as conn:
            collected = collect_live_once(
                conn,
                race_date=race_date,
                raw_dir=Path(args.raw_dir),
                sleep_seconds=args.sleep,
                jcd=args.jcd,
                rno=args.rno,
            )
            predicted = {"predicted": 0, "failed": 0}
            if model_path.exists():
                predicted = predict_open_races(
                    conn,
                    model_path=model_path,
                    race_date=race_date,
                    jcd=args.jcd,
                    rno=args.rno,
                )
            if args.retrain_every and loop > 0 and loop % args.retrain_every == 0:
                train_model(
                    conn,
                    model_path=model_path,
                    include_odds=args.include_odds,
                    min_examples=100,
                )
        print(
            json.dumps(
                {"loop": loop, "collected": collected, "predicted": predicted},
                ensure_ascii=False,
            ),
            flush=True,
        )
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.interval)


def cmd_predict(args: argparse.Namespace) -> int:
    init_db(args.db)
    from .modeling import predict_race

    rid = race_id(_parse_date(args.date).isoformat(), args.jcd, args.rno)
    with connection(args.db) as conn:
        rows = predict_race(
            conn,
            model_path=Path(args.model),
            race_id_value=rid,
            top_n=args.top_n,
            store=True,
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    init_db(args.db)
    from .server import serve

    print(f"Serving BOAT RACE AI Monitor on http://{args.host}:{args.port}", flush=True)
    serve(
        db_path=Path(args.db),
        host=args.host,
        port=args.port,
        backtest_path=Path(args.backtest) if args.backtest else None,
    )
    return 0


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


if __name__ == "__main__":
    raise SystemExit(main())
