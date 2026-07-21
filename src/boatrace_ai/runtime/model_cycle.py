from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from ..db import connection, init_db
from ..modeling import backtest_model, train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Periodically run historical or realtime-odds shadow evaluation."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model.joblib")
    parser.add_argument("--backtest", default="data/models/backtest.json")
    parser.add_argument("--state", default="data/models/model_cycle_state.json")
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--min-train-races", type=int, default=50)
    parser.add_argument("--min-eligible-races", type=int, default=0)
    parser.add_argument("--min-new-races", type=int, default=0)
    parser.add_argument("--min-odds-snapshots", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--from-date")
    parser.add_argument("--include-odds", action="store_true")
    parser.add_argument("--max-loops", type=int)
    args = parser.parse_args(argv)

    init_db(args.db)
    state_path = Path(args.state) if args.state else None
    previous_state = read_state(state_path) if state_path else {}
    last_evaluated_races = int(previous_state.get("last_evaluated_races") or 0)
    last_evaluated_at = previous_state.get("last_evaluated_at")
    model_path = Path(args.model)
    backtest_path = Path(args.backtest)
    loop = 0
    while True:
        event = {
            "loop": loop,
            "trained": False,
            "backtested": False,
            "mode": "realtime_odds_shadow" if args.include_odds else "historical_only",
            "from_date": args.from_date,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        if last_evaluated_races:
            event["last_evaluated_races"] = last_evaluated_races
            event["last_evaluated_at"] = last_evaluated_at
        try:
            with connection(args.db) as conn:
                counts = dataset_counts(
                    conn,
                    from_date=args.from_date,
                    require_odds=args.include_odds,
                    min_odds_snapshots=args.min_odds_snapshots,
                )
                eligible_races = counts["odds_result_races"] if args.include_odds else counts["races"]
                required_races = max(args.min_eligible_races, args.min_train_races + args.folds)
                ready = eligible_races >= required_races
                event.update(
                    {
                        "counts": counts,
                        "eligible_races": eligible_races,
                        "required_races": required_races,
                        "ready": ready,
                        "readiness": min(1.0, eligible_races / max(1, required_races)),
                    }
                )
                if not ready:
                    event["status"] = "waiting_for_data"
                elif counts["examples"] >= args.min_examples:
                    due = evaluation_due(
                        eligible_races=eligible_races,
                        last_evaluated_races=last_evaluated_races,
                        min_new_races=args.min_new_races,
                        artifacts_exist=model_path.exists() and backtest_path.exists(),
                    )
                    if not due:
                        event["status"] = "up_to_date"
                        event["next_evaluation_races"] = (
                            last_evaluated_races + max(1, args.min_new_races)
                        )
                    else:
                        train_meta = train_model(
                            conn,
                            model_path=model_path,
                            include_odds=args.include_odds,
                            from_date=args.from_date,
                            min_odds_snapshots=args.min_odds_snapshots,
                            complete_results_only=args.include_odds,
                            min_examples=args.min_examples,
                        )
                        event["trained"] = True
                        event["train"] = train_meta
                        bt = backtest_model(
                            conn,
                            output_path=backtest_path,
                            folds=args.folds,
                            include_odds=args.include_odds,
                            from_date=args.from_date,
                            min_odds_snapshots=args.min_odds_snapshots,
                            complete_results_only=args.include_odds,
                            min_train_races=args.min_train_races,
                        )
                        event["backtested"] = True
                        event["status"] = "evaluated"
                        event["backtest"] = {
                            "evaluated_races": bt.get("evaluated_races"),
                            "winner_top1_accuracy": bt.get("winner_top1_accuracy"),
                            "trifecta_top5_hit_rate": bt.get("trifecta_top5_hit_rate"),
                            "entry_log_loss": bt.get("entry_log_loss"),
                            "entry_brier": bt.get("entry_brier"),
                        }
                        last_evaluated_races = eligible_races
                        last_evaluated_at = event["generated_at"]
                        event["last_evaluated_races"] = last_evaluated_races
                        event["last_evaluated_at"] = last_evaluated_at
                else:
                    event["status"] = "waiting_for_examples"
        except Exception as exc:
            event["status"] = "error"
            event["error"] = str(exc)
        if state_path:
            try:
                write_state(state_path, event)
            except Exception as exc:
                event["state_error"] = str(exc)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.interval)


def read_state(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def evaluation_due(
    *,
    eligible_races: int,
    last_evaluated_races: int,
    min_new_races: int,
    artifacts_exist: bool,
) -> bool:
    if not artifacts_exist or last_evaluated_races <= 0:
        return True
    increment = max(0, int(min_new_races))
    return increment == 0 or eligible_races >= last_evaluated_races + increment


def write_state(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(event, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def dataset_counts(
    conn: sqlite3.Connection,
    *,
    from_date: str | None = None,
    require_odds: bool = False,
    min_odds_snapshots: int = 1,
) -> dict[str, int]:
    filters = ["rr.rank IS NOT NULL"]
    where_params: list[object] = []
    if from_date:
        filters.append("r.race_date >= ?")
        where_params.append(from_date)

    if require_odds:
        filters.append(
            "(SELECT COUNT(*) FROM odds_snapshots os "
            "WHERE os.race_id = r.race_id) >= ?"
        )
        where_params.append(max(1, int(min_odds_snapshots)))

    row = conn.execute(
        f"""
        SELECT COUNT(*) AS examples, COUNT(DISTINCT r.race_id) AS races
        FROM races r
        JOIN (
            SELECT race_id
            FROM race_results
            WHERE rank IS NOT NULL
            GROUP BY race_id
            HAVING COUNT(*) = 6
        ) eligible_results ON eligible_results.race_id = r.race_id
        JOIN entries e ON e.race_id = r.race_id
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {" AND ".join(filters)}
        """,
        where_params,
    ).fetchone()
    races = int(row["races"] if isinstance(row, sqlite3.Row) else row[1])
    examples = int(row["examples"] if isinstance(row, sqlite3.Row) else row[0])
    return {
        "examples": examples,
        "races": races,
        "odds_result_races": races if require_odds else 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
