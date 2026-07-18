from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .db import connection, init_db
from .modeling_no_odds_v7_modelrank import predict_race


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh all day predictions ranked by model probability.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v7.joblib")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)

    init_db(args.db)
    model_path = Path(args.model)
    predicted = 0
    failed = 0
    with connection(args.db) as conn:
        rows = conn.execute(
            """
            SELECT r.race_id
            FROM races r
            WHERE r.race_date = ?
              AND (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
            ORDER BY r.jcd, r.rno
            """,
            (args.date,),
        ).fetchall()
        for row in rows:
            try:
                predict_race(conn, model_path=model_path, race_id_value=row["race_id"])
                predicted += 1
            except Exception:
                failed += 1
    print(
        json.dumps(
            {
                "date": args.date,
                "model": str(model_path),
                "rank_basis": "model_probability",
                "targets": predicted + failed,
                "predicted": predicted,
                "failed": failed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
