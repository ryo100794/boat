from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from . import webserver_operational33 as base
from .db import connect


HTML = base.HTML


def archive_overview_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = base._one(query, "date", date.today().isoformat())
    with connect(db_path) as conn:
        totals = base._row(
            conn,
            """
            SELECT
              COUNT(*) AS races,
              MIN(race_date) AS first_date,
              MAX(race_date) AS last_date,
              (SELECT COUNT(*) FROM entries) AS entries,
              (SELECT COUNT(DISTINCT race_id) FROM race_results WHERE rank IS NOT NULL) AS result_races,
              (SELECT COUNT(*) FROM odds_snapshots) AS odds_snapshots,
              (SELECT COUNT(DISTINCT race_id) FROM predictions) AS prediction_races,
              (SELECT COUNT(DISTINCT race_id) FROM beforeinfo) AS beforeinfo_races
            FROM races
            """,
        )
        today = {
            "races": _scalar(conn, "SELECT COUNT(*) FROM races WHERE race_date = ?", (race_date,)),
            "entry_races": _scalar(
                conn,
                "SELECT COUNT(DISTINCT e.race_id) FROM entries e JOIN races r ON r.race_id = e.race_id WHERE r.race_date = ?",
                (race_date,),
            ),
            "result_races": _scalar(
                conn,
                "SELECT COUNT(DISTINCT rr.race_id) FROM race_results rr JOIN races r ON r.race_id = rr.race_id WHERE r.race_date = ? AND rr.rank IS NOT NULL",
                (race_date,),
            ),
            "odds_races": _scalar(
                conn,
                "SELECT COUNT(DISTINCT os.race_id) FROM odds_snapshots os JOIN races r ON r.race_id = os.race_id WHERE r.race_date = ?",
                (race_date,),
            ),
            "prediction_races": _scalar(
                conn,
                "SELECT COUNT(DISTINCT p.race_id) FROM predictions p JOIN races r ON r.race_id = p.race_id WHERE r.race_date = ?",
                (race_date,),
            ),
        }
        years = base._rows(
            conn,
            """
            SELECT
              substr(race_date, 1, 4) AS year,
              COUNT(*) AS races,
              NULL AS entry_races,
              NULL AS result_races,
              NULL AS prediction_races
            FROM races
            GROUP BY year
            ORDER BY year DESC
            LIMIT 14
            """,
        )
        venues = base._rows(
            conn,
            """
            SELECT
              jcd,
              MAX(venue_name) AS venue_name,
              COUNT(*) AS races,
              NULL AS entry_races,
              NULL AS result_races,
              NULL AS odds_races
            FROM races
            GROUP BY jcd
            ORDER BY jcd
            """,
        )
    return {
        "date": race_date,
        "generated_at": base._now(),
        "totals": totals,
        "today": today,
        "years": years,
        "venues": venues,
    }


def _scalar(conn, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row else 0


base.archive_overview = archive_overview_fast


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
