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
        years = _merged_group(
            base._rows(
                conn,
                """
                SELECT substr(race_date, 1, 4) AS key, substr(race_date, 1, 4) AS year, COUNT(*) AS races
                FROM races
                GROUP BY key
                ORDER BY key DESC
                LIMIT 14
                """,
            ),
            "key",
            [
                ("entry_races", "SELECT substr(r.race_date, 1, 4) AS key, COUNT(DISTINCT e.race_id) AS value FROM entries e JOIN races r ON r.race_id = e.race_id GROUP BY key"),
                ("result_races", "SELECT substr(r.race_date, 1, 4) AS key, COUNT(DISTINCT rr.race_id) AS value FROM race_results rr JOIN races r ON r.race_id = rr.race_id WHERE rr.rank IS NOT NULL GROUP BY key"),
                ("prediction_races", "SELECT substr(r.race_date, 1, 4) AS key, COUNT(DISTINCT p.race_id) AS value FROM predictions p JOIN races r ON r.race_id = p.race_id GROUP BY key"),
            ],
            conn,
        )
        venues = _merged_group(
            base._rows(
                conn,
                """
                SELECT jcd AS key, jcd, MAX(venue_name) AS venue_name, COUNT(*) AS races
                FROM races
                GROUP BY jcd
                ORDER BY jcd
                """,
            ),
            "key",
            [
                ("entry_races", "SELECT r.jcd AS key, COUNT(DISTINCT e.race_id) AS value FROM entries e JOIN races r ON r.race_id = e.race_id GROUP BY r.jcd"),
                ("result_races", "SELECT r.jcd AS key, COUNT(DISTINCT rr.race_id) AS value FROM race_results rr JOIN races r ON r.race_id = rr.race_id WHERE rr.rank IS NOT NULL GROUP BY r.jcd"),
                ("odds_races", "SELECT r.jcd AS key, COUNT(DISTINCT os.race_id) AS value FROM odds_snapshots os JOIN races r ON r.race_id = os.race_id GROUP BY r.jcd"),
            ],
            conn,
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


def _merged_group(
    rows: list[dict[str, Any]],
    key: str,
    metrics: list[tuple[str, str]],
    conn,
) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    wanted = {str(row[key]) for row in out}
    for metric, sql in metrics:
        values = {str(row["key"]): int(row["value"] or 0) for row in conn.execute(sql).fetchall()}
        for row in out:
            row[metric] = values.get(str(row[key]), 0)
    for row in out:
        row.pop("key", None)
    return [row for row in out if not wanted or str(row.get(key, "")) in wanted or key not in row]


base.archive_overview = archive_overview_fast


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
