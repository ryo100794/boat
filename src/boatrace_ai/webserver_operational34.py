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
        today = base._row(
            conn,
            """
            WITH
              entry_counts AS (SELECT race_id, COUNT(*) AS entries FROM entries GROUP BY race_id),
              result_counts AS (SELECT race_id, COUNT(*) AS results FROM race_results WHERE rank IS NOT NULL GROUP BY race_id),
              odds_counts AS (SELECT race_id, COUNT(*) AS odds FROM odds_snapshots GROUP BY race_id),
              pred_counts AS (SELECT race_id, COUNT(*) AS preds FROM predictions GROUP BY race_id)
            SELECT
              COUNT(*) AS races,
              SUM(CASE WHEN COALESCE(ec.entries, 0) = 6 THEN 1 ELSE 0 END) AS entry_races,
              SUM(CASE WHEN COALESCE(rc.results, 0) >= 3 THEN 1 ELSE 0 END) AS result_races,
              SUM(CASE WHEN COALESCE(oc.odds, 0) > 0 THEN 1 ELSE 0 END) AS odds_races,
              SUM(CASE WHEN COALESCE(pc.preds, 0) > 0 THEN 1 ELSE 0 END) AS prediction_races
            FROM races r
            LEFT JOIN entry_counts ec ON ec.race_id = r.race_id
            LEFT JOIN result_counts rc ON rc.race_id = r.race_id
            LEFT JOIN odds_counts oc ON oc.race_id = r.race_id
            LEFT JOIN pred_counts pc ON pc.race_id = r.race_id
            WHERE r.race_date = ?
            """,
            (race_date,),
        )
        years = base._rows(
            conn,
            """
            WITH
              entry_counts AS (SELECT race_id, COUNT(*) AS entries FROM entries GROUP BY race_id),
              result_counts AS (SELECT race_id, COUNT(*) AS results FROM race_results WHERE rank IS NOT NULL GROUP BY race_id),
              pred_counts AS (SELECT race_id, COUNT(*) AS preds FROM predictions GROUP BY race_id)
            SELECT
              substr(r.race_date, 1, 4) AS year,
              COUNT(*) AS races,
              SUM(CASE WHEN COALESCE(ec.entries, 0) = 6 THEN 1 ELSE 0 END) AS entry_races,
              SUM(CASE WHEN COALESCE(rc.results, 0) >= 3 THEN 1 ELSE 0 END) AS result_races,
              SUM(CASE WHEN COALESCE(pc.preds, 0) > 0 THEN 1 ELSE 0 END) AS prediction_races
            FROM races r
            LEFT JOIN entry_counts ec ON ec.race_id = r.race_id
            LEFT JOIN result_counts rc ON rc.race_id = r.race_id
            LEFT JOIN pred_counts pc ON pc.race_id = r.race_id
            GROUP BY year
            ORDER BY year DESC
            LIMIT 14
            """,
        )
        venues = base._rows(
            conn,
            """
            WITH
              entry_counts AS (SELECT race_id, COUNT(*) AS entries FROM entries GROUP BY race_id),
              result_counts AS (SELECT race_id, COUNT(*) AS results FROM race_results WHERE rank IS NOT NULL GROUP BY race_id),
              odds_counts AS (SELECT race_id, COUNT(*) AS odds FROM odds_snapshots GROUP BY race_id)
            SELECT
              r.jcd,
              MAX(r.venue_name) AS venue_name,
              COUNT(*) AS races,
              SUM(CASE WHEN COALESCE(ec.entries, 0) = 6 THEN 1 ELSE 0 END) AS entry_races,
              SUM(CASE WHEN COALESCE(rc.results, 0) >= 3 THEN 1 ELSE 0 END) AS result_races,
              SUM(CASE WHEN COALESCE(oc.odds, 0) > 0 THEN 1 ELSE 0 END) AS odds_races
            FROM races r
            LEFT JOIN entry_counts ec ON ec.race_id = r.race_id
            LEFT JOIN result_counts rc ON rc.race_id = r.race_id
            LEFT JOIN odds_counts oc ON oc.race_id = r.race_id
            GROUP BY r.jcd
            ORDER BY r.jcd
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


base.archive_overview = archive_overview_fast


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
