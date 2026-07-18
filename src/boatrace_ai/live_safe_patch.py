from __future__ import annotations

from datetime import date

from .constants import VENUE_BY_CODE
from .db import race_id


def install() -> None:
    from . import live

    live._ensure_minimal_race = _ensure_minimal_race_safe


def _ensure_minimal_race_safe(conn, *, race_date: date, jcd: str, rno: int, status: str) -> None:
    rid = race_id(race_date.isoformat(), jcd, rno)
    venue = VENUE_BY_CODE.get(jcd.zfill(2))
    conn.execute(
        """
        INSERT INTO races (race_id, race_date, jcd, venue_name, rno, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id) DO UPDATE SET
          status = CASE
            WHEN excluded.status IS NOT NULL THEN excluded.status
            ELSE races.status
          END,
          venue_name = COALESCE(NULLIF(races.venue_name, ''), excluded.venue_name),
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            rid,
            race_date.isoformat(),
            jcd.zfill(2),
            venue.name if venue else jcd.zfill(2),
            int(rno),
            status,
        ),
    )
