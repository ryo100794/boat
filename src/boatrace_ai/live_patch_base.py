from __future__ import annotations

from datetime import date

from .constants import VENUE_BY_CODE
from .db import race_id


def install() -> None:
    from . import live

    live._ensure_minimal_race = _ensure_minimal_race_safe
    live.upsert_race = upsert_race_preserve_existing


def upsert_race_preserve_existing(conn, payload) -> str:
    rid = payload.get("race_id") or race_id(
        payload["race_date"], payload["jcd"], payload["rno"]
    )
    values = {
        "race_id": rid,
        "race_date": payload["race_date"],
        "jcd": payload["jcd"].zfill(2),
        "venue_name": payload["venue_name"],
        "rno": int(payload["rno"]),
        "title": payload.get("title"),
        "race_type": payload.get("race_type"),
        "distance_m": payload.get("distance_m"),
        "deadline_at": payload.get("deadline_at"),
        "status": payload.get("status", "scheduled"),
        "source_url": payload.get("source_url"),
    }
    conn.execute(
        """
        INSERT INTO races (
          race_id, race_date, jcd, venue_name, rno, title, race_type,
          distance_m, deadline_at, status, source_url
        )
        VALUES (
          :race_id, :race_date, :jcd, :venue_name, :rno, :title, :race_type,
          :distance_m, :deadline_at, :status, :source_url
        )
        ON CONFLICT(race_id) DO UPDATE SET
          title=COALESCE(excluded.title, races.title),
          race_type=COALESCE(excluded.race_type, races.race_type),
          distance_m=COALESCE(excluded.distance_m, races.distance_m),
          deadline_at=COALESCE(excluded.deadline_at, races.deadline_at),
          status=CASE
            WHEN races.status = 'final' THEN races.status
            ELSE COALESCE(excluded.status, races.status)
          END,
          source_url=COALESCE(excluded.source_url, races.source_url),
          updated_at=CURRENT_TIMESTAMP
        """,
        values,
    )
    return rid


def _ensure_minimal_race_safe(conn, *, race_date: date, jcd: str, rno: int, status: str) -> None:
    rid = race_id(race_date.isoformat(), jcd, rno)
    venue = VENUE_BY_CODE.get(jcd.zfill(2))
    conn.execute(
        """
        INSERT INTO races (race_id, race_date, jcd, venue_name, rno, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id) DO UPDATE SET
          status = CASE
            WHEN races.status = 'final' THEN races.status
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
