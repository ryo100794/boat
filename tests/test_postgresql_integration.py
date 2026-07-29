import os

import psycopg
import pytest

from boatrace_ai.db import insert_odds_snapshot, upsert_race
from boatrace_ai.postgresql import Connection, convert_sql
from boatrace_ai.storage import upsert_result_status


def test_live_storage_round_trip_rolls_back() -> None:
    dsn = os.environ.get("BOATRACE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("BOATRACE_TEST_POSTGRES_DSN is not set")

    raw = psycopg.connect(dsn, connect_timeout=30)
    connection = Connection(raw)
    race_id = "2099-12-31-99-01"
    try:
        upsert_race(
            connection,
            {
                "race_id": race_id,
                "race_date": "2099-12-31",
                "jcd": "99",
                "venue_name": "integration-test",
                "rno": 1,
                "status": "scheduled",
            },
        )
        snapshot_id = insert_odds_snapshot(
            connection,
            race_id,
            "2099-12-31T00:00:00+00:00",
            "00:00",
            {"1-2-3": 12.3, "1-3-2": 21.0},
            "https://example.invalid/test",
            {"parsed_count": 2},
        )
        row = connection.execute(
            "SELECT count(*) FROM odds_trifecta WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        assert row[0] == 2
        upsert_race(
            connection,
            {
                "race_id": race_id,
                "race_date": "2099-12-31",
                "jcd": "99",
                "venue_name": "integration-test",
                "rno": 1,
                "status": "final",
            },
        )
        upsert_result_status(
            connection,
            race_id=race_id,
            row={
                "status": "final",
                "rows": [],
                "payouts": [],
                "trifecta_evaluable": False,
                "result_reason": "race_cancelled",
            },
        )
        row = connection.execute(
            "SELECT r.status, rs.status, rs.trifecta_evaluable, rs.reason "
            "FROM races r JOIN race_result_status rs USING (race_id) "
            "WHERE r.race_id = ?",
            (race_id,),
        ).fetchone()
        assert tuple(row) == ("final", "final", False, "race_cancelled")
    finally:
        raw.rollback()
        raw.close()


def test_cancelled_result_upsert_is_postgresql_compatible() -> None:
    statement = convert_sql(
        """
        INSERT INTO race_result_status (race_id, status, trifecta_evaluable, reason)
        VALUES (:race_id, :status, :trifecta_evaluable, :reason)
        ON CONFLICT(race_id) DO UPDATE SET
          status=excluded.status,
          trifecta_evaluable=excluded.trifecta_evaluable,
          reason=excluded.reason
        """
    )

    assert "%s" not in statement
    assert "%(race_id)s" in statement
    assert "%(trifecta_evaluable)s" in statement
    assert "ON CONFLICT(race_id) DO UPDATE" in statement
