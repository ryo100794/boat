import os

import psycopg
import pytest

from boatrace_ai.db import insert_odds_snapshot, upsert_race
from boatrace_ai.postgresql import Connection


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
    finally:
        raw.rollback()
        raw.close()
