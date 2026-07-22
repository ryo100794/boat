from __future__ import annotations

from boatrace_ai.db import connection, init_db, insert_odds_snapshot
from boatrace_ai.listwise.market_calibration import odds_data_signature
from boatrace_ai.odds_quality import TRIFECTA_COMBINATION_KEYS


def _insert_race(conn, race_id: str, *, rno: int, start_at: str) -> None:
    conn.execute(
        "INSERT INTO races(race_id, race_date, jcd, venue_name, rno, deadline_at) "
        "VALUES (?, '2026-07-22', '01', 'Kiryu', ?, ?)",
        (race_id, rno, start_at),
    )


def _complete_race(conn, race_id: str) -> None:
    conn.executemany(
        "INSERT INTO race_results(race_id, lane, rank) VALUES (?, ?, ?)",
        [(race_id, lane, lane) for lane in range(1, 7)],
    )


def _insert_snapshot(conn, race_id: str, captured_at: str, value: float) -> int:
    return insert_odds_snapshot(
        conn,
        race_id,
        captured_at,
        captured_at,
        {combination: value for combination in TRIFECTA_COMBINATION_KEYS},
        f"https://example.test/{race_id}/{captured_at}",
        {"parser_version": "odds3t_dom_v2"},
    )


def test_signature_uses_only_completed_races_and_latest_pre_t5_snapshot(tmp_path) -> None:
    database = tmp_path / "signature.sqlite"
    complete_id = "2026-07-22-01-01"
    future_id = "2026-07-22-01-02"
    init_db(database)

    with connection(database) as conn:
        _insert_race(
            conn,
            complete_id,
            rno=1,
            start_at="2026-07-22T12:00:00+09:00",
        )
        _insert_race(
            conn,
            future_id,
            rno=2,
            start_at="2026-07-22T12:30:00+09:00",
        )
        _complete_race(conn, complete_id)
        older_id = _insert_snapshot(
            conn, complete_id, "2026-07-22T02:45:00+00:00", 12.0
        )
        selected_id = _insert_snapshot(
            conn, complete_id, "2026-07-22T02:49:00+00:00", 11.0
        )

        baseline = odds_data_signature(
            conn, from_date="2026-07-22", through_date="2026-07-22"
        )

        _insert_snapshot(conn, complete_id, "2026-07-22T02:51:00+00:00", 10.0)
        _insert_snapshot(conn, future_id, "2026-07-22T03:19:00+00:00", 9.0)
        unchanged = odds_data_signature(
            conn, from_date="2026-07-22", through_date="2026-07-22"
        )

        _complete_race(conn, future_id)
        completed = odds_data_signature(
            conn, from_date="2026-07-22", through_date="2026-07-22"
        )

    assert older_id != selected_id
    assert baseline == {
        "complete_race_count": 1,
        "payout_race_count": 0,
        "snapshot_count": 1,
        "snapshot_id_sum": selected_id,
        "max_snapshot_id": selected_id,
    }
    assert unchanged == baseline
    assert completed["complete_race_count"] == 2
    assert completed["snapshot_count"] == 2
    assert completed["snapshot_id_sum"] > baseline["snapshot_id_sum"]


def test_signature_changes_when_trifecta_payout_arrives(tmp_path) -> None:
    database = tmp_path / "payout.sqlite"
    race_id = "2026-07-22-01-01"
    init_db(database)

    with connection(database) as conn:
        _insert_race(
            conn,
            race_id,
            rno=1,
            start_at="2026-07-22T12:00:00",
        )
        _complete_race(conn, race_id)
        _insert_snapshot(conn, race_id, "2026-07-22T02:49:00+00:00", 11.0)
        before = odds_data_signature(
            conn, from_date="2026-07-22", through_date="2026-07-22"
        )
        conn.execute(
            "INSERT INTO payouts(race_id, bet_type, combination, payout_yen) "
            "VALUES (?, ?, '1-2-3', 1230)",
            (race_id, "3\u9023\u5358"),
        )
        after = odds_data_signature(
            conn, from_date="2026-07-22", through_date="2026-07-22"
        )

    assert before["payout_race_count"] == 0
    assert after["payout_race_count"] == 1
    assert after["snapshot_id_sum"] == before["snapshot_id_sum"]
