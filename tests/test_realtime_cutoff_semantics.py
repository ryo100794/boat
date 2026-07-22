from boatrace_ai.db import connection, init_db, insert_odds_snapshot
from boatrace_ai.features import odds_lane_features
from boatrace_ai.odds_quality import TRIFECTA_COMBINATION_KEYS
from boatrace_ai.runtime.model_cycle import dataset_counts


def test_t5_gate_treats_naive_stored_start_as_jst(tmp_path) -> None:
    database = tmp_path / "cutoff.sqlite"
    race_id = "2026-07-22-01-01"
    odds = {combination: 10.0 for combination in TRIFECTA_COMBINATION_KEYS}
    init_db(database)
    with connection(database) as conn:
        conn.execute(
            "INSERT INTO races "
            "(race_id, race_date, jcd, venue_name, rno, deadline_at) "
            "VALUES (?, '2026-07-22', '01', '桐生', 1, "
            "'2026-07-22T10:20:00')",
            (race_id,),
        )
        for lane in range(1, 7):
            conn.execute(
                "INSERT INTO entries (race_id, lane) VALUES (?, ?)",
                (race_id, lane),
            )
            conn.execute(
                "INSERT INTO race_results (race_id, lane, rank) VALUES (?, ?, ?)",
                (race_id, lane, lane),
            )
        for minute in range(9):
            insert_odds_snapshot(
                conn,
                race_id,
                f"2026-07-22T01:{minute:02d}:00+00:00",
                None,
                odds,
                "test",
                {"parser_version": "odds3t_dom_v2"},
            )
        insert_odds_snapshot(
            conn,
            race_id,
            "2026-07-22T01:15:00+00:00",
            None,
            odds,
            "after-model-cutoff",
            {"parser_version": "odds3t_dom_v2"},
        )

        before = dataset_counts(
            conn,
            from_date="2026-07-22",
            require_odds=True,
            min_odds_snapshots=10,
        )
        lane_features = odds_lane_features(conn, race_id)
        insert_odds_snapshot(
            conn,
            race_id,
            "2026-07-22T01:09:00+00:00",
            None,
            odds,
            "tenth-safe-snapshot",
            {"parser_version": "odds3t_dom_v2"},
        )
        after = dataset_counts(
            conn,
            from_date="2026-07-22",
            require_odds=True,
            min_odds_snapshots=10,
        )

    assert before["odds_result_races"] == 0
    assert {row["snapshot_count"] for row in lane_features.values()} == {9.0}
    assert after["odds_result_races"] == 1
