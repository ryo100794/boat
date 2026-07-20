from __future__ import annotations

from datetime import datetime, timezone

from boatrace_ai.db import connection, init_db, insert_odds_snapshot
from boatrace_ai.web.intraday_bankroll import COMBINATIONS, day_bankroll_simulation


def _seed_race(conn, race_id: str, start_at: str, model: str) -> None:
    conn.execute(
        """
        INSERT INTO races(
          race_id, race_date, jcd, venue_name, rno, deadline_at, status
        ) VALUES (?, '2026-07-20', '01', '桐生', 1, ?, 'final')
        """,
        (race_id, start_at),
    )
    conn.executemany(
        "INSERT INTO race_results(race_id, lane, rank) VALUES (?, ?, ?)",
        [(race_id, lane, lane) for lane in range(1, 7)],
    )
    conn.execute(
        """
        INSERT INTO payouts(race_id, bet_type, combination, payout_yen)
        VALUES (?, '3連単', '1-2-3', 2000)
        """,
        (race_id,),
    )
    conn.executemany(
        """
        INSERT INTO predictions(
          race_id, generated_at, model_path, combination, probability
        ) VALUES (?, '2026-07-20T02:49:00+00:00', ?, ?, ?)
        """,
        [
            (race_id, model, combination, 0.20 if combination == "1-2-3" else 0.001)
            for combination in COMBINATIONS
        ],
    )
    insert_odds_snapshot(
        conn,
        race_id,
        "2026-07-20T02:49:30+00:00",
        "11:49",
        {combination: 10.0 for combination in COMBINATIONS},
        "test",
        {"parser_version": "odds3t_dom_v2"},
    )


def test_simulates_selected_model_from_first_race_with_reinvestment(tmp_path) -> None:
    db_path = tmp_path / "boat.sqlite"
    init_db(db_path)
    model = "data/models/win_model_no_odds_v8.joblib"
    with connection(db_path) as conn:
        _seed_race(conn, "2026-07-20-01-01", "2026-07-20T12:00:00", model)

    with connection(db_path) as conn:
        result = day_bankroll_simulation(
            conn,
            race_date="2026-07-20",
            model_path=model,
            now=datetime(2026, 7, 20, 4, 0, tzinfo=timezone.utc),
        )

    assert result["selected_model"] == model
    assert result["stats"]["evaluated_races"] == 1
    assert result["stats"]["valid_odds_races"] == 1
    assert result["stats"]["tickets"] == 1
    assert result["stats"]["stake_yen"] == 200
    assert result["stats"]["return_yen"] == 4000
    assert result["stats"]["current_bankroll_yen"] == 13_800
    assert result["series"][0]["profit_yen"] == 3800


def test_rejects_lane_header_values_mixed_into_odds(tmp_path) -> None:
    db_path = tmp_path / "boat.sqlite"
    init_db(db_path)
    model = "data/models/win_model_no_odds_v8.joblib"
    with connection(db_path) as conn:
        _seed_race(conn, "2026-07-20-01-01", "2026-07-20T12:00:00", model)
        insert_odds_snapshot(
            conn,
            "2026-07-20-01-01",
            "2026-07-20T02:49:40+00:00",
            "11:49",
            {
                combination: float((index % 6) + 1)
                for index, combination in enumerate(COMBINATIONS)
            },
            "broken",
            {"parser_version": "odds3t_dom_v2"},
        )

    with connection(db_path) as conn:
        result = day_bankroll_simulation(
            conn,
            race_date="2026-07-20",
            model_path=model,
            now=datetime(2026, 7, 20, 4, 0, tzinfo=timezone.utc),
        )

    assert result["stats"]["rejected_odds_snapshots"] >= 1
    assert result["stats"]["valid_odds_races"] == 1
