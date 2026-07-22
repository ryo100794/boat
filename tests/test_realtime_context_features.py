from itertools import permutations

from boatrace_ai.db import connection, init_db
from boatrace_ai.modeling import (
    REALTIME_ODDS_FEATURE_SET,
    _load_model_training_examples,
    train_model,
)


def _seed_context_race(conn, race_id: str = "context", rno: int = 1) -> None:
    conn.execute(
        "INSERT INTO races "
        "(race_id, race_date, jcd, venue_name, rno, deadline_at) "
        "VALUES (?, '2026-07-22', '01', '桐生', ?, '2026-07-22T10:20:00+09:00')",
        (race_id, rno),
    )
    for lane in range(1, 7):
        conn.execute(
            "INSERT INTO entries "
            "(race_id, lane, racer_no, racer_class, branch, origin, "
            "national_win_rate, local_win_rate, motor_2_rate, boat_2_rate) "
            "VALUES (?, ?, ?, 'A1', ?, ?, ?, ?, ?, ?)",
            (
                race_id,
                lane,
                4000 + lane,
                "群馬" if lane == 1 else "東京",
                "群馬" if lane == 1 else "東京",
                7.0 - lane * 0.2,
                7.2 - lane * 0.2,
                40.0 - lane,
                38.0 - lane,
            ),
        )
        conn.execute(
            "INSERT INTO race_results (race_id, lane, rank) VALUES (?, ?, ?)",
            (race_id, lane, lane),
        )
        conn.execute(
            "INSERT INTO beforeinfo "
            "(race_id, captured_at, lane, exhibition_time, course, start_timing, "
            "weather, wind_direction, wind_speed_m, wave_cm) "
            "VALUES (?, '2026-07-22T01:10:00+00:00', ?, ?, ?, ?, '雨', '北', 4.0, 3.0)",
            (race_id, lane, 6.60 + lane * 0.02, lane, 0.10 + lane * 0.01),
        )
        conn.execute(
            "INSERT INTO beforeinfo "
            "(race_id, captured_at, lane, exhibition_time, course, start_timing, "
            "weather, wind_direction, wind_speed_m, wave_cm) "
            "VALUES (?, '2026-07-22T01:16:00+00:00', ?, 9.99, 6, 0.99, "
            "'締切後', '南', 9.0, 9.0)",
            (race_id, lane),
        )
    combinations = ["-".join(map(str, value)) for value in permutations(range(1, 7), 3)]
    for minute in range(10):
        cursor = conn.execute(
            "INSERT INTO odds_snapshots "
            "(race_id, bet_type, captured_at, parser_version) "
            "VALUES (?, 'trifecta', ?, 'odds3t_dom_v2')",
            (race_id, f"2026-07-22T01:{minute:02d}:00+00:00"),
        )
        conn.executemany(
            "INSERT INTO odds_trifecta "
            "(snapshot_id, race_id, combination, odds) VALUES (?, ?, ?, ?)",
            [
                (cursor.lastrowid, race_id, combination, 10.0 + minute)
                for combination in combinations
            ],
        )


def test_realtime_training_uses_beforeinfo_weather_course_and_odds(tmp_path) -> None:
    db_path = tmp_path / "context.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        _seed_context_race(conn)
        features, labels, meta = _load_model_training_examples(
            conn,
            include_odds=True,
            from_date="2026-07-22",
            min_odds_snapshots=10,
            complete_results_only=True,
        )

    assert len(features) == len(labels) == len(meta) == 6
    lane1 = features[0]
    assert lane1["has_beforeinfo"] == 1
    assert lane1["weather"] == "雨"
    assert lane1["course"] == 1.0
    assert lane1["exhibition_time"] > 0
    assert lane1["odds_snapshot_count"] == 10.0
    assert lane1["research_home_branch"] == 1
    assert "motor_2_rate_rank" in lane1
    assert "boat_2_rate_rank" in lane1
    assert lane1["exhibition_time"] < 7.0


def test_realtime_model_artifact_records_v2_feature_contract(tmp_path) -> None:
    db_path = tmp_path / "context.sqlite"
    model_path = tmp_path / "model.joblib"
    init_db(db_path)
    with connection(db_path) as conn:
        for index in range(3):
            _seed_context_race(
                conn,
                race_id=f"context-{index}",
                rno=index + 1,
            )
        metadata = train_model(
            conn,
            model_path=model_path,
            include_odds=True,
            from_date="2026-07-22",
            min_odds_snapshots=10,
            complete_results_only=True,
            min_examples=12,
        )

    assert metadata["feature_set"] == REALTIME_ODDS_FEATURE_SET
    assert metadata["races"] == 3
