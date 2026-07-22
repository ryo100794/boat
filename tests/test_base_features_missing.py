from __future__ import annotations

import sqlite3

from boatrace_ai.base_features import (
    _ranks,
    iter_training_examples,
    race_relative_features,
)


def _entry(lane: int) -> dict:
    return {
        "lane": lane,
        "rno": 1,
        "jcd": "01",
        "racer_class": "A1",
        "age": 30,
        "weight_kg": 52,
        "f_count": 0,
        "l_count": 0,
        "avg_st": 0.15,
        "national_win_rate": 6.0,
        "national_2_rate": 40.0,
        "national_3_rate": 55.0,
        "local_win_rate": 6.0,
        "local_2_rate": 40.0,
        "local_3_rate": 55.0,
        "motor_2_rate": 35.0,
        "motor_3_rate": 50.0,
        "boat_2_rate": 35.0,
        "boat_3_rate": 50.0,
    }


def test_missing_values_are_not_ranked_by_lane() -> None:
    assert _ranks(
        {1: -1.0, 2: -1.0, 3: 5.0},
        high_is_good=True,
    ) == {1: 0, 2: 0, 3: 1}


def test_equal_values_share_competition_rank() -> None:
    assert _ranks(
        {1: 5.0, 2: 5.0, 3: 4.0},
        high_is_good=True,
    ) == {1: 1, 2: 1, 3: 3}


def test_missing_relative_value_uses_flag_and_neutral_derivatives() -> None:
    features = race_relative_features(
        [_entry(lane) for lane in range(1, 7)],
        {lane: {} for lane in range(1, 7)},
    )

    for lane in range(1, 7):
        assert features[lane]["has_exhibition_time"] == 0
        assert features[lane]["exhibition_time_rank"] == 0
        assert features[lane]["exhibition_time_vs_mean"] == 0.0
        assert features[lane]["exhibition_time_z"] == 0.0
        assert features[lane]["exhibition_time_best_gap"] == 0.0
        assert features[lane]["exhibition_time_scaled"] == 0.0


def test_training_example_stream_filters_races_and_uses_latest_beforeinfo() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE races (
          race_id TEXT, race_date TEXT, jcd TEXT, rno INTEGER,
          race_type TEXT, distance_m INTEGER, deadline_at TEXT
        );
        CREATE TABLE entries (
          race_id TEXT, lane INTEGER, racer_no INTEGER, racer_name TEXT,
          racer_class TEXT, branch TEXT, origin TEXT, age INTEGER,
          weight_kg REAL, f_count INTEGER, l_count INTEGER, avg_st REAL,
          national_win_rate REAL, national_2_rate REAL, national_3_rate REAL,
          local_win_rate REAL, local_2_rate REAL, local_3_rate REAL,
          motor_no INTEGER, motor_2_rate REAL, motor_3_rate REAL,
          boat_no INTEGER, boat_2_rate REAL, boat_3_rate REAL
        );
        CREATE TABLE race_results (
          race_id TEXT, lane INTEGER, rank INTEGER
        );
        CREATE TABLE beforeinfo (
          race_id TEXT, lane INTEGER, captured_at TEXT, weight_kg REAL,
          exhibition_time REAL, tilt REAL, adjusted_weight REAL, course INTEGER,
          start_timing REAL, wind_speed_m REAL, air_temp_c REAL,
          water_temp_c REAL, wave_cm REAL, weather TEXT, wind_direction TEXT,
          propeller TEXT, parts_exchange TEXT
        );
        """
    )
    race_ids = ("202601010101", "202601020101")
    for day, race_id in enumerate(race_ids, start=1):
        conn.execute(
            "INSERT INTO races VALUES (?, ?, '01', 1, 'general', 1800, ?)",
            (race_id, f"2026-01-{day:02d}", f"2026-01-{day:02d}T01:00:00+00:00"),
        )
        for lane in range(1, 7):
            conn.execute(
                """
                INSERT INTO entries VALUES (
                  ?, ?, ?, ?, 'A1', 'Tokyo', 'Tokyo', 30, 52.0, 0, 0, 0.15,
                  6.0, 40.0, 55.0, 6.1, 41.0, 56.0,
                  ?, 35.0, 50.0, ?, 36.0, 51.0
                )
                """,
                (race_id, lane, 4000 + lane, f"Racer {lane}", lane, lane),
            )
            conn.execute(
                "INSERT INTO race_results VALUES (?, ?, ?)",
                (race_id, lane, lane),
            )
            for captured_at, weight in (
                ("2026-01-01T00:00:00+00:00", 51.0),
                ("2026-01-01T00:01:00+00:00", 52.5),
            ):
                conn.execute(
                    """
                    INSERT INTO beforeinfo VALUES (
                      ?, ?, ?, ?, 6.70, 0.0, 0.0, ?, 0.12,
                      3.0, 20.0, 18.0, 3.0, 'sunny', 'north', '', ''
                    )
                    """,
                    (race_id, lane, captured_at, weight, lane),
                )

    rows = list(
        iter_training_examples(
            conn,
            include_odds=False,
            include_research=False,
            include_races={race_ids[1]},
        )
    )

    assert len(rows) == 6
    assert {str(meta["race_id"]) for _item, _label, meta in rows} == {race_ids[1]}
    assert [label for _item, label, _meta in rows] == [1, 0, 0, 0, 0, 0]
    assert all(item["before_weight_kg"] == 52.5 for item, _label, _meta in rows)
    assert not any(
        key.startswith("research_")
        for item, _label, _meta in rows
        for key in item
    )
