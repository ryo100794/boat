from __future__ import annotations

from boatrace_ai.base_features import is_home_branch, race_relative_features
from boatrace_ai.cache_entry_series_features import ensure_series_cache_table
from boatrace_ai.contextual_features import RollingState
from boatrace_ai.feature_tuning import build_race_features


def _entry(lane: int, *, branch: str = "東京", local_delta: float = 0.0) -> dict:
    return {
        "race_id": "202607200501",
        "race_date": "2026-07-20",
        "lane": lane,
        "rno": 1,
        "jcd": "05",
        "race_type": "一般",
        "distance_m": 1800,
        "racer_no": 4000 + lane,
        "racer_name": f"選手{lane}",
        "racer_class": "A1" if lane == 1 else "B1",
        "branch": branch,
        "origin": branch,
        "age": 30,
        "weight_kg": 52,
        "f_count": 0,
        "l_count": 0,
        "avg_st": 0.15,
        "national_win_rate": 6.0 if lane == 1 else 5.0,
        "national_2_rate": 40.0 if lane == 1 else 30.0,
        "national_3_rate": 55.0,
        "local_win_rate": (6.0 if lane == 1 else 5.0) + local_delta,
        "local_2_rate": (40.0 if lane == 1 else 30.0) + local_delta,
        "local_3_rate": 55.0,
        "motor_no": lane,
        "motor_2_rate": 35.0,
        "motor_3_rate": 50.0,
        "boat_no": lane,
        "boat_2_rate": 35.0,
        "boat_3_rate": 50.0,
        "rank": lane,
        "result_course": lane,
        "result_start_timing": 0.15,
    }


def _before(lane: int, *, course: int | None = None) -> dict:
    return {
        "weight_kg": 52.0,
        "exhibition_time": 6.70 + lane / 100,
        "tilt": 0.0,
        "adjusted_weight": 0.0,
        "course": lane if course is None else course,
        "start_timing": 0.10 + lane / 100,
        "weather": "雨",
        "wind_direction": "北",
        "wind_speed_m": 4.0,
        "air_temp_c": 25.0,
        "water_temp_c": 24.0,
        "wave_cm": 4.0,
        "propeller": "",
        "parts_exchange": "",
    }


def test_home_branch_maps_shared_venue_branches() -> None:
    assert is_home_branch("03", "東京")
    assert is_home_branch("05", "東京")
    assert is_home_branch("20", "福岡")
    assert not is_home_branch("05", "埼玉")


def test_research_features_separate_home_matchup_equipment_and_live_context() -> None:
    rows = [
        _entry(lane, local_delta=1.0 if lane == 1 else 0.0)
        for lane in range(1, 7)
    ]
    features = race_relative_features(
        rows,
        {lane: _before(lane) for lane in range(1, 7)},
    )

    assert features[1]["research_home_branch"] == 1
    assert features[1]["research_local_vs_national_win"] == 1.0
    assert features[1]["research_home_local_win_delta"] == 1.0
    assert (
        features[1]["research_racer_strength"]
        > features[2]["research_racer_strength"]
    )
    assert features[1]["research_racer_strength_rank"] == 1
    assert features[1]["research_waku_nari"] == 1
    assert features[1]["research_exhibition_top1"] == 1
    assert features[1]["research_exhibition_rank_weather"] == "1:雨"
    assert "research_equipment_strength" in features[1]


def test_course_change_and_research_group_ablation() -> None:
    rows = [_entry(lane) for lane in range(1, 7)]
    before = {lane: _before(lane) for lane in range(1, 7)}
    before[2] = _before(2, course=3)
    before[3] = _before(3, course=2)
    live_features = race_relative_features(rows, before)
    assert live_features[2]["research_waku_nari"] == 0
    assert live_features[2]["research_course_changed"] == 1
    assert live_features[2]["research_course_delta"] == 1

    full = build_race_features(
        rows,
        RollingState(),
        drop_feature_groups=("series_cached", "series_relative"),
    )
    dropped = build_race_features(
        rows,
        RollingState(),
        drop_feature_groups=(
            "research_correlates",
            "series_cached",
            "series_relative",
        ),
    )
    assert any(key.startswith("research_") for key in full[0]["features"])
    assert not any(key.startswith("research_") for key in dropped[0]["features"])


def test_postgresql_series_cache_check_is_read_only() -> None:
    class FakePostgresql:
        dialect = "postgresql"

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str) -> None:
            self.statements.append(statement)

        def executescript(self, _statement: str) -> None:
            raise AssertionError("PostgreSQL schema must not be mutated by evaluation")

    conn = FakePostgresql()
    ensure_series_cache_table(conn)
    assert conn.statements == ["SELECT 1 FROM entry_series_features LIMIT 0"]
