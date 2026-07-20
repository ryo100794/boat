from __future__ import annotations

from boatrace_ai.base_features import _ranks, race_relative_features


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
