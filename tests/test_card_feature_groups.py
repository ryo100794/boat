from __future__ import annotations

from boatrace_ai.contextual_features import RollingState
from boatrace_ai.feature_tuning import (
    DEFAULT_ABLATION_FEATURE_GROUPS,
    build_race_features,
    normalize_drop_feature_groups,
)


def _entry(lane: int) -> dict:
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
        "branch": "東京",
        "origin": "東京",
        "age": 30,
        "weight_kg": 52,
        "f_count": 0,
        "l_count": 0,
        "avg_st": 0.15,
        "national_win_rate": 6.0 if lane == 1 else 5.0,
        "national_2_rate": 40.0,
        "national_3_rate": 55.0,
        "local_win_rate": 6.0 if lane == 1 else 5.0,
        "local_2_rate": 40.0,
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


def _features(*dropped: str) -> dict:
    rows = [_entry(lane) for lane in range(1, 7)]
    groups = (*dropped, "series_cached", "series_relative")
    return build_race_features(
        rows,
        RollingState(),
        drop_feature_groups=groups,
    )[0]["features"]


def test_granular_card_groups_are_explicit_only() -> None:
    assert not {
        "card_identity_context",
        "card_numeric",
        "card_relative",
    } & set(DEFAULT_ABLATION_FEATURE_GROUPS)
    assert normalize_drop_feature_groups(
        "card_identity_context,card_numeric,card_relative"
    ) == ("card_identity_context", "card_numeric", "card_relative")


def test_card_groups_can_be_ablated_independently() -> None:
    identity_dropped = _features("card_identity_context")
    assert "lane" not in identity_dropped
    assert "race_month" not in identity_dropped
    assert "national_win_rate" in identity_dropped
    assert "national_win_rate_rank" in identity_dropped

    numeric_dropped = _features("card_numeric")
    assert "lane" in numeric_dropped
    assert "race_month" in numeric_dropped
    assert "national_win_rate" not in numeric_dropped
    assert "national_win_rate_rank" in numeric_dropped

    relative_dropped = _features("card_relative", "research_correlates")
    assert "lane" in relative_dropped
    assert "national_win_rate" in relative_dropped
    assert "national_win_rate_rank" not in relative_dropped
    assert "ability_score" not in relative_dropped


def test_base_pastlog_remains_the_umbrella_ablation() -> None:
    dropped = _features("base_pastlog")
    assert "lane" not in dropped
    assert "national_win_rate" not in dropped
    assert "national_win_rate_rank" not in dropped
    assert "hist_racer_win_rate_s" in dropped
