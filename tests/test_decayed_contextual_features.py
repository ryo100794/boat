from __future__ import annotations

import math

import pytest

from boatrace_ai.contextual_features import DecayedRollingState


def _row(
    race_date: str,
    *,
    rank: int = 1,
    racer_no: int = 4001,
    motor_no: int = 11,
    boat_no: int = 21,
    start_timing: float | None = 0.12,
) -> dict:
    return {
        "race_date": race_date,
        "jcd": "05",
        "lane": 1,
        "racer_no": racer_no,
        "motor_no": motor_no,
        "boat_no": boat_no,
        "rank": rank,
        "result_start_timing": start_timing,
    }


def test_effective_count_uses_date_based_half_life_decay() -> None:
    state = DecayedRollingState()
    state.update_race([_row("2026-01-01")])

    features = state.features_for(_row("2026-01-31"))

    assert features["decayed_racer_30d_effective_count"] == pytest.approx(0.5)
    assert features["decayed_racer_90d_effective_count"] == pytest.approx(2 ** (-1 / 3))
    assert features["decayed_racer_365d_effective_count"] == pytest.approx(2 ** (-30 / 365))


def test_same_day_updates_do_not_decay() -> None:
    state = DecayedRollingState()
    state.update_race([_row("2026-02-01", rank=1)])
    state.update_race([_row("2026-02-01", rank=2)])

    features = state.features_for(_row("2026-02-01"))

    for half_life in state.HALF_LIVES_DAYS:
        assert features[f"decayed_racer_{half_life}d_effective_count"] == 2.0


def test_missing_entity_ids_do_not_create_shared_buckets_or_features() -> None:
    state = DecayedRollingState()
    missing = _row("2026-03-01", racer_no=0, motor_no=0, boat_no=0)
    state.update_race([missing])

    features = state.features_for(missing)

    assert state.racer == {}
    assert state.racer_lane == {}
    assert state.racer_venue == {}
    assert state.motor == {}
    assert state.motor_lane == {}
    assert state.boat == {}
    assert features
    assert all(
        "racer" not in key and "motor" not in key and "boat" not in key
        for key in features
    )
    assert "decayed_venue_lane_30d_effective_count" in features


def test_out_of_order_update_and_past_feature_query_are_rejected() -> None:
    state = DecayedRollingState()
    state.update_race([_row("2026-04-10")])

    with pytest.raises(ValueError, match="backwards"):
        state.update_race([_row("2026-04-09")])
    with pytest.raises(ValueError, match="backwards"):
        state.features_for(_row("2026-04-09"))

    current = state.features_for(_row("2026-04-10"))
    assert current["decayed_racer_30d_effective_count"] == 1.0


def test_features_remain_finite_with_missing_start_timing() -> None:
    state = DecayedRollingState()
    state.update_race([_row("2026-05-01", start_timing=None)])

    features = state.features_for(_row("2036-05-01"))

    assert features
    assert all(math.isfinite(value) for value in features.values())
    assert features["decayed_racer_30d_avg_start_timing_s"] == pytest.approx(0.17)


def test_short_minus_long_trend_reflects_recent_improvement() -> None:
    state = DecayedRollingState()
    state.update_race([_row("2026-01-01", rank=6)])
    state.update_race([_row("2026-12-31", rank=1)])

    features = state.features_for(_row("2026-12-31"))

    assert features["decayed_racer_win_rate_s_30d_365d_trend"] > 0.0
    assert features["decayed_racer_avg_rank_s_30d_365d_trend"] < 0.0


def test_mixed_race_dates_are_rejected_before_any_update() -> None:
    state = DecayedRollingState()

    with pytest.raises(ValueError, match="same race_date"):
        state.update_race([_row("2026-06-01"), _row("2026-06-02")])

    assert state.venue_lane == {}
    assert state.racer == {}


def test_valid_unseen_entities_emit_prior_and_no_history_flag() -> None:
    state = DecayedRollingState()

    features = state.features_for(_row("2026-06-01"))

    expected_entities = {
        "venue_lane",
        "racer",
        "racer_lane",
        "racer_venue",
        "motor",
        "motor_lane",
        "boat",
    }
    for entity in expected_entities:
        assert features[f"decayed_{entity}_30d_effective_count"] == 0.0
        assert features[f"decayed_{entity}_30d_has_recent_history"] == 0.0
        assert features[f"decayed_{entity}_30d_win_rate_s"] == pytest.approx(1 / 6)
        assert features[f"decayed_{entity}_win_rate_s_30d_365d_trend"] == 0.0

    assert state.racer == {}
    assert state.motor == {}
    assert state.boat == {}
