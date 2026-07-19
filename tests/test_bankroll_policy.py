from __future__ import annotations

import pytest

from boatrace_ai.bankroll_policy import (
    filter_candidates,
    race_confidence,
    select_temporal_policy,
    split_calibration_dates,
)


def test_race_confidence_is_normalized() -> None:
    even = race_confidence({lane: 1.0 for lane in range(1, 7)})
    assert even["race_top_lane_probability"] == pytest.approx(1 / 6)
    assert even["race_top_lane_margin"] == pytest.approx(0.0)
    assert even["race_normalized_entropy"] == pytest.approx(1.0)

    decisive = race_confidence({1: 0.7, 2: 0.1, 3: 0.1, 4: 0.05, 5: 0.03, 6: 0.02})
    assert decisive["race_top_lane_probability"] == pytest.approx(0.7)
    assert decisive["race_top_lane_margin"] == pytest.approx(0.6)
    assert decisive["race_normalized_entropy"] < 0.7


def test_filter_candidates_applies_market_and_confidence_limits() -> None:
    rows = [
        {
            "probability": 0.02,
            "estimated_odds": 50.0,
            "race_top_lane_probability": 0.35,
            "race_top_lane_margin": 0.10,
            "race_normalized_entropy": 0.90,
        },
        {
            "probability": 0.005,
            "estimated_odds": 120.0,
            "race_top_lane_probability": 0.22,
            "race_top_lane_margin": 0.01,
            "race_normalized_entropy": 0.99,
        },
    ]
    policy = {
        "min_ticket_probability": 0.01,
        "max_estimated_odds": 60.0,
        "min_race_top_lane_probability": 0.30,
        "min_race_top_lane_margin": 0.05,
        "max_race_normalized_entropy": 0.95,
    }
    assert filter_candidates(rows, policy) == [rows[0]]
    assert filter_candidates(rows, {"no_bet": True}) == []


def test_split_calibration_dates_preserves_time_order() -> None:
    calibration, evaluation = split_calibration_dates(
        {"2026-01-04", "2026-01-01", "2026-01-03", "2026-01-02"},
        calibration_fraction=0.25,
    )
    assert calibration == ["2026-01-01"]
    assert evaluation == ["2026-01-02", "2026-01-03", "2026-01-04"]


def test_temporal_policy_can_select_no_bet_when_calibration_loses() -> None:
    def allocate_day(race_date, candidates, evaluated_races, **_kwargs):
        stake = len(candidates) * 100
        return {
            "stake_yen": stake,
            "return_yen": 0,
            "profit_yen": -stake,
            "tickets": len(candidates),
        }

    selected, rows = select_temporal_policy(
        ["2026-01-01"],
        {"2026-01-01": [{"probability": 0.02}]},
        {"2026-01-01": {"race-1"}},
        allocate_day=allocate_day,
        allocation_kwargs={},
        policies=[{"name": "baseline"}, {"name": "no_bet", "no_bet": True}],
    )
    assert selected == {"name": "no_bet", "no_bet": True}
    assert [row["profit_yen"] for row in rows] == [-100, 0]
