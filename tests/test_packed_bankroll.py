from __future__ import annotations

from boatrace_ai.adaptive_allocation import allocate_adaptive_day
from boatrace_ai.packed_bankroll import (
    candidate_ev_calibration,
    evaluate_packed_policy,
    pack_candidates,
)


POLICY = {
    "daily_budget_yen": 10_000,
    "fractional_kelly": 0.25,
    "max_daily_exposure_fraction": 0.60,
    "min_daily_exposure_fraction": 0.40,
    "race_cap_fraction": 0.10,
    "ticket_cap_fraction": 0.03,
    "max_daily_tickets": 4,
    "allocation_mode": "normalized_kelly",
    "stake_granularity_yen": 100,
    "min_stake_yen": 100,
    "ev_threshold": 1.0,
}


def _candidate(race_id: str, combination: str, probability: float, odds: float, *, hit: bool = False):
    return {
        "race_id": race_id,
        "combination": combination,
        "probability": probability,
        "estimated_odds": odds,
        "estimated_ev": probability * odds,
        "actual_payout_yen": int(odds * 100),
        "hit": hit,
    }


def test_candidate_ev_calibration_reports_realized_flat_returns() -> None:
    low_hit = _candidate("r1", "1-2-3", 0.10, 11.0, hit=True)
    low_hit["actual_payout_yen"] = 800
    middle_miss = _candidate("r2", "1-3-2", 0.10, 13.0)
    high_hit = _candidate("r3", "2-1-3", 0.10, 16.0, hit=True)
    high_hit["actual_payout_yen"] = 300
    packed = pack_candidates(
        {"2026-07-01": [low_hit, middle_miss, high_hit]},
        {"2026-07-01": 3},
    )

    rows = candidate_ev_calibration(packed)

    assert rows[0]["tickets"] == 1
    assert rows[0]["realized_roi"] == 8.0
    assert rows[1]["tickets"] == 1
    assert rows[1]["realized_roi"] == 0.0
    assert rows[2]["tickets"] == 1
    assert rows[2]["realized_roi"] == 3.0
    assert rows[-1]["upper_exclusive"] is None


def test_packed_policy_matches_reference_allocator() -> None:
    candidates = [
        _candidate("r1", "1-2-3", 0.20, 8.0, hit=True),
        _candidate("r1", "1-3-2", 0.18, 7.0),
        _candidate("r1", "2-1-3", 0.12, 10.0),
        _candidate("r2", "1-2-3", 0.25, 5.0),
        _candidate("r2", "2-1-3", 0.08, 15.0),
        _candidate("r2", "3-1-2", 0.01, 20.0),
    ]
    reference = allocate_adaptive_day(
        "2026-07-01",
        candidates,
        {"r1", "r2"},
        **{key: value for key, value in POLICY.items() if key != "ev_threshold"},
    )
    packed = pack_candidates({"2026-07-01": candidates}, {"2026-07-01": 2})
    result = evaluate_packed_policy(packed, POLICY)
    day = result["daily"][0]
    assert day["tickets"] == reference["tickets"]
    assert day["selected_races"] == reference["races_bet"]
    assert day["hit_tickets"] == reference["hit_tickets"]
    assert day["hit_races"] == reference["hit_races"]
    assert day["stake_yen"] == reference["stake_yen"]
    assert day["return_yen"] == reference["return_yen"]


def test_packed_policy_keeps_empty_days() -> None:
    packed = pack_candidates({}, {"2026-07-01": 12})
    result = evaluate_packed_policy(packed, POLICY)
    assert result["evaluated_races"] == 12
    assert result["tickets"] == 0
    assert result["daily"][0]["race_date"] == "2026-07-01"


def test_ev_threshold_is_applied_without_repacking() -> None:
    candidates = [
        _candidate("r1", "1-2-3", 0.20, 8.0, hit=True),
        _candidate("r1", "1-3-2", 0.11, 10.0),
    ]
    packed = pack_candidates({"2026-07-01": candidates}, {"2026-07-01": 1})
    result = evaluate_packed_policy(packed, {**POLICY, "ev_threshold": 1.2})
    assert result["candidate_tickets"] == 1


def test_probability_and_odds_tail_filters_apply_without_repacking() -> None:
    candidates = [
        _candidate("r1", "1-2-3", 0.20, 8.0, hit=True),
        _candidate("r1", "1-3-2", 0.01, 150.0),
        _candidate("r1", "2-1-3", 0.005, 250.0),
    ]
    packed = pack_candidates(
        {"2026-07-01": candidates},
        {"2026-07-01": 1},
    )
    result = evaluate_packed_policy(packed, {
        **POLICY,
        "min_ticket_probability": 0.01,
        "max_estimated_odds": 100.0,
    })

    assert result["candidate_tickets"] == 1
