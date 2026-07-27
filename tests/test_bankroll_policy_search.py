from __future__ import annotations

from boatrace_ai.bankroll_policy_search import (
    CONSERVATIVE_POLICY_ANCHORS,
    policy_candidates,
    recent_allocation_diagnostics,
    slice_day_range,
    successive_halving_search,
)
from boatrace_ai.packed_bankroll import pack_candidates


POLICY = {
    "daily_budget_yen": 10_000,
    "ev_threshold": 1.0,
    "min_ticket_probability": 0.0,
    "max_estimated_odds": None,
    "payout_prior_weight": 30.0,
    "fractional_kelly": 0.25,
    "max_daily_exposure_fraction": 0.60,
    "min_daily_exposure_fraction": 0.40,
    "race_cap_fraction": 0.10,
    "ticket_cap_fraction": 0.03,
    "max_daily_tickets": 30,
    "allocation_mode": "normalized_kelly",
    "stake_granularity_yen": 100,
    "min_stake_yen": 100,
}


def _packed_days():
    candidates_by_date = {}
    evaluated = {}
    for day in range(1, 9):
        date = f"2026-07-{day:02d}"
        candidates_by_date[date] = [
            {
                "race_id": f"r{day}",
                "estimated_odds": 8.0,
                "estimated_ev": 1.6,
                "probability": 0.2,
                "actual_payout_yen": 800,
                "hit": day % 2 == 0,
            },
            {
                "race_id": f"r{day}",
                "estimated_odds": 10.0,
                "estimated_ev": 1.1,
                "probability": 0.11,
                "actual_payout_yen": 1000,
                "hit": False,
            },
        ]
        evaluated[date] = 1
    return pack_candidates(candidates_by_date, evaluated)


def test_policy_candidates_are_unique_and_reproducible() -> None:
    first = policy_candidates(POLICY, count=16, seed=7)
    second = policy_candidates(POLICY, count=16, seed=7)
    assert first == second
    assert first[0] == POLICY
    assert len({
        tuple(candidate[key] for key in (
            "ev_threshold", "min_ticket_probability", "max_estimated_odds",
            "fractional_kelly", "max_daily_exposure_fraction",
            "min_daily_exposure_fraction", "race_cap_fraction",
            "ticket_cap_fraction", "max_daily_tickets",
        ))
        for candidate in first
    }) == 16


def test_successive_halving_bootstraps_only_finalists() -> None:
    result = successive_halving_search(
        _packed_days(),
        POLICY,
        candidate_count=9,
        finalists=2,
        bootstrap_samples=100,
        seed=7,
    )
    assert [row["evaluated_candidates"] for row in result["stages"]] == [9, 3, 2]
    assert len(result["finalists"]) == 2
    finalist_policies = [row["policy"] for row in result["finalists"]]
    for overrides in CONSERVATIVE_POLICY_ANCHORS:
        assert {**POLICY, **overrides} in finalist_policies
    assert [
        row["protected_anchor_count"] for row in result["stages"]
    ] == [2, 2, 2]
    assert result["selected"] == result["finalists"][0]
    assert "roi_ci95_lower" in result["selected"]["confidence"]
    assert len(result["selected"]["temporal_stability"]["folds"]) == 3
    assert (
        "minimum_temporal_roi_above_one"
        in result["selected"]["promotion_gate"]
    )


def test_policy_candidates_preserve_conservative_anchors() -> None:
    candidates = policy_candidates(POLICY, count=4, seed=7)

    for overrides in CONSERVATIVE_POLICY_ANCHORS:
        assert {**POLICY, **overrides} in candidates

    assert max(candidate["ev_threshold"] for candidate in candidates) >= 2.0


def test_slice_day_range_rebases_offsets() -> None:
    packed = _packed_days()
    sliced = slice_day_range(packed, 2, 5)
    assert sliced.dates == ("2026-07-03", "2026-07-04", "2026-07-05")
    assert sliced.offsets.tolist() == [0, 2, 4, 6]
    assert sliced.tickets == 6
    assert sliced.evaluated_races.tolist() == [1, 1, 1]


def test_recent_allocation_diagnostics_flags_purchase_spike() -> None:
    daily = [
        {"stake_yen": 100, "tickets": 1}
        for _ in range(21)
    ] + [
        {"stake_yen": 800, "tickets": 8}
        for _ in range(7)
    ]

    diagnostics = recent_allocation_diagnostics(daily)

    assert diagnostics["stake_multiplier"] == 8.0
    assert diagnostics["ticket_multiplier"] == 8.0
    assert diagnostics["stable"] is False


def test_recent_allocation_diagnostics_accepts_normal_variation() -> None:
    daily = [
        {"stake_yen": 200, "tickets": 2}
        for _ in range(21)
    ] + [
        {"stake_yen": 300, "tickets": 3}
        for _ in range(7)
    ]

    assert recent_allocation_diagnostics(daily)["stable"] is True
