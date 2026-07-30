from __future__ import annotations

import math

import pytest

from boatrace_ai.discrete_log_allocation import allocate_discrete_log_day


def _candidate(
    race_id: str,
    combination: str,
    *,
    probability: float = 0.055,
    estimated_odds: float = 20.0,
    hit: bool = False,
    actual_payout_yen: int = 2_000,
) -> dict:
    return {
        "race_id": race_id,
        "race_date": "2026-07-29",
        "combination": combination,
        "probability": probability,
        "estimated_odds": estimated_odds,
        "actual_combination": "1-2-3",
        "actual_payout_yen": actual_payout_yen,
        "hit": hit,
    }


def _allocate(candidates: list[dict], **overrides: object) -> dict:
    policy = {
        "daily_budget_yen": 10_000,
        "max_daily_exposure_fraction": 0.20,
        "race_cap_fraction": 0.03,
        "ticket_cap_fraction": 0.01,
        "max_daily_tickets": None,
        "stake_granularity_yen": 100,
        "min_stake_yen": 100,
        "max_tickets_per_race": 2,
    }
    policy.update(overrides)
    return allocate_discrete_log_day(
        "2026-07-29",
        candidates,
        {str(candidate["race_id"]) for candidate in candidates},
        **policy,
    )


def _selection(result: dict) -> list[tuple[str, str, int]]:
    return sorted(
        (
            str(row["race_id"]),
            str(row["combination"]),
            int(row["stake_yen"]),
        )
        for row in result["selected_sample"]
    )


def test_discrete_log_selects_one_unit_when_quarter_kelly_rounds_to_zero() -> None:
    probability = 0.055
    odds = 20.0
    full_kelly = (probability * odds - 1.0) / (odds - 1.0)
    quarter_kelly_stake = 10_000 * 0.25 * full_kelly
    assert quarter_kelly_stake < 100

    result = _allocate([_candidate("race-1", "1-2-3")])

    expected_log = probability * math.log(1.19) + (1.0 - probability) * math.log(
        0.99
    )
    assert expected_log > 0.0
    assert result["tickets"] == 1
    assert result["stake_yen"] == 100
    assert result["expected_log_growth"] == pytest.approx(expected_log)


def test_discrete_log_keeps_zero_ticket_option_for_negative_growth() -> None:
    result = _allocate(
        [
            _candidate(
                "race-1",
                "1-2-3",
                probability=0.04,
                estimated_odds=20.0,
            )
        ]
    )

    assert result["positive_edge_tickets"] == 0
    assert result["tickets"] == 0
    assert result["stake_yen"] == 0
    assert result["roi"] is None
    assert result["expected_log_growth"] == 0.0


def test_two_ticket_portfolio_uses_mutually_exclusive_outcomes() -> None:
    result = _allocate(
        [
            _candidate("race-1", "1-2-3"),
            _candidate("race-1", "1-3-2"),
        ],
        race_cap_fraction=0.02,
    )

    expected_log = (
        2 * 0.055 * math.log(1.18)
        + (1.0 - 2 * 0.055) * math.log(0.98)
    )
    assert result["tickets"] == 2
    assert result["races_bet"] == 1
    assert result["stake_yen"] == 200
    assert result["expected_log_growth"] == pytest.approx(expected_log)
    assert result["race_portfolios"] == [
        {
            "race_id": "race-1",
            "tickets": 2,
            "stake_yen": 200,
            "expected_log_growth": pytest.approx(expected_log),
        }
    ]


def test_ticket_race_day_and_count_caps_are_all_enforced() -> None:
    result = _allocate(
        [
            _candidate(
                "race-1",
                combination,
                probability=0.20,
                estimated_odds=10.0,
            )
            for combination in ("1-2-3", "1-3-2", "2-1-3")
        ],
        max_daily_exposure_fraction=0.03,
        race_cap_fraction=0.03,
        ticket_cap_fraction=0.02,
        max_daily_tickets=2,
    )

    assert result["tickets"] == 2
    assert result["stake_yen"] == 300
    assert result["max_stake_yen"] <= 200
    assert result["race_portfolios"][0]["stake_yen"] <= 300
    assert all(row["stake_yen"] <= 200 for row in result["selected_sample"])


def test_selection_is_deterministic_and_does_not_use_settlement_fields() -> None:
    original = [
        _candidate(
            "race-b",
            "1-2-3",
            hit=True,
            actual_payout_yen=1_000_000,
        ),
        _candidate("race-a", "1-2-3", hit=False, actual_payout_yen=0),
    ]
    changed = [
        {
            **original[1],
            "hit": True,
            "actual_combination": "1-2-3",
            "actual_payout_yen": 2_000_000,
        },
        {
            **original[0],
            "hit": False,
            "actual_combination": "6-5-4",
            "actual_payout_yen": 0,
        },
    ]

    first = _allocate(
        original,
        max_daily_exposure_fraction=0.01,
        race_cap_fraction=0.01,
    )
    second = _allocate(
        changed,
        max_daily_exposure_fraction=0.01,
        race_cap_fraction=0.01,
    )

    assert _selection(first) == _selection(second) == [("race-a", "1-2-3", 100)]
    assert first["expected_log_growth"] == second["expected_log_growth"]
    assert first["return_yen"] != second["return_yen"]


def test_actual_results_are_used_only_to_aggregate_profit_and_roi() -> None:
    hit = _candidate("race-1", "1-2-3")
    hit.pop("hit")
    miss = _candidate("race-2", "2-1-3")
    miss.pop("hit")

    result = _allocate(
        [hit, miss],
        max_daily_exposure_fraction=0.02,
        race_cap_fraction=0.01,
    )

    assert result["tickets"] == 2
    assert result["hit_tickets"] == 1
    assert result["hit_races"] == 1
    assert result["stake_yen"] == 200
    assert result["return_yen"] == 2_000
    assert result["profit_yen"] == 1_800
    assert result["roi"] == 10.0
    assert result["largest_hit_return_yen"] == 2_000
    assert result["hit_return_square_sum_yen2"] == 4_000_000
