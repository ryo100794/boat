from __future__ import annotations

import itertools
import math
import random

import pytest

from boatrace_ai.multinomial_kelly import (
    MultinomialKellyCandidate,
    allocate_multinomial_kelly,
)


def _brute_force(
    candidates: list[MultinomialKellyCandidate],
    *,
    bankroll_yen: int,
    unit_yen: int,
    cap_units: int,
) -> tuple[float, tuple[int, ...]]:
    best = math.log(bankroll_yen)
    best_units = (0,) * len(candidates)
    for units in itertools.product(range(cap_units + 1), repeat=len(candidates)):
        if sum(units) > cap_units:
            continue
        cash = bankroll_yen - sum(units) * unit_yen
        if cash <= 0:
            continue
        objective = math.fsum(
            candidate.probability
            * math.log(cash + count * unit_yen * candidate.final_odds)
            for candidate, count in zip(candidates, units, strict=True)
        )
        if objective > best:
            best = objective
            best_units = units
    return best, best_units


def test_matches_brute_force_small_random_problems() -> None:
    random_source = random.Random(20260728)
    for _ in range(30):
        raw_probabilities = [random_source.uniform(0.1, 1.0) for _ in range(3)]
        total = sum(raw_probabilities)
        candidates = [
            MultinomialKellyCandidate(
                selection=str(index + 1),
                probability=raw_probability / total,
                final_odds=random_source.uniform(1.1, 12.0),
            )
            for index, raw_probability in enumerate(raw_probabilities)
        ]
        expected_objective, expected_units = _brute_force(
            candidates,
            bankroll_yen=1_000,
            unit_yen=100,
            cap_units=3,
        )

        result = allocate_multinomial_kelly(
            1_000,
            candidates,
            race_cap_yen=300,
            daily_cap_yen=800,
            daily_staked_yen=200,
        )

        assert result.expected_log_wealth == pytest.approx(
            expected_objective, abs=1e-12
        )
        assert math.fsum(
            allocation.probability * math.log(result.wealth_if(allocation.selection))
            for allocation in result.allocations
        ) == pytest.approx(result.expected_log_wealth, abs=1e-12)

        assert tuple(item.units for item in result.allocations) == expected_units

def test_zero_units_when_cash_has_the_highest_expected_log_growth() -> None:
    result = allocate_multinomial_kelly(
        10_000,
        [
            MultinomialKellyCandidate("1-2-3", 0.5, 1.5),
            MultinomialKellyCandidate("2-1-3", 0.5, 1.5),
        ],
        race_cap_yen=1_000,
        daily_cap_yen=3_000,
    )

    assert result.total_units == 0
    assert result.total_stake_yen == 0
    assert result.cash_yen == 10_000
    assert result.purchased == ()
    assert result.expected_log_growth == pytest.approx(0.0)


def test_result_is_deterministic_and_input_order_invariant() -> None:
    candidates = [
        MultinomialKellyCandidate("3-2-1", 0.2, 8.0),
        MultinomialKellyCandidate("1-2-3", 0.4, 3.5),
        MultinomialKellyCandidate("2-1-3", 0.4, 3.5),
    ]
    forward = allocate_multinomial_kelly(
        10_000, candidates, race_cap_yen=500, daily_cap_yen=1_000
    )
    reverse = allocate_multinomial_kelly(
        10_000, reversed(candidates), race_cap_yen=500, daily_cap_yen=1_000
    )

    assert forward == reverse
    assert [allocation.selection for allocation in forward.allocations] == [
        "1-2-3",
        "2-1-3",
        "3-2-1",
    ]


def test_race_and_remaining_daily_caps_are_both_enforced() -> None:
    candidates = [
        MultinomialKellyCandidate("fav", 0.9, 2.0),
        MultinomialKellyCandidate("other", 0.1, 2.0),
    ]
    race_limited = allocate_multinomial_kelly(
        10_000,
        candidates,
        race_cap_yen=500,
        daily_cap_yen=3_000,
        daily_staked_yen=0,
    )
    day_limited = allocate_multinomial_kelly(
        10_000,
        candidates,
        race_cap_yen=1_000,
        daily_cap_yen=3_000,
        daily_staked_yen=2_700,
    )

    assert race_limited.total_stake_yen == 500
    assert race_limited.effective_cap_yen == 500
    assert day_limited.total_stake_yen == 300
    assert day_limited.effective_cap_yen == 300


def test_never_allows_bankruptcy_even_when_caps_exceed_bankroll() -> None:
    result = allocate_multinomial_kelly(
        300,
        [
            MultinomialKellyCandidate("fav", 1.0, 100.0),
            MultinomialKellyCandidate("impossible", 0.0, 2.0),
        ],
        race_cap_yen=10_000,
        daily_cap_yen=10_000,
    )

    assert result.total_stake_yen <= 200
    assert result.cash_yen >= 100
    assert all(result.wealth_if(item.selection) > 0.0 for item in result.allocations)


@pytest.mark.parametrize(
    "candidates, message",
    [
        (
            [
                MultinomialKellyCandidate("a", 0.4, 2.0),
                MultinomialKellyCandidate("b", 0.4, 2.0),
            ],
            "sum to 1",
        ),
        ([MultinomialKellyCandidate("a", -0.1, 2.0)], "nonnegative"),
        ([MultinomialKellyCandidate("a", 1.0, math.inf)], "finite and positive"),
        (
            [
                MultinomialKellyCandidate("a", 0.5, 2.0),
                MultinomialKellyCandidate("a", 0.5, 3.0),
            ],
            "duplicate",
        ),
    ],
)
def test_rejects_invalid_probability_or_outcome_data(
    candidates: list[MultinomialKellyCandidate], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        allocate_multinomial_kelly(10_000, candidates)


def test_probability_tolerance_validates_rounding_without_renormalizing() -> None:
    candidates = [
        MultinomialKellyCandidate("a", 0.3333333333, 3.1),
        MultinomialKellyCandidate("b", 0.3333333333, 3.1),
        MultinomialKellyCandidate("c", 0.3333333333, 3.1),
    ]

    result = allocate_multinomial_kelly(10_000, candidates)

    assert math.fsum(item.probability for item in result.allocations) == pytest.approx(
        0.9999999999
    )


def test_full_trifecta_field_stays_within_bounded_exact_search() -> None:
    candidates = [
        MultinomialKellyCandidate(
            selection=f"combination-{index:03d}",
            probability=1.0 / 120.0,
            final_odds=130.0 if index == 0 else 100.0,
        )
        for index in range(120)
    ]

    result = allocate_multinomial_kelly(
        10_000,
        candidates,
        race_cap_yen=500,
        daily_cap_yen=3_000,
    )

    assert len(result.allocations) == 120
    assert result.total_units <= 5
    assert result.cash_yen > 0


def test_requires_explicit_opt_in_for_unusually_large_search_grid() -> None:
    candidates = [MultinomialKellyCandidate("certain", 1.0, 2.0)]

    with pytest.raises(ValueError, match="increase max_search_units"):
        allocate_multinomial_kelly(
            2_000_000,
            candidates,
            race_cap_yen=2_000_000,
            daily_cap_yen=2_000_000,
        )
