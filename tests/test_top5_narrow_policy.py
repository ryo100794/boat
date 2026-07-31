from __future__ import annotations

import pytest

from boatrace_ai.runtime.top5_narrow_policy import (
    POLICY_NAME,
    daily_capital_limits,
    select_top5_narrow_candidates,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def distribution() -> dict[str, float]:
    values = {combination: 0.9 / 115 for combination in COMBINATIONS}
    for combination in sorted(COMBINATIONS)[:5]:
        values[combination] = 0.02
    return values


def select(*, capital: int = 10_000, odds: dict[str, float] | None = None):
    return select_top5_narrow_candidates(
        distribution(),
        odds or {combination: 51.0 for combination in COMBINATIONS},
        race_id="2026-07-31-01-01",
        race_date="2026-07-31",
        jcd="01",
        rno=1,
        snapshot_id=23,
        captured_at="2026-07-31T08:27:00+09:00",
        available_capital_yen=capital,
    )


def test_selects_only_top5_inside_registered_forecast_ev_band() -> None:
    odds = {combination: 200.0 for combination in COMBINATIONS}
    ranked = sorted(COMBINATIONS)[:5]
    odds[ranked[0]] = 50.0
    odds[ranked[1]] = 52.5
    odds[ranked[2]] = 53.0
    selected = select(odds=odds)

    assert [row["combination"] for row in selected] == ranked[:2]
    assert [row["probability_rank"] for row in selected] == [1, 2]
    assert all(row["stake_yen"] == 100 for row in selected)
    assert all(row["policy_name"] == POLICY_NAME for row in selected)
    assert all("hit" not in row and "return_yen" not in row for row in selected)


def test_capital_limits_ticket_count_and_invalid_vectors() -> None:
    assert len(select(capital=250)) == 2
    assert select(capital=99) == ()
    probabilities = distribution()
    probabilities.pop(next(iter(probabilities)))
    with pytest.raises(ValueError, match="aligned 120"):
        select_top5_narrow_candidates(
            probabilities,
            {combination: 51.0 for combination in COMBINATIONS},
            race_id="race",
            race_date="2026-07-31",
            jcd="01",
            rno=1,
            snapshot_id=1,
            captured_at="now",
            available_capital_yen=100,
        )


def test_daily_capital_uses_settlements_only_for_realized_allowance() -> None:
    limits = daily_capital_limits(
        [
            {"total_stake_yen": 100, "profit_yen": -100},
            {"total_stake_yen": 200, "profit_yen": 600},
            {"total_stake_yen": 100, "profit_yen": None},
        ],
        bankroll_yen=10_100,
    )

    assert limits == {
        "gross_stake_yen": 400,
        "realized_cumulative_profit_yen": 500,
        "gross_stake_allowance_yen": 10_500,
        "remaining_gross_stake_allowance_yen": 10_100,
        "allocatable_bankroll_yen": 10_100,
    }
