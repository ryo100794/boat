from __future__ import annotations

import pytest

from boatrace_ai.runtime.uncertainty_adjusted_top5_policy import (
    POLICY_NAME,
    select_uncertainty_adjusted_top5_candidates,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def distribution(top: list[str], top_value: float) -> dict[str, float]:
    remainder = (1.0 - len(top) * top_value) / (120 - len(top))
    values = {combination: remainder for combination in COMBINATIONS}
    values.update({combination: top_value for combination in top})
    return values


def select(ranking, probabilities, odds, *, capital=10_000):
    return select_uncertainty_adjusted_top5_candidates(
        ranking,
        probabilities,
        odds,
        race_id="2026-08-02-01-01",
        race_date="2026-08-02",
        jcd="01",
        rno=1,
        snapshot_id=24,
        captured_at="2026-08-02T08:27:00+09:00",
        available_capital_yen=capital,
    )


def test_ranking_and_ev_probability_heads_have_separate_roles() -> None:
    ranked = sorted(COMBINATIONS)[:5]
    ranking = distribution(ranked, 0.02)
    probabilities = distribution(ranked, 0.015)
    odds = {combination: 10.0 for combination in COMBINATIONS}
    for combination in ranked:
        odds[combination] = 70.0

    selected = select(ranking, probabilities, odds)

    assert [row["combination"] for row in selected] == ranked
    for rank, row in enumerate(selected, start=1):
        combination = row["combination"]
        assert row["probability_rank"] == rank
        assert row["probability"] == probabilities[combination]
        assert row["ranking_score"] == ranking[combination]
        assert row["estimated_ev"] == pytest.approx(
            probabilities[combination] * odds[combination]
        )
        assert row["policy_name"] == POLICY_NAME


def test_probability_head_controls_ev_without_changing_ranking() -> None:
    ranked = sorted(COMBINATIONS)[:5]
    ranking = distribution(ranked, 0.02)
    low = distribution([], 0.0)
    high = dict(low)
    promoted = ranked[2]
    delta = 0.015 - high[promoted]
    high[promoted] += delta
    high[sorted(COMBINATIONS)[-1]] -= delta
    odds = {combination: 80.0 for combination in COMBINATIONS}

    assert select(ranking, low, odds) == ()
    selected = select(ranking, high, odds)
    assert [row["combination"] for row in selected] == [promoted]
    assert selected[0]["probability_rank"] == 3


def test_requires_three_aligned_complete_vectors_and_honors_capital() -> None:
    ranked = sorted(COMBINATIONS)[:5]
    distribution120 = distribution(ranked, 0.02)
    odds = {combination: 60.0 for combination in COMBINATIONS}
    assert len(select(distribution120, distribution120, odds, capital=250)) == 2
    incomplete = dict(distribution120)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="aligned 120"):
        select(distribution120, incomplete, odds)
