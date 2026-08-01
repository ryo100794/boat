from __future__ import annotations

import pytest

from boatrace_ai.listwise.dual_head_conformal_policy_v32 import (
    simulate_dual_head_conformal_policy_v32,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def distribution(top: list[str], value: float) -> dict[str, float]:
    remainder = (1.0 - value * len(top)) / (120 - len(top))
    result = {combination: remainder for combination in COMBINATIONS}
    result.update({combination: value for combination in top})
    return result


def race(actual: str, *, payout: int = 7_000) -> dict:
    ranked = sorted(COMBINATIONS)[:5]
    lower = {combination: 10.0 for combination in COMBINATIONS}
    lower.update({combination: 70.0 for combination in ranked})
    return {
        "race_id": "2026-07-31-01-01",
        "race_date": "2026-07-31",
        "jcd": "01",
        "rno": 1,
        "model_probabilities": distribution(ranked, 0.02),
        "market_probabilities": distribution([], 0.0),
        "estimated_final_odds": lower,
        "actual_combination": actual,
        "actual_payout_yen": payout,
        "captured_at": "2026-07-31T10:00:00+09:00",
        "odds_deadline_at": "2026-07-31T10:05:00+09:00",
    }


def test_uses_ranking_head_for_order_and_probability_head_for_ev() -> None:
    ranked = sorted(COMBINATIONS)[:5]
    probability = distribution(ranked, 0.015)
    ranking = distribution(ranked, 0.02)

    def blender(model, market, *, model_weight, temperature):
        return probability if model_weight == 0.25 else ranking

    result = simulate_dual_head_conformal_policy_v32(
        [race(ranked[2])],
        probability_calibrator={"model_weight": 0.25, "temperature": 1.0},
        ranking_calibrator={"model_weight": 0.75, "temperature": 1.0},
        probability_blender=blender,
    )

    assert result["tickets"] == 5
    assert result["hit_tickets"] == 1
    assert result["stake_yen"] == 500
    assert result["return_yen"] == 7_000
    assert result["policy"]["ranking_source"] == "ranking_head"
    assert result["policy"]["probability_source"] == "probability_head"


def test_rejects_incomplete_outcome_vectors() -> None:
    item = race("1-2-3")
    item["estimated_final_odds"].pop(next(iter(item["estimated_final_odds"])))

    with pytest.raises(ValueError, match="aligned 120"):
        simulate_dual_head_conformal_policy_v32(
            [item],
            probability_calibrator={"model_weight": 0.25, "temperature": 1.0},
            ranking_calibrator={"model_weight": 0.75, "temperature": 1.0},
            probability_blender=lambda model, market, **kwargs: model,
        )
