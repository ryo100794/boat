from __future__ import annotations

from boatrace_ai.listwise.flat_policy import (
    select_flat_policy,
    simulate_flat_policy,
)


def _identity_blend(model, _market, **_kwargs):
    return model


def _race(index: int, *, hit: bool) -> dict:
    actual = "1-2-3" if hit else "6-5-4"
    return {
        "race_id": f"2026-07-{18 + index // 30:02d}-01-{index % 12 + 1:02d}",
        "race_date": f"2026-07-{18 + index // 30:02d}",
        "actual_combination": actual,
        "actual_payout_yen": 600,
        "model_probabilities": {"1-2-3": 0.25, "6-5-4": 0.01},
        "market_probabilities": {"1-2-3": 0.15, "6-5-4": 0.01},
        "odds": {"1-2-3": 6.0, "6-5-4": 80.0},
    }


def _policy() -> dict:
    return {
        "name": "test",
        "max_model_rank": 1,
        "min_odds": 5.0,
        "max_odds": 20.0,
        "ev_threshold": 1.0,
        "min_model_market_ratio": 1.0,
    }


def test_flat_policy_stakes_one_unit_per_selected_ticket() -> None:
    result = simulate_flat_policy(
        [_race(0, hit=True), _race(1, hit=False)],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy=_policy(),
        probability_blender=_identity_blend,
    )

    assert result["tickets"] == 2
    assert result["stake_yen"] == 200
    assert result["return_yen"] == 600
    assert result["profit_yen"] == 400
    assert result["roi"] == 3.0


def test_flat_policy_selection_requires_fifty_tickets_and_multiple_winning_days() -> None:
    profitable = [_race(index, hit=index % 3 == 0) for index in range(60)]
    selected, rows = select_flat_policy(
        profitable,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        probability_blender=_identity_blend,
        policies=[{"name": "no_bet", "no_bet": True}, _policy()],
    )

    assert selected["name"] == "test"
    candidate = next(row for row in rows if row["policy"]["name"] == "test")
    assert candidate["tickets"] == 60
    assert candidate["eligible"] is True


def test_flat_policy_selection_falls_back_to_no_bet_when_sample_is_small() -> None:
    selected, rows = select_flat_policy(
        [_race(index, hit=True) for index in range(20)],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        probability_blender=_identity_blend,
        policies=[{"name": "no_bet", "no_bet": True}, _policy()],
    )

    assert selected == {"name": "no_bet", "no_bet": True}
    candidate = next(row for row in rows if row["policy"]["name"] == "test")
    assert candidate["eligible"] is False
