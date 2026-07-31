from __future__ import annotations

from boatrace_ai.listwise.flat_policy import (
    select_flat_policy,
    simulate_chronological_flat_policy,
    simulate_flat_policy,
)
from datetime import datetime, timedelta, timezone


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


def test_flat_policy_can_restrict_a_preregistered_narrow_ev_band() -> None:
    race = _race(0, hit=True)
    race["model_probabilities"] = {"1-2-3": 0.10, "6-5-4": 0.09}
    race["market_probabilities"] = {"1-2-3": 0.10, "6-5-4": 0.09}
    race["odds"] = {"1-2-3": 10.2, "6-5-4": 12.0}
    result = simulate_flat_policy(
        [race],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy={
            "max_model_rank": 2,
            "ev_threshold": 1.0,
            "max_estimated_ev": 1.05,
            "min_model_market_ratio": 0.0,
        },
        probability_blender=_identity_blend,
    )

    assert result["tickets"] == 1
    assert result["hit_tickets"] == 1
    assert result["daily"][0]["races_bet"] == 1


def test_chronological_flat_policy_enforces_live_gross_capital_rule() -> None:
    start = datetime(2026, 7, 30, 1, 0, tzinfo=timezone.utc)
    races = []
    for index in range(101):
        decision = start + timedelta(minutes=index)
        races.append({
            **_race(index, hit=False),
            "race_date": "2026-07-30",
            "race_id": f"2026-07-30-01-{index + 1:03d}",
            "jcd": "01",
            "rno": index + 1,
            "captured_at": decision.isoformat(),
            "odds_deadline_at": (decision + timedelta(minutes=5)).isoformat(),
        })

    result = simulate_chronological_flat_policy(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy=_policy(),
        probability_blender=_identity_blend,
    )

    assert result["tickets"] == 100
    assert result["stake_yen"] == 10_000
    assert result["return_yen"] == 0
    assert result["daily"][0]["gross_stake_yen"] == 10_000
    assert result["daily"][0]["final_gross_stake_allowance_yen"] == 10_000
    assert result["daily"][0]["profit_reinvestment"] is True


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
