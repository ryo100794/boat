from __future__ import annotations

import pytest

from boatrace_ai.listwise.v25_t5_safety_policy_v35 import (
    attach_t5_safety_odds,
    fit_t5_safety_factors,
)


def _odds(value: float) -> dict[str, float]:
    return {f"c{index:03d}": value + index / 1000 for index in range(120)}


def _race(*, current: float = 10.0, closing_factor: float = 0.8) -> dict:
    t5 = _odds(current)
    closing = {key: value * closing_factor for key, value in t5.items()}
    return {
        "race_id": "r1",
        "race_date": "2026-07-20",
        "odds_checkpoints": {
            "300": {
                "odds": t5,
                "captured_age_seconds": 300,
                "captured_at": "2026-07-20T01:00:00+00:00",
            }
        },
        "official_closing_odds": closing,
        "actual_combination": "c000",
        "actual_payout_yen": 999999,
    }


def test_t5_safety_factor_uses_only_odds_ratio() -> None:
    first = _race()
    second = {**first, "actual_combination": "c119", "actual_payout_yen": 0}
    first_model = fit_t5_safety_factors([first], minimum_bucket_tickets=1)
    second_model = fit_t5_safety_factors([second], minimum_bucket_tickets=1)
    assert first_model == second_model
    assert first_model["global_factor"] == pytest.approx(0.8)
    assert first_model["uses_outcome_teacher"] is False
    assert first_model["uses_payout_teacher"] is False


def test_attach_t5_safety_odds_excludes_stale_checkpoint() -> None:
    model = fit_t5_safety_factors([_race()], minimum_bucket_tickets=1)
    eligible = _race()
    stale = _race()
    stale["race_id"] = "stale"
    stale["odds_checkpoints"]["300"]["captured_age_seconds"] = 421
    transformed, excluded = attach_t5_safety_odds([eligible, stale], model)
    assert excluded == 1
    assert len(transformed) == 1
    assert transformed[0]["captured_at"] == "2026-07-20T01:00:00+00:00"
    assert transformed[0]["estimated_final_odds"]["c000"] == pytest.approx(8.0)
