from __future__ import annotations

from datetime import date, timedelta

import pytest

from boatrace_ai.listwise.payout_weighted_ranking import (
    evaluate_temporal_payout_weighted_roles,
    fit_payout_weighted_ranking,
    payout_race_weight,
    payout_ranking_metrics,
)


def _race(day: date, *, actual: str = "1-2-3", payout: int = 250) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:00:00+09:00",
        "actual_combination": actual,
        "actual_payout_yen": payout,
        "odds": {"1-2-3": 2.5, "2-1-3": 8.0},
        "model_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "market_probabilities": {"1-2-3": 0.6, "2-1-3": 0.4},
        "lane_context": {
            str(lane): {
                "class_rank": 1.0 if lane == 1 else 4.0,
                "national_win_rate": 8.0 if lane == 1 else 4.0,
                "motor_2_rate": float(50 - lane),
                "boat_2_rate": float(45 - lane),
            }
            for lane in range(1, 7)
        },
    }


def test_payout_weight_is_neutral_at_zero_and_capped_for_tail() -> None:
    start = date(2026, 1, 1)
    assert payout_race_weight(_race(start), exponent=0.0) == 1.0
    tail = _race(start, payout=100_000)
    assert payout_race_weight(tail, exponent=0.45) <= 4.0
    assert payout_race_weight(tail, exponent=0.45) > 1.0


def test_payout_weight_rejects_invalid_teacher() -> None:
    race = _race(date(2026, 1, 1), payout=0)
    with pytest.raises(ValueError, match="positive actual payout"):
        payout_race_weight(race, exponent=0.3)


def test_weighted_ranking_fits_and_reports_role_metrics() -> None:
    start = date(2026, 1, 1)
    races = [
        _race(
            start + timedelta(days=index),
            actual="2-1-3" if index % 4 == 0 else "1-2-3",
            payout=2000 if index % 4 == 0 else 250,
        )
        for index in range(16)
    ]
    artifact = fit_payout_weighted_ranking(
        races, exponent=0.3, max_iterations=80
    )
    metrics = payout_ranking_metrics(races, artifact)
    assert artifact["role"] == "payout_weighted_ranking_only"
    assert artifact["feature_dimension"] == len(artifact["coefficients"])
    assert metrics["evaluated_races"] == 16
    assert metrics["top5_flat_stake_yen"] == 16 * 500


def test_role_evaluation_separates_ranking_policy_and_outer_days() -> None:
    start = date(2026, 1, 1)
    calibration = [
        _race(start + timedelta(days=index)) for index in range(42)
    ]
    evaluation = [_race(start + timedelta(days=42))]
    result = evaluate_temporal_payout_weighted_roles(
        calibration,
        evaluation,
        daily_budget_yen=10_000,
        exponents=(0.0, 0.3),
        bootstrap_samples=200,
    )
    assert result["status"] == "completed"
    assert result["ranking_training_through"] < result["policy_calibration_from"]
    assert result["policy_calibration_through"] < result["evaluation_from"]
    assert len(result["candidates"]) == 2
    assert result["probability_metrics"]["evaluated_races"] == 1
    assert result["ranking_metrics"]["evaluated_races"] == 1


def test_role_evaluation_refuses_short_calibration() -> None:
    start = date(2026, 1, 1)
    result = evaluate_temporal_payout_weighted_roles(
        [_race(start + timedelta(days=index)) for index in range(39)],
        [_race(start + timedelta(days=40))],
        daily_budget_yen=10_000,
    )
    assert result["status"] == "insufficient_calibration_days"
    assert result["required_calibration_days"] == 40
