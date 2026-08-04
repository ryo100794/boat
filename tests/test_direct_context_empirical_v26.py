from __future__ import annotations

from datetime import date, timedelta

from boatrace_ai.listwise.direct_context_empirical_v26 import (
    evaluate_temporal_direct_context_empirical,
)


def _race(day: date) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:00:00+09:00",
        "actual_combination": "1-2-3",
        "actual_payout_yen": 250,
        "odds": {"1-2-3": 2.5, "2-1-3": 2.5},
        "model_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "market_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "lane_context": {
            str(lane): {
                "national_win_rate": 9.0 if lane == 1 else 3.0,
                "local_win_rate": 8.0 if lane == 1 else 3.0,
            }
            for lane in range(1, 7)
        },
    }


def test_v26_separates_probability_ev_and_outer_periods() -> None:
    start = date(2026, 1, 1)
    calibration = [_race(start + timedelta(days=index)) for index in range(40)]
    evaluation = [_race(start + timedelta(days=index)) for index in range(40, 42)]
    result = evaluate_temporal_direct_context_empirical(
        calibration,
        evaluation,
        daily_budget_yen=10_000,
        policy_calibration_days=30,
        bootstrap_samples=200,
        min_tickets=60,
        min_candidate_days=20,
    )
    assert result["status"] == "completed"
    assert result["probability_training_through"] == "2026-01-10"
    assert result["policy_calibration_from"] == "2026-01-11"
    assert result["policy_calibration_through"] == "2026-02-09"
    assert result["evaluation_from"] == "2026-02-10"
    assert result["empirical_ev_calibration"]["trained_through_date"] < result[
        "evaluation_from"
    ]
    assert result["empirical_ev_calibration"]["ready"] is True
    # Refit probabilities move the outer raw EV beyond the zero-width local
    # support learned from this deliberately constant synthetic calibration.
    # The production policy must fail closed instead of extrapolating.
    assert result["bankroll"]["tickets"] == 0
    assert result["bankroll"]["roi"] is None


def test_v26_refuses_too_short_calibration() -> None:
    start = date(2026, 1, 1)
    result = evaluate_temporal_direct_context_empirical(
        [_race(start + timedelta(days=index)) for index in range(39)],
        [_race(start + timedelta(days=40))],
        daily_budget_yen=10_000,
        policy_calibration_days=30,
        bootstrap_samples=200,
    )
    assert result["status"] == "insufficient_calibration_days"
    assert result["required_calibration_days"] == 40
