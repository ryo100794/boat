from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from boatrace_ai.listwise.conditional_ticket_residual_v30 import (
    FEATURE_VARIANTS,
    _dimension,
    _objective_gradient,
    _prepare,
    conditional_metrics,
    evaluate_temporal_conditional_ticket_residual,
    fit_conditional_ticket_residual,
    ticket_feature_matrix,
)


def _race(day: date, *, actual: str = "1-2-3", jcd: str = "01", rno: int = 1) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-{jcd}-{rno:02d}",
        "race_date": race_date,
        "jcd": jcd,
        "rno": rno,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:00:00+09:00",
        "actual_combination": actual,
        "actual_payout_yen": 250 if actual == "1-2-3" else 800,
        "odds": {"1-2-3": 2.5, "2-1-3": 8.0},
        "model_probabilities": {"1-2-3": 0.55, "2-1-3": 0.45},
        "market_probabilities": {"1-2-3": 0.65, "2-1-3": 0.35},
        "lane_context": {
            str(lane): {
                "class_rank": float(lane),
                "national_win_rate": float(8 - lane),
                "research_local_vs_national_win": float(lane == 1),
                "research_home_branch": float(lane == 2),
                "motor_2_rate": float(50 - lane),
                "boat_2_rate": float(45 - lane),
                "hist_racer_win_rate_s": float(7 - lane),
                "hist_racer_venue_win_rate_s": float(6 - lane),
                "hist_motor_win_rate_s": float(5 - lane),
                "hist_boat_win_rate_s": float(4 - lane),
            }
            for lane in range(1, 7)
        },
    }


def test_ticket_features_capture_disagreement_and_context_without_results() -> None:
    race = _race(date(2026, 1, 1), jcd="07", rno=10)
    combinations = ("1-2-3", "2-1-3")
    active = FEATURE_VARIANTS["full_conditional"]

    matrix = ticket_feature_matrix(race, combinations, active)

    assert matrix.shape == (2, len(active))
    assert np.all(np.isfinite(matrix))
    venue_column = active.index("residual_venue_07")
    other_venue_column = active.index("residual_venue_06")
    assert np.any(matrix[:, venue_column] != 0.0)
    assert np.all(matrix[:, other_venue_column] == 0.0)


def test_v30_gradient_matches_finite_difference() -> None:
    variant = "race_shape_number"
    active = FEATURE_VARIANTS[variant]
    prepared = [_prepare(_race(date(2026, 1, 1)), active)]
    coefficients = np.zeros(_dimension(active), dtype=np.float64)
    _, gradient = _objective_gradient(coefficients, prepared, regularization=0.03)
    epsilon = 1e-6
    for index in (0, len(coefficients) - 1):
        forward = coefficients.copy()
        backward = coefficients.copy()
        forward[index] += epsilon
        backward[index] -= epsilon
        forward_loss, _ = _objective_gradient(
            forward, prepared, regularization=0.03
        )
        backward_loss, _ = _objective_gradient(
            backward, prepared, regularization=0.03
        )
        numerical = (forward_loss - backward_loss) / (2.0 * epsilon)
        assert abs(float(gradient[index]) - numerical) < 1e-6


def test_v30_fit_and_metrics_are_probability_normalized() -> None:
    start = date(2026, 1, 1)
    races = [
        _race(
            start + timedelta(days=index),
            actual="2-1-3" if index % 4 == 0 else "1-2-3",
            jcd=f"{index % 3 + 1:02d}",
            rno=index % 12 + 1,
        )
        for index in range(16)
    ]
    artifact = fit_conditional_ticket_residual(
        races,
        variant="race_shape_number",
        regularization=0.03,
        max_iterations=80,
    )
    metrics = conditional_metrics(races, artifact)

    assert artifact["feature_dimension"] == len(artifact["coefficients"])
    assert artifact["active_ticket_feature_count"] > 0
    assert metrics["evaluated_races"] == len(races)
    assert 0.0 <= metrics["trifecta_top5_hit_rate"] <= 1.0


def test_v30_temporal_roles_are_strictly_ordered() -> None:
    start = date(2026, 1, 1)
    calibration = [
        _race(
            start + timedelta(days=index),
            actual="2-1-3" if index % 5 == 0 else "1-2-3",
            rno=index % 12 + 1,
        )
        for index in range(42)
    ]
    evaluation = [_race(start + timedelta(days=42))]

    result = evaluate_temporal_conditional_ticket_residual(
        calibration,
        evaluation,
        daily_budget_yen=10_000,
        variants=("rank_disagreement",),
        regularizations=(0.03,),
        bootstrap_samples=100,
    )

    assert result["status"] == "completed"
    assert result["ranking_training_through"] < result["policy_calibration_from"]
    assert result["policy_calibration_through"] < result["evaluation_from"]
    assert result["metrics"]["evaluated_races"] == 1
