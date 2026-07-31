from __future__ import annotations

import math

import numpy as np
import pytest

from boatrace_ai.listwise.contextual_market_residual_v24 import (
    FEATURE_DIMENSION as BASE_FEATURE_DIMENSION,
)
from boatrace_ai.listwise.course_interaction_residual import (
    STRUCTURE_VARIANTS,
    _course_feature_dimension,
    _course_objective_gradient,
    _prepare_course_race,
    evaluate_temporal_course_interaction,
    fit_course_residual,
    structure_probabilities,
)
from boatrace_ai.listwise.pruned_direct_context_v27 import FEATURE_VARIANTS


def _race(day: int, actual: str = "1-2-3") -> dict:
    race_date = f"2026-02-{day:02d}"
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:00:00+09:00",
        "actual_combination": actual,
        "actual_payout_yen": 250,
        "odds": {"1-2-3": 2.5, "2-1-3": 2.5},
        "model_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "market_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "lane_context": {
            str(lane): {
                "class_rank": 4.0 if lane == 1 else 1.0,
                "national_win_rate": 9.0 if lane == 1 else 3.0,
                "national_2_rate": 70.0 if lane == 1 else 20.0,
                "motor_2_rate": float(50 - lane),
                "boat_2_rate": float(45 - lane),
            }
            for lane in range(1, 7)
        },
    }


def test_course_interaction_dimension_is_lane_gated() -> None:
    active = FEATURE_VARIANTS["ability_raw"]
    assert _course_feature_dimension(active) == (
        BASE_FEATURE_DIMENSION + 3 * 6 * 3 * 2
    )
    artifact = fit_course_residual(
        [_race(day) for day in range(1, 7)],
        feature_variant="ability_raw",
        regularization=0.1,
        max_iterations=30,
    )
    assert artifact["feature_dimension"] == BASE_FEATURE_DIMENSION + 108
    assert artifact["architecture"] == "finish_stage_by_starting_lane_context"


def test_course_interaction_gradient_matches_finite_difference() -> None:
    active = FEATURE_VARIANTS["ability_raw"]
    prepared = [_prepare_course_race(_race(1), active)]
    coefficients = np.zeros(_course_feature_dimension(active), dtype=np.float64)
    _, gradient = _course_objective_gradient(
        coefficients, prepared, regularization=0.1
    )
    index = BASE_FEATURE_DIMENSION
    epsilon = 1e-6
    forward = coefficients.copy()
    backward = coefficients.copy()
    forward[index] += epsilon
    backward[index] -= epsilon
    forward_loss, _ = _course_objective_gradient(
        forward, prepared, regularization=0.1
    )
    backward_loss, _ = _course_objective_gradient(
        backward, prepared, regularization=0.1
    )
    numerical = (forward_loss - backward_loss) / (2.0 * epsilon)
    assert gradient[index] == pytest.approx(numerical, abs=1e-6)


def test_course_interaction_learns_lane_specific_strength() -> None:
    artifact = fit_course_residual(
        [_race(day) for day in range(1, 13)],
        feature_variant="ability_raw",
        regularization=0.01,
        max_iterations=80,
    )
    probabilities = structure_probabilities(_race(13), artifact)
    assert probabilities["1-2-3"] > probabilities["2-1-3"]
    assert -math.log(probabilities["1-2-3"]) < math.log(2.0)


def test_temporal_structure_search_keeps_outer_period_separate() -> None:
    result = evaluate_temporal_course_interaction(
        [_race(day) for day in range(1, 11)],
        [_race(11)],
        policies=(),
        daily_budget_yen=10_000,
        structure_variants={
            "shared_independent_core": STRUCTURE_VARIANTS[
                "shared_independent_core"
            ],
            "course_ability_raw": STRUCTURE_VARIANTS["course_ability_raw"],
        },
        regularizations=(0.1,),
    )
    assert result["inner_fit_through"] < result["inner_validation_from"]
    assert result["metrics"]["evaluated_races"] == 1
    assert len(result["candidates"]) == 2
    assert result["purchase_diagnostics"] == []


def test_temporal_structure_search_rejects_changed_design() -> None:
    with pytest.raises(ValueError, match="preregistered"):
        evaluate_temporal_course_interaction(
            [_race(day) for day in range(1, 5)],
            [_race(5)],
            policies=(),
            daily_budget_yen=10_000,
            structure_variants={
                "course_ability_raw": ("course_gated", "equipment_history")
            },
            regularizations=(0.1,),
        )
