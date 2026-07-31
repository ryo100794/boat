from __future__ import annotations

import math

import pytest

from boatrace_ai.listwise.contextual_market_residual_v24 import (
    FEATURE_DIMENSION as BASE_FEATURE_DIMENSION,
)
from boatrace_ai.listwise.pruned_direct_context_evaluation_v27 import (
    evaluate_temporal_pruned_residual,
)
from boatrace_ai.listwise.pruned_direct_context_v27 import (
    FEATURE_VARIANTS,
    fit_pruned_residual,
    fit_temporal_pruned_residual,
    pruned_probabilities,
)


def _race(day: int, actual: str = "1-2-3") -> dict:
    race_date = f"2026-01-{day:02d}"
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
            }
            for lane in range(1, 7)
        },
    }


def test_v27_really_reduces_the_fitted_dimension() -> None:
    races = [_race(day) for day in range(1, 9)]
    market_only = fit_pruned_residual(
        races, variant="market_model_only", regularization=0.1, max_iterations=20
    )
    ability = fit_pruned_residual(
        races, variant="ability_raw", regularization=0.1, max_iterations=30
    )
    assert market_only["feature_dimension"] == BASE_FEATURE_DIMENSION
    assert ability["feature_dimension"] == BASE_FEATURE_DIMENSION + 3 * 3 * 2
    assert len(ability["coefficients"]) == ability["feature_dimension"]
    assert set(ability["active_context_features"]) == set(
        FEATURE_VARIANTS["ability_raw"]
    )


def test_v27_pruned_ability_learns_a_card_level_residual() -> None:
    artifact = fit_pruned_residual(
        [_race(day) for day in range(1, 13)],
        variant="ability_raw",
        regularization=0.01,
        max_iterations=50,
    )
    probabilities = pruned_probabilities(_race(13), artifact)
    assert probabilities["1-2-3"] > probabilities["2-1-3"]
    assert -math.log(probabilities["1-2-3"]) < math.log(2.0)


def test_v27_selection_prefers_smaller_equivalent_variant() -> None:
    result = fit_temporal_pruned_residual(
        [_race(day) for day in range(1, 11)],
        [_race(11)],
        variants={
            "market_model_only": FEATURE_VARIANTS["market_model_only"],
            "ability_raw": FEATURE_VARIANTS["ability_raw"],
        },
        regularizations=(0.1,),
        selection_tolerance=10.0,
    )
    assert result["selected_candidate"]["variant"] == "market_model_only"
    assert len(result["candidates"]) == 2


def test_v27_evaluation_includes_bankroll_diagnostics() -> None:
    result = evaluate_temporal_pruned_residual(
        [_race(day) for day in range(1, 11)],
        [_race(11)],
        policies=({
            "max_model_rank": 5,
            "ev_threshold": 1.0,
            "max_estimated_ev": 2.0,
            "max_odds": 100.0,
            "max_tickets_per_race": 5,
            "stake_per_ticket_yen": 100,
            "staking_mode": "flat",
        },),
        daily_budget_yen=10_000,
    )
    assert result["metrics"]["evaluated_races"] == 1
    assert len(result["purchase_diagnostics"]) == 1
    assert "bootstrap" in result["purchase_diagnostics"][0]


def test_v27_rejects_non_preregistered_feature_assignment() -> None:
    with pytest.raises(ValueError, match="preregistered"):
        fit_temporal_pruned_residual(
            [_race(day) for day in range(1, 5)],
            [_race(5)],
            variants={"ability_raw": ("motor_2_rate",)},
            regularizations=(0.1,),
        )
