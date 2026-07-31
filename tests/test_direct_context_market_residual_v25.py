from __future__ import annotations

import math

import numpy as np
import pytest

from boatrace_ai.listwise.direct_context_market_residual_v25 import (
    BASE_FEATURE_DIMENSION,
    CONTEXT_COLUMNS,
    CONTEXT_FEATURES,
    FEATURE_DIMENSION,
    _objective_gradient,
    direct_context_metrics,
    direct_context_probabilities,
    extract_lane_context,
    fit_direct_context_residual,
    prepare_context_race,
)


def _lane_context() -> dict[str, dict[str, float]]:
    return {
        str(lane): {
            "national_win_rate": 8.0 if lane == 1 else 4.0,
            "motor_2_rate": float(50 - lane),
            "hist_racer_win_rate_s": 0.4 if lane == 1 else 0.1,
        }
        for lane in range(1, 7)
    }


def _race(day: int, actual: str = "1-2-3") -> dict:
    return {
        "race_id": f"2026-01-{day:02d}-01-01",
        "race_date": f"2026-01-{day:02d}",
        "jcd": "01",
        "actual_combination": actual,
        "model_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "market_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "lane_context": _lane_context(),
    }


def test_zero_direct_context_residual_is_exact_market() -> None:
    probabilities = direct_context_probabilities(
        _race(1),
        {"coefficients": [0.0] * FEATURE_DIMENSION},
    )
    assert probabilities == pytest.approx({"1-2-3": 0.5, "2-1-3": 0.5})


def test_direct_context_layout_is_stage_specific() -> None:
    prepared = prepare_context_race(_race(1))
    assert prepared.lane_context.shape == (6, CONTEXT_COLUMNS)
    assert prepared.stage_lanes.shape == (2, 3)
    assert FEATURE_DIMENSION == BASE_FEATURE_DIMENSION + 3 * CONTEXT_COLUMNS
    assert prepared.lane_context[0, CONTEXT_FEATURES.index("national_win_rate")] > 0


def test_extract_lane_context_keeps_only_finite_selected_values() -> None:
    rows = []
    for lane in range(1, 7):
        rows.append({
            "meta": {"lane": lane},
            "features": {
                "national_win_rate": 5.0 + lane,
                "motor_2_rate": float("nan"),
                "not_selected": 99.0,
            },
        })
    result = extract_lane_context(rows)
    assert result["1"] == {"national_win_rate": 6.0}
    assert "not_selected" not in result["1"]


def test_direct_context_gradient_matches_finite_difference() -> None:
    prepared = [prepare_context_race(_race(1))]
    coefficients = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    loss, gradient = _objective_gradient(
        coefficients,
        prepared,
        regularization=0.1,
    )
    assert math.isfinite(loss)
    index = BASE_FEATURE_DIMENSION + CONTEXT_FEATURES.index("national_win_rate")
    epsilon = 1e-6
    forward = coefficients.copy()
    backward = coefficients.copy()
    forward[index] += epsilon
    backward[index] -= epsilon
    forward_loss, _ = _objective_gradient(forward, prepared, regularization=0.1)
    backward_loss, _ = _objective_gradient(backward, prepared, regularization=0.1)
    numerical = (forward_loss - backward_loss) / (2.0 * epsilon)
    assert gradient[index] == pytest.approx(numerical, abs=1e-6)


def test_direct_context_learns_market_residual_from_card_strength() -> None:
    training = [_race(day) for day in range(1, 13)]
    artifact = fit_direct_context_residual(
        training,
        regularization=0.01,
        max_iterations=40,
    )
    metrics = direct_context_metrics([_race(13)], artifact)
    assert artifact["feature_dimension"] == FEATURE_DIMENSION
    assert artifact["gradient_norm"] < 1e-4
    assert metrics["trifecta_log_loss"] < math.log(2.0)
    probabilities = direct_context_probabilities(_race(13), artifact)
    assert probabilities["1-2-3"] > probabilities["2-1-3"]
