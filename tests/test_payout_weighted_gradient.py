from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from boatrace_ai.listwise.payout_weighted_ranking import (
    _weighted_objective_gradient,
)
from boatrace_ai.listwise.pruned_direct_context_v27 import (
    FEATURE_VARIANTS,
    _feature_dimension,
    _prepare_race,
)


def _race() -> dict:
    return {
        "race_id": "2026-01-01-01-01",
        "race_date": date(2026, 1, 1).isoformat(),
        "jcd": "01",
        "actual_combination": "1-2-3",
        "actual_payout_yen": 1800,
        "model_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "market_probabilities": {"1-2-3": 0.5, "2-1-3": 0.5},
        "lane_context": {
            str(lane): {
                "class_rank": float(lane),
                "national_win_rate": float(8 - lane),
                "motor_2_rate": float(40 + lane),
            }
            for lane in range(1, 7)
        },
    }


@pytest.mark.parametrize("index_kind", ["base", "context"])
def test_payout_weighted_gradient_matches_finite_difference(
    index_kind: str,
) -> None:
    active = FEATURE_VARIANTS["independent_core"]
    prepared = [_prepare_race(_race(), active)]
    coefficients = np.zeros(_feature_dimension(active), dtype=np.float64)
    weights = np.asarray([1.7], dtype=np.float64)
    _, gradient = _weighted_objective_gradient(
        coefficients, prepared, weights, regularization=0.03
    )
    index = 0 if index_kind == "base" else len(coefficients) - 1
    epsilon = 1e-6
    forward = coefficients.copy()
    backward = coefficients.copy()
    forward[index] += epsilon
    backward[index] -= epsilon
    forward_loss, _ = _weighted_objective_gradient(
        forward, prepared, weights, regularization=0.03
    )
    backward_loss, _ = _weighted_objective_gradient(
        backward, prepared, weights, regularization=0.03
    )
    numerical = (forward_loss - backward_loss) / (2.0 * epsilon)
    assert gradient[index] == pytest.approx(numerical, abs=1e-6)
