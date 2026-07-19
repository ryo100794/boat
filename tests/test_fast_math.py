import math

import pytest

from boatrace_ai.fast_math import (
    TRIFECTA_COMBINATIONS,
    plackett_luce_probabilities,
)
from boatrace_ai.modeling import _pl_probability


def test_fast_math_matches_reference_plackett_luce() -> None:
    values = (0.48, 0.19, 0.12, 0.09, 0.07, 0.05)
    expected = tuple(
        _pl_probability(
            {lane: values[lane - 1] for lane in range(1, 7)},
            first,
            second,
            third,
        )
        for first, second, third in TRIFECTA_COMBINATIONS
    )

    actual = plackett_luce_probabilities(values)

    assert len(actual) == 120
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert math.isclose(sum(actual), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_fast_math_rejects_incomplete_lane_vector() -> None:
    with pytest.raises(ValueError, match="six lane"):
        plackett_luce_probabilities((0.5, 0.5))
