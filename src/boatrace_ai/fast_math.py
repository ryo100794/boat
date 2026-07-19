from __future__ import annotations

import math
from itertools import permutations
from typing import Iterable


TRIFECTA_COMBINATIONS = tuple(permutations(range(1, 7), 3))

try:
    from ._fast_boat_math import plackett_luce as _native_plackett_luce
except ImportError:
    _native_plackett_luce = None


def plackett_luce_probabilities(probabilities: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in probabilities)
    if len(values) != 6:
        raise ValueError(f"six lane probabilities are required: {len(values)}")
    if _native_plackett_luce is not None:
        return tuple(_native_plackett_luce(values))
    return tuple(
        _pl_probability(values, first - 1, second - 1, third - 1)
        for first, second, third in TRIFECTA_COMBINATIONS
    )


def native_available() -> bool:
    return _native_plackett_luce is not None


def _pl_probability(
    probabilities: tuple[float, ...],
    first: int,
    second: int,
    third: int,
) -> float:
    p_first = probabilities[first]
    p_second_lane = probabilities[second]
    after_first = max(1e-9, 1.0 - p_first)
    after_second = max(1e-9, 1.0 - p_first - p_second_lane)
    value = p_first * (p_second_lane / after_first) * (
        probabilities[third] / after_second
    )
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
