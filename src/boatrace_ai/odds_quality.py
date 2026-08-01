from __future__ import annotations

import math
from itertools import combinations, permutations
from typing import Any, Mapping

from .fast_math import TRIFECTA_COMBINATIONS


TRIFECTA_PARSER_VERSION = "odds3t_dom_v2"
TRIFECTA_COMBINATION_KEYS = tuple(
    "-".join(map(str, combination)) for combination in TRIFECTA_COMBINATIONS
)
MAX_LANE_MARKER_ODDS = 8
TRIFECTA_LANES = tuple(range(1, 7))
MIN_ACTIVE_TRIFECTA_LANES = 4


def describe_trifecta_market(
    odds: Mapping[str, float | None],
    *,
    allow_zero: bool = False,
) -> dict[str, Any] | None:
    """Validate a complete official market, including coherent absent-lane markets."""
    if set(odds) != set(TRIFECTA_COMBINATION_KEYS):
        return None

    available: dict[str, float] = {}
    try:
        for key in TRIFECTA_COMBINATION_KEYS:
            raw_value = odds[key]
            if raw_value is None:
                continue
            value = float(raw_value)
            minimum = 0.0 if allow_zero else 1.0
            if not math.isfinite(value) or value < minimum:
                return None
            available[key] = value
    except (KeyError, TypeError, ValueError):
        return None
    if not available or (
        allow_zero and not any(value > 0.0 for value in available.values())
    ):
        return None
    if (
        sum(
            value in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
            for value in available.values()
        )
        > MAX_LANE_MARKER_ODDS
    ):
        return None

    available_keys = set(available)
    for active_count in range(
        len(TRIFECTA_LANES), MIN_ACTIVE_TRIFECTA_LANES - 1, -1
    ):
        for active_lanes in combinations(TRIFECTA_LANES, active_count):
            expected = {
                "-".join(map(str, combination))
                for combination in permutations(active_lanes, 3)
            }
            if available_keys != expected:
                continue
            absent_lanes = sorted(set(TRIFECTA_LANES) - set(active_lanes))
            return {
                "active_lanes": list(active_lanes),
                "absent_lanes": absent_lanes,
                "active_combination_count": len(expected),
                "total_combination_count": len(TRIFECTA_COMBINATION_KEYS),
                "special_market": bool(absent_lanes),
                "model_supported": not absent_lanes,
            }
    return None


def plausible_trifecta_odds(odds: Mapping[str, float]) -> bool:
    description = describe_trifecta_market(odds)
    return bool(description and not description["special_market"])


def plausible_trifecta_capture(odds: Mapping[str, float]) -> bool:
    """Accept official in-sale snapshots while keeping zero odds out of inference."""
    description = describe_trifecta_market(odds, allow_zero=True)
    return bool(description and not description["special_market"])
