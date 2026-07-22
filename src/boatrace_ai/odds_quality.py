from __future__ import annotations

import math
from typing import Mapping

from .fast_math import TRIFECTA_COMBINATIONS


TRIFECTA_PARSER_VERSION = "odds3t_dom_v2"
TRIFECTA_COMBINATION_KEYS = tuple(
    "-".join(map(str, combination)) for combination in TRIFECTA_COMBINATIONS
)
MAX_LANE_MARKER_ODDS = 8


def plausible_trifecta_odds(odds: Mapping[str, float]) -> bool:
    if set(odds) != set(TRIFECTA_COMBINATION_KEYS):
        return False
    try:
        values = [float(odds[key]) for key in TRIFECTA_COMBINATION_KEYS]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) and value >= 1.0 for value in values)
        and sum(value in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0} for value in values)
        <= MAX_LANE_MARKER_ODDS
    )
