import math
from itertools import permutations
from pathlib import Path

import pytest

from scripts.analyze_market_curvature import (
    attach_disagreement_curvature,
    standardized_payload,
)
from scripts.analyze_market_momentum import evaluate_momentum_candidate


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _distribution(primary: str, probability: float) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(COMBINATIONS) - 1)
    return {
        combination: probability if combination == primary else remainder
        for combination in COMBINATIONS
    }


def _race(race_date: str, index: int) -> dict:
    actual = COMBINATIONS[index % len(COMBINATIONS)]
    market = _distribution(actual, 0.12)
    model = _distribution(actual, 0.28)
    return {
        "race_id": f"{race_date}-{index}",
        "race_date": race_date,
        "actual_combination": actual,
        "model_probabilities": model,
        "market_probabilities": market,
    }


def test_curvature_encoding_preserves_feature_up_to_softmax_constant() -> None:
    race = _race("2026-07-20", 0)
    transformed = attach_disagreement_curvature(
        [race],
        disagreement_clip=4.0,
    )[0]
    first, second = COMBINATIONS[:2]

    def encoded(combination: str) -> float:
        return math.log(race["market_probabilities"][combination]) - math.log(
            transformed["earlier_market_probabilities"][combination]
        )

    def expected(combination: str) -> float:
        disagreement = math.log(race["model_probabilities"][combination]) - math.log(
            race["market_probabilities"][combination]
        )
        clipped = min(4.0, max(-4.0, disagreement))
        return clipped * abs(clipped)

    assert encoded(first) - encoded(second) == pytest.approx(
        expected(first) - expected(second)
    )


def test_curvature_probe_uses_prior_days_and_never_promotes_directly() -> None:
    races = [
        _race(race_date, index)
        for race_date in ("2026-07-20", "2026-07-21", "2026-07-22")
        for index in range(8)
    ]
    transformed = attach_disagreement_curvature(
        races,
        disagreement_clip=4.0,
    )

    evaluation = evaluate_momentum_candidate(
        transformed,
        evaluation_date="2026-07-22",
    )
    payload = standardized_payload(
        evaluation,
        source_cache=Path("scores.joblib"),
        disagreement_clip=4.0,
    )

    assert evaluation["calibration_dates"] == ["2026-07-20", "2026-07-21"]
    assert evaluation["evaluation_date"] == "2026-07-22"
    assert payload["promotion_eligible"] is False
    assert payload["status"] in {
        "candidate_requires_new_day_confirmation",
        "rejected_no_incremental_value",
    }
