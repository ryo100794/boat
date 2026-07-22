from itertools import permutations

import pytest

from boatrace_ai.listwise.market_momentum import (
    fit_momentum_newton,
    momentum_log_pool_probabilities,
    momentum_probability_metrics,
    select_momentum_regularization_prequential,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _uniform() -> dict[str, float]:
    return {combination: 1.0 / len(COMBINATIONS) for combination in COMBINATIONS}


def _race(race_date: str, index: int) -> dict:
    actual = COMBINATIONS[index % len(COMBINATIONS)]
    earlier = {combination: 0.8 / (len(COMBINATIONS) - 1) for combination in COMBINATIONS}
    earlier[actual] = 0.2
    return {
        "race_id": f"{race_date}-{index}",
        "race_date": race_date,
        "actual_combination": actual,
        "model_probabilities": _uniform(),
        "market_probabilities": _uniform(),
        "earlier_market_probabilities": earlier,
    }


def test_momentum_log_pool_has_market_baseline_endpoint() -> None:
    market = _uniform()
    probabilities = momentum_log_pool_probabilities(
        market,
        market,
        market,
        model_coefficient=0.0,
        market_coefficient=1.0,
        momentum_coefficient=0.0,
    )

    assert probabilities == pytest.approx(market)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_momentum_newton_learns_direction_without_holdout_access() -> None:
    training = [
        _race(race_date, index)
        for race_date in ("2026-07-20", "2026-07-21")
        for index in range(12)
    ]
    holdout = [_race("2026-07-22", index) for index in range(12)]

    calibrator = fit_momentum_newton(training, regularization=0.01)
    metrics = momentum_probability_metrics(holdout, calibrator)

    assert calibrator["converged"] is True
    assert calibrator["momentum_coefficient"] < 0
    assert metrics["trifecta_log_loss"] < 4.0
    assert metrics["trifecta_top5_hit_rate"] == 1.0


def test_momentum_regularization_uses_forward_daily_folds() -> None:
    races = [
        _race(race_date, index)
        for race_date in ("2026-07-20", "2026-07-21")
        for index in range(8)
    ]

    selection = select_momentum_regularization_prequential(
        races,
        regularizations=(0.01, 1.0),
    )

    assert selection["dates"] == ["2026-07-20", "2026-07-21"]
    assert len(selection["candidates"]) == 2
    assert all(
        candidate["folds"][0]["training_dates"] == ["2026-07-20"]
        for candidate in selection["candidates"]
    )
    assert all(
        candidate["folds"][0]["evaluation_date"] == "2026-07-21"
        for candidate in selection["candidates"]
    )
