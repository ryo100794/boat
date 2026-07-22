from itertools import permutations

import pytest

from boatrace_ai.listwise.closing_odds_momentum import (
    attach_momentum_closing_odds,
    fit_momentum_closing_odds_model,
    momentum_closing_odds_metrics,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race(*, rising: bool) -> dict:
    odds = {combination: 20.0 + index for index, combination in enumerate(COMBINATIONS)}
    earlier = {combination: 1.0 / len(COMBINATIONS) for combination in COMBINATIONS}
    current = dict(earlier)
    closing = dict(odds)
    for index, combination in enumerate(COMBINATIONS):
        direction = 1.0 if (index % 2 == 0) == rising else -1.0
        current[combination] *= 1.0 + 0.10 * direction
        closing[combination] = odds[combination] * (1.0 - 0.08 * direction)
    total = sum(current.values())
    current = {key: value / total for key, value in current.items()}
    return {
        "odds": odds,
        "closing_odds": closing,
        "market_probabilities": current,
        "earlier_market_probabilities": earlier,
        "momentum_scale": 1.0,
    }


def test_momentum_price_model_uses_only_preclosing_trend() -> None:
    training = [_race(rising=True), _race(rising=False)]
    model = fit_momentum_closing_odds_model(training, regularization=0.001)
    holdout = [_race(rising=True)]
    metrics = momentum_closing_odds_metrics(holdout, model)
    augmented = attach_momentum_closing_odds(holdout, model)

    assert model["training_races"] == 2
    assert model["momentum_coefficient"] < 0.0
    assert metrics["forecast_mean_absolute_log_error"] < metrics[
        "baseline_mean_absolute_log_error"
    ]
    assert len(augmented[0]["estimated_final_odds"]) == 120
    assert all(value >= 1.0 for value in augmented[0]["estimated_final_odds"].values())


def test_momentum_price_model_rejects_incomplete_snapshots() -> None:
    with pytest.raises(ValueError, match="complete snapshots"):
        fit_momentum_closing_odds_model(
            [
                {
                    "odds": {"1-2-3": 10.0},
                    "closing_odds": {"1-2-3": 9.0},
                    "market_probabilities": {"1-2-3": 1.0},
                    "earlier_market_probabilities": {"1-2-3": 1.0},
                }
            ]
        )
