from itertools import permutations

from boatrace_ai.listwise.closing_odds_contextual import (
    fit_contextual_closing_odds_model,
    forecast_contextual_closing_odds,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race() -> dict:
    odds = {
        combination: 20.0 + index
        for index, combination in enumerate(COMBINATIONS)
    }
    earlier = {
        combination: 1.0 / len(COMBINATIONS)
        for combination in COMBINATIONS
    }
    current = {
        combination: probability * (1.1 if index % 2 == 0 else 0.9)
        for index, (combination, probability) in enumerate(earlier.items())
    }
    total = sum(current.values())
    current = {key: value / total for key, value in current.items()}
    closing = {
        key: value * (0.92 if index % 2 == 0 else 1.08)
        for index, (key, value) in enumerate(odds.items())
    }
    return {
        "odds": odds,
        "closing_odds": closing,
        "market_probabilities": current,
        "earlier_market_probabilities": earlier,
        "momentum_scale": 1.0,
    }


def test_contextual_forecast_is_complete_and_improves_synthetic_trend() -> None:
    training = [_race(), _race()]
    model = fit_contextual_closing_odds_model(training)
    holdout = _race()
    forecast = forecast_contextual_closing_odds(holdout, model)
    baseline_error = sum(
        abs(__import__("math").log(holdout["closing_odds"][key])
            - __import__("math").log(holdout["odds"][key]))
        for key in COMBINATIONS
    )
    forecast_error = sum(
        abs(__import__("math").log(holdout["closing_odds"][key])
            - __import__("math").log(forecast[key]))
        for key in COMBINATIONS
    )

    assert model["training_races"] == 2
    assert len(forecast) == 120
    assert forecast_error < baseline_error
