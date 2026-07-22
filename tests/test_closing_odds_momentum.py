from itertools import permutations

import pytest

from boatrace_ai.listwise.closing_odds_momentum import (
    attach_momentum_closing_odds,
    fit_momentum_closing_odds_model,
    momentum_closing_odds_metrics,
    select_closing_odds_model,
    attach_selected_closing_odds,
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



def test_price_model_selection_requires_prior_day_improvement() -> None:
    races = []
    for race_date in ("2026-07-21", "2026-07-22"):
        for rising in (True, False):
            race = _race(rising=rising)
            race["race_date"] = race_date
            races.append(race)

    selection = select_closing_odds_model(
        races, minimum_relative_improvement=0.01
    )

    assert selection["selected"] == "momentum"
    assert selection["prequential_evaluation_races"] == 2
    assert selection["prequential_momentum_mae"] < selection[
        "prequential_baseline_mae"
    ]


def test_price_model_selection_falls_back_without_two_momentum_days() -> None:
    race = _race(rising=True)
    race["race_date"] = "2026-07-21"
    selection = select_closing_odds_model([race])
    missing = dict(race)
    missing.pop("earlier_market_probabilities")
    augmented = attach_selected_closing_odds([missing], selection)

    assert selection["selected"] == "baseline"
    assert selection["prequential_evaluation_races"] == 0
    assert augmented[0]["closing_odds_forecast_source"] == "baseline"



def test_price_model_selection_excludes_incomplete_closing_snapshot() -> None:
    baseline_race = _race(rising=True)
    baseline_race["race_date"] = "2026-07-21"
    baseline_race.pop("earlier_market_probabilities")
    incomplete = _race(rising=False)
    incomplete["race_date"] = "2026-07-21"
    incomplete["closing_odds"] = {}

    selection = select_closing_odds_model([baseline_race, incomplete])

    assert selection["selected"] == "baseline"
    assert selection["eligible_momentum_races"] == 0
