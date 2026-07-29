from __future__ import annotations

import itertools

from boatrace_ai.listwise.odds_path_operational import (
    attach_odds_path_model,
    fit_odds_path_model,
    fit_performance_priors,
)


COMBINATIONS = ["-".join(map(str, row)) for row in itertools.permutations(range(1, 7), 3)]


def _race(index: int, *, actual: str = "1-2-3") -> dict:
    model = {key: 1.0 / 120.0 for key in COMBINATIONS}
    current_odds = {key: 120.0 for key in COMBINATIONS}
    path = []
    for minutes, winner_probability in ((25.0, 0.02), (15.0, 0.03), (5.0, 0.05), (0.0, 0.08)):
        remainder = (1.0 - winner_probability) / 119.0
        probabilities = {key: remainder for key in COMBINATIONS}
        probabilities["1-2-3"] = winner_probability
        path.append(
            {
                "minutes_before_decision": minutes,
                "market_probabilities": probabilities,
            }
        )
    market = dict(path[-1]["market_probabilities"])
    current_odds["1-2-3"] = 12.5
    return {
        "race_id": f"race-{index}",
        "race_date": f"2026-07-{18 + index:02d}",
        "jcd": "01",
        "rno": index + 1,
        "actual_combination": actual,
        "actual_payout_yen": 1250 if actual == "1-2-3" else 12000,
        "model_probabilities": model,
        "market_probabilities": market,
        "odds": current_odds,
        "odds_path": path,
    }


def test_odds_path_model_builds_probability_and_return_parameters() -> None:
    races = [_race(index) for index in range(3)]
    model = fit_odds_path_model(races, max_iterations=20)
    attached = attach_odds_path_model([_race(9)], model)[0]

    assert model["model_type"] == "odds_path_probability_and_return_v1"
    assert model["training_races"] == 3
    assert len(model["feature_names"]) == len(model["weights"])
    assert abs(sum(attached["model_probabilities"].values()) - 1.0) < 1e-12
    assert len(attached["historical_return_multipliers"]) == 120
    assert all(
        0.25 <= value <= 2.0
        for value in attached["historical_return_multipliers"].values()
    )


def test_holdout_result_and_payout_do_not_change_inference() -> None:
    model = fit_odds_path_model([_race(0), _race(1)], max_iterations=10)
    first = _race(5, actual="1-2-3")
    second = _race(5, actual="6-5-4")

    first_prediction = attach_odds_path_model([first], model)[0]
    second_prediction = attach_odds_path_model([second], model)[0]

    assert first_prediction["model_probabilities"] == second_prediction["model_probabilities"]
    assert first_prediction["historical_return_multipliers"] == second_prediction["historical_return_multipliers"]


def test_probability_only_model_does_not_double_count_closing_price_drift() -> None:
    model = fit_odds_path_model(
        [_race(0), _race(1)],
        max_iterations=10,
        use_return_multipliers=False,
    )
    attached = attach_odds_path_model([_race(9)], model)[0]

    assert model["model_type"] == "odds_path_probability_only_v2"
    assert model["return_multiplier_mode"] == "disabled_for_forecast_closing_price"
    assert set(attached["historical_return_multipliers"].values()) == {1.0}


def test_closing_return_model_uses_separate_return_price_basis() -> None:
    races = [_race(0), _race(1)]
    for race in races:
        race["performance_return_odds"] = {
            combination: odds * 0.9 for combination, odds in race["odds"].items()
        }
    model = fit_odds_path_model(
        races,
        max_iterations=10,
        return_price_basis="forecast_closing",
    )
    attached = attach_odds_path_model([_race(9)], model)[0]

    assert model["model_type"] == "odds_path_closing_return_v3"
    assert model["return_price_basis"] == "forecast_closing"
    assert model["return_multiplier_mode"] == (
        "historical_forecast_closing_to_payout_bucket"
    )
    assert any(
        value != 1.0 for value in attached["historical_return_multipliers"].values()
    )


def test_performance_priors_shrink_sparse_hit_and_payout_rates() -> None:
    priors = fit_performance_priors([_race(0)])

    assert priors["training_races"] == 1
    assert priors["buckets"]
    assert all(0.0 < row["hit_rate"] < 1.0 for row in priors["buckets"].values())
    assert all(0.25 <= row["return_multiplier"] <= 2.0 for row in priors["buckets"].values())
