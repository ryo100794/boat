import math

from boatrace_ai.listwise.closing_odds import (
    attach_forecast_closing_odds,
    closing_odds_metrics,
    decision_odds,
    fit_closing_odds_model,
    forecast_closing_odds,
)


def _race(scale: float) -> dict:
    odds = {f"c{index}": float(index + 2) for index in range(120)}
    return {
        "race_id": f"r{scale}",
        "odds": odds,
        "closing_odds": {
            combination: math.exp(-0.2 + 1.05 * math.log(value)) * scale
            for combination, value in odds.items()
        },
    }


def test_closing_odds_model_reduces_training_log_error() -> None:
    model = fit_closing_odds_model([_race(1.0), _race(1.01)])

    assert model["training_races"] == 2
    assert model["training_tickets"] == 240
    assert model["training_mean_absolute_log_error"] < model[
        "baseline_mean_absolute_log_error"
    ]
    assert model["log_odds_coefficient"] > 1.0

    metrics = closing_odds_metrics([_race(1.0)], model)
    assert metrics["evaluation_tickets"] == 120
    assert metrics["forecast_mean_absolute_log_error"] < metrics[
        "baseline_mean_absolute_log_error"
    ]


def test_forecast_is_attached_without_replacing_observed_t5_odds() -> None:
    race = _race(1.0)
    model = fit_closing_odds_model([race])

    attached = attach_forecast_closing_odds([race], model)[0]

    assert attached["odds"] is race["odds"]
    assert decision_odds(attached) is attached["estimated_final_odds"]
    assert decision_odds(race) is race["odds"]
    assert set(forecast_closing_odds(race["odds"], model)) == set(race["odds"])


def test_incomplete_closing_snapshot_is_not_used_for_fitting() -> None:
    race = _race(1.0)
    race["closing_odds"].pop("c0")

    try:
        fit_closing_odds_model([race])
    except ValueError as exc:
        assert "complete paired snapshots" in str(exc)
    else:
        raise AssertionError("incomplete closing snapshot must be rejected")



def test_expected_value_forecast_uses_conservative_race_cluster_correction() -> None:
    races = []
    for race_index in range(4):
        race = _race(1.0)
        race["race_id"] = f"cluster-{race_index}"
        race["closing_odds"] = {
            combination: value * (0.8 if index % 2 == 0 else 1.2)
            for index, (combination, value) in enumerate(race["closing_odds"].items())
        }
        races.append(race)

    model = fit_closing_odds_model(races)
    median = forecast_closing_odds(races[0]["odds"], model)
    expected = forecast_closing_odds(
        races[0]["odds"], model, expected_value=True
    )
    attached = attach_forecast_closing_odds([races[0]], model)[0]

    assert model["expected_odds_race_clusters"] == 4
    assert model["expected_odds_multiplier"] > 1.0
    assert expected["c0"] > median["c0"]
    assert attached["closing_odds_forecast_target"] == "conservative_expected_odds"
    assert decision_odds(attached)["c0"] == expected["c0"]
