from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from boatrace_ai.listwise.market_offset_calibration import (
    fit_market_offset_calibration,
)


COMBINATIONS = tuple(
    "-".join(map(str, lanes))
    for lanes in itertools.permutations(range(1, 7), 3)
)


def _normalized(weights: list[float]) -> dict[str, float]:
    total = math.fsum(weights)
    return {
        key: value / total for key, value in zip(COMBINATIONS, weights)
    }


def _vectors(seed: int = 0):
    market_weights = [
        1.0 / (2.0 + ((index * 17 + seed) % 80))
        for index in range(120)
    ]
    model_weights = [
        market_weights[index]
        * math.exp((((index * 13 + seed) % 19) - 9) / 6.0)
        for index in range(120)
    ]
    market = _normalized(market_weights)
    model = _normalized(model_weights)
    odds = {
        key: min(1e6, 0.75 / market[key])
        for key in COMBINATIONS
    }
    return model, market, odds


def _race(day: str, seed: int, *, actual: str | None = None):
    model, market, odds = _vectors(seed)
    return {
        "race_id": f"{day}-{seed:03d}",
        "race_date": day,
        "model_probabilities": model,
        "market_probabilities": market,
        "forecast_odds": odds,
        "actual_combination": actual or max(model, key=model.get),
        "payout_yen": 999_999,
        "profit_yen": 888_888,
    }


def _fit(races, *, day="2026-07-04"):
    return fit_market_offset_calibration(
        races,
        prediction_date=day,
        regularization=0.3,
        min_training_races=2,
    )


def test_fitted_probabilities_are_finite_sum_to_one_and_audited() -> None:
    races = [
        _race("2026-07-01", 1),
        _race("2026-07-02", 2),
        _race("2026-07-03", 3),
    ]
    artifact = _fit(races)
    model, market, odds = _vectors(11)
    prediction = artifact.predict(
        model, market, odds, prediction_date="2026-07-04"
    )

    assert artifact.fitted
    assert artifact.trained_through_date == "2026-07-03"
    assert artifact.training_dates == (
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
    )
    assert artifact.training_races == 3
    assert artifact.as_dict()["teacher"] == "one_hot_actual_combination"
    assert artifact.as_dict()["uses_profit_teacher"] is False
    assert prediction.mode == "market_offset"
    assert len(prediction.probabilities) == 120
    assert all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in prediction.probabilities.values()
    )
    assert math.fsum(prediction.probabilities.values()) == pytest.approx(
        1.0, abs=1e-15
    )


def test_fit_and_predict_are_input_order_invariant() -> None:
    races = [
        _race("2026-07-01", 1),
        _race("2026-07-02", 2),
        _race("2026-07-03", 3),
    ]
    forward = _fit(races)
    reverse = _fit(list(reversed(races)))
    assert reverse.coefficients == pytest.approx(forward.coefficients, abs=0.0)
    assert reverse.feature_mean == pytest.approx(forward.feature_mean, abs=0.0)
    assert reverse.feature_scale == pytest.approx(forward.feature_scale, abs=0.0)

    model, market, odds = _vectors(7)
    reversed_model = dict(reversed(list(model.items())))
    reversed_market = dict(reversed(list(market.items())))
    reversed_odds = dict(reversed(list(odds.items())))
    left = forward.predict(
        model, market, odds, prediction_date="2026-07-04"
    )
    right = forward.predict(
        reversed_model,
        reversed_market,
        reversed_odds,
        prediction_date="2026-07-04",
    )
    assert right.probabilities == left.probabilities


def test_same_day_and_future_teachers_are_excluded_before_feature_access() -> None:
    prior = [_race("2026-07-01", 1), _race("2026-07-02", 2)]
    malformed_non_past = [
        {
            "race_date": "2026-07-03",
            "actual_combination": object(),
            "profit_yen": float("nan"),
        },
        {
            "race_date": "2099-01-01",
            "actual_combination": object(),
            "profit_yen": float("inf"),
        },
    ]
    baseline = fit_market_offset_calibration(
        prior,
        prediction_date="2026-07-03",
        min_training_races=2,
    )
    audited = fit_market_offset_calibration(
        prior + malformed_non_past,
        prediction_date="2026-07-03",
        min_training_races=2,
    )

    assert audited.coefficients == pytest.approx(baseline.coefficients, abs=0.0)
    assert audited.objective == pytest.approx(baseline.objective, abs=0.0)
    assert audited.excluded_non_past_races == 2
    assert audited.trained_through_date == "2026-07-02"
    assert all(day < audited.prediction_date for day in audited.training_dates)


def test_insufficient_prior_data_falls_back_exactly_to_normalized_market() -> None:
    artifact = fit_market_offset_calibration(
        [{"race_date": "2026-07-03"}],
        prediction_date="2026-07-03",
        min_training_races=2,
    )
    model, market, odds = _vectors(4)
    unnormalized_market = {
        key: value * 123.0 for key, value in market.items()
    }
    prediction = artifact.predict(
        model,
        unnormalized_market,
        odds,
        prediction_date="2026-07-03",
    )

    assert not artifact.fitted
    assert artifact.fallback_reason == "insufficient_strictly_prior_races"
    assert artifact.coefficients == (0.0, 0.0, 0.0)
    assert prediction.mode == "market_only"
    assert prediction.probabilities == pytest.approx(market, abs=1e-15)


def test_extreme_finite_values_remain_stable() -> None:
    artifact = _fit([
        _race("2026-07-01", 1),
        _race("2026-07-02", 2),
    ], day="2026-07-03")
    model = {
        key: (1e308 if index == 0 else 1e-300)
        for index, key in enumerate(COMBINATIONS)
    }
    market = {
        key: (1e308 if index == 119 else 1e-300)
        for index, key in enumerate(COMBINATIONS)
    }
    odds = {
        key: (1e308 if index % 2 else 1e-300)
        for index, key in enumerate(COMBINATIONS)
    }
    prediction = artifact.predict(
        model, market, odds, prediction_date="2026-07-03"
    )

    values = np.asarray(list(prediction.probabilities.values()))
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert math.fsum(values) == pytest.approx(1.0, abs=1e-15)


def test_profit_and_payout_fields_are_not_training_teachers() -> None:
    races = [_race("2026-07-01", 1), _race("2026-07-02", 2)]
    changed = []
    for race in races:
        copy = dict(race)
        copy["profit_yen"] = -10**18
        copy["payout_yen"] = float("nan")
        copy["return_multiplier"] = float("inf")
        changed.append(copy)
    baseline = _fit(races, day="2026-07-03")
    profit_mutated = _fit(changed, day="2026-07-03")

    assert profit_mutated.coefficients == pytest.approx(
        baseline.coefficients, abs=0.0
    )
    assert profit_mutated.objective == pytest.approx(
        baseline.objective, abs=0.0
    )


def test_prediction_refuses_date_before_artifact_boundary() -> None:
    artifact = _fit([
        _race("2026-07-01", 1),
        _race("2026-07-02", 2),
    ])
    model, market, odds = _vectors(8)
    with pytest.raises(ValueError, match="precedes artifact boundary"):
        artifact.predict(
            model, market, odds, prediction_date="2026-07-03"
        )
