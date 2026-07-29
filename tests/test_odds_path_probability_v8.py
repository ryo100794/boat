from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

import boatrace_ai.listwise.odds_path_probability_v8 as v8


COMBINATIONS = tuple(
    "-".join(map(str, values))
    for values in permutations(range(1, 7), 3)
)


def _distribution(primary: str, probability: float) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(COMBINATIONS) - 1)
    return {
        combination: probability if combination == primary else remainder
        for combination in COMBINATIONS
    }


def _exact_market(primary: str) -> dict[str, float]:
    # Binary fractions make the 120 entries sum to exactly one.
    return {
        combination: 9.0 / 128.0 if combination == primary else 1.0 / 128.0
        for combination in COMBINATIONS
    }


def _race(
    race_date: str,
    rno: int,
    *,
    actual: str | None = None,
    market_primary: str | None = None,
) -> dict:
    primary = market_primary or COMBINATIONS[(rno * 7) % len(COMBINATIONS)]
    base_primary = COMBINATIONS[(rno * 11 + 3) % len(COMBINATIONS)]
    winner = actual or COMBINATIONS[(rno * 5 + 1) % len(COMBINATIONS)]
    market = _distribution(primary, 0.12)
    base = _distribution(base_primary, 0.15)
    earlier = dict(market)
    earlier[primary] *= 0.92
    earlier_total = sum(earlier.values())
    earlier = {key: value / earlier_total for key, value in earlier.items()}
    return {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "actual_combination": winner,
        "actual_payout_yen": 10_000,
        "model_probabilities": base,
        "market_probabilities": market,
        "odds_path": [
            {
                "minutes_before_decision": 10.0,
                "market_probabilities": earlier,
            },
            {
                "minutes_before_decision": 0.0,
                "market_probabilities": market,
            },
        ],
    }


def _races(days: int = 6, races_per_day: int = 3) -> list[dict]:
    return [
        _race(f"2026-07-{20 + day:02d}", rno)
        for day in range(days)
        for rno in range(1, races_per_day + 1)
    ]


def _zero_model() -> dict:
    return {
        "feature_names": list(v8.FEATURE_NAMES),
        "feature_mean": [0.0] * len(v8.FEATURE_NAMES),
        "feature_scale": [1.0] * len(v8.FEATURE_NAMES),
        "coefficients": [0.0] * len(v8.FEATURE_NAMES),
        "fixed_market_log_coefficient": 1.0,
    }


def test_zero_residual_reproduces_market_distribution_exactly() -> None:
    race = _race("2026-07-30", 1)
    race["market_probabilities"] = _exact_market("1-2-3")

    attached = v8.attach_odds_path_probability_v8([race], _zero_model())[0]

    assert attached["model_probabilities"] == race["market_probabilities"]
    assert sum(attached["model_probabilities"].values()) == 1.0


def test_fit_attaches_complete_normalized_distribution_and_metadata() -> None:
    races = _races()
    model = v8.fit_odds_path_probability_v8(races)
    attached = v8.attach_odds_path_probability_v8(
        [_race("2026-07-30", 9)], model
    )[0]

    assert len(attached["model_probabilities"]) == 120
    assert sum(attached["model_probabilities"].values()) == pytest.approx(1.0)
    assert model["fixed_market_log_coefficient"] == 1.0
    assert model["training_days"] == 6
    assert model["trained_through_date"] == "2026-07-25"
    assert len(model["feature_mean"]) == len(v8.FEATURE_NAMES)
    assert len(model["feature_scale"]) == len(v8.FEATURE_NAMES)
    assert model["selection_basis"]["status"] == "selected"


def test_nested_scalers_and_models_use_only_strictly_prior_dates() -> None:
    races = _races(days=5, races_per_day=2)
    selection = v8.select_regularization_nested(
        races, regularizations=(0.1, 1.0)
    )
    first_candidate = selection["candidates"][0]
    first_fold = first_candidate["folds"][0]
    training = [
        race
        for race in races
        if race["race_date"] < first_fold["validation_date"]
    ]
    _offsets, raw_features, _actual = v8._prepare_training_tensors(training)
    expected_mean, expected_scale = v8._fit_feature_scaler(raw_features)

    assert all(
        fold["trained_through_date"] < fold["validation_date"]
        for candidate in selection["candidates"]
        for fold in candidate["folds"]
    )
    assert first_fold["feature_mean"] == pytest.approx(expected_mean)
    assert first_fold["feature_scale"] == pytest.approx(expected_scale)


def test_holdout_outcome_and_payout_do_not_affect_attached_probability() -> None:
    model = v8.fit_odds_path_probability_v8(_races())
    first = _race("2026-07-30", 8, actual="1-2-3")
    second = dict(first)
    second["actual_combination"] = "6-5-4"
    second["actual_payout_yen"] = 999_900

    first_probability = v8.attach_odds_path_probability_v8([first], model)[0]
    second_probability = v8.attach_odds_path_probability_v8([second], model)[0]

    assert model["trained_through_date"] < first["race_date"]
    assert (
        first_probability["model_probabilities"]
        == second_probability["model_probabilities"]
    )


def test_probability_gradient_matches_finite_difference() -> None:
    offsets, raw_features, actual = v8._prepare_training_tensors(_races(2, 2))
    mean, scale = v8._fit_feature_scaler(raw_features)
    features = (raw_features - mean) / scale
    coefficients = np.asarray([0.03, -0.02, 0.01, 0.04, -0.01, 0.02])
    regularization = 0.1
    _objective, gradient, _hessian = v8._objective_gradient_hessian(
        offsets,
        features,
        actual,
        coefficients,
        regularization=regularization,
    )
    epsilon = 1e-6
    numerical = np.zeros_like(coefficients)
    for index in range(len(coefficients)):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_objective, _gradient, _hessian = v8._objective_gradient_hessian(
            offsets,
            features,
            actual,
            plus,
            regularization=regularization,
        )
        minus_objective, _gradient, _hessian = v8._objective_gradient_hessian(
            offsets,
            features,
            actual,
            minus,
            regularization=regularization,
        )
        numerical[index] = (plus_objective - minus_objective) / (2 * epsilon)

    assert gradient == pytest.approx(numerical, abs=1e-6)


def test_nested_regularization_selection_is_deterministic() -> None:
    races = _races()

    first = v8.fit_odds_path_probability_v8(
        races, regularizations=(0.01, 0.1, 1.0)
    )
    second = v8.fit_odds_path_probability_v8(
        list(reversed(races)), regularizations=(1.0, 0.1, 0.01)
    )

    assert first["regularization_selection"] == second["regularization_selection"]
    assert first["coefficients"] == pytest.approx(second["coefficients"])


def test_short_history_uses_strongest_regularization_fallback() -> None:
    races = _races(days=3, races_per_day=2)

    model = v8.fit_odds_path_probability_v8(
        races, regularizations=(0.01, 0.1, 2.0)
    )
    selection = model["regularization_selection"]

    assert selection["status"] == "conservative_fallback"
    assert selection["reason"] == "insufficient_nested_validation_days"
    assert selection["selected_regularization"] == 2.0
    assert model["regularization"] == 2.0
