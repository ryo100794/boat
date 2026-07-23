from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.payout_estimator import (
    FEATURE_COUNT,
    ConditionalPayoutRegressor,
    ConditionalPayoutStatistics,
    fit_conditional_payout,
    fit_conditional_payout_statistics,
    payout_features,
    predict_conditional_odds,
)


def test_prediction_uses_lognormal_mean_correction() -> None:
    model = ConditionalPayoutRegressor(
        weights=np.zeros(FEATURE_COUNT),
        residual_variance=2.0,
        ridge=10.0,
        training_samples=100,
    )

    predicted = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
    )

    median = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
        lognormal_mean_correction=False,
    )

    halfway = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
        mean_correction_factor=0.5,
    )

    np.testing.assert_allclose(predicted, [np.e])
    np.testing.assert_allclose(halfway, [np.sqrt(np.e)])
    np.testing.assert_allclose(median, [1.1])


def test_payout_features_encode_probability_order_venue_and_race_number() -> None:
    matrix = payout_features(
        [0.2, 0.05],
        ["1-2-3", "6-5-4"],
        [
            ("r1", "2026-07-01", "01", 2),
            ("r2", "2026-07-01", "24", 11),
        ],
    )

    assert matrix.shape == (2, FEATURE_COUNT)
    assert matrix[0, 3] == 1.0
    assert matrix[1, 8] == 1.0
    assert matrix[0, 21] == 1.0
    assert matrix[1, 44] == 1.0
    assert matrix[0, 45] == 1.0
    assert matrix[1, 47] == 1.0
    assert matrix[1, 53] > matrix[0, 48]
    assert np.count_nonzero(matrix[0]) == 9
    assert np.count_nonzero(matrix[1]) == 9


def test_conditional_payout_regression_learns_probability_payout_relation() -> None:
    rng = np.random.default_rng(20260722)
    probabilities = np.linspace(0.01, 0.35, 500)
    combinations = ["1-2-3" if index % 2 else "6-5-4" for index in range(500)]
    race_keys = [
        (f"r{index}", "2026-07-01", f"{index % 24 + 1:02d}", index % 12 + 1)
        for index in range(500)
    ]
    true_odds = np.clip(0.78 / probabilities, 1.1, 2_000.0)
    payouts = true_odds * 100.0 * np.exp(rng.normal(0.0, 0.03, len(probabilities)))

    model = fit_conditional_payout(
        probabilities,
        combinations,
        race_keys,
        payouts,
    )
    statistics = ConditionalPayoutStatistics.empty()
    statistics.update(
        probabilities[:250],
        combinations[:250],
        race_keys[:250],
        payouts[:250],
    )
    statistics.update(
        probabilities[250:],
        combinations[250:],
        race_keys[250:],
        payouts[250:],
    )
    incremental_model = fit_conditional_payout_statistics(statistics)
    predicted = predict_conditional_odds(
        model,
        [0.02, 0.20],
        ["1-2-3", "1-2-3"],
        [
            ("a", "2026-07-02", "01", 1),
            ("b", "2026-07-02", "01", 1),
        ],
    )

    assert model.training_samples == 500
    np.testing.assert_allclose(incremental_model.weights, model.weights, atol=1e-12)
    assert np.isclose(
        incremental_model.residual_variance,
        model.residual_variance,
        atol=1e-12,
    )
    assert predicted[0] > predicted[1] * 5.0
    np.testing.assert_allclose(predicted, [39.0, 3.9], rtol=0.25)
