from __future__ import annotations

import numpy as np
import json
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

from boatrace_ai.listwise.conditional_order import conditional_probabilities, identity_model
from boatrace_ai.listwise.model import ListwiseLinearModel
from boatrace_ai.listwise.venue_conditional_order import (
    _load_legacy_bankroll_reference,
    _pack_model,
    objective_gradient,
    score_feature_rows_streaming,
    venue_conditional_probabilities,
    venue_identity_model,
    venue_indices,
)


def test_streaming_scores_match_direct_hashed_matrix() -> None:
    keys = [
        ("r1", "2026-07-01", "01", 1),
        ("r2", "2026-07-01", "01", 2),
    ]
    rows = []
    flat = []
    for race_index, key in enumerate(keys):
        race = []
        for lane in range(1, 7):
            features = {"lane": lane, "race_bias": race_index + 1}
            flat.append(features)
            race.append({
                "features": features,
                "meta": {
                    "race_id": key[0],
                    "lane": lane,
                    "rank": 7 - lane,
                },
            })
        rows.append(race)
    hasher = FeatureHasher(n_features=64, input_type="dict", alternate_sign=False)
    matrix = hasher.transform(flat)
    scaler = StandardScaler(with_mean=False).fit(matrix)
    model = ListwiseLinearModel(
        weights=np.linspace(-0.5, 0.5, 64),
        scaler=scaler,
        target="top3_pl",
        alpha=0.001,
        learning_rate=0.02,
        epochs=1,
    )

    scores, ranks = score_feature_rows_streaming(
        iter(rows), race_keys=keys, model=model, batch_races=1
    )
    expected = np.asarray(scaler.transform(matrix).dot(model.weights)).reshape(2, 6)

    np.testing.assert_allclose(scores, expected)
    np.testing.assert_array_equal(ranks, np.tile([6, 5, 4, 3, 2, 1], (2, 1)))


def test_legacy_bankroll_reference_requires_identical_daily_period(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "evaluation_from": "2025-07-20",
        "evaluation_through": "2025-07-21",
        "conditional_payout_walk_forward": {"bankroll": {
            "roi": 0.9,
            "profit_yen": -100,
            "stake_yen": 1000,
            "return_yen": 900,
            "daily": [
                {"race_date": "2025-07-20", "stake_yen": 500, "return_yen": 400},
                {"race_date": "2025-07-21", "stake_yen": 500, "return_yen": 500},
            ],
        }},
    }), encoding="utf-8")

    bankroll, daily = _load_legacy_bankroll_reference(
        path,
        evaluation_from="2025-07-20",
        evaluation_through="2025-07-21",
        expected_dates=["2025-07-20", "2025-07-21"],
    )

    assert bankroll["roi"] == 0.9
    assert len(daily) == 2


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260723)
    scores = rng.normal(size=(8, 6))
    orders = np.asarray([rng.permutation(6)[:3] for _ in range(len(scores))])
    venues = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    return scores, orders, venues


def test_zero_venue_adjustments_equal_global_conditional_model() -> None:
    scores, _orders, venues = _sample()
    model = venue_identity_model()

    actual = venue_conditional_probabilities(scores, model, venues)
    expected = conditional_probabilities(scores, identity_model())

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_venue_objective_gradient_matches_finite_difference() -> None:
    scores, orders, venues = _sample()
    parameters = _pack_model(
        venue_identity_model(
            global_regularization=0.001,
            venue_regularization=0.01,
        )
    )
    rng = np.random.default_rng(7)
    parameters = parameters + rng.normal(scale=0.01, size=parameters.shape)
    value, gradient = objective_gradient(
        parameters,
        scores,
        orders,
        venues,
        global_regularization=0.001,
        venue_regularization=0.01,
    )

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    epsilon = 1e-6
    indices = np.asarray([0, 3, 38, 75, 110, 111, 146, 975, 1400, len(parameters) - 1])
    for index in indices:
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_value = objective_gradient(
            upper,
            scores,
            orders,
            venues,
            global_regularization=0.001,
            venue_regularization=0.01,
        )[0]
        lower_value = objective_gradient(
            lower,
            scores,
            orders,
            venues,
            global_regularization=0.001,
            venue_regularization=0.01,
        )[0]
        numeric = (upper_value - lower_value) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient[index], numeric, rtol=2e-4, atol=2e-6)


def test_venue_codes_are_mapped_to_zero_based_indices() -> None:
    keys = [
        ("race-1", "2026-01-01", "01", 1),
        ("race-2", "2026-01-01", "24", 1),
    ]

    np.testing.assert_array_equal(venue_indices(keys), np.asarray([0, 23]))
