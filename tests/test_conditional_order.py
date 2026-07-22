from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.conditional_order import (
    ConditionalOrderModel,
    _pack,
    bankroll_promotion_gate,
    conditional_probabilities,
    evaluate_probabilities,
    fit_conditional_order,
    identity_model,
    objective_gradient,
)
from boatrace_ai.listwise.stagewise_mlp import (
    COMBINATION_LANES,
    stagewise_trifecta_probabilities,
)


def test_bankroll_promotion_requires_absolute_and_paired_roi_confidence() -> None:
    candidate = {"roi": 1.05, "profit_yen": 5_000}
    baseline = {"roi": 0.90, "profit_yen": -10_000}

    weak = bankroll_promotion_gate(
        candidate,
        baseline,
        {"roi_ci95_lower": 0.99, "roi_delta_ci95_lower": 0.01},
    )
    strong = bankroll_promotion_gate(
        candidate,
        baseline,
        {"roi_ci95_lower": 1.01, "roi_delta_ci95_lower": 0.01},
    )

    assert weak["roi_pass"] is True
    assert weak["roi_ci_lower_above_one"] is False
    assert weak["pass"] is False
    assert strong["pass"] is True


def test_identity_matches_standard_pl_and_probabilities_sum_to_one() -> None:
    scores = np.asarray(
        [
            [1.2, 0.4, -0.1, 0.8, -0.7, 0.2],
            [-0.2, 0.1, 0.9, 0.3, 0.6, -0.4],
        ],
        dtype=np.float64,
    )

    actual = conditional_probabilities(scores, identity_model())
    expected = stagewise_trifecta_probabilities(
        np.repeat(np.exp(scores)[:, :, None], 3, axis=2)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), np.ones(len(scores)))


def test_objective_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(20260722)
    scores = rng.normal(size=(8, 6))
    orders = np.asarray([rng.permutation(6)[:3] for _ in range(len(scores))])
    parameters = _pack(identity_model()) + rng.normal(scale=0.03, size=111)
    parameters[:3] = np.maximum(parameters[:3], 0.1)
    regularization = 0.01
    _objective, gradient = objective_gradient(
        parameters,
        scores,
        orders,
        regularization=regularization,
    )

    epsilon = 1e-6
    for index in (0, 2, 3, 10, 38, 39, 70, 75, 100, 110):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_objective, _ = objective_gradient(
            plus, scores, orders, regularization=regularization
        )
        minus_objective, _ = objective_gradient(
            minus, scores, orders, regularization=regularization
        )
        numeric = (plus_objective - minus_objective) / (2.0 * epsilon)
        assert np.isclose(gradient[index], numeric, rtol=2e-5, atol=2e-6)


def _sample_orders(
    rng: np.random.Generator,
    scores: np.ndarray,
    model: ConditionalOrderModel,
) -> np.ndarray:
    probabilities = conditional_probabilities(scores, model)
    indices = np.asarray(
        [rng.choice(len(COMBINATION_LANES), p=row) for row in probabilities]
    )
    return COMBINATION_LANES[indices]


def test_fit_recovers_useful_conditional_order_signal() -> None:
    rng = np.random.default_rng(42)
    true_model = identity_model()
    second_bias = true_model.second_bias.copy()
    third_first_bias = true_model.third_first_bias.copy()
    third_second_bias = true_model.third_second_bias.copy()
    for first in range(6):
        second_bias[first, (first + 1) % 6] = 1.4
        third_first_bias[first, (first + 2) % 6] = 0.8
    for second in range(6):
        third_second_bias[second, (second + 1) % 6] = 1.0
    true_model = ConditionalOrderModel(
        scales=np.asarray([1.1, 0.9, 1.2]),
        second_bias=second_bias,
        third_first_bias=third_first_bias,
        third_second_bias=third_second_bias,
        regularization=0.0,
    )
    train_scores = rng.normal(size=(2_000, 6))
    train_orders = _sample_orders(rng, train_scores, true_model)
    fitted, diagnostics = fit_conditional_order(
        train_scores,
        train_orders,
        regularization=0.001,
        max_iterations=80,
    )
    test_scores = rng.normal(size=(1_000, 6))
    test_orders = _sample_orders(rng, test_scores, true_model)
    ranks = np.full((len(test_orders), 6), 6, dtype=np.int8)
    for rank, lanes in enumerate(test_orders.T, start=1):
        ranks[np.arange(len(ranks)), lanes] = rank

    baseline = evaluate_probabilities(
        conditional_probabilities(test_scores, identity_model()), ranks
    )
    candidate = evaluate_probabilities(
        conditional_probabilities(test_scores, fitted), ranks
    )

    assert diagnostics["iterations"] > 0
    assert diagnostics["gradient_norm"] < 0.05
    assert candidate["trifecta_log_loss"] < baseline["trifecta_log_loss"] - 0.05
