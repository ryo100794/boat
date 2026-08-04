from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.conditional_order import (
    ConditionalOrderModel,
    _pack,
    bankroll_promotion_gate,
    build_parser,
    conditional_probabilities,
    evaluate_probabilities,
    evaluate_direct_pair_diagnostics,
    evaluate_reversed_place_pair_structure,
    fit_conditional_order,
    identity_model,
    objective_gradient,
)
from boatrace_ai.listwise.direct_bankroll import (
    COMBINATION_LABELS,
    simulate_direct_bankroll,
)


def test_reversed_place_pair_structure_selects_same_winner_swap() -> None:
    probabilities = np.full((2, 120), 0.5 / 118.0, dtype=np.float64)
    left = int(np.flatnonzero(np.all(
        COMBINATION_LANES == np.asarray([0, 1, 2]), axis=1
    ))[0])
    right = int(np.flatnonzero(np.all(
        COMBINATION_LANES == np.asarray([0, 2, 1]), axis=1
    ))[0])
    probabilities[:, left] = 0.30
    probabilities[:, right] = 0.20
    ranks = np.asarray([
        [1, 2, 3, 4, 5, 6],
        [2, 1, 3, 4, 5, 6],
    ])

    metrics = evaluate_reversed_place_pair_structure(probabilities, ranks)

    assert metrics["evaluated_races"] == 2
    assert metrics["selected_pair_hit_rate"] == 0.5
    assert metrics["selected_winner_hit_rate"] == 0.5
    assert metrics["pair_hit_rate_given_winner"] == 1.0
    assert np.isclose(metrics["mean_selected_pair_probability"], 0.5)
    assert metrics["pair_calibration_gap"] == 0.0


def test_direct_pair_diagnostics_execute_all_four_research_paths() -> None:
    probabilities = np.full((1, 120), 1e-12, dtype=np.float64)
    probabilities[0, COMBINATION_LABELS.index("1-2-3")] = 0.5
    probabilities[0, COMBINATION_LABELS.index("1-3-2")] = 0.5
    race_keys = [("test", "2026-07-20", "01", 1)]
    payouts = {
        "train": {"combination": "1-2-3", "payout_yen": 2_000},
        "test": {"combination": "1-3-2", "payout_yen": 2_000},
    }
    full_baseline = simulate_direct_bankroll(
        probabilities,
        race_keys=race_keys,
        payouts=payouts,
        training_races={"train"},
    )

    result = evaluate_direct_pair_diagnostics(
        baseline_probabilities=probabilities,
        candidate_probabilities=probabilities,
        race_keys=race_keys,
        payouts=payouts,
        training_races={"train"},
        full_baseline_bankroll=full_baseline,
    )

    assert set(result) == {
        "baseline_exact_two",
        "baseline_reversed_place_pair",
        "conditional_exact_two",
        "conditional_reversed_place_pair",
    }
    for diagnostic in result.values():
        assert diagnostic["promotion_eligible"] is False
        assert diagnostic["bankroll"]["evaluated_races"] == 1
        assert diagnostic["bankroll"]["selected_tickets"] == 2
        assert diagnostic["diagnostic_gate"][
            "reused_holdout_research_only"
        ] is True
        assert diagnostic["diagnostic_gate"]["holdout_role_pass"] is False
        assert diagnostic["diagnostic_gate"]["pass"] is False
from boatrace_ai.listwise.stagewise_mlp import (
    COMBINATION_LANES,
    stagewise_trifecta_probabilities,
)


def test_policy_selection_cli_defaults_to_sixty_days() -> None:
    args = build_parser().parse_args(
        [
            "--cache-prefix", "cache",
            "--baseline-model", "model.joblib",
            "--training-through", "2025-07-23",
            "--evaluation-from", "2025-07-24",
            "--evaluation-through", "2026-07-23",
            "--model-output", "output.joblib",
            "--output", "output.json",
        ]
    )

    assert args.payout_policy_selection_days == 60
    assert args.return_policy_selection_days == 60


def test_bankroll_promotion_requires_absolute_and_paired_roi_confidence() -> None:
    candidate = {
        "roi": 1.05,
        "profit_yen": 5_000,
        "roi_without_largest_hit": 1.01,
        "evaluation_days": 365,
        "evaluated_races": 49_581,
        "selected_tickets": 300,
        "hit_tickets": 30,
        "effective_hit_count": 25.0,
        "winning_days": 220,
    }
    baseline = {"roi": 0.90, "profit_yen": -10_000}

    weak = bankroll_promotion_gate(
        candidate,
        baseline,
        {
            "roi_ci95_lower": 0.99,
            "roi_delta_ci95_lower": 0.01,
            "probability_roi_above_one": 0.99,
        },
    )
    weak_probability = bankroll_promotion_gate(
        candidate,
        baseline,
        {
            "roi_ci95_lower": 1.01,
            "roi_delta_ci95_lower": 0.01,
            "probability_roi_above_one": 0.94,
        },
    )
    strong = bankroll_promotion_gate(
        candidate,
        baseline,
        {
            "roi_ci95_lower": 1.01,
            "roi_delta_ci95_lower": 0.01,
            "probability_roi_above_one": 0.95,
        },
    )
    fragile = bankroll_promotion_gate(
        {**candidate, "roi_without_largest_hit": 0.99},
        baseline,
        {
            "roi_ci95_lower": 1.01,
            "roi_delta_ci95_lower": 0.01,
            "probability_roi_above_one": 0.95,
        },
    )
    too_small = bankroll_promotion_gate(
        {**candidate, "selected_tickets": 199},
        baseline,
        {
            "roi_ci95_lower": 1.01,
            "roi_delta_ci95_lower": 0.01,
            "probability_roi_above_one": 0.95,
        },
    )
    concentrated = bankroll_promotion_gate(
        {**candidate, "effective_hit_count": 19.99},
        baseline,
        {
            "roi_ci95_lower": 1.01,
            "roi_delta_ci95_lower": 0.01,
            "probability_roi_above_one": 0.95,
        },
    )

    assert weak["roi_pass"] is True
    assert weak["roi_ci_lower_above_one"] is False
    assert weak["pass"] is False
    assert weak_probability["probability_roi_above_one_pass"] is False
    assert weak_probability["pass"] is False
    assert fragile["largest_hit_excluded_roi_pass"] is False
    assert fragile["pass"] is False
    assert too_small["selected_tickets_pass"] is False
    assert too_small["pass"] is False
    assert concentrated["effective_hit_count_pass"] is False
    assert concentrated["pass"] is False
    assert strong["largest_hit_excluded_roi_pass"] is True
    assert strong["profitable_day_fraction_pass"] is True
    assert strong["probability_roi_above_one_pass"] is True
    assert strong["effective_hit_count_pass"] is True
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
