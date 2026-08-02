from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boatrace_ai.genetic_search import GeneticSearchSettings
from boatrace_ai.listwise.four_head_nested_v22 import (
    DecisionRace,
    LabeledRace,
    RaceOutcome,
    RacePrediction,
)
from boatrace_ai.listwise.learned_purchase_allocation_v33 import (
    AllocationConfig,
    _normalization,
    _objective_gradient,
    _prepare_pairs,
    allocation_decision,
    decision_feature_matrices,
    fit_learned_allocation_head,
)


CHOICES = 12


def _pair(date: str, sequence: int, *, profitable_context: bool):
    context = 1.0 if profitable_context else -1.0
    race_id = f"{date}-01-{sequence:03d}"
    decision = DecisionRace(
        race_id=race_id,
        race_date=date,
        features=tuple(
            (context, index / (CHOICES - 1)) for index in range(CHOICES)
        ),
        current_odds=tuple(8.0 + index / 4.0 for index in range(CHOICES)),
    )
    probability = np.linspace(CHOICES, 1.0, CHOICES)
    probability /= probability.sum()
    prediction = RacePrediction(
        race_id=race_id,
        race_date=date,
        probabilities=tuple(float(value) for value in probability),
        ranking_scores=tuple(float(value) for value in probability),
        predicted_closing_odds=tuple(
            8.5 + index / 4.0 for index in range(CHOICES)
        ),
        purchase_scores=tuple(-1.0 for _ in range(CHOICES)),
        selected_indices=(),
    )
    winner = 0 if profitable_context else CHOICES - 1
    closing = [8.5 + index / 4.0 for index in range(CHOICES)]
    closing[winner] = 8.0 if profitable_context else 2.0
    outcome = RaceOutcome(
        winner_index=winner,
        closing_odds=tuple(closing),
        ranking_order=(winner, *[index for index in range(CHOICES) if index != winner]),
    )
    return LabeledRace(decision, outcome), prediction


def _pairs(days: int = 12, races_per_day: int = 20):
    pairs = []
    for day in range(1, days + 1):
        date = f"2026-07-{day:02d}"
        for sequence in range(races_per_day):
            pairs.append(
                _pair(date, sequence, profitable_context=sequence % 2 == 0)
            )
    return tuple(zip(*pairs, strict=True))


def _payouts(races):
    return {
        race.decision.race_id: int(
            round(race.outcome.closing_odds[race.outcome.winner_index] * 100)
        )
        for race in races
    }


def test_decision_features_do_not_read_settlement_fields() -> None:
    race, prediction = _pair("2026-07-01", 1, profitable_context=True)
    before = decision_feature_matrices(race.decision, prediction)
    changed = replace(
        race,
        outcome=replace(race.outcome, winner_index=CHOICES - 1),
    )
    after = decision_feature_matrices(changed.decision, prediction)

    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)


def test_training_teacher_uses_official_payout_not_closing_odds() -> None:
    race, prediction = _pair("2026-07-01", 1, profitable_context=True)
    prepared, _digest = _prepare_pairs(
        (race,), (prediction,), {race.decision.race_id: 12_300}
    )

    assert prepared[0].payout_odds == 123.0


def test_training_teacher_rejects_missing_official_payout() -> None:
    race, prediction = _pair("2026-07-01", 1, profitable_context=True)
    with pytest.raises(ValueError, match="official realized payout"):
        _prepare_pairs((race,), (prediction,), {})


def test_analytic_gradient_matches_finite_difference() -> None:
    races, predictions = _pairs(days=4, races_per_day=4)
    prepared, _digest = _prepare_pairs(races, predictions, _payouts(races))
    normalization = _normalization(prepared)
    dimension = normalization[0].size + normalization[2].size + 1
    parameters = np.linspace(-0.03, 0.03, dimension)
    config = AllocationConfig("gradient", 0.07, 0.8)
    value, gradient = _objective_gradient(
        parameters,
        prepared,
        ticket_mean=normalization[0],
        ticket_scale=normalization[1],
        race_mean=normalization[2],
        race_scale=normalization[3],
        config=config,
    )

    assert np.isfinite(value)
    for index in (0, normalization[0].size - 1, dimension - 1):
        step = np.zeros(dimension)
        step[index] = 1e-6
        plus = _objective_gradient(
            parameters + step,
            prepared,
            ticket_mean=normalization[0],
            ticket_scale=normalization[1],
            race_mean=normalization[2],
            race_scale=normalization[3],
            config=config,
        )[0]
        minus = _objective_gradient(
            parameters - step,
            prepared,
            ticket_mean=normalization[0],
            ticket_scale=normalization[1],
            race_mean=normalization[2],
            race_scale=normalization[3],
            config=config,
        )[0]
        assert gradient[index] == pytest.approx(
            (plus - minus) / 2e-6, rel=2e-4, abs=2e-5
        )


def test_gate_intercept_is_not_zero_center_regularized() -> None:
    races, predictions = _pairs(days=4, races_per_day=4)
    prepared, _digest = _prepare_pairs(races, predictions, _payouts(races))
    normalization = _normalization(prepared)
    dimension = normalization[0].size + normalization[2].size + 1
    parameters = np.zeros(dimension)
    parameters[-1] = -4.0

    low = _objective_gradient(
        parameters,
        prepared,
        ticket_mean=normalization[0],
        ticket_scale=normalization[1],
        race_mean=normalization[2],
        race_scale=normalization[3],
        config=AllocationConfig("low", 0.01, 1.0),
    )
    high = _objective_gradient(
        parameters,
        prepared,
        ticket_mean=normalization[0],
        ticket_scale=normalization[1],
        race_mean=normalization[2],
        race_scale=normalization[3],
        config=AllocationConfig("high", 10.0, 1.0),
    )

    assert low[0] == pytest.approx(high[0])
    assert low[1][-1] == pytest.approx(high[1][-1])


def test_fit_rejects_non_prior_base_heads() -> None:
    races, predictions = _pairs(days=4, races_per_day=4)
    with pytest.raises(ValueError, match="strictly before"):
        fit_learned_allocation_head(
            races,
            predictions,
            _payouts(races),
            base_predictions_trained_through_date="2026-07-01",
            configs=(AllocationConfig("test", 0.1, 1.0),),
            max_iterations=10,
        )


def test_walk_forward_selection_uses_only_strictly_prior_days() -> None:
    races, predictions = _pairs(days=5, races_per_day=4)

    artifact = fit_learned_allocation_head(
        races,
        predictions,
        _payouts(races),
        base_predictions_trained_through_date="2026-06-30",
        configs=(AllocationConfig("walk", 0.1, 1.0),),
        max_iterations=20,
        selection_mode="walk-forward",
    )

    candidate = artifact.candidate_metrics[0]
    assert candidate["selection_mode"] == "walk-forward"
    assert [row["validation_date"] for row in candidate["folds"]] == [
        "2026-07-03",
        "2026-07-04",
        "2026-07-05",
    ]
    assert candidate["validation_days"] == 3
    assert candidate["validation_races"] == 12


def test_genetic_walk_forward_is_reproducible_and_audited() -> None:
    races, predictions = _pairs(days=5, races_per_day=4)
    kwargs = {
        "base_predictions_trained_through_date": "2026-06-30",
        "configs": (AllocationConfig("seed", 0.03, 0.5),),
        "max_iterations": 20,
        "selection_mode": "walk-forward",
        "genetic_search": GeneticSearchSettings(
            population_size=4,
            generations=2,
            elite_count=1,
            max_workers=2,
            seed=23,
        ),
    }

    first = fit_learned_allocation_head(
        races, predictions, _payouts(races), **kwargs
    )
    second = fit_learned_allocation_head(
        races, predictions, _payouts(races), **kwargs
    )

    assert first.selected_config == second.selected_config
    assert first.candidate_metrics == second.candidate_metrics
    assert first.search_protocol == "genetic-walk-forward-v1"
    assert first.search_history == second.search_history
    assert len(first.search_history) == 2
    assert all("new_evaluations" in row for row in first.search_history)


def test_learns_cash_exposure_and_ticket_allocation_then_rounds_to_units() -> None:
    races, predictions = _pairs()
    artifact = fit_learned_allocation_head(
        races,
        predictions,
        _payouts(races),
        base_predictions_trained_through_date="2026-06-30",
        configs=(AllocationConfig("test", 0.03, 0.5),),
        max_iterations=160,
    )
    profitable, profitable_prediction = _pair(
        "2026-07-13", 1, profitable_context=True
    )
    unprofitable, unprofitable_prediction = _pair(
        "2026-07-13", 2, profitable_context=False
    )

    buy = allocation_decision(
        artifact,
        profitable.decision,
        profitable_prediction,
        available_bankroll_yen=10_000,
    )
    hold = allocation_decision(
        artifact,
        unprofitable.decision,
        unprofitable_prediction,
        available_bankroll_yen=10_000,
    )

    assert artifact.outer_outcomes_used is False
    assert artifact.converged
    assert buy.proposed_stake_yen > 0
    assert buy.proposed_stake_yen <= 500
    assert buy.proposed_stake_yen >= hold.proposed_stake_yen
    assert buy.gate_probability > hold.gate_probability
    assert all(stake % 100 == 0 for stake in buy.stakes_yen)
    assert sum(buy.stakes_yen) == buy.proposed_stake_yen
    assert buy.stakes_yen[0] == max(buy.stakes_yen)


def test_inference_rejects_training_dates() -> None:
    races, predictions = _pairs(days=4, races_per_day=4)
    artifact = fit_learned_allocation_head(
        races,
        predictions,
        _payouts(races),
        base_predictions_trained_through_date="2026-06-30",
        configs=(AllocationConfig("test", 0.1, 1.0),),
        max_iterations=30,
    )
    race, prediction = _pair("2026-07-04", 99, profitable_context=True)
    with pytest.raises(ValueError, match="strictly after"):
        allocation_decision(
            artifact,
            race.decision,
            prediction,
            available_bankroll_yen=10_000,
        )
