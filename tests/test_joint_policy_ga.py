from __future__ import annotations

import pytest

from boatrace_ai.genetic_search import GeneticSearchSettings
from boatrace_ai.joint_market_value import JointMarketScenario
from boatrace_ai.joint_policy_ga import (
    JointPolicySearchConfig,
    optimize_joint_portfolio,
)


def _draws(count: int):
    return [
        [JointMarketScenario(
            {"A": 0.6, "B": 0.3, "C": 0.1},
            {"multipliers": {"A": 2.0, "B": 2.0, "C": 8.0}},
        )]
        for _ in range(count)
    ]


def _payoffs(scenario, bets):
    return {
        ticket: {
            ticket: round(
                amount * scenario.market_state["multipliers"][ticket]
            )
        }
        for ticket, amount in bets.items()
    }


def _settings(backend: str = "thread") -> GeneticSearchSettings:
    return GeneticSearchSettings(
        population_size=8,
        generations=4,
        elite_count=3,
        mutation_rate=0.4,
        random_injections=1,
        max_workers=1,
        execution_backend=backend,
        seed=41,
    )


def test_ga_optimizes_complete_stake_vector_and_reprices_selected_vector() -> None:
    result = optimize_joint_portfolio(
        _draws(20),
        candidate_tickets=("A", "B", "C"),
        gross_payoff_model=_payoffs,
        available_bankroll_yen=1_000,
        expected_outcomes=("A", "B", "C"),
        config=JointPolicySearchConfig(
            maximum_portfolio_stake_yen=1_000,
            maximum_ticket_stake_yen=1_000,
            maximum_selected_tickets=3,
            inner_tail_fraction=None,
            minimum_outer_draws=20,
        ),
        genetic_settings=_settings(),
    )

    assert result["selected"]["bets_yen"]
    assert 0 < result["selected"]["total_stake_yen"] < 1_000
    assert result["selected"]["fitness"] > 0.0
    assert result["purchase_authorized"] is True
    assert result["selected"]["bankroll_growth"]["growth"][
        "passes_growth_gate"
    ] is True
    assert result["selected"]["joint_value"][
        "marginal_contributions_computed"
    ] is True
    assert all(
        row["total_stake_yen"] <= 1_000
        for row in result["ranked_candidates"]
    )


def test_insufficient_outer_draws_selects_no_bet() -> None:
    result = optimize_joint_portfolio(
        _draws(2),
        candidate_tickets=("A", "B", "C"),
        gross_payoff_model=_payoffs,
        available_bankroll_yen=1_000,
        expected_outcomes=("A", "B", "C"),
        config=JointPolicySearchConfig(
            maximum_portfolio_stake_yen=1_000,
            maximum_ticket_stake_yen=1_000,
            maximum_selected_tickets=3,
            inner_tail_fraction=None,
            minimum_outer_draws=20,
        ),
        genetic_settings=_settings(),
    )

    assert result["selected"]["bets_yen"] == {}
    assert result["selected"]["fitness"] == 0.0
    assert result["purchase_authorized"] is False
    assert result["selected"]["joint_value"] is None
    assert result["feasible_candidates_found"] == 0
    assert result["best_search_candidate"]["bets_yen"] == {}


def test_rejected_vectors_keep_a_constraint_gradient_but_select_no_bet() -> None:
    def losing_payoffs(scenario, bets):
        multipliers = {"A": 1, "B": 1, "C": 1}
        return {
            ticket: {ticket: amount * multipliers[ticket]}
            for ticket, amount in bets.items()
        }

    result = optimize_joint_portfolio(
        _draws(20),
        candidate_tickets=("A", "B", "C"),
        gross_payoff_model=losing_payoffs,
        available_bankroll_yen=1_000,
        expected_outcomes=("A", "B", "C"),
        config=JointPolicySearchConfig(
            maximum_portfolio_stake_yen=1_000,
            maximum_ticket_stake_yen=1_000,
            maximum_selected_tickets=3,
            inner_tail_fraction=None,
            minimum_outer_draws=20,
        ),
        genetic_settings=_settings(),
    )

    assert result["purchase_authorized"] is False
    assert result["selected"]["bets_yen"] == {}
    assert result["selected"]["fitness"] == 0.0
    assert result["history"][0]["best_candidate"]["total_stake_yen"] > 0
    assert result["feasible_candidates_found"] == 0
    assert result["best_search_candidate"]["total_stake_yen"] > 0
    assert result["best_search_candidate"]["metrics"][
        "constraint_violation"
    ] > 0.0
    rejected = [
        row for row in result["ranked_candidates"]
        if row["total_stake_yen"] > 0
    ]
    assert rejected
    assert all(row["metrics"]["search_feasible"] is False for row in rejected)
    assert all(row["fitness"] > 0.0 for row in rejected)
    assert len({
        row["metrics"]["constraint_violation"] for row in rejected
    }) > 1


def test_validation_draws_must_not_reuse_search_draws() -> None:
    draws = _draws(20)
    with pytest.raises(ValueError, match="must be disjoint"):
        optimize_joint_portfolio(
            draws,
            validation_parameter_draws=draws,
            candidate_tickets=("A", "B", "C"),
            gross_payoff_model=_payoffs,
            available_bankroll_yen=1_000,
        )


def test_independent_validation_can_reject_search_winner() -> None:
    validation = [
        [JointMarketScenario(
            {"A": 0.6, "B": 0.3, "C": 0.1},
            {"multipliers": {"A": 0.5, "B": 0.5, "C": 0.5}},
        )]
        for _ in range(100)
    ]
    result = optimize_joint_portfolio(
        _draws(20),
        validation_parameter_draws=validation,
        validation_minimum_outer_draws=100,
        candidate_tickets=("A", "B", "C"),
        gross_payoff_model=_payoffs,
        available_bankroll_yen=1_000,
        expected_outcomes=("A", "B", "C"),
        config=JointPolicySearchConfig(
            maximum_portfolio_stake_yen=1_000,
            maximum_ticket_stake_yen=1_000,
            maximum_selected_tickets=3,
            inner_tail_fraction=None,
            minimum_outer_draws=20,
        ),
        genetic_settings=_settings(),
    )

    assert result["search_parameter_draws"] == 20
    assert result["validation_parameter_draws"] == 100
    assert result["validation_uses_separate_draw_set"] is True
    assert result["selected"]["bets_yen"]
    assert result["selected"]["metrics"]["portfolio"][
        "passes_purchase_gate"
    ] is True
    assert result["selected"]["joint_value"]["portfolio"][
        "passes_purchase_gate"
    ] is False
    assert result["purchase_authorized"] is False


def test_process_backend_matches_thread_backend() -> None:
    options = {
        "candidate_tickets": ("A", "B", "C"),
        "gross_payoff_model": _payoffs,
        "available_bankroll_yen": 1_000,
        "expected_outcomes": ("A", "B", "C"),
        "config": JointPolicySearchConfig(
            maximum_portfolio_stake_yen=1_000,
            maximum_ticket_stake_yen=1_000,
            maximum_selected_tickets=3,
            inner_tail_fraction=None,
            minimum_outer_draws=20,
        ),
    }
    thread = optimize_joint_portfolio(
        _draws(20),
        **options,
        genetic_settings=_settings("thread"),
    )
    process = optimize_joint_portfolio(
        _draws(20),
        **options,
        genetic_settings=GeneticSearchSettings(
            **{
                **_settings("process").__dict__,
                "max_workers": 2,
            }
        ),
    )

    assert process["selected"]["bets_yen"] == thread["selected"]["bets_yen"]
    assert process["selected"]["fitness"] == thread["selected"]["fitness"]
    assert process["purchase_authorized"] == thread["purchase_authorized"]
