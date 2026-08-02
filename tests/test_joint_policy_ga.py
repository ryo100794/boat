from __future__ import annotations

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


def _settings() -> GeneticSearchSettings:
    return GeneticSearchSettings(
        population_size=8,
        generations=4,
        elite_count=3,
        mutation_rate=0.4,
        random_injections=1,
        max_workers=1,
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
