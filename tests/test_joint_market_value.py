from __future__ import annotations

import math

import pytest

from boatrace_ai.joint_market_value import (
    JointMarketScenario,
    TRIFECTA_OUTCOMES,
    evaluate_joint_bankroll_growth,
    evaluate_joint_market_value,
    validate_probability_simplex,
    validate_trifecta_probability_simplex,
)


def test_bankroll_growth_penalizes_ruin_and_prefers_fractional_stake() -> None:
    draws = [[JointMarketScenario(
        {"A": 0.6, "B": 0.4},
        {"multipliers": {"A": 2.0, "B": 0.0}},
    )] for _ in range(20)]

    fractional = evaluate_joint_bankroll_growth(
        draws,
        bets_yen={"A": 200},
        gross_payoff_model=_ordinary_payoffs,
        available_bankroll_yen=1_000,
        expected_outcomes=("A", "B"),
        minimum_outer_draws=20,
    )
    all_in = evaluate_joint_bankroll_growth(
        draws,
        bets_yen={"A": 1_000},
        gross_payoff_model=_ordinary_payoffs,
        available_bankroll_yen=1_000,
        expected_outcomes=("A", "B"),
        minimum_outer_draws=20,
    )

    assert fractional["growth"]["lower_quantile"] > 0.0
    assert fractional["growth"]["passes_growth_gate"] is True
    assert fractional["inner_scenario_count_s_min"] == 1
    assert fractional["inner_scenario_count_s_max"] == 1
    assert fractional["inner_effective_samples_min"] == 1.0
    assert fractional["inner_tail_support_for_purchase"] is True
    assert all_in["growth"]["lower_quantile"] < 0.0
    assert all_in["growth"]["maximum_conditional_ruin_probability"] == pytest.approx(
        0.4
    )


def test_bankroll_growth_sparse_settlement_matches_terminal_state_sum() -> None:
    scenario = JointMarketScenario(
        {"A": 0.3, "B": 0.2, "cancelled": 0.5},
        {},
    )

    def payoffs(_scenario, _bets):
        return {
            "A": {"A": 400, "cancelled": 100},
            "B": {"B": 1_000, "cancelled": 200},
        }

    result = evaluate_joint_bankroll_growth(
        [[scenario]],
        bets_yen={"A": 100, "B": 200},
        gross_payoff_model=payoffs,
        available_bankroll_yen=1_000,
        expected_outcomes=("A", "B", "cancelled"),
        minimum_outer_draws=1,
    )
    expected = (
        0.3 * math.log(1_100 / 1_000)
        + 0.2 * math.log(1_700 / 1_000)
        + 0.5 * math.log(1_000 / 1_000)
    )

    assert result["growth"]["mean"] == pytest.approx(expected)
    assert result["growth"]["maximum_conditional_ruin_probability"] == 0.0


def _ordinary_payoffs(scenario, bets):
    multipliers = scenario.market_state["multipliers"]
    return {
        ticket: {ticket: round(stake * multipliers[ticket])}
        for ticket, stake in bets.items()
    }


def test_joint_value_keeps_probability_price_covariance() -> None:
    draw = [
        JointMarketScenario(
            {"1-2-3": 0.8, "1-3-2": 0.2},
            {"multipliers": {"1-2-3": 1.2}},
        ),
        JointMarketScenario(
            {"1-2-3": 0.2, "1-3-2": 0.8},
            {"multipliers": {"1-2-3": 6.0}},
        ),
    ]
    result = evaluate_joint_market_value(
        [draw],
        bets_yen={"1-2-3": 100},
        gross_payoff_model=_ordinary_payoffs,
        minimum_outer_draws=1,
    )
    moments = result["moments_by_draw"][0]["tickets"]["1-2-3"]
    assert moments["expected_probability_times_multiplier"] == pytest.approx(1.08)
    assert moments["independence_probability_times_multiplier"] == pytest.approx(1.80)
    assert moments["probability_multiplier_covariance"] == pytest.approx(-0.72)
    assert moments["expected_probability_times_multiplier"] == pytest.approx(
        moments["independence_probability_times_multiplier"]
        + moments["probability_multiplier_covariance"]
    )
    assert moments["joint_expected_edge"] == pytest.approx(0.08)
    assert moments["ordinary_hit_independence_approximation_edge"] == pytest.approx(
        0.8
    )


def test_portfolio_tail_is_taken_after_scenario_level_diversification() -> None:
    scenarios = [
        JointMarketScenario(
            {"A": 0.5, "B": 0.5},
            {"multipliers": {"A": 4.0, "B": 0.0}},
        ),
        JointMarketScenario(
            {"A": 0.5, "B": 0.5},
            {"multipliers": {"A": 0.0, "B": 4.0}},
        ),
    ]
    result = evaluate_joint_market_value(
        [scenarios],
        bets_yen={"A": 100, "B": 100},
        gross_payoff_model=_ordinary_payoffs,
        inner_tail_fraction=0.5,
        minimum_outer_draws=1,
        minimum_inner_tail_effective_samples=1.0,
    )

    assert result["tickets"]["A"]["mean"] == pytest.approx(-1.0)
    assert result["tickets"]["B"]["mean"] == pytest.approx(-1.0)
    assert result["portfolio"]["mean"] == pytest.approx(0.0)
    assert result["inner_aggregation"] == (
        "portfolio_path_weighted_lower_tail_mean"
    )
    assert result["inner_scenario_count_s_min"] == 2
    assert result["inner_scenario_count_s_max"] == 2
    assert result["inner_effective_samples_min"] == 2.0
    assert result["inner_effective_samples_mean"] == 2.0
    assert result["inner_effective_samples_max"] == 2.0
    assert result["inner_tail_effective_samples_min"] == 1.0
    assert result["inner_tail_effective_samples_mean"] == 1.0
    assert result["inner_tail_effective_samples_max"] == 1.0
    assert result["minimum_inner_tail_effective_samples"] == 1.0
    assert result["inner_tail_support_for_purchase"] is True


def test_cancelled_terminal_state_returns_principal_without_hit() -> None:
    scenario = JointMarketScenario(
        {"A": 0.0, "B": 0.0, "cancelled": 1.0}, {}
    )

    def refunds(_scenario, bets):
        return {ticket: {"cancelled": stake} for ticket, stake in bets.items()}

    result = evaluate_joint_market_value(
        [[scenario]],
        bets_yen={"A": 100},
        gross_payoff_model=refunds,
        expected_outcomes=("A", "B", "cancelled"),
        minimum_outer_draws=1,
    )

    assert result["portfolio"]["mean"] == pytest.approx(0.0)
    moments = result["moments_by_draw"][0]["tickets"]["A"]
    assert moments["mean_probability"] == 0.0
    assert moments["mean_other_terminal_receipts_per_yen"] == 1.0


def test_marginal_contribution_recomputes_payoffs_after_ticket_removal() -> None:
    observed_vectors = []

    def self_impact_payoffs(_scenario, bets):
        observed_vectors.append(dict(bets))
        total = sum(bets.values())
        return {
            ticket: {ticket: round(stake * (3.0 - total / 1_000.0))}
            for ticket, stake in bets.items()
        }

    scenario = JointMarketScenario({"A": 0.5, "B": 0.5}, {})
    result = evaluate_joint_market_value(
        [[scenario]],
        bets_yen={"A": 100, "B": 200},
        gross_payoff_model=self_impact_payoffs,
        minimum_outer_draws=1,
    )

    assert {tuple(sorted(row)) for row in observed_vectors} == {
        ("A", "B"),
        ("A",),
        ("B",),
    }
    assert result["marginal_contributions"]["A"]["available"] is True
    assert "passes_purchase_gate" not in result["tickets"]["A"]


def test_weighted_tail_uses_partial_boundary_mass() -> None:
    draw = [
        JointMarketScenario(
            {"A": 0.5, "B": 0.5},
            {"multipliers": {"A": 4.0}},
            weight=0.75,
        ),
        JointMarketScenario(
            {"A": 0.5, "B": 0.5},
            {"multipliers": {"A": 1.0}},
            weight=0.25,
        ),
    ]
    result = evaluate_joint_market_value(
        [draw],
        bets_yen={"A": 100},
        gross_payoff_model=_ordinary_payoffs,
        inner_tail_fraction=0.5,
        minimum_outer_draws=1,
        minimum_inner_tail_effective_samples=0.1,
    )

    assert result["portfolio"]["mean"] == pytest.approx(0.25)


def test_outer_quantile_uses_observed_order_statistic_not_interpolation() -> None:
    draws = [
        [JointMarketScenario({"A": 1.0}, {"multipliers": {"A": 1.0}})],
        [JointMarketScenario({"A": 1.0}, {"multipliers": {"A": 2.0}})],
    ]
    result = evaluate_joint_market_value(
        draws,
        bets_yen={"A": 100},
        gross_payoff_model=_ordinary_payoffs,
        outer_alpha=0.5,
        minimum_outer_draws=1,
    )

    assert result["portfolio"]["lower_quantile"] == pytest.approx(0.0)
    assert result["outer_quantile_method"] == "inverted_cdf"


def test_purchase_gate_is_disabled_when_outer_or_inner_evidence_is_small() -> None:
    draws = [
        [JointMarketScenario({"A": 1.0}, {"multipliers": {"A": 2.0}})]
    ] * 2
    result = evaluate_joint_market_value(
        draws,
        bets_yen={"A": 100},
        gross_payoff_model=_ordinary_payoffs,
        inner_tail_fraction=0.05,
    )

    assert result["portfolio"]["purchase_gate_evaluable"] is False
    assert result["portfolio"]["passes_purchase_gate"] is False
    assert set(result["portfolio"]["purchase_gate_reasons"]) == {
        "insufficient_outer_parameter_draws",
        "insufficient_inner_tail_effective_samples",
    }
    assert result["inner_scenario_count_s_min"] == 1
    assert result["inner_tail_effective_samples_min"] == 1.0
    assert result["minimum_inner_tail_effective_samples"] == 5.0
    assert result["inner_tail_support_for_purchase"] is False


def test_tail_ess_uses_actual_partial_scenario_mass() -> None:
    draw = [
        JointMarketScenario(
            {"A": 1.0}, {"multipliers": {"A": 0.5}}, weight=0.1
        ),
        JointMarketScenario(
            {"A": 1.0}, {"multipliers": {"A": 2.0}}, weight=0.9
        ),
    ]

    result = evaluate_joint_market_value(
        [draw],
        bets_yen={"A": 100},
        gross_payoff_model=_ordinary_payoffs,
        inner_tail_fraction=0.2,
        minimum_outer_draws=1,
        minimum_inner_tail_effective_samples=2.0,
    )

    # Tail mass is 0.1 from each path, so normalized weights are 0.5/0.5.
    assert result["inner_effective_samples_min"] == pytest.approx(1 / 0.82)
    assert result["inner_tail_effective_samples_min"] == pytest.approx(2.0)
    assert result["inner_tail_support_for_purchase"] is True


def test_trifecta_adapter_requires_exactly_the_120_outcomes() -> None:
    complete = {combination: 1.0 / 120.0 for combination in TRIFECTA_OUTCOMES}
    validated = validate_trifecta_probability_simplex(complete)
    assert len(validated) == 120

    incomplete = dict(complete)
    incomplete.pop(TRIFECTA_OUTCOMES[-1])
    incomplete[TRIFECTA_OUTCOMES[0]] += 1.0 / 120.0
    with pytest.raises(ValueError, match="expected set"):
        validate_trifecta_probability_simplex(incomplete)


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"A": 0.4, "B": 0.4},
        {"A": -0.1, "B": 1.1},
        {"A": float("nan"), "B": 1.0},
        {1: 0.5, "1": 0.5},
    ],
)
def test_probability_simplex_rejects_invalid_values_and_non_string_keys(
    probabilities,
) -> None:
    with pytest.raises(ValueError):
        validate_probability_simplex(probabilities)


def test_payoff_table_must_match_bet_vector_exactly() -> None:
    scenario = JointMarketScenario({"A": 1.0}, {})

    def extra_ticket(_scenario, _bets):
        return {"A": {"A": 100}, "B": {"A": 0}}

    with pytest.raises(ValueError, match="complete bet vector"):
        evaluate_joint_market_value(
            [[scenario]],
            bets_yen={"A": 100},
            gross_payoff_model=extra_ticket,
            minimum_outer_draws=1,
        )


def test_ga_evaluation_can_skip_marginal_repricing() -> None:
    observed_vectors = []

    def payoffs(_scenario, bets):
        observed_vectors.append(dict(bets))
        return {ticket: {ticket: amount * 2} for ticket, amount in bets.items()}

    result = evaluate_joint_market_value(
        [[JointMarketScenario({"A": 0.5, "B": 0.5})]],
        bets_yen={"A": 100, "B": 100},
        gross_payoff_model=payoffs,
        minimum_outer_draws=1,
        include_marginal_contributions=False,
        include_ticket_diagnostics=False,
    )

    assert observed_vectors == [{"A": 100, "B": 100}]
    assert result["marginal_contributions_computed"] is False
    assert result["marginal_contributions"] == {}
    assert result["ticket_diagnostics_computed"] is False
    assert result["tickets"] == {}
    assert result["moments_by_draw"] == []
