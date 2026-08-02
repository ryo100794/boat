from __future__ import annotations

import pytest

from boatrace_ai.joint_market_value import (
    JointMarketScenario,
    evaluate_joint_market_value,
    validate_probability_simplex,
)


def _payouts(scenario, bets):
    assert set(bets) == {"1-2-3"}
    return {"1-2-3": scenario.market_state["payout"]}


def test_joint_value_keeps_probability_price_covariance() -> None:
    draw = [
        JointMarketScenario(
            {"1-2-3": 0.8, "1-3-2": 0.2}, {"payout": 1.2}
        ),
        JointMarketScenario(
            {"1-2-3": 0.2, "1-3-2": 0.8}, {"payout": 6.0}
        ),
    ]
    result = evaluate_joint_market_value(
        [draw], bets_yen={"1-2-3": 100}, payout_model=_payouts
    )
    moments = result["moments_by_draw"][0]["tickets"]["1-2-3"]
    assert moments["probability_payout_covariance"] == pytest.approx(-0.72)
    assert moments["joint_expected_edge"] == pytest.approx(0.08)
    assert moments["independence_approximation_edge"] == pytest.approx(0.8)


def test_outer_quantile_is_over_path_integrated_parameter_draws() -> None:
    draws = [
        [JointMarketScenario({"1-2-3": 0.5, "1-3-2": 0.5}, {"payout": p}) for p in (2.4, 1.6)],
        [JointMarketScenario({"1-2-3": 0.5, "1-3-2": 0.5}, {"payout": p}) for p in (2.0, 1.2)],
    ]
    result = evaluate_joint_market_value(
        draws,
        bets_yen={"1-2-3": 100},
        payout_model=_payouts,
        outer_alpha=0.5,
    )
    ticket = result["tickets"]["1-2-3"]
    assert ticket["mean"] == pytest.approx(-0.1)
    assert ticket["lower_quantile"] == pytest.approx(-0.1)
    assert ticket["passes_purchase_gate"] is False


def test_inner_tail_mean_penalizes_bad_market_paths() -> None:
    draw = [
        JointMarketScenario(
            {"1-2-3": 0.5, "1-3-2": 0.5}, {"payout": 4.0}, weight=0.75
        ),
        JointMarketScenario(
            {"1-2-3": 0.5, "1-3-2": 0.5}, {"payout": 1.0}, weight=0.25
        ),
    ]
    result = evaluate_joint_market_value(
        [draw],
        bets_yen={"1-2-3": 100},
        payout_model=_payouts,
        inner_tail_fraction=0.25,
    )
    assert result["tickets"]["1-2-3"]["mean"] == pytest.approx(-0.5)
    assert result["inner_aggregation"] == "weighted_lower_tail_mean"


def test_payout_model_receives_complete_bet_vector_for_self_impact() -> None:
    observed = []

    def payout_model(scenario, bets):
        observed.append(dict(bets))
        return {key: 2.0 for key in bets}

    scenario = JointMarketScenario(
        {"1-2-3": 0.5, "1-3-2": 0.5}, {"pool": 10_000}
    )
    evaluate_joint_market_value(
        [[scenario]],
        bets_yen={"1-2-3": 100, "1-3-2": 200},
        payout_model=payout_model,
    )
    assert observed == [{"1-2-3": 100, "1-3-2": 200}]


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"1-2-3": 0.4, "1-3-2": 0.4},
        {"1-2-3": -0.1, "1-3-2": 1.1},
        {"1-2-3": float("nan"), "1-3-2": 1.0},
    ],
)
def test_probability_simplex_is_enforced(probabilities) -> None:
    with pytest.raises(ValueError):
        validate_probability_simplex(probabilities)
