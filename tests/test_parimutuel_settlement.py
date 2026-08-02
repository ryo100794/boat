from __future__ import annotations

import pytest

from boatrace_ai.joint_market_value import (
    JointMarketScenario,
    evaluate_joint_market_value,
)
from boatrace_ai.parimutuel_settlement import (
    ParimutuelSettlementRules,
    apportion_external_stakes,
    build_parimutuel_gross_payoff_model,
)


OUTCOMES = ("A", "B")


def _scenario(*, cancelled_probability: float = 0.0) -> JointMarketScenario:
    ordinary_mass = 1.0 - cancelled_probability
    return JointMarketScenario(
        {
            "A": ordinary_mass * 0.5,
            "B": ordinary_mass * 0.5,
            **({"cancelled": cancelled_probability} if cancelled_probability else {}),
        },
        {"external_ticket_stakes_yen": {"A": 100, "B": 300}},
    )


def test_integer_pool_settlement_recomputes_complete_bet_vector() -> None:
    settle = build_parimutuel_gross_payoff_model(ordinary_outcomes=OUTCOMES)

    single = settle(_scenario(), {"A": 100})
    portfolio = settle(_scenario(), {"A": 100, "B": 100})

    assert single == {"A": {"A": 180}}
    assert portfolio["A"]["A"] == 220
    assert portfolio["B"]["B"] == 110


def test_refund_states_return_principal_for_every_ticket() -> None:
    settle = build_parimutuel_gross_payoff_model(
        ordinary_outcomes=OUTCOMES,
        rules=ParimutuelSettlementRules(refund_terminal_states=("cancelled",)),
    )

    payoff = settle(_scenario(cancelled_probability=1.0), {"A": 200, "B": 100})

    assert payoff["A"]["cancelled"] == 200
    assert payoff["B"]["cancelled"] == 100

    value = evaluate_joint_market_value(
        [[_scenario(cancelled_probability=1.0)]],
        bets_yen={"A": 200, "B": 100},
        gross_payoff_model=settle,
        expected_outcomes=("A", "B", "cancelled"),
        minimum_outer_draws=1,
    )
    assert value["portfolio"]["mean"] == 0.0


def test_absolute_pool_is_required_for_self_impact() -> None:
    settle = build_parimutuel_gross_payoff_model(ordinary_outcomes=OUTCOMES)
    shares_only = JointMarketScenario(
        {"A": 0.5, "B": 0.5},
        {"final_market_shares": {"A": 0.25, "B": 0.75}},
    )

    with pytest.raises(ValueError, match="external_total_sales_yen"):
        settle(shares_only, {"A": 100})


def test_absolute_total_pool_and_shares_are_apportioned_in_integer_units() -> None:
    stakes = apportion_external_stakes(
        total_sales_yen=1_000,
        market_shares={"A": 0.333, "B": 0.667},
        ordinary_outcomes=OUTCOMES,
    )
    assert stakes == {"A": 330, "B": 670}
    assert sum(stakes.values()) == 1_000
    assert apportion_external_stakes(
        total_sales_yen=50,
        market_shares={"A": 0.5, "B": 0.5},
        ordinary_outcomes=OUTCOMES,
    ) == {"A": 30, "B": 20}

    settle = build_parimutuel_gross_payoff_model(ordinary_outcomes=OUTCOMES)
    scenario = JointMarketScenario(
        {"A": 0.5, "B": 0.5},
        {
            "external_total_sales_yen": 1_000,
            "final_market_shares": {"A": 0.333, "B": 0.667},
        },
    )
    assert set(settle(scenario, {"A": 100})) == {"A"}


def test_opt_in_external_stake_cache_preserves_complete_vector_settlement() -> None:
    settle = build_parimutuel_gross_payoff_model(
        ordinary_outcomes=OUTCOMES,
        cache_external_stakes=True,
    )
    scenario = _scenario()

    assert settle(scenario, {"A": 100}) == {"A": {"A": 180}}
    assert settle(scenario, {"A": 100, "B": 100}) == {
        "A": {"A": 220},
        "B": {"B": 110},
    }


def test_partial_refund_returns_only_affected_ticket_principal() -> None:
    settle = build_parimutuel_gross_payoff_model(ordinary_outcomes=OUTCOMES)
    scenario = JointMarketScenario(
        {"A": 0.4, "B": 0.4, "lane_withdrawal": 0.2},
        {
            "external_ticket_stakes_yen": {"A": 100, "B": 300},
            "partial_refund_tickets_by_state": {
                "lane_withdrawal": ["B"],
            },
        },
    )

    payoff = settle(scenario, {"A": 100, "B": 200})

    assert "lane_withdrawal" not in payoff["A"]
    assert payoff["B"]["lane_withdrawal"] == 200


@pytest.mark.parametrize("amount", [1, 10, 99, 150])
def test_purchase_stakes_must_use_100_yen_units(amount: int) -> None:
    settle = build_parimutuel_gross_payoff_model(ordinary_outcomes=OUTCOMES)
    with pytest.raises(ValueError, match="purchase-unit"):
        settle(_scenario(), {"A": amount})
