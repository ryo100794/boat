from __future__ import annotations

import pytest

from boatrace_ai.joint_market_value import (
    TRIFECTA_OUTCOMES,
    JointMarketScenario,
)
from boatrace_ai.parimutuel_settlement import (
    build_parimutuel_gross_payoff_model,
)
from boatrace_ai.pool_scale_lower_bound import (
    POOL_SCALE_METHOD,
    attach_pool_scale_lower_bound,
    infer_minimum_pool_scale,
)


OUTCOMES = ("A", "B", "C")


def test_minimum_pool_reproduces_integer_displayed_odds() -> None:
    # A known 100-face-unit pool gives payouts 25, 37 and 15 yen per face.
    result = infer_minimum_pool_scale(
        {"A": "2.5", "B": "3.7", "C": "1.5"},
        ordinary_outcomes=OUTCOMES,
        max_total_sales_yen=1_000,
        batch_size=7,
    )

    assert result.total_face_units <= 100
    assert sum(result.ticket_stakes_yen.values()) == result.total_sales_yen
    expected = {"A": 25, "B": 37, "C": 15}
    for outcome, payout_per_face in expected.items():
        stake_units = result.ticket_stakes_yen[outcome] // 10
        assert max(10, result.distributable_pool_yen // stake_units) == payout_per_face


def test_allocation_and_hash_are_deterministic() -> None:
    options = dict(
        displayed_odds={"A": 2.5, "B": 3.7, "C": 1.5},
        ordinary_outcomes=OUTCOMES,
        max_total_sales_yen=1_000,
    )
    first = infer_minimum_pool_scale(**options)
    second = infer_minimum_pool_scale(**options)

    assert first == second
    assert len(first.allocation_sha256) == 64


def test_complete_120_way_trifecta_price_vector_is_reconstructed() -> None:
    known_units = {
        outcome: index + 1 for index, outcome in enumerate(TRIFECTA_OUTCOMES)
    }
    total_units = sum(known_units.values())
    distributable = total_units * 10 * 75 // 100
    displayed = {
        outcome: max(10, distributable // units) / 10
        for outcome, units in known_units.items()
    }

    result = infer_minimum_pool_scale(
        displayed,
        ordinary_outcomes=TRIFECTA_OUTCOMES,
        max_total_sales_yen=total_units * 10,
    )

    assert len(result.ticket_stakes_yen) == 120
    assert set(result.ticket_stakes_yen) == set(TRIFECTA_OUTCOMES)
    assert result.total_face_units <= total_units
    for outcome in TRIFECTA_OUTCOMES:
        inferred_units = result.ticket_stakes_yen[outcome] // 10
        assert max(
            10, result.distributable_pool_yen // inferred_units
        ) == int(displayed[outcome] * 10)


def test_missing_or_impossible_odds_fail_closed() -> None:
    with pytest.raises(ValueError, match="must match"):
        infer_minimum_pool_scale(
            {"A": 2.0, "B": 3.0}, ordinary_outcomes=OUTCOMES
        )
    with pytest.raises(ValueError, match="allow_unpriced"):
        infer_minimum_pool_scale(
            {"A": 2.0, "B": "--", "C": 3.0},
            ordinary_outcomes=OUTCOMES,
        )
    with pytest.raises(ValueError, match="no odds-consistent pool"):
        infer_minimum_pool_scale(
            {"A": 100.0, "B": 100.0, "C": 100.0},
            ordinary_outcomes=OUTCOMES,
            max_total_sales_yen=1_000,
        )


def test_explicit_unpriced_outcome_receives_zero_in_audit_allocation() -> None:
    result = infer_minimum_pool_scale(
        {"A": 1.5, "B": "--", "C": 1.5},
        ordinary_outcomes=OUTCOMES,
        allow_unpriced=True,
        max_total_sales_yen=1_000,
    )
    assert result.unpriced_outcomes == ("B",)
    assert result.ticket_stakes_yen["B"] == 0


def test_tokoname_official_record_sheet_arithmetic() -> None:
    total_face_units = 1_815_020
    winning_face_units = 53_650
    distributable_yen = total_face_units * 10 * 75 // 100

    assert distributable_yen == 13_612_650
    assert distributable_yen // winning_face_units == 253
    assert (distributable_yen // winning_face_units) * 10 == 2_530


def test_lower_bound_attaches_without_changing_generated_shares() -> None:
    shares = {"A": 0.2, "B": 0.3, "C": 0.5}
    paths = [[JointMarketScenario(
        probabilities={"A": 0.4, "B": 0.3, "C": 0.3},
        market_state={"final_market_shares": shares, "path": 7},
        weight=0.25,
    )]]

    attached, lower_bound = attach_pool_scale_lower_bound(
        paths,
        displayed_odds={"A": 2.5, "B": 3.7, "C": 1.5},
        ordinary_outcomes=OUTCOMES,
        odds_asof="T-5",
        max_total_sales_yen=1_000,
    )
    scenario = attached[0][0]

    assert scenario.market_state["final_market_shares"] is shares
    assert (
        scenario.market_state["external_total_sales_yen"]
        == lower_bound.total_sales_yen
    )
    assert scenario.market_state["external_pool_scale_method"] == POOL_SCALE_METHOD
    assert scenario.market_state["external_pool_scale_asof"] == "T-5"
    assert scenario.weight == 0.25
    settle = build_parimutuel_gross_payoff_model(ordinary_outcomes=OUTCOMES)
    assert set(settle(scenario, {"A": 100})) == {"A"}


def test_existing_absolute_scale_is_not_overwritten() -> None:
    paths = [[JointMarketScenario(
        probabilities={"A": 0.4, "B": 0.3, "C": 0.3},
        market_state={
            "final_market_shares": {"A": 0.2, "B": 0.3, "C": 0.5},
            "external_total_sales_yen": 10_000,
        },
    )]]
    with pytest.raises(ValueError, match="already attached"):
        attach_pool_scale_lower_bound(
            paths,
            displayed_odds={"A": 2.5, "B": 3.7, "C": 1.5},
            ordinary_outcomes=OUTCOMES,
            odds_asof="T-5",
        )
