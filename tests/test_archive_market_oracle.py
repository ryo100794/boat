from __future__ import annotations

import pytest

from boatrace_ai.listwise.archive_market_oracle import (
    PRIMARY_POLICY,
    V23_TOP5_ORACLE_POLICY,
    restrict_probabilities_to_available,
)


def test_restrict_probabilities_renormalizes_after_withdrawal() -> None:
    probabilities = {"1-2-3": 0.2, "1-2-4": 0.3, "1-2-5": 0.5}
    restricted = restrict_probabilities_to_available(
        probabilities, {"1-2-3", "1-2-5"}
    )
    assert restricted == pytest.approx({"1-2-3": 2 / 7, "1-2-5": 5 / 7})
    assert sum(restricted.values()) == pytest.approx(1.0)


def test_restrict_probabilities_rejects_uncovered_market() -> None:
    with pytest.raises(ValueError, match="do not cover"):
        restrict_probabilities_to_available({"1-2-3": 1.0}, {"1-2-4"})


def test_primary_oracle_policy_is_fixed_and_conservative() -> None:
    assert PRIMARY_POLICY["ev_threshold"] == 1.05
    assert PRIMARY_POLICY["max_estimated_ev"] == 1.20
    assert PRIMARY_POLICY["max_tickets_per_race"] == 3
    assert PRIMARY_POLICY["staking_mode"] == "kelly_025"


def test_v23_top5_oracle_policy_matches_registered_band() -> None:
    assert V23_TOP5_ORACLE_POLICY["max_model_rank"] == 5
    assert V23_TOP5_ORACLE_POLICY["ev_threshold"] == 1.0
    assert V23_TOP5_ORACLE_POLICY["max_estimated_ev"] == 1.05
    assert V23_TOP5_ORACLE_POLICY["stake_per_ticket_yen"] == 100
