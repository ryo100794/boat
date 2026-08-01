from __future__ import annotations

import pytest

from boatrace_ai.listwise.archive_market_oracle import (
    PRIMARY_POLICY,
    V23_TOP5_ORACLE_POLICY,
    narrow_ev_diagnostic_policies,
    restrict_probabilities_to_available,
    temporal_residual_diagnostic,
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


def test_v25_narrow_ev_diagnostic_grid_is_fixed_and_non_overlapping() -> None:
    policies = narrow_ev_diagnostic_policies()
    assert len(policies) == 15
    assert {policy["max_model_rank"] for policy in policies} == {1, 3, 5}
    assert {
        (policy["ev_threshold"], policy["max_estimated_ev"])
        for policy in policies
    } == {
        (0.95, 1.00), (1.00, 1.025), (1.025, 1.05),
        (1.05, 1.10), (1.10, 1.20),
    }
    assert all(policy["max_odds"] == 80.0 for policy in policies)


def _residual_race(race_date: str, actual: str) -> dict:
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:00:00+09:00",
        "actual_combination": actual,
        "actual_payout_yen": 200 if actual == "1-2-3" else 300,
        "odds": {"1-2-3": 2.0, "1-3-2": 3.0},
        "model_probabilities": {"1-2-3": 0.7, "1-3-2": 0.3},
        "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
    }


def test_temporal_residual_uses_strictly_earlier_calibration_days() -> None:
    races = [
        _residual_race("2026-01-01", "1-2-3"),
        _residual_race("2026-01-02", "1-3-2"),
        _residual_race("2026-01-03", "1-2-3"),
        _residual_race("2026-01-04", "1-2-3"),
    ]
    result = temporal_residual_diagnostic(
        races,
        calibration_through="2026-01-02",
    )
    assert result["status"] == "completed"
    assert result["calibration_from"] == "2026-01-01"
    assert result["calibration_through"] == "2026-01-02"
    assert result["evaluation_from"] == "2026-01-03"
    assert result["evaluation_through"] == "2026-01-04"
    assert result["calibration_races"] == 2
    assert result["evaluation_races"] == 2
    assert result["metrics"]["evaluated_races"] == 2
    assert len(result["purchase_diagnostics"]) == 4
    assert all("bootstrap" in row for row in result["purchase_diagnostics"])


def test_temporal_residual_requires_four_days() -> None:
    races = [_residual_race("2026-01-01", "1-2-3")]
    result = temporal_residual_diagnostic(races)
    assert result == {
        "status": "insufficient_days",
        "dates": 1,
        "calibration_days": 0,
        "evaluation_days": 0,
    }
