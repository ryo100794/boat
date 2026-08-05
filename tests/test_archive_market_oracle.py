from __future__ import annotations

import pytest

from boatrace_ai.listwise.archive_market_oracle import (
    EVALUATION_VERSION,
    PRIMARY_POLICY,
    V23_TOP5_ORACLE_POLICY,
    narrow_ev_diagnostic_policies,
    restrict_probabilities_to_available,
    temporal_residual_diagnostic,
)

def test_v33_archive_protocol_uses_new_evaluation_version() -> None:
    assert EVALUATION_VERSION == 21


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

def test_targeted_mature_value_reuses_the_sealed_split_without_other_refits(
    monkeypatch,
) -> None:
    races = [
        _residual_race("2026-01-01", "1-2-3"),
        _residual_race("2026-01-02", "1-3-2"),
        _residual_race("2026-01-03", "1-2-3"),
        _residual_race("2026-01-04", "1-2-3"),
    ]
    calls: list[tuple[int, int, int]] = []

    def fake_mature(calibration, evaluation, *, daily_budget_yen, num_threads=4):
        del num_threads
        calls.append((len(calibration), len(evaluation), daily_budget_yen))
        return {
            "model": "mature_stacked_contextual_value_rank20",
            "status": "completed",
            "evaluation_probability_metrics": {
                "evaluated_races": len(evaluation),
                "trifecta_log_loss": 3.5,
                "market_trifecta_log_loss": 3.6,
            },
            "probability_artifact": {"artifact_sha256": "a" * 64},
            "probability_selection": {
                "selected_stack": "linear50_nonlinear50",
                "selected_weights": {"linear": 0.5, "nonlinear": 0.5},
            },
            "bankroll": {"stake_yen": 0},
            "promotion_eligible": False,
        }

    monkeypatch.setattr(
        "boatrace_ai.listwise.archive_market_oracle.evaluate_mature_stacked_value",
        fake_mature,
    )
    result = temporal_residual_diagnostic(
        races,
        calibration_through="2026-01-02",
        daily_budget_yen=12_000,
        temporal_component="mature_stacked_contextual_value",
    )

    assert calls == [(2, 2, 12_000)]
    assert result["targeted_temporal_component"] == (
        "mature_stacked_contextual_value"
    )
    assert result["calibration_through"] == "2026-01-02"
    assert result["evaluation_from"] == "2026-01-03"
    assert result["stacked_market_residual_v42"]["metrics"][
        "evaluated_races"
    ] == 2
    assert result["mature_stacked_contextual_value"]["status"] == "completed"
    assert "ticket_utility_calibration_aligned_v33" not in result


def test_targeted_temporal_component_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown temporal component"):
        temporal_residual_diagnostic(
            [_residual_race("2026-01-01", "1-2-3")],
            temporal_component="unknown",
        )
