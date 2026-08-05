from __future__ import annotations

import joblib
import pytest

from boatrace_ai.archive_closing_odds import (
    OFFICIAL_SOURCE_KEY,
    ensure_archive_schema,
    record_attempt,
)
from boatrace_ai.db import connection, init_db, upsert_race
from boatrace_ai.listwise.archive_market_oracle import (
    EVALUATION_VERSION,
    PRIMARY_POLICY,
    V23_TOP5_ORACLE_POLICY,
    externalize_targeted_mature_evidence,
    narrow_ev_diagnostic_policies,
    official_archive_coverage,
    restrict_probabilities_to_available,
    temporal_residual_diagnostic,
)

def test_v33_archive_protocol_uses_new_evaluation_version() -> None:
    assert EVALUATION_VERSION == 23


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


def test_targeted_mature_evidence_is_externalized_without_loss(
    tmp_path,
) -> None:
    result = {
        "temporal_residual_diagnostic": {
            "stacked_market_residual_v42": {
                "artifact": {"artifact_sha256": "a" * 64, "booster": "full"},
            },
            "mature_stacked_contextual_value": {
                "model": "mature_stacked_contextual_value_rank20",
                "probability_artifact": {
                    "artifact_sha256": "a" * 64,
                    "booster": "full",
                },
                "bankroll": {
                    "tickets": 0,
                    "daily": [{
                        "race_date": "2026-07-01",
                        "stake_yen": 0,
                        "candidate_decision_audit": [{"race_id": "r1"}],
                        "eligible_candidate_audit": [{"race_id": "r2"}],
                        "selected_sample": [{"race_id": "r3"}],
                    }],
                },
                "context_value_audit": {"status": "completed"},
            },
        },
    }

    compact = externalize_targeted_mature_evidence(
        result, tmp_path / "job.json"
    )

    sidecar = compact["research_sidecar"]
    assert len(sidecar["sha256"]) == 64
    assert sidecar["candidate_decision_count"] == 1
    assert sidecar["eligible_candidate_count"] == 1
    loaded = joblib.load(sidecar["path"])
    full = loaded["mature_stacked_contextual_value"]
    assert full["probability_artifact"]["booster"] == "full"
    assert full["bankroll"]["daily"][0]["candidate_decision_audit"] == [
        {"race_id": "r1"}
    ]
    mature = compact["temporal_residual_diagnostic"][
        "mature_stacked_contextual_value"
    ]
    assert mature["probability_artifact"]["externalized"] is True
    assert mature["context_value_audit"]["status"] == "completed"
    assert "candidate_decision_audit" not in mature["bankroll"]["daily"][0]
    assert "eligible_candidate_audit" not in mature["bankroll"]["daily"][0]
    assert "selected_sample" not in mature["bankroll"]["daily"][0]


def test_targeted_temporal_component_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown temporal component"):
        temporal_residual_diagnostic(
            [_residual_race("2026-01-01", "1-2-3")],
            temporal_component="unknown",
        )

def test_official_coverage_uses_all_targets_and_publishes_missing_reasons(
    tmp_path,
) -> None:
    db_path = tmp_path / "official-coverage.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        ensure_archive_schema(conn)
        race_ids = []
        for rno in range(1, 5):
            race_id = upsert_race(
                conn,
                {
                    "race_date": f"2026-01-0{rno}",
                    "jcd": "01",
                    "venue_name": "桐生",
                    "rno": rno,
                    "status": "final",
                },
            )
            race_ids.append(race_id)
            conn.execute(
                "INSERT INTO payouts("
                "race_id, bet_type, combination, payout_yen"
                ") VALUES (?, '3連単', '1-2-3', 1230)",
                (race_id,),
            )
        conn.execute(
            """
            INSERT INTO archive_closing_odds_snapshots(
              race_id, source_key, fetched_at, source_url, payload_sha256,
              parser_version, odds_count, verification_status, raw_json
            ) VALUES (?, ?, '2026-02-01T00:00:00+00:00', 'https://example.test',
                      ?, 'test', 120,
                      'official_primary_winner_payout_match', '{}')
            """,
            (race_ids[0], OFFICIAL_SOURCE_KEY, "a" * 64),
        )
        record_attempt(
            conn,
            race_id=race_ids[1],
            status="excluded_non_six_boat",
            source_key=OFFICIAL_SOURCE_KEY,
        )
        record_attempt(
            conn,
            race_id=race_ids[2],
            status="invalid",
            error="incomplete",
            source_key=OFFICIAL_SOURCE_KEY,
        )
        record_attempt(
            conn,
            race_id=race_ids[3],
            status="fetch_error",
            error="timeout",
            source_key=OFFICIAL_SOURCE_KEY,
        )
        conn.commit()

        coverage = official_archive_coverage(
            conn,
            from_date="2026-01-01",
            through_date="2026-01-31",
        )

    assert coverage["official_eligible_target_races"] == 4
    assert coverage["official_excluded_non_six_boat_races"] == 1
    assert coverage["official_expected_six_boat_races"] == 3
    assert coverage["official_snapshot_races"] == 1
    assert coverage["official_unresolved_races"] == 2
    assert coverage["official_invalid_attempt_races"] == 1
    assert coverage["official_fetch_failure_attempt_races"] == 1
    assert coverage["official_coverage_ratio"] == pytest.approx(1 / 3)
    assert coverage["official_minimum_required_coverage"] == 0.995
    assert coverage["official_coverage_ready"] is False
    assert coverage["official_monthly_coverage"] == [
        {
            "month": "2026-01",
            "eligible_target_races": 4,
            "excluded_non_six_boat_races": 1,
            "expected_six_boat_races": 3,
            "official_snapshot_races": 1,
            "unresolved_races": 2,
            "invalid_attempt_races": 1,
            "fetch_failure_attempt_races": 1,
            "coverage_ratio": pytest.approx(1 / 3),
            "coverage_ready": False,
        }
    ]
