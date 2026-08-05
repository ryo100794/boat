from __future__ import annotations

from datetime import date, timedelta

import pytest

from boatrace_ai.listwise import decision_v38_empirical_lcb as subject


def _race(day: date, index: int) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-01-{index:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 300,
        "odds": {"1-2-3": 3.0, "1-3-2": 4.0},
        "model_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "snapshot_id": index,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:01:00+09:00",
        "input_snapshot_age_seconds": 60.0,
    }


def _frozen() -> dict:
    return {
        "model": "decision_time_nonlinear_market_residual_v38",
        "training_status": "ready",
        "training_through": "2026-01-01",
        "official_closing_fields_used": False,
        "source_scored_cache_sha256": "a" * 64,
        "artifact": {"booster_sha256": "b" * 64},
    }


def test_v39_scores_frozen_v44_stack(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        subject,
        "stacked_probabilities",
        lambda race, artifact: calls.append((race, artifact)) or {
            "1-2-3": 0.7,
            "1-3-2": 0.3,
        },
    )
    frozen = {
        **_frozen(),
        "model": "decision_time_stacked_market_residual_v44",
        "artifact": {"artifact_sha256": "c" * 64},
    }
    scored = subject.score_frozen_v38_races(
        [_race(date(2026, 1, 2), 1)], frozen
    )
    assert scored[0]["model_probabilities"]["1-2-3"] == 0.7
    assert calls[0][1] == frozen["artifact"]


def test_v39_ledger_is_strictly_after_model_and_before_each_fold(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "nonlinear_residual_probabilities",
        lambda race, _artifact, *, shrinkage: dict(
            race["market_probabilities"]
        ),
    )
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=index), index) for index in range(6)]
    result = subject.walk_forward_decision_v38_lcb(
        races,
        _frozen(),
        registered_after="2026-01-03",
        minimum_ledger_days=2,
        minimum_ledger_candidates=2,
        minimum_ledger_candidate_days=2,
        minimum_local_candidates=1,
        minimum_local_candidate_days=1,
        minimum_local_ess=1.0,
        bootstrap_samples=100,
    )

    assert result["ledger_candidates"] == 6
    assert result["bankroll"]["evaluation_days"] == 3
    assert all(row["strict_prior_check"] for row in result["fold_audit"])
    assert result["fold_audit"][0]["prior_candidates"] == 0
    assert result["fold_audit"][0]["calibration_cutoff_date"] is None
    assert result["fold_audit"][1]["calibration_cutoff_date"] == "2026-01-04"
    assert result["fold_audit"][1]["max_training_settlement_date"] == (
        "2026-01-04"
    )
    assert len(result["fold_audit"][1]["decision_contract_hash"]) == 64
    assert result["fold_audit"][1]["frozen_model_hash"] == "a" * 64
    assert result["fold_audit"][1]["settlement_engine_hash"] == (
        result["settlement_engine_hash"]
    )
    assert result["fold_audit"][0]["stake_yen"] == 0
    assert result["fold_audit"][0]["candidate_decisions"] == 2
    assert result["fold_audit"][0]["denial_reason_counts"] == {
        "WARMUP_CALENDAR_DAYS": 2
    }
    assert result["fold_audit"][0]["buy_threshold"] == 1.0
    assert result["candidate_population"] == (
        "all_probability_top5_before_purchase_gate"
    )
    assert result["ticket_level_independence_assumed"] is False
    assert result["real_betting_enabled"] is False
    assert result["strict_prior_violation_count"] == 0
    assert result["same_race_calibrator_hash_count_max"] == 1
    first_decision = result["candidate_decision_audit"][0]
    assert first_decision["decision_time"].endswith("+09:00")
    assert first_decision["max_prior_settlement_time"] is None
    assert first_decision["candidate_population"] == (
        "all_pregate_probability_ranked"
    )
    assert first_decision["required_calendar_days"] == 2
    assert result["formal_purchase_rule"]["raw_v_buy_is_diagnostic_only"] is True
    assert result["formal_promotion_rule"][
        "probability_roi_above_one_is_diagnostic_only"
    ] is True
    assert result["formal_promotion_gate"]["positive_profit"] is False


def test_v45_can_predeclare_a_wider_or_narrower_candidate_rank(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "nonlinear_residual_probabilities",
        lambda race, _artifact, *, shrinkage: dict(
            race["market_probabilities"]
        ),
    )
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=index), index) for index in range(4)]

    result = subject.walk_forward_decision_v38_lcb(
        races,
        _frozen(),
        registered_after="2026-01-01",
        purchase_max_probability_rank=1,
        minimum_ledger_days=30,
        minimum_ledger_candidates=300,
        minimum_ledger_candidate_days=20,
        bootstrap_samples=100,
    )

    assert result["ledger_candidates"] == 3
    assert result["purchase_max_probability_rank"] == 1
    assert result["candidate_population"] == (
        "all_probability_top1_before_purchase_gate"
    )
    assert all(
        row["probability_rank"] == 1
        for row in result["candidate_decision_audit"]
    )


def test_v39_zero_stake_roi_is_na_not_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        subject,
        "nonlinear_residual_probabilities",
        lambda race, _artifact, *, shrinkage: dict(
            race["market_probabilities"]
        ),
    )
    races = [_race(date(2026, 1, 2), 1), _race(date(2026, 1, 3), 2)]
    result = subject.walk_forward_decision_v38_lcb(
        races,
        _frozen(),
        registered_after="2026-01-01",
        minimum_ledger_days=30,
        minimum_ledger_candidates=300,
        minimum_ledger_candidate_days=20,
        bootstrap_samples=100,
    )

    assert result["bankroll"]["stake_yen"] == 0
    assert result["bankroll"]["roi"] is None
    assert result["bankroll"]["roi_display"] == "N/A"
    assert result["promotion_eligible"] is False


def test_v45_calendar_warmup_is_separate_from_candidate_days(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "nonlinear_residual_probabilities",
        lambda race, _artifact, *, shrinkage: dict(
            race["market_probabilities"]
        ),
    )
    races = [
        _race(date(2026, 1, 2), 1),
        _race(date(2026, 1, 3), 2),
        _race(date(2026, 1, 4), 3),
    ]
    result = subject.walk_forward_decision_v38_lcb(
        races,
        _frozen(),
        registered_after="2026-01-01",
        minimum_ledger_days=30,
        minimum_ledger_candidates=2,
        minimum_ledger_candidate_days=2,
        minimum_local_candidates=1,
        minimum_local_candidate_days=1,
        minimum_local_ess=1.0,
        bootstrap_samples=100,
    )

    latest = result["fold_audit"][-1]
    assert latest["prior_candidate_days"] == 2
    assert latest["prior_calendar_span_days"] == 2
    assert latest["calibration_ready"] is False
    assert "insufficient_calendar_span_days" in latest["ready_reasons"]
    assert latest["authorized_tickets"] == 0
    assert latest["denial_reason_counts"] == {
        "WARMUP_CALENDAR_DAYS": 2
    }


def test_v39_refuses_registration_before_frozen_training() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        subject.walk_forward_decision_v38_lcb(
            [],
            _frozen(),
            registered_after="2025-12-31",
            bootstrap_samples=100,
        )


def test_v39_refuses_reuse_of_challenger_selection_period() -> None:
    frozen = {**_frozen(), "evaluation_through": "2026-01-07"}
    with pytest.raises(ValueError, match="selection data"):
        subject.walk_forward_decision_v38_lcb(
            [],
            frozen,
            registered_after="2026-01-06",
            bootstrap_samples=100,
        )


@pytest.mark.parametrize("maximum_rank", [0, 121])
def test_v45_refuses_invalid_candidate_rank(maximum_rank: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 120"):
        subject.walk_forward_decision_v38_lcb(
            [],
            _frozen(),
            registered_after="2026-01-01",
            purchase_max_probability_rank=maximum_rank,
            bootstrap_samples=100,
        )
