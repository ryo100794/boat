from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from boatrace_ai.evaluation_queue import build_command, summarize_result

from boatrace_ai.joint_edge_calibrated_replay import (
    _fit_bets_to_cash,
    _purchase_gate_outcome,
    run_joint_edge_calibrated_replay,
)


def _base_artifact(path: Path, *, day3_payout: int = 200) -> Path:
    daily = []
    for day in range(1, 5):
        payout = day3_payout if day == 3 else 200
        daily.append({
            "race_date": f"2026-07-{day:02d}",
            "races": [{
                "race_id": f"race-{day}",
                "evaluation_time_t": (
                    f"2026-07-{day:02d}T10:00:00+09:00"
                ),
                "settlement_available_at": (
                    f"2026-07-{day:02d}T10:10:00+09:00"
                ),
                "actual_combination": "1-2-3",
                "actual_payout_yen": payout,
                "best_search_stake_yen": 100,
                "best_search_hypothetical_return_yen": payout,
                "best_search_edge_excess": 0.20,
                "best_search_growth_excess": 0.01,
                "best_search_constraint_violation": 0.0,
                "best_search_bets_yen": {"1-2-3": 100},
                "validation_uses_separate_draw_set": True,
                "portfolio_lower_quantile": 0.20,
                "purchase_value_gate_passed": True,
                "bankroll_growth_lower_quantile": 0.01,
                "pregate_candidate_generated": True,
                "best_search_validation_portfolio_lower_quantile": 0.20,
                "best_search_validation_purchase_value_gate_passed": True,
                "best_search_validation_bankroll_growth_lower_quantile": 0.01,
                "best_search_validation_growth_gate_passed": True,
            }],
        })
    payload = {
        "model": "joint_bankroll_strict_walk_forward_v5",
        "evaluation_protocol_id": "base-protocol",
        "evaluation_protocol": {
            "evaluation_time_t": {
                "definition": "purchase_decision_timestamp",
                "source_field": "decision_at_else_odds_deadline_at",
            },
            "odds_snapshot_age": {"definition": "decision_minus_snapshot"},
            "population": {"wager_types": ["trifecta"]},
            "training_and_joint_distribution": {
                "search_outer_draws": 20,
                "validation_outer_draws": 100,
                "search_validation_draw_sets_disjoint": True,
            },
            "purchase_rule": {"inner_tail_fraction": 0.1},
        },
        "configuration": {"buy_margin": 0.0},
        "daily": daily,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(path: Path) -> dict:
    return run_joint_edge_calibrated_replay(
        path,
        initial_daily_bankroll_yen=1_000,
        calibration_bootstrap_samples=100,
        calibration_min_training_days=2,
        calibration_min_portfolios=2,
        calibration_min_candidate_days=2,
        bootstrap_samples=100,
        seed=17,
    )


def test_replay_uses_only_strictly_prior_days_for_purchase_gate(
    tmp_path: Path,
) -> None:
    profitable = _run(_base_artifact(tmp_path / "profitable.json"))
    changed_current = _run(
        _base_artifact(tmp_path / "changed.json", day3_payout=0)
    )

    assert profitable["calibration_ready_days"] == 2
    first_ready = profitable["daily"][2]["races"][0]
    changed_first_ready = changed_current["daily"][2]["races"][0]
    assert first_ready["purchase_authorized"] is True
    assert first_ready["raw_value_source"] == (
        "pregate_best_search_independent_validation_"
        "portfolio_lower_quantile"
    )
    assert changed_first_ready["purchase_authorized"] is True
    assert first_ready["calibrated_gross_return_lcb95"] == (
        changed_first_ready["calibrated_gross_return_lcb95"]
    )
    assert first_ready["calibration_trained_through_date"] == "2026-07-02"
    assert first_ready["return_yen"] == 200
    assert changed_first_ready["return_yen"] == 0

    assert profitable["primary_bankroll"]["roi"] == 2.0
    assert profitable["evaluation_protocol"][
        "base_evaluation_protocol_id"
    ] == "base-protocol"
    assert profitable["deployment_eligible"] is False
    assert profitable["promotion_eligible"] is False
    assert profitable["promotion_gate"][
        "independent_validation_value_only"
    ] is True
    assert profitable["calibration_input_sources"] == {
        "pregate_best_search_independent_validation_"
        "portfolio_lower_quantile": 4
    }
    independence = profitable["calibration_independence_audit"]
    assert independence["strict_prior_fold_violations"] == 0
    assert independence["strict_prior_training_for_every_ready_fold"] is True
    assert independence["strict_settlement_fold_violations"] == 0
    assert independence[
        "settlement_before_decision_for_every_ready_fold"
    ] is True
    assert independence[
        "settlement_before_decision_for_every_ready_candidate"
    ] is True
    assert independence["candidate_settlement_boundary_violations"] == 0
    assert independence["ready_candidate_calibration_boundaries"] == 2
    assert len(independence["candidate_boundary_manifest_sha256"]) == 64
    assert all(
        race[
            "calibration_latest_settlement_strictly_before_"
            "evaluation_time_t"
        ] is True
        for day in profitable["daily"] if day["calibration_ready"]
        for race in day["races"]
    )
    assert independence["same_race_teacher_fold_violations"] == 0
    assert independence[
        "same_race_excluded_for_every_ready_fold"
    ] is True
    batch_audit = profitable[
        "same_race_calibrator_settlement_batch_audit"
    ]
    assert batch_audit["ticket_calibrator_instance_violations"] == 0
    assert batch_audit[
        "all_tickets_in_race_share_one_prior_calibrator"
    ] is True
    assert batch_audit[
        "teacher_admission_before_settlement_violations"
    ] == 0
    assert batch_audit[
        "results_admitted_only_after_strict_settlement"
    ] is True
    population = profitable["calibration_learning_population_audit"]
    assert population["candidate_portfolios"] == 4
    assert population["pregate_candidates_generated"] == 4
    assert population["pregate_candidates_registered"] == 4
    assert population["pregate_candidates_missing_independent_value"] == 0
    assert population["all_pregate_candidates_registered"] is True
    assert population["unique_races"] == 4
    assert population["outcome_filter"] == "none"
    assert population["purchase_filter"] == (
        "none_includes_purchased_and_rejected"
    )
    assert population["positive_return_portfolios"] + population[
        "zero_return_portfolios"
    ] == population["candidate_portfolios"]
    warmup = profitable["calibration_warmup_audit"]
    assert warmup["logical_operator"] == "AND"
    assert warmup["logic_violations"] == 0
    assert warmup["ready_exactly_when_all_thresholds_pass"] is True
    assert warmup["first_ready_boundary"]["evaluation_date"] == (
        "2026-07-03"
    )
    assert warmup["first_ready_boundary"]["training_days"] == 2
    assert warmup["first_ready_boundary"]["candidate_portfolios"] == 2
    assert warmup["first_ready_boundary"]["candidate_days"] == 2
    assert warmup["pre_ready_purchases"] == 0
    assert warmup["pre_ready_stake_yen"] == 0
    assert warmup["pre_ready_nonempty_bet_vectors"] == 0
    assert warmup["pre_ready_purchase_authorizations"] == 0
    assert warmup["no_purchases_before_ready"] is True
    update_audit = profitable["calibrator_update_audit"]
    assert update_audit["updates_after_initialization"] == 3
    assert update_audit["update_logic_violations"] == 0
    assert update_audit[
        "updates_only_when_eligible_teacher_population_changes"
    ] is True
    assert update_audit[
        "unchanged_population_reuses_identical_calibrator"
    ] is True
    assert update_audit[
        "every_decision_bound_to_full_prior_ledger_artifact"
    ] is True
    assert update_audit["missing_decision_calibrator_bindings"] == 0
    assert update_audit["instance_artifact_collisions"] == 0
    assert update_audit["instance_ledger_collisions"] == 0
    range_audit = profitable["calibration_input_range_audit"]
    assert range_audit["out_of_range_candidates"] == 0
    assert range_audit["out_of_range_purchase_violations"] == 0
    assert independence["search_validation_draw_sets_disjoint"] is True
    assert independence["value_population_independent_validation_only"] is True
    assert independence["value_population_identical_realized_portfolios_only"] is True
    calibration = profitable["purchase_value_realization_calibration"]
    assert calibration["version"] == (
        "strict_prior_calibrated_value_realization_deciles_v1"
    )
    assert calibration["candidate_portfolios"] == 2
    assert calibration["excluded_mismatched_portfolios"] == 0
    assert calibration["independent_validation_value_only"] is True
    assert calibration["identical_realized_portfolio_only"] is True
    assert len(calibration["candidate_manifest_sha256"]) == 64
    assert all(
        row["predicted_roi"] == 2.0
        and row["conservative_predicted_roi"] == 2.0
        and row["realized_roi"] == 2.0
        for row in calibration["deciles"]
    )
    assert profitable["evaluation_protocol"][
        "training_and_joint_distribution"
    ]["search_validation_draw_sets_disjoint"] is True
    calibration_protocol = profitable["evaluation_protocol"]["calibration"]
    assert calibration_protocol["target_unit"] == (
        "gross_return_per_staked_yen_including_returned_principal"
    )
    assert calibration_protocol["purchase_condition"] == (
        "calibrated_gross_return_lcb95_greater_than_"
        "one_plus_calibration_margin"
    )
    assert calibration_protocol["independent_sample_unit"] == (
        "one_stake_weighted_candidate_portfolio_per_race"
    )
    assert profitable["formal_purchase_value"]["value_unit"] == (
        "net_expected_edge_equals_gross_return_minus_one"
    )
    assert "minimum_30_calibration_ready_days" in (
        profitable["promotion_gate_failed"]
    )


def test_independent_validation_gate_is_required_after_calibration_ready(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "validation-reject.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["daily"][2]["races"][0]["purchase_value_gate_passed"] = False
    payload["daily"][2]["races"][0][
        "best_search_validation_purchase_value_gate_passed"
    ] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)
    rejected = result["daily"][2]["races"][0]
    assert rejected["calibrated_gross_return_lcb95"] > 1.0
    assert rejected["base_joint_gate_feasible"] is False
    assert rejected["purchase_authorized"] is False
    assert rejected["rejection_reason"] == "base_joint_gate_not_feasible"
    assert result["calibration_learning_population_audit"][
        "candidate_portfolios"
    ] == 4
    assert result["calibration_learning_population_audit"][
        "structurally_rejected_portfolios"
    ] == 1


@pytest.mark.parametrize("settlement_time", [
    "2026-07-03T10:00:00+09:00",
    "2026-07-03T11:00:00+09:00",
])
def test_calibration_excludes_candidates_not_strictly_settled_before_decision(
    tmp_path: Path,
    settlement_time: str,
) -> None:
    artifact = _base_artifact(tmp_path / "unsettled.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["daily"][1]["races"][0]["settlement_available_at"] = (
        settlement_time
    )
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)

    day3 = result["daily"][2]
    assert day3["calibration_information_cutoff"] == (
        "2026-07-03T10:00:00+09:00"
    )
    assert day3["settlement_eligible_training_records"] == 1
    assert day3["settlement_excluded_training_records"] == 1
    assert day3["newly_admitted_settled_race_batches"] == 0
    assert day3["pending_unsettled_race_batches"] == 2
    assert day3["teacher_population_changed"] is False
    assert day3["calibrator_instance_changed"] is False
    assert day3["calibrator_cache_hit"] is True
    assert day3["calibration_instance_id"] == result["daily"][1][
        "calibration_instance_id"
    ]
    assert day3["calibrator_artifact_sha256"] == result["daily"][1][
        "calibrator_artifact_sha256"
    ]
    assert day3["calibration_ready"] is False
    assert day3["stake_yen"] == 0
    assert result["daily"][3]["calibration_ready"] is True
    assert result["daily"][3][
        "newly_admitted_settled_race_batches"
    ] == 2
    assert result["daily"][3]["stake_yen"] == 100
    assert result["calibration_independence_audit"][
        "strict_settlement_fold_violations"
    ] == 0
    assert result["same_race_calibrator_settlement_batch_audit"][
        "teacher_admission_before_settlement_violations"
    ] == 0


def test_calibration_excludes_prior_record_from_same_race(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "same-race.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    repeated_race_id = payload["daily"][0]["races"][0]["race_id"]
    payload["daily"][2]["races"][0]["race_id"] = repeated_race_id
    payload["daily"] = payload["daily"][:3]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)

    day3 = result["daily"][2]
    race3 = day3["races"][0]
    assert day3["same_race_excluded_training_records"] == 1
    assert day3["same_race_teacher_overlap_count"] == 0
    assert day3["eligible_training_unique_races"] == 1
    assert day3["calibration_ready"] is False
    assert race3["calibration_same_race_teacher_overlap_count"] == 0
    assert result["calibration_independence_audit"][
        "same_race_teacher_fold_violations"
    ] == 0
    assert result["calibration_learning_population_audit"][
        "duplicate_race_result_batches_excluded"
    ] == 1


def test_duplicate_race_id_within_evaluation_fold_fails_closed(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "duplicate-race.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    duplicate = dict(payload["daily"][0]["races"][0])
    payload["daily"][0]["races"].append(duplicate)
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="evaluation fold contains duplicate race_id"
    ):
        _run(artifact)


def test_out_of_training_range_raw_value_is_never_purchased(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "out-of-range.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    race = payload["daily"][2]["races"][0]
    race["best_search_validation_portfolio_lower_quantile"] = 0.50
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)

    replayed = result["daily"][2]["races"][0]
    assert replayed["calibration_ready"] is True
    assert replayed["raw_portfolio_gross_return_estimate"] == 1.5
    assert replayed["calibration_training_raw_input_min"] == 1.2
    assert replayed["calibration_training_raw_input_max"] == 1.2
    assert replayed["calibration_input_in_training_range"] is False
    assert replayed["purchase_authorized"] is False
    assert replayed["stake_yen"] == 0
    assert replayed["rejection_reason"] == (
        "calibration_input_out_of_training_range"
    )
    audit = result["calibration_input_range_audit"]
    assert audit["out_of_range_candidates"] == 1
    assert audit["out_of_range_purchase_violations"] == 0
    assert audit["all_out_of_range_inputs_rejected"] is True


def test_later_same_day_decision_rebuilds_from_settled_prior_race(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "same-day-update.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    first = payload["daily"][2]["races"][0]
    second = dict(first)
    second["race_id"] = "race-3-later"
    second["evaluation_time_t"] = "2026-07-03T11:00:00+09:00"
    second["settlement_available_at"] = "2026-07-03T11:10:00+09:00"
    payload["daily"][2]["races"].append(second)
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)

    folds = [
        fold for fold in result["calibration_folds"]
        if fold["evaluation_date"] == "2026-07-03"
    ]
    assert len(folds) == 2
    early, later = folds
    assert early["settlement_eligible_training_records"] == 2
    assert later["settlement_eligible_training_records"] == 3
    assert later["newly_admitted_settled_race_batches"] == 1
    assert later["latest_training_settlement_available_at"] == (
        "2026-07-03T10:10:00+09:00"
    )
    assert later["strict_settlement_before_decision"] is True
    assert later["trained_through_date"] == "2026-07-03"
    assert later["teacher_population_changed"] is True
    assert later["calibrator_instance_changed"] is True
    assert later["calibration_instance_id"] != early[
        "calibration_instance_id"
    ]
    assert result["calibrator_update_audit"][
        "every_decision_bound_to_full_prior_ledger_artifact"
    ] is True


def test_all_tickets_in_race_share_one_frozen_prior_calibrator(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "multi-ticket.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    race = payload["daily"][2]["races"][0]
    race["best_search_bets_yen"] = {
        "1-2-3": 100,
        "1-3-2": 100,
    }
    race["best_search_stake_yen"] = 200
    race["best_search_hypothetical_return_yen"] = 200
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)

    replayed = result["daily"][2]["races"][0]
    assert replayed["counterfactual_ticket_count"] == 2
    assert replayed["ticket_calibration_instance_count"] == 1
    assert len(replayed["calibration_instance_id"]) == 64
    assert result["same_race_calibrator_settlement_batch_audit"][
        "ticket_calibrator_instance_violations"
    ] == 0


def test_zero_generated_candidates_is_complete_but_not_warm(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "zero-candidates.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    for day in payload["daily"]:
        race = day["races"][0]
        race["pregate_candidate_generated"] = False
        race["best_search_stake_yen"] = 0
        race["best_search_bets_yen"] = {}
        race["best_search_hypothetical_return_yen"] = 0
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)

    population = result["calibration_learning_population_audit"]
    assert population["pregate_candidates_generated"] == 0
    assert population["pregate_candidates_registered"] == 0
    assert population["all_pregate_candidates_registered"] is True
    assert result["calibration_warmup_audit"][
        "first_ready_boundary"
    ] is None
    assert result["primary_bankroll"]["stake_yen"] == 0
    update_audit = result["calibrator_update_audit"]
    assert update_audit["updates_after_initialization"] == 0
    assert update_audit["unchanged_population_reuses"] == 3
    assert update_audit["unique_calibrator_instances"] == 1
    assert update_audit["calibrator_fits"] == 1


def test_legacy_search_edge_is_explicit_fallback_and_cannot_promote(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "legacy.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    for day in payload["daily"]:
        race = day["races"][0]
        race["validation_uses_separate_draw_set"] = False
        race.pop("portfolio_lower_quantile")
        race.pop("purchase_value_gate_passed")
        race.pop("bankroll_growth_lower_quantile")
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)
    assert result["daily"][2]["races"][0]["raw_value_source"] == (
        "legacy_search_edge_fallback"
    )
    assert result["promotion_gate"][
        "independent_validation_value_only"
    ] is False
    assert result["promotion_eligible"] is False


def test_replay_no_bets_until_calibration_support_is_ready(
    tmp_path: Path,
) -> None:
    result = _run(_base_artifact(tmp_path / "base.json"))

    assert result["daily"][0]["stake_yen"] == 0
    assert result["daily"][1]["stake_yen"] == 0
    assert result["daily"][0]["races"][0]["rejection_reason"] == (
        "calibration_not_ready"
    )
    assert result["daily"][2]["stake_yen"] == 100
    assert result["primary_bankroll"]["selected_races"] == 2
    assert result["formal_purchase_value"]["minimum"] > 0.0
    assert result["calibration_folds"][2]["quantile_method"] == "inverted_cdf"
    assert result["calibration_folds"][2]["total_exposure_weight"] == 200.0
    assert result["evaluation_protocol"]["calibration"]["quantile_method"] == (
        "inverted_cdf"
    )
    assert result["evaluation_protocol"]["calibration"]["sample_weight"] == (
        "candidate_portfolio_stake_yen"
    )
    gate = result["purchase_gate_operational_audit"]
    assert gate["safety_invariants_passed"] is True
    assert gate["pre_calibration_ready_purchases"] == 0
    assert gate["below_calibrated_lcb_threshold_purchases"] == 0
    assert gate["non_independent_value_purchases"] == 0
    assert gate["outcome"] == "accumulating_strict_prior_calibration"


def test_mature_zero_purchase_window_is_safe_abstention_not_gate_failure() -> None:
    assert _purchase_gate_outcome(
        mature_observation_window=True,
        observed_purchased_portfolios=0,
        safety_invariants_passed=True,
        promotion_evidence_passed=False,
    ) == "safe_abstention_no_demonstrated_price_advantage"

    assert _purchase_gate_outcome(
        mature_observation_window=True,
        observed_purchased_portfolios=0,
        safety_invariants_passed=False,
        promotion_evidence_passed=False,
    ) == "formal_purchase_evidence_rejected"


def test_cash_downscale_preserves_integer_units_and_ticket_proportions() -> None:
    scaled = _fit_bets_to_cash(
        {"1-2-3": 500, "1-3-2": 300, "2-1-3": 200},
        600,
    )

    assert sum(scaled.values()) == 600
    assert all(value % 100 == 0 for value in scaled.values())
    assert scaled == {"1-2-3": 300, "1-3-2": 200, "2-1-3": 100}


def test_queue_builds_constrained_calibrated_replay_command(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "data/models/evaluation_queue"
    cache_dir = tmp_path / "data/models/evaluation_cache/market_scored"
    result_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    base = result_dir / "job-00000100.json"
    base.write_text("{}", encoding="utf-8")
    cache = cache_dir / "source.joblib"
    joblib.dump({"races": []}, cache)

    command, output = build_command(
        {
            "job_id": 101,
            "task_type": "joint_edge_calibrated_replay",
            "parameters": {
                "base_artifact": str(base.relative_to(tmp_path)),
                "scored_cache": str(cache.relative_to(tmp_path)),
                "calibration_min_training_days": 3,
                "calibration_min_portfolios": 200,
                "timeout_seconds": 3600,
            },
        },
        app_root=tmp_path,
        python=Path("/venv/bin/python"),
        db="postgresql://unused",
    )

    assert output == result_dir / "job-00000101.json"
    assert command[:3] == [
        "/venv/bin/python",
        "-m",
        "boatrace_ai.joint_edge_calibrated_replay",
    ]
    assert command[command.index("--base-artifact") + 1] == str(base)
    assert command[command.index("--scored-cache") + 1] == str(cache)
    assert command[
        command.index("--calibration-min-portfolios") + 1
    ] == "200"


def test_queue_defaults_require_mature_strict_prior_calibration(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "data/models/evaluation_queue"
    result_dir.mkdir(parents=True)
    base = result_dir / "job-00000100.json"
    base.write_text("{}", encoding="utf-8")

    command, _output = build_command(
        {
            "job_id": 102,
            "task_type": "joint_edge_calibrated_replay",
            "parameters": {"base_artifact": str(base.relative_to(tmp_path))},
        },
        app_root=tmp_path,
        python=Path("/venv/bin/python"),
        db="postgresql://unused",
    )

    assert command[
        command.index("--calibration-bootstrap-samples") + 1
    ] == "5000"
    assert command[
        command.index("--calibration-min-training-days") + 1
    ] == "30"
    assert command[
        command.index("--calibration-min-portfolios") + 1
    ] == "300"
    assert command[
        command.index("--calibration-min-candidate-days") + 1
    ] == "20"


def test_calibrated_replay_summary_exposes_formal_value_and_protocol() -> None:
    summary = summarize_result({
        "model": "joint_edge_calibrated_replay_v9",
        "evaluation_protocol_id": "calibrated-protocol",
        "evaluation_protocol": {
            "calibration": {
                "target_unit": (
                    "gross_return_per_staked_yen_including_returned_principal"
                ),
                "raw_input_unit": (
                    "gross_return_multiple_including_principal"
                ),
                "purchase_condition": (
                    "calibrated_gross_return_lcb95_greater_than_"
                    "one_plus_calibration_margin"
                ),
                "independent_sample_unit": (
                    "one_stake_weighted_candidate_portfolio_per_race"
                ),
            },
        },
        "evaluation_days": 5,
        "evaluated_races": 713,
        "calibration_ready_days": 2,
        "calibration_ready_races": 280,
        "primary_bankroll": {
            "stake_yen": 1_000,
            "return_yen": 1_200,
            "profit_yen": 200,
            "roi": 1.2,
            "daily_cluster_bootstrap_roi_lower_95": 1.01,
        },
        "formal_purchase_value": {
            "minimum": 1.03,
            "safety_margin": 1.0,
            "selected_portfolios": 4,
            "all_above_safety_margin": True,
            "value_unit": (
                "net_expected_edge_equals_gross_return_minus_one"
            ),
        },
        "calibration_independence_audit": {
            "strict_prior_fold_violations": 0,
            "strict_prior_training_for_every_ready_fold": True,
            "strict_settlement_fold_violations": 0,
            "settlement_before_decision_for_every_ready_fold": True,
            "ready_candidate_calibration_boundaries": 280,
            "candidate_settlement_boundary_violations": 0,
            "settlement_before_decision_for_every_ready_candidate": True,
            "candidate_boundary_manifest_sha256": "candidate-boundaries",
            "settlement_boundary_definition": (
                "candidate_settlement_available_at_strictly_before_"
                "earliest_evaluation_time_t_of_fold"
            ),
            "same_race_teacher_fold_violations": 0,
            "same_race_excluded_for_every_ready_fold": True,
            "same_race_rule": (
                "evaluation_race_id_must_not_appear_in_calibration_teacher_"
                "and_one_candidate_portfolio_per_race"
            ),
            "search_validation_draw_sets_disjoint": True,
            "value_population_manifest_sha256": "abc123",
            "value_population_independent_validation_only": True,
            "value_population_identical_realized_portfolios_only": True,
        },
        "purchase_gate_operational_audit": {
            "outcome": "safe_abstention_no_demonstrated_price_advantage",
            "safety_invariants_passed": True,
            "mature_observation_window": True,
            "safe_abstention": True,
            "pre_calibration_ready_purchases": 0,
            "below_calibrated_lcb_threshold_purchases": 0,
            "non_independent_value_purchases": 0,
        },
        "same_race_calibrator_settlement_batch_audit": {
            "ticket_calibrator_instance_violations": 0,
            "all_tickets_in_race_share_one_prior_calibrator": True,
            "teacher_admitted_race_batches": 390,
            "pending_unsettled_race_batches": 10,
            "teacher_admission_before_settlement_violations": 0,
            "results_admitted_only_after_strict_settlement": True,
        },
        "calibration_learning_population_audit": {
            "independent_sample_unit": (
                "one_fixed_counterfactual_portfolio_per_race"
            ),
            "inclusion_rule": "all_ex_ante_candidates",
            "outcome_filter": "none",
            "purchase_filter": "none_includes_purchased_and_rejected",
            "candidate_portfolios": 400,
            "pregate_candidates_generated": 400,
            "pregate_candidates_registered": 400,
            "pregate_candidates_missing_independent_value": 0,
            "all_pregate_candidates_registered": True,
            "unique_races": 400,
            "positive_return_portfolios": 40,
            "zero_return_portfolios": 360,
            "population_manifest_sha256": "population",
        },
        "calibration_warmup_audit": {
            "logical_operator": "AND",
            "minimum_training_calendar_days": 30,
            "minimum_pregate_candidate_portfolios": 300,
            "minimum_candidate_days": 20,
            "logic_violations": 0,
            "ready_exactly_when_all_thresholds_pass": True,
            "first_ready_boundary": {"evaluation_date": "2026-07-01"},
            "pre_ready_purchases": 0,
            "pre_ready_stake_yen": 0,
            "pre_ready_nonempty_bet_vectors": 0,
            "pre_ready_purchase_authorizations": 0,
            "no_purchases_before_ready": True,
        },
        "calibrator_update_audit": {
            "updates_after_initialization": 30,
            "unchanged_population_reuses": 5,
            "unique_calibrator_instances": 31,
            "calibrator_fits": 31,
            "update_logic_violations": 0,
            "unchanged_population_reuse_violations": 0,
            "updates_only_when_eligible_teacher_population_changes": True,
            "unchanged_population_reuses_identical_calibrator": True,
            "missing_decision_calibrator_bindings": 0,
            "instance_artifact_collisions": 0,
            "instance_ledger_collisions": 0,
            "every_decision_bound_to_full_prior_ledger_artifact": True,
        },
        "calibration_input_range_audit": {
            "ready_candidates_with_raw_input": 280,
            "out_of_range_candidates": 3,
            "out_of_range_purchase_violations": 0,
            "all_out_of_range_inputs_rejected": True,
        },
        "purchase_value_realization_calibration": {
            "version": "strict_prior_calibrated_value_realization_deciles_v1",
            "candidate_portfolios": 400,
            "excluded_mismatched_portfolios": 0,
            "monotone_realized_roi": True,
            "deciles": [{"decile": 1, "candidate_portfolios": 40}],
        },
        "bankroll_confidence": {
            "formal_gate_passed": True,
            "condition_id": "resampling",
            "condition": {
                "primary_block": "complete_operating_day",
                "quantile_method": "inverted_cdf",
                "samples": 2000,
            },
            "sensitivity": {
                "day_venue": {"roi_lower": 0.98},
                "venue_meeting": {"roi_lower": 0.97},
            },
        },
    })

    assert summary["evaluation_protocol_id"] == "calibrated-protocol"
    assert summary["joint_purchase_value_minimum"] == 1.03
    assert summary["joint_purchase_value_gate_passed"] is True
    assert summary["formal_roi_gate_passed"] is True
    assert summary["calibration_strict_prior_fold_violations"] == 0
    assert summary["calibration_strict_prior_all_ready_folds"] is True
    assert summary["calibration_strict_settlement_fold_violations"] == 0
    assert summary[
        "calibration_settlement_before_decision_all_ready_folds"
    ] is True
    assert summary[
        "calibration_settlement_before_decision_all_ready_candidates"
    ] is True
    assert summary[
        "calibration_candidate_settlement_boundary_violations"
    ] == 0
    assert summary["calibration_ready_candidate_boundaries"] == 280
    assert summary[
        "calibration_same_race_teacher_fold_violations"
    ] == 0
    assert summary[
        "calibration_same_race_excluded_all_ready_folds"
    ] is True
    assert summary["calibration_independent_sample_unit"] == (
        "one_stake_weighted_candidate_portfolio_per_race"
    )
    assert summary[
        "calibration_same_race_ticket_calibrator_violations"
    ] == 0
    assert summary[
        "calibration_same_prior_for_all_tickets_in_race"
    ] is True
    assert summary[
        "calibration_teacher_admission_before_settlement_violations"
    ] == 0
    assert summary[
        "calibration_results_admitted_only_after_settlement"
    ] is True
    assert summary[
        "calibration_learning_population_candidate_portfolios"
    ] == 400
    assert summary["calibration_pregate_candidates_generated"] == 400
    assert summary["calibration_pregate_candidates_registered"] == 400
    assert summary["calibration_all_pregate_candidates_registered"] is True
    assert summary[
        "calibration_learning_population_unique_races"
    ] == 400
    assert summary[
        "calibration_learning_population_outcome_filter"
    ] == "none"
    assert summary["calibration_warmup_logical_operator"] == "AND"
    assert summary["calibration_warmup_logic_violations"] == 0
    assert summary["calibration_warmup_conjunction_consistent"] is True
    assert summary["calibration_warmup_pre_ready_purchases"] == 0
    assert summary["calibration_warmup_pre_ready_stake_yen"] == 0
    assert summary["calibration_warmup_pre_ready_nonempty_bets"] == 0
    assert summary["calibration_warmup_pre_ready_authorizations"] == 0
    assert summary["calibration_warmup_no_purchases_before_ready"] is True
    assert summary["calibrator_update_logic_violations"] == 0
    assert summary["calibrator_updates_only_on_teacher_change"] is True
    assert summary[
        "calibrator_reuses_identical_instance_when_unchanged"
    ] is True
    assert summary[
        "calibrator_every_decision_bound_to_prior_ledger_artifact"
    ] is True
    assert summary[
        "calibration_input_range_out_of_range_candidates"
    ] == 3
    assert summary["calibration_input_range_purchase_violations"] == 0
    assert summary["calibration_input_range_all_rejected"] is True
    assert summary["calibration_target_unit"] == (
        "gross_return_per_staked_yen_including_returned_principal"
    )
    assert summary["calibration_raw_input_unit"] == (
        "gross_return_multiple_including_principal"
    )
    assert summary["calibration_purchase_condition"] == (
        "calibrated_gross_return_lcb95_greater_than_"
        "one_plus_calibration_margin"
    )
    assert summary["formal_purchase_value_unit"] == (
        "net_expected_edge_equals_gross_return_minus_one"
    )
    assert summary["calibration_search_validation_draw_sets_disjoint"] is True
    assert summary["calibration_value_population_manifest_sha256"] == "abc123"
    assert summary["calibration_value_population_independent_only"] is True
    assert summary["calibration_value_population_identical_only"] is True
    assert summary["purchase_gate_operational_outcome"] == (
        "safe_abstention_no_demonstrated_price_advantage"
    )
    assert summary["purchase_gate_safety_invariants_passed"] is True
    assert summary["purchase_gate_mature_observation_window"] is True
    assert summary["purchase_gate_safe_abstention"] is True
    assert summary["purchase_gate_pre_ready_purchases"] == 0
    assert summary["purchase_gate_below_lcb_purchases"] == 0
    assert summary["purchase_gate_non_independent_purchases"] == 0
    assert summary["purchase_value_realization_candidate_portfolios"] == 400
    assert summary["purchase_value_realization_deciles"] == [
        {"decile": 1, "candidate_portfolios": 40}
    ]
    assert summary["bootstrap_condition_id"] == "resampling"
    assert summary["day_venue_roi_lower_95"] == 0.98
