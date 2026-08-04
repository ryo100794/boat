from __future__ import annotations

import json
from pathlib import Path

import joblib

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
        "independent_validation_portfolio_lower_quantile"
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
        "independent_validation_portfolio_lower_quantile": 4
    }
    independence = profitable["calibration_independence_audit"]
    assert independence["strict_prior_fold_violations"] == 0
    assert independence["strict_prior_training_for_every_ready_fold"] is True
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
    assert "minimum_30_calibration_ready_days" in (
        profitable["promotion_gate_failed"]
    )


def test_independent_validation_gate_is_required_after_calibration_ready(
    tmp_path: Path,
) -> None:
    artifact = _base_artifact(tmp_path / "validation-reject.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["daily"][2]["races"][0]["purchase_value_gate_passed"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(artifact)
    rejected = result["daily"][2]["races"][0]
    assert rejected["calibrated_gross_return_lcb95"] > 1.0
    assert rejected["base_joint_gate_feasible"] is False
    assert rejected["purchase_authorized"] is False
    assert rejected["rejection_reason"] == "base_joint_gate_not_feasible"


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
        "model": "joint_edge_calibrated_replay_v3",
        "evaluation_protocol_id": "calibrated-protocol",
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
        },
        "calibration_independence_audit": {
            "strict_prior_fold_violations": 0,
            "strict_prior_training_for_every_ready_fold": True,
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
