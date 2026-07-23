from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import (
    build_command,
    dedupe_key,
    prepare_standardized_workspace,
    result_decision,
    seed_default_jobs,
    summarize_result,
)


def _job(task_type: str, parameters: dict, *, job_id: int = 7) -> dict:
    return {
        "job_id": job_id,
        "task_type": task_type,
        "model_key": "candidate",
        "parameters": parameters,
    }


def test_dedupe_key_is_parameter_order_independent() -> None:
    assert dedupe_key("probe", "model", {"a": 1, "b": 2}) == dedupe_key(
        "probe", "model", {"b": 2, "a": 1}
    )


def test_market_curvature_command_uses_fixed_script_and_output(tmp_path) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "market_curvature",
            {"evaluation_date": "2026-07-22", "disagreement_clip": 2.0},
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1] == str(root / "scripts/analyze_market_curvature.py")
    assert command[-1] == str(root / "data/models/evaluation_queue/job-00000007.json")
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_task_parameters_cannot_select_arbitrary_command(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported task_type"):
        build_command(
            _job("shell", {"command": "rm -rf /"}),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_feature_search_rejects_unregistered_target(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported targets"):
        build_command(
            _job(
                "listwise_feature_search",
                {"targets": "future_result", "evaluation_date": "2026-07-22"},
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_result_summary_and_decision_use_nested_evaluation_metrics() -> None:
    payload = {
        "status": "candidate_requires_new_day_confirmation",
        "incremental_confidence_pass": True,
        "momentum_newton_residual": {
            "metrics": {
                "evaluated_races": 136,
                "trifecta_log_loss": 3.84,
                "trifecta_top5_hit_rate": 0.31,
            }
        },
    }

    summary = summarize_result(payload)

    assert summary["evaluated_races"] == 136
    assert summary["trifecta_log_loss"] == 3.84
    assert result_decision("market_curvature", summary) == "confirm_on_new_holdout"


def test_result_summary_preserves_paired_payout_feature_comparison() -> None:
    summary = summarize_result({
        "model": "venue",
        "venue_conditional_order": {
            "trifecta_log_loss": 3.79,
            "trifecta_top5_hit_rate": 0.35,
        },
        "payout_feature_comparison": {
            "candidate_schema": "conditional_payout_interactions_v2",
            "legacy_schema": "conditional_payout_additive_v1",
            "candidate_bankroll": {"roi": 1.03},
            "legacy_bankroll": {"roi": 0.90},
            "confidence": {
                "roi_delta": 0.13,
                "roi_delta_ci95_lower": 0.02,
                "roi_delta_ci95_upper": 0.24,
                "probability_roi_delta_above_zero": 0.99,
            },
            "gate": {
                "pass": True,
                "roi_ci95_lower": 1.01,
                "roi_delta_ci95_lower": 0.02,
                "roi_pass": True,
                "profit_pass": True,
                "baseline_improved": True,
            },
        },
    })

    assert summary["payout_feature_candidate_roi"] == 1.03
    assert summary["trifecta_log_loss"] == 3.79
    assert summary["trifecta_top5_hit_rate"] == 0.35
    assert summary["payout_feature_legacy_roi"] == 0.90
    assert summary["payout_feature_roi_delta_ci95_lower"] == 0.02
    assert summary["payout_feature_probability_roi_delta_above_zero"] == 0.99
    assert summary["payout_feature_promotion_eligible"] is True
    assert summary["payout_feature_gate_roi_ci95_lower"] == 1.01
    assert summary["payout_feature_candidate_schema"].endswith("v2")
    assert (
        result_decision("venue_conditional_order", summary)
        == "payout_feature_promotion_candidate"
    )


def test_default_seed_contains_parameter_sweep(monkeypatch) -> None:
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr("boatrace_ai.evaluation_queue.enqueue_job", fake_enqueue)

    inserted = seed_default_jobs(object(), evaluation_date="2026-07-22")

    assert len(inserted) == 11
    assert sum(row["task_type"] == "market_curvature" for row in calls) == 6
    assert sum(row["task_type"] == "listwise_feature_search" for row in calls) == 4
    assert all(
        row["parameters"]["evaluation_date"] == "2026-07-22" for row in calls
    )


def test_supervisor_runs_four_postgresql_queue_workers() -> None:
    config = Path(
        "scripts/deployment/supervisor-boatrace-evaluation-runner.ini"
    ).read_text(encoding="utf-8")

    assert "boatrace_ai.evaluation_queue run" in config
    assert "numprocs=4" in config
    assert "--seed-defaults" in config
    assert "--vm-limit-gib 0" in config


def test_standardized_workspace_rotates_stale_protocol_metadata(tmp_path) -> None:
    current = tmp_path / "data/models/standardized_365d_v2"
    current.mkdir(parents=True)
    (current / "protocol.json").write_text(
        '{"as_of_date_jst":"2026-07-20"}', encoding="utf-8"
    )
    (current / "manifest.json").write_text('{"ready":true}', encoding="utf-8")

    prepare_standardized_workspace(tmp_path, evaluation_date="2026-07-22")

    assert not (current / "protocol.json").exists()
    archive = tmp_path / "data/models/evaluation_queue/standardized_history/2026-07-20"
    assert (archive / "protocol.json").is_file()
    assert (archive / "manifest.json").is_file()
