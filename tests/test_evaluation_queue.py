from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import (
    TASK_PROFILES,
    build_command,
    dedupe_key,
    prepare_standardized_workspace,
    result_decision,
    seed_default_jobs,
    summarize_result,
)
from boatrace_ai.listwise.conditional_order import (
    build_parser as conditional_parser,
)
from boatrace_ai.listwise.venue_conditional_order import (
    build_parser as venue_conditional_parser,
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


def test_calibrated_mlp_recency_search_profile() -> None:
    assert TASK_PROFILES["calibrated_mlp_recency_search"] == {
        "category": "evaluation",
        "memory_mb": 16384,
        "disk_mb": 4096,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }


def test_conditional_payout_tail_profile_and_command_are_fixed(tmp_path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job(
            "conditional_payout_tail",
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "timeout_seconds": 3600,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    result = root / "data/models/evaluation_queue/job-00000007.json"
    assert TASK_PROFILES["conditional_payout_tail"] == {
        "category": "evaluation",
        "memory_mb": 12288,
        "disk_mb": 2048,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }
    assert command == [
        str(python),
        "-m",
        "boatrace_ai.listwise.conditional_order",
        "--db",
        "postgresql://test",
        "--cache-prefix",
        str(
            root
            / "data/models/standardized_365d_v2/listwise_search_cache"
            / "listwise_search_4096_drop_research_correlates"
        ),
        "--baseline-model",
        str(root / "data/models/standardized_365d_v2/listwise_newton.joblib"),
        "--training-through",
        "2025-07-19",
        "--evaluation-from",
        "2025-07-20",
        "--evaluation-through",
        "2026-07-19",
        "--model-output",
        str(result.with_suffix(".joblib")),
        "--output",
        str(result),
        "--validation-days",
        "90",
        "--batch-races",
        "4000",
        "--payout-mean-corrections",
        "0.0",
        "--promote-legacy-cache",
    ]
    assert output == result


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-21",
                "evaluation_through": "2026-07-19",
            },
            "adjacent",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2025-07-19",
            },
            "chronological",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-18",
            },
            "exactly 365 days",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "timeout_seconds": 299,
            },
            "timeout_seconds",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "command": "rm -rf /",
            },
            "unsupported",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "cache_prefix": "/tmp/untrusted",
            },
            "unsupported",
        ),
    ],
)
def test_conditional_payout_tail_rejects_invalid_parameters(
    tmp_path, parameters, message
) -> None:
    with pytest.raises(ValueError, match=message):
        build_command(
            _job("conditional_payout_tail", parameters),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_conditional_payout_mean_correction_defaults_disable_double_correction() -> None:
    conditional = conditional_parser().parse_args(
        [
            "--cache-prefix", "cache",
            "--baseline-model", "baseline.joblib",
            "--training-through", "2025-07-19",
            "--evaluation-from", "2025-07-20",
            "--evaluation-through", "2026-07-19",
            "--model-output", "model.joblib",
            "--output", "result.json",
        ]
    )
    venue = venue_conditional_parser().parse_args(
        [
            "--baseline-model", "baseline.joblib",
            "--training-through", "2025-07-19",
            "--evaluation-from", "2025-07-20",
            "--evaluation-through", "2026-07-19",
            "--model-output", "model.joblib",
            "--output", "result.json",
        ]
    )

    assert conditional.payout_mean_corrections == [0.0]
    assert venue.payout_mean_corrections == [0.0]


def test_calibrated_mlp_recency_search_command_is_fixed(tmp_path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job(
            "calibrated_mlp_recency_search",
            {
                "evaluation_date": "2026-07-22",
                "half_lives": "none,180,180.0,365",
                "calibration_days": 120,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command == [
        str(python),
        "-m",
        "boatrace_ai.recency_mlp_evaluation",
        "--db",
        "postgresql://test",
        "--output",
        str(root / "data/models/evaluation_queue/job-00000007.json"),
        "--evaluation-date",
        "2026-07-22",
        "--feature-cache",
        str(root / "data/models/calibrated_shadow_features_16384"),
        "--half-lives",
        "none,180,365",
        "--calibration-days",
        "120",
    ]
    assert output == root / "data/models/evaluation_queue/job-00000007.json"

    default_command, _ = build_command(
        _job("calibrated_mlp_recency_search", {"evaluation_date": "2026-07-22"}),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert default_command[-4:] == [
        "--half-lives",
        "none,180,365,730",
        "--calibration-days",
        "180",
    ]


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "evaluation_date is required"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none"}, "at least 2"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none,29"}, "finite numbers"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none,nan"}, "finite numbers"),
        ({"evaluation_date": "2026-07-22", "calibration_days": 29}, "calibration_days"),
        ({"evaluation_date": "2026-07-22", "timeout_seconds": 299}, "timeout_seconds"),
        ({"evaluation_date": "2026-07-22", "command": "rm -rf /"}, "unsupported"),
        ({"evaluation_date": "2026-07-22", "feature_cache": "/tmp/cache"}, "unsupported"),
    ],
)
def test_calibrated_mlp_recency_search_rejects_invalid_parameters(
    tmp_path, parameters, message
) -> None:
    with pytest.raises(ValueError, match=message):
        build_command(
            _job("calibrated_mlp_recency_search", parameters),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


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


def test_conditional_payout_tail_summary_respects_explicit_non_promotion() -> None:
    summary = summarize_result({
        "promotion_eligible": True,
        "roi": 1.50,
        "profit_yen": 50_000,
        "conditional_payout_walk_forward": {
            "promotion_eligible": False,
            "bankroll": {
                "roi": 1.08,
                "profit_yen": 8_000,
                "stake_yen": 100_000,
                "policy": {
                    "payout_tail_schema": "conditional_payout_tail_v1",
                    "payout_feature_schema": "conditional_payout_interactions_v2",
                },
            },
            "bankroll_confidence": {
                "roi_ci95_lower": 1.01,
                "probability_roi_above_one": 0.98,
            },
            "diagnostic_gate": {
                "pass": True,
                "roi_pass": True,
            },
        },
    })

    assert summary["payout_feature_candidate_roi"] == 1.08
    assert summary["payout_feature_candidate_profit_yen"] == 8_000
    assert summary["payout_feature_candidate_stake_yen"] == 100_000
    assert (
        summary["payout_feature_candidate_schema"]
        == "conditional_payout_tail_v1"
    )
    assert summary["payout_feature_roi_ci95_lower"] == 1.01
    assert summary["payout_feature_gate_pass"] is True
    assert summary["payout_feature_promotion_eligible"] is False
    assert (
        result_decision("conditional_payout_tail", summary)
        == "reject_or_research_only"
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
    assert not any(
        row["task_type"] == "calibrated_mlp_recency_search" for row in calls
    )
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
