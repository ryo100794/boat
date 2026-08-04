from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import boatrace_ai.evaluation_queue as evaluation_queue

from boatrace_ai.evaluation_queue import (
    DEFAULT_WORK_TICKETS,
    JobDependencyUnavailable,
    ObsoleteJob,
    ResourceSnapshot,
    TASK_PROFILES,
    build_command,
    claim_job,
    dedupe_key,
    defer_job,
    enqueue_job,
    enqueue_refined_market_evaluation,
    ensure_schema,
    fail_job,
    reconcile_refined_market_evaluations,
    prepare_standardized_workspace,
    result_decision,
    seed_default_jobs,
    seed_daily_genetic_jobs,
    seed_daily_market_jobs,
    seed_periodic_jobs,
    seed_work_tickets,
    summarize_result,
)
from boatrace_ai.feature_schema import (
    DECAYED_HISTORY_FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    MISSING_SAFE_FEATURE_SCHEMA_VERSION,
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
        "status": "running",
        "task_type": task_type,
        "model_key": "candidate",
        "parameters": parameters,
    }


def test_dedupe_key_is_parameter_order_independent() -> None:
    assert dedupe_key("probe", "model", {"a": 1, "b": 2}) == dedupe_key(
        "probe", "model", {"b": 2, "a": 1}
    )


def test_daily_genetic_seed_is_scoped_by_protocol_version(monkeypatch) -> None:
    class Result:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        protocol_exists = False

        def __init__(self):
            self.queries = []

        def execute(self, sql, parameters):
            self.queries.append((sql, parameters))
            return Result({"found": 1} if self.protocol_exists else None)

    conn = Connection()
    enqueued = []

    def fake_enqueue(_conn, **kwargs):
        enqueued.append(kwargs)
        return len(enqueued)

    monkeypatch.setattr(evaluation_queue, "enqueue_job", fake_enqueue)

    inserted = seed_daily_genetic_jobs(
        conn,
        evaluation_date="2026-07-27",
        now=datetime(2026, 7, 29, tzinfo=timezone.utc),
        island_count=2,
        max_generations=1,
    )

    assert inserted == [1, 2]
    assert len(enqueued) == 2
    assert {
        row["parameters"]["genetic_protocol_version"] for row in enqueued
    } == {4}
    query, parameters = conn.queries[0]
    assert "parameters->>'genetic_protocol_version'" in query
    assert parameters == ("2026-07-27", "4")

    conn.protocol_exists = True
    assert seed_daily_genetic_jobs(
        conn,
        evaluation_date="2026-07-27",
        now=datetime(2026, 7, 29, 0, 1, tzinfo=timezone.utc),
        island_count=2,
        max_generations=1,
    ) == []


def test_enqueue_parser_loads_parameters_from_file(tmp_path: Path) -> None:
    parameters_file = tmp_path / "parameters.json"
    parameters_file.write_text('{"from_year": 2016, "to_year": 2026}', encoding="utf-8")
    args = evaluation_queue.build_parser().parse_args(
        [
            "enqueue",
            "--task-type",
            "racer_stats_backfill",
            "--model-key",
            "official_racer_periods_2016_2026",
            "--parameters-file",
            str(parameters_file),
            "--priority",
            "93",
            "--max-attempts",
            "3",
        ]
    )

    assert evaluation_queue.load_job_parameters(args.parameters_file) == {
        "from_year": 2016,
        "to_year": 2026,
    }
    assert args.priority == 93
    assert args.max_attempts == 3


def test_load_job_parameters_rejects_non_object(tmp_path: Path) -> None:
    parameters_file = tmp_path / "parameters.json"
    parameters_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="one JSON object"):
        evaluation_queue.load_job_parameters(parameters_file)


def test_cpu_times_are_limited_to_process_affinity(tmp_path: Path) -> None:
    stat = tmp_path / "stat"
    stat.write_text(
        "cpu  300 0 100 600 30 0 0 0 0 0\n"
        "cpu0 100 0 20 200 10 0 0 0 0 0\n"
        "cpu1 80 0 30 100 5 0 0 0 0 0\n"
        "cpu2 120 0 50 300 15 0 0 0 0 0\n",
        encoding="utf-8",
    )

    assert evaluation_queue._read_cpu_times({0, 2}, stat_path=stat) == (
        525,
        815,
    )
    assert evaluation_queue._read_cpu_times(None, stat_path=stat) == (630, 1030)


def test_four_head_learned_value_command_records_learning_and_outer_periods(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "four_head_learned_value",
            {
                "source_model": "data/models/evaluation_queue/job-00002707.joblib",
                "training_from": "2026-07-20",
                "training_through": "2026-07-30",
                "outer_from": "2026-07-31",
                "outer_through": "2026-08-01",
                "projection_dimensions": 16,
                "purchase_teacher_version": 3,
                "purchase_loss": "ridge_capped_net",
                "alpha": 0.01,
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1:3] == ["-m", "boatrace_ai.listwise.four_head_v22_evaluation"]
    assert command[command.index("--training-through") + 1] == "2026-07-30"
    assert command[command.index("--outer-from") + 1] == "2026-07-31"
    assert command[command.index("--projection-dimensions") + 1] == "16"
    assert command[command.index("--alpha") + 1] == "0.01"
    assert command[command.index("--purchase-loss") + 1] == "ridge_capped_net"
    assert command[command.index("--data-cache") + 1].startswith(
        str(root / "data/models/evaluation_cache/four_head_v22")
    )
    assert output == root / "data/models/evaluation_queue/job-00000007.json"
    assert "four_head_learned_value" in TASK_PROFILES


def test_four_head_data_cache_identity_ignores_job_lineage(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/evaluation_queue/job-00002707.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_teacher_version": 3,
        "purchase_loss": "ridge_capped_net",
    }
    first = _job("four_head_learned_value", params, job_id=8)
    second = _job("four_head_learned_value", params, job_id=9)
    first["parent_job_id"] = 100
    second["parent_job_id"] = 200

    first_command, _ = build_command(
        first,
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    second_command, _ = build_command(
        second,
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert first_command[first_command.index("--data-cache") + 1] == (
        second_command[second_command.index("--data-cache") + 1]
    )


def test_four_head_poisson_purchase_loss_requires_teacher_version_four(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "poisson_capped_gross",
        "purchase_teacher_version": 4,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == "poisson_capped_gross"
    params["purchase_teacher_version"] = 3
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_tweedie_purchase_loss_requires_teacher_version_five(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "tweedie_capped_gross",
        "purchase_teacher_version": 5,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == "tweedie_capped_gross"
    params["purchase_teacher_version"] = 4
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_hurdle_purchase_loss_requires_teacher_version_six(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "hurdle_logistic_lognormal",
        "purchase_teacher_version": 6,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "hurdle_logistic_lognormal"
    )
    params["purchase_teacher_version"] = 5
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_calibrated_hurdle_requires_teacher_version_seven(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "hurdle_logistic_lognormal_calibrated",
        "purchase_teacher_version": 7,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "hurdle_logistic_lognormal_calibrated"
    )
    params["purchase_teacher_version"] = 6
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_contextual_hurdle_requires_teacher_version_eight(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "hurdle_contextual_lognormal",
        "purchase_teacher_version": 8,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "hurdle_contextual_lognormal"
    )
    params["purchase_teacher_version"] = 7
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_contextual_interactions_requires_teacher_version_nine(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "hurdle_contextual_interactions_lognormal",
        "purchase_teacher_version": 9,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "hurdle_contextual_interactions_lognormal"
    )
    params["purchase_teacher_version"] = 8
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_pairwise_rank_requires_teacher_version_ten(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "pairwise_contextual_rank_calibrated",
        "purchase_teacher_version": 10,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "pairwise_contextual_rank_calibrated"
    )
    params["purchase_teacher_version"] = 9
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )




def test_four_head_temporal_aggregate_uses_only_queue_result_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "four_head_temporal_aggregate",
            {"source_job_ids": [11198, 11216, 11246, 11247]},
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert TASK_PROFILES["four_head_temporal_aggregate"]["memory_mb"] == 512
    assert command[1:3] == [
        "-m",
        "boatrace_ai.listwise.four_head_temporal_aggregate",
    ]
    assert command.count("--input") == 4
    assert str(root / "data/models/evaluation_queue/job-00011198.json") in command
    assert command[-2:] == ["--output", str(output)]

    with pytest.raises(ValueError, match="source_job_ids"):
        build_command(
            _job("four_head_temporal_aggregate", {"source_job_ids": [11198]}),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_offset_tail_requires_teacher_version_eleven(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "multinomial_offset_uncapped_lognormal",
        "purchase_teacher_version": 11,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "multinomial_offset_uncapped_lognormal"
    )
    params["purchase_teacher_version"] = 10
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_all_choice_closing_requires_teacher_version_twelve(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "multinomial_offset_all_choice_closing",
        "purchase_teacher_version": 12,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "multinomial_offset_all_choice_closing"
    )
    params["purchase_teacher_version"] = 11
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_temperature_requires_teacher_version_thirteen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "multinomial_offset_all_choice_closing_temperature",
        "purchase_teacher_version": 13,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "multinomial_offset_all_choice_closing_temperature"
    )
    params["purchase_teacher_version"] = 12
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_market_offset_requires_teacher_version_fourteen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": "multinomial_market_offset_all_choice_closing",
        "purchase_teacher_version": 14,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "multinomial_market_offset_all_choice_closing"
    )
    params["purchase_teacher_version"] = 13
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_oof_scaled_market_requires_teacher_version_fifteen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    params = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": (
            "multinomial_market_offset_oof_scaled_all_choice_closing"
        ),
        "purchase_teacher_version": 15,
    }
    command, _output = build_command(
        _job("four_head_learned_value", params),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[command.index("--purchase-loss") + 1] == (
        "multinomial_market_offset_oof_scaled_all_choice_closing"
    )
    params["purchase_teacher_version"] = 14
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            _job("four_head_learned_value", params),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_four_head_learned_value_rejects_training_outer_overlap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    with pytest.raises(ValueError, match="outer period"):
        build_command(
            _job(
                "four_head_learned_value",
                {
                    "source_model": "data/models/source.joblib",
                    "training_from": "2026-07-20",
                    "training_through": "2026-07-31",
                    "outer_from": "2026-07-31",
                    "outer_through": "2026-08-01",
                    "purchase_teacher_version": 3,
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_summary_preserves_learned_purchase_value_calibration() -> None:
    value = {
        "schema_version": 1,
        "tickets": 240,
        "pearson_correlation": 0.12,
        "calibration_mae": 0.08,
        "positive_predicted_tickets": 12,
        "positive_predicted_fraction": 0.05,
        "positive_observed_capped_roi": 1.04,
        "calibration_deciles": [{"quantile": 10, "tickets": 24}],
    }
    summary = summarize_result({
        "purchase_value_diagnostics": value,
        "purchase_probability_temperature": 1.75,
    })


    assert summary["purchase_value_diagnostics"] == value
    assert summary["purchase_value_pearson_correlation"] == 0.12
    assert summary["purchase_value_positive_observed_capped_roi"] == 1.04
    assert summary["purchase_probability_temperature"] == 1.75


def test_summary_preserves_temporal_aggregate_page_metrics() -> None:
    summary = summarize_result(
        {
            "profitable_day_fraction": 1 / 7,
            "purchase_value_positive_predicted_tickets": 816,
            "purchase_value_positive_observed_capped_roi": 0.6334,
        }
    )

    assert summary["profitable_day_fraction"] == 1 / 7
    assert summary["purchase_value_positive_predicted_tickets"] == 816
    assert summary["purchase_value_positive_observed_capped_roi"] == 0.6334


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


def test_archive_market_oracle_command_is_period_bounded(tmp_path) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "archive_market_oracle",
            {
                "from_date": "2025-07-25",
                "through_date": "2026-07-24",
                "model_input": "data/models/evaluation_queue/job-00002606.joblib",
                "daily_budget_yen": 10000,
                "timeout_seconds": 43200,
                "temporal_calibration_through": "2026-07-17",
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1:3] == ["-m", "boatrace_ai.listwise.archive_market_oracle"]
    assert command[command.index("--from-date") + 1] == "2025-07-25"
    assert command[command.index("--temporal-calibration-through") + 1] == (
        "2026-07-17"
    )
    assert command[command.index("--daily-budget-yen") + 1] == "10000"
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_listwise_cutoff_refit_command_is_period_bounded(tmp_path) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "listwise_cutoff_refit",
            {
                "source_model": "data/models/listwise_newton_cg_v1.joblib",
                "training_cutoff": "2026-05-09",
                "evaluation_from": "2026-05-10",
                "evaluation_through": "2026-05-16",
                "timeout_seconds": 43200,
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1:3] == ["-m", "boatrace_ai.listwise.cutoff_refit"]
    assert command[command.index("--training-cutoff") + 1] == "2026-05-09"
    assert command[command.index("--evaluation-from") + 1] == "2026-05-10"
    assert command[command.index("--model-output") + 1] == str(
        root / "data/models/evaluation_queue/job-00000007.joblib"
    )
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_listwise_cutoff_refit_rejects_overlapping_dates(tmp_path) -> None:
    root = tmp_path / "boat"
    with pytest.raises(ValueError, match="training_cutoff"):
        build_command(
            _job(
                "listwise_cutoff_refit",
                {
                    "source_model": "data/models/listwise_newton_cg_v1.joblib",
                    "training_cutoff": "2026-05-10",
                    "evaluation_from": "2026-05-10",
                    "evaluation_through": "2026-05-16",
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_archive_closing_backfill_is_rate_limited_and_serial(tmp_path) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "archive_closing_backfill",
            {
                "from_date": "2026-06-01",
                "through_date": "2026-06-30",
                "sleep_seconds": 1.5,
                "max_pages": 4000,
                "timeout_seconds": 86400,
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert TASK_PROFILES["archive_closing_backfill"] == {
        "category": "collection",
        "memory_mb": 512,
        "disk_mb": 256,
        "idle_cpu": 3.0,
        "max_parallel": 1,
    }
    assert command[1:3] == ["-m", "boatrace_ai.archive_closing_odds"]
    assert command[command.index("--sleep-seconds") + 1] == "1.5"
    assert command[command.index("--max-pages") + 1] == "4000"
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_explicit_failed_promotion_gate_cannot_fall_through_to_roi_pass() -> None:
    assert result_decision(
        "four_head_learned_value",
        {
            "roi": 3.2666667,
            "profit_yen": 680,
            "promotion_eligible": False,
        },
    ) == "reject_or_research_only"


def test_archive_closing_backfill_result_is_collection_not_rejection() -> None:
    assert result_decision(
        "archive_closing_backfill",
        {"archive_stored": 4000, "archive_remaining": 12000},
    ) == "collection_complete"


def test_calibrated_mlp_recency_search_profile() -> None:
    assert TASK_PROFILES["calibrated_mlp_recency_search"] == {
        "category": "evaluation",
        "memory_mb": 16384,
        "disk_mb": 4096,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }


def test_model_cache_archive_uses_backup_profile_and_allowlisted_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    cache = root / "data/models/legacy.matrix.npz"
    command, output = build_command(
        _job(
            "gdrive_model_cache_archive",
            {
                "paths": ["data/models/legacy.matrix.npz"],
                "timeout_seconds": 86400,
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert TASK_PROFILES["gdrive_model_cache_archive"] == {
        "category": "backup",
        "memory_mb": 512,
        "disk_mb": 2048,
        "idle_cpu": 3.0,
        "max_parallel": 1,
    }
    assert command[-2:] == ["--path", str(cache)]
    assert output == root / "data/models/evaluation_queue/job-00000007.json"

    with pytest.raises(ValueError, match="inside data/models"):
        build_command(
            _job("gdrive_model_cache_archive", {"paths": ["../secret"]}),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_repository_hygiene_profile_is_low_resource_and_serial() -> None:
    assert TASK_PROFILES["repository_hygiene"] == {
        "category": "maintenance",
        "memory_mb": 256,
        "disk_mb": 256,
        "idle_cpu": 3.0,
        "max_parallel": 1,
    }



def test_persist_selected_cache_profile_and_wait_command(tmp_path: Path) -> None:
    assert TASK_PROFILES["persist_standard_selected_cache"] == {
        "category": "maintenance",
        "memory_mb": 512,
        "disk_mb": 1024,
        "idle_cpu": 3.0,
        "max_parallel": 1,
    }
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "persist_standard_selected_cache",
            {
                "artifact_mtime_after": 1_774_500_000.5,
                "timeout_seconds": 21_600,
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command == [
        str(root / ".venv/bin/python"),
        str(root / "scripts/persist_selected_feature_cache.py"),
        "--artifact",
        str(
            root
            / "data/models/standardized_365d_v2/raw/listwise_feature_teacher.json"
        ),
        "--destination-dir",
        str(root / "data/models/standardized_365d_v2/selected_cache"),
        "--wait-for-mtime-after",
        "1774500000.5",
        "--wait-timeout-seconds",
        "21300",
        "--output",
        str(root / "data/models/evaluation_queue/job-00000007.json"),
    ]
    assert output == root / "data/models/evaluation_queue/job-00000007.json"
    assert result_decision(
        "persist_standard_selected_cache", {"status": "persisted"}
    ) == "maintenance_complete"


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"artifact_mtime_after": 0.0},
        {"artifact_mtime_after": 1_774_500_000.5, "unexpected": True},
    ],
)
def test_persist_selected_cache_command_rejects_invalid_parameters(
    tmp_path: Path, parameters: dict
) -> None:
    with pytest.raises(ValueError):
        build_command(
            _job("persist_standard_selected_cache", parameters),
            app_root=tmp_path / "boat",
            python=tmp_path / "boat/.venv/bin/python",
            db="postgresql://test",
        )

def test_obsolete_feature_schema_refinement_is_cancelled_before_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    result = root / "data/models/evaluation_queue/job-00000395.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"feature_schema_version": MISSING_SAFE_FEATURE_SCHEMA_VERSION}),
        encoding="utf-8",
    )
    with pytest.raises(ObsoleteJob, match="obsolete"):
        build_command(
            _job(
                "listwise_newton_refine",
                {
                    "search_result": "data/models/evaluation_queue/job-00000395.json",
                    "cache_dir": str(root / "data/models/evaluation_cache/job-00000395"),
                },
            ),
            app_root=root,
            python=Path("/venv/python"),
            db="host=postgres dbname=boatrace",
        )


@pytest.mark.parametrize(
    "schema",
    (FEATURE_SCHEMA_VERSION, DECAYED_HISTORY_FEATURE_SCHEMA_VERSION),
    ids=("current", "decayed-history"),
)
@pytest.mark.parametrize(
    ("job_threshold", "parent_threshold", "expected_threshold"),
    [
        (1.2, 1.0, "1.2"),
        (None, 1.0, "1.0"),
        (None, None, "1.2"),
    ],
    ids=["explicit-job-wins", "parent-fallback", "default-fallback"],
)
def test_newton_refinement_ev_threshold_precedence(
    tmp_path: Path,
    job_threshold: float | None,
    schema: str,
    parent_threshold: float | None,
    expected_threshold: str,
) -> None:
    root = tmp_path / "boat"
    result = root / "data/models/evaluation_queue/job-00007011.json"
    result.parent.mkdir(parents=True)
    payload: dict[str, object] = {"feature_schema_version": schema}
    if parent_threshold is not None:
        payload["policy"] = {"ev_threshold": parent_threshold}
    result.write_text(json.dumps(payload), encoding="utf-8")
    parameters: dict[str, object] = {
        "search_result": "data/models/evaluation_queue/job-00007011.json",
        "cache_dir": str(root / "data/models/evaluation_cache/job-00007011"),
    }
    if job_threshold is not None:
        parameters["ev_threshold"] = job_threshold

    command, _output = build_command(
        _job("listwise_newton_refine", parameters, job_id=7369),
        app_root=root,
        python=Path("/venv/python"),
        db="host=postgres dbname=boatrace",
    )

    assert command[command.index("--ev-threshold") + 1] == expected_threshold


def test_series_feature_cache_profile_and_command(tmp_path: Path) -> None:
    assert TASK_PROFILES["series_feature_cache"] == {
        "category": "maintenance",
        "memory_mb": 512,
        "disk_mb": 256,
        "idle_cpu": 3.0,
        "max_parallel": 1,
    }
    command, output = build_command(
        _job(
            "series_feature_cache",
            {"from_date": "2026-07-09", "timeout_seconds": 600},
        ),
        app_root=tmp_path,
        python=Path("/venv/python"),
        db="host=postgres dbname=boatrace",
    )
    assert command == [
        "/venv/python", "-m", "boatrace_ai.cache_entry_series_features",
        "--db", "host=postgres dbname=boatrace",
        "--batch-size", "1000",
        "--from-date", "2026-07-09",
        "--output", str(output),
    ]
    assert output == tmp_path / "data/models/evaluation_queue/job-00000007.json"


def _write_standard_feature_artifact(
    root: Path,
    cache_dir: Path,
    *,
    variant: str = "drop_research_correlates",
    n_features: int = 4096,
    create_manifest: bool = True,
) -> Path:
    artifact = (
        root / "data/models/standardized_365d_v2/raw"
        / "listwise_feature_teacher.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    protocol = artifact.parent.parent / "protocol.json"
    protocol.write_text(
        json.dumps({
            "holdout_start": "2025-07-20",
            "holdout_end": "2026-07-19",
            "calendar_days": 365,
        }),
        encoding="utf-8",
    )
    artifact.write_text(json.dumps({
        "selected": {"feature_variant": variant},
        "selected_cache_dir": str(cache_dir),
        "n_features": n_features,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }), encoding="utf-8")
    cache_prefix = cache_dir / f"listwise_search_{n_features}_{variant}"
    if create_manifest:
        cache_dir.mkdir(parents=True, exist_ok=True)
        Path(str(cache_prefix) + ".manifest.json").write_text(
            json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION}),
            encoding="utf-8",
        )
    return cache_prefix


def test_standardized_selected_cache_root_is_fixed() -> None:
    assert evaluation_queue.STANDARDIZED_SELECTED_CACHE_DIR == Path(
        "/workspace/boat/data/models/standardized_365d_v2/selected_cache"
    )


def test_conditional_payout_tail_profile_and_command_are_fixed(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    cache_prefix = _write_standard_feature_artifact(root, cache_dir)
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
        str(cache_prefix),
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
        "365",
        "--batch-races",
        "4000",
        "--payout-mean-corrections",
        "0.0",
        "0.5",
        "1.0",
        "--payout-threshold-candidates",
        "1.05",
        "1.10",
        "1.20",
        "1.30",
        "1.50",
        "2.00",
        "--promote-legacy-cache",
    ]
    assert output == result


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("malformed", "incomplete or invalid"),
        ("incomplete", "incomplete or invalid"),
        ("unknown_variant", "unknown feature variant"),
        ("feature_range", "out of range"),
        ("cache_traversal", "must exactly match"),
    ],
)
def test_claimed_conditional_payout_fails_on_invalid_standard_artifact(
    tmp_path,
    monkeypatch,
    case,
    message,
) -> None:
    root = tmp_path / "boat"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    artifact = (
        root / "data/models/standardized_365d_v2/raw"
        / "listwise_feature_teacher.json"
    )
    if case != "missing":
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if case == "malformed":
            artifact.write_text("{", encoding="utf-8")
        else:
            payload = {
                "selected": {"feature_variant": "full"},
                "selected_cache_dir": str(cache_dir),
                "n_features": 4096,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            }
            if case == "incomplete":
                payload.pop("selected")
            elif case == "unknown_variant":
                payload["selected"]["feature_variant"] = "../full"
            elif case == "feature_range":
                payload["n_features"] = 999
            elif case == "cache_traversal":
                payload["selected_cache_dir"] = str(cache_dir / ".." / "escape")
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            if case != "missing_manifest":
                prefix = cache_dir / "listwise_search_4096_full"
                cache_dir.mkdir(parents=True, exist_ok=True)
                Path(str(prefix) + ".manifest.json").write_text(
                    json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION}),
                    encoding="utf-8",
                )

    with pytest.raises(ValueError, match=message):
        build_command(
            _job(
                "conditional_payout_tail",
                {
                    "training_through": "2025-07-19",
                    "evaluation_from": "2025-07-20",
                    "evaluation_through": "2026-07-19",
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


@pytest.mark.parametrize("stale_part", ["artifact", "manifest"])
def test_conditional_payout_defers_stale_feature_contract(
    tmp_path,
    monkeypatch,
    stale_part,
) -> None:
    root = tmp_path / "boat"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    cache_prefix = _write_standard_feature_artifact(root, cache_dir)
    target = (
        root
        / "data/models/standardized_365d_v2/raw/listwise_feature_teacher.json"
        if stale_part == "artifact"
        else Path(str(cache_prefix) + ".manifest.json")
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["feature_schema_version"] = MISSING_SAFE_FEATURE_SCHEMA_VERSION
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(JobDependencyUnavailable, match="stale feature schema"):
        build_command(
            _job(
                "conditional_payout_tail",
                {
                    "training_through": "2025-07-19",
                    "evaluation_from": "2025-07-20",
                    "evaluation_through": "2026-07-19",
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


@pytest.mark.parametrize("missing", ["artifact", "manifest"])
def test_conditional_payout_defers_until_selected_cache_exists(
    tmp_path,
    monkeypatch,
    missing,
) -> None:
    root = tmp_path / "boat"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    if missing == "manifest":
        _write_standard_feature_artifact(
            root,
            cache_dir,
            create_manifest=False,
        )

    with pytest.raises(JobDependencyUnavailable, match="not available yet"):
        build_command(
            _job(
                "conditional_payout_tail",
                {
                    "training_through": "2025-07-19",
                    "evaluation_from": "2025-07-20",
                    "evaluation_through": "2026-07-19",
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_conditional_payout_cancels_window_not_matching_standard_protocol(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "boat"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    _write_standard_feature_artifact(root, cache_dir)

    with pytest.raises(ObsoleteJob, match="does not match"):
        build_command(
            _job(
                "conditional_payout_tail",
                {
                    "training_through": "2025-07-20",
                    "evaluation_from": "2025-07-21",
                    "evaluation_through": "2026-07-20",
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


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
        "--model-output",
        str(root / "data/models/evaluation_queue/job-00000007.joblib"),
        "--deployment-model-output",
        str(root / "data/models/evaluation_queue/job-00000007.deployment.joblib"),
        "--incumbent-prediction",
        str(root / "data/models/standardized_365d_v2/raw/no_odds_v8_prediction.json"),
        "--incumbent-bankroll",
        str(root / "data/models/standardized_365d_v2/raw/no_odds_v8_bankroll.json"),
        "--evaluation-date",
        "2026-07-22",
        "--feature-cache",
        str(root / "data/models/calibrated_shadow_features_16384"),
        "--drop-feature-groups",
        "research_correlates",
        "--half-lives",
        "none,180,365",
        "--calibration-days",
        "120",
        "--selection-entry-log-loss-tolerance",
        "0.0005",
    ]
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


    base_command, _ = build_command(
        _job(
            "calibrated_mlp_recency_search",
            {
                "evaluation_date": "2026-07-22",
                "drop_feature_groups": "base_pastlog,base_pastlog",
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    cache_index = base_command.index("--feature-cache") + 1
    groups_index = base_command.index("--drop-feature-groups") + 1
    assert base_command[cache_index].endswith(
        "calibrated_shadow_features_16384__drop_base_pastlog"
    )
    assert base_command[groups_index] == "base_pastlog"

    research_command, _ = build_command(
        _job(
            "calibrated_mlp_recency_search",
            {
                "evaluation_date": "2026-07-22",
                "drop_feature_groups": "raw_equipment_identifiers",
                "protected_blend": True,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    research_cache_index = research_command.index("--feature-cache") + 1
    research_groups_index = research_command.index("--drop-feature-groups") + 1
    assert research_command[research_cache_index].endswith(
        "calibrated_shadow_features_16384__drop_raw_equipment_identifiers"
    )
    assert research_command[research_groups_index] == "raw_equipment_identifiers"

    official_command, _ = build_command(
        _job(
            "calibrated_mlp_recency_search",
            {
                "evaluation_date": "2026-07-22",
                "drop_feature_groups": (
                    "live_official_context,speculative_research,"
                    "raw_equipment_identifiers"
                ),
                "protected_blend": True,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    official_cache_index = official_command.index("--feature-cache") + 1
    official_groups_index = official_command.index("--drop-feature-groups") + 1
    assert official_command[official_cache_index].endswith(
        "calibrated_shadow_features_16384__drop_raw_equipment_identifiers_"
        "speculative_research_live_official_context"
    )
    assert official_command[official_groups_index] == (
        "raw_equipment_identifiers,speculative_research,live_official_context"
    )

    protected_command, _ = build_command(
        _job(
            "calibrated_mlp_recency_search",
            {"evaluation_date": "2026-07-22", "protected_blend": True},
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert protected_command[-2:] == [
        "--protected-baseline-model",
        str(root / "data/models/standardized_365d_v2/no_odds_v8.joblib"),
    ]

    default_command, _ = build_command(
        _job("calibrated_mlp_recency_search", {"evaluation_date": "2026-07-22"}),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert default_command[-6:] == [
        "--half-lives",
        "none,180,365,730",
        "--calibration-days",
        "180",
        "--selection-entry-log-loss-tolerance",
        "0.0005",
    ]

    selected_command, _ = build_command(
        _job(
            "calibrated_mlp_recency_search",
            {
                "evaluation_date": "2026-07-26",
                "half_lives": "none",
                "protected_blend": True,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    selected_index = selected_command.index("--half-lives") + 1
    assert selected_command[selected_index] == "none"


def test_fixed_model_conditional_order_uses_exact_artifact_and_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    model = root / "data/models/evaluation_queue/job-00012012.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixed-model")
    cache = (
        root
        / "data/models/evaluation_cache/job-00011732-combined"
        / "listwise_search_8192_keep_card_numeric"
    )
    cache.parent.mkdir(parents=True)
    for suffix in (".matrix.npz", ".ranks.npy", ".manifest.json"):
        Path(f"{cache}{suffix}").write_bytes(b"cache")
    python = root / ".venv/bin/python"

    command, output = build_command(
        _job(
            "fixed_model_conditional_order",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00012012.joblib"
                ),
                "cache_prefix": (
                    "data/models/evaluation_cache/job-00011732-combined/"
                    "listwise_search_8192_keep_card_numeric"
                ),
                "expected_model_sha256": hashlib.sha256(
                    b"fixed-model"
                ).hexdigest(),
                "training_through": "2025-07-25",
                "evaluation_from": "2025-07-26",
                "evaluation_through": "2026-08-02",
                "expected_evaluation_races": 49_581,
                "timeout_seconds": 21600,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert TASK_PROFILES["fixed_model_conditional_order"] == {
        "category": "evaluation",
        "memory_mb": 12288,
        "disk_mb": 2048,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }
    assert command[:3] == [
        str(python),
        "-m",
        "boatrace_ai.listwise.conditional_order",
    ]
    assert command[command.index("--baseline-model") + 1] == str(model)
    assert command[command.index("--cache-prefix") + 1] == str(cache)
    assert command[command.index("--training-through") + 1] == "2025-07-25"
    assert command[command.index("--evaluation-from") + 1] == "2025-07-26"
    assert command[command.index("--evaluation-through") + 1] == "2026-08-02"
    assert command[command.index("--validation-days") + 1] == "365"
    assert "--research-only-reused-holdout" in command
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_fixed_model_conditional_order_rejects_identity_or_period_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    model = root / "data/models/evaluation_queue/job-00012012.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixed-model")
    cache = root / "data/models/evaluation_cache/cache"
    cache.parent.mkdir(parents=True)
    for suffix in (".matrix.npz", ".ranks.npy", ".manifest.json"):
        Path(f"{cache}{suffix}").write_bytes(b"cache")
    base = {
        "model_input": "data/models/evaluation_queue/job-00012012.joblib",
        "cache_prefix": "data/models/evaluation_cache/cache",
        "expected_model_sha256": hashlib.sha256(b"other").hexdigest(),
        "training_through": "2025-07-25",
        "evaluation_from": "2025-07-26",
        "evaluation_through": "2026-08-02",
        "expected_evaluation_races": 49_581,
    }

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        build_command(
            _job("fixed_model_conditional_order", base),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )
    drifted = {
        **base,
        "expected_model_sha256": hashlib.sha256(
            b"fixed-model"
        ).hexdigest(),
        "evaluation_from": "2025-07-27",
    }
    with pytest.raises(ValueError, match="must be adjacent"):
        build_command(
            _job("fixed_model_conditional_order", drifted),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_fixed_model_conditional_order_result_contract(tmp_path) -> None:
    job = _job(
        "fixed_model_conditional_order",
        {
            "expected_model_sha256": "a" * 64,
            "training_through": "2025-07-25",
            "evaluation_from": "2025-07-26",
            "evaluation_through": "2026-08-02",
            "expected_evaluation_races": 49_581,
        },
    )
    tail_diagnostics = {
        "odds_field": "estimated_odds_at_purchase",
        "purchased_tickets": 30,
        "normal": {"tickets": 20},
        "tail": {"tickets": 10},
    }
    bankroll = {
        "evaluated_races": 49_581,
        "effective_hit_count": 25.0,
        "tail_portfolio_diagnostics": tail_diagnostics,
    }
    payload = {
        "source_model_sha256": "a" * 64,
        "reused_holdout_research_only": True,
        "promotion_eligible": False,
        "training_through": "2025-07-25",
        "evaluation_from": "2025-07-26",
        "evaluation_through": "2026-08-02",
        "evaluation_races": 49_581,
        "conditional_order": {"evaluated_races": 49_581},
        "listwise_baseline": {"evaluated_races": 49_581},
        "bankroll": bankroll,
        "baseline_bankroll": bankroll,
        "conditional_payout_walk_forward": {
            "artifact_state_saved": True,
            "state_schema": "conditional_payout_next_day_inference_v1",
            "state_trained_through": "2026-08-02",
            "state_role": "next_day_inference_after_evaluation",
            "bankroll": bankroll,
        },
        "expected_return_calibration": {
            "artifact_state_saved": True,
            "state_schema": "expected_return_next_day_inference_v1",
            "state_trained_through": "2026-08-02",
            "state_role": "next_day_inference_after_evaluation",
            "bankroll": bankroll,
        },
    }

    evaluation_queue._validate_job_result_contract(job, payload)

    raw_payload = json.loads(json.dumps(payload))
    raw_ticket = {
        "date": "2025-07-26",
        "race_id": "holdout-race",
        "odds": 20.0,
        "stake": 100,
        "return": 2_000,
    }
    for path in (
        ("bankroll",),
        ("baseline_bankroll",),
        ("conditional_payout_walk_forward", "bankroll"),
        ("expected_return_calibration", "bankroll"),
    ):
        bankroll = raw_payload
        for key in path:
            bankroll = bankroll[key]
        bankroll.pop("tail_portfolio_diagnostics")
        bankroll["daily"] = [
            {
                "race_date": "2025-07-26",
                "_tail_portfolio_rows": [raw_ticket],
            }
        ]
    raw_result = tmp_path / "fixed-conditional-order.json"
    raw_result.write_text(json.dumps(raw_payload), encoding="utf-8")
    loaded, _summary = evaluation_queue._load_result(raw_result)
    evaluation_queue._validate_job_result_contract(job, loaded)
    for path in (
        ("bankroll",),
        ("baseline_bankroll",),
        ("conditional_payout_walk_forward", "bankroll"),
        ("expected_return_calibration", "bankroll"),
    ):
        bankroll = loaded
        for key in path:
            bankroll = bankroll[key]
        assert bankroll["tail_portfolio_diagnostics"][
            "purchased_tickets"
        ] == 1
        assert "_tail_portfolio_rows" not in bankroll["daily"][0]

    with pytest.raises(ValueError, match="evaluation_races mismatch"):
        evaluation_queue._validate_job_result_contract(
            job, {**payload, "evaluation_races": 49_580}
        )
    with pytest.raises(ValueError, match="evaluation_through mismatch"):
        evaluation_queue._validate_job_result_contract(
            job, {**payload, "evaluation_through": "2026-08-01"}
        )
    with pytest.raises(ValueError, match="source_model_sha256 mismatch"):
        evaluation_queue._validate_job_result_contract(
            job, {**payload, "source_model_sha256": "b" * 64}
        )
    with pytest.raises(
        ValueError,
        match="conditional_payout_walk_forward.bankroll.evaluated_races mismatch",
    ):
        evaluation_queue._validate_job_result_contract(
            job,
            {
                **payload,
                "conditional_payout_walk_forward": {
                    "artifact_state_saved": True,
                    "state_schema": (
                        "conditional_payout_next_day_inference_v1"
                    ),
                    "state_trained_through": "2026-08-02",
                    "state_role": "next_day_inference_after_evaluation",
                    "bankroll": {"evaluated_races": 49_580}
                },
            },
        )
    with pytest.raises(
        ValueError,
        match="expected_return_calibration trained_through mismatch",
    ):
        evaluation_queue._validate_job_result_contract(
            job,
            {
                **payload,
                "expected_return_calibration": {
                    **payload["expected_return_calibration"],
                    "state_trained_through": "2026-08-01",
                },
            },
        )
    with pytest.raises(
        ValueError,
        match="expected_return_calibration.bankroll.tail_portfolio_diagnostics invalid",
    ):
        evaluation_queue._validate_job_result_contract(
            job,
            {
                **payload,
                "expected_return_calibration": {
                    **payload["expected_return_calibration"],
                    "bankroll": {"evaluated_races": 49_581},
                },
            },
        )
    with pytest.raises(
        ValueError,
        match="expected_return_calibration.bankroll.effective_hit_count invalid",
    ):
        evaluation_queue._validate_job_result_contract(
            job,
            {
                **payload,
                "expected_return_calibration": {
                    **payload["expected_return_calibration"],
                    "bankroll": {
                        **bankroll,
                        "effective_hit_count": None,
                    },
                },
            },
        )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "evaluation_date is required"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none,29"}, "finite numbers"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none,nan"}, "finite numbers"),
        ({"evaluation_date": "2026-07-22", "calibration_days": 29}, "calibration_days"),
        (
            {
                "evaluation_date": "2026-07-22",
                "selection_entry_log_loss_tolerance": 0.051,
            },
            "selection_entry_log_loss_tolerance",
        ),
        ({"evaluation_date": "2026-07-22", "timeout_seconds": 299}, "timeout_seconds"),
        ({"evaluation_date": "2026-07-22", "protected_blend": "yes"}, "boolean"),
        ({"evaluation_date": "2026-07-22", "command": "rm -rf /"}, "unsupported"),
        ({"evaluation_date": "2026-07-22", "feature_cache": "/tmp/cache"}, "unsupported"),
        ({"evaluation_date": "2026-07-22", "drop_feature_groups": "future"}, "unknown"),
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


def test_feature_search_accepts_explicit_feature_variants(tmp_path) -> None:
    command, _output = build_command(
        _job(
            "listwise_feature_search",
            {
                "evaluation_date": "2026-07-22",
                "feature_variants": (
                    "drop_card_numeric,drop_card_relative,"
                    "drop_card_numeric_card_relative,drop_card_numeric"
                ),
            },
        ),
        app_root=tmp_path,
        python=tmp_path / "python",
        db="postgresql://test",
    )
    index = command.index("--feature-variants")
    assert command[index + 1] == (
        "drop_card_numeric,drop_card_relative,drop_card_numeric_card_relative"
    )


def test_feature_search_can_reuse_verified_completed_job_output(tmp_path) -> None:
    command, _output = build_command(
        _job(
            "listwise_feature_search",
            {
                "evaluation_date": "2026-07-22",
                "reuse_search_job_id": 7476,
            },
        ),
        app_root=tmp_path,
        python=tmp_path / "python",
        db="postgresql://test",
    )
    index = command.index("--reuse-search-output")
    assert command[index + 1] == str(
        tmp_path
        / "data/models/evaluation_queue/job-00007476.json"
    )


def test_combined_search_rejects_reuse_search_job_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="supported only"):
        build_command(
            _job(
                "combined_feature_search",
                {
                    "evaluation_date": "2026-07-22",
                    "reuse_search_job_id": 7476,
                },
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_combined_search_accepts_registered_feature_variant_subset(tmp_path) -> None:
    command, _output = build_command(
        _job(
            "combined_feature_search",
            {
                "evaluation_date": "2026-07-22",
                "feature_variants": "keep_card_numeric",
                "targets": "winner",
                "alphas": "0.00001",
            },
        ),
        app_root=tmp_path,
        python=tmp_path / "python",
        db="postgresql://test",
    )
    index = command.index("--combined-feature-variants")
    assert command[index + 1] == "keep_card_numeric"


def test_feature_search_accepts_bounded_ev_policy_selection_grid(tmp_path) -> None:
    command, _output = build_command(
        _job(
            "listwise_feature_search",
            {
                "evaluation_date": "2026-07-22",
                "ev_thresholds": "1.0,1.2,1.5",
            },
        ),
        app_root=tmp_path,
        python=tmp_path / "python",
        db="postgresql://test",
    )

    index = command.index("--ev-thresholds")
    assert command[index + 1] == "1,1.2,1.5"
    assert "--ev-threshold" not in command


@pytest.mark.parametrize("parameter", ["variant_workers", "candidate_workers", "cache_dir"])
def test_feature_search_rejects_injected_worker_or_path(tmp_path, parameter) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported listwise_feature_search parameters",
    ):
        build_command(
            _job(
                "listwise_feature_search",
                {"evaluation_date": "2026-07-22", parameter: 3},
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_fresh_work_ticket_seed_registers_feature_search_parallelization(
    tmp_path: Path,
) -> None:
    expected = next(
        row for row in DEFAULT_WORK_TICKETS if row[0] == "OPS-EVAL-PERF-001"
    )
    conn = sqlite3.connect(tmp_path / "fresh.sqlite")
    conn.execute(
        """
        CREATE TABLE work_tickets (
          ticket_key TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          area TEXT NOT NULL,
          description TEXT NOT NULL,
          acceptance_criteria TEXT NOT NULL,
          priority INTEGER NOT NULL,
          status TEXT NOT NULL,
          progress INTEGER NOT NULL,
          source TEXT NOT NULL
        )
        """
    )
    class CursorWithoutRowcount:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchone(self):
            return self.cursor.fetchone()

    class ConnectionWithoutRowcount:
        def execute(self, statement, parameters=()):
            return CursorWithoutRowcount(conn.execute(statement, parameters))

    compat_conn = ConnectionWithoutRowcount()
    try:
        assert seed_work_tickets(compat_conn) == len(DEFAULT_WORK_TICKETS)
        assert seed_work_tickets(compat_conn) == 0
        actual = conn.execute(
            """
            SELECT ticket_key, title, area, description, acceptance_criteria,
                   priority, status, progress
            FROM work_tickets
            WHERE ticket_key = 'OPS-EVAL-PERF-001'
            """
        ).fetchone()
    finally:
        conn.close()
    assert actual == expected
    assert expected[1:5] == (
        "特徴探索の並列化と再現性保証",
        "モデル基盤",
        "特徴バリアント生成を資源制約付きで並列化し、評価待ち時間を短縮する。GitHub Issue: https://github.com/ryo100794/boat/issues/1",
        "workers=1/2で候補順・selected・holdout hash・資金評価が一致し、checkpoint再開可能。Git commit SHAとDBイベントを記録し、リモートが同SHAで稼働する",
    )
    assert expected[6:] == ("in_progress", 35)


def test_default_work_tickets_include_sync_hygiene_and_model_followups() -> None:
    keys = {row[0] for row in DEFAULT_WORK_TICKETS}
    assert {
        "OPS-EVAL-MEM-001",
        "OPS-GITHUB-SYNC-001",
        "OPS-REPO-SYNC-001",
        "DOCS-HIERARCHY-001",
        "MODEL-FEATURE-COMBINE-001",
        "MODEL-SERIES-CACHE-001",
        "MODEL-PAYOUT-001",
        "MODEL-RECENCY-001",
        "MODEL-VENUE-001",
        "MODEL-SEGMENT-001",
        "MODEL-MARKET-RESIDUAL-001",
        "MODEL-HISTORICAL-RESIDUAL-001",
        "MODEL-MARKET-POLICY-CAL-001",
        "MODEL-V21-PROSPECTIVE-EVIDENCE-001",
        "TEST-BASELINE-FAILURES-001",
        "UI-MODEL-DAILY-001",
    } <= keys
    memory_ticket = next(
        row for row in DEFAULT_WORK_TICKETS
        if row[0] == "OPS-EVAL-MEM-001"
    )
    assert memory_ticket[5:] == (98, "in_progress", 15)

    combined_ticket = next(
        row for row in DEFAULT_WORK_TICKETS
        if row[0] == "MODEL-FEATURE-COMBINE-001"
    )
    assert combined_ticket == (
        "MODEL-FEATURE-COMBINE-001",
        "Combined feature ablation and retraining",
        "Model",
        "Run selection-only search combining base_pastlog+research_correlates with inert series_cached/series_relative ablations",
        "Compare against single ablations on the same fixed 365-day holdout and evaluation axes without holdout leakage",
        88,
        "queued",
        10,
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


def test_result_summary_exposes_bankroll_gate_and_temporal_folds() -> None:
    summary = summarize_result({
        "promotion_gate": {
            "minimum_tickets": True,
            "minimum_betting_days": False,
            "roi_above_one": True,
        },
        "holdout_temporal_stability": {
            "minimum_roi": 0.94,
            "folds": [{"roi": 1.10}, {"roi": 0.94}, {"roi": 1.03}],
        },
    })

    assert summary["promotion_gate_passed"] == 2
    assert summary["promotion_gate_total"] == 3
    assert summary["promotion_gate_failed"] == ["minimum_betting_days"]
    assert summary["holdout_temporal_minimum_roi"] == 0.94
    assert summary["holdout_temporal_fold_rois"] == [1.10, 0.94, 1.03]


def test_result_summary_exposes_fixed_trend_point_prospective_evidence() -> None:
    summary = summarize_result({
        "source_model_sha256": "a" * 64,
        "prequential_conditional_order": {
            "status": "evaluated",
            "method": "strict_prior",
            "minimum_prior_days": 4,
            "available_days": 15,
            "transformed_days": 11,
            "transformed_races": 1630,
            "baseline_log_loss": 3.8575,
            "conditional_log_loss": 3.7908,
            "log_loss_difference": -0.0667,
            "baseline_top5_hit_rate": 0.3423,
            "conditional_top5_hit_rate": 0.3521,
            "top5_hit_rate_difference": 0.0098,
            "improving_days": 11,
            "daily": [{"must": "not leak into summary"}],
        },
        "trend_point_market_offset_kelly_walk_forward": {
            "status": "evaluating",
            "registered_after": "2026-08-03",
            "evaluation_days": 4,
            "evaluated_races": 612,
            "tickets": 104,
            "hit_tickets": 17,
            "stake_yen": 12_000,
            "return_yen": 15_300,
            "profit_yen": 3_300,
            "roi": 1.275,
            "roi_without_largest_hit": 1.08,
            "effective_hit_count": 12.4,
            "promotion_eligible": False,
            "bootstrap": {
                "roi_ci95_lower": 0.91,
                "probability_roi_above_one": 0.88,
            },
            "log_loss": {
                "races": 612,
                "challenger_delta_vs_market": -0.018,
                "challenger_improvement_confidence": 0.97,
                "challenger_top5_improvement_confidence": 0.96,
            },
            "purchase_probability_calibration": {
                "selected_races": 88,
                "observed_hits": 14,
                "expected_hits": 13.2,
                "probability_at_most_observed_hits": 0.71,
                "method": "exact_poisson_binomial_lower_tail_over_disjoint_race_selections",
            },
            "promotion_gate": {
                "sample_size_pass": False,
                "roi_pass": True,
                "pass": False,
            },
        },
        "trend_point_odds_safety_sweep": {
            "status": "diagnostic_only_not_promotion_evidence",
            "selection_data_through": "2026-08-03",
            "next_registration_must_be_after": "2026-08-03",
            "rows": [{
                "odds_safety_factor": 1.1,
                "retrospective": {
                    "evaluation_days": 9,
                    "evaluated_races": 1379,
                    "tickets": 80,
                    "hit_tickets": 13,
                    "stake_yen": 9000,
                    "return_yen": 12000,
                    "profit_yen": 3000,
                    "roi": 1.333,
                    "roi_without_largest_hit": 1.14,
                    "effective_hit_count": 11.2,
                    "bootstrap": {
                        "roi_ci95_lower": 0.92,
                        "probability_roi_above_one": 0.89,
                    },
                    "daily": [{"must": "not leak into summary"}],
                },
            }],
        },
    })

    assert summary["trend_point_prospective_registered_after"] == "2026-08-03"
    conditional = summary["prequential_conditional_order"]
    assert conditional["transformed_races"] == 1630
    assert conditional["log_loss_difference"] == -0.0667
    assert conditional["improving_days"] == 11
    assert "daily" not in conditional
    assert summary["source_model_sha256"] == "a" * 64
    assert summary["trend_point_prospective_evaluation_days"] == 4
    assert summary["trend_point_prospective_tickets"] == 104
    assert summary["trend_point_prospective_roi"] == 1.275
    assert summary["trend_point_prospective_roi_without_largest_hit"] == 1.08
    assert summary[
        "trend_point_prospective_daily_cluster_bootstrap_roi_lower_95"
    ] == 0.91
    assert summary["trend_point_prospective_probability_roi_above_one"] == 0.88
    assert summary[
        "trend_point_prospective_market_challenger_improvement_confidence"
    ] == 0.97
    assert summary[
        "trend_point_prospective_market_challenger_top5_improvement_confidence"
    ] == 0.96
    assert summary[
        "trend_point_prospective_probability_calibration"
    ]["probability_at_most_observed_hits"] == 0.71
    assert summary["trend_point_prospective_promotion_gate"]["pass"] is False
    safety = summary["trend_point_odds_safety_sweep"]
    assert safety["selection_data_through"] == "2026-08-03"
    assert safety["rows"][0]["odds_safety_factor"] == 1.1
    assert safety["rows"][0]["retrospective"]["roi_ci95_lower"] == 0.92
    assert "daily" not in safety["rows"][0]["retrospective"]


def test_result_summary_preserves_bankroll_model_protocol() -> None:
    summary = summarize_result({
        "comparison_role": "bankroll_policy_model",
        "coefficient_optimizer": "newton_cg",
        "ev_calibration_mode": "contextual_point",
        "ev_calibration_usage": "prior_selection_fit_then_applied_to_policy_and_holdout",
        "evaluation_from": "2025-07-25",
        "evaluation_through": "2026-07-24",
        "selection_races": 74331,
        "holdout_races": 49506,
    })

    assert summary == {
        "comparison_role": "bankroll_policy_model",
        "coefficient_optimizer": "newton_cg",
        "ev_calibration_mode": "contextual_point",
        "ev_calibration_usage": "prior_selection_fit_then_applied_to_policy_and_holdout",
        "evaluation_from": "2025-07-25",
        "evaluation_through": "2026-07-24",
        "selection_races": 74331,
        "holdout_races": 49506,
    }


def test_result_summary_combines_bankroll_and_nested_prediction_metrics() -> None:
    summary = summarize_result({
        "roi": 0.92,
        "profit_yen": -17000,
        "selection_prediction_metrics": {
            "entry_log_loss": 0.400,
            "winner_top1_accuracy": 0.500,
            "trifecta_top5_hit_rate": 0.250,
        },
        "holdout_prediction_metrics": {
            "entry_log_loss": 0.327,
            "winner_top1_accuracy": 0.568,
            "trifecta_top5_hit_rate": 0.330,
        },
        "bankroll_confidence": {
            "roi_ci95_lower": 0.805,
            "roi_ci95_upper": 1.042,
            "probability_roi_above_one": 0.098,
        },
    })

    assert summary["roi"] == 0.92
    assert summary["entry_log_loss"] == 0.327
    assert summary["winner_top1_accuracy"] == 0.568
    assert summary["trifecta_top5_hit_rate"] == 0.330
    assert summary["roi_ci95_lower"] == 0.805
    assert summary["roi_ci95_upper"] == 1.042
    assert summary["probability_roi_above_one"] == 0.098


def test_result_summary_preserves_raw_archive_transfer_metrics() -> None:
    summary = summarize_result({
        "status": "completed",
        "source_files_before": 35,
        "source_files_after": 24,
        "source_bytes_before": 664383,
        "source_bytes_after": 506907,
        "archived_files_removed": 11,
        "archived_bytes_removed": 157476,
        "staging_files": 0,
    })

    assert summary["archived_files_removed"] == 11
    assert summary["archived_bytes_removed"] == 157476
    assert summary["source_files_after"] == 24
    assert summary["staging_files"] == 0


def test_result_summary_preserves_archive_closing_odds_counts() -> None:
    summary = summarize_result({
        "status": "completed",
        "source_role": "secondary_archive_candidate_unverified",
        "source_key": "archive-v1",
        "from_date": "2026-07-20",
        "through_date": "2026-07-27",
        "targets": 50,
        "stored": 48,
        "invalid": 2,
        "fetch_failed": 0,
        "not_found": 0,
        "remaining": 1223,
    })

    assert summary["archive_targets"] == 50
    assert summary["archive_stored"] == 48
    assert summary["archive_invalid"] == 2
    assert summary["archive_remaining"] == 1223


def test_standardized_manifest_summary_reports_selected_and_best_candidate() -> None:
    summary = summarize_result({
        "protocol_id": "standard_365d_v2",
        "comparison_ready": True,
        "valid_model_count": 3,
        "models": [
            {
                "model_id": "incumbent",
                "entry_log_loss": 0.37,
                "winner_top1_accuracy": 0.56,
                "trifecta_top5_hit_rate": 0.31,
                "roi": 0.76,
                "profit_yen": -300,
            },
            {"model_id": "candidate-a", "roi": 0.83, "profit_yen": -100},
            {"model_id": "candidate-b", "roi": 0.80, "profit_yen": -200},
        ],
        "promotion_decision": {
            "incumbent_model_id": "incumbent",
            "selected_model_id": "incumbent",
            "eligible_candidate_ids": [],
            "status": "retain_incumbent",
        },
    })

    assert summary == {
        "entry_log_loss": 0.37,
        "winner_top1_accuracy": 0.56,
        "trifecta_top5_hit_rate": 0.31,
        "roi": 0.76,
        "profit_yen": -300,
        "model": "incumbent",
        "best_candidate_model": "candidate-a",
        "best_candidate_roi": 0.83,
        "best_candidate_profit_yen": -100,
        "comparison_ready": True,
        "valid_model_count": 3,
        "promotion_eligible": False,
        "status": "retain_incumbent",
    }


def test_genetic_island_summary_exposes_live_champion_metrics() -> None:
    summary = summarize_result({
        "model": "genetic_listwise_island_v1-20260726T010000Z-g00-i00",
        "cohort": "20260726T010000Z",
        "generation": 0,
        "island_id": 0,
        "population_size": 8,
        "history": [{"local_generation": 0}, {"local_generation": 1}],
        "champion": {
            "fitness": -1.23,
            "metrics": {
                "entry_log_loss": 0.34,
                "winner_top1_accuracy": 0.56,
                "trifecta_top5_hit_rate": 0.31,
                "evaluated_races": 3000,
            },
        },
    })

    assert summary["genetic_fitness"] == -1.23
    assert summary["genetic_cohort"] == "20260726T010000Z"
    assert summary["genetic_generation"] == 0
    assert summary["genetic_island_id"] == 0
    assert summary["genetic_evaluated_individuals"] == 16
    assert summary["entry_log_loss"] == 0.34
    assert summary["winner_top1_accuracy"] == 0.56
    assert summary["trifecta_top5_hit_rate"] == 0.31


def test_repository_sync_summary_preserves_deferred_reason() -> None:
    summary = summarize_result({
        "status": "completed",
        "action": "deferred_active_evaluation",
        "ahead": 48,
        "behind": 0,
        "active_evaluations": 1,
    })

    assert summary == {
        "action": "deferred_active_evaluation",
        "ahead": 48,
        "behind": 0,
        "active_evaluations": 1,
        "status": "completed",
    }
    assert result_decision("repository_sync", summary) == (
        "repository_sync_deferred"
    )
    assert result_decision(
        "repository_sync", {"action": "fast_forwarded"}
    ) == "maintenance_complete"


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


def test_market_walk_forward_requires_explicit_promotion_eligibility() -> None:
    summary = summarize_result({
        "model": "market-candidate",
        "promotion_eligible": False,
        "roi": 7.42,
        "profit_yen": 3_210,
        "evaluation_races": 170,
        "evaluation_days": 1,
        "winner_log_loss": 1.10,
        "winner_top1_accuracy": 0.58,
    })

    assert summary["winner_log_loss"] == 1.10
    assert summary["winner_top1_accuracy"] == 0.58
    assert result_decision("market_residual_walk_forward", summary) == (
        "accumulate_formal_evidence"
    )
    summary["promotion_eligible"] = True
    assert (
        result_decision("market_residual_walk_forward", summary)
        == "promotion_candidate"
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
                "return_yen": 108_000,
                "max_drawdown_yen": 12_000,
                "roi_without_largest_hit": 1.001,
                "largest_hit_return_yen": 900,
                "largest_hit_return_share": 900 / 101_000,
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
    assert summary["payout_feature_candidate_return_yen"] == 108_000
    assert summary["payout_feature_candidate_max_drawdown_yen"] == 12_000
    assert (
        summary["payout_feature_candidate_schema"]
        == "conditional_payout_tail_v1"
    )
    assert summary["payout_feature_roi_ci95_lower"] == 1.01
    assert summary["payout_feature_probability_roi_above_one"] == 0.98
    assert summary["payout_feature_gate_pass"] is True
    assert summary["payout_feature_promotion_eligible"] is False
    assert (
        result_decision("conditional_payout_tail", summary)
        == "reject_or_research_only"
    )


def test_result_summary_exposes_expected_return_holdout_metrics() -> None:
    tail = {
        "odds_field": "estimated_odds_at_purchase",
        "purchased_tickets": 30,
        "normal": {"tickets": 20, "roi": 1.04},
        "tail": {"tickets": 10, "roi": 0.72},
    }
    summary = summarize_result({
        "expected_return_calibration": {
            "promotion_eligible": False,
            "bankroll": {
                "roi": 1.01,
                "profit_yen": 1_000,
                "stake_yen": 100_000,
                "return_yen": 101_000,
                "max_drawdown_yen": 12_000,
                "roi_without_largest_hit": 1.001,
                "largest_hit_return_yen": 900,
                "largest_hit_return_share": 900 / 101_000,
                "selected_tickets": 30,
                "races_bet": 24,
                "hit_tickets": 8,
                "effective_hit_count": 6.4,
                "winning_days": 18,
                "losing_days": 12,
                "tail_portfolio_diagnostics": tail,
                "policy_selection": {
                    "source": "pre_evaluation_temporal_selection",
                    "selected_ev_threshold": 1.3,
                },
            },
            "bankroll_confidence": {
                "roi_ci95_lower": 0.97,
                "probability_roi_above_one": 0.61,
            },
            "diagnostic_gate": {"pass": False, "roi_pass": True},
        },
        "expected_return_fixed_threshold": {
            "bankroll": {
                "roi": 0.96,
                "profit_yen": -4_000,
                "stake_yen": 100_000,
                "return_yen": 96_000,
                "roi_without_largest_hit": 0.91,
                "largest_hit_return_yen": 5_000,
                "largest_hit_return_share": 5_000 / 96_000,
                "selected_tickets": 40,
                "races_bet": 31,
                "hit_tickets": 9,
                "effective_hit_count": 7.1,
                "tail_portfolio_diagnostics": tail,
            },
            "bankroll_confidence": {"roi_ci95_lower": 0.91},
        },
    })

    assert summary["expected_return_candidate_roi"] == 1.01
    assert summary["expected_return_candidate_selected_tickets"] == 30
    assert summary["expected_return_candidate_roi_without_largest_hit"] == 1.001
    assert summary["expected_return_candidate_effective_hit_count"] == 6.4
    assert summary["expected_return_roi_ci95_lower"] == 0.97
    assert summary["expected_return_probability_roi_above_one"] == 0.61
    assert summary["expected_return_gate_pass"] is False
    assert summary["expected_return_promotion_eligible"] is False
    assert summary["expected_return_selected_ev_threshold"] == 1.3
    assert summary["expected_return_tail_portfolio_diagnostics"] == tail
    assert summary["expected_return_fixed_roi"] == 0.96
    assert summary["expected_return_fixed_roi_without_largest_hit"] == 0.91
    assert summary["expected_return_fixed_effective_hit_count"] == 7.1
    assert summary["expected_return_fixed_roi_ci95_lower"] == 0.91
    assert summary["expected_return_fixed_tail_portfolio_diagnostics"] == tail


def test_daily_market_seed_uses_fixed_completed_sources(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "data" / "models" / "evaluation_queue"
    model_dir.mkdir(parents=True)
    sources = {
        "calibrated_mlp_prediction_deployment": model_dir / "protected.json",
        "calibrated_mlp_recency_card_features": model_dir / "mlp.json",
        "calibrated_lightgbm_recency_period_v6_4cpu": model_dir / "lightgbm.json",
    }
    for path in sources.values():
        path.with_suffix(".joblib").write_bytes(b"model")
    sources["calibrated_mlp_prediction_deployment"].with_name(
        "protected.deployment.joblib"
    ).write_bytes(b"model")

    class FakeConn:
        parameters = ()

        def execute(self, _sql, parameters):
            self.parameters = parameters
            return self

        def fetchone(self):
            source_key = self.parameters[1]
            return {"result_path": str(sources[source_key])}

    conn = FakeConn()
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(evaluation_queue, "enqueue_job", fake_enqueue)

    inserted = seed_daily_market_jobs(
        conn, app_root=tmp_path, evaluation_date="2026-07-25"
    )

    assert inserted == list(range(1, 21))
    assert {row["model_key"] for row in calls} == {
        "protected_mlp_prediction:market_residual:20260718-25",
        "calibrated_mlp_recency_selected:market_residual:20260718-25",
        "calibrated_lightgbm_recency_selected:market_residual:20260718-25",
        "odds_path_operational_daily:market_residual:20260718-25",
        "odds_path_probability_only_daily:market_residual:20260718-25",
        "odds_path_observed_closing_return_v4_daily:market_residual:20260718-25",
        "odds_path_observed_closing_return_robust_policy_v17_daily:market_residual:20260718-25",
        "odds_path_observed_closing_return_schedule_quota_v18_daily:market_residual:20260718-25",
        "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19_daily:market_residual:20260718-25",
        "odds_path_observed_closing_return_schedule_quota_dual_head_v20_daily:market_residual:20260718-25",
        "odds_path_observed_closing_return_schedule_quota_triple_head_v21_daily:market_residual:20260718-25",
        "odds_path_prequential_shrinkage_return_v6_daily:market_residual:20260718-25",
        "odds_path_crossfit_conservative_ev_v7_daily:market_residual:20260718-25",
        "odds_path_market_offset_crossfit_conservative_ev_v8_daily:market_residual:20260718-25",
        "odds_path_market_offset_discrete_log_ev_v9_daily:market_residual:20260718-25",
        "odds_path_market_offset_selection_conformal_discrete_ev_v10_daily:market_residual:20260718-25",
        "odds_path_role_integrated_multihorizon_v11_daily:market_residual:20260718-25",
        "odds_path_role_integrated_t300_nonlinear_v12_daily:market_residual:20260718-25",
        "odds_path_role_integrated_edge_conditional_lcb_v13_daily:market_residual:20260718-25",
        "odds_path_role_integrated_registered_band_lcb_v14_daily:market_residual:20260718-25",
    }
    protected = next(
        row for row in calls if row["model_key"].startswith("protected_mlp_prediction:")
    )
    assert Path(protected["parameters"]["model_input"]).name == (
        "protected.deployment.joblib"
    )
    assert all(
        row["parameters"]["through_date"] == "2026-07-25"
        and row["parameters"]["from_date"] == "2026-07-18"
        for row in calls
    )
    odds_path = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"] == "odds_path_return"
    )
    assert odds_path["priority"] == 98
    assert odds_path["parameters"]["timeout_seconds"] == 7200
    probability_only = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"] == "odds_path_probability"
    )
    assert probability_only["priority"] == 98
    assert probability_only["parameters"]["timeout_seconds"] == 7200
    observed_closing = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_observed_closing_return"
    )
    assert observed_closing["priority"] == 96
    assert observed_closing["parameters"]["timeout_seconds"] == 3600
    v19 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19"
    )
    assert v19["priority"] == 96
    assert v19["parameters"]["timeout_seconds"] == 3600
    v20 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_observed_closing_return_schedule_quota_dual_head_v20"
    )
    assert v20["priority"] == 100
    assert v19["priority"] < v20["priority"]
    assert v20["parameters"]["timeout_seconds"] == 3600
    v21 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
    )
    assert v21["priority"] == 97
    assert v21["priority"] < v20["priority"]
    assert v21["parameters"]["timeout_seconds"] == 3600
    prequential_v6 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_prequential_shrinkage_return"
    )
    assert prequential_v6["priority"] == 95
    assert prequential_v6["parameters"]["timeout_seconds"] == 7200
    crossfit_v7 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_crossfit_conservative_ev"
    )
    assert crossfit_v7["priority"] == 94
    assert crossfit_v7["parameters"]["timeout_seconds"] == 7200
    crossfit_v8 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_market_offset_crossfit_conservative_ev"
    )
    assert crossfit_v8["priority"] == 93
    assert crossfit_v8["parameters"]["timeout_seconds"] == 7200
    discrete_v9 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_market_offset_discrete_log_ev_v9"
    )
    assert discrete_v9["priority"] == 92
    assert discrete_v9["parameters"]["timeout_seconds"] == 7200
    selection_v10 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    )
    assert selection_v10["priority"] == 91
    assert selection_v10["parameters"]["timeout_seconds"] == 14_400
    assert selection_v10["parameters"]["model_input"] == discrete_v9["parameters"]["model_input"]
    role_integrated_v11 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_role_integrated_multihorizon_v11"
    )
    assert role_integrated_v11["priority"] == 90
    assert role_integrated_v11["parameters"] == {
        "model_input": discrete_v9["parameters"]["model_input"],
        "from_date": "2026-07-18",
        "through_date": "2026-07-25",
        "daily_budget_yen": 10000,
        "min_calibration_days": 2,
        "calibrator_strategy": "odds_path_role_integrated_multihorizon_v11",
        "minimum_day_coverage": 1.0,
        "timeout_seconds": 14_400,
    }
    role_integrated_v12 = next(
        row for row in calls
        if row["parameters"]["calibrator_strategy"]
        == "odds_path_role_integrated_t300_nonlinear_v12"
    )
    assert role_integrated_v12["priority"] == 104
    assert role_integrated_v12["parameters"] == {
        "model_input": discrete_v9["parameters"]["model_input"],
        "from_date": "2026-07-18",
        "through_date": "2026-07-25",
        "daily_budget_yen": 10000,
        "min_calibration_days": 2,
        "calibrator_strategy": "odds_path_role_integrated_t300_nonlinear_v12",
        "minimum_day_coverage": 1.0,
        "v12_closing_fallback_policy": "v11",
        "timeout_seconds": 14_400,
    }
    assert seed_daily_market_jobs(
        conn, app_root=tmp_path, evaluation_date="2026-07-17"
    ) == []


def test_default_seed_contains_parameter_sweep(monkeypatch) -> None:
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr("boatrace_ai.evaluation_queue.enqueue_job", fake_enqueue)

    inserted = seed_default_jobs(object(), evaluation_date="2026-07-22")

    assert len(inserted) == 15
    standardized = [
        row for row in calls if row["task_type"] == "standardized_365d"
    ]
    assert standardized[0]["parameters"]["timeout_seconds"] == 86400
    assert standardized[0]["max_attempts"] == 3
    assert standardized[0]["priority"] == 70
    drop_base_mlp = next(
        row for row in calls
        if row["model_key"] == "calibrated_mlp_recency_drop_base_pastlog"
    )
    assert drop_base_mlp["task_type"] == "calibrated_mlp_recency_search"
    assert drop_base_mlp["parameters"] == {
        "evaluation_date": "2026-07-22",
        "half_lives": "none,180,365,730",
        "calibration_days": 180,
        "drop_feature_groups": "base_pastlog",
        "timeout_seconds": 86400,
    }
    assert drop_base_mlp["priority"] == 90
    assert drop_base_mlp["max_attempts"] == 3
    payout = next(row for row in calls if row["task_type"] == "conditional_payout_tail")
    assert payout["parameters"] == {
        "training_through": "2025-07-22",
        "evaluation_from": "2025-07-23",
        "evaluation_through": "2026-07-22",
        "timeout_seconds": 86400,
    }
    assert payout["priority"] == 90
    assert payout["max_attempts"] == 3
    research = next(
        row for row in calls
        if row["task_type"] == "historical_research_logit"
    )
    assert research["model_key"] == "no_odds_v9_research_logit"
    assert research["parameters"] == {
        "evaluation_date": "2026-07-22",
        "timeout_seconds": 86400,
    }
    assert research["priority"] == 89
    assert sum(row["task_type"] == "market_curvature" for row in calls) == 6
    assert sum(row["task_type"] == "listwise_feature_search" for row in calls) == 4
    combined = [row for row in calls if row["task_type"] == "combined_feature_search"]
    assert len(combined) == 1
    assert combined[0]["priority"] == 85
    assert combined[0]["model_key"] == "listwise_combined_8192"
    assert combined[0]["parameters"]["n_features"] == 8192
    assert sum(
        row["task_type"] == "calibrated_mlp_recency_search" for row in calls
    ) == 1
    assert all(
        row["parameters"]["evaluation_date"] == "2026-07-22"
        for row in calls
        if row["task_type"] != "conditional_payout_tail"
    )

    calls.clear()
    inserted = seed_default_jobs(
        object(),
        evaluation_date="2026-07-23",
        include_standardized=False,
    )
    assert len(inserted) == 14
    assert all(row["task_type"] != "standardized_365d" for row in calls)


def test_standardized_evaluation_due_uses_weekly_cadence() -> None:
    class Result:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class DueConnection:
        def __init__(self, row):
            self.row = row

        def execute(self, statement, parameters=()):
            assert "task_type = ?" in statement
            assert parameters == ("standardized_365d",)
            return Result(self.row)

    assert evaluation_queue.standardized_evaluation_due(
        DueConnection(None), evaluation_date="2026-07-28"
    )
    completed = DueConnection({
        "evaluation_date": "2026-07-21",
        "status": "completed",
    })
    assert not evaluation_queue.standardized_evaluation_due(
        completed, evaluation_date="2026-07-27"
    )
    assert evaluation_queue.standardized_evaluation_due(
        completed, evaluation_date="2026-07-28"
    )
    running = DueConnection({
        "evaluation_date": "2026-07-01",
        "status": "running",
    })
    assert not evaluation_queue.standardized_evaluation_due(
        running, evaluation_date="2026-07-28"
    )


class _PeriodicScheduleConnection:
    def __init__(self):
        self.keys: set[str] = set()

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        assert "dedupe_key = ?" in sql
        key, _task_type = parameters
        return _QueryResult({"count": int(key in self.keys)})


def test_periodic_seed_uses_low_backup_priority_and_skips_completed_bucket(
    monkeypatch,
) -> None:
    conn = _PeriodicScheduleConnection()
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        conn.keys.add(dedupe_key(
            kwargs["task_type"], kwargs["model_key"], kwargs["parameters"]
        ))
        return len(calls)

    monkeypatch.setattr(evaluation_queue, "enqueue_job", fake_enqueue)
    now = datetime(2026, 7, 23, 12, 34, tzinfo=timezone.utc)

    assert seed_periodic_jobs(conn, now=now) == [1, 2, 3, 4, 5]
    assert seed_periodic_jobs(conn, now=now) == []
    assert len(calls) == 5
    backup = next(row for row in calls if row["task_type"] == "gdrive_raw_archive")
    assert backup["priority"] == 10
    assert backup["parameters"]["schedule_bucket"] == (
        "2026-07-23T12:00:00+00:00"
    )
    series = next(row for row in calls if row["task_type"] == "series_feature_cache")
    assert series["priority"] == 45
    assert series["parameters"]["from_date"] == "2026-07-09"
    sync = next(row for row in calls if row["task_type"] == "repository_sync")
    assert sync["priority"] == 25


def test_periodic_enqueue_retains_atomic_dedupe_conflict_guard() -> None:
    class RecordingConnection:
        def __init__(self):
            self.sql = ""

        def execute(self, statement, parameters=()):
            self.sql = " ".join(statement.split())
            return _QueryResult()

    conn = RecordingConnection()

    assert enqueue_job(
        conn,
        task_type="gdrive_raw_archive",
        model_key="raw-data",
        parameters={"schedule_bucket": "2026-07-23T12:30:00+00:00"},
    ) is None
    assert "ON CONFLICT(dedupe_key) DO NOTHING" in conn.sql


@pytest.mark.parametrize(
    ("task_type", "model_key", "parameters", "identity"),
    [
        (
            "standardized_365d",
            "all_registered_models",
            {"evaluation_date": "2026-07-23", "timeout_seconds": 86400},
            {"evaluation_date": "2026-07-23"},
        ),
        (
            "conditional_payout_tail",
            "listwise-conditional-payout",
            {
                "training_through": "2025-07-23",
                "evaluation_from": "2025-07-24",
                "evaluation_through": "2026-07-23",
                "timeout_seconds": 86400,
            },
            {
                "training_through": "2025-07-23",
                "evaluation_from": "2025-07-24",
                "evaluation_through": "2026-07-23",
            },
        ),
        (
            "market_residual_walk_forward",
            "odds-path-v21:market_residual:20260718-29",
            {
                "model_input": "data/models/evaluation_queue/job-00002707.joblib",
                "from_date": "2026-07-18",
                "through_date": "2026-07-29",
                "daily_budget_yen": 10000,
                "min_calibration_days": 2,
                "calibrator_strategy": (
                    "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
                ),
                "minimum_day_coverage": 1.0,
                "timeout_seconds": 14400,
            },
            {
                "model_input": "data/models/evaluation_queue/job-00002707.joblib",
                "from_date": "2026-07-18",
                "through_date": "2026-07-29",
                "daily_budget_yen": 10000,
                "min_calibration_days": 2,
                "calibrator_strategy": (
                    "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
                ),
                "minimum_day_coverage": 1.0,
            },
        ),
    ],
)
def test_long_evaluation_enqueue_semantically_dedupes_retry_parameter_changes(
    task_type,
    model_key,
    parameters,
    identity,
) -> None:
    class RecordingConnection:
        def __init__(self):
            self.inserted = False

        def execute(self, statement, values=()):
            sql = " ".join(statement.split())
            if sql.startswith("SELECT job_id"):
                assert "parameters @> CAST(? AS JSONB)" in sql
                assert values == (
                    task_type,
                    model_key,
                    json.dumps(
                        identity,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                return _QueryResult({"job_id": 387})
            if sql.startswith("INSERT INTO model_evaluation_jobs"):
                self.inserted = True
            raise AssertionError(f"unexpected SQL: {sql}")

    conn = RecordingConnection()

    assert enqueue_job(
        conn,
        task_type=task_type,
        model_key=model_key,
        parameters=parameters,
    ) is None
    assert conn.inserted is False


def test_leader_commits_maintenance_before_claim(monkeypatch, tmp_path) -> None:
    events = []
    connection_count = 0

    class Scope:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return object()

        def __exit__(self, exc_type, exc, traceback):
            events.append(f"commit:{self.name}")

    def fake_connection(_db):
        nonlocal connection_count
        names = ("startup", "maintenance", "claim")
        name = names[connection_count]
        connection_count += 1
        return Scope(name)

    monkeypatch.setattr(evaluation_queue, "connection", fake_connection)
    monkeypatch.setattr(evaluation_queue, "ensure_schema", lambda _conn: None)
    monkeypatch.setattr(evaluation_queue, "recover_worker_job", lambda *_a, **_k: 0)
    monkeypatch.setattr(evaluation_queue, "seed_work_tickets", lambda _conn: 0)
    monkeypatch.setattr(
        evaluation_queue,
        "reconcile_completed_job_runs",
        lambda *_a, **_k: events.append("recover-completed"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "requeue_stale_jobs",
        lambda *_a, **_k: events.append("requeue"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "reconcile_queue_state",
        lambda *_a, **_k: events.append("reconcile"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "reconcile_refined_market_evaluations",
        lambda *_a, **_k: events.append("reconcile-refined"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_default_jobs",
        lambda *_a, **_k: events.append("seed-defaults"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "cancel_superseded_daily_jobs",
        lambda *_a, **_k: events.append("cancel-superseded"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "standardized_evaluation_due",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_daily_market_jobs",
        lambda *_a, **_k: events.append("seed-market"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_daily_genetic_jobs",
        lambda *_a, **_k: events.append("seed-genetic"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "genetic_cache_evaluation_date",
        lambda _root: (
            evaluation_queue.datetime.now(evaluation_queue.JST).date()
            - evaluation_queue.timedelta(days=1)
        ).isoformat(),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_periodic_jobs",
        lambda *_a, **_k: events.append("seed-periodic"),
    )
    resources = ResourceSnapshot(32000, 10000, 100.0, 16, 0.0)
    monkeypatch.setattr(
        evaluation_queue, "system_resources", lambda **_kwargs: resources
    )
    monkeypatch.setattr(
        evaluation_queue,
        "claim_job",
        lambda *_a, **_k: events.append("claim") or None,
    )
    monkeypatch.setattr(evaluation_queue.time, "monotonic", lambda: 10000.0)
    args = evaluation_queue.build_parser().parse_args([
        "run",
        "--db", "postgresql://test",
        "--app-root", str(tmp_path),
        "--python", "python",
        "--worker-id", "evaluator-00",
        "--seed-defaults",
        "--schedule-periodic",
        "--once",
    ])

    assert evaluation_queue.run_worker(args) == 0
    assert events.index("recover-completed") < events.index("requeue")
    assert events.index("recover-completed") < events.index("commit:maintenance")
    assert events.index("commit:maintenance") < events.index("enter:claim")
    assert events.index("seed-market") < events.index("commit:maintenance")
    assert events.index("reconcile") < events.index("commit:maintenance")
    assert events.index("cancel-superseded") < events.index("seed-market")
    assert events.index("seed-genetic") < events.index("commit:maintenance")
    assert events.index("seed-periodic") < events.index("commit:maintenance")
    assert events.index("enter:claim") < events.index("claim")


def test_scheduler_seeds_without_claiming_jobs(monkeypatch, tmp_path) -> None:
    events = []

    class Scope:
        def __enter__(self):
            events.append("enter")
            return object()

        def __exit__(self, exc_type, exc, traceback):
            events.append("commit")

    monkeypatch.setattr(evaluation_queue, "connection", lambda _db: Scope())
    monkeypatch.setattr(evaluation_queue, "ensure_schema", lambda _conn: None)
    monkeypatch.setattr(
        evaluation_queue,
        "seed_work_tickets",
        lambda _conn: events.append("seed-work"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "reconcile_completed_job_runs",
        lambda *_a, **_k: events.append("recover-completed"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "requeue_stale_jobs",
        lambda *_a, **_k: events.append("requeue"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "reconcile_queue_state",
        lambda *_a, **_k: events.append("reconcile"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_default_jobs",
        lambda *_a, **_k: events.append("seed-defaults"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "cancel_superseded_daily_jobs",
        lambda *_a, **_k: events.append("cancel-superseded"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "standardized_evaluation_due",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_daily_market_jobs",
        lambda *_a, **_k: events.append("seed-market"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "genetic_cache_evaluation_date",
        lambda _root: None,
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_periodic_jobs",
        lambda *_a, **_k: events.append("seed-periodic"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "claim_job",
        lambda *_a, **_k: pytest.fail("scheduler must not claim evaluation jobs"),
    )
    monkeypatch.setattr(evaluation_queue.time, "monotonic", lambda: 10000.0)
    args = evaluation_queue.build_parser().parse_args([
        "schedule",
        "--db", "postgresql://test",
        "--app-root", str(tmp_path),
        "--seed-defaults",
        "--once",
    ])

    assert evaluation_queue.run_scheduler(args) == 0
    assert events == [
        "enter", "seed-work", "commit",
        "enter", "recover-completed", "requeue", "reconcile", "seed-defaults",
        "cancel-superseded", "seed-market", "seed-periodic", "commit",
    ]


def test_supervisor_runs_four_postgresql_queue_workers() -> None:
    config = Path(
        "scripts/deployment/supervisor-boatrace-evaluation-runner.ini"
    ).read_text(encoding="utf-8")

    assert "boatrace_ai.evaluation_queue run" in config
    assert "numprocs=4" in config
    assert "--seed-defaults" not in config
    assert "--schedule-periodic" not in config
    assert "--vm-limit-gib 0" in config
    assert 'BOATRACE_PG_APPLICATION_NAME="boatrace_evaluator"' in config
    assert 'BOATRACE_PG_WORK_MEM="128MB"' in config

    scheduler = Path(
        "scripts/deployment/supervisor-boatrace-evaluation-scheduler.ini"
    ).read_text(encoding="utf-8")
    assert "boatrace_ai.evaluation_queue schedule" in scheduler
    assert "--seed-defaults" in scheduler
    assert "--schedule-interval 60" in scheduler
    assert 'BOATRACE_PG_APPLICATION_NAME="boatrace_scheduler"' in scheduler


def test_worker_sets_database_memory_defaults_without_overriding(monkeypatch) -> None:
    monkeypatch.delenv("BOATRACE_PG_APPLICATION_NAME", raising=False)
    monkeypatch.setenv("BOATRACE_PG_WORK_MEM", "256MB")

    evaluation_queue._configure_worker_database_memory()

    assert os.environ["BOATRACE_PG_APPLICATION_NAME"] == "boatrace_evaluator"
    assert os.environ["BOATRACE_PG_WORK_MEM"] == "256MB"


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


def test_feature_search_profiles_fit_the_32gb_quota_and_migrate_old_defaults() -> None:
    assert TASK_PROFILES["standardized_365d"]["memory_mb"] == 14336
    assert TASK_PROFILES["listwise_feature_search"]["memory_mb"] == 14336
    assert TASK_PROFILES["combined_feature_search"] == {
        "category": "evaluation",
        "memory_mb": 14336,
        "disk_mb": 4096,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }

    class RecordingPostgres:
        dialect = "postgresql"

        def __init__(self):
            self.calls = []

        def execute(self, statement, params=()):
            self.calls.append((statement, params))

        def executescript(self, statement):
            self.calls.append((statement, ()))

    conn = RecordingPostgres()
    ensure_schema(conn)
    migration = next(
        (statement, params)
        for statement, params in conn.calls
        if "status = 'queued'" in statement
        and "min_free_memory_mb = ?" in statement
    )
    assert migration[1] == (
        14336,
        "standardized_365d",
        "listwise_feature_search",
        16384,
    )
    lightgbm_memory_migration = next(
        (statement, params)
        for statement, params in conn.calls
        if params == (14336, "lightgbm_recency_search", 65536)
    )
    assert "status = 'queued'" in lightgbm_memory_migration[0]
    lightgbm_worker_migration = next(
        (statement, params)
        for statement, params in conn.calls
        if params == (4, "lightgbm_recency_search", "16")
    )
    assert "jsonb_set" in lightgbm_worker_migration[0]
    assert "status = 'queued'" in lightgbm_worker_migration[0]

class _QueryResult:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class _ClaimConnection:
    def __init__(self, state):
        self.state = state
        self.saved_timeouts = []
        self.candidate_sql = ""
        self.update_sql = ""
        self.events = []

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        self.events.append(sql)
        if "pg_advisory_xact_lock" in sql or sql.startswith("LOCK TABLE"):
            return _QueryResult()
        if "SELECT jobs.*" in sql:
            self.candidate_sql = sql
            return _QueryResult(dict(self.state))
        if "UPDATE model_evaluation_jobs" in sql and "RETURNING *" in sql:
            self.update_sql = sql
            saved = json.loads(parameters[1])
            self.saved_timeouts.append(saved["timeout_seconds"])
            self.state.update({
                "status": "running",
                "worker_id": parameters[0],
                "attempt": int(self.state["attempt"]) + 1,
                "parameters": saved,
                "error": None,
            })
            return _QueryResult(dict(self.state))
        if "INSERT INTO model_evaluation_job_runs" in sql:
            return _QueryResult()
        raise AssertionError(f"unexpected SQL: {sql}")


class _LifecycleConnection:
    def __init__(self, parent):
        self.parent = dict(parent)
        self.run = {
            "status": "running",
            "error": None,
        }

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        if "UPDATE model_evaluation_jobs" in sql:
            if "max_attempts = max_attempts + 1" in sql:
                self.parent.update({
                    "status": "queued",
                    "max_attempts": int(self.parent["max_attempts"]) + 1,
                    "error": parameters[0],
                })
            else:
                self.parent.update({
                    "status": parameters[0],
                    "error": parameters[2],
                })
            return _QueryResult()
        if "UPDATE model_evaluation_job_runs" in sql:
            self.run.update({
                "status": "failed",
                "error": parameters[0],
            })
            return _QueryResult()
        raise AssertionError(f"unexpected SQL: {sql}")


def test_high_memory_evaluations_reserve_six_gib_for_services() -> None:
    resources = ResourceSnapshot(
        available_memory_mb=28000,
        available_disk_mb=10000,
        idle_cpu_percent=100.0,
        cpu_count=16,
        load_1m=0.0,
        memory_limit_mb=32000,
    )

    assert evaluation_queue._evaluation_reservation_mb(resources) == 25856


def test_timeout_retry_doubles_once_when_job_387_is_next_claimed() -> None:
    state = {
        "job_id": 387,
        "task_type": "listwise_feature_search",
        "category": "evaluation",
        "model_key": "feature-search",
        "parameters": {"timeout_seconds": 28800},
        "status": "queued",
        "attempt": 1,
        "max_attempts": 3,
        "error": "TimeoutExpired: command timed out after 21600 seconds",
    }
    conn = _ClaimConnection(state)
    resources = ResourceSnapshot(
        available_memory_mb=32000,
        available_disk_mb=10000,
        idle_cpu_percent=100.0,
        cpu_count=16,
        load_1m=0.0,
    )

    claimed = claim_job(conn, worker_id="evaluator-00", resources=resources)

    assert claimed is not None
    assert claimed["parameters"]["timeout_seconds"] == 43200
    assert conn.saved_timeouts == [43200]
    assert conn.events[0].startswith("SELECT pg_advisory_xact_lock")
    assert conn.events[1].startswith("SELECT jobs.*")
    assert not any(event.startswith("LOCK TABLE") for event in conn.events)
    assert "jobs.parent_job_id IS NULL" in conn.candidate_sql
    assert "parent.status = 'completed'" in conn.candidate_sql
    assert "SUM(running.min_free_memory_mb)" in conn.candidate_sql
    assert "model_evaluation_control" in conn.candidate_sql
    assert "control.enabled = TRUE" in conn.candidate_sql
    assert "started_at = CURRENT_TIMESTAMP" in conn.update_sql

    state.update({
        "status": "queued",
        "max_attempts": 4,
        "error": None,
    })
    claimed_again = claim_job(
        conn,
        worker_id="evaluator-00",
        resources=resources,
    )

    assert claimed_again is not None
    assert claimed_again["parameters"]["timeout_seconds"] == 43200
    assert conn.saved_timeouts == [43200, 43200]


def test_recover_worker_closes_interrupted_attempt() -> None:
    events = []

    class Result:
        def __init__(self, rows=()):
            self.rows = list(rows)

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, statement, parameters=()):
            sql = " ".join(statement.split())
            events.append((sql, parameters))
            if "UPDATE model_evaluation_jobs" in sql:
                return Result([{"job_id": 3566, "attempt": 2}])
            if "UPDATE model_evaluation_job_runs" in sql:
                return Result()
            raise AssertionError(f"unexpected SQL: {sql}")

    recovered = evaluation_queue.recover_worker_job(
        Connection(),
        worker_id="evaluator-02",
    )

    assert recovered == 1
    run_sql, run_parameters = events[1]
    assert "status = 'failed'" in run_sql
    assert "completed_at = CURRENT_TIMESTAMP" in run_sql
    assert run_parameters == (3566, 2)


def test_requeue_stale_jobs_closes_matching_running_attempt() -> None:
    events = []

    class Result:
        def __init__(self, rows=()):
            self.rows = list(rows)

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, statement, parameters=()):
            sql = " ".join(statement.split())
            events.append((sql, parameters))
            if "UPDATE model_evaluation_jobs" in sql:
                return Result([{"job_id": 3566, "attempt": 2}])
            if "UPDATE model_evaluation_job_runs" in sql:
                return Result()
            raise AssertionError(f"unexpected SQL: {sql}")

    requeued = evaluation_queue.requeue_stale_jobs(
        Connection(),
        stale_minutes=90,
    )

    assert requeued == 1
    assert len(events) == 1
    sql, parameters = events[0]
    assert "WITH locked_jobs AS MATERIALIZED" in sql
    assert "ORDER BY job_id FOR UPDATE SKIP LOCKED" in sql
    assert "), stale_jobs AS" in sql
    assert "UPDATE model_evaluation_jobs" in sql
    assert "UPDATE model_evaluation_job_runs AS runs" in sql
    assert "RETURNING jobs.job_id, jobs.attempt" in sql
    assert "runs.status = 'running'" in sql
    assert "runs.job_id = stale_jobs.job_id" in sql
    assert "runs.attempt = stale_jobs.attempt" in sql
    assert "completed_at = CURRENT_TIMESTAMP" in sql
    assert parameters == (90, "worker lease expired", "worker lease expired")


def test_reconcile_queue_state_only_cancels_exhausted_queued_jobs() -> None:
    events = []

    class Result:
        def fetchall(self):
            return [{"job_id": 3564}]

    class Connection:
        def execute(self, statement, parameters=()):
            events.append((" ".join(statement.split()), parameters))
            return Result()

    cancelled = evaluation_queue.reconcile_queue_state(Connection())

    assert cancelled == 1
    sql, parameters = events[0]
    assert "SET status = 'cancelled'" in sql
    assert "completed_at = CURRENT_TIMESTAMP" in sql
    assert "WHERE status = 'queued' AND attempt >= max_attempts" in sql
    assert "WITH orphaned_runs AS" in sql
    assert "UPDATE model_evaluation_job_runs AS runs" in sql
    assert "runs.status = 'running'" in sql
    assert "jobs.status = 'running'" in sql
    assert "jobs.attempt = runs.attempt" in sql
    assert parameters == (
        "queue reconciliation closed orphaned running attempt",
        "queue reconciliation cancelled exhausted job: "
        "attempt reached max_attempts",
    )


def test_cancel_superseded_daily_jobs_requires_exact_newer_track() -> None:
    events = []

    class Result:
        def fetchall(self):
            return [{"job_id": 4007}, {"job_id": 4009}]

    class Connection:
        def execute(self, statement, parameters=()):
            events.append((" ".join(statement.split()), parameters))
            return Result()

    cancelled = evaluation_queue.cancel_superseded_daily_jobs(
        Connection(),
        evaluation_date="2026-07-27",
    )

    assert cancelled == [4007, 4009]
    sql, parameters = events[0]
    assert "old.status = 'queued'" in sql
    assert "newer.task_type = old.task_type" in sql
    assert "newer.model_key = old.model_key" in sql
    assert "newer.status IN ('queued', 'running', 'completed')" in sql
    assert "superseded_by_newer_daily_evaluation" in sql
    assert "RETURNING old.job_id" in sql
    assert parameters == ("2026-07-27", "2026-07-27")


def test_timeout_retry_never_shortens_a_larger_configured_limit() -> None:
    parameters = evaluation_queue._timeout_retry_parameters(
        {"timeout_seconds": 86400},
        task_type="standardized_365d",
        previous_error="TimeoutExpired: command timed out after 21600 seconds",
    )

    assert parameters["timeout_seconds"] == 86400


def test_dependency_defer_preserves_job_1069_remaining_attempt() -> None:
    job = {
        "job_id": 1069,
        "attempt": 2,
        "max_attempts": 2,
    }
    conn = _LifecycleConnection(job)

    defer_job(
        conn,
        job=job,
        reason="selected standardized feature cache is not available yet",
    )

    assert conn.parent["status"] == "queued"
    assert conn.parent["max_attempts"] == 3
    assert conn.parent["max_attempts"] - job["attempt"] == 1
    assert conn.run["status"] == "failed"
    assert conn.parent["error"].startswith("dependency deferred:")
    assert conn.run["error"] == conn.parent["error"]


def test_invalid_artifact_uses_normal_terminal_failure() -> None:
    job = {
        "job_id": 1069,
        "attempt": 2,
        "max_attempts": 2,
    }
    conn = _LifecycleConnection(job)

    fail_job(
        conn,
        job=job,
        error="ValueError: standardized feature artifact is incomplete or invalid",
    )

    assert conn.parent["status"] == "failed"
    assert conn.parent["max_attempts"] == 2
    assert conn.run["status"] == "failed"
    assert conn.parent["error"].startswith("ValueError:")


class _ReprioritizeConnection:
    def __init__(self, *, job_status: str = "queued", ticket_exists: bool = True):
        self.job = {"job_id": 1069, "priority": 70, "status": job_status}
        self.ticket = (
            {"ticket_key": "MODEL-PAYOUT-001", "progress": 65}
            if ticket_exists
            else None
        )
        self.events = []

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        if "UPDATE model_evaluation_jobs" in sql:
            priority, job_id = parameters
            if int(job_id) != self.job["job_id"] or self.job["status"] != "queued":
                return _QueryResult()
            self.job["priority"] = int(priority)
            return _QueryResult(dict(self.job))
        if "UPDATE work_tickets" in sql:
            if self.ticket is None or parameters[0] != self.ticket["ticket_key"]:
                return _QueryResult()
            self.ticket["progress"] = max(int(self.ticket["progress"]), 70)
            return _QueryResult(dict(self.ticket))
        if "INSERT INTO work_ticket_events" in sql:
            self.events.append(parameters)
            return _QueryResult()
        raise AssertionError(f"unexpected SQL: {sql}")


def test_reprioritize_job_is_bounded_audited_and_parser_exposed() -> None:
    args = evaluation_queue.build_parser().parse_args([
        "reprioritize",
        "--job-id", "1069",
        "--priority", "90",
        "--reason", "Run payout policy before recency search",
        "--ticket-key", "MODEL-PAYOUT-001",
    ])
    conn = _ReprioritizeConnection()

    result = evaluation_queue.reprioritize_job(
        conn,
        job_id=args.job_id,
        priority=args.priority,
        reason=args.reason,
        ticket_key=args.ticket_key,
    )

    assert result == {"job_id": 1069, "priority": 90, "status": "queued"}
    assert conn.ticket["progress"] == 70
    assert conn.events == [
        (
            "MODEL-PAYOUT-001",
            70,
            "Run payout policy before recency search",
        )
    ]


@pytest.mark.parametrize(
    ("job_id", "priority", "reason", "message"),
    [
        (0, 90, "reason", "job_id"),
        (1069, -1, "reason", "priority"),
        (1069, 1001, "reason", "priority"),
        (1069, 90, " ", "reason"),
        (1069, 90, "x" * 501, "reason"),
    ],
)
def test_reprioritize_job_rejects_unbounded_input(
    job_id, priority, reason, message
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluation_queue.reprioritize_job(
            _ReprioritizeConnection(),
            job_id=job_id,
            priority=priority,
            reason=reason,
        )


def test_reprioritize_job_requires_queued_job_and_known_ticket() -> None:
    with pytest.raises(ValueError, match="queued"):
        evaluation_queue.reprioritize_job(
            _ReprioritizeConnection(job_status="running"),
            job_id=1069,
            priority=90,
            reason="reason",
        )
    with pytest.raises(ValueError, match="unknown ticket"):
        evaluation_queue.reprioritize_job(
            _ReprioritizeConnection(ticket_exists=False),
            job_id=1069,
            priority=90,
            reason="reason",
            ticket_key="MODEL-PAYOUT-001",
        )


def test_combined_feature_search_command_is_fixed_and_isolated(tmp_path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job(
            "combined_feature_search",
            {
                "evaluation_date": "2026-07-23",
                "n_features": 4096,
                "targets": "winner,top3_pl",
                "alphas": "0.00001,0.0001",
                "timeout_seconds": 21600,
            },
            job_id=77,
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command[0:3] == [
        str(python),
        "-m",
        "boatrace_ai.listwise.combined_feature_search",
    ]
    assert command[command.index("--cache-dir") + 1] == (
        "/tmp/boatrace-evaluation/job-00000077/combined"
    )
    assert command[command.index("--selected-cache-dir") + 1] == str(
        root / "data/models/evaluation_cache/job-00000077-combined"
    )
    assert command[command.index("--variant-workers") + 1] == "1"
    assert command[command.index("--candidate-workers") + 1] == "2"
    assert command[command.index("--as-of-date") + 1] == "2026-07-23"
    assert output == root / "data/models/evaluation_queue/job-00000077.json"
    assert result_decision("combined_feature_search", {"roi": 0.8}) == (
        "refine_selected_candidate"
    )
    assert result_decision(
        "combined_feature_search",
        {"roi": 1.2, "profit_yen": 2000, "promotion_eligible": True},
    ) == "refine_selected_candidate"


def test_newton_refinement_enqueues_market_evaluation_from_its_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    result = (
        tmp_path
        / "data/models/evaluation_queue/job-00009840.json"
    )
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"as_of_date": "2026-07-31"}),
        encoding="utf-8",
    )
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return 10440

    monkeypatch.setattr(evaluation_queue, "enqueue_job", fake_enqueue)

    job_id = enqueue_refined_market_evaluation(
        object(),
        {
            "job_id": 10027,
            "task_type": "listwise_newton_refine",
            "model_key": "listwise_combined_8192:newton",
            "priority": 86,
            "parameters": {
                "search_result": (
                    "data/models/evaluation_queue/job-00009840.json"
                )
            },
        },
        app_root=tmp_path,
    )

    assert job_id == 10440
    assert calls == [{
        "task_type": "market_residual_walk_forward",
        "model_key": (
            "listwise_combined_8192:newton:v21_market:20260718-31"
        ),
        "parameters": {
            "model_input": (
                "data/models/evaluation_queue/job-00010027.joblib"
            ),
            "from_date": "2026-07-18",
            "through_date": "2026-07-31",
            "daily_budget_yen": 10000,
            "calibrator_strategy": (
                "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
            ),
            "min_calibration_days": 2,
            "minimum_day_coverage": 1.0,
            "timeout_seconds": 7200,
        },
        "priority": 88,
        "max_attempts": 2,
        "parent_job_id": 10027,
    }]


def test_reconcile_recovers_refinement_completed_before_worker_reload(
    monkeypatch, tmp_path: Path
) -> None:
    completed = {
        "job_id": 10463,
        "task_type": "listwise_newton_refine",
        "model_key": "candidate:newton",
        "priority": 81,
        "parameters": {
            "search_result": "data/models/evaluation_queue/job-00009837.json"
        },
    }

    class Result:
        def fetchall(self):
            return [completed]

    class Connection:
        def execute(self, sql):
            assert "NOT EXISTS" in sql
            assert "INTERVAL '48 hours'" in sql
            assert "market_residual_walk_forward" in sql
            return Result()

    calls = []

    def fake_enqueue(conn, job, *, app_root):
        calls.append((conn, job, app_root))
        return 10470

    monkeypatch.setattr(
        evaluation_queue,
        "enqueue_refined_market_evaluation",
        fake_enqueue,
    )
    conn = Connection()

    assert reconcile_refined_market_evaluations(
        conn, app_root=tmp_path
    ) == [10470]
    assert calls == [(conn, completed, tmp_path)]


def test_listwise_feature_search_command_preserves_fixed_loss_blend(tmp_path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, _output = build_command(
        _job(
            "listwise_feature_search",
            {
                "evaluation_date": "2026-07-29",
                "targets": "top3_pl",
                "loss_blend": 0.375,
                "timeout_seconds": 21600,
            },
            job_id=88,
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command[command.index("--targets") + 1] == "top3_pl"
    assert command[command.index("--loss-blend") + 1] == "0.375"


def test_genetic_island_command_preserves_full_day_embargo(tmp_path) -> None:
    command, _output = build_command(
        _job(
            "genetic_island_search",
            {
                "evaluation_date": "2026-07-29",
                "cohort": "ga-v4-test",
                "generation": 0,
                "island_id": 0,
                "island_count": 2,
                "seed": 17,
                "train_races": 2000,
                "validation_races": 500,
                "embargo_days": 2,
            },
            job_id=89,
        ),
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--embargo-days") + 1] == "2"


@pytest.mark.parametrize("parameter", ["variant_workers", "candidate_workers", "cache_dir"])
def test_combined_feature_search_rejects_injected_worker_or_path(
    tmp_path, parameter
) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported combined_feature_search parameters",
    ):
        build_command(
            _job(
                "combined_feature_search",
                {"evaluation_date": "2026-07-23", parameter: 2},
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )

def test_lightgbm_recency_search_profile() -> None:
    assert TASK_PROFILES["lightgbm_recency_search"] == {
        "category": "evaluation",
        "memory_mb": 14336,
        "disk_mb": 1024,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }


def test_lightgbm_recency_search_command_is_fixed(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job(
            "lightgbm_recency_search",
            {
                "evaluation_date": "2026-07-24",
                "half_lives": "none,365",
                "drop_feature_groups": "legacy_composites",
                "n_estimators": 400,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 200,
                "feature_fraction": 0.8,
                "max_bin": 127,
                "n_jobs": 24,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command[0:3] == [
        str(python),
        "-m",
        "boatrace_ai.lightgbm_recency_evaluation",
    ]

    single_half_life_command, _ = build_command(
        _job(
            "lightgbm_recency_search",
            {
                "evaluation_date": "2026-07-24",
                "half_lives": "none",
                "architecture_presets": "compact,balanced,interaction",
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert single_half_life_command[
        single_half_life_command.index("--half-lives") + 1
    ] == "none"
    assert command[command.index("--feature-cache") + 1].endswith(
        "lightgbm_v6_features_16384_drop_legacy_composites"
    )
    assert "--no-write-feature-cache" in command
    assert command[command.index("--drop-feature-groups") + 1] == (
        "legacy_composites"
    )
    expected = {
        "--half-lives": "none,365",
        "--n-estimators": "400",
        "--num-leaves": "63",
        "--max-depth": "8",
        "--min-child-samples": "200",
        "--feature-fraction": "0.8",
        "--max-bin": "127",
        "--n-jobs": "24",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_lightgbm_recency_search_accepts_card_feature_ablation(
    tmp_path: Path,
) -> None:
    command, _ = build_command(
        _job(
            "lightgbm_recency_search",
            {
                "evaluation_date": "2026-07-31",
                "drop_feature_groups": (
                    "card_identity_context,card_relative,"
                    "research_correlates"
                ),
            },
        ),
        app_root=tmp_path,
        python=tmp_path / "python",
        db="postgresql://test",
    )

    assert command[command.index("--drop-feature-groups") + 1] == (
        "card_identity_context,card_relative,research_correlates"
    )
    assert command[command.index("--feature-cache") + 1].endswith(
        "drop_card_identity_context_card_relative_research_correlates"
    )


def test_lightgbm_structural_presets_are_forwarded(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    incumbent = root / "data/models/evaluation_queue/job-00002707.json"
    incumbent.parent.mkdir(parents=True)
    incumbent.write_text("{}", encoding="utf-8")
    command, _ = build_command(
        _job(
            "lightgbm_recency_search",
            {
                "evaluation_date": "2026-07-24",
                "architecture_presets": "compact,balanced,interaction",
                "selection_entry_log_loss_tolerance": 0.0005,
                "incumbent_result": (
                    "data/models/evaluation_queue/job-00002707.json"
                ),
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--architecture-presets") + 1] == (
        "compact,balanced,interaction"
    )
    assert command[
        command.index("--selection-entry-log-loss-tolerance") + 1
    ] == "0.0005"
    assert command[command.index("--incumbent-prediction") + 1] == str(
        incumbent
    )
    assert command[command.index("--incumbent-bankroll") + 1] == str(
        incumbent
    )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "evaluation_date is required"),
        ({"evaluation_date": "2026-07-24", "max_depth": 0}, "max_depth"),
        ({"evaluation_date": "2026-07-24", "n_estimators": 9}, "n_estimators"),
        (
            {"evaluation_date": "2026-07-24", "drop_feature_groups": "future"},
            "unknown",
        ),
        (
            {"evaluation_date": "2026-07-24", "command": "arbitrary"},
            "unsupported",
        ),
    ],
)
def test_lightgbm_recency_search_rejects_invalid_parameters(
    tmp_path: Path,
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_command(
            _job("lightgbm_recency_search", parameters),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_market_residual_walk_forward_command_is_fixed(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    model = root / "data/models/evaluation_queue/job-00002606.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"artifact")
    v25_artifact = model.with_suffix(".json")
    v25_artifact.write_text("{}", encoding="utf-8")
    python = root / ".venv/bin/python"

    command, output = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": "data/models/evaluation_queue/job-00002606.joblib",
                "from_date": "2026-07-18",
                "through_date": "2026-07-24",
                "calibrator_strategy": "newton_residual",
                "v25_probability_artifact": "data/models/evaluation_queue/job-00002606.json",
                "closing_odds_min_training_days": 4,
                "closing_odds_min_training_races": 250,
                "trend_point_odds_safety_sweep": True,
                "trend_point_required_ticket_count": 2,
                "prequential_conditional_order": True,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert TASK_PROFILES["market_residual_walk_forward"] == {
        "category": "evaluation",
        "memory_mb": 8192,
        "disk_mb": 256,
        "idle_cpu": 5.0,
        "max_parallel": 2,
    }
    assert command[:3] == [
        str(python),
        "-m",
        "boatrace_ai.listwise.market_calibration",
    ]
    expected_cache = (
        root / "data/models/evaluation_cache/market_scored"
        / "job-00002606_2026-07-18_2026-07-24.races.joblib"
    )
    assert command[command.index("--scored-cache") + 1] == str(expected_cache)
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--calibrator-strategy") + 1] == "newton_residual"
    assert command[command.index("--v25-probability-artifact") + 1] == str(
        v25_artifact
    )
    assert command[
        command.index("--closing-odds-min-training-days") + 1
    ] == "4"
    assert command[
        command.index("--closing-odds-min-training-races") + 1
    ] == "250"
    assert command[command.index("--through-date") + 1] == "2026-07-24"
    assert "--trend-point-odds-safety-sweep" in command
    assert command[
        command.index("--trend-point-required-ticket-count") + 1
    ] == "2"
    assert "--prequential-conditional-order" in command
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_market_residual_walk_forward_builds_fixed_model_blend_command(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    candidate = root / "data/models/evaluation_queue/candidate.joblib"
    baseline = root / "data/models/baseline.joblib"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")

    command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": "data/models/evaluation_queue/candidate.joblib",
                "baseline_model_input": "data/models/baseline.joblib",
                "candidate_weight": 0.35,
                "from_date": "2026-07-18",
            },
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--baseline-model") + 1] == str(baseline)
    assert command[command.index("--candidate-weight") + 1] == "0.35"


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"baseline_model_input": "data/models/baseline.joblib"}, "provided together"),
        ({"candidate_weight": 0.5}, "provided together"),
        (
            {"baseline_model_input": None, "candidate_weight": None},
            "non-empty string",
        ),
        (
            {
                "baseline_model_input": "data/models/baseline.joblib",
                "candidate_weight": "0.5",
            },
            "must be a number",
        ),
        (
            {
                "baseline_model_input": "data/models/baseline.joblib",
                "candidate_weight": 1.01,
            },
            "finite and in",
        ),
        (
            {
                "baseline_model_input": "../../baseline.joblib",
                "candidate_weight": 0.5,
            },
            "inside data/models",
        ),
    ],
)
def test_market_residual_walk_forward_rejects_invalid_fixed_model_blend(
    tmp_path: Path,
    parameters: dict[str, object],
    message: str,
) -> None:
    root = tmp_path / "boat"
    candidate = root / "data/models/candidate.joblib"
    baseline = root / "data/models/baseline.joblib"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")
    with pytest.raises(ValueError, match=message):
        build_command(
            _job(
                "market_residual_walk_forward",
                {
                    "model_input": "data/models/candidate.joblib",
                    "from_date": "2026-07-18",
                    **parameters,
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_market_residual_walk_forward_accepts_v18_schedule_quota(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    model = root / "data/models/evaluation_queue/job-00002606.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"artifact")
    python = root / ".venv/bin/python"
    expected_cache = (
        root / "data/models/evaluation_cache/market_scored"
        / "job-00002606_2026-07-18_2026-07-24.races.joblib"
    )

    command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-29",
                "calibrator_strategy": (
                    "odds_path_observed_closing_return_schedule_quota_v18"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command[command.index("--calibrator-strategy") + 1] == (
        "odds_path_observed_closing_return_schedule_quota_v18"
    )

    v19_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-29",
                "calibrator_strategy": (
                    "odds_path_observed_closing_return_schedule_quota_"
                    "raw_nonregression_v19"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert v19_command[v19_command.index("--calibrator-strategy") + 1] == (
        "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19"
    )

    v20_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-29",
                "calibrator_strategy": (
                    "odds_path_observed_closing_return_schedule_quota_"
                    "dual_head_v20"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert v20_command[v20_command.index("--calibrator-strategy") + 1] == (
        "odds_path_observed_closing_return_schedule_quota_dual_head_v20"
    )

    v21_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-29",
                "calibrator_strategy": (
                    "odds_path_observed_closing_return_schedule_quota_"
                    "triple_head_v21"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert v21_command[v21_command.index("--calibrator-strategy") + 1] == (
        "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
    )

    odds_path_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": "data/models/evaluation_queue/job-00002606.joblib",
                "from_date": "2026-07-18",
                "through_date": "2026-07-24",
                "calibrator_strategy": "odds_path_return",
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert odds_path_command[
        odds_path_command.index("--scored-cache") + 1
    ] == str(expected_cache)
    assert odds_path_command[
        odds_path_command.index("--calibrator-strategy") + 1
    ] == "odds_path_return"

    v7_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-24",
                "calibrator_strategy": (
                    "odds_path_crossfit_conservative_ev"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert v7_command[
        v7_command.index("--calibrator-strategy") + 1
    ] == "odds_path_crossfit_conservative_ev"
    v8_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-24",
                "calibrator_strategy": (
                    "odds_path_market_offset_crossfit_conservative_ev"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert v8_command[
        v8_command.index("--calibrator-strategy") + 1
    ] == "odds_path_market_offset_crossfit_conservative_ev"
    v10_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-24",
                "calibrator_strategy": (
                    "odds_path_market_offset_selection_conformal_discrete_ev_v10"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert v10_command[
        v10_command.index("--calibrator-strategy") + 1
    ] == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    v11_command, v11_output = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": (
                    "data/models/evaluation_queue/job-00002606.joblib"
                ),
                "from_date": "2026-07-18",
                "through_date": "2026-07-24",
                "daily_budget_yen": 10000,
                "calibrator_strategy": (
                    "odds_path_role_integrated_multihorizon_v11"
                ),
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert v11_command[:3] == [
        str(python),
        "-m",
        "boatrace_ai.listwise.market_calibration",
    ]
    assert v11_command[
        v11_command.index("--calibrator-strategy") + 1
    ] == "odds_path_role_integrated_multihorizon_v11"
    assert v11_command[v11_command.index("--from-date") + 1] == "2026-07-18"
    assert v11_command[v11_command.index("--through-date") + 1] == "2026-07-24"
    assert v11_command[v11_command.index("--daily-budget-yen") + 1] == "10000"
    assert v11_command[v11_command.index("--output") + 1] == str(v11_output)
    orthogonal_command, _ = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": "data/models/evaluation_queue/job-00002606.joblib",
                "from_date": "2026-07-18",
                "through_date": "2026-07-24",
                "calibrator_strategy": "orthogonal_residual",
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert orthogonal_command[
        orthogonal_command.index("--calibrator-strategy") + 1
    ] == "orthogonal_residual"

    with pytest.raises(ValueError, match="inside data/models"):
        build_command(
            _job(
                "market_residual_walk_forward",
                {"model_input": "../../outside.joblib", "from_date": "2026-07-18"},
            ),
            app_root=root,
            python=python,
            db="postgresql://test",
        )
    with pytest.raises(ValueError, match="unsupported"):
        build_command(
            _job(
                "market_residual_walk_forward",
                {
                    "model_input": "data/models/candidate.joblib",
                    "from_date": "2026-07-18",
                    "command": "arbitrary",
                },
            ),
            app_root=root,
            python=python,
            db="postgresql://test",
        )


def test_result_summary_preserves_v7_prospective_gate_metrics() -> None:
    summary = summarize_result({
        "model": "odds_path_crossfit_conservative_ev_v7",
        "prospective_crossfit_conservative_ev_v7_walk_forward": {
            "status": "evaluating",
            "registered_after": "2026-07-29",
            "evaluation_days": 31,
            "evaluated_races": 4_100,
            "tickets": 320,
            "hit_tickets": 25,
            "roi": 1.08,
            "roi_without_largest_hit": 1.02,
            "daily_cluster_bootstrap_roi_lower_95": 1.01,
            "effective_hit_count": 22.0,
            "largest_hit_return_share": 0.12,
            "calibrated_trifecta_log_loss": 3.70,
            "closing_q20_pinball_loss": 0.04,
            "closing_q20_lower_coverage": 0.81,
            "promotion_eligible": True,
        },
    })

    assert summary["prospective_v7_registered_after"] == "2026-07-29"
    assert summary["prospective_v7_roi"] == 1.08
    assert summary["prospective_v7_roi_without_largest_hit"] == 1.02
    assert (
        summary[
            "prospective_v7_daily_cluster_bootstrap_roi_lower_95"
        ]
        == 1.01
    )
    assert summary["prospective_v7_closing_q20_lower_coverage"] == 0.81
    assert summary["prospective_v7_promotion_eligible"] is True


def test_load_result_preserves_purchase_decision_diagnostics(tmp_path) -> None:
    path = tmp_path / "v8-result.json"
    diagnostics = {
        "threshold_pass_candidates": 0,
        "candidates_after_race_cap": 0,
        "purchases_after_allocation": 0,
        "safe_ev_max": 1.041,
        "safe_ev_p95": 0.982,
        "safe_ev_p99": 1.012,
        "safe_ev_at_least": {"1.00": 8, "1.05": 0},
    }
    path.write_text(
        json.dumps({
            "model": "odds_path_market_offset_crossfit_conservative_ev_v8",
            "purchase_decision_diagnostics": diagnostics,
        }),
        encoding="utf-8",
    )

    payload, summary = evaluation_queue._load_result(path)

    assert payload["purchase_decision_diagnostics"] == diagnostics
    assert summary["purchase_decision_diagnostics"] == diagnostics
    assert summary["threshold_pass_candidates"] == 0
    assert summary["candidates_after_race_cap"] == 0
    assert summary["purchases_after_allocation"] == 0
    assert summary["safe_ev_max"] == 1.041
    assert summary["safe_ev_p95"] == 0.982
    assert summary["safe_ev_p99"] == 1.012


def test_result_summary_exposes_high_ev_holdout_calibration() -> None:
    summary = summarize_result({
        "holdout_top5_flat_diagnostic": {
            "evaluated_races": 100, "tickets": 500, "hit_races": 30,
            "hit_rate": 0.3, "stake_yen": 50000,
            "return_yen": 45000, "profit_yen": -5000, "roi": 0.9,
            "average_hit_payout_yen": 1500,
            "breakeven_average_hit_payout_yen": 1666.6667,
        },
        "holdout_candidate_ev_calibration": [
            {
                "lower_inclusive": 2.0, "upper_exclusive": 2.5,
                "tickets": 10, "hits": 1, "flat_stake_yen": 1000,
                "flat_return_yen": 800, "realized_roi": 0.8,
                "mean_estimated_ev": 2.2,
            },
            {
                "lower_inclusive": 2.5, "upper_exclusive": None,
                "tickets": 5, "hits": 1, "flat_stake_yen": 500,
                "flat_return_yen": 1000, "realized_roi": 2.0,
                "mean_estimated_ev": 2.8,
            },
        ],
    })

    assert summary["top5_flat_tickets"] == 500
    assert summary["top5_flat_hit_rate"] == 0.3
    assert summary["top5_flat_roi"] == 0.9
    assert summary["high_ev_tickets"] == 15
    assert summary["high_ev_realized_roi"] == 1.2
    assert len(summary["holdout_candidate_ev_calibration"]) == 2


def test_result_summary_exposes_registered_ev_band_separately() -> None:
    summary = summarize_result({
        "roi": 0.33,
        "registered_ev_band_walk_forward": {
            "status": "evaluating",
            "registered_after": "2026-07-25",
            "evaluation_days": 1,
            "evaluated_races": 132,
            "tickets": 4,
            "hit_tickets": 1,
            "stake_yen": 400,
            "return_yen": 500,
            "roi": 1.25,
            "profit_yen": 100,
            "winning_days": 1,
            "profitable_day_fraction": 1.0,
            "largest_hit_return_share": 0.8,
            "effective_hit_count": 1.5,
            "roi_without_largest_hit": 0.25,
            "daily_cluster_bootstrap_roi_lower_95": 1.25,
            "probability_roi_above_one": 1.0,
        },
        "prospective_normalized_ev_walk_forward": {
            "status": "waiting_for_first_unseen_day",
            "registered_after": "2026-07-27",
            "evaluation_days": 0,
            "evaluated_races": 0,
            "tickets": 0,
            "hit_tickets": 0,
            "roi": 0.0,
            "profit_yen": 0,
        },
        "prospective_top5_narrow_ev_walk_forward": {
            "status": "waiting_for_first_unseen_day",
            "registered_after": "2026-07-28",
            "evaluation_days": 0,
            "evaluated_races": 0,
            "tickets": 0,
            "hit_tickets": 0,
            "roi": 0.0,
            "profit_yen": 0,
        },
        "top5_narrow_retrospective_diagnostic": {
            "status": "diagnostic_only_not_promotion_evidence",
            "evaluation_days": 8,
            "evaluated_races": 1179,
            "tickets": 609,
            "hit_tickets": 71,
            "roi": 1.312,
            "profit_yen": 19010,
            "roi_without_largest_hit": 1.28,
            "effective_hit_count": 50.0,
            "promotion_evidence": False,
        },
        "prospective_observed_closing_return_v4_walk_forward": {
            "status": "waiting_for_first_unseen_day",
            "registered_after": "2026-07-29",
            "evaluation_days": 0,
            "evaluated_races": 0,
            "tickets": 0,
            "hit_tickets": 0,
            "roi": 0.0,
            "profit_yen": 0,
            "roi_without_largest_hit": None,
        },
    })

    assert summary["roi"] == 0.33
    assert summary["registered_ev_band_roi"] == 1.25
    assert summary["registered_ev_band_evaluation_days"] == 1
    assert summary["registered_ev_band_stake_yen"] == 400
    assert summary["registered_ev_band_return_yen"] == 500
    assert summary["registered_ev_band_winning_days"] == 1
    assert summary["registered_ev_band_profitable_day_fraction"] == 1.0
    assert summary["registered_ev_band_largest_hit_return_share"] == 0.8
    assert summary["registered_ev_band_effective_hit_count"] == 1.5
    assert summary["registered_ev_band_roi_without_largest_hit"] == 0.25
    assert summary["registered_ev_band_daily_cluster_bootstrap_roi_lower_95"] == 1.25
    assert summary["registered_ev_band_probability_roi_above_one"] == 1.0
    assert summary["prospective_normalized_ev_status"] == (
        "waiting_for_first_unseen_day"
    )
    assert summary["prospective_normalized_ev_registered_after"] == "2026-07-27"
    assert summary["prospective_top5_narrow_ev_status"] == (
        "waiting_for_first_unseen_day"
    )
    assert summary["prospective_top5_narrow_ev_registered_after"] == "2026-07-28"
    assert summary["top5_narrow_retrospective_evaluation_days"] == 8
    assert summary["top5_narrow_retrospective_roi"] == 1.312
    assert summary["top5_narrow_retrospective_promotion_evidence"] is False
    assert summary["prospective_observed_closing_v4_status"] == (
        "waiting_for_first_unseen_day"
    )
    assert summary["prospective_observed_closing_v4_registered_after"] == (
        "2026-07-29"
    )


def test_result_summary_preserves_v9_prospective_and_allocation_diagnostics() -> None:
    diagnostics = {
        "threshold_pass_candidates": 40,
        "candidates_before_allocation": 20,
        "allocation_candidate_tickets": 12,
        "purchases_after_allocation": 8,
        "zero_purchase_days": 2,
        "zero_reason_counts": {"no_positive_discrete_log_growth": 2},
    }
    summary = summarize_result({
        "model": "odds_path_market_offset_discrete_log_ev_v9",
        "purchase_decision_diagnostics": diagnostics,
        "prospective_market_offset_discrete_log_ev_v9_walk_forward": {
            "status": "evaluating",
            "registered_after": "2026-07-29",
            "evaluation_days": 31,
            "evaluated_races": 4_100,
            "tickets": 320,
            "roi": 1.08,
            "roi_without_largest_hit": 1.02,
            "promotion_eligible": True,
        },
    })

    assert summary["model"] == "odds_path_market_offset_discrete_log_ev_v9"
    assert summary["purchase_decision_diagnostics"] == diagnostics
    assert summary["candidates_before_allocation"] == 20
    assert summary["allocation_candidate_tickets"] == 12
    assert summary["zero_reason_counts"] == {
        "no_positive_discrete_log_growth": 2
    }
    assert summary["prospective_v9_registered_after"] == "2026-07-29"
    assert summary["prospective_v9_roi"] == 1.08
    assert summary["prospective_v9_promotion_eligible"] is True


def test_result_summary_preserves_v10_selection_conformal_diagnostics() -> None:
    diagnostics = {
        "raw_selected_candidates": 9,
        "guarded_threshold_candidates": 3,
        "purchases_after_allocation": 2,
        "zero_purchase_days": 1,
        "zero_reason_counts": {"no_candidate_after_selection_conformal": 1},
    }
    conformal = {
        "selection_evaluation_candidates": 9,
        "selection_raw_covered_candidates": 2,
        "selection_guarded_covered_candidates": 8,
        "selection_raw_closing_coverage": 2 / 9,
        "selection_guarded_closing_coverage": 8 / 9,
        "selection_closing_ratio_mean": 0.72,
        "selection_closing_ratio_p10": 0.51,
        "selection_closing_ratio_median": 0.68,
        "haircut_latest": 0.55,
        "haircut_min": 0.48,
        "haircut_max": 0.60,
        "training_days_latest": 6,
        "training_candidates_latest": 41,
        "trained_through_date_latest": "2026-07-29",
    }
    summary = summarize_result({
        "model": "odds_path_market_offset_selection_conformal_discrete_ev_v10",
        "purchase_decision_diagnostics": diagnostics,
        "selection_conformal": conformal,
        "prospective_market_offset_selection_conformal_discrete_ev_v10_walk_forward": {
            "status": "evaluating",
            "registered_after": "2026-07-29",
            "evaluation_days": 31,
            "evaluated_races": 4_100,
            "tickets": 301,
            "roi": 1.06,
            "roi_without_largest_hit": 1.01,
            "promotion_eligible": True,
            "selection_conformal": conformal,
        },
    })

    assert summary["purchase_decision_diagnostics"] == diagnostics
    assert summary["raw_selected_candidates"] == 9
    assert summary["guarded_threshold_candidates"] == 3
    assert summary["zero_reason_counts"] == {
        "no_candidate_after_selection_conformal": 1
    }
    assert summary["selection_conformal"] == conformal
    assert summary["selection_raw_closing_coverage"] == 2 / 9
    assert summary["selection_guarded_closing_coverage"] == 8 / 9
    assert summary["haircut_latest"] == 0.55
    assert summary["training_days_latest"] == 6
    assert summary["training_candidates_latest"] == 41
    assert summary["prospective_v10_registered_after"] == "2026-07-29"
    assert summary["prospective_v10_roi"] == 1.06
    assert summary["prospective_v10_promotion_eligible"] is True
    assert summary["prospective_v10_selection_conformal"] == conformal


def test_v12_role_stack_build_command_carries_explicit_fallback_contract(
    tmp_path: Path,
) -> None:
    model_input = tmp_path / "data/models/source.joblib"
    model_input.parent.mkdir(parents=True)
    model_input.write_bytes(b"artifact")
    command, _output = build_command(
        _job(
            "market_residual_walk_forward",
            {
                "model_input": "data/models/source.joblib",
                "from_date": "2026-07-18",
                "through_date": "2026-07-29",
                "daily_budget_yen": 10_000,
                "calibrator_strategy": (
                    "odds_path_role_integrated_t300_nonlinear_v12"
                ),
                "v12_closing_fallback_policy": "no_bet",
            },
        ),
        app_root=tmp_path,
        python=Path("/venv/bin/python"),
        db="postgresql://test",
    )

    assert command[
        command.index("--calibrator-strategy") + 1
    ] == "odds_path_role_integrated_t300_nonlinear_v12"
    assert command[
        command.index("--v12-closing-fallback-policy") + 1
    ] == "no_bet"


def test_v12_result_summary_preserves_closing_model_identity() -> None:
    identity = {
        "requested_model": "closing_odds_t300_nonlinear_v12",
        "fallback_policy": "v11",
        "selected_model_latest": "closing_odds_multihorizon_v11",
        "selected_model_fold_counts": {
            "closing_odds_multihorizon_v11": 2,
            "closing_odds_t300_nonlinear_v12": 1,
        },
        "evaluation_folds": 3,
        "v12_ready_folds": 3,
        "v12_adopted_folds": 1,
        "v11_fallback_folds": 2,
        "no_bet_folds": 0,
    }
    summary = summarize_result({
        "model": "odds_path_role_integrated_t300_nonlinear_v12",
        "roi": 1.05,
        "profit_yen": 500,
        "calibrated_trifecta_log_loss": 3.7,
        "closing_q20_lower_coverage": 0.82,
        "closing_model_identity": identity,
        "prospective_role_integrated_v12_walk_forward": {
            "evaluation_days": 3,
            "evaluated_races": 360,
            "tickets": 12,
            "profit_yen": 500,
            "roi": 1.05,
            "promotion_eligible": False,
            "closing_model_identity": identity,
        },
    })

    assert summary["roi"] == 1.05
    assert summary["calibrated_trifecta_log_loss"] == 3.7
    assert summary["closing_q20_lower_coverage"] == 0.82
    assert summary["closing_model_identity"] == identity
    assert summary["closing_model_requested"] == (
        "closing_odds_t300_nonlinear_v12"
    )
    assert summary["closing_model_selected"] == (
        "closing_odds_multihorizon_v11"
    )
    assert summary["closing_fallback_policy"] == "v11"
    assert summary["closing_v12_adopted_folds"] == 1
    assert summary["prospective_v12_roi"] == 1.05
    assert summary["prospective_v12_closing_model_identity"] == identity


def test_result_summary_exposes_v33_forecast_only_metrics() -> None:
    summary = summarize_result({
        "v33_v25_top1_narrow_retrospective_diagnostic": {
            "evaluation_days": 7,
            "tickets": 119,
            "roi": 0.9958,
            "profit_yen": -50,
            "promotion_evidence": False,
        },
        "v33_v25_top1_narrow_forecast_only_diagnostic": {
            "evaluation_days": 2,
            "tickets": 23,
            "roi": 0.8217,
            "profit_yen": -410,
            "promotion_evidence": False,
        },
    })

    assert summary["v33_v25_top1_narrow_retrospective_roi"] == 0.9958
    assert summary["v33_v25_top1_narrow_retrospective_tickets"] == 119
    assert summary["v33_v25_top1_narrow_forecast_only_roi"] == 0.8217
    assert summary["v33_v25_top1_narrow_forecast_only_evaluation_days"] == 2
    assert summary["v33_v25_top1_narrow_forecast_only_promotion_evidence"] is False


def test_result_summary_derives_v33_forecast_only_metrics_from_folds() -> None:
    summary = summarize_result({
        "folds": [
            {
                "closing_odds_policy_input": "observed_t5_fallback",
                "v33_v25_top1_narrow_retrospective_bankroll": {
                    "evaluated_races": 100,
                    "tickets": 10,
                    "hit_tickets": 2,
                    "stake_yen": 1000,
                    "return_yen": 1500,
                    "largest_hit_return_yen": 900,
                    "hit_return_square_sum_yen2": 1170000,
                },
            },
            {
                "closing_odds_policy_input": "oof_forecast_final_from_real_t5",
                "v33_v25_top1_narrow_retrospective_bankroll": {
                    "evaluated_races": 120,
                    "tickets": 20,
                    "hit_tickets": 3,
                    "stake_yen": 2000,
                    "return_yen": 1800,
                    "largest_hit_return_yen": 800,
                    "hit_return_square_sum_yen2": 1080000,
                },
            },
        ],
    })

    prefix = "v33_v25_top1_narrow_forecast_only"
    assert summary[f"{prefix}_evaluation_days"] == 1
    assert summary[f"{prefix}_evaluated_races"] == 120
    assert summary[f"{prefix}_tickets"] == 20
    assert summary[f"{prefix}_roi"] == 0.9
    assert summary[f"{prefix}_profit_yen"] == -200
    assert summary[f"{prefix}_roi_without_largest_hit"] == 0.5
    assert summary[f"{prefix}_promotion_evidence"] is False
