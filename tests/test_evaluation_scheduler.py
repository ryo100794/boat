from __future__ import annotations

import errno
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import (
    PRODUCTION_TREND_POINT_MODEL_INPUT,
    PRODUCTION_TREND_POINT_MODEL_KEY,
    PRODUCTION_TREND_POINT_REGISTERED_AFTER,
    PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID,
    PRODUCTION_TREND_POINT_TWO_TICKET_MODEL_KEY,
    ResourceSnapshot,
    SCHEMA,
    build_command,
    job_workspace_reservation_mb,
    resources_allow,
    seed_periodic_jobs,
    workspace_quota_allows,
)


class _CountRow(dict):
    pass


class _IdleQueue:
    def execute(self, statement, params=()):
        assert "COUNT(*)" in statement or "SELECT job_id" in statement
        return self

    def fetchone(self):
        return _CountRow(count=0)

    def fetchall(self):
        return []


def test_resource_gate_requires_memory_disk_and_idle_cpu() -> None:
    resources = ResourceSnapshot(
        available_memory_mb=32_768,
        available_disk_mb=8_192,
        idle_cpu_percent=42.0,
        cpu_count=16,
        load_1m=3.0,
    )

    assert resources_allow(
        resources,
        min_free_memory_mb=16_384,
        min_free_disk_mb=4_096,
        min_idle_cpu_percent=15.0,
    )
    assert not resources_allow(
        resources,
        min_free_memory_mb=65_536,
        min_free_disk_mb=4_096,
        min_idle_cpu_percent=15.0,
    )
    assert not resources_allow(
        resources,
        min_free_memory_mb=16_384,
        min_free_disk_mb=16_384,
        min_idle_cpu_percent=15.0,
    )
    assert not resources_allow(
        resources,
        min_free_memory_mb=16_384,
        min_free_disk_mb=4_096,
        min_idle_cpu_percent=50.0,
    )


def test_workspace_quota_probe_reserves_target_and_removes_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_fallocate(descriptor: int, offset: int, length: int) -> None:
        assert os.fstat(descriptor).st_size == 0
        calls.append((offset, length))

    monkeypatch.setattr(os, "posix_fallocate", fake_fallocate)

    assert workspace_quota_allows(tmp_path, required_mb=12) is True
    assert calls == [(0, 12 * 1024**2)]
    assert not list((tmp_path / "data" / "archive-staging").iterdir())


def test_workspace_quota_probe_rejects_quota_and_removes_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_close = os.close

    def fail_fallocate(_descriptor: int, _offset: int, _length: int) -> None:
        raise OSError(errno.EDQUOT, "quota")

    def close_then_report_quota(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError(errno.EDQUOT, "quota on close")

    monkeypatch.setattr(os, "posix_fallocate", fail_fallocate)
    monkeypatch.setattr(os, "close", close_then_report_quota)

    assert workspace_quota_allows(tmp_path, required_mb=12) is False
    monkeypatch.setattr(os, "close", real_close)
    assert not list((tmp_path / "data" / "archive-staging").iterdir())


def test_mlp_workspace_reservation_drops_after_complete_cache(
    tmp_path: Path,
) -> None:
    job = {
        "task_type": "calibrated_mlp_recency_search",
        "min_free_disk_mb": 12288,
        "parameters": {
            "drop_feature_groups": (
                "raw_equipment_identifiers,speculative_research,"
                "live_official_context"
            )
        },
    }

    assert job_workspace_reservation_mb(job, tmp_path) == 1024

    prefix = (
        tmp_path
        / "data/models/"
        "calibrated_shadow_features_16384__drop_"
        "raw_equipment_identifiers_speculative_research_live_official_context"
    )
    prefix.parent.mkdir(parents=True)
    for suffix in ("matrix.npz", "ranks.npy", "manifest.json"):
        Path(f"{prefix}.{suffix}").write_bytes(b"complete")

    assert job_workspace_reservation_mb(job, tmp_path) == 256


def test_non_mlp_workspace_reservation_uses_profile_requirement(
    tmp_path: Path,
) -> None:
    assert job_workspace_reservation_mb(
        {
            "task_type": "bankroll_policy_nested_annual",
            "min_free_disk_mb": 4096,
        },
        tmp_path,
    ) == 256


def test_periodic_scheduler_enqueues_backup_aggregation_and_hygiene(monkeypatch) -> None:
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr("boatrace_ai.evaluation_queue.enqueue_job", fake_enqueue)

    inserted = seed_periodic_jobs(
        _IdleQueue(), now=datetime(2026, 7, 23, 12, 34, tzinfo=timezone.utc)
    )

    assert inserted == [1, 2, 3, 4, 5]
    assert [row["task_type"] for row in calls] == [
        "gdrive_raw_archive",
        "evaluation_aggregate",
        "series_feature_cache",
        "repository_sync",
        "repository_hygiene",
    ]
    assert all("schedule_bucket" in row["parameters"] for row in calls)
    hygiene = calls[-1]
    assert hygiene["model_key"] == "repository"
    assert hygiene["parameters"]["timeout_seconds"] == 300
    assert hygiene["priority"] == 20
    sync = next(row for row in calls if row["task_type"] == "repository_sync")
    assert sync["parameters"]["timeout_seconds"] == 300
    assert sync["priority"] == 25


def test_periodic_scheduler_registers_fixed_trend_candidate_after_first_unseen_day(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr("boatrace_ai.evaluation_queue.enqueue_job", fake_enqueue)
    root = tmp_path / "boat"
    result = (
        root / "data" / "models" / "evaluation_queue"
        / f"job-{PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID:08d}.json"
    )
    result.parent.mkdir(parents=True)
    expected_sha256 = "a" * 64
    result.write_text(
        json.dumps({
            "source_model": PRODUCTION_TREND_POINT_MODEL_INPUT,
            "source_model_sha256": expected_sha256,
        }),
        encoding="utf-8",
    )

    inserted = seed_periodic_jobs(
        _IdleQueue(),
        now=datetime(2026, 8, 5, 3, 1, tzinfo=timezone.utc),
        app_root=root,
    )

    assert inserted == [1, 2, 3, 4, 5, 6, 7]
    candidate = calls[-2]
    assert candidate["task_type"] == "market_residual_walk_forward"
    assert candidate["model_key"] == PRODUCTION_TREND_POINT_MODEL_KEY
    assert candidate["parameters"]["model_input"] == PRODUCTION_TREND_POINT_MODEL_INPUT
    assert candidate["parameters"]["through_date"] == "2026-08-04"
    assert candidate["parameters"]["trend_point_registered_after"] == (
        PRODUCTION_TREND_POINT_REGISTERED_AFTER
    )
    assert candidate["parameters"]["expected_model_sha256"] == expected_sha256
    registration = candidate["parameters"]["prospective_candidate"]
    assert registration == {
        "source_model_job_id": 12_012,
        "source_evaluation_job_id": 12_051,
        "expected_model_sha256": expected_sha256,
        "policy": "trend_point_market_offset_discrete_multinomial_kelly",
        "registered_after": PRODUCTION_TREND_POINT_REGISTERED_AFTER,
        "evidence_dates": "strictly_after_registered_after",
        "selection_data_is_diagnostic_only": True,
        "real_betting_enabled": False,
    }
    assert candidate["priority"] == 44
    two_ticket = calls[-1]
    assert two_ticket["task_type"] == "market_residual_walk_forward"
    assert (
        two_ticket["model_key"]
        == PRODUCTION_TREND_POINT_TWO_TICKET_MODEL_KEY
    )
    assert two_ticket["parameters"]["trend_point_required_ticket_count"] == 2
    assert two_ticket["parameters"]["prospective_candidate"] == {
        "source_model_job_id": 12_012,
        "source_evaluation_job_id": 12_051,
        "expected_model_sha256": expected_sha256,
        "policy": (
            "trend_point_market_offset_discrete_multinomial_kelly_"
            "exact_two_ticket"
        ),
        "required_ticket_count": 2,
        "registered_after": PRODUCTION_TREND_POINT_REGISTERED_AFTER,
        "evidence_dates": "strictly_after_registered_after",
        "selection_data_is_diagnostic_only": True,
        "real_betting_enabled": False,
    }
    assert two_ticket["priority"] == 43


def test_fixed_trend_candidate_command_preserves_registration_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    model = root / PRODUCTION_TREND_POINT_MODEL_INPUT
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixed-model")
    command, _output = build_command(
        {
            "job_id": 16,
            "task_type": "market_residual_walk_forward",
            "parameters": {
                "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
                "from_date": "2026-07-20",
                "through_date": "2026-08-04",
                "trend_point_registered_after": (
                    PRODUCTION_TREND_POINT_REGISTERED_AFTER
                ),
                "trend_point_required_ticket_count": 2,
                "expected_model_sha256": hashlib.sha256(
                    b"fixed-model"
                ).hexdigest(),
                "prospective_candidate": {
                    "source_job_id": 12_012,
                    "registered_after": PRODUCTION_TREND_POINT_REGISTERED_AFTER,
                },
            },
        },
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    boundary_index = command.index("--trend-point-registered-after")
    assert command[boundary_index + 1] == PRODUCTION_TREND_POINT_REGISTERED_AFTER
    ticket_count_index = command.index("--trend-point-required-ticket-count")
    assert command[ticket_count_index + 1] == "2"

    with pytest.raises(ValueError, match="prospective registration"):
        build_command(
            {
                "job_id": 17,
                "task_type": "market_residual_walk_forward",
                "parameters": {
                    "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
                    "from_date": "2026-07-20",
                    "expected_model_sha256": "0" * 64,
                },
            },
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_maintenance_commands_are_allowlisted(tmp_path) -> None:
    root = tmp_path / "boat"
    aggregate, aggregate_output = build_command(
        {
            "job_id": 12,
            "task_type": "evaluation_aggregate",
            "parameters": {},
        },
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    backup, backup_output = build_command(
        {
            "job_id": 13,
            "task_type": "gdrive_raw_archive",
            "parameters": {},
        },
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    hygiene, hygiene_output = build_command(
        {
            "job_id": 14,
            "task_type": "repository_hygiene",
            "parameters": {},
        },
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert "aggregate-evaluations" in aggregate
    assert "backup-raw" in backup
    assert hygiene == [
        str(root / ".venv/bin/python"),
        "-m",
        "boatrace_ai.maintenance_tasks",
        "repository-hygiene",
        "--app-root",
        str(root),
        "--output",
        str(root / "data/models/evaluation_queue/job-00000014.json"),
    ]
    sync, sync_output = build_command(
        {
            "job_id": 15,
            "task_type": "repository_sync",
            "parameters": {},
        },
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert aggregate_output.name == "job-00000012.json"
    assert backup_output.name == "job-00000013.json"
    assert hygiene_output.name == "job-00000014.json"
    assert sync == [
        str(root / ".venv/bin/python"),
        "-m",
        "boatrace_ai.maintenance_tasks",
        "repository-sync",
        "--db",
        "postgresql://test",
        "--app-root",
        str(root),
        "--output",
        str(root / "data/models/evaluation_queue/job-00000015.json"),
    ]
    assert sync_output.name == "job-00000015.json"


def test_schema_tracks_attempts_resources_and_work_tickets() -> None:
    assert "model_evaluation_job_runs" in SCHEMA
    assert "last_resource_snapshot" in SCHEMA
    assert "min_free_disk_mb" in SCHEMA
    assert "CREATE TABLE IF NOT EXISTS work_tickets" in SCHEMA
    assert "CREATE TABLE IF NOT EXISTS work_ticket_events" in SCHEMA
    for column in (
        "repository_full_name",
        "github_issue_number",
        "github_issue_url",
        "github_issue_updated_at",
        "last_synced_at",
    ):
        assert f"ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS {column}" in SCHEMA
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_work_tickets_github_issue" in SCHEMA
    assert "WHERE github_issue_number IS NOT NULL" in SCHEMA
    assert "SET min_free_memory_mb = 8192" in SCHEMA
    assert "task_type = 'market_residual_walk_forward'" in SCHEMA


def test_supervisor_separates_periodic_scheduler_from_workers() -> None:
    runner = Path(
        "scripts/deployment/supervisor-boatrace-evaluation-runner.ini"
    ).read_text(encoding="utf-8")
    scheduler = Path(
        "scripts/deployment/supervisor-boatrace-evaluation-scheduler.ini"
    ).read_text(encoding="utf-8")

    assert "--schedule-periodic" not in runner
    assert "boatrace_ai.evaluation_queue schedule" in scheduler
    assert "--seed-defaults" in scheduler
