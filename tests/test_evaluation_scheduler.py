from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from boatrace_ai.evaluation_queue import (
    ResourceSnapshot,
    SCHEMA,
    build_command,
    resources_allow,
    seed_periodic_jobs,
)


class _CountRow(dict):
    pass


class _IdleQueue:
    def execute(self, statement, params=()):
        assert "COUNT(*)" in statement
        return self

    def fetchone(self):
        return _CountRow(count=0)


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


def test_periodic_scheduler_enqueues_backup_aggregation_and_hygiene(monkeypatch) -> None:
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr("boatrace_ai.evaluation_queue.enqueue_job", fake_enqueue)

    inserted = seed_periodic_jobs(
        _IdleQueue(), now=datetime(2026, 7, 23, 12, 34, tzinfo=timezone.utc)
    )

    assert inserted == [1, 2, 3, 4]
    assert [row["task_type"] for row in calls] == [
        "gdrive_raw_archive",
        "evaluation_aggregate",
        "series_feature_cache",
        "repository_hygiene",
    ]
    assert all("schedule_bucket" in row["parameters"] for row in calls)
    hygiene = calls[-1]
    assert hygiene["model_key"] == "repository"
    assert hygiene["parameters"]["timeout_seconds"] == 300
    assert hygiene["priority"] == 20


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
    assert aggregate_output.name == "job-00000012.json"
    assert backup_output.name == "job-00000013.json"
    assert hygiene_output.name == "job-00000014.json"


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


def test_supervisor_enables_periodic_scheduler() -> None:
    config = Path(
        "scripts/deployment/supervisor-boatrace-evaluation-runner.ini"
    ).read_text(encoding="utf-8")

    assert "--schedule-periodic" in config
