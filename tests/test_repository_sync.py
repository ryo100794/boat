from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from boatrace_ai import maintenance_tasks


MIDDAY_UTC = datetime(2026, 8, 3, 11, 0, tzinfo=timezone.utc)
OVERNIGHT_UTC = datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc)


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _commit(root: Path, content: str, message: str) -> None:
    (root / "tracked.txt").write_text(content, encoding="utf-8")
    _git("add", "tracked.txt", cwd=root)
    _git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        message,
        cwd=root,
    )


def _repositories(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    _git("init", "--bare", str(remote))
    _git("init", str(seed))
    _commit(seed, "one\n", "initial")
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "-u", "origin", "master", cwd=seed)
    _git("clone", str(remote), str(checkout))
    return seed, checkout


def _connection(active: int, events: list[str] | None = None):
    class RowResult:
        def fetchone(self):
            return {"count": active}

    class Connection:
        def execute(self, statement, _parameters=None):
            if events is not None:
                events.append(" ".join(str(statement).split()))
            return RowResult()

    @contextmanager
    def connect(_db):
        yield Connection()

    return connect


def test_repository_sync_fetches_but_defers_during_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed, checkout = _repositories(tmp_path)
    before = _git("rev-parse", "HEAD", cwd=checkout)
    _commit(seed, "two\n", "upstream")
    _git("push", cwd=seed)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(1))

    output = tmp_path / "deferred.json"
    payload = maintenance_tasks.repository_sync(
        checkout, output, db="test", now=MIDDAY_UTC
    )

    assert payload["action"] == "deferred_active_evaluation"
    assert payload["active_evaluations"] == 1
    assert payload["behind"] == 1
    assert payload["before_head"] == payload["after_head"] == before
    assert payload["service_refresh"]["action"] == "deferred_repository_not_ready"
    assert json.loads(output.read_text(encoding="utf-8"))["action"] == payload["action"]


def test_repository_sync_drain_survives_until_idle_then_forces_refresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed, checkout = _repositories(tmp_path)
    _commit(seed, "two\n", "upstream")
    _git("push", cwd=seed)
    state = {"active": 1, "enabled": False, "target_head": ""}

    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def execute(self, statement, parameters=()):
            sql = " ".join(str(statement).split())
            if "COUNT(*) AS count" in sql:
                return Result({"count": state["active"]})
            if "SELECT enabled" in sql:
                return Result({
                    "enabled": state["enabled"],
                    "reason": "repository_update",
                    "target_head": state["target_head"],
                    "requested_at": None,
                    "updated_at": None,
                })
            if "INSERT INTO model_evaluation_control" in sql:
                state["enabled"] = True
                state["target_head"] = parameters[0]
                return Result()
            if "pg_advisory_xact_lock" in sql:
                return Result()
            raise AssertionError(sql)

    @contextmanager
    def connect(_db):
        yield Connection()

    monkeypatch.setattr(maintenance_tasks, "connection", connect)
    first = maintenance_tasks.repository_sync(
        checkout, tmp_path / "draining.json", db="test", now=MIDDAY_UTC
    )
    assert first["action"] == "deferred_active_evaluation"
    assert state["enabled"] is True
    assert first["evaluation_drain"]["enabled"] is True

    state["active"] = 0
    scheduled = []

    def schedule(
        app_root: Path, *, db: str, head: str,
        base_head: str | None, now: datetime
    ):
        scheduled.append((head, base_head))
        return {"action": "scheduled", "head": head}

    second = maintenance_tasks.repository_sync(
        checkout,
        tmp_path / "idle.json",
        db="test",
        now=MIDDAY_UTC,
        schedule_refresh=schedule,
    )
    assert second["action"] == "fast_forwarded"
    assert second["service_refresh"]["action"] == "scheduled"
    assert scheduled == [(second["after_head"], second["before_head"])]


def test_repository_sync_fast_forwards_clean_idle_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed, checkout = _repositories(tmp_path)
    _commit(seed, "two\n", "upstream")
    _git("push", cwd=seed)
    expected = _git("rev-parse", "HEAD", cwd=seed)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    payload = maintenance_tasks.repository_sync(
        checkout, tmp_path / "updated.json", db="test", now=MIDDAY_UTC
    )

    assert payload["action"] == "fast_forwarded"
    assert payload["behind"] == 1
    assert payload["after_head"] == expected
    assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "two\n"
    assert (
        payload["service_refresh"]["action"]
        == "deferred_outside_maintenance_window"
    )


def test_repository_sync_defers_dirty_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed, checkout = _repositories(tmp_path)
    _commit(seed, "two\n", "upstream")
    _git("push", cwd=seed)
    (checkout / "tracked.txt").write_text("local\n", encoding="utf-8")
    before = _git("rev-parse", "HEAD", cwd=checkout)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    payload = maintenance_tasks.repository_sync(
        checkout, tmp_path / "dirty.json", db="test", now=MIDDAY_UTC
    )

    assert payload["action"] == "deferred_dirty_worktree"
    assert payload["dirty_paths"]
    assert payload["after_head"] == before


def test_repository_sync_schedules_overnight_refresh_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seed, checkout = _repositories(tmp_path)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))
    scheduled: list[tuple[Path, str, str, str | None, datetime]] = []

    def schedule(
        app_root: Path, *, db: str, head: str,
        base_head: str | None, now: datetime
    ):
        scheduled.append((app_root, db, head, base_head, now))
        return {"action": "scheduled", "head": head, "pid": 123}

    payload = maintenance_tasks.repository_sync(
        checkout,
        tmp_path / "scheduled.json",
        db="test",
        now=OVERNIGHT_UTC,
        schedule_refresh=schedule,
    )

    head = _git("rev-parse", "HEAD", cwd=checkout)
    assert payload["action"] == "up_to_date"
    assert payload["service_refresh"] == {
        "action": "scheduled",
        "head": head,
        "pid": 123,
    }
    assert scheduled == [
        (checkout.resolve(), "test", head, None, OVERNIGHT_UTC)
    ]


def test_repository_sync_does_not_duplicate_recent_refresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seed, checkout = _repositories(tmp_path)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))
    head = _git("rev-parse", "HEAD", cwd=checkout)
    monkeypatch.setattr(
        maintenance_tasks,
        "_deployment_state",
        lambda _root: {
            "status": "scheduled",
            "head": head,
            "scheduled_at": (OVERNIGHT_UTC - timedelta(minutes=30)).isoformat(),
        },
    )

    payload = maintenance_tasks.repository_sync(
        checkout,
        tmp_path / "already-scheduled.json",
        db="test",
        now=OVERNIGHT_UTC,
        schedule_refresh=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("refresh must not be scheduled twice")
        ),
    )

    assert payload["service_refresh"]["action"] == "already_scheduled"


def test_repository_sync_recognizes_deployed_head(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seed, checkout = _repositories(tmp_path)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))
    head = _git("rev-parse", "HEAD", cwd=checkout)
    monkeypatch.setattr(
        maintenance_tasks,
        "_deployment_state",
        lambda _root: {"status": "completed", "head": head},
    )

    payload = maintenance_tasks.repository_sync(
        checkout,
        tmp_path / "deployed.json",
        db="test",
        now=OVERNIGHT_UTC,
    )

    assert payload["service_refresh"] == {"action": "up_to_date", "head": head}


def test_refresh_services_without_base_restarts_only_control_plane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    head = "a" * 40
    runner = "boatrace-evaluation-runner:boatrace-evaluation-runner_00"
    scheduler = "boatrace-evaluation-scheduler"
    calls: list[list[str]] = []
    sql_events: list[str] = []
    status_text = "\n".join(
        [
            "boatrace-dashboard RUNNING pid 1, uptime 1:00:00",
            "boatrace-daily-shadow-bundle-update STOPPED Not started",
            f"{runner} RUNNING pid 2, uptime 1:00:00",
            f"{scheduler} RUNNING pid 3, uptime 1:00:00",
            "unrelated-service RUNNING pid 4, uptime 1:00:00",
        ]
    )

    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        maintenance_tasks, "connection", _connection(0, sql_events)
    )

    def run(command, **_kwargs):
        calls.append([str(value) for value in command])
        stdout = status_text if command[-1] == "status" else "ok"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(maintenance_tasks.subprocess, "run", run)
    payload = maintenance_tasks.refresh_services(
        tmp_path, db="test", head=head, delay_seconds=0
    )

    assert payload["restarted"] == [runner, scheduler]
    assert [(command[-2], command[-1]) for command in calls] == [
        (str(tmp_path / "scripts/deployment/supervisord.conf"), "status"),
        ("stop", scheduler),
        ("stop", runner),
        ("start", runner),
        ("start", scheduler),
    ]
    assert payload["control_plane_stopped"] == [scheduler, runner]
    assert payload["control_plane_resumed"] == [runner, scheduler]
    assert [command[-1] for command in calls] == [
        "status",
        scheduler,
        runner,
        runner,
        scheduler,
    ]
    assert "boatrace-daily-shadow-bundle-update" in payload["skipped_stopped"]
    assert "boatrace-dashboard" in payload["skipped_unchanged"]
    assert "unrelated-service" not in payload["restarted"]
    assert "pg_advisory_lock" in sql_events[0]
    assert "status = 'running'" in sql_events[1]
    assert "pg_advisory_unlock" in sql_events[2]


def test_refresh_services_accepts_supervisor_partial_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    head = "e" * 40
    status_text = "\n".join(
        [
            "boatrace-dashboard RUNNING pid 1, uptime 1:00:00",
            "boatrace-daily-shadow-bundle-update STOPPED Not started",
        ]
    )
    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            3 if command[-1] == "status" else 0,
            status_text if command[-1] == "status" else "ok",
            "",
        )

    monkeypatch.setattr(maintenance_tasks.subprocess, "run", run)

    payload = maintenance_tasks.refresh_services(
        tmp_path, db="test", head=head, delay_seconds=0
    )

    assert payload["status"] == "completed"
    assert payload["restarted"] == []
    assert "boatrace-daily-shadow-bundle-update" in payload["skipped_stopped"]
    assert "boatrace-dashboard" in payload["skipped_unchanged"]


def test_refresh_services_skips_unchanged_consumers_for_control_plane_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_head = "d" * 40
    head = "e" * 40
    runner = "boatrace-evaluation-runner:boatrace-evaluation-runner_00"
    scheduler = "boatrace-evaluation-scheduler"
    status_text = "\n".join(
        [
            "boatrace-dashboard RUNNING pid 1, uptime 1:00:00",
            f"{runner} RUNNING pid 2, uptime 1:00:00",
            f"{scheduler} RUNNING pid 3, uptime 1:00:00",
        ]
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )

    def git_command(_root: Path, *args: str):
        stdout = (
            "src/boatrace_ai/evaluation_queue.py\n"
            "docs/PROJECT_STATUS.md\n"
            if args[:2] == ("diff", "--name-only")
            else ""
        )
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(maintenance_tasks, "_git_command", git_command)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    def run(command, **_kwargs):
        action = command[-2] if command[-1] != "status" else "status"
        name = command[-1]
        calls.append((action, name))
        stdout = status_text if name == "status" else "ok"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(maintenance_tasks.subprocess, "run", run)

    payload = maintenance_tasks.refresh_services(
        tmp_path,
        db="test",
        head=head,
        base_head=base_head,
        delay_seconds=0,
    )

    assert payload["status"] == "completed"
    assert payload["restarted"] == [runner, scheduler]
    assert payload["changed_paths"] == [
        "src/boatrace_ai/evaluation_queue.py",
        "docs/PROJECT_STATUS.md",
    ]
    assert "boatrace-dashboard" in payload["skipped_unchanged"]
    assert ("restart", "boatrace-dashboard") not in calls
    assert ("stop", runner) in calls
    assert ("stop", scheduler) in calls
    assert ("start", runner) in calls
    assert ("start", scheduler) in calls


def test_refresh_services_uses_sibling_service_manager_supervisorctl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    head = "d" * 40
    app_root = tmp_path / "boat"
    supervisorctl = (
        tmp_path / "service-manager" / ".venv" / "bin" / "supervisorctl"
    )
    supervisorctl.parent.mkdir(parents=True)
    supervisorctl.touch()
    commands: list[list[str]] = []
    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    def run(command, **_kwargs):
        commands.append([str(value) for value in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(maintenance_tasks.subprocess, "run", run)

    maintenance_tasks.refresh_services(
        app_root, db="test", head=head, delay_seconds=0
    )

    assert commands[0][0] == str(supervisorctl)


def test_refresh_services_defers_if_evaluation_started_during_delay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    head = "b" * 40
    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(1))
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append([str(value) for value in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(maintenance_tasks.subprocess, "run", run)

    payload = maintenance_tasks.refresh_services(
        tmp_path, db="test", head=head, delay_seconds=0
    )

    assert payload["status"] == "deferred_active_evaluation"
    assert payload["active_evaluations"] == 1
    assert payload["restarted"] == []
    assert [command[-1] for command in calls] == ["status"]


def test_refresh_services_recovers_scheduler_if_runner_stop_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    head = "c" * 40
    runner = "boatrace-evaluation-runner:boatrace-evaluation-runner_00"
    scheduler = "boatrace-evaluation-scheduler"
    calls: list[tuple[str, str]] = []
    status_text = "\n".join(
        [
            f"{runner} RUNNING pid 2, uptime 1:00:00",
            f"{scheduler} RUNNING pid 3, uptime 1:00:00",
        ]
    )
    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    def run(command, **_kwargs):
        action = command[-2] if command[-1] != "status" else "status"
        name = command[-1]
        calls.append((action, name))
        if action == "stop" and name == runner:
            return subprocess.CompletedProcess(command, 1, "stop failed", "")
        stdout = status_text if name == "status" else "ok"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(maintenance_tasks.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="stop failed"):
        maintenance_tasks.refresh_services(
            tmp_path, db="test", head=head, delay_seconds=0
        )

    assert ("stop", scheduler) in calls
    assert ("start", scheduler) in calls


def test_service_refresh_allowlist_is_limited_to_changed_code_consumers(
) -> None:
    assert set(maintenance_tasks.SERVICE_REFRESH_PROGRAMS) == {
        "boatrace-dashboard",
        "boatrace-daily-shadow-bundle-update",
        "boatrace-intraday-t300-daily-bundles",
        "boatrace-intraday-t300-shadow",
        "boatrace-intraday-v23-shadow",
        "boatrace-intraday-v32-shadow",
        "boatrace-stable-cell-shadow",
        "boatrace-quota-ceil-shadow",
        "boatrace-raw-guard-shadow",
    }
